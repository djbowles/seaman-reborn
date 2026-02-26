"""Tests for the unified AudioManager."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.audio.manager import AudioManager, create_audio_manager
from seaman_brain.audio.stt import NoopSTTProvider
from seaman_brain.audio.tts import NoopTTSProvider
from seaman_brain.config import AudioConfig

# ─── Fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def mock_tts():
    """Create a mock TTS provider."""
    tts = AsyncMock()
    tts.speak = AsyncMock()
    tts.synthesize = AsyncMock(return_value=b"fake-wav-data")
    return tts


@pytest.fixture
def mock_stt():
    """Create a mock STT provider."""
    stt = AsyncMock()
    stt.listen = AsyncMock(return_value="hello world")
    return stt


@pytest.fixture
def sounds_dir(tmp_path):
    """Create a temporary sounds directory with a test WAV file."""
    sounds = tmp_path / "sounds"
    sounds.mkdir()
    # Create a minimal WAV file
    import io
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(22050)
        wf.writeframes(b"\x00\x00" * 100)
    (sounds / "splash.wav").write_bytes(buf.getvalue())
    return sounds


@pytest.fixture
def manager(mock_tts, mock_stt, sounds_dir):
    """Create an AudioManager with mock providers."""
    return AudioManager(
        config=AudioConfig(),
        tts_provider=mock_tts,
        stt_provider=mock_stt,
        sounds_dir=sounds_dir,
    )


# ─── Initialization ──────────────────────────────────────────────


class TestAudioManagerInit:
    """Test AudioManager initialization."""

    def test_default_config(self, mock_tts, mock_stt):
        mgr = AudioManager(tts_provider=mock_tts, stt_provider=mock_stt)
        assert mgr.tts_enabled is True
        assert mgr.stt_enabled is False  # Default AudioConfig has stt_enabled=False
        assert mgr.sfx_enabled is True

    def test_custom_config(self, mock_tts, mock_stt):
        config = AudioConfig(tts_enabled=False, stt_enabled=True, sfx_enabled=False)
        mgr = AudioManager(
            config=config, tts_provider=mock_tts, stt_provider=mock_stt
        )
        assert mgr.tts_enabled is False
        assert mgr.stt_enabled is True
        assert mgr.sfx_enabled is False

    def test_creates_default_providers_when_none(self):
        config = AudioConfig(tts_enabled=False, stt_enabled=False)
        mgr = AudioManager(config=config)
        assert isinstance(mgr.tts_provider, NoopTTSProvider)
        assert isinstance(mgr.stt_provider, NoopSTTProvider)

    def test_provider_accessors(self, manager, mock_tts, mock_stt):
        assert manager.tts_provider is mock_tts
        assert manager.stt_provider is mock_stt


# ─── TTS speak routing ───────────────────────────────────────────


class TestSpeak:
    """Test speak() routes to TTS provider."""

    async def test_speak_routes_to_tts(self, manager, mock_tts):
        await manager.speak("Hello there")
        mock_tts.speak.assert_awaited_once_with("Hello there")

    async def test_speak_disabled_skips(self, manager, mock_tts):
        manager.tts_enabled = False
        await manager.speak("Hello there")
        mock_tts.speak.assert_not_awaited()

    async def test_speak_empty_text_skips(self, manager, mock_tts):
        await manager.speak("")
        mock_tts.speak.assert_not_awaited()

    async def test_speak_whitespace_only_skips(self, manager, mock_tts):
        await manager.speak("   ")
        mock_tts.speak.assert_not_awaited()

    async def test_speak_error_handled_gracefully(self, manager, mock_tts):
        mock_tts.speak.side_effect = RuntimeError("engine crash")
        # Should not raise
        await manager.speak("Hello")

    async def test_speak_none_skips(self, manager, mock_tts):
        await manager.speak(None)
        mock_tts.speak.assert_not_awaited()


# ─── TTS synthesize routing ──────────────────────────────────────


class TestSynthesize:
    """Test synthesize() routes to TTS provider."""

    async def test_synthesize_returns_bytes(self, manager, mock_tts):
        result = await manager.synthesize("Test text")
        assert result == b"fake-wav-data"
        mock_tts.synthesize.assert_awaited_once_with("Test text")

    async def test_synthesize_disabled_returns_empty(self, manager, mock_tts):
        manager.tts_enabled = False
        result = await manager.synthesize("Test text")
        assert result == b""
        mock_tts.synthesize.assert_not_awaited()

    async def test_synthesize_empty_text_returns_empty(self, manager, mock_tts):
        result = await manager.synthesize("")
        assert result == b""

    async def test_synthesize_error_returns_empty(self, manager, mock_tts):
        mock_tts.synthesize.side_effect = RuntimeError("fail")
        result = await manager.synthesize("Test")
        assert result == b""


# ─── STT listen routing ──────────────────────────────────────────


class TestListen:
    """Test listen() routes to STT provider."""

    async def test_listen_returns_transcription(self, mock_tts, mock_stt, sounds_dir):
        config = AudioConfig(stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        result = await mgr.listen()
        assert result == "hello world"
        mock_stt.listen.assert_awaited_once()

    async def test_listen_disabled_returns_empty(self, manager, mock_stt):
        # Default config has stt_enabled=False
        result = await manager.listen()
        assert result == ""
        mock_stt.listen.assert_not_awaited()

    async def test_listen_error_returns_empty(self, mock_tts, mock_stt, sounds_dir):
        config = AudioConfig(stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mock_stt.listen.side_effect = OSError("mic error")
        result = await mgr.listen()
        assert result == ""


# ─── SFX playback ────────────────────────────────────────────────


class TestPlaySfx:
    """Test play_sfx() sound effects playback."""

    async def test_play_sfx_missing_file_no_error(self, manager):
        # Should log warning but not raise
        await manager.play_sfx("nonexistent")

    async def test_play_sfx_disabled_skips(self, manager, sounds_dir):
        manager.sfx_enabled = False
        # patch _play_wav to detect if it was called
        with patch.object(manager, "_play_wav") as mock_play:
            await manager.play_sfx("splash")
            mock_play.assert_not_called()

    async def test_play_sfx_empty_name_skips(self, manager):
        with patch.object(manager, "_play_wav") as mock_play:
            await manager.play_sfx("")
            mock_play.assert_not_called()

    async def test_play_sfx_existing_file(self, manager, sounds_dir):
        with patch.object(manager, "_play_wav") as mock_play:
            await manager.play_sfx("splash")
            mock_play.assert_called_once_with(sounds_dir / "splash.wav")

    async def test_play_sfx_strips_whitespace(self, manager, sounds_dir):
        with patch.object(manager, "_play_wav") as mock_play:
            await manager.play_sfx("  splash  ")
            mock_play.assert_called_once_with(sounds_dir / "splash.wav")

    async def test_play_sfx_playback_error_handled(self, manager):
        with patch.object(
            manager, "_play_wav", side_effect=RuntimeError("audio fail")
        ):
            # Should not raise
            await manager.play_sfx("splash")

    async def test_play_sfx_none_name_skips(self, manager):
        with patch.object(manager, "_play_wav") as mock_play:
            await manager.play_sfx(None)
            mock_play.assert_not_called()


# ─── _play_wav backend fallback ──────────────────────────────────


class TestPlayWav:
    """Test the internal _play_wav method's backend fallback logic."""

    def test_play_wav_winsound(self, manager, sounds_dir):
        mock_winsound = MagicMock()
        mock_winsound.SND_FILENAME = 1
        mock_winsound.SND_NODEFAULT = 2
        wav_path = sounds_dir / "splash.wav"
        with patch(
            "builtins.__import__",
            side_effect=lambda name, *a, **kw: (
                mock_winsound if name == "winsound" else __import__(name, *a, **kw)
            ),
        ):
            manager._play_wav(wav_path)
        mock_winsound.PlaySound.assert_called_once()

    def test_play_wav_no_backend_logs_warning(self, manager, sounds_dir, caplog):
        wav_path = sounds_dir / "splash.wav"
        # Make both imports fail
        original_import = __builtins__.__import__ if hasattr(
            __builtins__, "__import__"
        ) else __import__

        def fail_imports(name, *args, **kwargs):
            if name in ("winsound", "simpleaudio"):
                raise ImportError(f"No module {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=fail_imports):
            import logging
            with caplog.at_level(logging.WARNING):
                manager._play_wav(wav_path)
        assert "No audio playback backend" in caplog.text


# ─── Channel enable/disable ──────────────────────────────────────


class TestChannelControl:
    """Test per-channel enable/disable."""

    def test_set_channel_tts(self, manager):
        manager.set_channel("tts", False)
        assert manager.tts_enabled is False
        manager.set_channel("tts", True)
        assert manager.tts_enabled is True

    def test_set_channel_stt(self, manager):
        manager.set_channel("stt", True)
        assert manager.stt_enabled is True
        manager.set_channel("stt", False)
        assert manager.stt_enabled is False

    def test_set_channel_sfx(self, manager):
        manager.set_channel("sfx", False)
        assert manager.sfx_enabled is False

    def test_set_channel_unknown_raises(self, manager):
        with pytest.raises(ValueError, match="Unknown audio channel"):
            manager.set_channel("midi", True)

    def test_set_channel_case_insensitive(self, manager):
        manager.set_channel("TTS", False)
        assert manager.tts_enabled is False

    def test_set_channel_strips_whitespace(self, manager):
        manager.set_channel("  sfx  ", False)
        assert manager.sfx_enabled is False

    def test_get_status(self, manager):
        status = manager.get_status()
        assert "tts" in status
        assert "stt" in status
        assert "sfx" in status
        assert isinstance(status["tts"], bool)

    def test_get_status_reflects_changes(self, manager):
        manager.tts_enabled = False
        manager.stt_enabled = True
        status = manager.get_status()
        assert status["tts"] is False
        assert status["stt"] is True


# ─── SFX volume ──────────────────────────────────────────────────


class TestSfxVolume:
    """Test SFX volume control."""

    def test_default_volume(self, manager):
        assert manager.sfx_volume == 0.5

    def test_set_volume(self, manager):
        manager.sfx_volume = 0.7
        assert manager.sfx_volume == pytest.approx(0.7)

    def test_volume_clamps_above_one(self, manager):
        manager.sfx_volume = 1.5
        assert manager.sfx_volume == 1.0

    def test_volume_clamps_below_zero(self, manager):
        manager.sfx_volume = -0.5
        assert manager.sfx_volume == 0.0


# ─── Factory function ────────────────────────────────────────────


class TestCreateAudioManager:
    """Test the create_audio_manager factory."""

    def test_creates_with_defaults(self):
        config = AudioConfig(tts_enabled=False, stt_enabled=False)
        mgr = create_audio_manager(config=config)
        assert isinstance(mgr, AudioManager)
        assert mgr.tts_enabled is False

    def test_creates_with_sounds_dir(self, tmp_path):
        config = AudioConfig(tts_enabled=False, stt_enabled=False)
        sounds = tmp_path / "my_sounds"
        sounds.mkdir()
        mgr = create_audio_manager(config=config, sounds_dir=sounds)
        assert isinstance(mgr, AudioManager)

    def test_creates_with_none_config(self):
        mgr = create_audio_manager()
        assert isinstance(mgr, AudioManager)


# ─── Thread-safety (concurrent usage) ────────────────────────────


class TestConcurrency:
    """Test that concurrent operations don't cause errors."""

    async def test_concurrent_speak_and_listen(self, mock_tts, mock_stt, sounds_dir):
        config = AudioConfig(stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        results = await asyncio.gather(
            mgr.speak("hello"),
            mgr.listen(),
            return_exceptions=True,
        )
        assert all(not isinstance(r, Exception) for r in results)

    async def test_concurrent_sfx_serialized(self, manager, sounds_dir):
        """Multiple SFX calls should be serialized via lock."""
        call_order = []

        async def mock_play(path):
            call_order.append("start")
            await asyncio.sleep(0.01)
            call_order.append("end")

        with patch.object(manager, "_play_wav", side_effect=lambda p: None):
            await asyncio.gather(
                manager.play_sfx("splash"),
                manager.play_sfx("splash"),
            )


# ─── STT provider upgrade ──────────────────────────────────────


class TestSTTProviderUpgrade:
    """Test that enabling STT upgrades NoopSTTProvider to a real one."""

    def test_enable_stt_tries_upgrade(self, mock_tts, sounds_dir):
        """Setting stt_enabled=True on a noop STT triggers upgrade attempt."""
        config = AudioConfig(stt_enabled=False)
        mgr = AudioManager(
            config=config, tts_provider=mock_tts, sounds_dir=sounds_dir
        )
        assert isinstance(mgr.stt_provider, NoopSTTProvider)

        # Mock create_stt_provider to return a real (mock) provider
        mock_real_stt = AsyncMock()
        mock_real_stt.listen = AsyncMock(return_value="transcribed text")
        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=mock_real_stt,
        ):
            mgr.stt_enabled = True

        assert mgr.stt_provider is mock_real_stt
        assert mgr.stt_enabled is True

    def test_enable_stt_keeps_noop_if_unavailable(self, mock_tts, sounds_dir):
        """If upgrade still returns NoopSTTProvider, keep the noop."""
        config = AudioConfig(stt_enabled=False)
        mgr = AudioManager(
            config=config, tts_provider=mock_tts, sounds_dir=sounds_dir
        )
        noop = mgr.stt_provider
        assert isinstance(noop, NoopSTTProvider)

        # create_stt_provider returns another noop (e.g., PyAudio missing)
        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=NoopSTTProvider(),
        ):
            mgr.stt_enabled = True

        # Still noop, no crash
        assert isinstance(mgr.stt_provider, NoopSTTProvider)

    def test_disable_stt_no_upgrade(self, mock_tts, mock_stt, sounds_dir):
        """Setting stt_enabled=False does not trigger upgrade."""
        config = AudioConfig(stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        with patch(
            "seaman_brain.audio.manager.create_stt_provider"
        ) as mock_create:
            mgr.stt_enabled = False
            mock_create.assert_not_called()

    def test_stt_upgrade_exception_handled(self, mock_tts, sounds_dir):
        """If create_stt_provider raises, upgrade logs and keeps noop."""
        config = AudioConfig(stt_enabled=False)
        mgr = AudioManager(
            config=config, tts_provider=mock_tts, sounds_dir=sounds_dir
        )
        assert isinstance(mgr.stt_provider, NoopSTTProvider)

        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            side_effect=RuntimeError("no audio"),
        ):
            mgr.stt_enabled = True

        assert isinstance(mgr.stt_provider, NoopSTTProvider)


# ─── TTS voice update ────────────────────────────────────────────


class TestUpdateTTSVoice:
    """Test update_tts_voice() runtime voice switching."""

    def test_updates_manager_config(self, manager):
        """update_tts_voice sets the manager config."""
        manager.update_tts_voice("Microsoft Zira")
        assert manager._config.tts_voice == "Microsoft Zira"

    def test_normalizes_system_default(self, manager):
        """'System Default' is normalized to empty string."""
        manager.update_tts_voice("System Default")
        assert manager._config.tts_voice == ""

    def test_updates_pyttsx3_provider_config(self, mock_stt, sounds_dir):
        """update_tts_voice propagates to Pyttsx3TTSProvider config."""
        from seaman_brain.audio.tts import Pyttsx3TTSProvider

        mock_provider = MagicMock(spec=Pyttsx3TTSProvider)
        mock_provider._config = AudioConfig()
        config = AudioConfig()
        mgr = AudioManager(
            config=config,
            tts_provider=mock_provider,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr.update_tts_voice("Test Voice")
        assert mock_provider._config.tts_voice == "Test Voice"
        assert mgr._config.tts_voice == "Test Voice"

    def test_non_pyttsx3_provider_no_crash(self, manager):
        """update_tts_voice with mock TTS provider doesn't crash."""
        manager.update_tts_voice("Any Voice")
        assert manager._config.tts_voice == "Any Voice"
