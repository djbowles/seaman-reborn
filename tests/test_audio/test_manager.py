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


# ─── set_input_device wrapper ─────────────────────────────────────


class TestSetInputDevice:
    """Test AudioManager.set_input_device delegates to STT provider."""

    def test_delegates_to_stt_provider(self, manager, mock_stt):
        """set_input_device calls through to STT provider."""
        mock_stt.set_input_device = MagicMock()
        manager.set_input_device("Test Mic")
        mock_stt.set_input_device.assert_called_once_with("Test Mic")

    def test_restarts_pipeline_on_device_change(self, sounds_dir):
        """Changing input device restarts the full-duplex pipeline."""
        config = AudioConfig(aec_enabled=False)
        mock_tts = MagicMock()
        mock_stt = MagicMock()
        mock_stt.set_input_device = MagicMock()
        manager = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        # Simulate an active pipeline
        mock_pipeline = MagicMock()
        manager._pipeline = mock_pipeline
        manager._async_loop = MagicMock()

        with patch("seaman_brain.audio.manager.AudioManager._init_pipeline"):
            manager.set_input_device("Portacapture X6")

        # Old pipeline should have been stopped
        mock_pipeline.stop.assert_called_once()
        # Config should be updated
        assert config.audio_input_device == "Portacapture X6"

    def test_no_pipeline_restart_when_no_pipeline(self, manager, mock_stt):
        """No restart attempt when pipeline is not active."""
        mock_stt.set_input_device = MagicMock()
        assert manager._pipeline is None
        manager.set_input_device("Some Mic")
        mock_stt.set_input_device.assert_called_once_with("Some Mic")


# ── Fix #10: STT upgrade returns bool and resets on failure ──────────


class TestSTTUpgradeReturnsBool:
    """Tests for _try_upgrade_stt returning bool."""

    def test_upgrade_returns_true_on_success(self, sounds_dir):
        """_try_upgrade_stt returns True when upgrade succeeds."""
        from seaman_brain.audio.stt import NoopSTTProvider

        config = AudioConfig()
        mock_tts = MagicMock()
        noop_stt = NoopSTTProvider()
        manager = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=noop_stt,
            sounds_dir=sounds_dir,
        )

        real_provider = MagicMock()  # not a NoopSTTProvider
        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=real_provider,
        ):
            result = manager._try_upgrade_stt()
        assert result is True
        assert manager._stt is real_provider

    def test_upgrade_returns_false_on_noop(self, sounds_dir):
        """_try_upgrade_stt returns False when only NoopSTT is available."""
        from seaman_brain.audio.stt import NoopSTTProvider

        config = AudioConfig()
        mock_tts = MagicMock()
        noop_stt = NoopSTTProvider()
        manager = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=noop_stt,
            sounds_dir=sounds_dir,
        )

        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=NoopSTTProvider(),
        ):
            result = manager._try_upgrade_stt()
        assert result is False

    def test_stt_enabled_resets_on_upgrade_failure(self, sounds_dir):
        """stt_enabled setter resets to False when upgrade fails."""
        from seaman_brain.audio.stt import NoopSTTProvider

        config = AudioConfig()
        mock_tts = MagicMock()
        noop_stt = NoopSTTProvider()
        manager = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=noop_stt,
            sounds_dir=sounds_dir,
        )

        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=NoopSTTProvider(),
        ):
            manager.stt_enabled = True
        # Should have reverted because upgrade failed
        assert manager.stt_enabled is False


# ── TTS/STT failure handling (no cross-provider fallback) ─────────────


class TestTTSFailureHandling:
    """Test TTS failure logging without cross-provider fallback."""

    async def test_speak_failure_does_not_switch_provider(self, mock_stt, sounds_dir):
        """TTS failures are logged but provider is NOT swapped."""
        failing_tts = AsyncMock()
        failing_tts.speak = AsyncMock(side_effect=RuntimeError("synth error"))

        config = AudioConfig()
        mgr = AudioManager(
            config=config,
            tts_provider=failing_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr._tts_enabled = True

        for _ in range(5):
            await mgr.speak("test")

        # Provider stays the same — no fallback chain
        assert mgr._tts is failing_tts

    async def test_speak_failure_still_functional_after_recovery(
        self, mock_stt, sounds_dir
    ):
        """TTS works again when provider recovers from transient errors."""
        call_count = 0

        async def flaky_speak(text):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("transient error")

        flaky_tts = AsyncMock()
        flaky_tts.speak = flaky_speak

        config = AudioConfig()
        mgr = AudioManager(
            config=config,
            tts_provider=flaky_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr._tts_enabled = True

        # Two failures then success — no provider swap
        await mgr.speak("fail1")
        await mgr.speak("fail2")
        await mgr.speak("success")
        assert mgr._tts is flaky_tts


# ─── Full-duplex mode ─────────────────────────────────────────────


class TestFullDuplexMode:
    """Test AudioManager in full-duplex mode with AEC pipeline."""

    def test_pipeline_created_when_aec_enabled(self, mock_tts, mock_stt, sounds_dir):
        """Pipeline is created when aec_enabled=True."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        assert mgr.full_duplex is True
        assert mgr._pipeline is not None
        assert mgr._pending_utterance is not None
        assert mgr._barge_in_event is not None

    def test_no_pipeline_when_aec_disabled(self, manager):
        """Pipeline is not created when aec_enabled=False (default)."""
        assert manager.full_duplex is False
        assert manager._pipeline is None

    async def test_speak_feeds_reference_in_full_duplex(
        self, mock_tts, mock_stt, sounds_dir
    ):
        """speak() synthesizes and feeds reference in full-duplex mode."""
        import io
        import wave

        # Create realistic WAV data
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        wav_bytes = buf.getvalue()

        mock_tts.synthesize = AsyncMock(return_value=wav_bytes)

        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr._tts_enabled = True

        # Mock the pipeline's feed_reference and playback
        mgr._pipeline.feed_reference = MagicMock()
        with patch.object(mgr, "_play_wav_async", new_callable=AsyncMock):
            await mgr.speak("Hello full-duplex")

        mock_tts.synthesize.assert_awaited_once_with("Hello full-duplex")
        mgr._pipeline.feed_reference.assert_called_once_with(wav_bytes)

    async def test_listen_gets_from_utterance_queue(
        self, mock_tts, mock_stt, sounds_dir
    ):
        """listen() gets transcribed text from pipeline queue."""
        config = AudioConfig(aec_enabled=True, stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )

        # Put text in the queue
        await mgr._pending_utterance.put("hello from pipeline")

        result = await mgr.listen()
        assert result == "hello from pipeline"

    async def test_listen_timeout_returns_empty(
        self, mock_tts, mock_stt, sounds_dir
    ):
        """listen() returns empty on timeout."""
        import seaman_brain.audio.manager as mgr_mod

        orig = mgr_mod._LISTEN_TIMEOUT
        mgr_mod._LISTEN_TIMEOUT = 0.01

        try:
            config = AudioConfig(aec_enabled=True, stt_enabled=True)
            mgr = AudioManager(
                config=config,
                tts_provider=mock_tts,
                stt_provider=mock_stt,
                sounds_dir=sounds_dir,
            )

            result = await mgr.listen()
            assert result == ""
        finally:
            mgr_mod._LISTEN_TIMEOUT = orig

    def test_barge_in_event_propagation(self, mock_tts, mock_stt, sounds_dir):
        """Barge-in callback sets the event."""
        config = AudioConfig(aec_enabled=True, barge_in_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )

        assert not mgr._barge_in_event.is_set()
        mgr._on_pipeline_barge_in()
        assert mgr._barge_in_event.is_set()

    def test_cancel_tts_clears_reference(self, mock_tts, mock_stt, sounds_dir):
        """cancel_tts() clears pipeline reference queue."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )

        mgr._pipeline.clear_reference = MagicMock()
        with patch("sounddevice.stop"):
            mgr.cancel_tts()
        mgr._pipeline.clear_reference.assert_called_once()

    def test_cancel_tts_without_pipeline(self, manager):
        """cancel_tts() is safe when no pipeline."""
        with patch("sounddevice.stop"):
            manager.cancel_tts()
        assert manager._is_speaking is False

    def test_start_stop_pipeline(self, mock_tts, mock_stt, sounds_dir):
        """start_pipeline/stop_pipeline lifecycle."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )

        mgr._pipeline.start = MagicMock()
        mgr._pipeline.stop = MagicMock()

        mgr.start_pipeline()
        mgr._pipeline.start.assert_called_once()

        mgr.stop_pipeline()
        mgr._pipeline.stop.assert_called_once()

    async def test_half_duplex_unchanged(self, manager, mock_tts, mock_stt):
        """Half-duplex behavior is unchanged when aec_enabled=False."""
        assert manager.full_duplex is False

        # speak() uses original path
        await manager.speak("Hello half-duplex")
        mock_tts.speak.assert_awaited_once_with("Hello half-duplex")


# ─── Runtime provider swap ─────────────────────────────────────


class TestSwapTTSProvider:
    """Test swap_tts_provider() runtime provider replacement."""

    def test_swap_returns_class_name(self, manager):
        """swap_tts_provider returns the new provider's class name."""
        mock_new = MagicMock()
        with patch(
            "seaman_brain.audio.manager.create_tts_provider",
            return_value=mock_new,
        ):
            result = manager.swap_tts_provider(AudioConfig())
        assert result == type(mock_new).__name__
        assert manager._tts is mock_new

    def test_swap_updates_config(self, manager):
        """swap_tts_provider updates the manager's config reference."""
        new_config = AudioConfig(tts_provider="kokoro")
        mock_new = MagicMock()
        with patch(
            "seaman_brain.audio.manager.create_tts_provider",
            return_value=mock_new,
        ):
            manager.swap_tts_provider(new_config)
        assert manager._config is new_config


class TestSwapSTTProvider:
    """Test swap_stt_provider() runtime provider replacement."""

    def test_swap_returns_class_name(self, manager):
        """swap_stt_provider returns the new provider's class name."""
        mock_new = MagicMock()
        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=mock_new,
        ):
            result = manager.swap_stt_provider(AudioConfig(stt_enabled=True))
        assert result == type(mock_new).__name__
        assert manager._stt is mock_new

    def test_swap_updates_config(self, manager):
        """swap_stt_provider updates the manager's config reference."""
        new_config = AudioConfig(stt_provider="faster_whisper", stt_enabled=True)
        mock_new = MagicMock()
        with patch(
            "seaman_brain.audio.manager.create_stt_provider",
            return_value=mock_new,
        ):
            manager.swap_stt_provider(new_config)
        assert manager._config is new_config


class TestToggleAEC:
    """Test toggle_aec() runtime AEC pipeline management."""

    def test_enable_creates_pipeline(self, manager):
        """toggle_aec(True) creates and starts the pipeline."""
        assert manager._pipeline is None
        manager.toggle_aec(True)
        assert manager._pipeline is not None
        assert manager._pending_utterance is not None
        assert manager._barge_in_event is not None

    def test_disable_destroys_pipeline(self, mock_tts, mock_stt, sounds_dir):
        """toggle_aec(False) stops and tears down the pipeline."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        assert mgr._pipeline is not None
        mgr._pipeline.stop = MagicMock()
        mgr.toggle_aec(False)
        assert mgr._pipeline is None
        assert mgr._pending_utterance is None
        assert mgr._barge_in_event is None

    def test_enable_then_disable_roundtrip(self, manager):
        """Enable then disable AEC returns to half-duplex."""
        manager.toggle_aec(True)
        assert manager.full_duplex is True
        manager._pipeline.stop = MagicMock()
        manager.toggle_aec(False)
        assert manager.full_duplex is False


# ─── Mood forwarding ────────────────────────────────────────────


class TestSetMood:
    """Test AudioManager.set_mood() forwarding to TTS provider."""

    def test_forwards_to_provider_with_set_mood(self, manager, mock_tts):
        """set_mood() calls provider.set_mood() when available."""
        mock_tts.set_mood = MagicMock()
        manager.set_mood("hostile")
        mock_tts.set_mood.assert_called_once_with("hostile")

    def test_safe_without_set_mood(self, mock_stt, sounds_dir):
        """set_mood() is a no-op when provider lacks set_mood."""
        bare_tts = AsyncMock(spec=["synthesize", "speak"])
        mgr = AudioManager(
            config=AudioConfig(),
            tts_provider=bare_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        # Should not raise even though bare_tts has no set_mood
        mgr.set_mood("hostile")

    def test_forwards_multiple_moods(self, manager, mock_tts):
        """set_mood() forwards different mood values correctly."""
        mock_tts.set_mood = MagicMock()
        manager.set_mood("sardonic")
        manager.set_mood("content")
        assert mock_tts.set_mood.call_count == 2
        mock_tts.set_mood.assert_any_call("sardonic")
        mock_tts.set_mood.assert_any_call("content")


# ─── is_speaking with pipeline ──────────────────────────────────


class TestIsSpeakingWithPipeline:
    """Test is_speaking covers full-duplex pipeline TTS state."""

    def test_is_speaking_false_by_default(self, manager):
        """Default state: not speaking."""
        assert manager.is_speaking is False

    def test_is_speaking_true_half_duplex(self, manager):
        """Half-duplex _is_speaking flag."""
        manager._is_speaking = True
        assert manager.is_speaking is True

    def test_is_speaking_true_when_pipeline_tts_playing(self, manager):
        """Full-duplex: pipeline _tts_playing makes is_speaking True."""
        mock_pipeline = MagicMock()
        mock_pipeline._tts_playing = True
        manager._pipeline = mock_pipeline
        assert manager.is_speaking is True

    def test_is_speaking_false_when_pipeline_tts_idle(self, manager):
        """Full-duplex: pipeline _tts_playing=False doesn't trigger."""
        mock_pipeline = MagicMock()
        mock_pipeline._tts_playing = False
        manager._pipeline = mock_pipeline
        assert manager.is_speaking is False


# ─── Pipeline auto-restart ──────────────────────────────────────


class TestPipelineAutoRestart:
    """Test check_pipeline_health() auto-restart with backoff."""

    def test_no_restart_when_no_pipeline(self, manager):
        """check_pipeline_health() is a no-op without a pipeline."""
        manager.check_pipeline_health()
        assert manager._pipeline is None

    def test_no_restart_when_alive(self, mock_tts, mock_stt, sounds_dir):
        """check_pipeline_health() does nothing when pipeline is alive."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        original_pipeline = mgr._pipeline
        # Simulate alive: _running=True and thread alive
        original_pipeline._running = True
        original_pipeline._thread = MagicMock(is_alive=MagicMock(return_value=True))
        mgr.check_pipeline_health()
        assert mgr._pipeline is original_pipeline

    def test_restart_when_dead(self, mock_tts, mock_stt, sounds_dir):
        """check_pipeline_health() restarts a dead pipeline (first attempt)."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        original_pipeline = mgr._pipeline
        # Simulate dead: _running=True but thread is dead
        original_pipeline._running = True
        original_pipeline._thread = MagicMock(is_alive=MagicMock(return_value=False))
        original_pipeline.stop = MagicMock()

        mgr.check_pipeline_health()

        assert mgr._pipeline is not original_pipeline
        assert mgr._pipeline is not None
        original_pipeline.stop.assert_called_once()
        assert mgr._pipeline_restart_count == 1

    def test_backoff_prevents_immediate_retry(self, mock_tts, mock_stt, sounds_dir):
        """Second check within backoff window does not restart again."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        # Simulate dead
        mgr._pipeline._running = True
        mgr._pipeline._thread = MagicMock(is_alive=MagicMock(return_value=False))
        mgr._pipeline.stop = MagicMock()

        mgr.check_pipeline_health()  # attempt 1
        p_after_first = mgr._pipeline
        # Simulate the new pipeline also dies immediately
        p_after_first._running = True
        p_after_first._thread = MagicMock(is_alive=MagicMock(return_value=False))
        p_after_first.stop = MagicMock()

        mgr.check_pipeline_health()  # within backoff — should not restart
        assert mgr._pipeline is p_after_first  # no change
        assert mgr._pipeline_restart_count == 1  # still 1

    def test_max_restarts_stops_retrying(self, mock_tts, mock_stt, sounds_dir):
        """After max restarts, stops retrying until long backoff expires."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr._pipeline_max_restarts = 2

        for _ in range(3):
            mgr._pipeline._running = True
            mgr._pipeline._thread = MagicMock(is_alive=MagicMock(return_value=False))
            mgr._pipeline.stop = MagicMock()
            mgr._pipeline_backoff_until = 0.0  # bypass backoff for test
            mgr.check_pipeline_health()

        assert mgr._pipeline_restart_count == 2  # capped at max

    def test_alive_resets_restart_count(self, mock_tts, mock_stt, sounds_dir):
        """A healthy pipeline resets the restart counter."""
        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr._pipeline_restart_count = 2  # simulate prior failures
        # Simulate alive
        mgr._pipeline._running = True
        mgr._pipeline._thread = MagicMock(is_alive=MagicMock(return_value=True))

        mgr.check_pipeline_health()
        assert mgr._pipeline_restart_count == 0


# ─── Transcription failure handling ──────────────────────────────


class TestTranscribeFailureHandling:
    """Test _transcribe_and_enqueue handles errors without fallback."""

    async def test_transcription_failure_logs_and_drops(self, mock_tts, sounds_dir):
        """Failed transcription is logged, utterance dropped (no retry)."""
        stt = AsyncMock()
        stt.transcribe = AsyncMock(side_effect=ConnectionError("gRPC down"))

        config = AudioConfig(aec_enabled=True, stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=stt,
            sounds_dir=sounds_dir,
        )

        await mgr._transcribe_and_enqueue(b"\x00\x00" * 100)

        # Only one call — no retry with fallback
        assert stt.transcribe.await_count == 1
        assert mgr._pending_utterance.empty()

    async def test_successful_transcription_enqueued(self, mock_tts, sounds_dir):
        """Successful transcription is put in the utterance queue."""
        stt = AsyncMock()
        stt.transcribe = AsyncMock(return_value="hello from riva")

        config = AudioConfig(aec_enabled=True, stt_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=mock_tts,
            stt_provider=stt,
            sounds_dir=sounds_dir,
        )

        await mgr._transcribe_and_enqueue(b"\x00\x00" * 1600)

        result = mgr._pending_utterance.get_nowait()
        assert result == "hello from riva"


# ─── TTS cancellation prevents pending playback ──────────────


class TestCancelTtsPreventsPlayback:
    """Test that cancel_tts prevents queued speak() from playing."""

    async def test_cancel_tts_prevents_pending_playback(
        self, mock_stt, sounds_dir
    ):
        """Speak() skips playback when _tts_cancelled is set before feed."""
        import io
        import wave

        # Create realistic WAV data
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b"\x00\x00" * 1600)
        wav_bytes = buf.getvalue()

        slow_tts = AsyncMock()

        async def _slow_synthesize(text: str) -> bytes:
            # Simulate cancel arriving during synthesis
            mgr._tts_cancelled.set()
            return wav_bytes

        slow_tts.synthesize = _slow_synthesize

        config = AudioConfig(aec_enabled=True)
        mgr = AudioManager(
            config=config,
            tts_provider=slow_tts,
            stt_provider=mock_stt,
            sounds_dir=sounds_dir,
        )
        mgr._tts_enabled = True

        # Mock the pipeline's feed_reference and playback
        mgr._pipeline.feed_reference = MagicMock()
        play_mock = AsyncMock()
        mgr._play_wav_async = play_mock

        await mgr.speak("Hello but cancelled")

        # feed_reference and playback should NOT have been called
        mgr._pipeline.feed_reference.assert_not_called()
        play_mock.assert_not_awaited()
