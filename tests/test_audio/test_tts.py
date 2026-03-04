"""Tests for TTS provider abstraction and implementations."""

from __future__ import annotations

import io
import sys
import wave
from unittest.mock import MagicMock, patch

import pytest

from seaman_brain.audio.tts import (
    KokoroTTSProvider,
    NoopTTSProvider,
    Pyttsx3TTSProvider,
    RivaTTSProvider,
    TTSProvider,
    create_tts_provider,
)
from seaman_brain.config import AudioConfig


def _mock_pyttsx3(engine=None):
    """Create a mock pyttsx3 module and return (mock_module, mock_engine)."""
    mock_module = MagicMock()
    mock_engine = engine or MagicMock()
    mock_module.init.return_value = mock_engine
    mock_engine.getProperty.return_value = []
    return mock_module, mock_engine


# ─── Protocol compliance ───────────────────────────────────────────


class TestTTSProviderProtocol:
    """Verify TTSProvider protocol is runtime-checkable."""

    def test_noop_implements_protocol(self):
        provider = NoopTTSProvider()
        assert isinstance(provider, TTSProvider)

    def test_pyttsx3_implements_protocol(self):
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
        assert isinstance(provider, TTSProvider)

    def test_arbitrary_class_not_protocol(self):
        class NotAProvider:
            pass
        assert not isinstance(NotAProvider(), TTSProvider)

    def test_partial_implementation_not_protocol(self):
        class Partial:
            async def synthesize(self, text: str) -> bytes:
                return b""
        assert not isinstance(Partial(), TTSProvider)


# ─── NoopTTSProvider ───────────────────────────────────────────────


class TestNoopTTSProvider:
    """Test the silent fallback provider."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_valid_wav(self):
        provider = NoopTTSProvider()
        data = await provider.synthesize("hello")
        assert len(data) > 0
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 22050
            assert wf.getnframes() == 0

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        provider = NoopTTSProvider()
        data = await provider.synthesize("")
        assert len(data) > 0

    @pytest.mark.asyncio
    async def test_speak_does_nothing(self):
        provider = NoopTTSProvider()
        await provider.speak("hello world")

    @pytest.mark.asyncio
    async def test_speak_empty_text(self):
        provider = NoopTTSProvider()
        await provider.speak("")


# ─── Pyttsx3TTSProvider init ───────────────────────────────────────


class TestPyttsx3TTSProviderInit:
    """Test initialization and configuration."""

    def test_init_success(self):
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            assert provider.available is True

    def test_init_with_custom_config(self):
        config = AudioConfig(tts_rate=200, tts_volume=0.5, tts_voice="english")
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            assert provider.available is True
            assert provider._config.tts_rate == 200
            assert provider._config.tts_volume == 0.5

    def test_init_failure_marks_unavailable(self):
        mock_mod = MagicMock()
        mock_mod.init.side_effect = RuntimeError("No audio device")
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            assert provider.available is False
            assert "No audio device" in provider._init_error

    def test_init_import_error_marks_unavailable(self):
        # pyttsx3 set to None in sys.modules triggers ImportError
        with patch.dict(sys.modules, {"pyttsx3": None}):
            provider = Pyttsx3TTSProvider()
            assert provider.available is False


# ─── Pyttsx3TTSProvider engine ─────────────────────────────────────


class TestPyttsx3TTSProviderEngine:
    """Test engine creation and configuration."""

    def test_create_engine_applies_rate(self):
        config = AudioConfig(tts_rate=180)
        mock_mod, mock_engine = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            engine.setProperty.assert_any_call("rate", 180)

    def test_create_engine_applies_volume(self):
        config = AudioConfig(tts_volume=0.6)
        mock_mod, mock_engine = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            engine.setProperty.assert_any_call("volume", 0.6)

    def test_create_engine_clamps_volume_high(self):
        config = AudioConfig(tts_volume=1.5)
        mock_mod, mock_engine = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            engine.setProperty.assert_any_call("volume", 1.0)

    def test_create_engine_clamps_volume_low(self):
        config = AudioConfig(tts_volume=-0.5)
        mock_mod, mock_engine = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            engine.setProperty.assert_any_call("volume", 0.0)

    def test_create_engine_selects_voice_by_id(self):
        config = AudioConfig(tts_voice="english")
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        voice_obj = MagicMock()
        voice_obj.id = "com.apple.speech.synthesis.voice.english"
        voice_obj.name = "Apple English"
        mock_engine.getProperty.return_value = [voice_obj]
        mock_mod.init.return_value = mock_engine
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            engine.setProperty.assert_any_call(
                "voice", "com.apple.speech.synthesis.voice.english"
            )

    def test_create_engine_selects_voice_by_name(self):
        """Voice matching works when config stores display name (not registry id)."""
        config = AudioConfig(
            tts_voice="Microsoft David Desktop - English (United States)"
        )
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        voice_obj = MagicMock()
        voice_obj.id = "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Tokens\\TTS_MS_DAVID"
        voice_obj.name = "Microsoft David Desktop - English (United States)"
        mock_engine.getProperty.return_value = [voice_obj]
        mock_mod.init.return_value = mock_engine
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            engine.setProperty.assert_any_call(
                "voice",
                "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Speech\\Tokens\\TTS_MS_DAVID",
            )

    def test_create_engine_no_matching_voice(self):
        config = AudioConfig(tts_voice="nonexistent_voice")
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        voice_obj = MagicMock()
        voice_obj.id = "com.apple.speech.synthesis.voice.english"
        voice_obj.name = "Apple English"
        mock_engine.getProperty.return_value = [voice_obj]
        mock_mod.init.return_value = mock_engine
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider(config)
            engine = provider._create_engine()
            voice_calls = [
                c for c in engine.setProperty.call_args_list
                if c[0][0] == "voice"
            ]
            assert len(voice_calls) == 0


# ─── Pyttsx3TTSProvider synthesize ─────────────────────────────────


class TestPyttsx3TTSProviderSynthesize:
    """Test async synthesize method."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self):
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        mock_mod.init.return_value = mock_engine

        def fake_save(text, path):
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b"\x00" * 100)
            with open(path, "wb") as f:
                f.write(buf.getvalue())

        mock_engine.save_to_file.side_effect = fake_save
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            data = await provider.synthesize("Hello world")
            assert isinstance(data, bytes)
            assert len(data) > 0

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            data = await provider.synthesize("")
            assert isinstance(data, bytes)
            assert len(data) > 0

    @pytest.mark.asyncio
    async def test_synthesize_whitespace_only(self):
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            data = await provider.synthesize("   ")
            assert isinstance(data, bytes)

    @pytest.mark.asyncio
    async def test_synthesize_unavailable_raises(self):
        mock_mod = MagicMock()
        mock_mod.init.side_effect = RuntimeError("No device")
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            assert not provider.available
            with pytest.raises(RuntimeError, match="TTS engine unavailable"):
                await provider.synthesize("hello")


# ─── Pyttsx3TTSProvider speak ──────────────────────────────────────


class TestPyttsx3TTSProviderSpeak:
    """Test async speak method."""

    @pytest.mark.asyncio
    async def test_speak_calls_engine(self):
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        mock_mod.init.return_value = mock_engine
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            await provider.speak("Hello")
            mock_engine.say.assert_called_once_with("Hello")
            mock_engine.runAndWait.assert_called_once()

    @pytest.mark.asyncio
    async def test_speak_empty_text_skips(self):
        mock_mod, mock_engine = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            await provider.speak("")
            # _speak_sync should return early, no say() call
            # But we need a fresh engine to check — the engine from init is different
            # from the one created in _speak_sync. Since speak("") returns early
            # before creating an engine, we can check that mock_mod.init was called
            # exactly once (during __init__._initialize, not in _speak_sync)
            # Actually, _initialize calls init twice (test + stop), so just check no say
            assert not any(
                c for c in mock_engine.say.call_args_list
            )

    @pytest.mark.asyncio
    async def test_speak_unavailable_no_error(self):
        mock_mod = MagicMock()
        mock_mod.init.side_effect = RuntimeError("No device")
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            await provider.speak("hello")

    @pytest.mark.asyncio
    async def test_speak_strips_whitespace(self):
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        mock_mod.init.return_value = mock_engine
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            await provider.speak("  Hello  ")
            mock_engine.say.assert_called_once_with("Hello")


# ─── create_tts_provider factory ───────────────────────────────────


class TestCreateTTSProvider:
    """Test the factory function."""

    def test_disabled_returns_noop(self):
        config = AudioConfig(tts_enabled=False)
        provider = create_tts_provider(config)
        assert isinstance(provider, NoopTTSProvider)

    def test_pyttsx3_available(self):
        config = AudioConfig(tts_provider="pyttsx3")
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = create_tts_provider(config)
            assert isinstance(provider, Pyttsx3TTSProvider)

    def test_pyttsx3_unavailable_falls_back(self):
        config = AudioConfig(tts_provider="pyttsx3")
        mock_mod = MagicMock()
        mock_mod.init.side_effect = RuntimeError("No audio")
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = create_tts_provider(config)
            assert isinstance(provider, NoopTTSProvider)

    def test_unknown_provider_falls_back(self):
        config = AudioConfig(tts_provider="unknown_engine")
        provider = create_tts_provider(config)
        assert isinstance(provider, NoopTTSProvider)

    def test_default_config_uses_pyttsx3(self):
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = create_tts_provider()
            assert isinstance(provider, Pyttsx3TTSProvider)

    def test_case_insensitive_provider_name(self):
        config = AudioConfig(tts_provider="PYTTSX3")
        mock_mod, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = create_tts_provider(config)
            assert isinstance(provider, Pyttsx3TTSProvider)


# ─── Edge cases ────────────────────────────────────────────────────


class TestTTSEdgeCases:
    """Edge case and integration tests."""

    @pytest.mark.asyncio
    async def test_noop_synthesize_produces_parseable_wav(self):
        provider = NoopTTSProvider()
        data = await provider.synthesize("test")
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() >= 1
            assert wf.getsampwidth() >= 1

    def test_empty_wav_helper(self):
        data = Pyttsx3TTSProvider._empty_wav()
        assert isinstance(data, bytes)
        buf = io.BytesIO(data)
        with wave.open(buf, "rb") as wf:
            assert wf.getnframes() == 0

    @pytest.mark.asyncio
    async def test_synthesize_engine_failure_in_save(self):
        mock_mod = MagicMock()
        mock_engine = MagicMock()
        mock_mod.init.return_value = mock_engine
        mock_engine.save_to_file.side_effect = RuntimeError("Engine crash")
        with patch.dict(sys.modules, {"pyttsx3": mock_mod}):
            provider = Pyttsx3TTSProvider()
            with pytest.raises(RuntimeError, match="Engine crash"):
                await provider.synthesize("hello")


# ─── KokoroTTSProvider ───────────────────────────────────────────


def _mock_kokoro():
    """Create a mock kokoro module and return (mock_module, mock_pipeline)."""
    import numpy as np

    mock_module = MagicMock()
    mock_pipeline = MagicMock()

    # Make pipeline callable — returns generator of (gs, ps, audio) tuples
    sample_audio = np.zeros(2400, dtype=np.float32)  # 0.1s at 24kHz
    mock_pipeline.__call__ = MagicMock(
        return_value=iter([(None, None, sample_audio)])
    )
    mock_module.KPipeline.return_value = mock_pipeline
    return mock_module, mock_pipeline


def _mock_soundfile():
    """Create a mock soundfile module."""
    mock_sf = MagicMock()

    def fake_write(buf, data, samplerate, format=None, subtype=None):
        """Write a minimal valid WAV to the buffer."""
        import struct
        num_samples = len(data)
        byte_data = struct.pack(f"<{num_samples}h", *[int(s * 32767) for s in data[:100]])
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(byte_data)
        buf.write(wav_buf.getvalue())
        buf.seek(0)

    mock_sf.write.side_effect = fake_write
    mock_sf.read.return_value = ([0.0] * 100, 24000)
    return mock_sf


class TestKokoroTTSProviderInit:
    """Test initialization of KokoroTTSProvider."""

    def test_init_lazy_no_import(self):
        """Constructor does NOT import kokoro — lazy init."""
        provider = KokoroTTSProvider()
        assert provider._pipeline is None
        assert provider.available is False  # Not initialized yet

    def test_initialize_success(self):
        """_initialize() loads kokoro pipeline."""
        mock_kokoro, _ = _mock_kokoro()
        with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
            provider = KokoroTTSProvider()
            provider._initialize()
            assert provider.available is True
            assert provider._pipeline is not None

    def test_initialize_import_error(self):
        """_initialize() handles missing kokoro gracefully."""
        with patch.dict(sys.modules, {"kokoro": None}):
            provider = KokoroTTSProvider()
            provider._initialize()
            assert provider.available is False
            assert "not installed" in provider._init_error

    def test_initialize_idempotent(self):
        """Calling _initialize() twice does not recreate the pipeline."""
        mock_kokoro, _ = _mock_kokoro()
        with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
            provider = KokoroTTSProvider()
            provider._initialize()
            pipeline_ref = provider._pipeline
            provider._initialize()
            assert provider._pipeline is pipeline_ref

    def test_implements_protocol(self):
        """KokoroTTSProvider satisfies TTSProvider protocol."""
        provider = KokoroTTSProvider()
        assert isinstance(provider, TTSProvider)


class TestKokoroTTSProviderSynthesize:
    """Test synthesis with mocked kokoro."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_bytes(self):
        """synthesize() returns WAV bytes."""
        mock_kokoro, _ = _mock_kokoro()
        mock_sf = _mock_soundfile()
        with patch.dict(sys.modules, {
            "kokoro": mock_kokoro,
            "soundfile": mock_sf,
        }):
            provider = KokoroTTSProvider()
            data = await provider.synthesize("Hello world")
            assert isinstance(data, bytes)
            assert len(data) > 0

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """synthesize() returns empty WAV for blank input."""
        mock_kokoro, _ = _mock_kokoro()
        with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
            provider = KokoroTTSProvider()
            provider._initialize()
            data = await provider.synthesize("")
            assert isinstance(data, bytes)
            # Verify it's a valid WAV
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                assert wf.getnframes() == 0

    @pytest.mark.asyncio
    async def test_synthesize_unavailable_raises(self):
        """synthesize() raises when kokoro is unavailable."""
        with patch.dict(sys.modules, {"kokoro": None}):
            provider = KokoroTTSProvider()
            with pytest.raises(RuntimeError, match="Kokoro TTS unavailable"):
                await provider.synthesize("hello")

    @pytest.mark.asyncio
    async def test_voice_config_applied(self):
        """Voice and speed config are passed to pipeline."""
        config = AudioConfig(tts_voice="am_michael", tts_speed=1.5)
        mock_kokoro, mock_pipeline = _mock_kokoro()
        mock_sf = _mock_soundfile()
        with patch.dict(sys.modules, {
            "kokoro": mock_kokoro,
            "soundfile": mock_sf,
        }):
            provider = KokoroTTSProvider(config)
            await provider.synthesize("Test")
            mock_pipeline.assert_called_once_with(
                "Test", voice="am_michael", speed=1.5
            )


    @pytest.mark.asyncio
    async def test_invalid_voice_falls_back_to_default(self):
        """Invalid voice name (e.g. pyttsx3 placeholder) falls back to af_heart."""
        config = AudioConfig(tts_voice="Some Voice")
        mock_kokoro, mock_pipeline = _mock_kokoro()
        mock_sf = _mock_soundfile()
        with patch.dict(sys.modules, {
            "kokoro": mock_kokoro,
            "soundfile": mock_sf,
        }):
            provider = KokoroTTSProvider(config)
            await provider.synthesize("Test")
            mock_pipeline.assert_called_once_with(
                "Test", voice="af_heart", speed=1.0
            )


class TestKokoroTTSProviderSpeak:
    """Test speak with mocked kokoro + sounddevice."""

    @pytest.mark.asyncio
    async def test_speak_unavailable_no_error(self):
        """speak() silently skips when unavailable."""
        with patch.dict(sys.modules, {"kokoro": None}):
            provider = KokoroTTSProvider()
            provider._initialize()
            await provider.speak("hello")  # Should not raise

    @pytest.mark.asyncio
    async def test_speak_empty_skips(self):
        """speak() does nothing for empty text."""
        mock_kokoro, _ = _mock_kokoro()
        with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
            provider = KokoroTTSProvider()
            provider._initialize()
            await provider.speak("")


class TestKokoroTTSFactory:
    """Test factory with kokoro provider."""

    def test_factory_creates_kokoro_when_available(self):
        """Factory creates KokoroTTSProvider when provider='kokoro' and installed."""
        config = AudioConfig(tts_provider="kokoro")
        mock_kokoro, _ = _mock_kokoro()
        with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
            provider = create_tts_provider(config)
            assert isinstance(provider, KokoroTTSProvider)

    def test_factory_falls_back_to_pyttsx3(self):
        """Factory falls back to pyttsx3 when kokoro not installed."""
        config = AudioConfig(tts_provider="kokoro")
        mock_pyttsx3, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {"kokoro": None, "pyttsx3": mock_pyttsx3}):
            provider = create_tts_provider(config)
            assert isinstance(provider, Pyttsx3TTSProvider)

    def test_factory_falls_back_to_noop(self):
        """Factory falls back to noop when both kokoro and pyttsx3 unavailable."""
        config = AudioConfig(tts_provider="kokoro")
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = RuntimeError("No audio")
        with patch.dict(sys.modules, {"kokoro": None, "pyttsx3": mock_pyttsx3}):
            provider = create_tts_provider(config)
            assert isinstance(provider, NoopTTSProvider)


# ── Fix #3: TTS executor timeout ─────────────────────────────────────


class TestTTSTimeout:
    """Tests for TTS executor timeout handling."""

    async def test_synthesize_timeout_returns_empty_wav(self):
        """Pyttsx3 synthesize timeout returns empty WAV bytes."""
        import seaman_brain.audio.tts as tts_mod

        orig = tts_mod._TTS_TIMEOUT
        tts_mod._TTS_TIMEOUT = 0.01  # Very short for test

        try:
            mock_pyttsx3, mock_engine = _mock_pyttsx3()
            with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
                provider = Pyttsx3TTSProvider()

                # Make sync call block forever
                def _block(*args, **kwargs):
                    import time
                    time.sleep(10)

                provider._synthesize_sync = _block

                result = await provider.synthesize("test")
                # Should get empty WAV, not hang
                assert len(result) > 0  # WAV header at minimum
        finally:
            tts_mod._TTS_TIMEOUT = orig

    async def test_speak_timeout_does_not_hang(self):
        """Pyttsx3 speak timeout completes without hanging."""
        import seaman_brain.audio.tts as tts_mod

        orig = tts_mod._TTS_TIMEOUT
        tts_mod._TTS_TIMEOUT = 0.01

        try:
            mock_pyttsx3, mock_engine = _mock_pyttsx3()
            with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
                provider = Pyttsx3TTSProvider()

                def _block(*args, **kwargs):
                    import time
                    time.sleep(10)

                provider._speak_sync = _block

                # Should complete, not hang
                await provider.speak("test")
        finally:
            tts_mod._TTS_TIMEOUT = orig


# ── Fix #6: Kokoro retry cooldown ────────────────────────────────────


class TestKokoroRetryCooldown:
    """Tests for Kokoro init retry cooldown."""

    def test_retry_skipped_during_cooldown(self):
        """Kokoro init skipped if last failure was recent."""
        provider = KokoroTTSProvider()
        provider._last_failure_time = 999999999.0  # far future
        provider._retry_interval = 60.0

        # Patch time.monotonic to return something before cooldown expires
        with patch("time.monotonic", return_value=999999999.0 + 30):
            provider._initialize()

        # Should still be unavailable (skipped due to cooldown)
        assert not provider._available

    def test_retry_attempted_after_cooldown(self):
        """Kokoro init retried after cooldown expires."""
        provider = KokoroTTSProvider()
        provider._last_failure_time = 100.0
        provider._retry_interval = 60.0

        # After cooldown (100 + 60 < 200)
        with patch("time.monotonic", return_value=200.0):
            with patch.dict(sys.modules, {"kokoro": None}):
                provider._initialize()

        # Attempted but failed (kokoro not installed)
        assert not provider._available
        assert provider._last_failure_time > 0


# ── Fix #25: Empty WAV detection ─────────────────────────────────────


class TestKokoroCleanForTTS:
    """Test _clean_for_tts markup stripping for Kokoro G2P safety."""

    def test_strips_think_block(self):
        text = "<think>\nReasoning here.\n</think>\nHello."
        result = KokoroTTSProvider._clean_for_tts(text)
        assert "<think>" not in result
        assert "Reasoning" not in result
        assert "Hello." in result

    def test_strips_remaining_tags(self):
        text = "Hello <b>world</b>."
        result = KokoroTTSProvider._clean_for_tts(text)
        assert "<b>" not in result
        assert result == "Hello world."

    def test_strips_asterisks(self):
        text = "*sighs deeply* Whatever."
        result = KokoroTTSProvider._clean_for_tts(text)
        assert "*" not in result
        assert "sighs deeply" in result
        assert "Whatever." in result

    def test_collapses_whitespace(self):
        text = "Hello    world."
        result = KokoroTTSProvider._clean_for_tts(text)
        assert "  " not in result
        assert result == "Hello world."

    def test_strips_complex_qwen3_output(self):
        text = (
            "<think>\nI should keep it short - 1-2 sentences.\n</think>\n"
            "Oh wonderful, another day."
        )
        result = KokoroTTSProvider._clean_for_tts(text)
        assert "1-2 sentences" not in result
        assert "Oh wonderful, another day." in result

    def test_empty_after_stripping_returns_empty(self):
        text = "<think>only reasoning</think>"
        result = KokoroTTSProvider._clean_for_tts(text)
        assert result == ""

    def test_plain_text_unchanged(self):
        text = "The water is cold."
        result = KokoroTTSProvider._clean_for_tts(text)
        assert result == "The water is cold."


class TestKokoroPerSentenceFallback:
    """Test that per-sentence processing skips G2P failures gracefully."""

    @pytest.mark.asyncio
    async def test_synthesize_skips_bad_sentences(self):
        """Synthesize produces audio even when some sentences fail G2P."""
        import numpy as np

        mock_kokoro, mock_pipeline = _mock_kokoro()
        mock_sf = _mock_soundfile()

        # Use 100 samples to fit _mock_soundfile's fake_write limit
        sample_audio = np.zeros(100, dtype=np.float32)

        def fake_pipeline(text, voice=None, speed=None):
            if "podfish" in text.lower():
                raise TypeError(
                    "unsupported operand type(s) for +: 'NoneType' and 'str'"
                )
            return iter([(None, None, sample_audio)])

        mock_pipeline.side_effect = fake_pipeline

        with patch.dict(sys.modules, {
            "kokoro": mock_kokoro,
            "soundfile": mock_sf,
        }):
            provider = KokoroTTSProvider()
            data = await provider.synthesize(
                "A podfish sat there. The water was cold."
            )
            assert isinstance(data, bytes)
            assert len(data) > 0

    @pytest.mark.asyncio
    async def test_synthesize_all_sentences_fail(self):
        """Synthesize returns empty WAV when all sentences fail G2P."""
        mock_kokoro, mock_pipeline = _mock_kokoro()

        def always_fail(text, voice=None, speed=None):
            raise TypeError("NoneType + str")

        mock_pipeline.side_effect = always_fail

        with patch.dict(sys.modules, {"kokoro": mock_kokoro}):
            provider = KokoroTTSProvider()
            data = await provider.synthesize("podfish. xyzfoo.")
            assert isinstance(data, bytes)
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                assert wf.getnframes() == 0


class TestEmptyWAVDetection:
    """Tests for header-only WAV detection in pyttsx3."""

    def test_header_only_wav_raises(self):
        """Header-only WAV output raises RuntimeError."""
        mock_pyttsx3, mock_engine = _mock_pyttsx3()
        with patch.dict(sys.modules, {"pyttsx3": mock_pyttsx3}):
            provider = Pyttsx3TTSProvider()

            # Create a header-only WAV (44 bytes)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(22050)
                wf.writeframes(b"")
            header_only = buf.getvalue()
            assert len(header_only) == 44

            # Patch temp file to produce header-only WAV
            with patch("tempfile.NamedTemporaryFile") as mock_tmp:
                mock_tmp.return_value.__enter__ = lambda s: s
                mock_tmp.return_value.__exit__ = MagicMock(return_value=False)
                mock_tmp.return_value.name = "fake.wav"

                from pathlib import Path
                with patch.object(Path, "exists", return_value=True):
                    with patch.object(Path, "stat") as mock_stat:
                        mock_stat.return_value.st_size = 44
                        with patch.object(Path, "read_bytes", return_value=header_only):
                            with patch.object(Path, "unlink"):
                                with pytest.raises(
                                    RuntimeError, match="TTS produced empty audio"
                                ):
                                    provider._synthesize_sync("test")


# ─── RivaTTSProvider ─────────────────────────────────────────────


def _mock_riva_tts():
    """Create mock riva.client module for TTS tests."""
    mock_riva = MagicMock()
    mock_client = MagicMock()

    # Auth and SpeechSynthesisService
    mock_auth = MagicMock()
    mock_service = MagicMock()
    mock_client.Auth.return_value = mock_auth
    mock_client.SpeechSynthesisService.return_value = mock_service
    mock_client.SynthesizeSpeechRequest = MagicMock
    mock_client.AudioEncoding = MagicMock()
    mock_client.AudioEncoding.LINEAR_PCM = 1

    mock_riva.client = mock_client
    return mock_riva, mock_client, mock_service


class TestRivaTTSProviderInit:
    """Test initialization of RivaTTSProvider."""

    def test_implements_protocol(self):
        """RivaTTSProvider satisfies TTSProvider protocol."""
        with patch.dict(sys.modules, {"riva": MagicMock(), "riva.client": MagicMock()}):
            provider = RivaTTSProvider()
        assert isinstance(provider, TTSProvider)

    def test_init_success(self):
        """Successful init marks provider as available."""
        mock_riva, mock_client, _ = _mock_riva_tts()
        with patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}):
            provider = RivaTTSProvider()
            assert provider.available is True

    def test_init_import_error(self):
        """Missing riva.client marks provider unavailable."""
        with patch.dict(sys.modules, {"riva": None, "riva.client": None}):
            provider = RivaTTSProvider()
            assert provider.available is False
            assert "not installed" in provider._init_error

    def test_init_connection_error(self):
        """Connection failure marks provider unavailable."""
        mock_riva, mock_client, _ = _mock_riva_tts()
        mock_client.Auth.side_effect = RuntimeError("Connection refused")
        with patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}):
            provider = RivaTTSProvider()
            assert provider.available is False


class TestRivaTTSProviderSynthesize:
    """Test Riva TTS synthesis."""

    @pytest.mark.asyncio
    async def test_synthesize_returns_wav_bytes(self):
        """synthesize() returns WAV bytes wrapping PCM."""
        mock_riva, mock_client, mock_service = _mock_riva_tts()
        # Mock response with PCM audio
        mock_resp = MagicMock()
        mock_resp.audio = b"\x00\x00" * 1600  # 0.1s at 16kHz
        mock_service.synthesize.return_value = mock_resp

        with patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}):
            provider = RivaTTSProvider()
            data = await provider.synthesize("Hello")
            assert isinstance(data, bytes)
            assert len(data) > 44  # More than just WAV header

    @pytest.mark.asyncio
    async def test_synthesize_unavailable_raises(self):
        """synthesize() raises when Riva unavailable."""
        with patch.dict(sys.modules, {"riva": None, "riva.client": None}):
            provider = RivaTTSProvider()
            with pytest.raises(RuntimeError, match="Riva TTS unavailable"):
                await provider.synthesize("hello")

    @pytest.mark.asyncio
    async def test_synthesize_empty_text(self):
        """synthesize() returns empty WAV for blank input."""
        mock_riva, mock_client, _ = _mock_riva_tts()
        with patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}):
            provider = RivaTTSProvider()
            data = await provider.synthesize("")
            assert isinstance(data, bytes)
            buf = io.BytesIO(data)
            with wave.open(buf, "rb") as wf:
                assert wf.getnframes() == 0

    def test_pcm_to_wav(self):
        """_pcm_to_wav wraps PCM bytes in valid WAV container."""
        pcm = b"\x00\x00" * 100
        wav = RivaTTSProvider._pcm_to_wav(pcm, 16000)
        buf = io.BytesIO(wav)
        with wave.open(buf, "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getsampwidth() == 2
            assert wf.getframerate() == 16000
            assert wf.getnframes() == 100


class TestRivaTTSFactory:
    """Test factory with Riva provider."""

    def test_factory_creates_riva_when_available(self):
        """Factory creates RivaTTSProvider when provider='riva' and installed."""
        config = AudioConfig(tts_provider="riva")
        mock_riva, mock_client, _ = _mock_riva_tts()
        with patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}):
            provider = create_tts_provider(config)
            assert isinstance(provider, RivaTTSProvider)

    def test_factory_riva_falls_back_to_kokoro(self):
        """Factory falls back to kokoro when Riva unavailable."""
        config = AudioConfig(tts_provider="riva")
        mock_kokoro, _ = _mock_kokoro()
        with patch.dict(sys.modules, {
            "riva": None, "riva.client": None,
            "kokoro": mock_kokoro,
        }):
            provider = create_tts_provider(config)
            assert isinstance(provider, KokoroTTSProvider)

    def test_factory_riva_falls_back_to_pyttsx3(self):
        """Factory falls back to pyttsx3 when Riva and Kokoro unavailable."""
        config = AudioConfig(tts_provider="riva")
        mock_pyttsx3, _ = _mock_pyttsx3()
        with patch.dict(sys.modules, {
            "riva": None, "riva.client": None,
            "kokoro": None,
            "pyttsx3": mock_pyttsx3,
        }):
            provider = create_tts_provider(config)
            assert isinstance(provider, Pyttsx3TTSProvider)

    def test_factory_riva_falls_back_to_noop(self):
        """Factory falls back to noop when all unavailable."""
        config = AudioConfig(tts_provider="riva")
        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.side_effect = RuntimeError("No audio")
        with patch.dict(sys.modules, {
            "riva": None, "riva.client": None,
            "kokoro": None,
            "pyttsx3": mock_pyttsx3,
        }):
            provider = create_tts_provider(config)
            assert isinstance(provider, NoopTTSProvider)
