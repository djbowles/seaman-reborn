"""Tests for TTS provider abstraction and implementations."""

from __future__ import annotations

import io
import sys
import wave
from unittest.mock import MagicMock, patch

import pytest

from seaman_brain.audio.tts import (
    NoopTTSProvider,
    Pyttsx3TTSProvider,
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
