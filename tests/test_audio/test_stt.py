"""Tests for STT provider abstraction and implementations."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from seaman_brain.audio.stt import (
    FasterWhisperSTTProvider,
    NoopSTTProvider,
    SpeechRecognitionSTTProvider,
    STTProvider,
    create_stt_provider,
)
from seaman_brain.config import AudioConfig


def _mock_sr(recognizer=None, mic_ok=True):
    """Create a mock speech_recognition module and return (mock_module, mock_recognizer).

    Args:
        recognizer: Optional pre-configured mock recognizer.
        mic_ok: If True, Microphone context manager works fine. If False, raises OSError.
    """
    mock_module = MagicMock()
    mock_rec = recognizer or MagicMock()
    mock_module.Recognizer.return_value = mock_rec

    # Set up exception classes on the mock module
    mock_module.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
    mock_module.UnknownValueError = type("UnknownValueError", (Exception,), {})
    mock_module.RequestError = type("RequestError", (Exception,), {})

    # Set up Microphone as a context manager
    mock_mic_instance = MagicMock()
    mock_mic_instance.__enter__ = MagicMock(return_value=mock_mic_instance)
    mock_mic_instance.__exit__ = MagicMock(return_value=False)

    if mic_ok:
        mock_module.Microphone.return_value = mock_mic_instance
    else:
        mock_module.Microphone.side_effect = OSError("No microphone found")

    return mock_module, mock_rec


# ─── Protocol compliance ───────────────────────────────────────────


class TestSTTProviderProtocol:
    """Verify STTProvider protocol is runtime-checkable."""

    def test_noop_implements_protocol(self):
        provider = NoopSTTProvider()
        assert isinstance(provider, STTProvider)

    def test_speech_recognition_implements_protocol(self):
        mock_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
        assert isinstance(provider, STTProvider)

    def test_arbitrary_class_not_protocol(self):
        class NotAProvider:
            pass
        assert not isinstance(NotAProvider(), STTProvider)

    def test_partial_implementation_not_protocol(self):
        class Partial:
            def not_listen(self) -> str:
                return ""
        assert not isinstance(Partial(), STTProvider)


# ─── NoopSTTProvider ───────────────────────────────────────────────


class TestNoopSTTProvider:
    """Test the silent fallback provider."""

    @pytest.mark.asyncio
    async def test_listen_returns_empty_string(self):
        provider = NoopSTTProvider()
        result = await provider.listen()
        assert result == ""

    @pytest.mark.asyncio
    async def test_listen_returns_str_type(self):
        provider = NoopSTTProvider()
        result = await provider.listen()
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_listen_multiple_calls(self):
        provider = NoopSTTProvider()
        r1 = await provider.listen()
        r2 = await provider.listen()
        assert r1 == r2 == ""


# ─── SpeechRecognitionSTTProvider init ─────────────────────────────


class TestSpeechRecognitionSTTProviderInit:
    """Test initialization and availability."""

    def test_init_success(self):
        mock_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            assert provider.available is True

    def test_init_with_custom_config(self):
        config = AudioConfig(stt_provider="vosk")
        mock_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider(config)
            assert provider.available is True
            assert provider._config.stt_provider == "vosk"

    def test_init_no_microphone(self):
        mock_mod, _ = _mock_sr(mic_ok=False)
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            assert provider.available is False
            assert "No microphone" in provider._init_error

    def test_init_import_error(self):
        with patch.dict(sys.modules, {"speech_recognition": None}):
            provider = SpeechRecognitionSTTProvider()
            assert provider.available is False
            assert "not installed" in provider._init_error

    def test_init_pyaudio_missing(self):
        """PyAudio not installed — Microphone raises AttributeError."""
        mock_mod, _ = _mock_sr()
        mock_mod.Microphone.side_effect = AttributeError("No PyAudio")
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            assert provider.available is False
            assert "PyAudio" in provider._init_error


# ─── SpeechRecognitionSTTProvider listen ──────────────────────────


class TestSpeechRecognitionSTTProviderListen:
    """Test async listen method with mocked microphone input."""

    @pytest.mark.asyncio
    async def test_listen_google_success(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.listen.return_value = mock_audio
        mock_rec.recognize_google.return_value = "hello world"

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = await provider.listen()
            assert result == "hello world"
            mock_rec.recognize_google.assert_called_once_with(mock_audio)

    @pytest.mark.asyncio
    async def test_listen_vosk_success(self):
        config = AudioConfig(stt_provider="vosk")
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.listen.return_value = mock_audio
        mock_rec.recognize_vosk.return_value = "hello from vosk"

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider(config)
            result = await provider.listen()
            assert result == "hello from vosk"
            mock_rec.recognize_vosk.assert_called_once_with(mock_audio)

    @pytest.mark.asyncio
    async def test_listen_timeout_returns_empty(self):
        mock_mod, mock_rec = _mock_sr()
        mock_rec.listen.side_effect = mock_mod.WaitTimeoutError("timeout")

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = await provider.listen()
            assert result == ""

    @pytest.mark.asyncio
    async def test_listen_unintelligible_returns_empty(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.listen.return_value = mock_audio
        mock_rec.recognize_google.side_effect = mock_mod.UnknownValueError()

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = await provider.listen()
            assert result == ""

    @pytest.mark.asyncio
    async def test_listen_request_error_returns_empty(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.listen.return_value = mock_audio
        mock_rec.recognize_google.side_effect = mock_mod.RequestError("API down")

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = await provider.listen()
            assert result == ""

    @pytest.mark.asyncio
    async def test_listen_unavailable_returns_empty(self):
        mock_mod, _ = _mock_sr(mic_ok=False)
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            assert not provider.available
            result = await provider.listen()
            assert result == ""

    @pytest.mark.asyncio
    async def test_listen_strips_whitespace(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.listen.return_value = mock_audio
        mock_rec.recognize_google.return_value = "  hello  "

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = await provider.listen()
            assert result == "hello"

    @pytest.mark.asyncio
    async def test_listen_mic_error_during_listen(self):
        mock_mod, mock_rec = _mock_sr()
        # First Microphone() call in __init__ succeeds (mic_ok=True)
        # Second Microphone() call in _listen_sync raises OSError
        call_count = 0
        original_mic = mock_mod.Microphone

        def mic_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return original_mic.return_value
            raise OSError("Microphone disconnected")

        mock_mod.Microphone.side_effect = mic_side_effect
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            assert provider.available
            result = await provider.listen()
            assert result == ""


# ─── SpeechRecognitionSTTProvider recognize ───────────────────────


class TestSpeechRecognitionSTTProviderRecognize:
    """Test the _recognize helper with different backends."""

    def test_recognize_unknown_backend_falls_back_to_google(self):
        config = AudioConfig(stt_provider="unknown_backend")
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.recognize_google.return_value = "fallback result"

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider(config)
            result = provider._recognize(mock_audio)
            assert result == "fallback result"
            mock_rec.recognize_google.assert_called_once_with(mock_audio)

    def test_recognize_empty_result(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.recognize_google.return_value = ""

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = provider._recognize(mock_audio)
            assert result == ""

    def test_recognize_none_result(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.recognize_google.return_value = None

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = provider._recognize(mock_audio)
            assert result == ""

    def test_recognize_unexpected_exception(self):
        mock_mod, mock_rec = _mock_sr()
        mock_audio = MagicMock()
        mock_rec.recognize_google.side_effect = RuntimeError("Unexpected")

        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            result = provider._recognize(mock_audio)
            assert result == ""


# ─── create_stt_provider factory ──────────────────────────────────


class TestCreateSTTProvider:
    """Test the factory function."""

    def test_disabled_returns_noop(self):
        config = AudioConfig(stt_enabled=False)
        provider = create_stt_provider(config)
        assert isinstance(provider, NoopSTTProvider)

    def test_enabled_with_mic_returns_speech_recognition(self):
        config = AudioConfig(stt_enabled=True)
        mock_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = create_stt_provider(config)
            assert isinstance(provider, SpeechRecognitionSTTProvider)

    def test_enabled_no_mic_falls_back(self):
        config = AudioConfig(stt_enabled=True)
        mock_mod, _ = _mock_sr(mic_ok=False)
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = create_stt_provider(config)
            assert isinstance(provider, NoopSTTProvider)

    def test_default_config_disabled(self):
        # Default AudioConfig has stt_enabled=False
        provider = create_stt_provider()
        assert isinstance(provider, NoopSTTProvider)

    def test_none_config_uses_defaults(self):
        provider = create_stt_provider(None)
        assert isinstance(provider, NoopSTTProvider)

    def test_enabled_import_error_falls_back(self):
        config = AudioConfig(stt_enabled=True)
        with patch.dict(sys.modules, {"speech_recognition": None}):
            provider = create_stt_provider(config)
            assert isinstance(provider, NoopSTTProvider)


# ─── FasterWhisperSTTProvider ─────────────────────────────────────


def _mock_faster_whisper():
    """Create a mock faster_whisper module and return (mock_module, mock_model)."""
    mock_module = MagicMock()
    mock_model = MagicMock()
    mock_module.WhisperModel.return_value = mock_model
    return mock_module, mock_model


class TestFasterWhisperSTTProviderInit:
    """Test initialization of FasterWhisperSTTProvider."""

    def test_init_lazy_no_import(self):
        """Constructor does NOT import faster_whisper — lazy init."""
        provider = FasterWhisperSTTProvider()
        assert provider._model is None
        assert provider.available is False

    def test_initialize_success(self):
        """_initialize() loads WhisperModel."""
        mock_fw, _ = _mock_faster_whisper()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            provider = FasterWhisperSTTProvider()
            provider._initialize()
            assert provider.available is True
            assert provider._model is not None

    def test_initialize_import_error(self):
        """_initialize() handles missing faster-whisper gracefully."""
        with patch.dict(sys.modules, {"faster_whisper": None}):
            provider = FasterWhisperSTTProvider()
            provider._initialize()
            assert provider.available is False
            assert "not installed" in provider._init_error

    def test_initialize_idempotent(self):
        """Calling _initialize() twice does not recreate the model."""
        mock_fw, _ = _mock_faster_whisper()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            provider = FasterWhisperSTTProvider()
            provider._initialize()
            model_ref = provider._model
            provider._initialize()
            assert provider._model is model_ref

    def test_implements_protocol(self):
        """FasterWhisperSTTProvider satisfies STTProvider protocol."""
        provider = FasterWhisperSTTProvider()
        assert isinstance(provider, STTProvider)

    def test_custom_config_applied(self):
        """Config values are stored correctly."""
        config = AudioConfig(
            stt_model="base",
            stt_silence_threshold=0.02,
            stt_silence_duration=2.0,
        )
        provider = FasterWhisperSTTProvider(config)
        assert provider._config.stt_model == "base"
        assert provider._config.stt_silence_threshold == 0.02
        assert provider._config.stt_silence_duration == 2.0


class TestFasterWhisperSTTProviderListen:
    """Test listen with mocked whisper + sounddevice."""

    @pytest.mark.asyncio
    async def test_listen_unavailable_returns_empty(self):
        """listen() returns empty when model not available."""
        with patch.dict(sys.modules, {"faster_whisper": None}):
            provider = FasterWhisperSTTProvider()
            result = await provider.listen()
            assert result == ""

    @pytest.mark.asyncio
    async def test_listen_success(self):
        """listen() returns transcribed text with mocked audio."""
        import numpy as np

        mock_fw, mock_model = _mock_faster_whisper()

        # Mock segment result
        mock_segment = MagicMock()
        mock_segment.text = "hello world"
        mock_model.transcribe.return_value = (iter([mock_segment]), MagicMock())

        # Mock sounddevice
        mock_sd = MagicMock()
        block_count = 0

        class FakeStream:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, n):
                nonlocal block_count
                block_count += 1
                if block_count <= 3:
                    # Speech — above threshold
                    return np.full((n, 1), 0.1, dtype=np.float32), False
                # Silence blocks — trigger stop after silence_dur
                return np.zeros((n, 1), dtype=np.float32), False

        mock_sd.InputStream.return_value = FakeStream()
        mock_sd.query_devices.return_value = []

        with patch.dict(sys.modules, {
            "faster_whisper": mock_fw,
            "sounddevice": mock_sd,
            "numpy": np,
        }):
            config = AudioConfig(stt_silence_duration=0.5)  # 5 blocks at 0.1s
            provider = FasterWhisperSTTProvider(config)
            provider._initialize()
            result = await provider.listen()
            assert result == "hello world"

    @pytest.mark.asyncio
    async def test_listen_no_speech_returns_empty(self):
        """listen() returns empty when only silence is captured."""
        import numpy as np

        mock_fw, mock_model = _mock_faster_whisper()

        mock_sd = MagicMock()

        class SilentStream:
            def __init__(self):
                self._count = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self, n):
                self._count += 1
                if self._count > 150:  # max_duration / 0.1
                    raise StopIteration
                return np.zeros((n, 1), dtype=np.float32), False

        mock_sd.InputStream.return_value = SilentStream()
        mock_sd.query_devices.return_value = []

        with patch.dict(sys.modules, {
            "faster_whisper": mock_fw,
            "sounddevice": mock_sd,
            "numpy": np,
        }):
            provider = FasterWhisperSTTProvider()
            provider._initialize()
            result = await provider.listen()
            assert result == ""


class TestFasterWhisperSTTFactory:
    """Test factory with faster_whisper provider."""

    def test_factory_creates_faster_whisper_when_available(self):
        """Factory creates FasterWhisperSTTProvider when configured."""
        config = AudioConfig(stt_enabled=True, stt_provider="faster_whisper")
        mock_fw, _ = _mock_faster_whisper()
        with patch.dict(sys.modules, {"faster_whisper": mock_fw}):
            provider = create_stt_provider(config)
            assert isinstance(provider, FasterWhisperSTTProvider)

    def test_factory_falls_back_to_speech_recognition(self):
        """Factory falls back to speech_recognition when faster-whisper unavailable."""
        config = AudioConfig(stt_enabled=True, stt_provider="faster_whisper")
        mock_sr_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {
            "faster_whisper": None,
            "speech_recognition": mock_sr_mod,
        }):
            provider = create_stt_provider(config)
            assert isinstance(provider, SpeechRecognitionSTTProvider)

    def test_factory_falls_back_to_noop(self):
        """Factory falls back to noop when all providers unavailable."""
        config = AudioConfig(stt_enabled=True, stt_provider="faster_whisper")
        with patch.dict(sys.modules, {
            "faster_whisper": None,
            "speech_recognition": None,
        }):
            provider = create_stt_provider(config)
            assert isinstance(provider, NoopSTTProvider)
