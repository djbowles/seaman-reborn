"""Tests for STT provider abstraction and implementations."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from seaman_brain.audio.stt import (
    FasterWhisperSTTProvider,
    NoopSTTProvider,
    RivaSTTProvider,
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

    def test_factory_faster_whisper_unavailable_returns_noop(self):
        """Factory returns noop when faster-whisper unavailable (no fallback chain)."""
        config = AudioConfig(stt_enabled=True, stt_provider="faster_whisper")
        with patch.dict(sys.modules, {"faster_whisper": None}):
            provider = create_stt_provider(config)
            assert isinstance(provider, NoopSTTProvider)


# ─── set_input_device ─────────────────────────────────────────────


class TestSetInputDevice:
    """Test set_input_device on all STT providers."""

    def test_noop_set_input_device_no_error(self):
        """NoopSTTProvider.set_input_device is a silent no-op."""
        provider = NoopSTTProvider()
        provider.set_input_device("Some Mic")  # Should not raise

    def test_speech_recognition_set_input_device(self):
        """SpeechRecognitionSTTProvider stores resolved mic index."""
        mock_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            # Mock _resolve_mic_index to return a known index
            with patch.object(provider, "_resolve_mic_index", return_value=3):
                provider.set_input_device("Portacapture X6")
            assert provider._mic_index == 3

    def test_speech_recognition_set_input_device_default(self):
        """System Default resets mic index to None."""
        mock_mod, _ = _mock_sr()
        with patch.dict(sys.modules, {"speech_recognition": mock_mod}):
            provider = SpeechRecognitionSTTProvider()
            provider.set_input_device("System Default")
            assert provider._mic_index is None

    def test_faster_whisper_set_input_device(self):
        """FasterWhisperSTTProvider stores device name in config."""
        config = AudioConfig()
        provider = FasterWhisperSTTProvider(config)
        provider.set_input_device("Test Microphone")
        assert provider._config.audio_input_device == "Test Microphone"

    def test_faster_whisper_set_input_device_default(self):
        """System Default stored as-is in config."""
        config = AudioConfig()
        provider = FasterWhisperSTTProvider(config)
        provider.set_input_device("System Default")
        assert provider._config.audio_input_device == "System Default"


# ─── FasterWhisperSTTProvider transcribe ──────────────────────────


class TestFasterWhisperTranscribe:
    """Test the transcribe() method for pre-captured audio."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self):
        """transcribe() returns transcribed text from PCM bytes."""
        import numpy as np

        mock_fw, mock_model = _mock_faster_whisper()

        mock_segment = MagicMock()
        mock_segment.text = "hello pipeline"
        mock_model.transcribe.return_value = (iter([mock_segment]), MagicMock())

        with patch.dict(sys.modules, {"faster_whisper": mock_fw, "numpy": np}):
            provider = FasterWhisperSTTProvider()
            provider._initialize()

            # Create PCM bytes (16kHz 16-bit mono)
            samples = np.zeros(1600, dtype=np.int16)
            result = await provider.transcribe(samples.tobytes())
            assert result == "hello pipeline"

    @pytest.mark.asyncio
    async def test_transcribe_unavailable_returns_empty(self):
        """transcribe() returns empty when model unavailable."""
        with patch.dict(sys.modules, {"faster_whisper": None}):
            provider = FasterWhisperSTTProvider()
            result = await provider.transcribe(b"\x00\x00" * 100)
            assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_error_returns_empty(self):
        """transcribe() handles errors gracefully."""
        import numpy as np

        mock_fw, mock_model = _mock_faster_whisper()
        mock_model.transcribe.side_effect = RuntimeError("decode error")

        with patch.dict(sys.modules, {"faster_whisper": mock_fw, "numpy": np}):
            provider = FasterWhisperSTTProvider()
            provider._initialize()
            result = await provider.transcribe(b"\x00\x00" * 100)
            assert result == ""


# ─── RivaSTTProvider ──────────────────────────────────────────────


@contextmanager
def _grpc_reachable():
    """Mock gRPC connectivity check to report server as reachable."""
    mock_channel = MagicMock()
    mock_future = MagicMock()
    mock_future.result.return_value = None
    with (
        patch("grpc.insecure_channel", return_value=mock_channel),
        patch("grpc.channel_ready_future", return_value=mock_future),
    ):
        yield


def _mock_riva_stt():
    """Create mock riva.client module for STT tests."""
    mock_riva = MagicMock()
    mock_client = MagicMock()

    mock_auth = MagicMock()
    mock_service = MagicMock()
    mock_client.Auth.return_value = mock_auth
    mock_client.ASRService.return_value = mock_service
    mock_client.RecognitionConfig = MagicMock
    mock_client.AudioEncoding = MagicMock()
    mock_client.AudioEncoding.LINEAR_PCM = 1

    mock_riva.client = mock_client
    return mock_riva, mock_client, mock_service


class TestRivaSTTProviderInit:
    """Test initialization of RivaSTTProvider."""

    def test_implements_protocol(self):
        """RivaSTTProvider satisfies STTProvider protocol."""
        with patch.dict(sys.modules, {"riva": MagicMock(), "riva.client": MagicMock()}):
            provider = RivaSTTProvider()
        assert isinstance(provider, STTProvider)

    def test_init_success(self):
        mock_riva, mock_client, _ = _mock_riva_stt()
        with (
            patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}),
            _grpc_reachable(),
        ):
            provider = RivaSTTProvider()
            assert provider.available is True

    def test_init_import_error(self):
        with patch.dict(sys.modules, {"riva": None, "riva.client": None}):
            provider = RivaSTTProvider()
            assert provider.available is False
            assert "not installed" in provider._init_error

    def test_init_connection_error(self):
        mock_riva, mock_client, _ = _mock_riva_stt()
        mock_client.Auth.side_effect = RuntimeError("Connection refused")
        with patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}):
            provider = RivaSTTProvider()
            assert provider.available is False

    def test_init_grpc_unreachable(self):
        """gRPC connectivity timeout marks provider unavailable."""
        import grpc

        mock_riva, mock_client, _ = _mock_riva_stt()
        mock_future = MagicMock()
        mock_future.result.side_effect = grpc.FutureTimeoutError()
        with (
            patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}),
            patch("grpc.insecure_channel", return_value=MagicMock()),
            patch("grpc.channel_ready_future", return_value=mock_future),
        ):
            provider = RivaSTTProvider()
            assert provider.available is False
            assert "not reachable" in provider._init_error

    def test_set_input_device(self):
        """set_input_device stores device name."""
        with patch.dict(sys.modules, {"riva": MagicMock(), "riva.client": MagicMock()}):
            provider = RivaSTTProvider()
        provider.set_input_device("Test Mic")
        assert provider._config.audio_input_device == "Test Mic"


class TestRivaSTTProviderTranscribe:
    """Test Riva STT transcription."""

    @pytest.mark.asyncio
    async def test_transcribe_returns_text(self):
        """transcribe() returns text from PCM bytes."""
        mock_riva, mock_client, mock_service = _mock_riva_stt()

        mock_alt = MagicMock()
        mock_alt.transcript = "hello riva"
        mock_result = MagicMock()
        mock_result.alternatives = [mock_alt]
        mock_resp = MagicMock()
        mock_resp.results = [mock_result]
        mock_service.offline_recognize.return_value = mock_resp

        with (
            patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}),
            _grpc_reachable(),
        ):
            provider = RivaSTTProvider()
            result = await provider.transcribe(b"\x00\x00" * 1600)
            assert result == "hello riva"

    @pytest.mark.asyncio
    async def test_transcribe_unavailable_returns_empty(self):
        with patch.dict(sys.modules, {"riva": None, "riva.client": None}):
            provider = RivaSTTProvider()
            result = await provider.transcribe(b"\x00\x00" * 100)
            assert result == ""

    @pytest.mark.asyncio
    async def test_transcribe_error_returns_empty(self):
        mock_riva, mock_client, mock_service = _mock_riva_stt()
        mock_service.offline_recognize.side_effect = RuntimeError("gRPC error")

        with (
            patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}),
            _grpc_reachable(),
        ):
            provider = RivaSTTProvider()
            result = await provider.transcribe(b"\x00\x00" * 100)
            assert result == ""


class TestRivaSTTFactory:
    """Test factory with Riva provider."""

    def test_factory_creates_riva_when_available(self):
        config = AudioConfig(stt_enabled=True, stt_provider="riva")
        mock_riva, mock_client, _ = _mock_riva_stt()
        with (
            patch.dict(sys.modules, {"riva": mock_riva, "riva.client": mock_client}),
            _grpc_reachable(),
        ):
            provider = create_stt_provider(config)
            assert isinstance(provider, RivaSTTProvider)

    def test_factory_riva_unavailable_returns_noop(self):
        """Factory returns noop when Riva unavailable (no fallback chain)."""
        config = AudioConfig(stt_enabled=True, stt_provider="riva")
        with patch.dict(sys.modules, {
            "riva": None, "riva.client": None,
        }):
            provider = create_stt_provider(config)
            assert isinstance(provider, NoopSTTProvider)
