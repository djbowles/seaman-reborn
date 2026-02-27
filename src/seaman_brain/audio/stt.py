"""Speech-to-text provider abstraction and implementations.

Provides an STTProvider Protocol for pluggable STT backends, with a
SpeechRecognition-based implementation as the default. STT listen() is
blocking (waits for speech), so all operations run in a thread pool to
keep the async event loop free.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

from seaman_brain.config import AudioConfig

logger = logging.getLogger(__name__)

# Shared thread pool for STT operations (blocking mic I/O, keep off event loop)
_stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")


@runtime_checkable
class STTProvider(Protocol):
    """Protocol for speech-to-text providers.

    Implementations must provide an async listen() method that blocks
    until speech is detected and returns the transcribed text.
    """

    async def listen(self) -> str:
        """Listen for speech and return transcribed text.

        Blocks until speech is detected or timeout occurs.

        Returns:
            Transcribed text, or empty string on timeout/unintelligible input.
        """
        ...


class NoopSTTProvider:
    """Silent STT provider used as fallback when no microphone is available."""

    async def listen(self) -> str:
        """Return empty string — no microphone available."""
        logger.debug("NoopSTT: listen() called, returning empty")
        return ""


class SpeechRecognitionSTTProvider:
    """STT provider using the SpeechRecognition library.

    Supports configurable recognizer backends (google, vosk) from AudioConfig.
    Runs all blocking microphone operations in a dedicated thread pool.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self._available = False
        self._init_error: str = ""
        self._mic_index: int | None = None
        self._initialize()

    def _initialize(self) -> None:
        """Try to import speech_recognition and verify microphone access."""
        try:
            import speech_recognition as sr

            self._sr = sr
            self._recognizer = sr.Recognizer()
            # Test that Microphone can be instantiated
            with sr.Microphone() as _:
                pass
            self._available = True
        except ImportError as exc:
            self._available = False
            self._init_error = f"speech_recognition not installed: {exc}"
            logger.warning("STT unavailable: %s", self._init_error)
        except AttributeError as exc:
            # PyAudio not installed — Microphone raises AttributeError
            self._available = False
            self._init_error = f"PyAudio not available: {exc}"
            logger.warning("STT unavailable: %s", self._init_error)
        except OSError as exc:
            # No microphone device found
            self._available = False
            self._init_error = f"No microphone: {exc}"
            logger.warning("STT unavailable: %s", self._init_error)
        except Exception as exc:
            self._available = False
            self._init_error = str(exc)
            logger.warning("STT unavailable: %s", exc)

    @property
    def available(self) -> bool:
        """Whether the STT engine is operational."""
        return self._available

    def _resolve_mic_index(self, device_name: str) -> int | None:
        """Map an audio input device name to a PyAudio device index.

        Args:
            device_name: The device name from settings (e.g. "Portacapture X6").

        Returns:
            PyAudio device index, or None for system default.
        """
        if not device_name or device_name == "System Default":
            return None
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if (
                    info.get("maxInputChannels", 0) > 0
                    and device_name in info.get("name", "")
                ):
                    pa.terminate()
                    return i
            pa.terminate()
        except Exception as exc:
            logger.debug("Could not resolve mic device %r: %s", device_name, exc)
        return None

    def set_input_device(self, device_name: str) -> None:
        """Change the microphone device at runtime.

        Args:
            device_name: Device name from settings. "System Default" or empty
                resets to the default device.
        """
        self._mic_index = self._resolve_mic_index(device_name)
        logger.info(
            "STT input device set to: %s (index=%s)", device_name, self._mic_index
        )

    def _listen_sync(self) -> str:
        """Synchronous listen — runs in thread pool.

        Returns:
            Transcribed text, or empty string on timeout/failure.
        """
        if not self._available:
            logger.warning("STT unavailable, returning empty")
            return ""

        sr = self._sr
        try:
            with sr.Microphone(device_index=self._mic_index) as source:
                self._recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = self._recognizer.listen(source, timeout=5, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            logger.debug("STT: listen timeout, no speech detected")
            return ""
        except OSError as exc:
            logger.warning("STT: microphone error: %s", exc)
            return ""
        except Exception as exc:
            logger.warning("STT: unexpected error during listen: %s", exc)
            return ""

        return self._recognize(audio)

    def _recognize(self, audio: object) -> str:
        """Run the configured recognizer backend on captured audio.

        Args:
            audio: An AudioData instance from speech_recognition.

        Returns:
            Transcribed text, or empty string on failure.
        """
        sr = self._sr
        backend = self._config.stt_provider.lower()

        try:
            if backend == "vosk":
                text = self._recognizer.recognize_vosk(audio)
            elif backend in ("google", "speech_recognition"):
                text = self._recognizer.recognize_google(audio)
            else:
                # Default to Google as fallback recognizer
                logger.warning(
                    "Unknown STT backend %r, falling back to google", backend
                )
                text = self._recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            logger.debug("STT: speech unintelligible")
            return ""
        except sr.RequestError as exc:
            logger.warning("STT: recognizer request error: %s", exc)
            return ""
        except Exception as exc:
            logger.warning("STT: recognition error: %s", exc)
            return ""

        return text.strip() if text else ""

    async def listen(self) -> str:
        """Listen for speech and return transcribed text asynchronously.

        Blocks (in a thread pool) until speech is detected or timeout occurs.

        Returns:
            Transcribed text, or empty string on timeout/unintelligible input.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_stt_executor, self._listen_sync)


class FasterWhisperSTTProvider:
    """STT provider using Faster-Whisper for local GPU-accelerated transcription.

    Uses CTranslate2 for 5.4x realtime speed on CUDA. The model is
    lazy-loaded on first listen() to avoid startup VRAM allocation (~3GB).
    Includes RMS-based silence detection for continuous listening.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self._available = False
        self._init_error: str = ""
        self._model: object | None = None

    def _initialize(self) -> None:
        """Lazy-load WhisperModel on first use."""
        if self._model is not None:
            return
        try:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._config.stt_model,
                device="cuda",
                compute_type="float16",
            )
            self._available = True
        except ImportError as exc:
            self._available = False
            self._init_error = f"faster-whisper not installed: {exc}"
            logger.warning("Faster-Whisper STT unavailable: %s", self._init_error)
        except Exception as exc:
            self._available = False
            self._init_error = str(exc)
            logger.warning("Faster-Whisper STT init failed: %s", exc)

    @property
    def available(self) -> bool:
        """Whether the STT engine is operational."""
        return self._available

    def _listen_sync(self) -> str:
        """Synchronous listen with RMS-based silence detection.

        Returns:
            Transcribed text, or empty string on timeout/silence.
        """
        self._initialize()
        if not self._available or self._model is None:
            logger.warning("Faster-Whisper STT unavailable, returning empty")
            return ""

        try:
            import numpy as np
            import sounddevice as sd

            samplerate = 16000
            block_size = int(samplerate * 0.1)  # 100ms blocks
            threshold = self._config.stt_silence_threshold
            silence_dur = self._config.stt_silence_duration
            max_duration = 15.0  # Safety cutoff

            frames: list = []
            recording = False
            silence_blocks = 0
            silence_limit = int(silence_dur / 0.1)  # blocks
            max_blocks = int(max_duration / 0.1)
            total_blocks = 0

            # Resolve input device
            device = None
            if self._config.audio_input_device:
                device_name = self._config.audio_input_device
                if device_name != "System Default":
                    devices = sd.query_devices()
                    for i, dev in enumerate(devices):
                        if (
                            device_name in dev["name"]
                            and dev["max_input_channels"] > 0
                        ):
                            device = i
                            break

            with sd.InputStream(
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                device=device,
            ) as stream:
                while total_blocks < max_blocks:
                    data, _overflowed = stream.read(block_size)
                    total_blocks += 1

                    rms = float(np.sqrt(np.mean(data ** 2)))

                    if rms > threshold:
                        recording = True
                        silence_blocks = 0
                        frames.append(data.copy())
                    elif recording:
                        silence_blocks += 1
                        frames.append(data.copy())
                        if silence_blocks >= silence_limit:
                            break
                    # else: pre-speech silence, skip

            if not frames:
                return ""

            audio = np.concatenate(frames).flatten()

            segments, _info = self._model.transcribe(
                audio,
                language="en",
                vad_filter=True,
                beam_size=5,
            )

            text = " ".join(seg.text for seg in segments).strip()
            return text

        except Exception as exc:
            logger.warning("Faster-Whisper listen failed: %s", exc)
            return ""

    async def listen(self) -> str:
        """Listen for speech and return transcribed text asynchronously.

        Blocks (in a thread pool) until speech is detected, silence timeout,
        or max phrase cutoff.

        Returns:
            Transcribed text, or empty string on timeout/failure.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_stt_executor, self._listen_sync)


def create_stt_provider(config: AudioConfig | None = None) -> STTProvider:
    """Create an STT provider based on configuration.

    Falls back to NoopSTTProvider if the configured provider is unavailable.

    Args:
        config: Audio configuration. Uses defaults if None.

    Returns:
        An STTProvider instance.
    """
    config = config or AudioConfig()

    if not config.stt_enabled:
        logger.info("STT disabled by configuration")
        return NoopSTTProvider()

    provider_name = config.stt_provider.lower()

    if provider_name == "faster_whisper":
        fw_provider = FasterWhisperSTTProvider(config)
        fw_provider._initialize()
        if fw_provider.available:
            logger.info(
                "STT initialized: faster-whisper (model=%s)", config.stt_model
            )
            return fw_provider
        logger.warning("Faster-Whisper unavailable, falling back to speech_recognition")
        # Fall through to speech_recognition

    provider = SpeechRecognitionSTTProvider(config)
    if provider.available:
        logger.info("STT initialized: speech_recognition (backend=%s)", config.stt_provider)
        return provider

    logger.warning("STT unavailable, falling back to silent mode")
    return NoopSTTProvider()
