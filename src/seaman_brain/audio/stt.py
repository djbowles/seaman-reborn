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
            with sr.Microphone() as source:
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

    provider = SpeechRecognitionSTTProvider(config)
    if provider.available:
        logger.info("STT initialized: speech_recognition (backend=%s)", config.stt_provider)
        return provider

    logger.warning("STT unavailable, falling back to silent mode")
    return NoopSTTProvider()
