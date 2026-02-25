"""Text-to-speech provider abstraction and implementations.

Provides a TTSProvider Protocol for pluggable TTS backends, with a pyttsx3
offline implementation as the default. TTS runs on CPU to preserve GPU for LLM.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
import wave
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, runtime_checkable

from seaman_brain.config import AudioConfig

logger = logging.getLogger(__name__)

# Shared thread pool for TTS operations (CPU-bound, keep off event loop)
_tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts")


@runtime_checkable
class TTSProvider(Protocol):
    """Protocol for text-to-speech providers.

    Implementations must provide async synthesize() and speak() methods.
    synthesize() returns raw audio bytes (WAV format).
    speak() plays audio through the default output device.
    """

    async def synthesize(self, text: str) -> bytes:
        """Convert text to audio bytes (WAV format).

        Args:
            text: The text to synthesize.

        Returns:
            Raw WAV audio bytes.
        """
        ...

    async def speak(self, text: str) -> None:
        """Speak text through the default audio output.

        Args:
            text: The text to speak aloud.
        """
        ...


class NoopTTSProvider:
    """Silent TTS provider used as fallback when no TTS engine is available."""

    async def synthesize(self, text: str) -> bytes:
        """Return empty WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"")
        return buf.getvalue()

    async def speak(self, text: str) -> None:
        """Do nothing — silent mode."""
        logger.debug("NoopTTS: would speak %r", text[:50] if text else "")


class Pyttsx3TTSProvider:
    """Offline TTS provider using pyttsx3 (eSpeak/SAPI5/NSSpeechSynthesizer).

    Runs all pyttsx3 calls in a dedicated thread to avoid blocking the
    async event loop. pyttsx3 engines are NOT thread-safe, so all operations
    are serialized through a single-thread executor.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self._engine_factory: Callable | None = None
        self._available = False
        self._init_error: str = ""
        self._initialize()

    def _initialize(self) -> None:
        """Try to import pyttsx3 and verify engine creation."""
        try:
            import pyttsx3
            self._engine_factory = pyttsx3.init
            # Test that an engine can be created
            engine = pyttsx3.init()
            engine.stop()
            self._available = True
        except Exception as exc:
            self._available = False
            self._init_error = str(exc)
            logger.warning("pyttsx3 TTS unavailable: %s", exc)

    @property
    def available(self) -> bool:
        """Whether the TTS engine is operational."""
        return self._available

    def _create_engine(self):
        """Create and configure a fresh pyttsx3 engine."""
        import pyttsx3

        engine = pyttsx3.init()

        # Apply voice settings
        if self._config.tts_voice:
            voices = engine.getProperty("voices")
            for voice in voices:
                if self._config.tts_voice.lower() in voice.id.lower():
                    engine.setProperty("voice", voice.id)
                    break

        engine.setProperty("rate", self._config.tts_rate)
        engine.setProperty("volume", max(0.0, min(1.0, self._config.tts_volume)))

        return engine

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis — runs in thread pool."""
        if not self._available:
            raise RuntimeError(f"TTS engine unavailable: {self._init_error}")

        if not text or not text.strip():
            return self._empty_wav()

        engine = self._create_engine()
        try:
            # Save to temp file and read back
            with tempfile.NamedTemporaryFile(
                suffix=".wav", delete=False
            ) as tmp:
                tmp_path = tmp.name

            engine.save_to_file(text.strip(), tmp_path)
            engine.runAndWait()

            path = Path(tmp_path)
            if path.exists() and path.stat().st_size > 0:
                data = path.read_bytes()
                path.unlink(missing_ok=True)
                return data

            # File missing or empty — engine may have failed silently
            path.unlink(missing_ok=True)
            return self._empty_wav()
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise
        finally:
            engine.stop()

    def _speak_sync(self, text: str) -> None:
        """Synchronous speak — runs in thread pool."""
        if not self._available:
            logger.warning("TTS unavailable, skipping speak")
            return

        if not text or not text.strip():
            return

        engine = self._create_engine()
        try:
            engine.say(text.strip())
            engine.runAndWait()
        finally:
            engine.stop()

    @staticmethod
    def _empty_wav() -> bytes:
        """Generate minimal valid WAV with no audio data."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(22050)
            wf.writeframes(b"")
        return buf.getvalue()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to WAV audio bytes asynchronously.

        Args:
            text: The text to synthesize.

        Returns:
            Raw WAV audio bytes.

        Raises:
            RuntimeError: If the TTS engine is unavailable.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_tts_executor, self._synthesize_sync, text)

    async def speak(self, text: str) -> None:
        """Speak text through default audio output asynchronously.

        Args:
            text: The text to speak aloud.
        """
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_tts_executor, self._speak_sync, text)


def create_tts_provider(config: AudioConfig | None = None) -> TTSProvider:
    """Create a TTS provider based on configuration.

    Falls back to NoopTTSProvider if the configured provider is unavailable.

    Args:
        config: Audio configuration. Uses defaults if None.

    Returns:
        A TTSProvider instance.
    """
    config = config or AudioConfig()

    if not config.tts_enabled:
        logger.info("TTS disabled by configuration")
        return NoopTTSProvider()

    provider_name = config.tts_provider.lower()

    if provider_name == "pyttsx3":
        provider = Pyttsx3TTSProvider(config)
        if provider.available:
            logger.info("TTS initialized: pyttsx3")
            return provider
        logger.warning("pyttsx3 unavailable, falling back to silent mode")
        return NoopTTSProvider()

    logger.warning("Unknown TTS provider %r, falling back to silent mode", provider_name)
    return NoopTTSProvider()
