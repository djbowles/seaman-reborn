"""Text-to-speech provider abstraction and implementations.

Provides a TTSProvider Protocol for pluggable TTS backends, with a pyttsx3
offline implementation as the default. TTS runs on CPU to preserve GPU for LLM.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
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

_TTS_TIMEOUT = 30.0  # seconds before TTS executor call is abandoned


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

        # Apply voice settings — match against both name and id
        if self._config.tts_voice:
            voices = engine.getProperty("voices")
            target = self._config.tts_voice.lower()
            for voice in voices:
                if target in voice.name.lower() or target in voice.id.lower():
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
                if len(data) <= 44:
                    logger.warning("TTS produced header-only WAV (no audio data)")
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
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_tts_executor, self._synthesize_sync, text),
                timeout=_TTS_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("TTS synthesize timed out after %.0fs", _TTS_TIMEOUT)
            return self._empty_wav()

    async def speak(self, text: str) -> None:
        """Speak text through default audio output asynchronously.

        Args:
            text: The text to speak aloud.
        """
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(_tts_executor, self._speak_sync, text),
                timeout=_TTS_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("TTS speak timed out after %.0fs", _TTS_TIMEOUT)


class KokoroTTSProvider:
    """Neural TTS provider using Kokoro for natural-sounding speech.

    Kokoro runs locally on GPU (~2GB VRAM) and produces high-quality
    24kHz audio. The model is lazy-loaded on first use to avoid VRAM
    allocation at startup. All operations run in a dedicated thread pool
    to keep the async event loop free.
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self._available = False
        self._init_error: str = ""
        self._pipeline: object | None = None
        self._last_failure_time: float = 0.0
        self._retry_interval: float = 60.0

    def _initialize(self) -> None:
        """Lazy-load Kokoro pipeline on first use."""
        if self._pipeline is not None:
            return
        # Skip retry if last failure was recent
        import time as _time
        if self._last_failure_time and (
            _time.monotonic() - self._last_failure_time < self._retry_interval
        ):
            return
        try:
            from kokoro import KPipeline

            lang = "a"  # American English
            # Force CPU to avoid VRAM contention with Ollama's LLM.
            # Kokoro is only 82M params (~300MB) — fast enough on CPU.
            self._pipeline = KPipeline(lang_code=lang, device="cpu")
            self._available = True
        except ImportError as exc:
            self._available = False
            self._init_error = f"kokoro not installed: {exc}"
            self._last_failure_time = _time.monotonic()
            logger.warning("Kokoro TTS unavailable: %s", self._init_error)
        except Exception as exc:
            self._available = False
            self._pipeline = None
            self._init_error = str(exc)
            self._last_failure_time = _time.monotonic()
            logger.warning("Kokoro TTS init failed: %s", exc)

    @property
    def available(self) -> bool:
        """Whether the TTS engine is operational."""
        return self._available

    @staticmethod
    def _clean_for_tts(text: str) -> str:
        """Strip markup that crashes Kokoro's G2P (misaki).

        The G2P tokenizer produces tokens with ``phonemes=None`` for certain
        input patterns, which causes a ``TypeError`` inside ``misaki/en.py``.
        Remove ``<think>…</think>`` reasoning blocks (Qwen3), remaining
        ``<…>`` tags, and ``*action*`` markers so only speakable text
        reaches the pipeline.
        """
        # Strip full <think>...</think> blocks first (Qwen3 reasoning)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        # Strip any remaining <...> tags
        text = re.sub(r"<[^>]*>", "", text)
        # Strip asterisks (LLM emphasis/action markers)
        text = text.replace("*", "")
        text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def _synthesize_sync(self, text: str) -> bytes:
        """Synchronous synthesis — runs in thread pool."""
        self._initialize()
        if not self._available or self._pipeline is None:
            raise RuntimeError(f"Kokoro TTS unavailable: {self._init_error}")

        if not text or not text.strip():
            return self._empty_wav()

        try:
            import soundfile as sf

            voice = self._config.tts_voice or "af_heart"
            speed = max(0.5, min(2.0, self._config.tts_speed))
            text = self._clean_for_tts(text)

            if not text:
                return self._empty_wav()

            # Split into sentences and process each individually.
            # Kokoro's G2P (misaki) crashes with TypeError on unknown words
            # and certain punctuation patterns (e.g. "word - word").
            # Per-sentence processing lets us skip failures while still
            # producing audio for the sentences that succeed.
            sentences = re.split(r"(?<=[.!?])\s+", text)
            audio_chunks: list = []
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    for _gs, _ps, audio in self._pipeline(
                        sentence, voice=voice, speed=speed
                    ):
                        audio_chunks.append(audio)
                except TypeError:
                    logger.debug("Kokoro G2P skipped sentence: %r", sentence[:60])

            if not audio_chunks:
                return self._empty_wav()

            import numpy as np

            full_audio = np.concatenate(audio_chunks)

            # Write to WAV bytes
            buf = io.BytesIO()
            sf.write(buf, full_audio, 24000, format="WAV", subtype="PCM_16")
            return buf.getvalue()
        except Exception as exc:
            logger.warning("Kokoro synthesize failed: %s", exc, exc_info=True)
            raise

    def _speak_sync(self, text: str) -> None:
        """Synchronous speak — synthesize then play via sounddevice."""
        self._initialize()
        if not self._available:
            logger.warning("Kokoro TTS unavailable, skipping speak")
            return

        if not text or not text.strip():
            return

        try:
            import soundfile as sf

            wav_bytes = self._synthesize_sync(text)
            buf = io.BytesIO(wav_bytes)
            data, samplerate = sf.read(buf)

            import sounddevice as sd

            sd.play(data, samplerate)
            sd.wait()
        except Exception as exc:
            logger.warning("Kokoro speak failed: %s", exc, exc_info=True)

    @staticmethod
    def _empty_wav() -> bytes:
        """Generate minimal valid WAV with no audio data."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"")
        return buf.getvalue()

    async def synthesize(self, text: str) -> bytes:
        """Convert text to WAV audio bytes asynchronously.

        Args:
            text: The text to synthesize.

        Returns:
            Raw WAV audio bytes (24kHz 16-bit PCM).

        Raises:
            RuntimeError: If the TTS engine is unavailable.
        """
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_tts_executor, self._synthesize_sync, text),
                timeout=_TTS_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("Kokoro TTS synthesize timed out after %.0fs", _TTS_TIMEOUT)
            return self._empty_wav()

    async def speak(self, text: str) -> None:
        """Speak text through default audio output asynchronously.

        Args:
            text: The text to speak aloud.
        """
        loop = asyncio.get_running_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(_tts_executor, self._speak_sync, text),
                timeout=_TTS_TIMEOUT,
            )
        except TimeoutError:
            logger.warning("Kokoro TTS speak timed out after %.0fs", _TTS_TIMEOUT)


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

    if provider_name == "kokoro":
        kokoro_provider = KokoroTTSProvider(config)
        # Trigger lazy init to check availability
        kokoro_provider._initialize()
        if kokoro_provider.available:
            logger.info("TTS initialized: kokoro (voice=%s)", config.tts_voice or "af_heart")
            return kokoro_provider
        logger.warning("Kokoro unavailable, falling back to pyttsx3")
        # Fall through to pyttsx3

    if provider_name in ("pyttsx3", "kokoro"):
        pyttsx3_provider = Pyttsx3TTSProvider(config)
        if pyttsx3_provider.available:
            logger.info("TTS initialized: pyttsx3")
            return pyttsx3_provider
        logger.warning("pyttsx3 unavailable, falling back to silent mode")
        return NoopTTSProvider()

    logger.warning("Unknown TTS provider %r, falling back to silent mode", provider_name)
    return NoopTTSProvider()
