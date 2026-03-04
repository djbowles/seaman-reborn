"""Unified audio manager coordinating TTS, STT, and sound effects.

Provides a single entry point for all audio operations (speak, listen, SFX)
with per-channel enable/disable and thread-safe concurrent usage. Supports
optional full-duplex mode with AEC pipeline for barge-in.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from seaman_brain.audio.stt import NoopSTTProvider, STTProvider, create_stt_provider
from seaman_brain.audio.tts import TTSProvider, create_tts_provider
from seaman_brain.config import AudioConfig

if TYPE_CHECKING:
    from seaman_brain.audio.pipeline import AudioIOPipeline

logger = logging.getLogger(__name__)

_TTS_FALLBACK_THRESHOLD = 3
_LISTEN_TIMEOUT = 15.0  # seconds for full-duplex utterance wait


class AudioManager:
    """Unified audio manager coordinating TTS, STT, and sound effects.

    Thread-safe for concurrent GUI usage. Each channel (tts, stt, sfx) can be
    independently enabled or disabled at runtime.

    When ``aec_enabled=True``, operates in full-duplex mode with an
    ``AudioIOPipeline`` for continuous mic processing, AEC, and barge-in.
    """

    def __init__(
        self,
        config: AudioConfig | None = None,
        tts_provider: TTSProvider | None = None,
        stt_provider: STTProvider | None = None,
        sounds_dir: str | Path | None = None,
    ) -> None:
        self._config = config or AudioConfig()
        self._tts: TTSProvider = tts_provider or create_tts_provider(self._config)
        self._stt: STTProvider = stt_provider or create_stt_provider(self._config)
        self._sounds_dir = Path(sounds_dir) if sounds_dir else Path("assets/sounds")

        # TTS failure tracking for auto-fallback
        self._tts_fail_count: int = 0

        # Per-channel enable/disable (runtime toggle)
        self._tts_enabled: bool = self._config.tts_enabled
        self._stt_enabled: bool = self._config.stt_enabled
        self._sfx_enabled: bool = self._config.sfx_enabled
        self._sfx_volume: float = max(0.0, min(1.0, self._config.sfx_volume))

        # Echo suppression: pause STT while TTS is playing + cooldown after
        self._is_speaking: bool = False
        self._speaking_until: float = 0.0  # monotonic time; STT blocked until past
        self._echo_cooldown: float = 0.5  # seconds to keep STT paused after TTS ends

        # Lock for thread-safe SFX playback
        self._sfx_lock = asyncio.Lock()

        # Full-duplex pipeline (created when aec_enabled=True)
        self._pipeline: AudioIOPipeline | None = None
        self._pending_utterance: asyncio.Queue[str] | None = None
        self._barge_in_event: asyncio.Event | None = None

        if self._config.aec_enabled:
            self._init_pipeline()

    @property
    def tts_provider(self) -> TTSProvider:
        """Current TTS provider."""
        return self._tts

    @property
    def stt_provider(self) -> STTProvider:
        """Current STT provider."""
        return self._stt

    @property
    def tts_enabled(self) -> bool:
        """Whether TTS output is enabled."""
        return self._tts_enabled

    @tts_enabled.setter
    def tts_enabled(self, value: bool) -> None:
        self._tts_enabled = value
        logger.info("TTS %s", "enabled" if value else "disabled")

    @property
    def stt_enabled(self) -> bool:
        """Whether STT input is enabled."""
        return self._stt_enabled

    @stt_enabled.setter
    def stt_enabled(self, value: bool) -> None:
        self._stt_enabled = value
        logger.info("STT %s", "enabled" if value else "disabled")
        # Recreate STT provider if enabling and current is noop
        if value and isinstance(self._stt, NoopSTTProvider):
            if not self._try_upgrade_stt():
                self._stt_enabled = False

    def _try_upgrade_stt(self) -> bool:
        """Attempt to replace NoopSTTProvider with a real one.

        Returns:
            True if upgrade succeeded, False otherwise.
        """
        logger.info("Attempting STT upgrade from %s...", type(self._stt).__name__)
        self._config.stt_enabled = True
        try:
            provider = create_stt_provider(self._config)
        except Exception:
            logger.warning("STT provider creation failed", exc_info=True)
            return False
        if isinstance(provider, NoopSTTProvider):
            logger.warning(
                "STT upgrade returned NoopSTTProvider — check that PyAudio "
                "and speech_recognition are installed"
            )
            return False
        self._stt = provider
        logger.info("STT provider upgraded to %s", type(provider).__name__)
        return True

    def _try_fallback_tts(self) -> None:
        """Switch from Kokoro to pyttsx3 after repeated TTS failures."""
        if self._tts_fail_count < _TTS_FALLBACK_THRESHOLD:
            return
        from seaman_brain.audio.tts import KokoroTTSProvider, Pyttsx3TTSProvider
        if not isinstance(self._tts, KokoroTTSProvider):
            return
        logger.warning(
            "Kokoro TTS failed %d times, falling back to pyttsx3",
            self._tts_fail_count,
        )
        try:
            fallback = Pyttsx3TTSProvider(self._config)
            if fallback.available:
                self._tts = fallback
                self._tts_fail_count = 0
                logger.info("TTS provider switched to pyttsx3 (fallback)")
            else:
                logger.warning("pyttsx3 fallback also unavailable")
        except Exception as exc:
            logger.warning("pyttsx3 fallback creation failed: %s", exc)

    def set_input_device(self, device_name: str) -> None:
        """Change the STT microphone device at runtime.

        Args:
            device_name: Device name from settings.
        """
        self._stt.set_input_device(device_name)

    def update_tts_voice(self, voice_name: str) -> None:
        """Update the TTS voice at runtime.

        Normalizes "System Default" to empty string (engine default).
        Updates both the manager config and the provider config so the
        next speak/synthesize call picks up the new voice.

        Args:
            voice_name: Display name of the voice, or "System Default".
        """
        normalized = "" if voice_name == "System Default" else voice_name
        self._config.tts_voice = normalized
        # Also update the provider's config if it's pyttsx3
        from seaman_brain.audio.tts import Pyttsx3TTSProvider
        if isinstance(self._tts, Pyttsx3TTSProvider):
            self._tts._config.tts_voice = normalized
        logger.info("TTS voice updated to %r", normalized or "(system default)")

    @property
    def is_speaking(self) -> bool:
        """Whether TTS is playing or cooldown is active (for echo suppression)."""
        return self._is_speaking or time.monotonic() < self._speaking_until

    @property
    def sfx_enabled(self) -> bool:
        """Whether sound effects are enabled."""
        return self._sfx_enabled

    @sfx_enabled.setter
    def sfx_enabled(self, value: bool) -> None:
        self._sfx_enabled = value
        logger.info("SFX %s", "enabled" if value else "disabled")

    @property
    def sfx_volume(self) -> float:
        """Current SFX volume (0.0 to 1.0)."""
        return self._sfx_volume

    @sfx_volume.setter
    def sfx_volume(self, value: float) -> None:
        self._sfx_volume = max(0.0, min(1.0, value))

    async def speak(self, text: str) -> None:
        """Speak text through the TTS provider.

        In full-duplex mode, synthesizes WAV, feeds it to the pipeline as
        reference, and plays via sounddevice. In half-duplex mode, sets
        ``is_speaking`` True for the duration so STT can pause.

        Args:
            text: The text to speak aloud.
        """
        if not self._tts_enabled:
            logger.debug("TTS disabled, skipping speak")
            return
        if not text or not text.strip():
            return

        # Full-duplex: synthesize -> feed reference -> play
        if self._pipeline is not None:
            try:
                wav_bytes = await self._tts.synthesize(text)
                self._tts_fail_count = 0
                if wav_bytes and len(wav_bytes) > 44:
                    self._pipeline.feed_reference(wav_bytes)
                    await self._play_wav_async(wav_bytes)
            except Exception as exc:
                logger.warning("TTS speak failed (full-duplex): %s", exc)
                self._tts_fail_count += 1
                self._try_fallback_tts()
            return

        # Half-duplex: original behavior
        self._is_speaking = True
        try:
            await self._tts.speak(text)
            self._tts_fail_count = 0
        except Exception as exc:
            logger.warning("TTS speak failed: %s", exc)
            self._tts_fail_count += 1
            self._try_fallback_tts()
        finally:
            self._is_speaking = False
            # Keep STT paused briefly after TTS stops so residual speaker
            # audio doesn't get picked up by the microphone.
            self._speaking_until = time.monotonic() + self._echo_cooldown

    async def synthesize(self, text: str) -> bytes:
        """Synthesize text to audio bytes via the TTS provider.

        Returns empty bytes if TTS is disabled or text is empty.

        Args:
            text: The text to synthesize.

        Returns:
            Raw WAV audio bytes, or b"" if disabled.
        """
        if not self._tts_enabled:
            logger.debug("TTS disabled, skipping synthesize")
            return b""
        if not text or not text.strip():
            return b""
        try:
            result = await self._tts.synthesize(text)
            self._tts_fail_count = 0
            return result
        except Exception as exc:
            logger.warning("TTS synthesize failed: %s", exc)
            self._tts_fail_count += 1
            self._try_fallback_tts()
            return b""

    async def listen(self) -> str:
        """Listen for speech via the STT provider.

        In full-duplex mode, waits for transcribed utterance from the
        pipeline queue. In half-duplex mode, waits for TTS to finish
        first (echo suppression) then listens.

        Returns:
            Transcribed text, or empty string on timeout/failure/disabled.
        """
        if not self._stt_enabled:
            logger.debug("STT disabled, skipping listen")
            return ""

        # Full-duplex: get from pipeline utterance queue
        if self._pipeline is not None and self._pending_utterance is not None:
            try:
                return await asyncio.wait_for(
                    self._pending_utterance.get(),
                    timeout=_LISTEN_TIMEOUT,
                )
            except TimeoutError:
                return ""
            except Exception as exc:
                logger.warning("Full-duplex listen failed: %s", exc)
                return ""

        # Half-duplex: echo suppression wait + STT listen
        while self._is_speaking or time.monotonic() < self._speaking_until:
            await asyncio.sleep(0.05)
        try:
            return await self._stt.listen()
        except Exception as exc:
            logger.warning("STT listen failed: %s", exc)
            return ""

    # ── Full-duplex pipeline methods ─────────────────────────────────

    def _init_pipeline(self) -> None:
        """Create the full-duplex audio pipeline."""
        from seaman_brain.audio.pipeline import AudioIOPipeline

        self._pending_utterance = asyncio.Queue()
        self._barge_in_event = asyncio.Event()
        self._pipeline = AudioIOPipeline(
            config=self._config,
            on_utterance=self._on_pipeline_utterance,
            on_barge_in=self._on_pipeline_barge_in,
        )

    @property
    def full_duplex(self) -> bool:
        """Whether the manager is in full-duplex mode."""
        return self._pipeline is not None

    @property
    def barge_in_event(self) -> asyncio.Event | None:
        """Event set when barge-in is detected (full-duplex only)."""
        return self._barge_in_event

    def start_pipeline(self) -> None:
        """Start the full-duplex audio pipeline."""
        if self._pipeline is not None:
            self._pipeline.start()
            logger.info("Full-duplex audio pipeline started")

    def stop_pipeline(self) -> None:
        """Stop the full-duplex audio pipeline."""
        if self._pipeline is not None:
            self._pipeline.stop()
            logger.info("Full-duplex audio pipeline stopped")

    def cancel_tts(self) -> None:
        """Cancel current TTS playback and clear pipeline reference."""
        if self._pipeline is not None:
            self._pipeline.clear_reference()
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        self._is_speaking = False
        logger.debug("TTS cancelled")

    def _on_pipeline_utterance(self, pcm_bytes: bytes) -> None:
        """Callback from pipeline thread: transcribe and enqueue result."""
        if self._pending_utterance is None:
            return

        # Check if STT provider has transcribe() (duck typing)
        if hasattr(self._stt, "transcribe"):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._transcribe_and_enqueue(pcm_bytes), loop
                    )
                    return
            except RuntimeError:
                pass

        # Fallback: put raw bytes info as empty (no transcribe method)
        logger.debug("STT provider lacks transcribe(), dropping utterance")

    async def _transcribe_and_enqueue(self, pcm_bytes: bytes) -> None:
        """Transcribe PCM bytes and put result in utterance queue."""
        try:
            text = await self._stt.transcribe(pcm_bytes)
            if text and text.strip() and self._pending_utterance is not None:
                await self._pending_utterance.put(text.strip())
        except Exception as exc:
            logger.warning("Pipeline transcription failed: %s", exc)

    def _on_pipeline_barge_in(self) -> None:
        """Callback from pipeline thread: signal barge-in event."""
        if self._barge_in_event is not None:
            self._barge_in_event.set()
        logger.debug("Barge-in detected")

    async def _play_wav_async(self, wav_bytes: bytes) -> None:
        """Play WAV bytes via sounddevice in executor."""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._play_wav_sd, wav_bytes)

    @staticmethod
    def _play_wav_sd(wav_bytes: bytes) -> None:
        """Synchronous WAV playback via sounddevice."""
        try:
            import io as _io
            import wave as _wave

            import numpy as np
            import sounddevice as sd

            buf = _io.BytesIO(wav_bytes)
            with _wave.open(buf, "rb") as wf:
                raw = wf.readframes(wf.getnframes())
                rate = wf.getframerate()
            data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            sd.play(data, rate)
            sd.wait()
        except Exception as exc:
            logger.warning("sounddevice playback failed: %s", exc)

    async def play_sfx(self, sound_name: str) -> None:
        """Play a sound effect from the sounds directory.

        Looks for <sound_name>.wav in the configured sounds directory.
        No-op if SFX is disabled or the sound file doesn't exist.

        Args:
            sound_name: Name of the sound (without extension).
        """
        if not self._sfx_enabled:
            logger.debug("SFX disabled, skipping %s", sound_name)
            return
        if not sound_name or not sound_name.strip():
            return

        sound_path = self._sounds_dir / f"{sound_name.strip()}.wav"
        if not sound_path.exists():
            logger.warning("SFX file not found: %s", sound_path)
            return

        async with self._sfx_lock:
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._play_wav, sound_path)
            except Exception as exc:
                logger.warning("SFX playback failed for %s: %s", sound_name, exc)

    def _play_wav(self, path: Path) -> None:
        """Synchronous WAV playback — runs in executor.

        Uses wave module to read and simpleaudio/winsound to play.
        Falls back gracefully if no audio backend is available.
        """
        try:
            import winsound
            winsound.PlaySound(
                str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT
            )
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("winsound failed: %s", exc)

        # Fallback: try simpleaudio
        try:
            import simpleaudio as sa
            wave_obj = sa.WaveObject.from_wave_file(str(path))
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return
        except ImportError:
            pass
        except Exception as exc:
            logger.debug("simpleaudio failed: %s", exc)

        logger.warning("No audio playback backend available for SFX")

    def set_channel(self, channel: str, enabled: bool) -> None:
        """Enable or disable an audio channel by name.

        Args:
            channel: One of "tts", "stt", "sfx".
            enabled: Whether the channel should be enabled.

        Raises:
            ValueError: If channel name is not recognized.
        """
        channel = channel.lower().strip()
        if channel == "tts":
            self.tts_enabled = enabled
        elif channel == "stt":
            self.stt_enabled = enabled
        elif channel == "sfx":
            self.sfx_enabled = enabled
        else:
            raise ValueError(f"Unknown audio channel: {channel!r}")

    def get_status(self) -> dict[str, bool]:
        """Return the enabled/disabled status of all channels.

        Returns:
            Dict with keys "tts", "stt", "sfx" and boolean values.
        """
        return {
            "tts": self._tts_enabled,
            "stt": self._stt_enabled,
            "sfx": self._sfx_enabled,
        }


def create_audio_manager(
    config: AudioConfig | None = None,
    sounds_dir: str | Path | None = None,
) -> AudioManager:
    """Create an AudioManager from configuration.

    Args:
        config: Audio configuration. Uses defaults if None.
        sounds_dir: Path to sound effects directory. Defaults to assets/sounds/.

    Returns:
        A configured AudioManager instance.
    """
    config = config or AudioConfig()
    return AudioManager(config=config, sounds_dir=sounds_dir)
