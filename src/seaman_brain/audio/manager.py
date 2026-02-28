"""Unified audio manager coordinating TTS, STT, and sound effects.

Provides a single entry point for all audio operations (speak, listen, SFX)
with per-channel enable/disable and thread-safe concurrent usage.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from seaman_brain.audio.stt import NoopSTTProvider, STTProvider, create_stt_provider
from seaman_brain.audio.tts import TTSProvider, create_tts_provider
from seaman_brain.config import AudioConfig

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_TTS_FALLBACK_THRESHOLD = 3


class AudioManager:
    """Unified audio manager coordinating TTS, STT, and sound effects.

    Thread-safe for concurrent GUI usage. Each channel (tts, stt, sfx) can be
    independently enabled or disabled at runtime.
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

        # Lock for thread-safe SFX playback
        self._sfx_lock = asyncio.Lock()

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

        No-op if TTS is disabled or text is empty.

        Args:
            text: The text to speak aloud.
        """
        if not self._tts_enabled:
            logger.debug("TTS disabled, skipping speak")
            return
        if not text or not text.strip():
            return
        try:
            await self._tts.speak(text)
            self._tts_fail_count = 0
        except Exception as exc:
            logger.warning("TTS speak failed: %s", exc)
            self._tts_fail_count += 1
            self._try_fallback_tts()

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

        Returns empty string if STT is disabled.

        Returns:
            Transcribed text, or empty string on timeout/failure/disabled.
        """
        if not self._stt_enabled:
            logger.debug("STT disabled, skipping listen")
            return ""
        try:
            return await self._stt.listen()
        except Exception as exc:
            logger.warning("STT listen failed: %s", exc)
            return ""

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
