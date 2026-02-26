"""Pygame audio bridge - ambient sounds, creature voice, UI SFX.

Connects the AudioManager to Pygame's mixer system for in-game audio:
- Ambient loops (water bubbling for aquarium, nature sounds for terrarium)
- Creature voice output via TTS -> Pygame mixer
- UI sound effects (button clicks, feeding splash, glass tap, evolution chime)
- Per-channel volume controls (ambient, voice, sfx)
- Microphone input toggle for STT
- Graceful fallback to visual-only mode when audio hardware is missing
"""

from __future__ import annotations

import asyncio
import io
import logging
import threading
from enum import Enum
from typing import Any

from seaman_brain.config import AudioConfig

logger = logging.getLogger(__name__)


class AudioChannel(Enum):
    """Named audio channels with independent volume controls."""

    AMBIENT = "ambient"
    VOICE = "voice"
    SFX = "sfx"


class PygameAudioBridge:
    """Connects AudioManager to Pygame mixer for in-game audio.

    Manages three independent channels (ambient, voice, SFX) with per-channel
    volume. Handles TTS voice output by synthesizing to WAV bytes and playing
    through Pygame mixer. Ambient loops and UI SFX use Pygame Sound objects.

    Gracefully degrades to visual-only mode if Pygame mixer or audio hardware
    is unavailable.

    Attributes:
        audio_available: Whether Pygame mixer initialized successfully.
        mic_active: Whether microphone input (STT) is currently active.
    """

    def __init__(
        self,
        audio_manager: Any | None = None,
        audio_config: AudioConfig | None = None,
        async_loop: asyncio.AbstractEventLoop | None = None,
        on_stt_result: Any | None = None,
    ) -> None:
        """Initialize the Pygame audio bridge.

        Args:
            audio_manager: An AudioManager instance for TTS/STT.
            audio_config: Audio configuration for volume levels.
            async_loop: Background asyncio loop for async TTS calls.
            on_stt_result: Callback ``(text: str) -> None`` invoked with
                each transcribed STT result. If None, results are logged
                but not forwarded.
        """
        self._audio_manager = audio_manager
        self._config = audio_config or AudioConfig()
        self._async_loop = async_loop
        self._on_stt_result = on_stt_result

        # Channel volumes (0.0 to 1.0)
        self._volumes: dict[AudioChannel, float] = {
            AudioChannel.AMBIENT: max(0.0, min(1.0, self._config.ambient_volume)),
            AudioChannel.VOICE: max(0.0, min(1.0, self._config.tts_volume)),
            AudioChannel.SFX: max(0.0, min(1.0, self._config.sfx_volume)),
        }

        # Mixer state
        self.audio_available = False
        self._mixer_initialized = False
        self._ambient_channel: Any | None = None  # pygame.mixer.Channel
        self._voice_channel: Any | None = None
        self._sfx_channel: Any | None = None

        # Ambient loop state
        self._ambient_sound: Any | None = None  # pygame.mixer.Sound
        self._ambient_playing = False
        self._current_ambient: str = ""

        # Microphone state
        self.mic_active = False
        self._stt_lock = threading.Lock()

        # Initialize mixer
        self._init_mixer()

    def _init_mixer(self) -> None:
        """Try to initialize Pygame mixer with 3 channels."""
        try:
            import pygame.mixer

            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

            # Reserve 3 channels: ambient(0), voice(1), sfx(2)
            pygame.mixer.set_num_channels(max(pygame.mixer.get_num_channels(), 3))
            self._ambient_channel = pygame.mixer.Channel(0)
            self._voice_channel = pygame.mixer.Channel(1)
            self._sfx_channel = pygame.mixer.Channel(2)

            # Apply initial volumes
            self._ambient_channel.set_volume(self._volumes[AudioChannel.AMBIENT])
            self._voice_channel.set_volume(self._volumes[AudioChannel.VOICE])
            self._sfx_channel.set_volume(self._volumes[AudioChannel.SFX])

            self._mixer_initialized = True
            self.audio_available = True
            logger.info("Pygame mixer initialized for audio bridge")
        except Exception as exc:
            self.audio_available = False
            self._mixer_initialized = False
            logger.warning("Pygame mixer unavailable, visual-only mode: %s", exc)

    def get_volume(self, channel: AudioChannel) -> float:
        """Get the volume level for a channel.

        Args:
            channel: The audio channel.

        Returns:
            Volume level between 0.0 and 1.0.
        """
        return self._volumes.get(channel, 0.0)

    def set_volume(self, channel: AudioChannel, volume: float) -> None:
        """Set the volume level for a channel.

        Args:
            channel: The audio channel to adjust.
            volume: Volume level between 0.0 and 1.0.
        """
        volume = max(0.0, min(1.0, volume))
        self._volumes[channel] = volume

        if not self._mixer_initialized:
            return

        channel_obj = {
            AudioChannel.AMBIENT: self._ambient_channel,
            AudioChannel.VOICE: self._voice_channel,
            AudioChannel.SFX: self._sfx_channel,
        }.get(channel)

        if channel_obj is not None:
            channel_obj.set_volume(volume)
            logger.debug("Volume %s set to %.2f", channel.value, volume)

    def start_ambient(self, environment: str = "aquarium") -> None:
        """Start the ambient sound loop for the current environment.

        Args:
            environment: "aquarium" for water sounds, "terrarium" for nature.
        """
        if not self._mixer_initialized or self._ambient_channel is None:
            return

        # Don't restart if already playing the same ambient
        if self._ambient_playing and self._current_ambient == environment:
            return

        self.stop_ambient()

        sound_name = "water_bubbling" if environment == "aquarium" else "nature_ambient"

        try:
            import pygame.mixer

            from seaman_brain.audio.manager import AudioManager

            # Try to load the ambient sound file
            sounds_dir = "assets/sounds"
            if self._audio_manager and isinstance(self._audio_manager, AudioManager):
                sounds_dir = str(self._audio_manager._sounds_dir)

            import os
            sound_path = os.path.join(sounds_dir, f"{sound_name}.wav")

            if os.path.exists(sound_path):
                self._ambient_sound = pygame.mixer.Sound(sound_path)
                self._ambient_sound.set_volume(self._volumes[AudioChannel.AMBIENT])
                self._ambient_channel.play(self._ambient_sound, loops=-1)
                self._ambient_playing = True
                self._current_ambient = environment
                logger.info("Ambient loop started: %s", sound_name)
            else:
                # No ambient file available — silent ambient
                self._ambient_playing = False
                self._current_ambient = environment
                logger.debug("Ambient sound not found: %s (silent mode)", sound_path)
        except Exception as exc:
            self._ambient_playing = False
            logger.warning("Failed to start ambient loop: %s", exc)

    def stop_ambient(self) -> None:
        """Stop the ambient sound loop."""
        if self._ambient_channel is not None:
            try:
                self._ambient_channel.stop()
            except Exception as exc:
                logger.debug("Error stopping ambient: %s", exc)
        self._ambient_playing = False
        self._ambient_sound = None
        self._current_ambient = ""

    @property
    def ambient_playing(self) -> bool:
        """Whether ambient sound is currently looping."""
        return self._ambient_playing

    def play_voice(self, text: str) -> None:
        """Play creature voice via TTS.

        Prefers AudioManager.speak() which uses pyttsx3's native audio output
        (works with system default device). Falls back to synthesize → mixer
        path if speak() is unavailable.

        Args:
            text: The text for the creature to speak.
        """
        if not text or not text.strip():
            return
        if self._audio_manager is None:
            return

        if self._async_loop is not None:
            asyncio.run_coroutine_threadsafe(
                self._play_voice_async(text), self._async_loop
            )
        else:
            logger.debug("No async loop available for voice playback")

    async def _play_voice_async(self, text: str) -> None:
        """Async voice playback — use native speak() or synthesize fallback.

        Args:
            text: Text to synthesize and play.
        """
        if self._audio_manager is None:
            return

        try:
            # Prefer native speak() — uses pyttsx3's built-in audio output
            await self._audio_manager.speak(text)
        except Exception as exc:
            logger.warning("Native speak failed, trying synthesize fallback: %s", exc)
            try:
                wav_bytes = await self._audio_manager.synthesize(text)
                if wav_bytes and self._voice_channel is not None:
                    self._play_wav_bytes(wav_bytes, self._voice_channel)
            except Exception as exc2:
                logger.warning("Voice playback failed: %s", exc2)

    def _play_wav_bytes(self, wav_bytes: bytes, channel: Any) -> None:
        """Play WAV bytes through a Pygame mixer channel.

        Args:
            wav_bytes: Raw WAV audio data.
            channel: Pygame mixer channel to play on.
        """
        if not wav_bytes or channel is None:
            return

        try:
            import pygame.mixer

            sound = pygame.mixer.Sound(file=io.BytesIO(wav_bytes))
            channel.play(sound)
        except Exception as exc:
            logger.warning("Failed to play WAV bytes: %s", exc)

    def play_sfx(self, sound_name: str) -> None:
        """Play a UI sound effect.

        Supported SFX names: button_click, feeding_splash, glass_tap,
        evolution_chime, and any .wav file in the sounds directory.

        Args:
            sound_name: Name of the sound effect (without .wav extension).
        """
        if not sound_name or not sound_name.strip():
            return
        if not self._mixer_initialized or self._sfx_channel is None:
            return

        try:
            import os

            import pygame.mixer

            sounds_dir = "assets/sounds"
            if self._audio_manager is not None:
                from seaman_brain.audio.manager import AudioManager
                if isinstance(self._audio_manager, AudioManager):
                    sounds_dir = str(self._audio_manager._sounds_dir)

            sound_path = os.path.join(sounds_dir, f"{sound_name.strip()}.wav")

            if os.path.exists(sound_path):
                sound = pygame.mixer.Sound(sound_path)
                sound.set_volume(self._volumes[AudioChannel.SFX])
                self._sfx_channel.play(sound)
                logger.debug("SFX played: %s", sound_name)
            else:
                logger.debug("SFX file not found: %s (silent)", sound_path)
        except Exception as exc:
            logger.warning("SFX playback failed for %s: %s", sound_name, exc)

    def toggle_microphone(self) -> bool:
        """Toggle microphone input for STT.

        Returns:
            The new microphone active state.
        """
        with self._stt_lock:
            self.mic_active = not self.mic_active

            if self._audio_manager is not None:
                self._audio_manager.stt_enabled = self.mic_active

            if self.mic_active:
                logger.info("Microphone activated")
                self._start_listening()
            else:
                logger.info("Microphone deactivated")

            return self.mic_active

    def _start_listening(self) -> None:
        """Begin STT listening in the background async loop."""
        if self._audio_manager is None or self._async_loop is None:
            return

        asyncio.run_coroutine_threadsafe(
            self._listen_async(), self._async_loop
        )

    async def _listen_async(self) -> None:
        """Continuous async STT listening loop.

        Listens repeatedly while ``mic_active`` is True. Each successful
        transcription is forwarded to the ``on_stt_result`` callback.
        """
        if self._audio_manager is None:
            return

        while self.mic_active:
            try:
                text = await self._audio_manager.listen()
                if text and text.strip() and self._on_stt_result is not None:
                    self._on_stt_result(text.strip())
            except Exception as exc:
                logger.warning("STT listen failed: %s", exc)
                # Brief pause before retrying to avoid tight error loops
                await asyncio.sleep(1.0)

    def handle_key_event(self, key: int) -> bool:
        """Handle keyboard input for audio controls.

        Currently handles: M key to toggle microphone.

        Args:
            key: Pygame key constant.

        Returns:
            True if the key was handled, False otherwise.
        """
        try:
            import pygame
            if key == pygame.K_m:
                self.toggle_microphone()
                return True
        except Exception:
            pass
        return False

    def update(self, dt: float) -> None:
        """Per-frame update for audio state.

        Args:
            dt: Delta time in seconds since last frame.
        """
        # Check if ambient finished unexpectedly and restart
        if self._ambient_playing and self._ambient_channel is not None:
            try:
                if not self._ambient_channel.get_busy():
                    # Channel stopped — may need restart
                    self._ambient_playing = False
            except Exception:
                pass

    def shutdown(self) -> None:
        """Clean shutdown — stop all audio channels."""
        self.stop_ambient()
        self.mic_active = False

        if self._mixer_initialized:
            try:
                if self._voice_channel is not None:
                    self._voice_channel.stop()
                if self._sfx_channel is not None:
                    self._sfx_channel.stop()
            except Exception as exc:
                logger.debug("Error stopping channels during shutdown: %s", exc)

        logger.info("PygameAudioBridge shutdown complete")

    def get_status(self) -> dict[str, Any]:
        """Return current audio bridge status.

        Returns:
            Dict with audio state information.
        """
        return {
            "audio_available": self.audio_available,
            "ambient_playing": self._ambient_playing,
            "ambient_environment": self._current_ambient,
            "mic_active": self.mic_active,
            "volumes": {ch.value: vol for ch, vol in self._volumes.items()},
        }
