"""Audio subsystem - TTS voice output, STT speech input, and sound effects."""

from seaman_brain.audio.tts import (
    NoopTTSProvider,
    Pyttsx3TTSProvider,
    TTSProvider,
    create_tts_provider,
)

__all__ = [
    "NoopTTSProvider",
    "Pyttsx3TTSProvider",
    "TTSProvider",
    "create_tts_provider",
]
