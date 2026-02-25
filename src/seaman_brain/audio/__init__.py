"""Audio subsystem - TTS voice output, STT speech input, and sound effects."""

from seaman_brain.audio.stt import (
    NoopSTTProvider,
    SpeechRecognitionSTTProvider,
    STTProvider,
    create_stt_provider,
)
from seaman_brain.audio.tts import (
    NoopTTSProvider,
    Pyttsx3TTSProvider,
    TTSProvider,
    create_tts_provider,
)

__all__ = [
    "NoopSTTProvider",
    "NoopTTSProvider",
    "Pyttsx3TTSProvider",
    "STTProvider",
    "SpeechRecognitionSTTProvider",
    "TTSProvider",
    "create_stt_provider",
    "create_tts_provider",
]
