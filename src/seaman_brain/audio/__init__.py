"""Audio subsystem - TTS voice output, STT speech input, and sound effects."""

from seaman_brain.audio.manager import AudioManager, create_audio_manager
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
    "AudioManager",
    "NoopSTTProvider",
    "NoopTTSProvider",
    "Pyttsx3TTSProvider",
    "STTProvider",
    "SpeechRecognitionSTTProvider",
    "TTSProvider",
    "create_audio_manager",
    "create_stt_provider",
    "create_tts_provider",
]
