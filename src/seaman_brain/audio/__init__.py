"""Audio subsystem - TTS voice output, STT speech input, and sound effects."""

from seaman_brain.audio.aec import NLMSEchoCanceller
from seaman_brain.audio.manager import AudioManager, create_audio_manager
from seaman_brain.audio.pipeline import AudioIOPipeline
from seaman_brain.audio.stt import (
    NoopSTTProvider,
    RivaSTTProvider,
    SpeechRecognitionSTTProvider,
    STTProvider,
    create_stt_provider,
)
from seaman_brain.audio.tts import (
    NoopTTSProvider,
    Pyttsx3TTSProvider,
    RivaTTSProvider,
    TTSProvider,
    create_tts_provider,
)

__all__ = [
    "AudioIOPipeline",
    "AudioManager",
    "NLMSEchoCanceller",
    "NoopSTTProvider",
    "NoopTTSProvider",
    "Pyttsx3TTSProvider",
    "RivaSTTProvider",
    "RivaTTSProvider",
    "STTProvider",
    "SpeechRecognitionSTTProvider",
    "TTSProvider",
    "create_audio_manager",
    "create_stt_provider",
    "create_tts_provider",
]
