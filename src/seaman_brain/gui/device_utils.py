"""Device enumeration utilities for audio I/O and webcam selection.

Probes available hardware devices and returns friendly names for use
in settings dropdowns. All functions gracefully handle missing libraries.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _get_preferred_api_idx() -> int | None:
    """Return the host API index for WASAPI / CoreAudio / ALSA, or None."""
    try:
        import sounddevice as sd  # type: ignore[import-untyped]

        preferred = ("Windows WASAPI", "Core Audio", "ALSA")
        for i, api in enumerate(sd.query_hostapis()):
            if api["name"] in preferred:
                return i
    except Exception:
        pass
    return None


_SKIP_NAMES = frozenset({
    "Microsoft Sound Mapper - Output",
    "Microsoft Sound Mapper - Input",
    "Primary Sound Driver",
    "Primary Sound Capture Driver",
})


def list_audio_output_devices() -> list[tuple[int, str]]:
    """Enumerate available audio output devices.

    Filters to a single host API (WASAPI on Windows) to avoid duplicates.

    Returns:
        List of (device_index, name) tuples. First entry is "System Default".
    """
    devices: list[tuple[int, str]] = [(0, "System Default")]
    try:
        import sounddevice as sd  # type: ignore[import-untyped]

        api_idx = _get_preferred_api_idx()
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_output_channels"] <= 0:
                continue
            if dev["name"] in _SKIP_NAMES:
                continue
            if api_idx is not None and dev["hostapi"] != api_idx:
                continue
            devices.append((i + 1, dev["name"]))
    except Exception:
        logger.debug("sounddevice not available for output enumeration")
    return devices


def list_audio_input_devices() -> list[tuple[int, str]]:
    """Enumerate available audio input (microphone) devices.

    Filters to a single host API (WASAPI on Windows) to avoid duplicates.

    Returns:
        List of (device_index, name) tuples. First entry is "System Default".
    """
    devices: list[tuple[int, str]] = [(0, "System Default")]
    try:
        import sounddevice as sd  # type: ignore[import-untyped]

        api_idx = _get_preferred_api_idx()
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] <= 0:
                continue
            if dev["name"] in _SKIP_NAMES:
                continue
            if api_idx is not None and dev["hostapi"] != api_idx:
                continue
            devices.append((i + 1, dev["name"]))
    except Exception:
        logger.debug("sounddevice not available for input enumeration")
    return devices


def _get_dshow_camera_names() -> list[str]:
    """Get DirectShow video input device names on Windows via pygrabber.

    Returns:
        List of device friendly names in DirectShow enumeration order,
        or empty list if pygrabber is not available.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore[import-untyped]

        return FilterGraph().get_input_devices()
    except Exception:
        return []


def list_webcams() -> list[tuple[int, str]]:
    """Probe webcam device indices and return those that open.

    Uses DirectShow device names (via ``pygrabber``) on Windows for
    friendly names. Falls back to "Camera N" when names aren't available.

    Returns:
        List of (index, name) tuples. First entry is "System Default".
    """
    devices: list[tuple[int, str]] = [(-1, "System Default")]

    # Get friendly names from DirectShow (Windows only, best-effort)
    dshow_names = _get_dshow_camera_names()

    try:
        import cv2  # type: ignore[import-untyped]

        for idx in range(5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                name = dshow_names[idx] if idx < len(dshow_names) else f"Camera {idx}"
                devices.append((idx, name))
                cap.release()
    except Exception:
        logger.debug("OpenCV not available for webcam enumeration")

    # If OpenCV isn't available but we have DirectShow names, list those
    if len(devices) == 1 and dshow_names:
        for idx, name in enumerate(dshow_names):
            devices.append((idx, name))

    return devices


def list_tts_providers() -> list[tuple[str, str]]:
    """Enumerate available TTS provider backends.

    Checks whether each provider library is importable and appends
    "(not installed)" when it is not.

    Returns:
        List of (provider_key, display_name) tuples.
    """
    providers: list[tuple[str, str]] = []

    # pyttsx3
    try:
        import pyttsx3  # type: ignore[import-untyped]  # noqa: F401
        providers.append(("pyttsx3", "pyttsx3 (System)"))
    except ImportError:
        providers.append(("pyttsx3", "pyttsx3 (not installed)"))

    # kokoro
    try:
        import kokoro  # type: ignore[import-untyped]  # noqa: F401
        providers.append(("kokoro", "Kokoro (Neural)"))
    except ImportError:
        providers.append(("kokoro", "Kokoro (not installed)"))

    # riva
    try:
        import riva.client  # type: ignore[import-untyped]  # noqa: F401
        providers.append(("riva", "NVIDIA Riva"))
    except ImportError:
        providers.append(("riva", "Riva (not installed)"))

    return providers


def list_stt_providers() -> list[tuple[str, str]]:
    """Enumerate available STT provider backends.

    Checks whether each provider library is importable and appends
    "(not installed)" when it is not.

    Returns:
        List of (provider_key, display_name) tuples.
    """
    providers: list[tuple[str, str]] = []

    # speech_recognition
    try:
        import speech_recognition  # type: ignore[import-untyped]  # noqa: F401
        providers.append(("speech_recognition", "Google Speech (Online)"))
    except ImportError:
        providers.append(("speech_recognition", "Speech Recognition (not installed)"))

    # faster_whisper
    try:
        import faster_whisper  # type: ignore[import-untyped]  # noqa: F401
        providers.append(("faster_whisper", "Faster Whisper (Local)"))
    except ImportError:
        providers.append(("faster_whisper", "Faster Whisper (not installed)"))

    # riva
    try:
        import riva.client  # type: ignore[import-untyped]  # noqa: F401
        providers.append(("riva", "NVIDIA Riva"))
    except ImportError:
        providers.append(("riva", "Riva (not installed)"))

    return providers


def list_kokoro_voices() -> list[tuple[str, str]]:
    """Enumerate available Kokoro TTS voices.

    Kokoro voices follow a naming convention: ``af_*`` for American female,
    ``am_*`` for American male, ``bf_*`` for British female, etc.

    Returns:
        List of (voice_id, friendly_name) tuples with "" as first entry
        for the default voice.
    """
    voices: list[tuple[str, str]] = [("", "Default (af_heart)")]

    # Known Kokoro voice IDs and their friendly names
    kokoro_voices = [
        ("af_heart", "Heart (American Female)"),
        ("af_alloy", "Alloy (American Female)"),
        ("af_aoede", "Aoede (American Female)"),
        ("af_bella", "Bella (American Female)"),
        ("af_jessica", "Jessica (American Female)"),
        ("af_kore", "Kore (American Female)"),
        ("af_nicole", "Nicole (American Female)"),
        ("af_nova", "Nova (American Female)"),
        ("af_river", "River (American Female)"),
        ("af_sarah", "Sarah (American Female)"),
        ("af_sky", "Sky (American Female)"),
        ("am_adam", "Adam (American Male)"),
        ("am_echo", "Echo (American Male)"),
        ("am_eric", "Eric (American Male)"),
        ("am_liam", "Liam (American Male)"),
        ("am_michael", "Michael (American Male)"),
        ("am_onyx", "Onyx (American Male)"),
        ("bf_emma", "Emma (British Female)"),
        ("bf_isabella", "Isabella (British Female)"),
        ("bm_daniel", "Daniel (British Male)"),
        ("bm_fable", "Fable (British Male)"),
        ("bm_george", "George (British Male)"),
        ("bm_lewis", "Lewis (British Male)"),
    ]

    try:
        import kokoro  # type: ignore[import-untyped]  # noqa: F401

        voices.extend(kokoro_voices)
    except ImportError:
        logger.debug("kokoro not available for voice enumeration")
    return voices


_RIVA_FALLBACK_VOICES: list[tuple[str, str]] = [
    # FastPitch HiFi-GAN (works on WSL2)
    ("English-US.Male-1", "Male 1 (FastPitch)"),
    ("English-US.Female-1", "Female 1 (FastPitch)"),
    # Magpie Multilingual (requires native Linux — broken on WSL2)
    ("Magpie-Multilingual.EN-US.Male.Male-1", "Male (Magpie)"),
    ("Magpie-Multilingual.EN-US.Female.Female-1", "Female (Magpie)"),
]


def list_riva_tts_voices(uri: str = "localhost:50051") -> list[tuple[str, str]]:
    """Enumerate available Riva TTS voices via gRPC.

    Queries the Riva server for available voice models and parses the
    ``subvoices`` parameter to list individual speaker/emotion variants.
    Falls back to known voice names if the server is unreachable.

    Args:
        uri: Riva server gRPC URI.

    Returns:
        List of (voice_name, friendly_name) tuples with "" as first entry.
    """
    voices: list[tuple[str, str]] = [("", "Default")]
    try:
        import riva.client  # type: ignore[import-untyped]
        import riva.client.proto.riva_tts_pb2 as tts_pb2  # type: ignore[import-untyped]

        auth = riva.client.Auth(uri=uri)
        service = riva.client.SpeechSynthesisService(auth)
        config = service.stub.GetRivaSynthesisConfig(
            tts_pb2.RivaSynthesisConfigRequest()
        )
        for model in config.model_config:
            params = dict(model.parameters)
            base_voice = params.get("voice_name", model.model_name)
            subvoices_str = params.get("subvoices", "")
            if subvoices_str:
                for entry in subvoices_str.split(","):
                    # Format: "SubvoiceName:speaker_id"
                    name = entry.rsplit(":", 1)[0]
                    full_name = f"{base_voice}.{name}"
                    voices.append((full_name, f"{name} ({base_voice})"))
            else:
                voices.append((base_voice, base_voice))
        if len(voices) > 1:
            return voices
    except ImportError:
        logger.debug("nvidia-riva-client not available for voice enumeration")
    except Exception as exc:
        logger.debug("Riva voice enumeration failed: %s", exc)

    # Fallback: known Riva voices (FastPitch + Magpie)
    voices.extend(_RIVA_FALLBACK_VOICES)
    return voices


def list_tts_voices(provider: str = "pyttsx3") -> list[tuple[str, str]]:
    """Enumerate available TTS voices for the given provider.

    Args:
        provider: TTS provider name (``"pyttsx3"``, ``"kokoro"``, or ``"riva"``).

    Returns:
        List of (voice_id, friendly_name) tuples with "" as first entry
        for the system default voice.
    """
    if provider.lower() == "kokoro":
        return list_kokoro_voices()

    if provider.lower() == "riva":
        return list_riva_tts_voices()

    # Default: pyttsx3
    voices: list[tuple[str, str]] = [("", "System Default")]
    try:
        import pyttsx3  # type: ignore[import-untyped]

        engine = pyttsx3.init()
        for v in engine.getProperty("voices"):
            voices.append((v.id, v.name))
        engine.stop()
    except Exception:
        logger.debug("pyttsx3 not available for voice enumeration")
    return voices
