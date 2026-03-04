"""Tests for device enumeration utilities (gui/device_utils.py).

All external libraries (sounddevice, cv2, pyttsx3) are mocked to avoid
hardware dependencies in CI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch


class TestListAudioOutputDevices:
    """Tests for list_audio_output_devices."""

    def test_returns_system_default_when_no_sounddevice(self):
        """Returns only system default when sounddevice is unavailable."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            from seaman_brain.gui.device_utils import list_audio_output_devices

            devices = list_audio_output_devices()
            assert len(devices) >= 1
            assert devices[0] == (0, "System Default")

    def test_returns_devices_from_sounddevice(self):
        """Returns devices when sounddevice is available."""
        mock_sd = MagicMock()
        mock_sd.query_hostapis.return_value = [
            {"name": "Windows WASAPI", "default_output_device": 0, "default_input_device": 2},
        ]
        mock_sd.query_devices.return_value = [
            {"name": "Speakers", "max_output_channels": 2, "max_input_channels": 0, "hostapi": 0},
            {"name": "HDMI", "max_output_channels": 8, "max_input_channels": 0, "hostapi": 0},
            {"name": "Mic", "max_output_channels": 0, "max_input_channels": 1, "hostapi": 0},
        ]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            # Force re-import to pick up mock
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            devices = device_utils.list_audio_output_devices()

        assert devices[0] == (0, "System Default")
        # Should include Speakers and HDMI but not Mic
        names = [name for _, name in devices]
        assert "Speakers" in names
        assert "HDMI" in names
        assert "Mic" not in names

    def test_handles_sounddevice_exception(self):
        """Gracefully handles sounddevice exceptions."""
        mock_sd = MagicMock()
        mock_sd.query_hostapis.side_effect = RuntimeError("No host APIs")
        mock_sd.query_devices.side_effect = RuntimeError("No devices")
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            devices = device_utils.list_audio_output_devices()

        assert len(devices) == 1
        assert devices[0] == (0, "System Default")


class TestListAudioInputDevices:
    """Tests for list_audio_input_devices."""

    def test_returns_system_default_when_no_sounddevice(self):
        """Returns only system default when sounddevice is unavailable."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            from seaman_brain.gui.device_utils import list_audio_input_devices

            devices = list_audio_input_devices()
            assert len(devices) >= 1
            assert devices[0] == (0, "System Default")

    def test_returns_input_devices(self):
        """Returns only input devices from sounddevice."""
        mock_sd = MagicMock()
        mock_sd.query_hostapis.return_value = [
            {"name": "Windows WASAPI", "default_output_device": 0, "default_input_device": 1},
        ]
        mock_sd.query_devices.return_value = [
            {"name": "Speakers", "max_output_channels": 2, "max_input_channels": 0, "hostapi": 0},
            {"name": "Mic", "max_output_channels": 0, "max_input_channels": 1, "hostapi": 0},
        ]
        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            devices = device_utils.list_audio_input_devices()

        names = [name for _, name in devices]
        assert "Mic" in names
        assert "Speakers" not in names


class TestListWebcams:
    """Tests for list_webcams."""

    def test_returns_system_default_when_no_cv2(self):
        """Returns only system default when OpenCV is unavailable."""
        with patch.dict("sys.modules", {"cv2": None}):
            from seaman_brain.gui.device_utils import list_webcams

            devices = list_webcams()
            assert len(devices) >= 1
            assert devices[0] == (-1, "System Default")

    def test_returns_found_cameras_generic_names(self):
        """Returns cameras with generic names when pygrabber is unavailable."""
        mock_cv2 = MagicMock()

        caps = []
        for i in range(5):
            cap = MagicMock()
            cap.isOpened.return_value = i < 2  # Only indices 0 and 1 available
            caps.append(cap)

        mock_cv2.VideoCapture.side_effect = caps

        with (
            patch.dict("sys.modules", {
                "cv2": mock_cv2,
                "pygrabber": None,
                "pygrabber.dshow_graph": None,
            }),
        ):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            devices = device_utils.list_webcams()

        assert devices[0] == (-1, "System Default")
        assert (0, "Camera 0") in devices
        assert (1, "Camera 1") in devices
        assert len(devices) == 3  # default + 2 cameras

    def test_returns_found_cameras_friendly_names(self):
        """Returns cameras with friendly names when pygrabber is available."""
        mock_cv2 = MagicMock()

        caps = []
        for i in range(5):
            cap = MagicMock()
            cap.isOpened.return_value = i < 2
            caps.append(cap)

        mock_cv2.VideoCapture.side_effect = caps

        mock_graph = MagicMock()
        mock_graph_cls = MagicMock(return_value=mock_graph)
        mock_graph.get_input_devices.return_value = ["Logitech C920", "OBS Virtual Cam"]

        mock_dshow = MagicMock()
        mock_dshow.FilterGraph = mock_graph_cls

        with (
            patch.dict("sys.modules", {
                "cv2": mock_cv2,
                "pygrabber": MagicMock(),
                "pygrabber.dshow_graph": mock_dshow,
            }),
        ):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            devices = device_utils.list_webcams()

        assert devices[0] == (-1, "System Default")
        assert (0, "Logitech C920") in devices
        assert (1, "OBS Virtual Cam") in devices


class TestListTTSProviders:
    """Tests for list_tts_providers."""

    def test_always_returns_three_providers(self):
        """Always returns exactly three provider entries."""
        from seaman_brain.gui.device_utils import list_tts_providers

        providers = list_tts_providers()
        assert len(providers) == 3

    def test_expected_keys_present(self):
        """All expected provider keys are present."""
        from seaman_brain.gui.device_utils import list_tts_providers

        providers = list_tts_providers()
        keys = [key for key, _ in providers]
        assert "pyttsx3" in keys
        assert "kokoro" in keys
        assert "riva" in keys

    def test_missing_lib_shows_not_installed(self):
        """Providers with missing libraries show '(not installed)' suffix."""
        with patch.dict("sys.modules", {"kokoro": None}):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            providers = device_utils.list_tts_providers()

        kokoro_entry = [name for key, name in providers if key == "kokoro"]
        assert kokoro_entry
        assert "not installed" in kokoro_entry[0]


class TestListSTTProviders:
    """Tests for list_stt_providers."""

    def test_always_returns_three_providers(self):
        """Always returns exactly three provider entries."""
        from seaman_brain.gui.device_utils import list_stt_providers

        providers = list_stt_providers()
        assert len(providers) == 3

    def test_expected_keys_present(self):
        """All expected provider keys are present."""
        from seaman_brain.gui.device_utils import list_stt_providers

        providers = list_stt_providers()
        keys = [key for key, _ in providers]
        assert "speech_recognition" in keys
        assert "faster_whisper" in keys
        assert "riva" in keys

    def test_missing_lib_shows_not_installed(self):
        """Providers with missing libraries show '(not installed)' suffix."""
        with patch.dict("sys.modules", {"faster_whisper": None}):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            providers = device_utils.list_stt_providers()

        fw_entry = [name for key, name in providers if key == "faster_whisper"]
        assert fw_entry
        assert "not installed" in fw_entry[0]


class TestListTTSVoices:
    """Tests for list_tts_voices."""

    def test_returns_system_default_when_no_pyttsx3(self):
        """Returns only system default when pyttsx3 is unavailable."""
        with patch.dict("sys.modules", {"pyttsx3": None}):
            from seaman_brain.gui.device_utils import list_tts_voices

            voices = list_tts_voices()
            assert len(voices) >= 1
            assert voices[0] == ("", "System Default")

    def test_returns_voices_from_pyttsx3(self):
        """Returns voices when pyttsx3 is available."""
        mock_engine = MagicMock()
        voice1 = MagicMock()
        voice1.id = "voice-1"
        voice1.name = "Microsoft David"
        voice2 = MagicMock()
        voice2.id = "voice-2"
        voice2.name = "Microsoft Zira"
        mock_engine.getProperty.return_value = [voice1, voice2]

        mock_pyttsx3 = MagicMock()
        mock_pyttsx3.init.return_value = mock_engine

        with patch.dict("sys.modules", {"pyttsx3": mock_pyttsx3}):
            import importlib

            from seaman_brain.gui import device_utils

            importlib.reload(device_utils)
            voices = device_utils.list_tts_voices()

        assert voices[0] == ("", "System Default")
        ids = [vid for vid, _ in voices]
        assert "voice-1" in ids
        assert "voice-2" in ids
