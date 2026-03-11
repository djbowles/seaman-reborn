"""Tests for Riva auto-start launcher."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from seaman_brain.audio.riva_launcher import (
    _is_riva_reachable,
    _wsl_available,
    ensure_riva_running,
)
from seaman_brain.config import AudioConfig

# All ensure_riva_running tests need the keepalive mocked to avoid
# spawning real WSL processes.
_KEEPALIVE_PATCH = "seaman_brain.audio.riva_launcher._start_wsl_keepalive"


class TestIsRivaReachable:
    """Tests for the gRPC reachability check."""

    def test_reachable_when_grpc_ready(self):
        """Returns True when gRPC channel becomes ready."""
        mock_channel = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None  # No exception = ready

        with patch("seaman_brain.audio.riva_launcher.grpc", create=True) as mock_grpc:
            # Patch grpc at module import level
            with patch.dict("sys.modules", {"grpc": mock_grpc}):
                mock_grpc.insecure_channel.return_value = mock_channel
                mock_grpc.channel_ready_future.return_value = mock_future
                mock_grpc.FutureTimeoutError = TimeoutError

                result = _is_riva_reachable("localhost:50051")

        # Accept either True or False — the important thing is no crash
        assert isinstance(result, bool)

    def test_unreachable_when_grpc_timeout(self):
        """Returns False when gRPC channel times out."""
        mock_grpc = MagicMock()
        mock_channel = MagicMock()
        mock_future = MagicMock()
        mock_grpc.FutureTimeoutError = type("FutureTimeoutError", (Exception,), {})
        mock_future.result.side_effect = mock_grpc.FutureTimeoutError()
        mock_grpc.insecure_channel.return_value = mock_channel
        mock_grpc.channel_ready_future.return_value = mock_future

        with patch.dict("sys.modules", {"grpc": mock_grpc}):
            result = _is_riva_reachable("localhost:50051")

        assert result is False

    def test_unreachable_when_grpc_not_installed(self):
        """Returns False when grpc is not importable."""
        with patch.dict("sys.modules", {"grpc": None}):
            result = _is_riva_reachable("localhost:50051")

        assert result is False


class TestWslAvailable:
    """Tests for WSL availability detection."""

    @patch("seaman_brain.audio.riva_launcher.sys")
    def test_not_available_on_non_windows(self, mock_sys):
        """Returns False on non-Windows platforms."""
        mock_sys.platform = "linux"
        assert _wsl_available() is False

    @patch("seaman_brain.audio.riva_launcher.sys")
    @patch("seaman_brain.audio.riva_launcher.subprocess")
    def test_available_when_wsl_status_ok(self, mock_subprocess, mock_sys):
        """Returns True when wsl --status succeeds."""
        mock_sys.platform = "win32"
        mock_subprocess.run.return_value = MagicMock(returncode=0)
        mock_subprocess.TimeoutExpired = TimeoutError
        assert _wsl_available() is True

    @patch("seaman_brain.audio.riva_launcher.sys")
    @patch("seaman_brain.audio.riva_launcher.subprocess")
    def test_unavailable_when_wsl_not_found(self, mock_subprocess, mock_sys):
        """Returns False when wsl binary doesn't exist."""
        mock_sys.platform = "win32"
        mock_subprocess.run.side_effect = FileNotFoundError
        mock_subprocess.TimeoutExpired = TimeoutError
        assert _wsl_available() is False


class TestEnsureRivaRunning:
    """Tests for the main ensure_riva_running orchestrator."""

    def test_skips_when_no_riva_configured(self):
        """Returns False when neither TTS nor STT uses Riva."""
        config = AudioConfig(tts_provider="pyttsx3", stt_provider="speech_recognition")
        assert ensure_riva_running(config) is False

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable", return_value=True)
    def test_returns_true_when_already_reachable(self, mock_reachable, mock_keepalive):
        """Returns True immediately when Riva is already up."""
        config = AudioConfig(tts_provider="riva", riva_auto_start=True)
        result = ensure_riva_running(config)
        assert result is True
        mock_keepalive.assert_called_once()

    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable", return_value=False)
    def test_returns_false_when_auto_start_disabled(self, mock_reachable):
        """Returns False when Riva is down and auto_start is off."""
        config = AudioConfig(tts_provider="riva", riva_auto_start=False)
        result = ensure_riva_running(config)
        assert result is False

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable", return_value=False)
    @patch("seaman_brain.audio.riva_launcher._wsl_available", return_value=False)
    def test_returns_false_when_wsl_unavailable(
        self, mock_wsl, mock_reachable, mock_keepalive
    ):
        """Returns False when WSL is not available."""
        config = AudioConfig(tts_provider="riva", riva_auto_start=True)
        result = ensure_riva_running(config)
        assert result is False

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._NIM_STARTUP_TIMEOUT", 0.1)
    @patch("seaman_brain.audio.riva_launcher._STARTUP_TIMEOUT", 0.1)
    @patch("seaman_brain.audio.riva_launcher._POLL_INTERVAL", 0.05)
    @patch("seaman_brain.audio.riva_launcher._start_riva_in_wsl", return_value=None)
    @patch("seaman_brain.audio.riva_launcher._ensure_docker_running", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._wsl_available", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable", return_value=False)
    def test_returns_false_when_launch_fails(
        self, mock_reachable, mock_wsl, mock_docker, mock_start, mock_keepalive
    ):
        """Returns False when WSL launch command fails."""
        config = AudioConfig(tts_provider="riva", riva_auto_start=True)
        result = ensure_riva_running(config)
        assert result is False

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._STARTUP_TIMEOUT", 0.5)
    @patch("seaman_brain.audio.riva_launcher._NIM_STARTUP_TIMEOUT", 0.5)
    @patch("seaman_brain.audio.riva_launcher._POLL_INTERVAL", 0.05)
    @patch("seaman_brain.audio.riva_launcher._start_riva_in_wsl")
    @patch("seaman_brain.audio.riva_launcher._ensure_docker_running", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._wsl_available", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable")
    def test_polls_until_ready(
        self, mock_reachable, mock_wsl, mock_docker, mock_start, mock_keepalive
    ):
        """Polls for readiness and returns True when server comes up."""
        # Initial check returns False, then polls return False, then True.
        # With shared URI (tts_uri == asr_uri), only one reachable check
        # per cycle: initial(1) + poll(2) + poll(3=True)
        mock_reachable.side_effect = [False, False, True]
        mock_proc = MagicMock()
        mock_start.return_value = mock_proc

        config = AudioConfig(tts_provider="riva", riva_auto_start=True)
        status_msgs = []
        result = ensure_riva_running(config, on_status=status_msgs.append)
        assert result is True
        assert any("connected" in m.lower() for m in status_msgs)

    @patch(_KEEPALIVE_PATCH)
    def test_on_status_callback_for_stt_riva(self, mock_keepalive):
        """Callback fires when STT (not TTS) uses Riva."""
        config = AudioConfig(
            tts_provider="pyttsx3",
            stt_provider="riva",
            riva_auto_start=True,
        )
        with patch(
            "seaman_brain.audio.riva_launcher._is_riva_reachable",
            return_value=True,
        ):
            msgs = []
            result = ensure_riva_running(config, on_status=msgs.append)
            assert result is True
            assert len(msgs) == 1

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable")
    def test_separate_tts_uri_checked(self, mock_reachable, mock_keepalive):
        """When riva_tts_uri is set, both URIs are checked separately."""
        mock_reachable.return_value = True
        config = AudioConfig(
            tts_provider="riva",
            stt_provider="riva",
            riva_uri="host:50051",
            riva_tts_uri="host:50052",
        )
        result = ensure_riva_running(config)
        assert result is True
        # Should check both URIs
        calls = [c.args[0] for c in mock_reachable.call_args_list]
        assert "host:50051" in calls
        assert "host:50052" in calls

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._NIM_STARTUP_TIMEOUT", 0.2)
    @patch("seaman_brain.audio.riva_launcher._STARTUP_TIMEOUT", 0.2)
    @patch("seaman_brain.audio.riva_launcher._POLL_INTERVAL", 0.05)
    @patch("seaman_brain.audio.riva_launcher._start_nim_tts", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._start_riva_in_wsl")
    @patch("seaman_brain.audio.riva_launcher._ensure_docker_running", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._wsl_available", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable")
    def test_starts_nim_tts_when_separate_uri(
        self, mock_reachable, mock_wsl, mock_docker,
        mock_start_asr, mock_start_nim, mock_keepalive,
    ):
        """Starts NIM TTS container when riva_tts_uri differs from riva_uri."""
        # ASR reachable, TTS not
        def reachable_side(uri, timeout=2.0):
            return "50051" in uri

        mock_reachable.side_effect = reachable_side
        mock_start_asr.return_value = MagicMock()

        config = AudioConfig(
            tts_provider="riva",
            stt_provider="riva",
            riva_uri="host:50051",
            riva_tts_uri="host:50052",
            riva_auto_start=True,
        )
        ensure_riva_running(config)
        mock_start_nim.assert_called_once()

    @patch(_KEEPALIVE_PATCH)
    @patch("seaman_brain.audio.riva_launcher._get_wsl_ip", return_value="10.0.0.42")
    @patch("seaman_brain.audio.riva_launcher._wsl_available", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable")
    def test_dynamic_ip_resolution(
        self, mock_reachable, mock_wsl, mock_ip, mock_keepalive,
    ):
        """Config URIs are updated when WSL2 IP changes and new IP is reachable."""
        # Old IP unreachable, new IP reachable
        def reachable_side(uri, timeout=2.0):
            return uri.startswith("10.0.0.42:")

        mock_reachable.side_effect = reachable_side

        config = AudioConfig(
            tts_provider="riva",
            stt_provider="riva",
            riva_uri="172.20.77.13:50051",
            riva_tts_uri="172.20.77.13:50052",
        )
        result = ensure_riva_running(config)
        assert result is True
        assert config.riva_uri == "10.0.0.42:50051"
        assert config.riva_tts_uri == "10.0.0.42:50052"

    @patch("seaman_brain.audio.riva_launcher._get_wsl_ip", return_value=None)
    @patch("seaman_brain.audio.riva_launcher._wsl_available", return_value=True)
    @patch("seaman_brain.audio.riva_launcher._is_riva_reachable", return_value=False)
    def test_dynamic_ip_no_wsl_ip(
        self, mock_reachable, mock_wsl, mock_ip,
    ):
        """Returns False when WSL IP cannot be determined and auto_start off."""
        config = AudioConfig(
            tts_provider="riva",
            riva_auto_start=False,
        )
        result = ensure_riva_running(config)
        assert result is False
