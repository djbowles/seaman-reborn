"""Auto-start NVIDIA Riva Speech Servers via WSL2 Docker.

Manages two Docker containers:
- **riva-speech** (Quickstart 2.19.0): ASR on gRPC port 50051
- **riva-nim-tts** (NIM Magpie 1.7.0): TTS on gRPC port 50052

Checks if each service is already reachable. If not, and
``riva_auto_start`` is enabled, starts Docker containers in WSL2
and polls until the gRPC endpoints respond.

A background WSL "keepalive" process is maintained to prevent WSL2
from idle-terminating (which would kill Docker and all containers).
"""

from __future__ import annotations

import atexit
import logging
import subprocess
import sys
import time

from seaman_brain.config import AudioConfig

logger = logging.getLogger(__name__)

_RIVA_SEARCH_PATHS = [
    "/opt/riva-quickstart/riva_quickstart_v2.19.0/riva_start.sh",
    "/opt/riva-quickstart/riva_quickstart_v*/riva_start.sh",
    "/tmp/riva_quickstart/riva_quickstart_v*/riva_start.sh",
    "$HOME/riva_quickstart/riva_start.sh",
]
_POLL_INTERVAL = 5.0  # seconds between gRPC readiness checks
_STARTUP_TIMEOUT = 300.0  # max seconds — Riva model loading can take 3-5 min
_NIM_STARTUP_TIMEOUT = 600.0  # NIM first-run builds TRT engines (~10 min)

# Container names
_ASR_CONTAINER = "riva-speech"
_TTS_CONTAINER = "riva-nim-tts"

# Module-level keepalive process — killed on interpreter exit.
_wsl_keepalive: subprocess.Popen | None = None


def _is_riva_reachable(uri: str, timeout: float = 2.0) -> bool:
    """Quick gRPC channel check — returns True if Riva is responding."""
    try:
        import grpc

        channel = grpc.insecure_channel(uri)
        try:
            grpc.channel_ready_future(channel).result(timeout=timeout)
            return True
        except grpc.FutureTimeoutError:
            return False
        finally:
            channel.close()
    except ImportError:
        logger.debug("grpc not installed, cannot check Riva reachability")
        return False
    except Exception as exc:
        logger.debug("Riva reachability check failed: %s", exc)
        return False


def _wsl_available() -> bool:
    """Check if WSL is usable on this machine."""
    if sys.platform != "win32":
        return False
    try:
        result = subprocess.run(
            ["wsl", "--status"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _ensure_docker_running(distro: str) -> bool:
    """Start the Docker daemon inside WSL if it isn't running."""
    try:
        check = subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-c",
             "docker info > /dev/null 2>&1 && echo OK"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "OK" in check.stdout:
            return True

        logger.info("Docker not running in WSL, starting daemon...")
        subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-c",
             "sudo service docker start"],
            capture_output=True,
            timeout=15,
        )
        # Verify it came up
        time.sleep(2.0)
        verify = subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-c",
             "docker info > /dev/null 2>&1 && echo OK"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "OK" in verify.stdout:
            logger.info("Docker daemon started")
            return True
        logger.warning("Docker daemon failed to start in WSL")
        return False
    except Exception as exc:
        logger.warning("Docker check/start failed: %s", exc)
        return False


def _find_riva_start_script(distro: str) -> str | None:
    """Locate riva_start.sh inside WSL2 by searching known paths."""
    search_cmd = " ".join(
        f"ls -1 {p} 2>/dev/null;" for p in _RIVA_SEARCH_PATHS
    )
    try:
        result = subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-c", search_cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in result.stdout.strip().splitlines():
            line = line.strip()
            if line.endswith("riva_start.sh"):
                logger.info("Found Riva start script: %s", line)
                return line
    except Exception as exc:
        logger.debug("Riva script search failed: %s", exc)
    return None


def _start_wsl_keepalive(distro: str) -> None:
    """Start a background WSL session to prevent WSL2 idle shutdown.

    WSL2 terminates the VM when no sessions are active, which kills
    Docker and all containers. This starts a long-running ``sleep``
    in the background and registers an atexit handler to clean it up.
    """
    global _wsl_keepalive
    if _wsl_keepalive is not None and _wsl_keepalive.poll() is None:
        return  # Already running

    try:
        _wsl_keepalive = subprocess.Popen(
            ["wsl", "-d", distro, "--", "bash", "-c",
             "sleep infinity"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        logger.info("WSL keepalive started (pid=%d)", _wsl_keepalive.pid)
    except Exception as exc:
        logger.warning("Failed to start WSL keepalive: %s", exc)


def _kill_wsl_keepalive() -> None:
    """Kill the WSL keepalive process on interpreter exit."""
    global _wsl_keepalive
    if _wsl_keepalive is not None:
        try:
            _wsl_keepalive.terminate()
            _wsl_keepalive.wait(timeout=3)
        except Exception:
            pass
        _wsl_keepalive = None


atexit.register(_kill_wsl_keepalive)


def _start_riva_in_wsl(distro: str) -> subprocess.Popen | None:
    """Launch riva_start.sh inside WSL2.

    Auto-discovers the script path by searching known install locations.
    Sets ``--restart unless-stopped`` on the container so Docker
    auto-restarts it if it crashes.

    Returns the Popen handle, or None on failure.
    """
    script = _find_riva_start_script(distro)
    if script is None:
        logger.warning(
            "riva_start.sh not found in WSL (%s). "
            "Run docker/riva_setup.sh first to install Riva.",
            distro,
        )
        return None

    script_dir = script.rsplit("/", 1)[0]
    cmd = [
        "wsl", "-d", distro, "--", "bash", "-c",
        f"cd {script_dir} && bash riva_start.sh > /tmp/riva_start.log 2>&1"
        f" && docker update --restart unless-stopped {_ASR_CONTAINER}"
        f" > /dev/null 2>&1",
    ]
    logger.info("Starting Riva ASR via WSL2 (%s): %s", distro, script)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        return proc
    except Exception as exc:
        logger.warning("Failed to launch Riva in WSL: %s", exc)
        return None


def _start_nim_tts(distro: str) -> bool:
    """Start the Riva NIM TTS container if not already running.

    Uses ``docker start`` for existing containers, which is fast.
    The container should already exist from initial setup via
    ``docker/riva_setup.sh``.

    Returns True if the container is running.
    """
    try:
        # Check if container exists and is running
        check = subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-c",
             f"docker ps -q -f name={_TTS_CONTAINER}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if check.stdout.strip():
            return True  # Already running

        # Try to start existing container
        result = subprocess.run(
            ["wsl", "-d", distro, "--", "bash", "-c",
             f"docker start {_TTS_CONTAINER}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            logger.info("Started NIM TTS container (%s)", _TTS_CONTAINER)
            return True

        logger.warning(
            "NIM TTS container '%s' not found. "
            "Run docker/riva_setup.sh to create it.",
            _TTS_CONTAINER,
        )
        return False
    except Exception as exc:
        logger.warning("Failed to start NIM TTS: %s", exc)
        return False


def _get_wsl_ip(distro: str) -> str | None:
    """Get the WSL2 VM IP address for cross-OS connectivity."""
    try:
        result = subprocess.run(
            ["wsl", "-d", distro, "--", "hostname", "-I"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split()[0]
    except Exception:
        pass
    return None


def ensure_riva_running(
    config: AudioConfig,
    on_status: callable | None = None,
) -> bool:
    """Ensure Riva servers are running before provider initialization.

    Manages two containers:
    - ASR (Quickstart) on ``config.riva_uri`` (default: port 50051)
    - TTS (NIM Magpie) on ``config.riva_tts_uri`` (default: port 50052)

    Also starts a WSL keepalive process to prevent WSL2 from idle-
    terminating while the application is running.

    Args:
        config: Audio configuration with Riva settings.
        on_status: Optional callback ``(message: str) -> None`` for
            progress updates (shown in splash/loading screen).

    Returns:
        True if at least one Riva service is reachable, False otherwise.
    """
    wants_riva = (
        config.tts_provider.lower() == "riva"
        or config.stt_provider.lower() == "riva"
    )
    if not wants_riva:
        return False

    asr_uri = config.riva_uri
    tts_uri = config.riva_tts_uri or config.riva_uri

    # Check what's already reachable
    asr_ok = _is_riva_reachable(asr_uri)
    tts_ok = _is_riva_reachable(tts_uri) if tts_uri != asr_uri else asr_ok

    if asr_ok and tts_ok:
        logger.info("Riva services already reachable (ASR=%s, TTS=%s)", asr_uri, tts_uri)
        if on_status:
            on_status("Riva connected")
        _start_wsl_keepalive(config.riva_wsl_distro)
        return True

    # Try dynamic WSL2 IP resolution before giving up on current URIs.
    # WSL2 IP changes on every reboot — auto-detect and retry.
    if _wsl_available() and not (asr_ok and tts_ok):
        wsl_ip = _get_wsl_ip(config.riva_wsl_distro)
        if wsl_ip:
            asr_port = asr_uri.rsplit(":", 1)[-1] if ":" in asr_uri else "50051"
            new_asr_uri = f"{wsl_ip}:{asr_port}"
            if new_asr_uri != asr_uri:
                logger.info("WSL2 IP changed — trying %s", new_asr_uri)
                if _is_riva_reachable(new_asr_uri):
                    config.riva_uri = new_asr_uri
                    asr_uri = new_asr_uri
                    asr_ok = True

            if tts_uri != asr_uri or config.riva_tts_uri:
                tts_port = (
                    tts_uri.rsplit(":", 1)[-1] if ":" in tts_uri else "50052"
                )
                new_tts_uri = f"{wsl_ip}:{tts_port}"
                if new_tts_uri != tts_uri:
                    if _is_riva_reachable(new_tts_uri):
                        config.riva_tts_uri = new_tts_uri
                        tts_uri = new_tts_uri
                        tts_ok = True
            elif asr_ok:
                tts_ok = True
                tts_uri = asr_uri

            if asr_ok and tts_ok:
                logger.info(
                    "Riva reachable after IP update (ASR=%s, TTS=%s)",
                    asr_uri, tts_uri,
                )
                if on_status:
                    on_status("Riva connected")
                _start_wsl_keepalive(config.riva_wsl_distro)
                return True

    if not config.riva_auto_start:
        logger.info("Riva not fully reachable and riva_auto_start=False")
        return asr_ok or tts_ok

    # Need to auto-start — check prerequisites
    if not _wsl_available():
        logger.warning("WSL not available — cannot auto-start Riva")
        return False

    # Start WSL keepalive BEFORE launching containers
    _start_wsl_keepalive(config.riva_wsl_distro)

    if on_status:
        on_status("Checking Docker...")
    if not _ensure_docker_running(config.riva_wsl_distro):
        return False

    any_started = False

    # Start Quickstart container if ASR or shared-URI TTS needs it
    needs_quickstart = (
        (config.stt_provider.lower() == "riva" and not asr_ok)
        or (config.tts_provider.lower() == "riva" and not asr_ok and tts_uri == asr_uri)
    )
    if needs_quickstart:
        if on_status:
            on_status("Starting Riva server...")
        proc = _start_riva_in_wsl(config.riva_wsl_distro)
        if proc is not None:
            any_started = True

    # Start NIM TTS container when using a separate TTS endpoint
    if config.tts_provider.lower() == "riva" and not tts_ok and tts_uri != asr_uri:
        if on_status:
            on_status("Starting Riva TTS (NIM)...")
        if _start_nim_tts(config.riva_wsl_distro):
            any_started = True

    if not any_started and not (asr_ok or tts_ok):
        return False

    # Poll for readiness — check both endpoints
    if on_status:
        on_status("Waiting for Riva services...")
    start_time = time.monotonic()
    timeout = _NIM_STARTUP_TIMEOUT if not tts_ok else _STARTUP_TIMEOUT

    while time.monotonic() - start_time < timeout:
        time.sleep(_POLL_INTERVAL)

        elapsed = time.monotonic() - start_time
        if on_status:
            on_status(f"Waiting for Riva... ({elapsed:.0f}s)")

        if not asr_ok:
            asr_ok = _is_riva_reachable(asr_uri, timeout=3.0)
        if tts_uri == asr_uri:
            tts_ok = asr_ok  # Same endpoint — keep in sync
        elif not tts_ok:
            tts_ok = _is_riva_reachable(tts_uri, timeout=3.0)

        # Success if we have what we need
        needs_asr = config.stt_provider.lower() == "riva"
        needs_tts = config.tts_provider.lower() == "riva"
        if (not needs_asr or asr_ok) and (not needs_tts or tts_ok):
            logger.info("Riva services ready after %.0fs", elapsed)
            if on_status:
                on_status("Riva connected")
            return True

        if elapsed > 60 and (elapsed % 30) < _POLL_INTERVAL:
            logger.info(
                "Riva startup poll: %.0fs (ASR=%s, TTS=%s)",
                elapsed, asr_ok, tts_ok,
            )

    logger.warning("Riva did not become fully ready within %.0fs", timeout)
    if on_status:
        on_status("Riva startup timed out")
    # Return True if at least one service is up
    return asr_ok or tts_ok
