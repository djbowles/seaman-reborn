"""Game loop integration bridge for the vision pipeline.

VisionBridge manages periodic and on-demand frame capture, dispatches
async vision model calls, and maintains a bounded buffer of recent
observations. Follows the same pattern as ``PygameAudioBridge``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

from seaman_brain.config import VisionConfig
from seaman_brain.vision.capture import SurfaceCapture, WebcamCapture
from seaman_brain.vision.observer import VisionObserver

logger = logging.getLogger(__name__)

_OBSERVATION_TIMEOUT = 30.0


class VisionBridge:
    """Connects the vision pipeline to the Pygame game loop.

    Handles periodic background captures, on-demand triggers (V key),
    source switching, and observation buffering. Async vision calls
    are dispatched to the background event loop via
    ``asyncio.run_coroutine_threadsafe``.

    Attributes:
        source: Current capture source (``"webcam"``, ``"tank"``, ``"off"``).
    """

    def __init__(
        self,
        vision_config: VisionConfig,
        async_loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._config = vision_config
        self._async_loop = async_loop

        self._webcam = WebcamCapture(device_index=vision_config.webcam_index)
        self._surface = SurfaceCapture()
        self._observer = VisionObserver(config=vision_config)

        self._observations: deque[tuple[str, datetime]] = deque(
            maxlen=vision_config.max_observations
        )
        self._timer: float = 0.0
        self._pending: Any | None = None  # Future from run_coroutine_threadsafe
        self._last_capture_failed: bool = False
        self._pending_start_time: float = 0.0

        self.source: str = vision_config.source

    def update(self, dt: float, surface: Any | None = None) -> None:
        """Per-frame update — accumulate timer and trigger periodic capture.

        Args:
            dt: Delta time in seconds since last frame.
            surface: The current Pygame display surface (for tank capture).
        """
        if self.source == "off":
            return

        # Check if pending observation completed
        self._check_pending()

        self._timer += dt
        if self._timer >= self._config.capture_interval and self._pending is None:
            self._timer = 0.0
            self._do_capture(surface)

    def trigger_observation(self, surface: Any | None = None) -> None:
        """Trigger an on-demand observation (e.g. from V key press).

        Args:
            surface: The current Pygame display surface (for tank capture).
        """
        if self.source == "off":
            return
        self._do_capture(surface)

    def get_recent_observations(self) -> list[str]:
        """Return recent observation texts from the buffer.

        Returns:
            List of observation strings, newest first.
        """
        return [text for text, _ts in reversed(self._observations)]

    def set_webcam_index(self, index: int) -> None:
        """Change the webcam device index at runtime.

        Recreates the internal ``WebcamCapture`` with the new index so
        subsequent captures use the correct camera.

        Args:
            index: OpenCV device index. ``-1`` means use device 0.
        """
        effective = 0 if index < 0 else index
        self._webcam = WebcamCapture(device_index=effective)
        self._config.webcam_index = index
        logger.info("Webcam index changed to: %d (effective %d)", index, effective)

    def set_source(self, source: str) -> None:
        """Switch the capture source at runtime.

        Args:
            source: ``"webcam"``, ``"tank"``, or ``"off"``.
        """
        self.source = source
        self._timer = 0.0
        logger.info("Vision source changed to: %s", source)

    def shutdown(self) -> None:
        """Clean shutdown — cancel any pending future."""
        if self._pending is not None:
            try:
                self._pending.cancel()
            except Exception:
                pass
            self._pending = None
        logger.info("VisionBridge shutdown complete")

    # ── Internal ──────────────────────────────────────────────────────

    def _do_capture(self, surface: Any | None) -> None:
        """Capture a frame and dispatch async observation."""
        frame_bytes: bytes | None = None

        if self.source == "webcam":
            if self._webcam.available:
                frame_bytes = self._webcam.capture()
            else:
                logger.warning("Webcam not available, skipping capture")
                self._last_capture_failed = True
                return
        elif self.source == "tank":
            frame_bytes = self._surface.capture(surface)
        else:
            return

        if frame_bytes is None:
            logger.warning("No frame captured from %s", self.source)
            self._last_capture_failed = True
            return

        if self._async_loop is None:
            logger.warning("No async loop available for vision observation")
            self._last_capture_failed = True
            return

        source = self.source

        async def _observe() -> str:
            return await self._observer.observe(frame_bytes, source=source)

        try:
            self._pending = asyncio.run_coroutine_threadsafe(
                _observe(), self._async_loop
            )
            self._pending_start_time = time.monotonic()
            self._last_capture_failed = False
        except Exception as exc:
            logger.warning("Failed to dispatch vision observation: %s", exc)
            self._last_capture_failed = True

    def _check_pending(self) -> None:
        """Check if the pending observation future is done."""
        if self._pending is None:
            return

        # Timeout check — cancel if running too long
        if not self._pending.done():
            elapsed = time.monotonic() - self._pending_start_time
            if elapsed > _OBSERVATION_TIMEOUT:
                logger.warning("Vision observation timed out after %.0fs", elapsed)
                try:
                    self._pending.cancel()
                except Exception:
                    pass
                self._pending = None
                self._last_capture_failed = True
            return

        try:
            result = self._pending.result(timeout=0)
            if result:
                self._observations.append((result, datetime.now(UTC)))
                logger.debug("Vision observation stored: %s", result[:60])
            else:
                logger.warning("Vision observation returned empty result")
        except Exception as exc:
            logger.warning("Vision observation failed: %s", exc)
        finally:
            self._pending = None
