"""Frame source implementations for the vision pipeline.

WebcamCapture grabs a frame from a USB/built-in camera via OpenCV.
SurfaceCapture grabs the current Pygame display surface.

Both follow the same interface: ``capture(...) -> bytes | None`` and
expose an ``available`` property indicating hardware readiness.
"""

from __future__ import annotations

import io
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Lazy-check for OpenCV
try:
    import cv2

    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False


class WebcamCapture:
    """Captures a single JPEG frame from a webcam via OpenCV.

    Opens and releases the camera per capture to avoid holding the device
    between 30-second intervals.

    Attributes:
        available: Whether OpenCV is importable.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._device_index = device_index

    @property
    def available(self) -> bool:
        """Whether OpenCV is installed and importable."""
        return _CV2_AVAILABLE

    def capture(self) -> bytes | None:
        """Open the camera, grab one frame, encode to JPEG, and release.

        Returns:
            JPEG bytes, or None on failure.
        """
        if not _CV2_AVAILABLE:
            return None

        cap = None
        try:
            cap = cv2.VideoCapture(self._device_index)
            if not cap.isOpened():
                logger.warning("Webcam %d could not be opened", self._device_index)
                return None

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Webcam %d returned no frame", self._device_index)
                return None

            success, buf = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning("Webcam %d JPEG encoding failed", self._device_index)
                return None
            return bytes(buf)
        except Exception as exc:
            logger.warning("Webcam capture failed: %s", exc)
            return None
        finally:
            if cap is not None:
                cap.release()


class SurfaceCapture:
    """Captures the current Pygame display surface as JPEG bytes.

    Uses ``pygame.image.tostring`` to extract raw pixels, then PIL to
    encode to JPEG for the vision model.

    Attributes:
        available: Always True when pygame is loaded.
    """

    @property
    def available(self) -> bool:
        """Whether surface capture is possible (always True at runtime)."""
        return True

    def capture(self, surface: Any) -> bytes | None:
        """Convert a Pygame Surface to JPEG bytes.

        Args:
            surface: A ``pygame.Surface`` instance.

        Returns:
            JPEG bytes, or None on failure.
        """
        if surface is None:
            return None

        try:
            import pygame

            raw = pygame.image.tostring(surface, "RGB")
            w = surface.get_width()
            h = surface.get_height()
        except Exception as exc:
            logger.warning("Surface pixel extraction failed: %s", exc)
            return None

        try:
            from PIL import Image

            img = Image.frombytes("RGB", (w, h), raw)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            return buf.getvalue()
        except ImportError:
            # Pillow not installed — fall back to raw BMP via pygame
            try:
                buf = io.BytesIO()
                pygame.image.save(surface, buf, "bmp")
                return buf.getvalue()
            except Exception as exc2:
                logger.warning("Surface capture fallback failed: %s", exc2)
                return None
        except Exception as exc:
            logger.warning("Surface JPEG encoding failed: %s", exc)
            return None
