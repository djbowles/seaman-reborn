"""Tests for vision frame capture sources."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from seaman_brain.vision.capture import SurfaceCapture, WebcamCapture

_PIL_AVAILABLE = True
try:
    from PIL import Image  # noqa: F401
except ImportError:
    _PIL_AVAILABLE = False

# ── WebcamCapture ─────────────────────────────────────────────────────


class TestWebcamCaptureAvailability:
    """Tests for WebcamCapture.available property."""

    def test_available_when_cv2_importable(self):
        """available reflects whether cv2 was importable at module load."""
        wc = WebcamCapture()
        # Result depends on test environment — just verify it's a bool
        assert isinstance(wc.available, bool)

    def test_default_device_index(self):
        """Default device index is 0."""
        wc = WebcamCapture()
        assert wc._device_index == 0

    def test_custom_device_index(self):
        """Custom device index is stored."""
        wc = WebcamCapture(device_index=2)
        assert wc._device_index == 2


class TestWebcamCapture:
    """Tests for WebcamCapture.capture() with mocked cv2."""

    def test_capture_returns_bytes_on_success(self):
        """Successful capture returns JPEG bytes."""
        import seaman_brain.vision.capture as cap_mod

        wc = WebcamCapture()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imencode.return_value = (True, b"\xff\xd8\xff\xe0fake-jpeg")

        original = getattr(cap_mod, "cv2", None)
        try:
            cap_mod.cv2 = mock_cv2
            with patch.object(cap_mod, "_CV2_AVAILABLE", True):
                result = wc.capture()
        finally:
            if original is not None:
                cap_mod.cv2 = original
            elif hasattr(cap_mod, "cv2"):
                delattr(cap_mod, "cv2")

        assert result is not None
        assert isinstance(result, bytes)
        mock_cap.release.assert_called_once()

    def test_capture_returns_none_when_cv2_unavailable(self):
        """Returns None when OpenCV is not installed."""
        wc = WebcamCapture()
        with patch("seaman_brain.vision.capture._CV2_AVAILABLE", False):
            assert wc.capture() is None

    def test_capture_returns_none_when_camera_cannot_open(self):
        """Returns None when camera fails to open."""
        import seaman_brain.vision.capture as cap_mod

        wc = WebcamCapture()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        original = getattr(cap_mod, "cv2", None)
        try:
            cap_mod.cv2 = mock_cv2
            with patch.object(cap_mod, "_CV2_AVAILABLE", True):
                result = wc.capture()
        finally:
            if original is not None:
                cap_mod.cv2 = original
            elif hasattr(cap_mod, "cv2"):
                delattr(cap_mod, "cv2")

        assert result is None
        mock_cap.release.assert_called_once()

    def test_capture_returns_none_when_read_fails(self):
        """Returns None when frame read fails."""
        import seaman_brain.vision.capture as cap_mod

        wc = WebcamCapture()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        original = getattr(cap_mod, "cv2", None)
        try:
            cap_mod.cv2 = mock_cv2
            with patch.object(cap_mod, "_CV2_AVAILABLE", True):
                result = wc.capture()
        finally:
            if original is not None:
                cap_mod.cv2 = original
            elif hasattr(cap_mod, "cv2"):
                delattr(cap_mod, "cv2")

        assert result is None
        mock_cap.release.assert_called_once()

    def test_capture_returns_none_when_encode_fails(self):
        """Returns None when JPEG encoding fails."""
        import seaman_brain.vision.capture as cap_mod

        wc = WebcamCapture()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, MagicMock())

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap
        mock_cv2.imencode.return_value = (False, None)

        original = getattr(cap_mod, "cv2", None)
        try:
            cap_mod.cv2 = mock_cv2
            with patch.object(cap_mod, "_CV2_AVAILABLE", True):
                result = wc.capture()
        finally:
            if original is not None:
                cap_mod.cv2 = original
            elif hasattr(cap_mod, "cv2"):
                delattr(cap_mod, "cv2")

        assert result is None

    def test_capture_releases_camera_on_exception(self):
        """Camera is released even when an exception occurs."""
        import seaman_brain.vision.capture as cap_mod

        wc = WebcamCapture()
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = RuntimeError("camera glitch")

        mock_cv2 = MagicMock()
        mock_cv2.VideoCapture.return_value = mock_cap

        original = getattr(cap_mod, "cv2", None)
        try:
            cap_mod.cv2 = mock_cv2
            with patch.object(cap_mod, "_CV2_AVAILABLE", True):
                result = wc.capture()
        finally:
            if original is not None:
                cap_mod.cv2 = original
            elif hasattr(cap_mod, "cv2"):
                delattr(cap_mod, "cv2")

        assert result is None
        mock_cap.release.assert_called_once()


# ── SurfaceCapture ────────────────────────────────────────────────────


class TestSurfaceCapture:
    """Tests for SurfaceCapture."""

    def test_available_always_true(self):
        """SurfaceCapture.available is always True."""
        sc = SurfaceCapture()
        assert sc.available is True

    def test_capture_none_surface(self):
        """Returns None when surface is None."""
        sc = SurfaceCapture()
        assert sc.capture(None) is None

    @pytest.mark.skipif(not _PIL_AVAILABLE, reason="Pillow not installed")
    def test_capture_with_pil(self):
        """Captures surface via PIL when available."""
        sc = SurfaceCapture()

        # Create a real 4x4 red image's raw RGB bytes
        w, h = 4, 4
        pixels = b"\xff\x00\x00" * (w * h)

        mock_surface = MagicMock()
        mock_surface.get_width.return_value = w
        mock_surface.get_height.return_value = h

        mock_pygame = MagicMock()
        mock_pygame.image.tostring.return_value = pixels

        # The capture() method does `import pygame` locally, so we need
        # sys.modules to point at our mock for that to work.
        with patch.dict("sys.modules", {"pygame": mock_pygame}):
            result = sc.capture(mock_surface)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_capture_returns_bytes_or_none(self):
        """Capture returns bytes or None, regardless of PIL availability."""
        sc = SurfaceCapture()
        mock_surface = MagicMock()
        mock_surface.get_width.return_value = 2
        mock_surface.get_height.return_value = 2

        mock_pygame = MagicMock()
        mock_pygame.image.tostring.return_value = b"\x00" * (2 * 2 * 3)

        with patch.dict("sys.modules", {"pygame": mock_pygame}):
            result = sc.capture(mock_surface)

        assert result is None or isinstance(result, bytes)

    def test_capture_handles_pygame_error(self):
        """Returns None when pygame.image.tostring raises."""
        sc = SurfaceCapture()
        mock_surface = MagicMock()

        mock_pygame = MagicMock()
        mock_pygame.image.tostring.side_effect = RuntimeError("no display")

        with (
            patch.dict("sys.modules", {"pygame": mock_pygame}),
            patch("seaman_brain.vision.capture.pygame", mock_pygame, create=True),
        ):
            result = sc.capture(mock_surface)

        assert result is None
