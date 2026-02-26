"""Tests for VisionBridge — game loop integration."""

from __future__ import annotations

from concurrent.futures import Future
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from seaman_brain.config import VisionConfig
from seaman_brain.vision.bridge import VisionBridge


@pytest.fixture
def config():
    """VisionConfig with short interval for tests."""
    return VisionConfig(
        enabled=True,
        source="webcam",
        capture_interval=5.0,
        max_observations=3,
    )


@pytest.fixture
def bridge(config):
    """VisionBridge with mocked internals (no real async loop)."""
    b = VisionBridge(vision_config=config, async_loop=None)
    # Mock the webcam/surface/observer to avoid real hardware
    b._webcam = MagicMock()
    b._webcam.available = True
    b._webcam.capture.return_value = b"fake-jpeg"
    b._surface = MagicMock()
    b._surface.capture.return_value = b"fake-surface-jpeg"
    b._observer = MagicMock()
    return b


# ── Construction Tests ────────────────────────────────────────────────


class TestBridgeConstruction:
    """Tests for VisionBridge initialization."""

    def test_default_source(self, config):
        """Bridge source matches config."""
        b = VisionBridge(vision_config=config)
        assert b.source == "webcam"

    def test_initial_timer_zero(self, config):
        """Timer starts at zero."""
        b = VisionBridge(vision_config=config)
        assert b._timer == 0.0

    def test_initial_observations_empty(self, config):
        """Observation buffer starts empty."""
        b = VisionBridge(vision_config=config)
        assert b.get_recent_observations() == []

    def test_initial_no_pending(self, config):
        """No pending observation at start."""
        b = VisionBridge(vision_config=config)
        assert b._pending is None


# ── Timer and Periodic Capture Tests ──────────────────────────────────


class TestPeriodicCapture:
    """Tests for periodic timer-driven capture."""

    def test_timer_accumulates(self, bridge):
        """Timer accumulates dt across frames."""
        bridge.update(1.0)
        assert bridge._timer == pytest.approx(1.0, abs=0.01)
        bridge.update(2.0)
        assert bridge._timer == pytest.approx(3.0, abs=0.01)

    def test_timer_resets_at_interval(self, bridge):
        """Timer resets when capture interval is reached."""
        mock_loop = MagicMock()
        bridge._async_loop = mock_loop

        future = Future()
        future.set_result("Observation text")

        with patch(
            "seaman_brain.vision.bridge.asyncio.run_coroutine_threadsafe",
            return_value=future,
        ):
            bridge.update(6.0)  # > 5.0 interval

        assert bridge._timer == pytest.approx(0.0, abs=0.01)

    def test_no_capture_when_off(self, bridge):
        """No capture when source is 'off'."""
        bridge.set_source("off")
        bridge.update(100.0)
        assert bridge._timer == 0.0  # Timer not updated
        bridge._webcam.capture.assert_not_called()

    def test_no_double_capture_while_pending(self, bridge):
        """No new capture while a previous observation is pending."""
        bridge._pending = MagicMock()  # Simulate pending
        bridge._pending.done.return_value = False
        bridge.update(6.0)  # Timer exceeds interval
        # Should not trigger a new capture because pending is set
        bridge._webcam.capture.assert_not_called()


# ── On-Demand Capture Tests ───────────────────────────────────────────


class TestOnDemandCapture:
    """Tests for trigger_observation()."""

    def test_trigger_webcam(self, bridge):
        """Trigger captures from webcam and dispatches async call."""
        mock_loop = MagicMock()
        bridge._async_loop = mock_loop

        future = Future()
        future.set_result("Human looks bored")

        with patch(
            "seaman_brain.vision.bridge.asyncio.run_coroutine_threadsafe",
            return_value=future,
        ):
            bridge.trigger_observation()

        bridge._webcam.capture.assert_called_once()

    def test_trigger_tank(self, bridge):
        """Trigger captures from tank surface."""
        bridge.set_source("tank")
        mock_loop = MagicMock()
        bridge._async_loop = mock_loop
        mock_surface = MagicMock()

        future = Future()
        future.set_result("Tank is clean")

        with patch(
            "seaman_brain.vision.bridge.asyncio.run_coroutine_threadsafe",
            return_value=future,
        ):
            bridge.trigger_observation(surface=mock_surface)

        bridge._surface.capture.assert_called_once_with(mock_surface)

    def test_trigger_when_off_is_noop(self, bridge):
        """Trigger when source is 'off' does nothing."""
        bridge.set_source("off")
        bridge.trigger_observation()
        bridge._webcam.capture.assert_not_called()

    def test_trigger_no_async_loop(self, bridge):
        """Trigger without async loop logs but doesn't crash."""
        bridge._async_loop = None
        bridge.trigger_observation()
        # Should not raise

    def test_trigger_no_frame_captured(self, bridge):
        """Trigger when capture returns None is handled."""
        bridge._webcam.capture.return_value = None
        bridge._async_loop = MagicMock()
        bridge.trigger_observation()
        # No dispatch should happen


# ── Observation Buffer Tests ──────────────────────────────────────────


class TestObservationBuffer:
    """Tests for observation storage and retrieval."""

    def test_get_recent_empty(self, bridge):
        """Empty buffer returns empty list."""
        assert bridge.get_recent_observations() == []

    def test_observations_stored_on_completion(self, bridge):
        """Completed observation is stored in buffer."""
        future = Future()
        future.set_result("The human is smiling")
        bridge._pending = future

        bridge._check_pending()

        obs = bridge.get_recent_observations()
        assert len(obs) == 1
        assert obs[0] == "The human is smiling"

    def test_buffer_bounded(self, bridge):
        """Buffer respects max_observations limit."""
        for i in range(5):
            bridge._observations.append((f"Observation {i}", datetime.now(UTC)))

        # max_observations=3, so only last 3 should remain
        obs = bridge.get_recent_observations()
        assert len(obs) == 3
        assert obs[0] == "Observation 4"  # Most recent first

    def test_empty_observation_not_stored(self, bridge):
        """Empty string result is not stored."""
        future = Future()
        future.set_result("")
        bridge._pending = future

        bridge._check_pending()

        assert bridge.get_recent_observations() == []

    def test_failed_observation_not_stored(self, bridge):
        """Failed observation (exception) is not stored."""
        future = Future()
        future.set_exception(RuntimeError("LLM down"))
        bridge._pending = future

        bridge._check_pending()

        assert bridge.get_recent_observations() == []
        assert bridge._pending is None


# ── Source Switching Tests ────────────────────────────────────────────


class TestSourceSwitching:
    """Tests for set_source()."""

    def test_set_source_webcam(self, bridge):
        """Can switch to webcam source."""
        bridge.set_source("webcam")
        assert bridge.source == "webcam"

    def test_set_source_tank(self, bridge):
        """Can switch to tank source."""
        bridge.set_source("tank")
        assert bridge.source == "tank"

    def test_set_source_off(self, bridge):
        """Can switch to off."""
        bridge.set_source("off")
        assert bridge.source == "off"

    def test_set_source_resets_timer(self, bridge):
        """Switching source resets the capture timer."""
        bridge._timer = 15.0
        bridge.set_source("tank")
        assert bridge._timer == 0.0


# ── Shutdown Tests ────────────────────────────────────────────────────


class TestShutdown:
    """Tests for VisionBridge.shutdown()."""

    def test_shutdown_cancels_pending(self, bridge):
        """Shutdown cancels any pending future."""
        mock_future = MagicMock()
        bridge._pending = mock_future

        bridge.shutdown()

        mock_future.cancel.assert_called_once()
        assert bridge._pending is None

    def test_shutdown_no_pending(self, bridge):
        """Shutdown with no pending future is safe."""
        bridge._pending = None
        bridge.shutdown()  # Should not raise


# ── Webcam Unavailable Fallback Tests ─────────────────────────────────


class TestGracefulDegradation:
    """Tests for graceful degradation when webcam is unavailable."""

    def test_webcam_unavailable_skips_capture(self, bridge):
        """When webcam not available, capture is skipped."""
        bridge._webcam.available = False
        bridge._async_loop = MagicMock()
        bridge.trigger_observation()
        bridge._webcam.capture.assert_not_called()
