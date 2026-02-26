"""Tests for the GameWindow Pygame window and main game loop (US-035).

Pygame is mocked to avoid requiring a display server in CI.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Mock pygame before importing window module
_pygame_mock = MagicMock()
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.KEYUP = 769
_pygame_mock.MOUSEBUTTONDOWN = 1025
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_RETURN = 13
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None
_pygame_mock.quit.return_value = None

# Display surface mock
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768
_pygame_mock.display.set_mode.return_value = _surface_mock
_pygame_mock.display.set_caption.return_value = None
_pygame_mock.display.flip.return_value = None

# Clock mock
_clock_mock = MagicMock()
_clock_mock.tick.return_value = 33  # ~30fps
_clock_mock.get_fps.return_value = 30.0
_pygame_mock.time.Clock.return_value = _clock_mock

# Font mock
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_mock.render.return_value = _font_surface
_pygame_mock.font.SysFont.return_value = _font_mock

# Draw mock
_pygame_mock.draw.rect.return_value = None


@pytest.fixture(autouse=True)
def _mock_pygame():
    """Patch pygame for all tests in this module."""
    with patch.dict(sys.modules, {"pygame": _pygame_mock}):
        # Force reimport to pick up mock
        if "seaman_brain.gui.window" in sys.modules:
            del sys.modules["seaman_brain.gui.window"]
        if "seaman_brain.gui" in sys.modules:
            del sys.modules["seaman_brain.gui"]
        yield


def _make_event(event_type: int, **kwargs: object) -> MagicMock:
    """Create a mock Pygame event."""
    ev = MagicMock()
    ev.type = event_type
    for k, v in kwargs.items():
        setattr(ev, k, v)
    return ev


# ── Construction and Config ──────────────────────────────────────────


class TestGameWindowConstruction:
    """Tests for GameWindow initialization and configuration."""

    def test_default_config(self):
        """GameWindow uses default GUIConfig when none provided."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        assert win.width == 1024
        assert win.height == 768
        assert win.fps == 30

    def test_custom_gui_config(self):
        """GameWindow respects custom GUIConfig."""
        from seaman_brain.config import GUIConfig
        from seaman_brain.gui.window import GameWindow

        cfg = GUIConfig(window_width=800, window_height=600, fps=60)
        win = GameWindow(gui_config=cfg)
        assert win.width == 800
        assert win.height == 600
        assert win.fps == 60

    def test_custom_full_config(self):
        """GameWindow extracts GUI config from full SeamanConfig."""
        from seaman_brain.config import GUIConfig, SeamanConfig
        from seaman_brain.gui.window import GameWindow

        full = SeamanConfig(gui=GUIConfig(window_width=640, fps=24))
        win = GameWindow(config=full)
        assert win.width == 640
        assert win.fps == 24

    def test_not_running_initially(self):
        """GameWindow starts in non-running state."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        assert win.running is False

    def test_screen_none_before_init(self):
        """Screen is None before initialize()."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        assert win.screen is None

    def test_manager_none_before_init(self):
        """ConversationManager is None before run()."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        assert win.manager is None


# ── Pygame Initialization ────────────────────────────────────────────


class TestPygameInitialization:
    """Tests for Pygame init sequence."""

    def test_initialize_creates_screen(self):
        """initialize() creates the display surface."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        assert win.screen is not None

    def test_initialize_idempotent(self):
        """Calling initialize() twice doesn't reinitialize."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        screen1 = win.screen
        win.initialize()
        assert win.screen is screen1

    def test_initialize_sets_caption(self):
        """initialize() sets the window title."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        _pygame_mock.display.set_caption.assert_called_with("Seaman Reborn")

    def test_initialize_creates_clock(self):
        """initialize() creates a Pygame clock."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        assert win._clock is not None


# ── Event Handling ───────────────────────────────────────────────────


class TestEventHandling:
    """Tests for Pygame event routing."""

    def test_quit_event_stops_loop(self):
        """QUIT event sets running to False."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        quit_ev = _make_event(_pygame_mock.QUIT)
        _pygame_mock.event.get.return_value = [quit_ev]

        win.handle_events()
        assert win.running is False

    def test_escape_key_dispatched_to_handlers(self):
        """Escape key is dispatched to registered KEYDOWN handlers (not hard-quit)."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        handler = MagicMock()
        win.register_event_handler(_pygame_mock.KEYDOWN, handler)

        esc_ev = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_ESCAPE)
        _pygame_mock.event.get.return_value = [esc_ev]

        win.handle_events()
        # ESC is now dispatched to handlers, not consumed by window
        handler.assert_called_once_with(esc_ev)
        assert win.running is True  # Window doesn't quit on ESC directly

    def test_registered_handler_called(self):
        """Registered event handlers are called for matching events."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        handler = MagicMock()
        win.register_event_handler(_pygame_mock.MOUSEBUTTONDOWN, handler)

        click_ev = _make_event(_pygame_mock.MOUSEBUTTONDOWN, pos=(100, 200))
        _pygame_mock.event.get.return_value = [click_ev]

        win.handle_events()
        handler.assert_called_once_with(click_ev)

    def test_unregister_handler(self):
        """Unregistered handlers are no longer called."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        handler = MagicMock()
        win.register_event_handler(_pygame_mock.KEYDOWN, handler)
        win.unregister_event_handler(_pygame_mock.KEYDOWN, handler)

        key_ev = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN)
        _pygame_mock.event.get.return_value = [key_ev]

        win.handle_events()
        handler.assert_not_called()

    def test_handler_error_does_not_crash(self):
        """An error in a handler is logged but doesn't crash the loop."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        bad_handler = MagicMock(side_effect=ValueError("boom"))
        win.register_event_handler(_pygame_mock.MOUSEBUTTONDOWN, bad_handler)

        click_ev = _make_event(_pygame_mock.MOUSEBUTTONDOWN, pos=(50, 50))
        _pygame_mock.event.get.return_value = [click_ev]

        # Should not raise
        win.handle_events()
        assert win.running is True

    def test_multiple_handlers_same_type(self):
        """Multiple handlers for the same event type all get called."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        h1 = MagicMock()
        h2 = MagicMock()
        win.register_event_handler(_pygame_mock.MOUSEBUTTONDOWN, h1)
        win.register_event_handler(_pygame_mock.MOUSEBUTTONDOWN, h2)

        click_ev = _make_event(_pygame_mock.MOUSEBUTTONDOWN, pos=(10, 10))
        _pygame_mock.event.get.return_value = [click_ev]

        win.handle_events()
        h1.assert_called_once()
        h2.assert_called_once()

    def test_empty_event_queue(self):
        """Empty event queue doesn't crash."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.running = True

        _pygame_mock.event.get.return_value = []
        win.handle_events()
        assert win.running is True


# ── Update Loop ──────────────────────────────────────────────────────


class TestUpdateLoop:
    """Tests for the per-frame update cycle."""

    def test_update_increments_frame_count(self):
        """update() increments the internal frame counter."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        assert win._frame_count == 0
        win.update(0.033)
        assert win._frame_count == 1

    def test_update_callbacks_called(self):
        """Registered update callbacks receive delta time."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        cb = MagicMock()
        win.register_update(cb)

        win.update(0.016)
        cb.assert_called_once_with(0.016)

    def test_update_callback_error_isolated(self):
        """A failing update callback doesn't prevent others from running."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        bad_cb = MagicMock(side_effect=RuntimeError("fail"))
        good_cb = MagicMock()
        win.register_update(bad_cb)
        win.register_update(good_cb)

        win.update(0.033)
        bad_cb.assert_called_once()
        good_cb.assert_called_once()


# ── Render Loop ──────────────────────────────────────────────────────


class TestRenderLoop:
    """Tests for the rendering cycle."""

    def test_render_clears_screen(self):
        """render() fills the screen with background color."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.render()
        _surface_mock.fill.assert_called()

    def test_render_flips_display(self):
        """render() calls pygame.display.flip()."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.render()
        _pygame_mock.display.flip.assert_called()

    def test_render_callbacks_called(self):
        """Registered render callbacks receive the surface."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        cb = MagicMock()
        win.register_renderer(cb)

        win.render()
        cb.assert_called_once_with(_surface_mock)

    def test_render_callback_error_isolated(self):
        """A failing render callback doesn't prevent display flip."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        bad_cb = MagicMock(side_effect=RuntimeError("render fail"))
        win.register_renderer(bad_cb)

        win.render()  # Should not raise
        _pygame_mock.display.flip.assert_called()

    def test_render_noop_without_screen(self):
        """render() is a no-op if screen is not initialized."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.render()  # Should not raise


# ── Async Bridge ─────────────────────────────────────────────────────


class TestAsyncBridge:
    """Tests for the background asyncio loop and ConversationManager bridge."""

    def test_start_async_bridge(self):
        """_start_async_bridge() creates a running event loop thread."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win._start_async_bridge()
        try:
            assert win._loop is not None
            assert win._loop.is_running()
            assert win._async_thread is not None
            assert win._async_thread.is_alive()
        finally:
            win._loop.call_soon_threadsafe(win._loop.stop)
            win._async_thread.join(timeout=2.0)

    def test_submit_async_runs_coroutine(self):
        """submit_async() runs a coroutine on the background loop."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win._start_async_bridge()
        try:

            async def _get_value() -> int:
                return 42

            future = win.submit_async(_get_value())
            result = future.result(timeout=2.0)
            assert result == 42
        finally:
            win._loop.call_soon_threadsafe(win._loop.stop)
            win._async_thread.join(timeout=2.0)

    def test_submit_async_without_bridge_raises(self):
        """submit_async() raises RuntimeError when bridge not started."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        with pytest.raises(RuntimeError, match="Async bridge not started"):

            async def _noop() -> None:
                pass

            win.submit_async(_noop())


# ── Shutdown ─────────────────────────────────────────────────────────


class TestShutdown:
    """Tests for clean shutdown sequence."""

    def test_shutdown_quits_pygame(self):
        """shutdown() calls pygame.quit()."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        _pygame_mock.quit.reset_mock()
        win.shutdown()
        _pygame_mock.quit.assert_called_once()

    def test_shutdown_clears_screen(self):
        """shutdown() sets screen to None."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        assert win.screen is not None
        win.shutdown()
        assert win.screen is None

    def test_shutdown_sets_not_running(self):
        """shutdown() sets running to False."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.running = True
        win.shutdown()
        assert win.running is False

    def test_shutdown_saves_manager_state(self):
        """shutdown() calls manager.shutdown() to save creature state."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win._start_async_bridge()
        try:
            mock_manager = MagicMock()
            mock_manager.shutdown = AsyncMock()
            win._manager = mock_manager
            win.shutdown()
            mock_manager.shutdown.assert_called_once()
        finally:
            # Cleanup if needed
            if win._loop is not None and win._loop.is_running():
                win._loop.call_soon_threadsafe(win._loop.stop)
            if win._async_thread is not None:
                win._async_thread.join(timeout=2.0)

    def test_shutdown_stops_async_thread(self):
        """shutdown() stops the background async loop thread."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win._start_async_bridge()
        assert win._async_thread is not None
        assert win._async_thread.is_alive()
        win.shutdown()
        assert win._async_thread is None

    def test_shutdown_idempotent(self):
        """Calling shutdown() twice doesn't raise."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        win.initialize()
        win.shutdown()
        win.shutdown()  # Should not raise


# ── Main Loop Integration ────────────────────────────────────────────


class TestMainLoopIntegration:
    """Tests for the full run() loop (short-circuited)."""

    def test_run_exits_on_quit(self):
        """run() exits cleanly when QUIT event is received."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        call_count = 0

        def _fake_get_events():
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                return [_make_event(_pygame_mock.QUIT)]
            return []

        _pygame_mock.event.get.side_effect = _fake_get_events

        # Patch _start_async_bridge and _init_manager_async to avoid real threads
        with (
            patch.object(win, "_start_async_bridge"),
            patch.object(win, "_init_manager_async"),
            patch.object(win, "shutdown"),
        ):
            win.run()

        assert call_count >= 2

    def test_run_calls_update_and_render(self):
        """run() calls update and render each frame."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        frames = 0

        def _fake_get_events():
            nonlocal frames
            frames += 1
            if frames >= 3:
                return [_make_event(_pygame_mock.QUIT)]
            return []

        _pygame_mock.event.get.side_effect = _fake_get_events

        update_spy = MagicMock()
        render_spy = MagicMock()

        with (
            patch.object(win, "_start_async_bridge"),
            patch.object(win, "_init_manager_async"),
            patch.object(win, "shutdown"),
            patch.object(win, "update", update_spy),
            patch.object(win, "render", render_spy),
        ):
            win.run()

        # update and render called for each frame before quit
        assert update_spy.call_count >= 2
        assert render_spy.call_count >= 2

    def test_run_calls_shutdown_on_exit(self):
        """run() calls shutdown() even when exiting normally."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        _pygame_mock.event.get.return_value = [_make_event(_pygame_mock.QUIT)]

        shutdown_spy = MagicMock()
        with (
            patch.object(win, "_start_async_bridge"),
            patch.object(win, "_init_manager_async"),
            patch.object(win, "shutdown", shutdown_spy),
        ):
            win.run()

        shutdown_spy.assert_called_once()


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_gui_config_from_full_config(self):
        """GUIConfig values propagate correctly from SeamanConfig."""
        from seaman_brain.config import GUIConfig, SeamanConfig
        from seaman_brain.gui.window import GameWindow

        full = SeamanConfig(gui=GUIConfig(
            window_width=1920, window_height=1080, fps=144, show_debug_hud=True
        ))
        win = GameWindow(config=full)
        assert win.config.window_width == 1920
        assert win.config.window_height == 1080
        assert win.config.fps == 144
        assert win.config.show_debug_hud is True

    def test_gui_config_override_takes_precedence(self):
        """gui_config parameter overrides config.gui."""
        from seaman_brain.config import GUIConfig, SeamanConfig
        from seaman_brain.gui.window import GameWindow

        full = SeamanConfig(gui=GUIConfig(window_width=1920))
        override = GUIConfig(window_width=640)
        win = GameWindow(config=full, gui_config=override)
        assert win.width == 640

    def test_unregister_nonexistent_handler_noop(self):
        """Unregistering a handler that was never registered is a no-op."""
        from seaman_brain.gui.window import GameWindow

        win = GameWindow()
        handler = MagicMock()
        win.unregister_event_handler(999, handler)  # Should not raise
