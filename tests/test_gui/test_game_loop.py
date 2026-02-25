"""Tests for the GameEngine full game loop integration (US-042).

Pygame is mocked at module level to avoid requiring a display server.
Uses the established pattern: sys.modules["pygame"] = mock + import once,
with autouse fixture to re-install mock between tests.
"""

from __future__ import annotations

import sys
from concurrent.futures import Future
from unittest.mock import MagicMock, patch

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

_pygame_mock = MagicMock()
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.KEYUP = 769
_pygame_mock.MOUSEBUTTONDOWN = 1025
_pygame_mock.MOUSEMOTION = 1024
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_RETURN = 13
_pygame_mock.K_TAB = 9
_pygame_mock.K_h = 104
_pygame_mock.K_m = 109
_pygame_mock.K_KP_ENTER = 271
_pygame_mock.K_BACKSPACE = 8
_pygame_mock.K_LEFT = 276
_pygame_mock.K_RIGHT = 275
_pygame_mock.K_HOME = 278
_pygame_mock.K_END = 279
_pygame_mock.K_DELETE = 127
_pygame_mock.K_PAGEUP = 280
_pygame_mock.K_PAGEDOWN = 281
_pygame_mock.SRCALPHA = 65536
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
_clock_mock.tick.return_value = 33
_clock_mock.get_fps.return_value = 30.0
_pygame_mock.time.Clock.return_value = _clock_mock

# Font mock
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 16
_font_mock.render.return_value = _font_surface
_font_mock.get_linesize.return_value = 18
_font_mock.size.return_value = (100, 16)
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock

# Draw mocks
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.line.return_value = None
_pygame_mock.draw.ellipse.return_value = None
_pygame_mock.draw.lines.return_value = None
_pygame_mock.draw.polygon.return_value = None

# Surface constructor mock
_overlay_surface = MagicMock()
_overlay_surface.get_width.return_value = 1024
_overlay_surface.get_height.return_value = 768
_pygame_mock.Surface.return_value = _overlay_surface
_pygame_mock.Rect = lambda x, y, w, h: MagicMock(x=x, y=y, width=w, height=h)

# Mixer mock
_mixer_mock = MagicMock()
_mixer_mock.get_init.return_value = True
_mixer_mock.get_num_channels.return_value = 8
_channel_mock = MagicMock()
_mixer_mock.Channel.return_value = _channel_mock
_mixer_mock.Sound.return_value = MagicMock()
_pygame_mock.mixer = _mixer_mock

# event.get mock (empty by default)
_pygame_mock.event.get.return_value = []

# Install pygame mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame.mixer"] = _mixer_mock
sys.modules["pygame.font"] = _pygame_mock.font

from seaman_brain.config import SeamanConfig  # noqa: E402
from seaman_brain.gui.game_loop import GameEngine  # noqa: E402
from seaman_brain.needs.death import DeathCause  # noqa: E402
from seaman_brain.types import CreatureStage  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Re-install pygame mock and reset between tests."""
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame.mixer"] = _mixer_mock
    sys.modules["pygame.font"] = _pygame_mock.font

    _pygame_mock.reset_mock(side_effect=True)
    _mixer_mock.reset_mock(side_effect=True)
    _channel_mock.reset_mock(side_effect=True)

    # Restore return values after reset
    _pygame_mock.QUIT = 256
    _pygame_mock.KEYDOWN = 768
    _pygame_mock.KEYUP = 769
    _pygame_mock.MOUSEBUTTONDOWN = 1025
    _pygame_mock.MOUSEMOTION = 1024
    _pygame_mock.K_ESCAPE = 27
    _pygame_mock.K_RETURN = 13
    _pygame_mock.K_TAB = 9
    _pygame_mock.K_h = 104
    _pygame_mock.K_m = 109
    _pygame_mock.K_KP_ENTER = 271
    _pygame_mock.K_BACKSPACE = 8
    _pygame_mock.K_LEFT = 276
    _pygame_mock.K_RIGHT = 275
    _pygame_mock.K_HOME = 278
    _pygame_mock.K_END = 279
    _pygame_mock.K_DELETE = 127
    _pygame_mock.K_PAGEUP = 280
    _pygame_mock.K_PAGEDOWN = 281
    _pygame_mock.SRCALPHA = 65536
    _pygame_mock.init.return_value = (6, 0)
    _pygame_mock.font.init.return_value = None
    _pygame_mock.quit.return_value = None
    _pygame_mock.display.set_mode.return_value = _surface_mock
    _pygame_mock.display.set_caption.return_value = None
    _pygame_mock.display.flip.return_value = None
    _pygame_mock.time.Clock.return_value = _clock_mock
    _clock_mock.tick.return_value = 33
    _clock_mock.get_fps.return_value = 30.0
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.render.return_value = _font_surface
    _font_mock.get_linesize.return_value = 18
    _font_mock.size.return_value = (100, 16)
    _font_surface.get_width.return_value = 100
    _font_surface.get_height.return_value = 16
    _pygame_mock.Surface.return_value = _overlay_surface
    _overlay_surface.get_width.return_value = 1024
    _overlay_surface.get_height.return_value = 768
    _surface_mock.get_width.return_value = 1024
    _surface_mock.get_height.return_value = 768
    _pygame_mock.Rect = lambda x, y, w, h: MagicMock(x=x, y=y, width=w, height=h)
    _pygame_mock.draw.rect.return_value = None
    _pygame_mock.draw.circle.return_value = None
    _pygame_mock.draw.line.return_value = None
    _pygame_mock.draw.ellipse.return_value = None
    _pygame_mock.draw.lines.return_value = None
    _pygame_mock.draw.polygon.return_value = None
    _pygame_mock.event.get.return_value = []
    _mixer_mock.get_init.return_value = True
    _mixer_mock.get_num_channels.return_value = 8
    _mixer_mock.Channel.return_value = _channel_mock
    _mixer_mock.Sound.return_value = MagicMock()


@pytest.fixture()
def config() -> SeamanConfig:
    """Default config for tests."""
    return SeamanConfig()


@pytest.fixture()
def engine(config: SeamanConfig) -> GameEngine:
    """Create and initialize a GameEngine without starting the loop."""
    eng = GameEngine(config=config)
    eng.initialize()
    return eng


# ── Happy Path Tests ─────────────────────────────────────────────────


class TestGameEngineInit:
    """Test GameEngine initialization."""

    def test_creates_with_default_config(self):
        """GameEngine creates successfully with default config."""
        engine = GameEngine()
        assert engine.window is not None
        assert engine.creature_state is not None
        assert engine.tank is not None
        assert engine.game_over is False

    def test_creates_with_custom_config(self, config: SeamanConfig):
        """GameEngine uses provided config."""
        engine = GameEngine(config=config)
        assert engine._config is config

    def test_initialize_sets_up_subsystems(self, engine: GameEngine):
        """After initialize, all subsystems are wired up."""
        assert engine._tank_renderer is not None
        assert engine._creature_renderer is not None
        assert engine._chat_panel is not None
        assert engine._hud is not None
        assert engine._interaction_manager is not None
        assert engine._audio_bridge is not None

    def test_initial_creature_state(self, engine: GameEngine):
        """Creature starts as Mushroomer stage."""
        assert engine.creature_state.stage == CreatureStage.MUSHROOMER
        assert engine.creature_state.mood == "neutral"


class TestGameLoopTick:
    """Test the per-frame update cycle."""

    def test_update_advances_subsystems(self, engine: GameEngine):
        """A single update tick advances animation and HUD timers."""
        initial_session = engine._hud.session_time
        engine._update(0.1)
        assert engine._hud.session_time > initial_session

    def test_update_accumulates_needs_timer(self, engine: GameEngine):
        """Needs timer accumulates across frames."""
        engine._update(0.3)
        assert engine._needs_timer == pytest.approx(0.3, abs=0.01)
        engine._update(0.3)
        assert engine._needs_timer == pytest.approx(0.6, abs=0.01)

    def test_needs_update_triggers_at_interval(self, engine: GameEngine):
        """Needs update fires when timer exceeds interval."""
        engine._update(1.1)  # > 1.0s interval
        # Timer should reset after firing
        assert engine._needs_timer < 0.5

    def test_update_no_op_when_game_over(self, engine: GameEngine):
        """Update does nothing when game is over."""
        engine.game_over = True
        initial_session = engine._hud.session_time
        engine._update(1.0)
        assert engine._hud.session_time == initial_session

    def test_tank_degrades_over_time(self, engine: GameEngine):
        """Tank cleanliness and oxygen degrade during updates."""
        initial_clean = engine.tank.cleanliness
        # Run several ticks
        for _ in range(10):
            engine._update(0.1)
        assert engine.tank.cleanliness <= initial_clean

    def test_notifications_expire(self, engine: GameEngine):
        """Notifications disappear after their duration."""
        engine._add_notification("Test notification")
        assert len(engine._notifications) == 1
        # Simulate enough time for expiry
        engine._update(5.0)
        assert len(engine._notifications) == 0


class TestAsyncConversationBridge:
    """Test async conversation bridge to ConversationManager."""

    def test_chat_submit_without_manager(self, engine: GameEngine):
        """Chat submit without manager shows fallback message."""
        engine.window._manager = None
        engine._on_chat_submit("Hello creature")
        assert engine._chat_panel.message_count >= 1

    def test_chat_submit_with_manager(self, engine: GameEngine):
        """Chat submit with manager starts async processing."""
        mock_manager = MagicMock()
        engine.window._manager = mock_manager
        engine.window._loop = MagicMock()

        # Mock run_coroutine_threadsafe to return a Future
        future = Future()
        with patch("seaman_brain.gui.game_loop.asyncio.run_coroutine_threadsafe",
                    return_value=future):
            engine._on_chat_submit("Hello creature")

        assert engine._pending_response is future
        assert engine._chat_panel.is_streaming

    def test_pending_response_delivered(self, engine: GameEngine):
        """Completed async response is added to chat."""
        future = Future()
        future.set_result("I am the creature")
        engine._pending_response = future

        engine._check_pending_response()

        assert engine._pending_response is None
        # Should have added the response message
        assert engine._chat_panel.message_count >= 1

    def test_pending_response_error_handled(self, engine: GameEngine):
        """Failed async response shows error in chat."""
        future = Future()
        future.set_exception(RuntimeError("LLM exploded"))
        engine._pending_response = future

        engine._check_pending_response()

        assert engine._pending_response is None
        assert engine._chat_panel.message_count >= 1

    def test_empty_submit_ignored(self, engine: GameEngine):
        """Empty string submission is ignored."""
        initial_count = engine._chat_panel.message_count
        engine._on_chat_submit("")
        assert engine._chat_panel.message_count == initial_count

    def test_submit_during_game_over_ignored(self, engine: GameEngine):
        """Chat submit during game over is ignored."""
        engine.game_over = True
        initial_count = engine._chat_panel.message_count
        engine._on_chat_submit("Hello")
        assert engine._chat_panel.message_count == initial_count


class TestEvolutionSequence:
    """Test evolution transitions with visual celebration."""

    def test_start_evolution_sets_flags(self, engine: GameEngine):
        """Starting evolution activates the celebration state."""
        engine._start_evolution(CreatureStage.GILLMAN)
        assert engine._evolution_active is True
        assert engine._evolution_timer == 0.0
        assert engine.creature_state.stage == CreatureStage.GILLMAN

    def test_evolution_celebration_runs(self, engine: GameEngine):
        """Evolution celebration runs for the configured duration."""
        engine._start_evolution(CreatureStage.GILLMAN)
        assert engine._evolution_active is True

        # Simulate frames during celebration
        engine._update_evolution_celebration(1.0)
        assert engine._evolution_active is True  # Still active

        engine._update_evolution_celebration(2.1)  # Past 3.0s total
        assert engine._evolution_active is False

    def test_evolution_adds_notification(self, engine: GameEngine):
        """Evolution adds a notification message."""
        engine._start_evolution(CreatureStage.GILLMAN)
        assert len(engine._notifications) > 0
        assert "Evolution" in engine._notifications[0][0]

    def test_evolution_updates_renderer_stage(self, engine: GameEngine):
        """After celebration, renderer stage matches new stage."""
        engine._start_evolution(CreatureStage.GILLMAN)
        engine._update_evolution_celebration(3.1)
        assert engine._creature_renderer.stage == CreatureStage.GILLMAN


class TestDeathAndRestart:
    """Test death handling and game restart."""

    def test_death_sets_game_over(self, engine: GameEngine):
        """Death triggers game over state."""
        engine._handle_death(DeathCause.STARVATION)
        assert engine.game_over is True
        assert engine._death_cause == DeathCause.STARVATION
        assert engine._death_message != ""

    def test_restart_resets_state(self, engine: GameEngine):
        """Restart after death resets all game state."""
        engine._handle_death(DeathCause.STARVATION)
        assert engine.game_over is True

        engine._restart_game()
        assert engine.game_over is False
        assert engine._death_cause is None
        assert engine.creature_state.stage == CreatureStage.MUSHROOMER
        assert engine._chat_panel.message_count >= 1  # "new egg" message

    def test_click_during_game_over_restarts(self, engine: GameEngine):
        """Clicking during game over triggers restart."""
        engine._handle_death(DeathCause.STARVATION)
        assert engine.game_over is True

        event = MagicMock()
        event.pos = (500, 400)
        engine._on_mouse_click(event)
        assert engine.game_over is False


class TestRendering:
    """Test rendering methods don't crash."""

    def test_render_normal_state(self, engine: GameEngine):
        """Rendering in normal state runs without error."""
        engine._render(_surface_mock)

    def test_render_game_over(self, engine: GameEngine):
        """Rendering game over screen runs without error."""
        engine._handle_death(DeathCause.STARVATION)
        engine._render(_surface_mock)

    def test_render_with_notifications(self, engine: GameEngine):
        """Rendering with active notifications works."""
        engine._add_notification("Test 1")
        engine._add_notification("Test 2")
        engine._render(_surface_mock)

    def test_render_evolution_overlay(self, engine: GameEngine):
        """Rendering evolution celebration works."""
        engine._evolution_active = True
        engine._evolution_timer = 1.0
        engine._render(_surface_mock)


class TestInteractionHandling:
    """Test mouse and keyboard interaction routing."""

    def test_mouse_move_updates_creature_eyes(self, engine: GameEngine):
        """Mouse motion updates creature eye tracking."""
        event = MagicMock()
        event.pos = (300, 400)
        engine._on_mouse_move(event)
        assert engine._creature_renderer._mouse_x == 300.0
        assert engine._creature_renderer._mouse_y == 400.0

    def test_key_h_toggles_hud_mode(self, engine: GameEngine):
        """Pressing H toggles HUD between compact and expanded."""
        initial = engine._hud.compact
        event = MagicMock()
        event.key = 104  # K_h
        # Need to make chat panel not consume the event
        engine._chat_panel.visible = False
        engine._on_key_down(event)
        assert engine._hud.compact != initial

    def test_shutdown_cleans_up(self, engine: GameEngine):
        """Shutdown doesn't raise exceptions."""
        engine.shutdown()
