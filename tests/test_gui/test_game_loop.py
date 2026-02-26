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
_pygame_mock.K_F2 = 283
_pygame_mock.K_h = 104
_pygame_mock.K_m = 109
_pygame_mock.K_v = 118
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
from seaman_brain.gui.game_loop import GameEngine, GameState  # noqa: E402
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
    _pygame_mock.K_F2 = 283
    _pygame_mock.K_h = 104
    _pygame_mock.K_m = 109
    _pygame_mock.K_v = 118
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

    def test_escape_quits_during_gameplay(self, engine: GameEngine):
        """ESC during gameplay sets window.running to False."""
        engine.window.running = True
        event = MagicMock()
        event.key = 27  # K_ESCAPE
        engine._on_key_down(event)
        assert engine.window.running is False

    def test_escape_closes_settings(self, engine: GameEngine):
        """ESC while settings open closes settings instead of quitting."""
        engine.window.running = True
        engine._game_state = GameState.SETTINGS
        if engine._settings_panel is not None:
            engine._settings_panel.visible = True

        event = MagicMock()
        event.key = 27  # K_ESCAPE
        engine._on_key_down(event)

        assert engine._game_state == GameState.PLAYING
        assert engine.window.running is True

    def test_settings_x_button_returns_to_playing(self, engine: GameEngine):
        """Clicking settings X button returns game state to PLAYING."""
        engine._game_state = GameState.SETTINGS
        if engine._settings_panel is not None:
            engine._settings_panel.visible = True

        # Simulate the X button close callback
        engine._on_settings_close()

        assert engine._game_state == GameState.PLAYING

    def test_f1_opens_settings(self, engine: GameEngine):
        """F1 toggles settings overlay open."""
        assert engine._game_state == GameState.PLAYING
        _pygame_mock.K_F1 = 282
        import seaman_brain.gui.game_loop as gl_mod
        gl_mod.pygame = _pygame_mock
        event = MagicMock()
        event.key = 282
        engine._on_key_down(event)
        assert engine._game_state == GameState.SETTINGS

    def test_f1_closes_settings(self, engine: GameEngine):
        """F1 toggles settings overlay closed."""
        _pygame_mock.K_F1 = 282
        import seaman_brain.gui.game_loop as gl_mod
        gl_mod.pygame = _pygame_mock
        engine._game_state = GameState.SETTINGS
        if engine._settings_panel is not None:
            engine._settings_panel.visible = True

        event = MagicMock()
        event.key = 282
        engine._on_key_down(event)
        assert engine._game_state == GameState.PLAYING

    def test_update_skipped_during_settings(self, engine: GameEngine):
        """Gameplay updates are skipped when settings overlay is open."""
        engine._game_state = GameState.SETTINGS
        initial_needs_timer = engine._needs_timer
        engine._update(1.0)
        assert engine._needs_timer == initial_needs_timer

    def test_settings_panel_created_on_init(self, engine: GameEngine):
        """Settings panel is created during initialize."""
        assert engine._settings_panel is not None

    def test_render_with_settings_open(self, engine: GameEngine):
        """Rendering with settings open doesn't crash."""
        engine._game_state = GameState.SETTINGS
        if engine._settings_panel is not None:
            engine._settings_panel.visible = True
        engine._render(_surface_mock)

    def test_mouse_click_routed_to_settings(self, engine: GameEngine):
        """Mouse clicks route to settings panel when it's open."""
        engine._game_state = GameState.SETTINGS
        if engine._settings_panel is not None:
            engine._settings_panel.visible = True
        event = MagicMock()
        event.pos = (500, 400)
        # Should not crash or restart game
        engine._on_mouse_click(event)
        assert engine.game_over is False

    def test_mouse_move_routed_to_settings(self, engine: GameEngine):
        """Mouse moves route to settings panel when it's open."""
        engine._game_state = GameState.SETTINGS
        event = MagicMock()
        event.pos = (500, 400)
        # Should not update creature eye tracking
        initial_x = engine._creature_renderer._mouse_x
        engine._on_mouse_move(event)
        assert engine._creature_renderer._mouse_x == initial_x

    def test_keys_consumed_during_settings(self, engine: GameEngine):
        """Non-ESC/F1 keys are consumed when settings are open."""
        engine._game_state = GameState.SETTINGS
        event = MagicMock()
        event.key = 104  # K_h
        initial_compact = engine._hud.compact
        engine._on_key_down(event)
        assert engine._hud.compact == initial_compact  # Not toggled


# ── Vision Bridge Integration Tests ──────────────────────────────────


class TestVisionBridgeIntegration:
    """Tests for vision bridge integration in GameEngine."""

    def test_no_vision_bridge_by_default(self, engine: GameEngine):
        """Vision bridge is None when config.vision.enabled is False."""
        assert engine._vision_bridge is None

    def test_vision_bridge_created_when_enabled(self, config: SeamanConfig):
        """Vision bridge is created when vision.enabled is True."""
        config.vision.enabled = True
        eng = GameEngine(config=config)
        eng.initialize()
        assert eng._vision_bridge is not None

    def test_v_key_without_bridge_is_noop(self, engine: GameEngine):
        """Pressing V without vision bridge doesn't crash."""
        import seaman_brain.gui.game_loop as gl_mod
        gl_mod.pygame = _pygame_mock
        engine._chat_panel.visible = False
        event = MagicMock()
        event.key = 118  # K_v
        initial_notifs = len(engine._notifications)
        engine._on_key_down(event)
        # No notification added since no vision bridge
        assert len(engine._notifications) == initial_notifs

    def test_v_key_with_bridge_triggers_observation(self, config: SeamanConfig):
        """Pressing V with vision bridge triggers observation and notification."""
        config.vision.enabled = True
        eng = GameEngine(config=config)
        eng.initialize()

        import seaman_brain.gui.game_loop as gl_mod
        gl_mod.pygame = _pygame_mock

        # Mock the vision bridge
        eng._vision_bridge = MagicMock()
        eng._chat_panel.visible = False

        event = MagicMock()
        event.key = 118  # K_v
        eng._on_key_down(event)

        eng._vision_bridge.trigger_observation.assert_called_once()
        assert any("Looking" in n[0] for n in eng._notifications)

    def test_shutdown_with_vision_bridge(self, config: SeamanConfig):
        """Shutdown cleans up vision bridge."""
        config.vision.enabled = True
        eng = GameEngine(config=config)
        eng.initialize()
        eng._vision_bridge = MagicMock()
        eng.shutdown()
        eng._vision_bridge.shutdown.assert_called_once()

    def test_on_vision_change_source(self, engine: GameEngine):
        """_on_vision_change with source key updates vision bridge."""
        engine._vision_bridge = MagicMock()
        engine._on_vision_change("source", "tank")
        engine._vision_bridge.set_source.assert_called_once_with("tank")

    def test_on_vision_change_creates_bridge(self, engine: GameEngine):
        """_on_vision_change with source=webcam creates bridge on demand."""
        assert engine._vision_bridge is None
        engine._on_vision_change("source", "webcam")
        assert engine._vision_bridge is not None


# ── ActionBar Integration Tests ──────────────────────────────────────


class TestActionBarIntegration:
    """Tests for ActionBar wiring in GameEngine."""

    def test_action_bar_created(self, engine: GameEngine):
        """ActionBar is created during init."""
        assert engine._action_bar is not None

    def test_action_bar_has_buttons(self, engine: GameEngine):
        """ActionBar has 7 buttons after initialization."""
        assert len(engine._action_bar.buttons) == 7

    def test_interaction_buttons_disabled(self, engine: GameEngine):
        """Old tiny interaction buttons are disabled."""
        assert engine._interaction_manager._buttons_enabled is False

    def test_action_bar_feed(self, engine: GameEngine):
        """Feed action from action bar feeds the creature."""
        engine._on_action_bar("feed")
        # Should add a notification
        assert len(engine._notifications) > 0

    def test_action_bar_temp_up(self, engine: GameEngine):
        """Temp+ action increases tank temperature."""
        initial_temp = engine.tank.temperature
        engine._on_action_bar("temp_up")
        assert engine.tank.temperature > initial_temp

    def test_action_bar_temp_down(self, engine: GameEngine):
        """Temp- action decreases tank temperature."""
        initial_temp = engine.tank.temperature
        engine._on_action_bar("temp_down")
        assert engine.tank.temperature < initial_temp

    def test_action_bar_clean(self, engine: GameEngine):
        """Clean action from action bar triggers clean."""
        engine._on_action_bar("clean")
        assert len(engine._notifications) > 0

    def test_action_bar_tap_glass(self, engine: GameEngine):
        """Tap action from action bar adds notification."""
        engine._on_action_bar("tap_glass")
        assert any("tap" in n[0].lower() for n in engine._notifications)

    def test_action_bar_ignored_during_game_over(self, engine: GameEngine):
        """Action bar actions are ignored during game over."""
        engine.game_over = True
        initial_notifs = len(engine._notifications)
        engine._on_action_bar("feed")
        assert len(engine._notifications) == initial_notifs

    def test_action_bar_click_routed(self, engine: GameEngine):
        """Mouse click on action bar button triggers action."""
        btn = engine._action_bar.buttons[0]  # Feed
        event = MagicMock()
        event.pos = (btn.x + 5, btn.y + 5)
        engine._on_mouse_click(event)
        # Should have added a notification
        assert len(engine._notifications) > 0

    def test_action_bar_aerate_aquarium(self, engine: GameEngine):
        """Aerate action boosts oxygen in aquarium mode."""
        engine._tank.oxygen_level = 0.5
        engine._on_action_bar("aerate")
        assert engine._tank.oxygen_level > 0.5
        assert len(engine._notifications) > 0

    def test_action_bar_aerate_terrarium_uses_sprinkle(self, engine: GameEngine):
        """Aerate action uses sprinkle in terrarium mode."""
        from seaman_brain.environment.tank import EnvironmentType
        engine._tank.environment_type = EnvironmentType.TERRARIUM
        engine._tank.water_level = 0.0
        engine._tank.oxygen_level = 0.5
        engine._on_action_bar("aerate")
        assert engine._tank.oxygen_level > 0.5
        assert len(engine._notifications) > 0

    def test_action_bar_render(self, engine: GameEngine):
        """ActionBar renders without crash."""
        engine._render(_surface_mock)


# ── Look Now Timeout / Failure Tests ─────────────────────────────────


class TestLookNowFailures:
    """Tests for Look Now timeout, capture failure, and success notifications."""

    def test_look_now_timeout_resets_and_notifies(self, engine: GameEngine):
        """Look Now timeout resets tracking state and shows notification."""
        import time as _time

        bridge = MagicMock()
        bridge._last_capture_failed = False
        bridge.get_recent_observations.return_value = ["old"]
        bridge.source = "webcam"
        engine._vision_bridge = bridge
        engine._vision_look_prev_count = 1
        engine._vision_look_start_time = _time.monotonic() - 31.0

        engine._check_vision_look_result()

        assert engine._vision_look_prev_count is None
        assert any("timed out" in n[0] for n in engine._notifications)

    def test_look_now_capture_failure_notifies(self, engine: GameEngine):
        """Look Now capture failure shows notification."""
        bridge = MagicMock()
        bridge._last_capture_failed = True
        bridge.source = "webcam"
        engine._vision_bridge = bridge
        engine._vision_look_prev_count = 0

        engine._check_vision_look_result()

        assert engine._vision_look_prev_count is None
        assert any("capture failed" in n[0] for n in engine._notifications)
        # Flag should be reset
        assert bridge._last_capture_failed is False

    def test_look_now_success_shows_observation(self, engine: GameEngine):
        """Look Now success shows 'Saw: ...' notification."""
        import time as _time

        bridge = MagicMock()
        bridge._last_capture_failed = False
        bridge.get_recent_observations.return_value = ["Human is waving", "old"]
        bridge.source = "webcam"
        engine._vision_bridge = bridge
        engine._vision_look_prev_count = 1
        engine._vision_look_start_time = _time.monotonic()

        engine._check_vision_look_result()

        assert engine._vision_look_prev_count is None
        assert any("Saw:" in n[0] for n in engine._notifications)


# ── STT Toggle Tests ─────────────────────────────────────────────────


class TestSTTToggle:
    """Tests for STT enable/disable wiring through _on_audio_change."""

    def test_stt_enable_without_audio_manager(self, engine: GameEngine):
        """Enabling STT without AudioManager shows 'not yet available'."""
        bridge = MagicMock()
        bridge._config = MagicMock()
        bridge._audio_manager = None
        engine._audio_bridge = bridge

        engine._on_audio_change("stt_enabled", True)

        assert any("not yet available" in n[0] for n in engine._notifications)

    def test_stt_disable_shows_notification(self, engine: GameEngine):
        """Disabling STT shows 'disabled' notification."""
        bridge = MagicMock()
        bridge._config = MagicMock()
        bridge.mic_active = False
        engine._audio_bridge = bridge

        engine._on_audio_change("stt_enabled", False)

        assert any("disabled" in n[0] for n in engine._notifications)


# ── Audio Config Attribute Regression Test ───────────────────────────


class TestAudioConfigAttribute:
    """Regression test: audio setting changes use correct attribute."""

    def test_audio_config_attribute_no_raise(self, engine: GameEngine):
        """Setting audio values via _on_audio_change doesn't raise AttributeError."""
        bridge = MagicMock()
        bridge._config = MagicMock()
        engine._audio_bridge = bridge

        # All these should access _config, not _audio_config
        engine._on_audio_change("tts_enabled", True)
        engine._on_audio_change("sfx_enabled", False)
        engine._on_audio_change("tts_volume", 0.8)
        engine._on_audio_change("sfx_volume", 0.5)
        engine._on_audio_change("ambient_volume", 0.3)

        assert bridge._config.tts_enabled is True
        assert bridge._config.sfx_enabled is False
        assert bridge._config.tts_volume == 0.8
        assert bridge._config.sfx_volume == 0.5
        assert bridge._config.ambient_volume == 0.3


# ── STT Callback Wiring Tests ───────────────────────────────────────


class TestSTTCallback:
    """Tests for STT result callback wiring in GameEngine."""

    def test_stt_result_submits_to_chat(self, engine: GameEngine):
        """STT result is forwarded as chat input."""
        initial_count = engine._chat_panel.message_count
        engine._on_stt_result("hello creature")
        assert engine._chat_panel.message_count > initial_count

    def test_stt_result_empty_ignored(self, engine: GameEngine):
        """Empty STT result is ignored."""
        initial_count = engine._chat_panel.message_count
        engine._on_stt_result("")
        assert engine._chat_panel.message_count == initial_count

    def test_audio_bridge_receives_callback(self, engine: GameEngine):
        """Audio bridge is initialized with the STT callback."""
        assert engine._audio_bridge._on_stt_result is not None
        assert engine._audio_bridge._on_stt_result == engine._on_stt_result


# ── Webcam Index Propagation Tests ───────────────────────────────────


class TestWebcamIndexPropagation:
    """Tests for webcam index propagation to live VisionBridge."""

    def test_webcam_index_change_updates_bridge(self, engine: GameEngine):
        """Changing webcam_index propagates to existing VisionBridge."""
        bridge = MagicMock()
        engine._vision_bridge = bridge
        engine._on_vision_change("webcam_index", 2)
        bridge.set_webcam_index.assert_called_once_with(2)
