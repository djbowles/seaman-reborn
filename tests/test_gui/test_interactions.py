"""Tests for interactive elements - feeding, temperature controls, glass tapping (US-040).

Pygame is mocked at module level to avoid requiring a display server in CI.
Uses the pattern from test_hud.py: sys.modules["pygame"] = mock, import once.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None

# Surface mock
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768

# Font mock
_font_mock = MagicMock()
_font_mock.get_linesize.return_value = 14
_font_mock.size.return_value = (60, 14)
_text_surf_mock = MagicMock()
_text_surf_mock.get_width.return_value = 60
_text_surf_mock.get_height.return_value = 14
_font_mock.render.return_value = _text_surf_mock
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock

# Draw mock
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.line.return_value = None
_pygame_mock.draw.circle.return_value = None

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)


# Surface constructor mock
def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 100
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 100
    return s


_pygame_mock.Surface = _make_surface

# Install pygame mock before importing gui modules
sys.modules["pygame"] = _pygame_mock

from seaman_brain.config import GUIConfig, NeedsConfig  # noqa: E402
from seaman_brain.creature.state import CreatureState  # noqa: E402
from seaman_brain.environment.tank import TankEnvironment  # noqa: E402
from seaman_brain.gui.interactions import (  # noqa: E402
    FoodDropEffect,
    InteractionManager,
    InteractionType,
    RippleEffect,
    TapReaction,
)
from seaman_brain.needs.feeding import FoodType  # noqa: E402
from seaman_brain.types import CreatureStage  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset draw mocks and re-install pygame mock between tests.

    Other test_gui modules also set sys.modules["pygame"] at module level,
    so we must re-install ours before each test to avoid cross-contamination.
    """
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.interactions as interactions_mod
    interactions_mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    _pygame_mock.Surface = _make_surface
    _pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.render.return_value = _text_surf_mock
    _font_mock.get_linesize.return_value = 14
    _font_mock.size.return_value = (60, 14)
    _text_surf_mock.get_width.return_value = 60
    _text_surf_mock.get_height.return_value = 14


@pytest.fixture()
def creature() -> CreatureState:
    """A creature with default values."""
    return CreatureState()


@pytest.fixture()
def hungry_creature() -> CreatureState:
    """A hungry Gillman creature (can eat pellets and worms)."""
    return CreatureState(stage=CreatureStage.GILLMAN, hunger=0.5)


@pytest.fixture()
def tank() -> TankEnvironment:
    """A tank with default values."""
    return TankEnvironment()


@pytest.fixture()
def manager() -> InteractionManager:
    """An InteractionManager with default config and zero-second feeding cooldown."""
    cfg = NeedsConfig(feeding_cooldown_seconds=0)
    return InteractionManager(needs_config=cfg)


# ── Construction Tests ───────────────────────────────────────────────


class TestConstruction:
    """Tests for InteractionManager initialization."""

    def test_default_config(self):
        """Manager uses default configs when none provided."""
        mgr = InteractionManager()
        assert mgr._gui_config.window_width == 1024
        assert mgr._gui_config.window_height == 768

    def test_custom_config(self):
        """Manager accepts custom configs."""
        gui = GUIConfig(window_width=800, window_height=600)
        mgr = InteractionManager(gui_config=gui)
        assert mgr._gui_config.window_width == 800

    def test_food_menu_starts_closed(self):
        """Food menu starts closed."""
        mgr = InteractionManager()
        assert mgr.food_menu_open is False

    def test_engines_created(self):
        """Feeding and care engines are created."""
        mgr = InteractionManager()
        assert mgr.feeding_engine is not None
        assert mgr.care_engine is not None


# ── Feeding Interaction Tests ────────────────────────────────────────


class TestFeedingInteraction:
    """Tests for feeding via click interaction."""

    def test_click_tank_center_opens_food_menu(
        self, manager: InteractionManager, creature: CreatureState, tank: TankEnvironment
    ):
        """Clicking center of tank opens food selection menu."""
        # Click center of tank (away from walls)
        result = manager.handle_click(512, 400, creature, tank)
        assert result is not None
        assert result.interaction_type == InteractionType.FEED
        assert manager.food_menu_open is True

    def test_food_menu_shows_stage_foods(
        self, manager: InteractionManager, hungry_creature: CreatureState,
        tank: TankEnvironment
    ):
        """Food menu shows foods appropriate for creature stage."""
        # Gillman can eat pellets and worms
        manager.handle_click(512, 400, hungry_creature, tank)
        assert manager.food_menu_open is True
        menu_items = manager._food_menu_items
        assert FoodType.PELLET in menu_items
        assert FoodType.WORM in menu_items
        assert FoodType.INSECT not in menu_items

    def test_select_food_feeds_creature(
        self, manager: InteractionManager, hungry_creature: CreatureState,
        tank: TankEnvironment
    ):
        """Selecting food from menu feeds the creature and reduces hunger."""
        initial_hunger = hungry_creature.hunger

        # Open menu
        manager.handle_click(512, 400, hungry_creature, tank)
        assert manager.food_menu_open is True

        # Click first menu item (menu is at click position)
        menu_x = manager._food_menu_x + 10
        menu_y = manager._food_menu_y + 5
        result = manager.handle_click(menu_x, menu_y, hungry_creature, tank)

        assert result is not None
        assert result.feeding_result is not None
        assert result.feeding_result.success is True
        assert hungry_creature.hunger < initial_hunger
        assert manager.food_menu_open is False

    def test_food_drop_effect_spawned_on_feed(
        self, manager: InteractionManager, hungry_creature: CreatureState,
        tank: TankEnvironment
    ):
        """Successful feeding spawns a food drop visual effect."""
        # Open and select food
        manager.handle_click(512, 400, hungry_creature, tank)
        menu_x = manager._food_menu_x + 10
        menu_y = manager._food_menu_y + 5
        manager.handle_click(menu_x, menu_y, hungry_creature, tank)

        assert len(manager.food_drops) == 1

    def test_click_outside_menu_closes_it(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Clicking outside the food menu closes it."""
        manager.handle_click(512, 400, creature, tank)
        assert manager.food_menu_open is True

        # Click far away from menu
        manager.handle_click(50, 50, creature, tank)
        assert manager.food_menu_open is False

    def test_interaction_count_increments_on_feed(
        self, manager: InteractionManager, hungry_creature: CreatureState,
        tank: TankEnvironment
    ):
        """Creature interaction count increases on successful feeding."""
        initial_count = hungry_creature.interaction_count
        manager.handle_click(512, 400, hungry_creature, tank)
        menu_x = manager._food_menu_x + 10
        menu_y = manager._food_menu_y + 5
        manager.handle_click(menu_x, menu_y, hungry_creature, tank)

        assert hungry_creature.interaction_count == initial_count + 1


# ── Temperature Control Tests ────────────────────────────────────────


class TestTemperatureControl:
    """Tests for temperature up/down buttons."""

    def test_temp_up_increases_temperature(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Clicking temp up button increases temperature."""
        initial_temp = tank.temperature
        # Force build buttons
        manager._build_buttons()

        # Find temp up button
        btn = [b for b in manager._buttons if b.action == InteractionType.TEMP_UP][0]
        result = manager.handle_click(
            btn.x + 1, btn.y + 1, creature, tank
        )

        assert result is not None
        assert result.interaction_type == InteractionType.TEMP_UP
        assert tank.temperature > initial_temp
        assert result.care_result is not None
        assert result.care_result.success is True

    def test_temp_down_decreases_temperature(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Clicking temp down button decreases temperature."""
        initial_temp = tank.temperature
        manager._build_buttons()

        btn = [b for b in manager._buttons if b.action == InteractionType.TEMP_DOWN][0]
        result = manager.handle_click(
            btn.x + 1, btn.y + 1, creature, tank
        )

        assert result is not None
        assert result.interaction_type == InteractionType.TEMP_DOWN
        assert tank.temperature < initial_temp

    def test_temp_result_has_message(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Temperature adjustment returns a descriptive message."""
        manager._build_buttons()
        btn = [b for b in manager._buttons if b.action == InteractionType.TEMP_UP][0]
        result = manager.handle_click(
            btn.x + 1, btn.y + 1, creature, tank
        )
        assert result is not None
        assert len(result.message) > 0
        assert "Temperature" in result.message or "temperature" in result.message


# ── Glass Tap Tests ──────────────────────────────────────────────────


class TestGlassTap:
    """Tests for tapping on the tank glass."""

    def test_tap_near_wall_triggers_reaction(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Clicking near tank wall triggers a glass tap reaction."""
        # Click near left wall
        result = manager.handle_click(5, 400, creature, tank)

        assert result is not None
        assert result.interaction_type == InteractionType.TAP_GLASS
        assert result.tap_reaction is not None

    def test_tap_startled_for_low_trust(
        self, manager: InteractionManager, tank: TankEnvironment
    ):
        """Low trust creature is startled by glass tap."""
        creature = CreatureState(trust_level=0.1)
        result = manager.handle_click(5, 400, creature, tank)

        assert result is not None
        assert result.tap_reaction == TapReaction.STARTLED

    def test_tap_curious_for_high_trust(
        self, manager: InteractionManager, tank: TankEnvironment
    ):
        """High trust creature is curious about glass tap."""
        creature = CreatureState(trust_level=0.8)
        result = manager.handle_click(5, 400, creature, tank)

        assert result is not None
        assert result.tap_reaction == TapReaction.CURIOUS

    def test_repeated_taps_cause_annoyance(
        self, manager: InteractionManager, tank: TankEnvironment
    ):
        """Tapping glass 3+ times causes annoyance."""
        creature = CreatureState(trust_level=0.5)

        # Tap 3 times
        manager.handle_click(5, 400, creature, tank)
        manager.handle_click(5, 400, creature, tank)
        result = manager.handle_click(5, 400, creature, tank)

        assert result is not None
        assert result.tap_reaction == TapReaction.ANNOYED

    def test_tap_increments_interaction_count(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Glass tap increments the creature interaction counter."""
        initial = creature.interaction_count
        manager.handle_click(5, 400, creature, tank)
        assert creature.interaction_count == initial + 1

    def test_tap_creates_ripple_effect(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Glass tap spawns a ripple visual effect."""
        manager.handle_click(5, 400, creature, tank)
        assert len(manager.ripples) >= 1


# ── Clean Tank Tests ─────────────────────────────────────────────────


class TestCleanTank:
    """Tests for the clean tank button."""

    def test_clean_increases_cleanliness(
        self, manager: InteractionManager, creature: CreatureState,
    ):
        """Clicking clean button increases tank cleanliness."""
        dirty_tank = TankEnvironment(cleanliness=0.3)
        manager._build_buttons()

        btn = [b for b in manager._buttons if b.action == InteractionType.CLEAN][0]
        result = manager.handle_click(
            btn.x + 1, btn.y + 1, creature, dirty_tank
        )

        assert result is not None
        assert result.interaction_type == InteractionType.CLEAN
        assert dirty_tank.cleanliness > 0.3

    def test_clean_creates_ripple(
        self, manager: InteractionManager, creature: CreatureState,
    ):
        """Clean button creates a ripple visual effect."""
        dirty_tank = TankEnvironment(cleanliness=0.3)
        manager._build_buttons()

        btn = [b for b in manager._buttons if b.action == InteractionType.CLEAN][0]
        manager.handle_click(btn.x + 1, btn.y + 1, creature, dirty_tank)

        assert len(manager.ripples) >= 1


# ── Drain/Fill Tests ─────────────────────────────────────────────────


class TestDrainFill:
    """Tests for the drain/fill tank button."""

    def test_drain_switches_to_terrarium(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Drain button switches tank to terrarium mode."""
        manager._build_buttons()

        btn = [b for b in manager._buttons if b.action == InteractionType.DRAIN][0]
        result = manager.handle_click(
            btn.x + 1, btn.y + 1, creature, tank
        )

        assert result is not None
        assert result.interaction_type == InteractionType.DRAIN
        assert tank.water_level == 0.0

    def test_fill_switches_to_aquarium(
        self, manager: InteractionManager, creature: CreatureState,
    ):
        """Fill button switches tank back to aquarium mode."""
        drained_tank = TankEnvironment(water_level=0.0)
        drained_tank.drain()  # Set to terrarium

        manager._build_buttons()
        btn = [b for b in manager._buttons if b.action == InteractionType.DRAIN][0]
        result = manager.handle_click(
            btn.x + 1, btn.y + 1, creature, drained_tank
        )

        assert result is not None
        assert result.care_result is not None


# ── Visual Effect Tests ──────────────────────────────────────────────


class TestVisualEffects:
    """Tests for ripple and food drop animations."""

    def test_ripple_fades_over_time(self):
        """Ripple alpha decreases as it expands."""
        r = RippleEffect(x=100, y=100)
        initial_alpha = r.alpha
        r.update(0.5)
        assert r.alpha < initial_alpha

    def test_ripple_dies_when_fully_expanded(self):
        """Ripple becomes not-alive when fully expanded."""
        r = RippleEffect(x=100, y=100, max_radius=10, speed=100)
        for _ in range(10):
            r.update(0.1)
        assert r.alive is False

    def test_food_drop_falls_to_target(self):
        """Food drop moves downward toward target_y."""
        f = FoodDropEffect(x=100, y=100, target_y=200, speed=200)
        f.update(0.3)
        assert f.y > 100
        assert f.landed is False

    def test_food_drop_lands_at_target(self):
        """Food drop stops at target_y and marks as landed."""
        f = FoodDropEffect(x=100, y=100, target_y=110, speed=200)
        f.update(1.0)
        assert f.y == 110.0
        assert f.landed is True

    def test_food_drop_fades_after_landing(self):
        """Food drop alpha decreases after landing."""
        f = FoodDropEffect(x=100, y=100, target_y=110, speed=200)
        f.update(1.0)  # Land
        assert f.landed is True
        f.update(0.5)  # Start fading
        assert f.alpha < 1.0

    def test_update_clears_dead_effects(self, manager: InteractionManager):
        """Manager update removes dead visual effects."""
        # Add a ripple that will die quickly
        r = RippleEffect(x=100, y=100, max_radius=10, speed=1000)
        manager._ripples.append(r)

        # Update enough for it to die
        manager.update(1.0)
        assert len(manager.ripples) == 0


# ── Rendering Tests ──────────────────────────────────────────────────


class TestRendering:
    """Tests for render() calls."""

    def test_render_draws_buttons(self, manager: InteractionManager):
        """Render draws button rectangles."""
        manager.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count > 0

    def test_render_with_food_menu_open(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Render with food menu open draws additional elements."""
        manager.handle_click(512, 400, creature, tank)
        assert manager.food_menu_open is True

        _pygame_mock.draw.reset_mock()
        manager.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count > 0

    def test_render_ripples(self, manager: InteractionManager):
        """Render draws active ripples."""
        manager._ripples.append(RippleEffect(x=100, y=100))
        manager.render(_surface_mock)
        assert _pygame_mock.draw.circle.call_count > 0

    def test_render_food_drops(self, manager: InteractionManager):
        """Render draws active food drops."""
        manager._food_drops.append(
            FoodDropEffect(x=100, y=100, target_y=200)
        )
        manager.render(_surface_mock)
        assert _pygame_mock.draw.circle.call_count > 0


# ── Mouse Move Tests ─────────────────────────────────────────────────


class TestMouseMove:
    """Tests for mouse hover detection."""

    def test_hover_over_button(self, manager: InteractionManager):
        """Moving mouse over a button sets its hover state."""
        manager._build_buttons()
        btn = manager._buttons[0]

        manager.handle_mouse_move(btn.x + 1, btn.y + 1)
        assert btn.hover is True

    def test_hover_away_from_button(self, manager: InteractionManager):
        """Moving mouse away from button clears hover state."""
        manager._build_buttons()
        btn = manager._buttons[0]

        manager.handle_mouse_move(btn.x + 1, btn.y + 1)
        assert btn.hover is True

        manager.handle_mouse_move(0, 0)
        assert btn.hover is False


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_click_outside_tank_returns_none(
        self, manager: InteractionManager, creature: CreatureState,
        tank: TankEnvironment
    ):
        """Clicking outside the tank area returns None."""
        # Set small tank area
        manager.set_tank_area(100, 100, 200, 200)
        result = manager.handle_click(50, 50, creature, tank)
        assert result is None

    def test_set_tank_area_resets_buttons(self, manager: InteractionManager):
        """Setting tank area forces button rebuild."""
        manager._build_buttons()
        assert manager._buttons_built is True

        manager.set_tank_area(0, 0, 800, 600)
        assert manager._buttons_built is False

    def test_all_stages_have_food_menu(
        self, manager: InteractionManager, tank: TankEnvironment
    ):
        """All creature stages can open the food menu."""
        for stage in CreatureStage:
            creature = CreatureState(stage=stage)
            manager.food_menu_open = False
            result = manager.handle_click(512, 400, creature, tank)
            assert result is not None
            assert result.interaction_type == InteractionType.FEED

    def test_tap_cooldown_resets_after_time(
        self, manager: InteractionManager
    ):
        """Tap count resets after cooldown expires."""
        manager._recent_tap_count = 5
        manager._tap_cooldown = 1.0

        manager.update(2.0)
        assert manager._recent_tap_count == 0
