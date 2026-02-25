"""Tests for the HUD and status display (US-039).

Pygame is mocked at module level to avoid requiring a display server in CI.
Uses the pattern from test_chat_panel.py: sys.modules["pygame"] = mock, import once.
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
_font_mock.get_linesize.return_value = 16
_font_mock.size.return_value = (80, 16)
_text_surf_mock = MagicMock()
_text_surf_mock.get_width.return_value = 80
_text_surf_mock.get_height.return_value = 16
_font_mock.render.return_value = _text_surf_mock
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock

# Draw mock
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.line.return_value = None
_pygame_mock.draw.circle.return_value = None

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)


# Surface constructor mock — returns a fresh MagicMock each time
def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 100
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 100
    return s


_pygame_mock.Surface = _make_surface

# Install pygame mock before importing gui modules
sys.modules["pygame"] = _pygame_mock

from seaman_brain.config import GUIConfig  # noqa: E402
from seaman_brain.creature.state import CreatureState  # noqa: E402
from seaman_brain.environment.tank import TankEnvironment  # noqa: E402
from seaman_brain.gui.hud import (  # noqa: E402
    _COLOR_BLUE,
    _COLOR_GREEN,
    _COLOR_RED,
    _COLOR_YELLOW,
    _MOOD_COLORS,
    _STAGE_NAMES,
    HUD,
    HUDMetric,
    _status_color,
)
from seaman_brain.types import CreatureStage  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset draw mocks between tests."""
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    # Restore render return value after reset
    _font_mock.render.return_value = _text_surf_mock
    _font_mock.get_linesize.return_value = 16
    _font_mock.size.return_value = (80, 16)
    _text_surf_mock.get_width.return_value = 80
    _text_surf_mock.get_height.return_value = 16


@pytest.fixture()
def default_creature() -> CreatureState:
    """A creature with default values."""
    return CreatureState()


@pytest.fixture()
def default_tank() -> TankEnvironment:
    """A tank with default values."""
    return TankEnvironment()


@pytest.fixture()
def hud() -> HUD:
    """An HUD with default config."""
    return HUD()


# ── Construction Tests ───────────────────────────────────────────────


class TestHUDConstruction:
    """Tests for HUD initialization."""

    def test_default_config(self):
        """HUD uses default GUIConfig when none provided."""
        h = HUD()
        assert h._config.window_width == 1024
        assert h._config.window_height == 768

    def test_custom_config(self):
        """HUD accepts custom GUIConfig."""
        cfg = GUIConfig(window_width=800, window_height=600)
        h = HUD(gui_config=cfg)
        assert h._config.window_width == 800
        assert h._config.window_height == 600

    def test_starts_in_expanded_mode(self):
        """HUD defaults to expanded (not compact) mode."""
        h = HUD()
        assert h.compact is False

    def test_session_time_starts_at_zero(self):
        """Session timer starts at zero."""
        h = HUD()
        assert h.session_time == 0.0


# ── Color Threshold Tests ────────────────────────────────────────────


class TestColorThresholds:
    """Tests for _status_color function — green/yellow/red thresholds."""

    def test_high_value_is_green(self):
        """Value >= 0.6 should return green."""
        assert _status_color(0.8) == _COLOR_GREEN
        assert _status_color(1.0) == _COLOR_GREEN
        assert _status_color(0.6) == _COLOR_GREEN

    def test_medium_value_is_yellow(self):
        """Value 0.3-0.59 should return yellow."""
        assert _status_color(0.5) == _COLOR_YELLOW
        assert _status_color(0.3) == _COLOR_YELLOW

    def test_low_value_is_red(self):
        """Value < 0.3 should return red."""
        assert _status_color(0.1) == _COLOR_RED
        assert _status_color(0.0) == _COLOR_RED
        assert _status_color(0.29) == _COLOR_RED

    def test_inverted_high_value_is_red(self):
        """Inverted: high value (e.g. hunger 1.0) should be red."""
        assert _status_color(0.8, inverted=True) == _COLOR_RED
        assert _status_color(1.0, inverted=True) == _COLOR_RED

    def test_inverted_low_value_is_green(self):
        """Inverted: low value (e.g. hunger 0.0) should be green."""
        assert _status_color(0.0, inverted=True) == _COLOR_GREEN
        assert _status_color(0.1, inverted=True) == _COLOR_GREEN

    def test_inverted_medium_is_yellow(self):
        """Inverted: medium value should be yellow."""
        assert _status_color(0.5, inverted=True) == _COLOR_YELLOW


# ── Mode Switching Tests ─────────────────────────────────────────────


class TestModeSwitching:
    """Tests for compact/expanded mode toggling."""

    def test_toggle_to_compact(self, hud: HUD):
        """toggle_mode() switches from expanded to compact."""
        assert hud.compact is False
        hud.toggle_mode()
        assert hud.compact is True

    def test_toggle_back_to_expanded(self, hud: HUD):
        """toggle_mode() twice returns to expanded."""
        hud.toggle_mode()
        hud.toggle_mode()
        assert hud.compact is False

    def test_compact_renders_without_crash(
        self, hud: HUD, default_creature: CreatureState, default_tank: TankEnvironment
    ):
        """Compact mode render does not crash."""
        hud.compact = True
        hud.render(_surface_mock, default_creature, default_tank)
        # If we got here, no crash

    def test_expanded_renders_without_crash(
        self, hud: HUD, default_creature: CreatureState, default_tank: TankEnvironment
    ):
        """Expanded mode render does not crash."""
        hud.compact = False
        hud.render(_surface_mock, default_creature, default_tank)


# ── Update Tests ─────────────────────────────────────────────────────


class TestHUDUpdate:
    """Tests for HUD update (session timer)."""

    def test_session_time_increments(self, hud: HUD):
        """Session timer increases with update()."""
        hud.update(1.5)
        assert hud.session_time == pytest.approx(1.5)

    def test_multiple_updates_accumulate(self, hud: HUD):
        """Multiple update() calls accumulate session time."""
        hud.update(1.0)
        hud.update(2.5)
        hud.update(0.5)
        assert hud.session_time == pytest.approx(4.0)


# ── Bar Rendering Tests ──────────────────────────────────────────────


class TestBarRendering:
    """Tests for metric bar rendering via render()."""

    def test_render_calls_draw_rect(
        self, hud: HUD, default_creature: CreatureState, default_tank: TankEnvironment
    ):
        """Rendering draws rectangles for bars."""
        hud.render(_surface_mock, default_creature, default_tank)
        # Should have drawn rect calls for bar backgrounds, fills, and borders
        assert _pygame_mock.draw.rect.call_count > 0

    def test_render_calls_font_render(
        self, hud: HUD, default_creature: CreatureState, default_tank: TankEnvironment
    ):
        """Rendering draws text for labels and values."""
        hud.render(_surface_mock, default_creature, default_tank)
        assert _font_mock.render.call_count > 0

    def test_starving_creature_hunger_bar_red(self, hud: HUD):
        """Hunger bar color is red when creature is starving."""
        creature = CreatureState(hunger=0.9)
        metrics = hud._build_need_metrics(creature)
        hunger = metrics[0]
        assert hunger.color == _COLOR_RED

    def test_healthy_creature_health_bar_green(self, hud: HUD):
        """Health bar color is green when creature is healthy."""
        creature = CreatureState(health=1.0)
        metrics = hud._build_need_metrics(creature)
        health = metrics[1]
        assert health.color == _COLOR_GREEN

    def test_low_health_bar_red(self, hud: HUD):
        """Health bar color is red when health is low."""
        creature = CreatureState(health=0.1)
        metrics = hud._build_need_metrics(creature)
        health = metrics[1]
        assert health.color == _COLOR_RED

    def test_medium_comfort_bar_yellow(self, hud: HUD):
        """Comfort bar color is yellow when comfort is medium."""
        creature = CreatureState(comfort=0.4)
        metrics = hud._build_need_metrics(creature)
        comfort = metrics[2]
        assert comfort.color == _COLOR_YELLOW

    def test_hunger_bar_value_is_inverted(self, hud: HUD):
        """Hunger bar_value shows fullness (1 - hunger)."""
        creature = CreatureState(hunger=0.3)
        metrics = hud._build_need_metrics(creature)
        hunger = metrics[0]
        assert hunger.bar_value == pytest.approx(0.7)

    def test_health_bar_value_direct(self, hud: HUD):
        """Health bar_value is the direct health value."""
        creature = CreatureState(health=0.8)
        metrics = hud._build_need_metrics(creature)
        health = metrics[1]
        assert health.bar_value == pytest.approx(0.8)


# ── Tank Indicator Tests ─────────────────────────────────────────────


class TestTankIndicators:
    """Tests for tank environment indicators."""

    def test_optimal_temperature_is_green(self, hud: HUD):
        """Temperature in optimal range shows green."""
        tank = TankEnvironment(temperature=24.0)
        metrics = hud._build_tank_metrics(tank)
        temp = metrics[0]
        assert temp.color == _COLOR_GREEN

    def test_cold_temperature_is_blue(self, hud: HUD):
        """Cold temperature shows blue."""
        tank = TankEnvironment(temperature=15.0)
        metrics = hud._build_tank_metrics(tank)
        temp = metrics[0]
        assert temp.color == _COLOR_BLUE

    def test_hot_temperature_is_red(self, hud: HUD):
        """Hot temperature shows red."""
        tank = TankEnvironment(temperature=35.0)
        metrics = hud._build_tank_metrics(tank)
        temp = metrics[0]
        assert temp.color == _COLOR_RED

    def test_clean_tank_is_green(self, hud: HUD):
        """Cleanliness at 1.0 shows green."""
        tank = TankEnvironment(cleanliness=1.0)
        metrics = hud._build_tank_metrics(tank)
        clean = metrics[1]
        assert clean.color == _COLOR_GREEN

    def test_dirty_tank_is_red(self, hud: HUD):
        """Cleanliness near 0.0 shows red."""
        tank = TankEnvironment(cleanliness=0.1)
        metrics = hud._build_tank_metrics(tank)
        clean = metrics[1]
        assert clean.color == _COLOR_RED

    def test_oxygen_bar_present(self, hud: HUD):
        """Oxygen bar is included in tank metrics."""
        tank = TankEnvironment(oxygen_level=0.8)
        metrics = hud._build_tank_metrics(tank)
        assert len(metrics) == 3
        oxygen = metrics[2]
        assert oxygen.label == "Oxygen"
        assert oxygen.bar_value == pytest.approx(0.8)

    def test_temperature_display_text_format(self, hud: HUD):
        """Temperature display text shows Celsius."""
        tank = TankEnvironment(temperature=24.5)
        metrics = hud._build_tank_metrics(tank)
        temp = metrics[0]
        assert temp.display_text == "24.5C"


# ── Trust Meter Tests ────────────────────────────────────────────────


class TestTrustMeter:
    """Tests for trust level display."""

    def test_zero_trust_is_red(self, hud: HUD):
        """Zero trust shows red."""
        creature = CreatureState(trust_level=0.0)
        metric = hud._build_trust_metric(creature)
        assert metric.color == _COLOR_RED

    def test_full_trust_is_green(self, hud: HUD):
        """Full trust shows green."""
        creature = CreatureState(trust_level=1.0)
        metric = hud._build_trust_metric(creature)
        assert metric.color == _COLOR_GREEN

    def test_trust_bar_value_matches(self, hud: HUD):
        """Trust bar value matches creature trust level."""
        creature = CreatureState(trust_level=0.5)
        metric = hud._build_trust_metric(creature)
        assert metric.bar_value == pytest.approx(0.5)
        assert metric.display_text == "50%"


# ── Top Bar Tests ────────────────────────────────────────────────────


class TestTopBar:
    """Tests for the top status bar."""

    def test_stage_names_cover_all_stages(self):
        """All CreatureStage values have display names."""
        for stage in CreatureStage:
            assert stage in _STAGE_NAMES

    def test_mood_colors_cover_expected_moods(self):
        """Known mood strings have color mappings."""
        expected = {"hostile", "irritated", "sardonic", "neutral",
                    "curious", "amused", "philosophical", "content"}
        assert set(_MOOD_COLORS.keys()) == expected

    def test_session_time_format_minutes(self, hud: HUD):
        """Session time formats as MM:SS for short sessions."""
        hud._session_time = 65.0  # 1 minute 5 seconds
        result = hud._format_session_time()
        assert result == "01:05"

    def test_session_time_format_hours(self, hud: HUD):
        """Session time formats as HH:MM:SS for long sessions."""
        hud._session_time = 3661.0  # 1 hour 1 minute 1 second
        result = hud._format_session_time()
        assert result == "1:01:01"

    def test_session_time_format_zero(self, hud: HUD):
        """Session time formats as 00:00 at start."""
        result = hud._format_session_time()
        assert result == "00:00"


# ── Edge Case Tests ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_extreme_hunger_clamp(self, hud: HUD):
        """Creature at max hunger (1.0) produces valid bar."""
        creature = CreatureState(hunger=1.0)
        metrics = hud._build_need_metrics(creature)
        hunger = metrics[0]
        assert hunger.bar_value == pytest.approx(0.0)
        assert hunger.color == _COLOR_RED

    def test_all_needs_zero(self, hud: HUD):
        """Creature with all needs at worst produces valid bars."""
        creature = CreatureState(hunger=1.0, health=0.0, comfort=0.0)
        metrics = hud._build_need_metrics(creature)
        assert len(metrics) == 3
        for m in metrics:
            assert m.color == _COLOR_RED

    def test_perfect_creature_all_green(self, hud: HUD):
        """Creature in perfect state has all green bars."""
        creature = CreatureState(hunger=0.0, health=1.0, comfort=1.0, trust_level=1.0)
        need_metrics = hud._build_need_metrics(creature)
        trust = hud._build_trust_metric(creature)
        for m in need_metrics:
            assert m.color == _COLOR_GREEN
        assert trust.color == _COLOR_GREEN

    def test_all_stages_render(
        self, hud: HUD, default_tank: TankEnvironment
    ):
        """HUD renders without crash for all creature stages."""
        for stage in CreatureStage:
            creature = CreatureState(stage=stage)
            hud.render(_surface_mock, creature, default_tank)

    def test_render_with_extreme_temperature(
        self, hud: HUD, default_creature: CreatureState
    ):
        """HUD renders without crash for extreme temperatures."""
        tank = TankEnvironment(temperature=5.0)
        hud.render(_surface_mock, default_creature, tank)
        tank2 = TankEnvironment(temperature=40.0)
        hud.render(_surface_mock, default_creature, tank2)

    def test_hudmetric_dataclass(self):
        """HUDMetric dataclass creates correctly."""
        m = HUDMetric(
            icon="H", label="Hunger", value=0.5,
            display_text="50%", color=(60, 200, 100), bar_value=0.5
        )
        assert m.icon == "H"
        assert m.label == "Hunger"
        assert m.value == 0.5
        assert m.display_text == "50%"
        assert m.color == (60, 200, 100)
        assert m.bar_value == 0.5
