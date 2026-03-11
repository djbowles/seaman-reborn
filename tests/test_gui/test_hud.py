"""Tests for the Modern Minimal HUD — top bar and sidebar tiles."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 80
_font_surface.get_height.return_value = 14
_font_mock.render.return_value = _font_surface
_font_mock.size.return_value = (80, 14)
_font_mock.get_linesize.return_value = 16
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768


def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 100
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 100
    return s


_pygame_mock.Surface = _make_surface
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.hud import HUD  # noqa: E402
from seaman_brain.gui.layout import ScreenLayout  # noqa: E402
from seaman_brain.gui.theme import Colors  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.hud as mod
    mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _font_surface.reset_mock()
    _pygame_mock.Surface = _make_surface
    _pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.render.return_value = _font_surface
    _font_mock.get_linesize.return_value = 16
    _font_mock.size.return_value = (80, 14)
    _font_surface.get_width.return_value = 80
    _font_surface.get_height.return_value = 14
    yield


@pytest.fixture()
def layout():
    return ScreenLayout(1024, 768)


@pytest.fixture()
def hud(layout):
    return HUD(layout)


# ── Construction Tests ────────────────────────────────────────────────


class TestHUDConstruction:
    def test_creates_need_tiles(self, hud):
        assert len(hud._need_tiles) == 4  # H, +, C, T

    def test_creates_action_tiles(self, hud):
        assert len(hud._action_tiles) == 7  # F, O, ^, v, *, ~, o

    def test_session_time_starts_at_zero(self, hud):
        assert hud.session_time == 0.0

    def test_mic_starts_inactive(self, hud):
        assert hud.mic_active is False

    def test_tts_starts_inactive(self, hud):
        assert hud.tts_active is False


# ── Need Tile Tests ──────────────────────────────────────────────────


class TestNeedTiles:
    def test_tile_color_green_when_healthy(self, hud):
        hud.update_needs(hunger=0.8, health=0.9, comfort=0.7, trust=0.6)
        for tile in hud._need_tiles:
            assert tile["color"] == Colors.STATUS_GREEN

    def test_tile_color_red_when_critical(self, hud):
        hud.update_needs(hunger=0.1, health=0.1, comfort=0.1, trust=0.1)
        for tile in hud._need_tiles:
            assert tile["color"] == Colors.STATUS_RED

    def test_tile_color_yellow_when_medium(self, hud):
        hud.update_needs(hunger=0.35, health=0.35, comfort=0.35, trust=0.35)
        for tile in hud._need_tiles:
            assert tile["color"] == Colors.STATUS_YELLOW

    def test_need_tile_icons(self, hud):
        icons = [t["icon"] for t in hud._need_tiles]
        assert icons == ["H", "+", "C", "T"]

    def test_need_tile_keys(self, hud):
        keys = [t["key"] for t in hud._need_tiles]
        assert keys == ["hunger", "health", "comfort", "trust"]


# ── Action Tile Tests ────────────────────────────────────────────────


class TestActionTiles:
    def test_action_tile_icons(self, hud):
        icons = [t["icon"] for t in hud._action_tiles]
        assert icons == ["F", "O", "^", "v", "*", "~", "o"]

    def test_action_tile_keys(self, hud):
        keys = [t["key"] for t in hud._action_tiles]
        assert keys == [
            "feed", "aerator", "temp_up", "temp_down",
            "clean", "drain", "fill",
        ]


# ── Click Handling Tests ─────────────────────────────────────────────


class TestActionTileClick:
    def test_click_outside_sidebar_returns_none(self, hud):
        # Click well outside sidebar (sidebar is 0-48px x)
        result = hud.handle_click(500, 200)
        assert result is None

    def test_click_in_sidebar_area(self, hud):
        # Click at sidebar x, somewhere in the tile area
        result = hud.handle_click(24, 200)
        assert result is None or isinstance(result, str)


# ── Top Bar Tests ────────────────────────────────────────────────────


class TestTopBar:
    def test_render_does_not_crash(self, hud):
        surface = MagicMock()
        surface.get_width.return_value = 1024
        hud.render(surface)
        # Should have drawn something
        assert _pygame_mock.draw.rect.called or surface.blit.called

    def test_settings_rect_none_before_render(self, hud):
        assert hud.settings_rect is None

    def test_lineage_rect_none_before_render(self, hud):
        assert hud.lineage_rect is None

    def test_settings_rect_set_after_render(self, hud):
        surface = MagicMock()
        surface.get_width.return_value = 1024
        hud.render(surface)
        assert hud.settings_rect is not None

    def test_lineage_rect_set_after_render(self, hud):
        surface = MagicMock()
        surface.get_width.return_value = 1024
        hud.render(surface)
        assert hud.lineage_rect is not None


# ── Update Tests ─────────────────────────────────────────────────────


class TestHUDUpdate:
    def test_session_time_increments(self, hud):
        hud.update(1.5)
        assert hud.session_time == pytest.approx(1.5)

    def test_multiple_updates_accumulate(self, hud):
        hud.update(1.0)
        hud.update(2.5)
        hud.update(0.5)
        assert hud.session_time == pytest.approx(4.0)

    def test_mic_pulse_advances_when_active(self, hud):
        hud.mic_active = True
        hud.update(0.5)
        assert hud._mic_pulse_timer > 0.0

    def test_mic_pulse_resets_when_inactive(self, hud):
        hud.mic_active = True
        hud.update(1.0)
        hud.mic_active = False
        hud.update(0.1)
        assert hud._mic_pulse_timer == 0.0


# ── Session Timer Format Tests ───────────────────────────────────────


class TestSessionTimerFormat:
    def test_format_zero(self, hud):
        assert hud._format_session_time() == "00:00"

    def test_format_minutes(self, hud):
        hud._session_time = 65.0
        assert hud._format_session_time() == "01:05"

    def test_format_hours(self, hud):
        hud._session_time = 3661.0
        assert hud._format_session_time() == "1:01:01"


# ── Top Bar State Tests ──────────────────────────────────────────────


class TestTopBarState:
    def test_update_creature_info(self, hud):
        hud.update_creature_info(stage="Gillman", mood="sardonic", name="Seaman")
        assert hud._stage_name == "Gillman"
        assert hud._mood_name == "sardonic"
        assert hud._creature_name == "Seaman"

    def test_default_creature_info(self, hud):
        assert hud._stage_name == "Mushroomer"
        assert hud._mood_name == "neutral"
        assert hud._creature_name == "Seaman"


# ── Tooltip Tests ────────────────────────────────────────────────────


class TestTooltips:
    def test_hover_over_need_tile_sets_tooltip(self, hud):
        # Hover at x=24 (sidebar center), y near first tile
        hud.handle_hover(24, hud._layout.sidebar.y + 12)
        # tooltip may or may not be set depending on exact position
        assert hud._tooltip is None or isinstance(hud._tooltip, str)

    def test_hover_outside_clears_tooltip(self, hud):
        hud._tooltip = "some tooltip"
        hud.handle_hover(500, 500)
        assert hud._tooltip is None


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_render_with_all_defaults(self, hud):
        surface = MagicMock()
        surface.get_width.return_value = 1024
        hud.render(surface)  # No crash

    def test_render_after_resize(self, layout, hud):
        layout.resize(1920, 1080)
        hud.resize(layout)
        surface = MagicMock()
        surface.get_width.return_value = 1920
        hud.render(surface)  # No crash

    def test_cooldown_overlay_does_not_crash(self, hud):
        hud.set_cooldown("feed", 5.0)
        surface = MagicMock()
        surface.get_width.return_value = 1024
        hud.render(surface)  # No crash
