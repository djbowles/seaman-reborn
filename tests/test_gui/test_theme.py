"""Tests for the Modern Minimal theme system."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Module-level pygame mock
_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 16
_font_mock.render.return_value = _font_surface
_font_mock.size.side_effect = lambda text: (len(text) * 8, 16)
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
sys.modules["pygame"] = _pygame_mock

import pytest  # noqa: E402

from seaman_brain.gui.theme import (  # noqa: E402
    VOID_BG,
    Colors,
    Fonts,
    Sizes,
    mood_glow_color,
    render_spaced_text,
    status_color,
)


@pytest.fixture(autouse=True)
def _reinstall_pygame_mock():
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.theme as mod
    mod.pygame = _pygame_mock
    yield


class TestColors:
    def test_void_bg_is_near_black(self):
        assert VOID_BG == (8, 8, 15)

    def test_surface_opacity_values(self):
        # 3% of 255 ~ 8, 5% ~ 13, 6% ~ 15
        assert Colors.SURFACE_3 == (255, 255, 255, 8)
        assert Colors.SURFACE_5 == (255, 255, 255, 13)
        assert Colors.BORDER == (255, 255, 255, 15)

    def test_status_colors_exist(self):
        assert Colors.STATUS_GREEN == (74, 222, 128)
        assert Colors.STATUS_YELLOW == (245, 158, 11)
        assert Colors.STATUS_RED == (239, 68, 68)


class TestStatusColor:
    def test_green_above_50(self):
        assert status_color(0.75) == Colors.STATUS_GREEN

    def test_yellow_between_25_and_50(self):
        assert status_color(0.35) == Colors.STATUS_YELLOW

    def test_red_below_25(self):
        assert status_color(0.1) == Colors.STATUS_RED

    def test_boundary_50_is_green(self):
        assert status_color(0.5) == Colors.STATUS_GREEN

    def test_boundary_25_is_yellow(self):
        assert status_color(0.25) == Colors.STATUS_YELLOW


class TestMoodGlowColor:
    def test_neutral_is_amber(self):
        r, g, b = mood_glow_color("neutral")
        assert (r, g, b) == (210, 140, 80)

    def test_hostile_is_red(self):
        r, g, b = mood_glow_color("hostile")
        assert (r, g, b) == (210, 80, 60)

    def test_content_is_gold(self):
        r, g, b = mood_glow_color("content")
        assert (r, g, b) == (230, 190, 80)

    def test_unknown_mood_defaults_to_amber(self):
        r, g, b = mood_glow_color("nonexistent")
        assert (r, g, b) == (210, 140, 80)


class TestSizes:
    def test_top_bar_height(self):
        assert Sizes.TOP_BAR_H == 32

    def test_sidebar_width(self):
        assert Sizes.SIDEBAR_W == 48

    def test_tile_size(self):
        assert Sizes.TILE == 24

    def test_chat_height(self):
        assert Sizes.CHAT_H == 130


class TestFonts:
    def test_init_creates_fonts(self):
        Fonts.init()
        assert Fonts.label is not None
        assert Fonts.body is not None
        assert Fonts.header is not None


class TestRenderSpacedText:
    def test_returns_surface(self):
        Fonts.init()
        surf = render_spaced_text("TEST", Fonts.label, (255, 255, 255), spacing=2)
        assert surf is not None
