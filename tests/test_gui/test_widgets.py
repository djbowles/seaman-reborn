"""Tests for Modern Minimal widgets."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 60
_font_surface.get_height.return_value = 14
_font_mock.render.return_value = _font_surface
_font_mock.size.return_value = (60, 14)
_font_mock.get_linesize.return_value = 16
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
_surface_mock = MagicMock()
_pygame_mock.Surface.return_value = _surface_mock
_pygame_mock.Rect = lambda x, y, w, h: type(
    "Rect", (), {
        "x": x, "y": y, "w": w, "h": h, "width": w, "height": h,
        "left": x, "top": y, "right": x + w, "bottom": y + h,
        "collidepoint": lambda self, px, py: (x <= px < x + w and y <= py < y + h),
    }
)()
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.widgets import (  # noqa: E402
    Button,
    Dropdown,
    Slider,
    Toggle,
)


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.widgets as mod
    mod.pygame = _pygame_mock
    _pygame_mock.Rect = lambda x, y, w, h: type(
        "Rect", (), {
            "x": x, "y": y, "w": w, "h": h, "width": w, "height": h,
            "left": x, "top": y, "right": x + w, "bottom": y + h,
            "collidepoint": lambda self, px, py: (
                x <= px < x + w and y <= py < y + h
            ),
        }
    )()
    yield


# ── Button Tests ─────────────────────────────────────────────────────


class TestButton:
    def test_construction(self):
        b = Button(10, 10, 100, 30, "Test")
        assert b.label == "Test"

    def test_click_inside_fires_callback(self):
        cb = MagicMock()
        b = Button(10, 10, 100, 30, "Test", on_click=cb)
        assert b.handle_click(50, 25) is True
        cb.assert_called_once()

    def test_click_outside_returns_false(self):
        cb = MagicMock()
        b = Button(10, 10, 100, 30, "Test", on_click=cb)
        assert b.handle_click(200, 200) is False
        cb.assert_not_called()

    def test_disabled_blocks_click(self):
        cb = MagicMock()
        b = Button(10, 10, 100, 30, "Test", on_click=cb)
        b.enabled = False
        assert b.handle_click(50, 25) is False

    def test_render_does_not_crash(self):
        b = Button(10, 10, 100, 30, "Test")
        b.render(_surface_mock)

    def test_hover_state(self):
        b = Button(10, 10, 100, 30, "Test")
        b.handle_mouse_move(50, 25)
        assert b._hovered is True
        b.handle_mouse_move(200, 200)
        assert b._hovered is False


# ── Toggle Tests ─────────────────────────────────────────────────────


class TestToggle:
    def test_initial_state(self):
        t = Toggle(10, 10, 50, 24, on=False)
        assert t.on is False

    def test_click_toggles(self):
        t = Toggle(10, 10, 50, 24, on=False)
        t.handle_click(30, 22)
        assert t.on is True

    def test_click_toggles_back(self):
        t = Toggle(10, 10, 50, 24, on=True)
        t.handle_click(30, 22)
        assert t.on is False

    def test_callback_fires(self):
        cb = MagicMock()
        t = Toggle(10, 10, 50, 24, on=False, on_change=cb)
        t.handle_click(30, 22)
        cb.assert_called_once_with(True)

    def test_click_outside_returns_false(self):
        t = Toggle(10, 10, 50, 24, on=False)
        assert t.handle_click(200, 200) is False
        assert t.on is False

    def test_render_does_not_crash(self):
        t = Toggle(10, 10, 50, 24, on=True)
        t.render(_surface_mock)


# ── Slider Tests ─────────────────────────────────────────────────────


class TestSlider:
    def test_initial_value(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50)
        assert s.value == 50

    def test_drag_changes_value(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50)
        s.handle_click(110, 20)  # middle of slider
        assert 40 <= s.value <= 60

    def test_callback_fires_on_drag(self):
        cb = MagicMock()
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50, on_change=cb)
        s.handle_click(110, 20)
        assert cb.called

    def test_clamps_to_range(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=150)
        assert s.value == 100

    def test_render_does_not_crash(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50)
        s.render(_surface_mock)

    def test_mouse_up_stops_dragging(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50)
        s.handle_click(110, 20)
        assert s._dragging is True
        s.handle_mouse_up()
        assert s._dragging is False


# ── Dropdown Tests ───────────────────────────────────────────────────


class TestDropdown:
    def test_construction(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0)
        assert d.selected_text == "A"

    def test_click_expands(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0)
        d.handle_click(80, 24)
        assert d.expanded is True

    def test_select_item(self):
        cb = MagicMock()
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0, on_change=cb)
        d.handle_click(80, 24)  # expand
        d.handle_click(80, 42)  # click first item in list (y=28+14=42)
        cb.assert_called()

    def test_click_outside_closes(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0)
        d.handle_click(80, 24)  # expand
        assert d.expanded is True
        d.handle_click(500, 500)  # click outside
        assert d.expanded is False

    def test_selected_text_empty_items(self):
        d = Dropdown(10, 10, 150, 28, items=[], selected=0)
        assert d.selected_text is None

    def test_set_items(self):
        d = Dropdown(10, 10, 150, 28, items=["A"], selected=0)
        d.set_items(["X", "Y", "Z"], selected=1)
        assert d.selected_text == "Y"

    def test_render_collapsed_does_not_crash(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B"], selected=0)
        d.render(_surface_mock)

    def test_render_expanded_does_not_crash(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B"], selected=0)
        d.expanded = True
        d.render(_surface_mock)

    def test_scroll_handling(self):
        items = [f"Item {i}" for i in range(20)]
        d = Dropdown(10, 10, 150, 28, items=items, selected=0)
        d.expanded = True
        assert d.handle_scroll(-1) is True
        assert d._scroll_offset == 1
