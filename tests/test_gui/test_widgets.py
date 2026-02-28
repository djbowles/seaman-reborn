"""Tests for the reusable Pygame widget library (gui/widgets.py).

Pygame is mocked at module level to avoid requiring a display server in CI.
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
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.line.return_value = None

# Rect mock — return a real-ish object with collidepoint
_real_rects: list = []


def _make_rect(x, y, w, h):
    r = MagicMock()
    r.x = x
    r.y = y
    r.width = w
    r.height = h
    r.collidepoint = lambda mx, my: x <= mx <= x + w and y <= my <= y + h
    _real_rects.append(r)
    return r


_pygame_mock.Rect = _make_rect

# Surface constructor mock
_pygame_mock.Surface.return_value = _surface_mock

# Install pygame mock before importing gui modules
sys.modules["pygame"] = _pygame_mock

from seaman_brain.gui.widgets import Button, Dropdown, Slider, Toggle  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset mocks and re-install pygame mock between tests."""
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.widgets as widgets_mod
    widgets_mod.pygame = _pygame_mock
    widgets_mod._FONT = None  # Reset lazy font
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    _real_rects.clear()
    # Restore return values
    _pygame_mock.Rect = _make_rect
    _pygame_mock.Surface.return_value = _surface_mock
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.render.return_value = _text_surf_mock
    _font_mock.get_linesize.return_value = 16
    _font_mock.size.return_value = (80, 16)
    _text_surf_mock.get_width.return_value = 80
    _text_surf_mock.get_height.return_value = 16


# ── Button Tests ──────────────────────────────────────────────────────


class TestButton:
    """Tests for the Button widget."""

    def test_create_button(self):
        """Button stores rect, label, and callback."""
        cb = MagicMock()
        btn = Button(10, 20, 100, 30, "Click Me", on_click=cb)
        assert btn.label == "Click Me"
        assert btn.enabled is True
        assert btn.on_click is cb

    def test_render_draws_rect_and_text(self):
        """Button render draws background, border, and label."""
        btn = Button(10, 20, 100, 30, "OK")
        btn.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count >= 2  # bg + border
        assert _font_mock.render.call_count >= 1

    def test_click_inside_calls_callback(self):
        """Clicking inside the button triggers the callback."""
        cb = MagicMock()
        btn = Button(10, 20, 100, 30, "OK", on_click=cb)
        result = btn.handle_click(50, 35)
        assert result is True
        cb.assert_called_once()

    def test_click_outside_does_nothing(self):
        """Clicking outside the button does not trigger callback."""
        cb = MagicMock()
        btn = Button(10, 20, 100, 30, "OK", on_click=cb)
        result = btn.handle_click(200, 200)
        assert result is False
        cb.assert_not_called()

    def test_disabled_button_ignores_click(self):
        """Disabled buttons do not trigger callbacks."""
        cb = MagicMock()
        btn = Button(10, 20, 100, 30, "OK", on_click=cb)
        btn.enabled = False
        result = btn.handle_click(50, 35)
        assert result is False
        cb.assert_not_called()

    def test_hover_state(self):
        """Mouse move updates hover state."""
        btn = Button(10, 20, 100, 30, "OK")
        assert btn._hovered is False
        btn.handle_mouse_move(50, 35)
        assert btn._hovered is True
        btn.handle_mouse_move(200, 200)
        assert btn._hovered is False

    def test_selected_state(self):
        """Button selected state can be toggled."""
        btn = Button(10, 20, 100, 30, "OK")
        assert btn.selected is False
        btn.selected = True
        assert btn.selected is True

    def test_click_without_callback(self):
        """Clicking a button with no callback returns True but doesn't crash."""
        btn = Button(10, 20, 100, 30, "OK", on_click=None)
        result = btn.handle_click(50, 35)
        assert result is True

    def test_mouse_up_is_noop(self):
        """handle_mouse_up does nothing for buttons."""
        btn = Button(10, 20, 100, 30, "OK")
        btn.handle_mouse_up()  # Should not raise


# ── Toggle Tests ──────────────────────────────────────────────────────


class TestToggle:
    """Tests for the Toggle widget."""

    def test_create_toggle(self):
        """Toggle stores label, value, and callback."""
        cb = MagicMock()
        tog = Toggle(10, 20, 200, 24, "Sound", value=True, on_change=cb)
        assert tog.label == "Sound"
        assert tog.value is True

    def test_toggle_default_off(self):
        """Toggle defaults to off."""
        tog = Toggle(10, 20, 200, 24, "Mute")
        assert tog.value is False

    def test_render_draws_elements(self):
        """Toggle render draws label, track, and knob."""
        tog = Toggle(10, 20, 200, 24, "TTS")
        tog.render(_surface_mock)
        assert _font_mock.render.call_count >= 1  # label
        assert _pygame_mock.draw.rect.call_count >= 1  # track
        assert _pygame_mock.draw.circle.call_count >= 1  # knob

    def test_click_toggles_value(self):
        """Clicking toggles the value and fires callback."""
        cb = MagicMock()
        tog = Toggle(10, 20, 200, 24, "TTS", value=False, on_change=cb)
        result = tog.handle_click(100, 32)
        assert result is True
        assert tog.value is True
        cb.assert_called_once_with(True)

    def test_click_toggles_back(self):
        """Clicking twice returns to original value."""
        tog = Toggle(10, 20, 200, 24, "TTS", value=False)
        tog.handle_click(100, 32)
        assert tog.value is True
        tog.handle_click(100, 32)
        assert tog.value is False

    def test_click_outside_does_nothing(self):
        """Clicking outside the toggle area does nothing."""
        cb = MagicMock()
        tog = Toggle(10, 20, 200, 24, "TTS", on_change=cb)
        result = tog.handle_click(300, 300)
        assert result is False
        cb.assert_not_called()

    def test_toggle_without_callback(self):
        """Toggle works without a callback."""
        tog = Toggle(10, 20, 200, 24, "TTS")
        tog.handle_click(100, 32)
        assert tog.value is True

    def test_mouse_up_is_noop(self):
        """handle_mouse_up does nothing for toggles."""
        tog = Toggle(10, 20, 200, 24, "TTS")
        tog.handle_mouse_up()  # Should not raise


# ── Slider Tests ──────────────────────────────────────────────────────


class TestSlider:
    """Tests for the Slider widget."""

    def test_create_slider(self):
        """Slider stores label, range, and value."""
        s = Slider(10, 20, 300, 24, "Volume", value=0.5, min_val=0.0, max_val=1.0)
        assert s.label == "Volume"
        assert s.value == pytest.approx(0.5)
        assert s.min_val == 0.0
        assert s.max_val == 1.0

    def test_value_clamped_to_range(self):
        """Slider clamps initial value to [min, max]."""
        s = Slider(10, 20, 300, 24, "Vol", value=2.0, min_val=0.0, max_val=1.0)
        assert s.value == pytest.approx(1.0)
        s2 = Slider(10, 20, 300, 24, "Vol", value=-1.0, min_val=0.0, max_val=1.0)
        assert s2.value == pytest.approx(0.0)

    def test_render_draws_elements(self):
        """Slider render draws label, track, fill, knob, and value text."""
        s = Slider(10, 20, 300, 24, "Volume", value=0.5)
        s.render(_surface_mock)
        assert _font_mock.render.call_count >= 2  # label + value
        assert _pygame_mock.draw.rect.call_count >= 1  # track
        assert _pygame_mock.draw.circle.call_count >= 1  # knob

    def test_click_starts_dragging(self):
        """Clicking on slider track starts drag mode."""
        s = Slider(10, 20, 300, 24, "Volume", value=0.5)
        # Click in the track area (after label_width=100)
        result = s.handle_click(160, 32)
        assert result is True
        assert s._dragging is True

    def test_click_outside_does_nothing(self):
        """Clicking outside slider does nothing."""
        s = Slider(10, 20, 300, 24, "Volume")
        result = s.handle_click(500, 500)
        assert result is False
        assert s._dragging is False

    def test_drag_updates_value(self):
        """Dragging the slider updates the value."""
        cb = MagicMock()
        s = Slider(10, 20, 300, 24, "Volume", value=0.0, on_change=cb)
        # Start drag at track midpoint
        track_x = s._track_x
        track_w = s._track_w
        mid_x = track_x + track_w // 2
        s.handle_click(mid_x, 32)
        assert s._dragging is True
        assert s.value == pytest.approx(0.5, abs=0.05)
        cb.assert_called()

    def test_mouse_up_stops_drag(self):
        """Releasing mouse stops drag mode."""
        s = Slider(10, 20, 300, 24, "Volume")
        s._dragging = True
        s.handle_mouse_up()
        assert s._dragging is False

    def test_mouse_move_while_dragging(self):
        """Mouse move during drag updates value."""
        s = Slider(10, 20, 300, 24, "Volume", value=0.0)
        s._dragging = True
        track_x = s._track_x
        track_w = s._track_w
        # Move to 75% of track
        s.handle_mouse_move(track_x + int(track_w * 0.75), 32)
        assert s.value == pytest.approx(0.75, abs=0.05)

    def test_mouse_move_no_drag(self):
        """Mouse move without drag only updates hover."""
        s = Slider(10, 20, 300, 24, "Volume", value=0.5)
        old_val = s.value
        s.handle_mouse_move(200, 32)
        assert s.value == pytest.approx(old_val)

    def test_normalized_property(self):
        """_normalized returns 0-1 proportion."""
        s = Slider(10, 20, 300, 24, "Temp", value=1.0, min_val=0.0, max_val=2.0)
        assert s._normalized == pytest.approx(0.5)

    def test_custom_range(self):
        """Slider works with non-0-1 ranges."""
        s = Slider(10, 20, 300, 24, "Temp", value=1.0, min_val=0.0, max_val=2.0)
        assert s.value == pytest.approx(1.0)


# ── Dropdown Tests ────────────────────────────────────────────────────


class TestDropdown:
    """Tests for the Dropdown widget."""

    def test_create_dropdown(self):
        """Dropdown stores label, items, and selection."""
        items = ["Alpha", "Beta", "Gamma"]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items, selected_index=1)
        assert dd.label == "Model"
        assert dd.items == items
        assert dd.selected_index == 1
        assert dd.selected_value == "Beta"

    def test_empty_dropdown(self):
        """Dropdown with no items has selected_index -1."""
        dd = Dropdown(10, 20, 300, 24, "Model")
        assert dd.selected_index == -1
        assert dd.selected_value is None

    def test_render_collapsed(self):
        """Collapsed dropdown renders box and selected text."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B"])
        dd.render(_surface_mock)
        assert _font_mock.render.call_count >= 2  # label + selected + arrow

    def test_render_expanded(self):
        """Expanded dropdown renders item list."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B", "C"])
        dd.expanded = True
        dd.render(_surface_mock)
        # Should render more calls for items
        assert _font_mock.render.call_count >= 5  # label + selected + arrow + 3 items

    def test_click_box_toggles_expanded(self):
        """Clicking the dropdown box toggles expanded state."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B"])
        assert dd.expanded is False
        # Click inside the box area (after label_width=100)
        dd.handle_click(150, 32)
        assert dd.expanded is True
        dd.handle_click(150, 32)
        assert dd.expanded is False

    def test_click_item_selects_it(self):
        """Clicking an item in expanded dropdown selects it."""
        cb = MagicMock()
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B", "C"], on_change=cb)
        dd.expanded = True
        # Items start at y=44 (20+24), each is 24px tall
        # Click on second item (index 1): y = 44 + 24 = 68 (middle of second item)
        dd.handle_click(150, 56)
        assert dd.selected_index == 0  # First item at y=44-67
        assert dd.expanded is False
        cb.assert_called_once_with(0, "A")

    def test_click_outside_closes(self):
        """Clicking outside expanded dropdown closes it."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B"])
        dd.expanded = True
        dd.handle_click(500, 500)
        assert dd.expanded is False

    def test_set_items_replaces_list(self):
        """set_items replaces the item list and resets selection."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B"])
        dd.set_items(["X", "Y", "Z"], selected_index=2)
        assert dd.items == ["X", "Y", "Z"]
        assert dd.selected_index == 2
        assert dd.selected_value == "Z"

    def test_set_items_empty_list(self):
        """set_items with empty list resets to -1."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A"])
        dd.set_items([])
        assert dd.selected_index == -1
        assert dd.selected_value is None

    def test_hover_item_tracking(self):
        """Mouse move updates hovered item index when expanded."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B", "C"])
        dd.expanded = True
        # Hover over first item area
        dd.handle_mouse_move(150, 48)
        assert dd._hovered_index == 0

    def test_hover_no_tracking_when_collapsed(self):
        """Mouse move doesn't track items when collapsed."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B"])
        dd.expanded = False
        dd.handle_mouse_move(150, 48)
        assert dd._hovered_index == -1

    def test_mouse_up_is_noop(self):
        """handle_mouse_up does nothing for dropdown."""
        dd = Dropdown(10, 20, 300, 24, "Model")
        dd.handle_mouse_up()  # Should not raise

    def test_click_outside_collapsed_does_nothing(self):
        """Click outside a collapsed dropdown returns False."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A"])
        dd.expanded = False
        result = dd.handle_click(500, 500)
        assert result is False

    def test_scroll_down_advances_offset(self):
        """Scrolling down in an expanded dropdown with overflow advances scroll offset."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        assert dd._scroll_offset == 0
        consumed = dd.handle_scroll(-1)  # scroll down
        assert consumed is True
        assert dd._scroll_offset == 1

    def test_scroll_up_retreats_offset(self):
        """Scrolling up decreases scroll offset (clamped at 0)."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        dd._scroll_offset = 3
        dd.handle_scroll(1)  # scroll up
        assert dd._scroll_offset == 2

    def test_scroll_clamped_at_zero(self):
        """Cannot scroll above the first item."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        dd._scroll_offset = 0
        dd.handle_scroll(1)  # scroll up
        assert dd._scroll_offset == 0

    def test_scroll_clamped_at_max(self):
        """Cannot scroll past the last page of items."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        max_offset = len(items) - dd._max_visible  # 10 - 6 = 4
        dd._scroll_offset = max_offset
        dd.handle_scroll(-1)  # scroll down
        assert dd._scroll_offset == max_offset

    def test_scroll_ignored_when_collapsed(self):
        """Scroll events are ignored when dropdown is collapsed."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = False
        consumed = dd.handle_scroll(-1)
        assert consumed is False
        assert dd._scroll_offset == 0

    def test_scroll_ignored_when_few_items(self):
        """Scroll events are ignored when all items fit without scrolling."""
        dd = Dropdown(10, 20, 300, 24, "Model", items=["A", "B", "C"])
        dd.expanded = True
        consumed = dd.handle_scroll(-1)
        assert consumed is False

    def test_click_item_with_scroll_offset(self):
        """Clicking an item accounts for scroll offset."""
        cb = MagicMock()
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items, on_change=cb)
        dd.expanded = True
        dd._scroll_offset = 3
        # Click first visible row (y=44, which is list_y=44, item 0 in view = index 3)
        dd.handle_click(150, 56)
        assert dd.selected_index == 3
        cb.assert_called_once_with(3, "Item 3")

    def test_render_scrollbar_when_overflow(self):
        """Expanded dropdown with >6 items renders scrollbar rects."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        dd.render(_surface_mock)
        # Should have extra draw.rect calls for scrollbar track + thumb
        rect_calls = _pygame_mock.draw.rect.call_count
        # Minimum: bg + border (collapsed) + bg + border (list) + scrollbar track + thumb = 6
        assert rect_calls >= 6

    def test_scrollbar_click_does_not_close(self):
        """Clicking the scrollbar area scrolls without closing the dropdown."""
        items = [f"Item {i}" for i in range(10)]
        # Dropdown at x=10, w=300, label_width=100 → box_x=110, box_w=200
        # Scrollbar is rightmost 6px: x=304..310
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        dd._scroll_offset = 0
        # Click on scrollbar area (x=305), below the thumb
        # list_y=44, list_h=144, thumb_h=86 (6/10 ratio), thumb starts at y=44
        # so thumb_bottom=130, click at 160 is below thumb
        result = dd.handle_click(305, 160)  # below thumb → page down
        assert result is True
        assert dd.expanded is True  # stays open!
        assert dd._scroll_offset > 0  # scrolled

    def test_scrollbar_click_above_thumb_pages_up(self):
        """Clicking above the scrollbar thumb pages up."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        dd._scroll_offset = 4  # scrolled to bottom
        # Thumb is near bottom; click near top of scrollbar (y=45)
        dd.handle_click(305, 45)
        assert dd.expanded is True
        assert dd._scroll_offset < 4

    def test_scrollbar_click_below_thumb_pages_down(self):
        """Clicking below the scrollbar thumb pages down."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        dd._scroll_offset = 0
        # Click near bottom of scrollbar (last item row)
        list_y = 20 + 24  # 44
        list_bottom = list_y + 6 * 24  # 188
        dd.handle_click(305, list_bottom - 5)
        assert dd.expanded is True
        assert dd._scroll_offset > 0

    def test_hover_ignores_scrollbar_zone(self):
        """Mouse hovering over scrollbar area does not highlight items."""
        items = [f"Item {i}" for i in range(10)]
        dd = Dropdown(10, 20, 300, 24, "Model", items=items)
        dd.expanded = True
        # Hover over scrollbar zone (x=305)
        dd.handle_mouse_move(305, 56)
        assert dd._hovered_index == -1
