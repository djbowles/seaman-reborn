"""Tests for the ActionBar right-side panel.

Pygame is mocked at module level to avoid requiring a display server.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
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

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: MagicMock(x=x, y=y, width=w, height=h)


# Surface constructor mock
def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 100
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 100
    return s


_pygame_mock.Surface = _make_surface

# Install pygame mock before importing gui modules
sys.modules["pygame"] = _pygame_mock

from seaman_brain.gui.action_bar import ActionBar, ActionButton  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset mocks and re-install pygame between tests."""
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.action_bar as ab_mod
    ab_mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    _pygame_mock.Surface = _make_surface
    _pygame_mock.Rect = lambda x, y, w, h: MagicMock(x=x, y=y, width=w, height=h)
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.render.return_value = _text_surf_mock
    _font_mock.get_linesize.return_value = 16
    _font_mock.size.return_value = (80, 16)
    _text_surf_mock.get_width.return_value = 80
    _text_surf_mock.get_height.return_value = 16


# ── ActionButton Tests ───────────────────────────────────────────────


class TestActionButton:
    """Tests for ActionButton dataclass."""

    def test_button_contains_point_inside(self):
        """contains() returns True for point inside button."""
        btn = ActionButton(key="feed", icon="F", label="Feed",
                           x=100, y=50, width=140, height=40)
        assert btn.contains(120, 70) is True

    def test_button_contains_point_outside(self):
        """contains() returns False for point outside button."""
        btn = ActionButton(key="feed", icon="F", label="Feed",
                           x=100, y=50, width=140, height=40)
        assert btn.contains(50, 30) is False

    def test_button_contains_edge(self):
        """contains() returns True for point on button edge."""
        btn = ActionButton(key="feed", icon="F", label="Feed",
                           x=100, y=50, width=140, height=40)
        assert btn.contains(100, 50) is True
        assert btn.contains(240, 90) is True

    def test_button_default_hover_false(self):
        """Button starts with hover=False."""
        btn = ActionButton(key="clean", icon="*", label="Clean")
        assert btn.hover is False


# ── ActionBar Construction Tests ─────────────────────────────────────


class TestActionBarConstruction:
    """Tests for ActionBar initialization."""

    def test_creates_with_no_callback(self):
        """ActionBar creates successfully without callback."""
        bar = ActionBar()
        assert bar.buttons == []

    def test_creates_with_callback(self):
        """ActionBar stores the on_action callback."""
        cb = MagicMock()
        bar = ActionBar(on_action=cb)
        assert bar._on_action is cb

    def test_set_panel_area_builds_buttons(self):
        """set_panel_area() creates 7 buttons."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        assert len(bar.buttons) == 7

    def test_button_keys(self):
        """Buttons have correct action keys."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        keys = [b.key for b in bar.buttons]
        assert keys == [
            "feed", "aerate", "temp_up", "temp_down", "clean", "drain", "tap_glass",
        ]

    def test_buttons_positioned_within_panel(self):
        """All buttons are positioned within the panel area."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        for btn in bar.buttons:
            assert btn.x >= 864
            assert btn.x + btn.width <= 864 + 160
            assert btn.y >= 45


# ── Click Handling Tests ─────────────────────────────────────────────


class TestActionBarClick:
    """Tests for ActionBar click handling."""

    def test_click_on_button_invokes_callback(self):
        """Clicking a button invokes on_action with the correct key."""
        cb = MagicMock()
        bar = ActionBar(on_action=cb)
        bar.set_panel_area(864, 45, 160, 723)
        # Click on first button (Feed)
        btn = bar.buttons[0]
        result = bar.handle_click(btn.x + 5, btn.y + 5)
        assert result is True
        cb.assert_called_once_with("feed")

    def test_click_outside_buttons_returns_false(self):
        """Clicking outside buttons returns False."""
        cb = MagicMock()
        bar = ActionBar(on_action=cb)
        bar.set_panel_area(864, 45, 160, 723)
        result = bar.handle_click(10, 10)
        assert result is False
        cb.assert_not_called()

    def test_click_each_button(self):
        """Each button invokes callback with its key."""
        cb = MagicMock()
        bar = ActionBar(on_action=cb)
        bar.set_panel_area(864, 45, 160, 723)
        expected_keys = [
            "feed", "aerate", "temp_up", "temp_down", "clean", "drain", "tap_glass",
        ]
        for btn, expected_key in zip(bar.buttons, expected_keys):
            cb.reset_mock()
            bar.handle_click(btn.x + 5, btn.y + 5)
            cb.assert_called_once_with(expected_key)

    def test_click_without_callback_no_error(self):
        """Clicking without a callback doesn't raise."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        btn = bar.buttons[0]
        result = bar.handle_click(btn.x + 5, btn.y + 5)
        assert result is True


# ── Hover Tests ──────────────────────────────────────────────────────


class TestActionBarHover:
    """Tests for ActionBar hover state."""

    def test_hover_sets_button_hover(self):
        """Moving mouse over a button sets its hover flag."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        btn = bar.buttons[0]
        bar.handle_mouse_move(btn.x + 5, btn.y + 5)
        assert btn.hover is True

    def test_hover_clears_other_buttons(self):
        """Moving mouse sets only the hovered button."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        # Hover first button
        btn0 = bar.buttons[0]
        btn1 = bar.buttons[1]
        bar.handle_mouse_move(btn0.x + 5, btn0.y + 5)
        assert btn0.hover is True
        assert btn1.hover is False

    def test_hover_outside_clears_all(self):
        """Moving mouse outside all buttons clears hover flags."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.buttons[0].hover = True
        bar.handle_mouse_move(10, 10)
        assert all(not b.hover for b in bar.buttons)


# ── Render Tests ─────────────────────────────────────────────────────


class TestActionBarRender:
    """Tests for ActionBar rendering."""

    def test_render_without_crash(self):
        """Render runs without error."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.render(_surface_mock)

    def test_render_draws_panel_bg(self):
        """Render blits the panel background."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.render(_surface_mock)
        assert _surface_mock.blit.called

    def test_render_draws_header_text(self):
        """Render draws 'Actions' header."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.render(_surface_mock)
        rendered_texts = [call.args[0] for call in _font_mock.render.call_args_list]
        assert any("Actions" in str(t) for t in rendered_texts)

    def test_render_draws_button_rects(self):
        """Render draws rect for each button."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.render(_surface_mock)
        # At least 12 rect calls (bg + border per button = 12, plus header bg)
        assert _pygame_mock.draw.rect.call_count >= 12

    def test_render_draws_button_labels(self):
        """Render draws label text for each button."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.render(_surface_mock)
        # Should render icon + label for each button (+ header)
        assert _font_mock.render.call_count >= 12  # 6 icons + 6 labels

    def test_render_with_hover(self):
        """Render with hovered button doesn't crash."""
        bar = ActionBar()
        bar.set_panel_area(864, 45, 160, 723)
        bar.buttons[0].hover = True
        bar.render(_surface_mock)
