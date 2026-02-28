"""Reusable Pygame widget library for settings UI.

Provides Button, Toggle, Slider, and Dropdown widgets with consistent
styling, hover states, and callback-driven interaction.

Each widget exposes: render(surface), handle_click(mx, my) -> bool,
handle_mouse_move(mx, my), handle_mouse_up().
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pygame

# ── Colors ───────────────────────────────────────────────────────────

_BG_COLOR = (25, 35, 55)
_BG_HOVER = (35, 50, 75)
_BG_ACTIVE = (45, 65, 95)
_BORDER_COLOR = (60, 80, 110)
_TEXT_COLOR = (200, 220, 240)
_TEXT_DIM = (130, 150, 170)
_ACCENT = (80, 160, 220)
_ACCENT_HOVER = (100, 180, 240)
_TOGGLE_ON = (60, 200, 100)
_TOGGLE_OFF = (100, 110, 130)
_SLIDER_TRACK = (40, 50, 70)
_SLIDER_FILL = (60, 140, 220)
_SLIDER_KNOB = (200, 220, 240)
_DROPDOWN_BG = (20, 30, 50)
_DROPDOWN_ITEM_HOVER = (40, 60, 90)
_SCROLLBAR_TRACK = (30, 40, 60)
_SCROLLBAR_THUMB = (80, 100, 130)

_FONT_SIZE = 14
_FONT: pygame.font.Font | None = None


def _ensure_font() -> pygame.font.Font:
    """Lazy-initialize and return the shared widget font."""
    global _FONT  # noqa: PLW0603
    if _FONT is None:
        try:
            _FONT = pygame.font.SysFont("consolas", _FONT_SIZE)
        except Exception:
            _FONT = pygame.font.Font(None, _FONT_SIZE)
    return _FONT


# ── Button ───────────────────────────────────────────────────────────


class Button:
    """Clickable button with label, hover highlight, and click callback.

    Attributes:
        rect: Bounding rectangle (x, y, w, h).
        label: Display text.
        enabled: Whether the button is clickable.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        on_click: Callable[[], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.on_click = on_click
        self.enabled = True
        self._hovered = False
        self._selected = False

    @property
    def selected(self) -> bool:
        """Whether this button is in selected/active state."""
        return self._selected

    @selected.setter
    def selected(self, value: bool) -> None:
        self._selected = value

    def render(self, surface: pygame.Surface) -> None:
        """Draw the button on the surface."""
        font = _ensure_font()

        if self._selected:
            bg = _BG_ACTIVE
            border = _ACCENT
        elif self._hovered and self.enabled:
            bg = _BG_HOVER
            border = _ACCENT_HOVER
        else:
            bg = _BG_COLOR
            border = _BORDER_COLOR

        pygame.draw.rect(surface, bg, self.rect)
        pygame.draw.rect(surface, border, self.rect, 1)

        text_color = _TEXT_COLOR if self.enabled else _TEXT_DIM
        text_surf = font.render(self.label, True, text_color)
        tx = self.rect.x + (self.rect.width - text_surf.get_width()) // 2
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle a click. Returns True if consumed."""
        if self.enabled and self.rect.collidepoint(mx, my):
            if self.on_click is not None:
                self.on_click()
            return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update hover state."""
        self._hovered = self.rect.collidepoint(mx, my)

    def handle_mouse_up(self) -> None:
        """No-op for buttons (click-only)."""


# ── Toggle ───────────────────────────────────────────────────────────


class Toggle:
    """On/off toggle switch with colored indicator.

    Attributes:
        rect: Bounding rectangle for the entire toggle area.
        value: Current on/off state.
        label: Display text shown next to the toggle.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        value: bool = False,
        on_change: Callable[[bool], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.value = value
        self.on_change = on_change
        self._hovered = False

    def render(self, surface: pygame.Surface) -> None:
        """Draw the toggle on the surface."""
        font = _ensure_font()

        # Label
        label_surf = font.render(self.label, True, _TEXT_COLOR)
        surface.blit(label_surf, (self.rect.x, self.rect.y + 2))

        # Toggle track (right side of rect)
        track_w = 36
        track_h = 18
        track_x = self.rect.x + self.rect.width - track_w - 4
        track_y = self.rect.y + (self.rect.height - track_h) // 2

        track_color = _TOGGLE_ON if self.value else _TOGGLE_OFF
        pygame.draw.rect(
            surface, track_color, (track_x, track_y, track_w, track_h), border_radius=9
        )

        # Knob
        knob_r = 7
        knob_x = track_x + track_w - knob_r - 3 if self.value else track_x + knob_r + 3
        knob_y = track_y + track_h // 2
        pygame.draw.circle(surface, _SLIDER_KNOB, (knob_x, knob_y), knob_r)

    def handle_click(self, mx: int, my: int) -> bool:
        """Toggle the value if clicked. Returns True if consumed."""
        if self.rect.collidepoint(mx, my):
            self.value = not self.value
            if self.on_change is not None:
                self.on_change(self.value)
            return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update hover state."""
        self._hovered = self.rect.collidepoint(mx, my)

    def handle_mouse_up(self) -> None:
        """No-op for toggles."""


# ── Slider ───────────────────────────────────────────────────────────


class Slider:
    """Draggable slider with track, fill, and knob.

    Attributes:
        rect: Bounding rectangle for the entire slider area.
        value: Current float value in [min_val, max_val].
        label: Display text shown above/beside the slider.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        value: float = 0.5,
        min_val: float = 0.0,
        max_val: float = 1.0,
        on_change: Callable[[float], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = max(min_val, min(max_val, value))
        self.on_change = on_change
        self._dragging = False
        self._hovered = False
        self._label_width = 100
        self._value_width = 45

    @property
    def _track_x(self) -> int:
        return self.rect.x + self._label_width

    @property
    def _track_w(self) -> int:
        return self.rect.width - self._label_width - self._value_width

    @property
    def _normalized(self) -> float:
        rng = self.max_val - self.min_val
        if rng <= 0:
            return 0.0
        return (self.value - self.min_val) / rng

    def render(self, surface: pygame.Surface) -> None:
        """Draw the slider on the surface."""
        font = _ensure_font()

        # Label
        label_surf = font.render(self.label, True, _TEXT_COLOR)
        label_y = self.rect.y + (self.rect.height - label_surf.get_height()) // 2
        surface.blit(label_surf, (self.rect.x, label_y))

        # Track
        track_h = 6
        track_y = self.rect.y + (self.rect.height - track_h) // 2
        track_x = self._track_x
        track_w = self._track_w

        pygame.draw.rect(surface, _SLIDER_TRACK, (track_x, track_y, track_w, track_h))

        # Fill
        fill_w = int(track_w * self._normalized)
        if fill_w > 0:
            pygame.draw.rect(surface, _SLIDER_FILL, (track_x, track_y, fill_w, track_h))

        # Knob
        knob_x = track_x + fill_w
        knob_y = self.rect.y + self.rect.height // 2
        knob_r = 7
        knob_color = _ACCENT_HOVER if self._dragging else _SLIDER_KNOB
        pygame.draw.circle(surface, knob_color, (knob_x, knob_y), knob_r)

        # Value text
        val_text = f"{self.value:.2f}"
        val_surf = font.render(val_text, True, _TEXT_DIM)
        val_x = track_x + track_w + 6
        val_y = self.rect.y + (self.rect.height - val_surf.get_height()) // 2
        surface.blit(val_surf, (val_x, val_y))

    def handle_click(self, mx: int, my: int) -> bool:
        """Start dragging if click is on the slider track area."""
        if self.rect.collidepoint(mx, my):
            track_x = self._track_x
            track_w = self._track_w
            if track_x <= mx <= track_x + track_w:
                self._dragging = True
                self._update_value_from_x(mx)
                return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update value while dragging."""
        self._hovered = self.rect.collidepoint(mx, my)
        if self._dragging:
            self._update_value_from_x(mx)

    def handle_mouse_up(self) -> None:
        """Stop dragging."""
        self._dragging = False

    def _update_value_from_x(self, mx: int) -> None:
        """Compute new value from mouse x position."""
        track_x = self._track_x
        track_w = self._track_w
        if track_w <= 0:
            return
        t = max(0.0, min(1.0, (mx - track_x) / track_w))
        self.value = self.min_val + t * (self.max_val - self.min_val)
        if self.on_change is not None:
            self.on_change(self.value)


# ── Dropdown ─────────────────────────────────────────────────────────


class Dropdown:
    """Collapsed/expanded dropdown menu with scrollable item list.

    Attributes:
        rect: Bounding rectangle for the collapsed dropdown.
        items: List of selectable string items.
        selected_index: Index of currently selected item (-1 for none).
        expanded: Whether the dropdown list is visible.
    """

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        label: str,
        items: list[str] | None = None,
        selected_index: int = 0,
        on_change: Callable[[int, str], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.items: list[str] = items or []
        self.selected_index = selected_index if items else -1
        self.on_change = on_change
        self.expanded = False
        self._hovered_index = -1
        self._max_visible = 6
        self._scroll_offset = 0
        self._label_width = 100

    @property
    def selected_value(self) -> str | None:
        """The currently selected item string, or None."""
        if 0 <= self.selected_index < len(self.items):
            return self.items[self.selected_index]
        return None

    def set_items(self, items: list[str], selected_index: int = 0) -> None:
        """Replace the item list and reset selection."""
        self.items = items
        self.selected_index = selected_index if items else -1
        self._scroll_offset = 0

    def render(self, surface: pygame.Surface) -> None:
        """Draw the dropdown (collapsed or expanded) on the surface."""
        font = _ensure_font()

        # Label
        label_surf = font.render(self.label, True, _TEXT_COLOR)
        label_y = self.rect.y + (self.rect.height - label_surf.get_height()) // 2
        surface.blit(label_surf, (self.rect.x, label_y))

        # Collapsed box
        box_x = self.rect.x + self._label_width
        box_w = self.rect.width - self._label_width
        box_rect = pygame.Rect(box_x, self.rect.y, box_w, self.rect.height)

        bg = _BG_HOVER if self.expanded else _BG_COLOR
        pygame.draw.rect(surface, bg, box_rect)
        pygame.draw.rect(surface, _BORDER_COLOR, box_rect, 1)

        # Selected text
        display = self.selected_value or "—"
        text_surf = font.render(display, True, _TEXT_COLOR)
        tx = box_x + 6
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

        # Arrow indicator
        arrow = "v" if not self.expanded else "^"
        arrow_surf = font.render(arrow, True, _TEXT_DIM)
        ax = box_x + box_w - arrow_surf.get_width() - 6
        surface.blit(arrow_surf, (ax, ty))

        # Expanded item list
        if self.expanded and self.items:
            self._render_items(surface, box_x, box_w)

    def _render_items(self, surface: pygame.Surface, box_x: int, box_w: int) -> None:
        """Render the expanded dropdown item list."""
        font = _ensure_font()
        item_h = self.rect.height
        visible_count = min(len(self.items), self._max_visible)
        list_h = visible_count * item_h
        list_y = self.rect.y + self.rect.height
        scrollable = len(self.items) > self._max_visible
        scrollbar_w = 6 if scrollable else 0
        text_area_w = box_w - scrollbar_w

        # Background
        pygame.draw.rect(surface, _DROPDOWN_BG, (box_x, list_y, box_w, list_h))
        pygame.draw.rect(surface, _BORDER_COLOR, (box_x, list_y, box_w, list_h), 1)

        for i in range(visible_count):
            idx = i + self._scroll_offset
            if idx >= len(self.items):
                break

            iy = list_y + i * item_h

            # Hover highlight
            if idx == self._hovered_index:
                pygame.draw.rect(
                    surface, _DROPDOWN_ITEM_HOVER, (box_x + 1, iy, text_area_w - 2, item_h)
                )

            # Selection indicator
            if idx == self.selected_index:
                pygame.draw.rect(surface, _ACCENT, (box_x + 1, iy, 3, item_h))

            text_surf = font.render(self.items[idx], True, _TEXT_COLOR)
            surface.blit(text_surf, (box_x + 8, iy + (item_h - text_surf.get_height()) // 2))

        # Scrollbar
        if scrollable and list_h > 0:
            sb_x = box_x + box_w - scrollbar_w
            # Track
            pygame.draw.rect(surface, _SCROLLBAR_TRACK, (sb_x, list_y, scrollbar_w, list_h))
            # Thumb
            max_offset = len(self.items) - self._max_visible
            thumb_h = max(8, int(list_h * self._max_visible / len(self.items)))
            thumb_travel = list_h - thumb_h
            ratio = self._scroll_offset / max_offset if max_offset > 0 else 0
            thumb_y = list_y + int(thumb_travel * ratio)
            pygame.draw.rect(surface, _SCROLLBAR_THUMB, (sb_x, thumb_y, scrollbar_w, thumb_h))

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle click: toggle expanded, or select item."""
        box_x = self.rect.x + self._label_width
        box_w = self.rect.width - self._label_width

        if self.expanded and self.items:
            # Check if click is on an item
            item_h = self.rect.height
            list_y = self.rect.y + self.rect.height
            visible_count = min(len(self.items), self._max_visible)
            list_bottom = list_y + visible_count * item_h

            if box_x <= mx <= box_x + box_w and list_y <= my <= list_bottom:
                clicked_i = (my - list_y) // item_h
                idx = clicked_i + self._scroll_offset
                if 0 <= idx < len(self.items):
                    self.selected_index = idx
                    self.expanded = False
                    if self.on_change is not None:
                        self.on_change(idx, self.items[idx])
                    return True

        # Check if click is on the collapsed box
        box_rect = pygame.Rect(box_x, self.rect.y, box_w, self.rect.height)
        if box_rect.collidepoint(mx, my):
            self.expanded = not self.expanded
            return True

        # Click outside — close
        if self.expanded:
            self.expanded = False
            return True

        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update hovered item index."""
        if not self.expanded:
            self._hovered_index = -1
            return

        box_x = self.rect.x + self._label_width
        box_w = self.rect.width - self._label_width
        item_h = self.rect.height
        list_y = self.rect.y + self.rect.height

        if box_x <= mx <= box_x + box_w and my >= list_y:
            i = (my - list_y) // item_h + self._scroll_offset
            if 0 <= i < len(self.items):
                self._hovered_index = i
                return

        self._hovered_index = -1

    def handle_scroll(self, direction: int) -> bool:
        """Scroll the dropdown list. Returns True if consumed.

        Args:
            direction: Positive = scroll up (show earlier items),
                       negative = scroll down (show later items).
                       Matches pygame MOUSEWHEEL event.y sign.
        """
        if not self.expanded or len(self.items) <= self._max_visible:
            return False
        max_offset = len(self.items) - self._max_visible
        if direction > 0:
            self._scroll_offset = max(0, self._scroll_offset - 1)
        elif direction < 0:
            self._scroll_offset = min(max_offset, self._scroll_offset + 1)
        return True

    def handle_mouse_up(self) -> None:
        """No-op for dropdown."""
