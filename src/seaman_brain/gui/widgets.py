"""Modern Minimal widget library — Button, Toggle, Slider, Dropdown.

All colors from theme.py. No pygame_gui dependency. Pure pygame.draw calls.
Each widget: render(surface), handle_click(mx, my) -> bool,
handle_mouse_move(mx, my), handle_mouse_up().
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pygame

from seaman_brain.gui.theme import Colors, Fonts

# ── Shared Font ──────────────────────────────────────────────────────

_FONT: pygame.font.Font | None = None


def _ensure_font() -> pygame.font.Font:
    """Lazy-initialize and return the shared widget font."""
    global _FONT  # noqa: PLW0603
    if _FONT is None:
        if Fonts.body is not None:
            _FONT = Fonts.body
        else:
            for name in ("consolas", "couriernew", "courier"):
                try:
                    _FONT = pygame.font.SysFont(name, 11)
                    return _FONT
                except Exception:
                    continue
            _FONT = pygame.font.Font(None, 11)
    return _FONT


# ── Styling Constants ────────────────────────────────────────────────

_BG = Colors.SURFACE_3[:3]
_BG_HOVER = Colors.SURFACE_5[:3]
_BORDER = Colors.BORDER[:3]
_TEXT = Colors.TEXT_90
_TEXT_DIM = Colors.TEXT_50
_ACCENT = Colors.ACCENT
_TOGGLE_ON = Colors.STATUS_GREEN
_TOGGLE_OFF = Colors.TEXT_30
_SLIDER_TRACK = Colors.SURFACE_3[:3]
_SLIDER_FILL = Colors.ACCENT
_SLIDER_KNOB = Colors.TEXT_90
_DROPDOWN_BG = (12, 12, 18)
_DROPDOWN_HOVER = Colors.SURFACE_5[:3]


# ── Button ───────────────────────────────────────────────────────────


class Button:
    """Clickable button with label and callback."""

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
        return self._selected

    @selected.setter
    def selected(self, value: bool) -> None:
        self._selected = value

    def render(self, surface: pygame.Surface) -> None:
        font = _ensure_font()
        if self._selected:
            bg = _BG_HOVER
            border = _ACCENT
        elif self._hovered and self.enabled:
            bg = _BG_HOVER
            border = _ACCENT
        else:
            bg = _BG
            border = _BORDER

        pygame.draw.rect(surface, bg, self.rect)
        pygame.draw.rect(surface, border, self.rect, 1)

        text_color = _TEXT if self.enabled else _TEXT_DIM
        text_surf = font.render(self.label, True, text_color)
        tx = self.rect.x + (self.rect.width - text_surf.get_width()) // 2
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

    def handle_click(self, mx: int, my: int) -> bool:
        if self.enabled and self.rect.collidepoint(mx, my):
            if self.on_click is not None:
                self.on_click()
            return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        self._hovered = self.rect.collidepoint(mx, my)

    def handle_mouse_up(self) -> None:
        pass


# ── Toggle ───────────────────────────────────────────────────────────


class Toggle:
    """On/off toggle switch."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        on: bool = False,
        on_change: Callable[[bool], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.on = on
        self.on_change = on_change
        self._hovered = False

    def render(self, surface: pygame.Surface) -> None:
        track_w = min(36, self.rect.width)
        track_h = min(18, self.rect.height)
        track_x = self.rect.x + (self.rect.width - track_w) // 2
        track_y = self.rect.y + (self.rect.height - track_h) // 2

        track_color = _TOGGLE_ON if self.on else _TOGGLE_OFF
        pygame.draw.rect(
            surface, track_color,
            (track_x, track_y, track_w, track_h),
            border_radius=9,
        )

        knob_r = max(3, track_h // 2 - 2)
        if self.on:
            knob_x = track_x + track_w - knob_r - 3
        else:
            knob_x = track_x + knob_r + 3
        knob_y = track_y + track_h // 2
        pygame.draw.circle(surface, _SLIDER_KNOB, (knob_x, knob_y), knob_r)

    def handle_click(self, mx: int, my: int) -> bool:
        if self.rect.collidepoint(mx, my):
            self.on = not self.on
            if self.on_change is not None:
                self.on_change(self.on)
            return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        self._hovered = self.rect.collidepoint(mx, my)

    def handle_mouse_up(self) -> None:
        pass


# ── Slider ───────────────────────────────────────────────────────────


class Slider:
    """Draggable slider with track, fill, and knob."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        value: float = 0.5,
        min_val: float = 0.0,
        max_val: float = 1.0,
        on_change: Callable[[float], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.min_val = min_val
        self.max_val = max_val
        self.value = max(min_val, min(max_val, value))
        self.on_change = on_change
        self._dragging = False
        self._hovered = False

    @property
    def _normalized(self) -> float:
        rng = self.max_val - self.min_val
        if rng <= 0:
            return 0.0
        return (self.value - self.min_val) / rng

    def render(self, surface: pygame.Surface) -> None:
        track_h = 6
        track_y = self.rect.y + (self.rect.height - track_h) // 2
        track_x = self.rect.x
        track_w = self.rect.width

        pygame.draw.rect(
            surface, _SLIDER_TRACK, (track_x, track_y, track_w, track_h)
        )

        fill_w = int(track_w * self._normalized)
        if fill_w > 0:
            pygame.draw.rect(
                surface, _SLIDER_FILL, (track_x, track_y, fill_w, track_h)
            )

        knob_x = track_x + fill_w
        knob_y = self.rect.y + self.rect.height // 2
        knob_r = 7
        knob_color = _ACCENT if self._dragging else _SLIDER_KNOB
        pygame.draw.circle(surface, knob_color, (knob_x, knob_y), knob_r)

    def handle_click(self, mx: int, my: int) -> bool:
        if self.rect.collidepoint(mx, my):
            self._dragging = True
            self._update_value_from_x(mx)
            return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        self._hovered = self.rect.collidepoint(mx, my)
        if self._dragging:
            self._update_value_from_x(mx)

    def handle_mouse_up(self) -> None:
        self._dragging = False

    def _update_value_from_x(self, mx: int) -> None:
        track_x = self.rect.x
        track_w = self.rect.width
        if track_w <= 0:
            return
        t = max(0.0, min(1.0, (mx - track_x) / track_w))
        self.value = self.min_val + t * (self.max_val - self.min_val)
        if self.on_change is not None:
            self.on_change(self.value)


# ── Dropdown ─────────────────────────────────────────────────────────


class Dropdown:
    """Collapsed/expanded dropdown menu with scrollable item list."""

    def __init__(
        self,
        x: int,
        y: int,
        w: int,
        h: int,
        items: list[str] | None = None,
        selected: int = 0,
        on_change: Callable[[int, str], Any] | None = None,
    ) -> None:
        self.rect = pygame.Rect(x, y, w, h)
        self.items: list[str] = items or []
        self.selected_index = selected if items else -1
        self.on_change = on_change
        self.expanded = False
        self._hovered_index = -1
        self._max_visible = 6
        self._scroll_offset = 0

    @property
    def selected_text(self) -> str | None:
        if 0 <= self.selected_index < len(self.items):
            return self.items[self.selected_index]
        return None

    def set_items(
        self, items: list[str], selected: int = 0
    ) -> None:
        self.items = items
        self.selected_index = selected if items else -1
        self._scroll_offset = 0

    def render(self, surface: pygame.Surface) -> None:
        font = _ensure_font()

        bg = _BG_HOVER if self.expanded else _BG
        pygame.draw.rect(surface, bg, self.rect)
        pygame.draw.rect(surface, _BORDER, self.rect, 1)

        display = self.selected_text or "—"
        text_surf = font.render(display, True, _TEXT)
        tx = self.rect.x + 6
        ty = self.rect.y + (self.rect.height - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

        arrow = "^" if self.expanded else "v"
        arrow_surf = font.render(arrow, True, _TEXT_DIM)
        ax = self.rect.x + self.rect.width - arrow_surf.get_width() - 6
        surface.blit(arrow_surf, (ax, ty))

        if self.expanded and self.items:
            self._render_items(surface)

    def _render_items(self, surface: pygame.Surface) -> None:
        font = _ensure_font()
        item_h = self.rect.height
        visible_count = min(len(self.items), self._max_visible)
        list_h = visible_count * item_h
        list_y = self.rect.y + self.rect.height
        scrollable = len(self.items) > self._max_visible
        scrollbar_w = 6 if scrollable else 0
        text_area_w = self.rect.width - scrollbar_w

        pygame.draw.rect(
            surface, _DROPDOWN_BG,
            (self.rect.x, list_y, self.rect.width, list_h),
        )
        pygame.draw.rect(
            surface, _BORDER,
            (self.rect.x, list_y, self.rect.width, list_h), 1,
        )

        for i in range(visible_count):
            idx = i + self._scroll_offset
            if idx >= len(self.items):
                break
            iy = list_y + i * item_h

            if idx == self._hovered_index:
                pygame.draw.rect(
                    surface, _DROPDOWN_HOVER,
                    (self.rect.x + 1, iy, text_area_w - 2, item_h),
                )

            if idx == self.selected_index:
                pygame.draw.rect(
                    surface, _ACCENT,
                    (self.rect.x + 1, iy, 3, item_h),
                )

            text_surf = font.render(self.items[idx], True, _TEXT)
            surface.blit(
                text_surf,
                (self.rect.x + 8,
                 iy + (item_h - text_surf.get_height()) // 2),
            )

        if scrollable and list_h > 0:
            sb_x = self.rect.x + self.rect.width - scrollbar_w
            pygame.draw.rect(
                surface, _SLIDER_TRACK, (sb_x, list_y, scrollbar_w, list_h)
            )
            max_offset = len(self.items) - self._max_visible
            thumb_h = max(
                8, int(list_h * self._max_visible / len(self.items))
            )
            thumb_travel = list_h - thumb_h
            ratio = self._scroll_offset / max_offset if max_offset > 0 else 0
            thumb_y = list_y + int(thumb_travel * ratio)
            pygame.draw.rect(
                surface, _TEXT_DIM, (sb_x, thumb_y, scrollbar_w, thumb_h)
            )

    def handle_click(self, mx: int, my: int) -> bool:
        if self.expanded and self.items:
            item_h = self.rect.height
            list_y = self.rect.y + self.rect.height
            visible_count = min(len(self.items), self._max_visible)
            list_bottom = list_y + visible_count * item_h

            if (self.rect.x <= mx <= self.rect.x + self.rect.width
                    and list_y <= my <= list_bottom):
                clicked_i = (my - list_y) // item_h
                idx = clicked_i + self._scroll_offset
                if 0 <= idx < len(self.items):
                    self.selected_index = idx
                    self.expanded = False
                    if self.on_change is not None:
                        self.on_change(idx, self.items[idx])
                    return True

        if self.rect.collidepoint(mx, my):
            self.expanded = not self.expanded
            return True

        if self.expanded:
            self.expanded = False
            return True

        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        if not self.expanded:
            self._hovered_index = -1
            return

        item_h = self.rect.height
        list_y = self.rect.y + self.rect.height

        if self.rect.x <= mx < self.rect.x + self.rect.width and my >= list_y:
            i = (my - list_y) // item_h + self._scroll_offset
            if 0 <= i < len(self.items):
                self._hovered_index = i
                return
        self._hovered_index = -1

    def handle_scroll(self, direction: int) -> bool:
        if not self.expanded or len(self.items) <= self._max_visible:
            return False
        max_offset = len(self.items) - self._max_visible
        if direction > 0:
            self._scroll_offset = max(0, self._scroll_offset - 1)
        elif direction < 0:
            self._scroll_offset = min(max_offset, self._scroll_offset + 1)
        return True

    def handle_mouse_up(self) -> None:
        pass
