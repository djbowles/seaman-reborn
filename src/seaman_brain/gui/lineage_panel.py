"""Lineage slide-out drawer for managing creature bloodlines.

Slides in from the right edge, same pattern as settings_panel.
Shows a scrollable list of bloodlines with the active one highlighted.
Select / New / Delete buttons at the bottom.

render(surface, progress) — progress 0-1 controls slide animation position.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pygame

from seaman_brain.gui.theme import Colors, Fonts

# ── Layout Constants ──────────────────────────────────────────────────

_HEADER_H = 36
_ITEM_H = 32
_CONTENT_PAD = 12
_BUTTON_H = 28
_BUTTON_GAP = 8
_DRAWER_ALPHA = 230


class LineagePanel:
    """Slide-out drawer for bloodline management."""

    def __init__(
        self,
        width: int = 400,
        *,
        on_select: Callable[[str], Any] | None = None,
        on_new: Callable[[str], Any] | None = None,
        on_delete: Callable[[str], Any] | None = None,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        self._width = width
        self.on_select = on_select
        self.on_new = on_new
        self.on_delete = on_delete
        self.on_close = on_close

        self._bloodlines: list[str] = []
        self._active: str | None = None
        self._selected_index: int = 0
        self._hovered_index: int = -1
        self._scroll_offset: int = 0
        self._font: pygame.font.Font | None = None

    # ── Data ──────────────────────────────────────────────────────────

    def set_bloodlines(
        self, names: list[str], active: str | None = None,
    ) -> None:
        """Set the bloodline list and active name."""
        self._bloodlines = list(names)
        self._active = active
        self._scroll_offset = 0
        # Select the active bloodline if present
        if active and active in names:
            self._selected_index = names.index(active)
        else:
            self._selected_index = 0

    def _select(self, name: str) -> None:
        """Select a bloodline and fire the callback."""
        self._active = name
        if name in self._bloodlines:
            self._selected_index = self._bloodlines.index(name)
        if self.on_select is not None:
            self.on_select(name)

    # ── Font ──────────────────────────────────────────────────────────

    def _ensure_font(self) -> None:
        if self._font is None:
            if Fonts.body is not None:
                self._font = Fonts.body
            else:
                for name in ("consolas", "couriernew", "courier"):
                    try:
                        self._font = pygame.font.SysFont(name, 11)
                        return
                    except Exception:
                        continue
                self._font = pygame.font.Font(None, 11)

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface, progress: float = 1.0) -> None:
        """Render the drawer. progress 0-1 controls slide position."""
        if progress <= 0:
            return
        self._ensure_font()
        if self._font is None:
            return

        screen_w = surface.get_width()
        screen_h = surface.get_height()
        slide_x = screen_w - int(self._width * progress)

        # Drawer background
        bg = pygame.Surface((self._width, screen_h), pygame.SRCALPHA)
        bg.fill((8, 8, 15, _DRAWER_ALPHA))
        surface.blit(bg, (slide_x, 0))

        # Left border
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (slide_x, 0), (slide_x, screen_h), 1,
        )

        font = self._font

        # Header
        title_surf = font.render("Lineage", True, Colors.TEXT_90)
        surface.blit(
            title_surf,
            (slide_x + _CONTENT_PAD, (_HEADER_H - title_surf.get_height()) // 2),
        )
        # Header separator
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (slide_x + _CONTENT_PAD, _HEADER_H),
            (slide_x + self._width - _CONTENT_PAD, _HEADER_H), 1,
        )

        # Bloodline list
        y = _HEADER_H + 8
        for i, name in enumerate(self._bloodlines):
            item_y = y + i * _ITEM_H
            is_active = name == self._active
            is_selected = i == self._selected_index

            # Highlight
            if is_selected:
                pygame.draw.rect(
                    surface, Colors.SURFACE_5[:3],
                    (slide_x + 4, item_y, self._width - 8, _ITEM_H),
                )

            # Active accent
            if is_active:
                pygame.draw.rect(
                    surface, Colors.ACCENT,
                    (slide_x + 4, item_y, 3, _ITEM_H),
                )

            text_color = Colors.TEXT_90 if is_active else Colors.TEXT_50
            name_surf = font.render(name, True, text_color)
            surface.blit(
                name_surf,
                (slide_x + _CONTENT_PAD + 8, item_y + (_ITEM_H - name_surf.get_height()) // 2),
            )

    # ── Input handling ────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle mouse click. Returns True if consumed."""
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Handle mouse movement for hover states."""

    def handle_mouse_up(self) -> None:
        """Handle mouse button release."""
