"""Proportional layout engine — computes pixel rects from screen size.

All layout constants come from theme.Sizes. This module computes the
actual pixel rectangles that components use for rendering and hit-testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from seaman_brain.gui.theme import Sizes


@dataclass
class Region:
    """A rectangular region on screen."""

    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.w, self.h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def collidepoint(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


class ScreenLayout:
    """Computes all screen regions from window dimensions.

    Call resize() when the window changes size. All components should
    read their regions from here rather than computing positions themselves.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._compute()

    def resize(self, width: int, height: int) -> None:
        """Recompute all regions for new window dimensions."""
        self.width = width
        self.height = height
        self._compute()

    def _compute(self) -> None:
        w, h = self.width, self.height
        tb = Sizes.TOP_BAR_H
        sb = Sizes.SIDEBAR_W
        ch = Sizes.CHAT_H

        self.top_bar = Region(0, 0, w, tb)
        self.sidebar = Region(0, tb, sb, h - tb)
        self.tank = Region(sb, tb, w - sb, h - tb - ch)
        self.chat = Region(0, h - ch, w, ch)
        self.drawer_width = int(w * Sizes.DRAWER_WIDTH_RATIO)
