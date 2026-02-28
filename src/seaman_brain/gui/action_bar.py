"""Right-side action bar with large labeled buttons.

Provides a vertical panel of interaction buttons (Feed, Temp+, Temp-,
Clean, Drain, Tap) as an alternative to the tiny in-tank buttons.
Each button has an icon, label, hover state, cooldown indicator, and click callback.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import pygame

# ── Colors ───────────────────────────────────────────────────────────────

_PANEL_BG = (12, 22, 42, 200)
_HEADER_COLOR = (180, 200, 220)
_BUTTON_BG = (25, 40, 65)
_BUTTON_BG_HOVER = (40, 65, 100)
_BUTTON_BORDER = (70, 100, 140)
_BUTTON_ICON_COLOR = (180, 220, 255)
_BUTTON_LABEL_COLOR = (180, 200, 220)

# Cooldown state colors
_BUTTON_BG_COOLDOWN = (15, 20, 35)
_COOLDOWN_BAR_COLOR = (60, 140, 220)
_COOLDOWN_TEXT_COLOR = (140, 160, 180)

# ── Constants ────────────────────────────────────────────────────────────

_BUTTON_HEIGHT = 40
_BUTTON_MARGIN = 6
_BUTTON_PADDING_X = 10
_HEADER_HEIGHT = 28
_FONT_SIZE = 14
_ICON_FONT_SIZE = 16
_COOLDOWN_BAR_HEIGHT = 3


@dataclass
class ActionButton:
    """A single action bar button with icon, label, hover state."""

    key: str
    icon: str
    label: str
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = _BUTTON_HEIGHT
    hover: bool = False
    cooldown: float = field(default=0.0)
    cooldown_max: float = field(default=0.0)

    def contains(self, mx: int, my: int) -> bool:
        """Check if a point is inside this button."""
        return (
            self.x <= mx <= self.x + self.width
            and self.y <= my <= self.y + self.height
        )


# Button definitions: (key, icon, label)
_BUTTON_DEFS: list[tuple[str, str, str]] = [
    ("feed", "F", "Feed"),
    ("aerate", "O", "Aerate"),
    ("temp_up", "^", "Temp +"),
    ("temp_down", "v", "Temp -"),
    ("clean", "*", "Clean"),
    ("drain", "~", "Drain"),
    ("tap_glass", "o", "Tap"),
]


class ActionBar:
    """Right-side vertical panel of interaction buttons.

    Attributes:
        buttons: List of ActionButton instances.
    """

    def __init__(self, on_action: Callable[[str], None] | None = None) -> None:
        """Initialize the action bar.

        Args:
            on_action: Callback invoked with the action key when a button is clicked.
        """
        self._on_action = on_action
        self.buttons: list[ActionButton] = []

        # Panel area (set by set_panel_area)
        self._x = 0
        self._y = 0
        self._w = 160
        self._h = 0

        # Fonts (lazy-initialized)
        self._font: pygame.font.Font | None = None
        self._icon_font: pygame.font.Font | None = None

    def _ensure_fonts(self) -> None:
        """Initialize fonts if not yet done."""
        if self._font is None:
            for name in ("consolas", "couriernew", "courier"):
                try:
                    self._font = pygame.font.SysFont(name, _FONT_SIZE)
                    self._icon_font = pygame.font.SysFont(name, _ICON_FONT_SIZE, bold=True)
                    break
                except Exception:
                    continue
            if self._font is None:
                self._font = pygame.font.Font(None, _FONT_SIZE)
                self._icon_font = pygame.font.Font(None, _ICON_FONT_SIZE)

    def set_panel_area(self, x: int, y: int, w: int, h: int) -> None:
        """Set the panel area and rebuild button positions.

        Args:
            x: Left edge of the panel.
            y: Top edge of the panel.
            w: Width of the panel.
            h: Height of the panel.
        """
        self._x = x
        self._y = y
        self._w = w
        self._h = h
        self._build_buttons()

    def _build_buttons(self) -> None:
        """Build button instances positioned within the panel."""
        self.buttons.clear()
        btn_x = self._x + _BUTTON_MARGIN
        btn_w = self._w - 2 * _BUTTON_MARGIN
        btn_y = self._y + _HEADER_HEIGHT + _BUTTON_MARGIN

        for key, icon, label in _BUTTON_DEFS:
            self.buttons.append(ActionButton(
                key=key,
                icon=icon,
                label=label,
                x=btn_x,
                y=btn_y,
                width=btn_w,
                height=_BUTTON_HEIGHT,
            ))
            btn_y += _BUTTON_HEIGHT + _BUTTON_MARGIN

    def update_cooldowns(
        self,
        feed_remaining: float = 0.0,
        feed_max: float = 0.0,
        clean_remaining: float = 0.0,
        clean_max: float = 0.0,
        aerate_remaining: float = 0.0,
        aerate_max: float = 0.0,
    ) -> None:
        """Update cooldown state for buttons that have cooldowns.

        Args:
            feed_remaining: Seconds remaining on feed cooldown.
            feed_max: Maximum feed cooldown duration.
            clean_remaining: Seconds remaining on clean cooldown.
            clean_max: Maximum clean cooldown duration.
            aerate_remaining: Seconds remaining on aerate cooldown.
            aerate_max: Maximum aerate cooldown duration.
        """
        mapping = {
            "feed": (feed_remaining, feed_max),
            "clean": (clean_remaining, clean_max),
            "aerate": (aerate_remaining, aerate_max),
        }
        for btn in self.buttons:
            remaining, maximum = mapping.get(btn.key, (0.0, 0.0))
            btn.cooldown = remaining
            btn.cooldown_max = maximum

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle a mouse click. Returns True if a button was clicked.

        Args:
            mx: Mouse x position.
            my: Mouse y position.

        Returns:
            True if a button was clicked and the action callback was invoked.
        """
        for btn in self.buttons:
            if btn.contains(mx, my):
                if btn.cooldown > 0:
                    return True  # Consumed click but don't fire action
                if self._on_action is not None:
                    self._on_action(btn.key)
                return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update button hover states.

        Args:
            mx: Mouse x position.
            my: Mouse y position.
        """
        for btn in self.buttons:
            btn.hover = btn.contains(mx, my)

    def render(self, surface: pygame.Surface) -> None:
        """Render the action bar panel and buttons.

        Args:
            surface: Pygame surface to draw on.
        """
        self._ensure_fonts()
        if self._font is None or self._icon_font is None:
            return

        # Panel background
        panel_surf = pygame.Surface((self._w, self._h), pygame.SRCALPHA)
        panel_surf.fill(_PANEL_BG)
        surface.blit(panel_surf, (self._x, self._y))

        # Header text
        header_surf = self._font.render("Actions", True, _HEADER_COLOR)
        hx = self._x + (self._w - header_surf.get_width()) // 2
        hy = self._y + (_HEADER_HEIGHT - header_surf.get_height()) // 2
        surface.blit(header_surf, (hx, hy))

        # Buttons
        for btn in self.buttons:
            on_cooldown = btn.cooldown > 0
            rect = pygame.Rect(btn.x, btn.y, btn.width, btn.height)

            if on_cooldown:
                # Darkened background during cooldown
                pygame.draw.rect(surface, _BUTTON_BG_COOLDOWN, rect)
                pygame.draw.rect(surface, _BUTTON_BORDER, rect, 1)

                # Dimmed icon
                icon_surf = self._icon_font.render(btn.icon, True, _COOLDOWN_TEXT_COLOR)
                ix = btn.x + _BUTTON_PADDING_X
                iy = btn.y + (btn.height - icon_surf.get_height()) // 2
                surface.blit(icon_surf, (ix, iy))

                # Countdown text replacing label
                cd_text = f"{btn.cooldown:.0f}s"
                cd_surf = self._font.render(cd_text, True, _COOLDOWN_TEXT_COLOR)
                lx = btn.x + _BUTTON_PADDING_X + icon_surf.get_width() + 8
                ly = btn.y + (btn.height - cd_surf.get_height()) // 2
                surface.blit(cd_surf, (lx, ly))

                # Progress bar at button bottom (fills left to right as cooldown expires)
                if btn.cooldown_max > 0:
                    progress = 1.0 - (btn.cooldown / btn.cooldown_max)
                    bar_w = max(0, int(btn.width * progress))
                    bar_y = btn.y + btn.height - _COOLDOWN_BAR_HEIGHT
                    if bar_w > 0:
                        pygame.draw.rect(
                            surface, _COOLDOWN_BAR_COLOR,
                            pygame.Rect(btn.x, bar_y, bar_w, _COOLDOWN_BAR_HEIGHT),
                        )
            else:
                # Normal rendering
                bg = _BUTTON_BG_HOVER if btn.hover else _BUTTON_BG
                pygame.draw.rect(surface, bg, rect)
                pygame.draw.rect(surface, _BUTTON_BORDER, rect, 1)

                # Icon
                icon_surf = self._icon_font.render(btn.icon, True, _BUTTON_ICON_COLOR)
                ix = btn.x + _BUTTON_PADDING_X
                iy = btn.y + (btn.height - icon_surf.get_height()) // 2
                surface.blit(icon_surf, (ix, iy))

                # Label
                label_surf = self._font.render(btn.label, True, _BUTTON_LABEL_COLOR)
                lx = btn.x + _BUTTON_PADDING_X + icon_surf.get_width() + 8
                ly = btn.y + (btn.height - label_surf.get_height()) // 2
                surface.blit(label_surf, (lx, ly))
