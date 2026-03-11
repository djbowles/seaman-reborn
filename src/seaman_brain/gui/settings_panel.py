"""Settings slide-out drawer with vertical tab strip.

Slides in from the right edge. Vertical tab strip on the left with
Personality / LLM / Audio / Vision icons. Content area renders widgets
for the active tab. All colors from theme.py. No pygame_gui dependency.

render(surface, progress) — progress 0-1 controls slide animation position.
"""
from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

import pygame

from seaman_brain.gui.theme import Colors, Fonts

# ── Layout Constants ──────────────────────────────────────────────────

_TAB_STRIP_W = 40
_TAB_H = 40
_TAB_GAP = 4
_CONTENT_PAD = 12
_ROW_H = 28
_LABEL_W = 120
_DRAWER_ALPHA = 230


class SettingsTab(Enum):
    """Settings panel tabs."""

    PERSONALITY = "Personality"
    LLM = "LLM Model"
    AUDIO = "Audio"
    VISION = "Vision"


_TAB_ICONS: dict[SettingsTab, str] = {
    SettingsTab.PERSONALITY: "P",
    SettingsTab.LLM: "L",
    SettingsTab.AUDIO: "A",
    SettingsTab.VISION: "V",
}


class SettingsPanel:
    """Slide-out settings drawer with vertical tab strip and callbacks."""

    def __init__(
        self,
        width: int = 400,
        *,
        on_personality_change: Callable[[dict[str, float]], Any] | None = None,
        on_llm_apply: Callable[[str, float], Any] | None = None,
        on_audio_change: Callable[[str, Any], Any] | None = None,
        on_vision_change: Callable[[str, Any], Any] | None = None,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        self._width = width
        self.on_personality_change = on_personality_change
        self.on_llm_apply = on_llm_apply
        self.on_audio_change = on_audio_change
        self.on_vision_change = on_vision_change
        self.on_close = on_close

        self.active_tab = SettingsTab.PERSONALITY
        self._font: pygame.font.Font | None = None

    # ── Tab management ────────────────────────────────────────────────

    def set_tab(self, tab: SettingsTab) -> None:
        """Switch the active tab."""
        self.active_tab = tab

    # ── Callback helpers ──────────────────────────────────────────────

    def _fire_personality_change(self, traits: dict[str, float]) -> None:
        if self.on_personality_change is not None:
            self.on_personality_change(traits)

    def _fire_audio_change(self, key: str, value: Any) -> None:
        if self.on_audio_change is not None:
            self.on_audio_change(key, value)

    def _fire_llm_apply(self, model: str, temperature: float) -> None:
        if self.on_llm_apply is not None:
            self.on_llm_apply(model, temperature)

    def _fire_vision_change(self, key: str, value: Any) -> None:
        if self.on_vision_change is not None:
            self.on_vision_change(key, value)

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

        # Tab strip
        self._render_tab_strip(surface, slide_x, screen_h)

        # Content area
        content_x = slide_x + _TAB_STRIP_W
        content_w = self._width - _TAB_STRIP_W
        self._render_content(surface, content_x, content_w, screen_h)

    def _render_tab_strip(
        self, surface: pygame.Surface, x: int, h: int,
    ) -> None:
        """Render vertical tab strip with icons."""
        font = self._font
        y = 8

        # Tab strip background
        pygame.draw.rect(
            surface, Colors.SURFACE_3[:3],
            (x + 1, 0, _TAB_STRIP_W - 1, h),
        )

        for tab in SettingsTab:
            icon = _TAB_ICONS[tab]
            is_active = tab == self.active_tab

            if is_active:
                # Accent line on left edge
                pygame.draw.rect(
                    surface, Colors.ACCENT,
                    (x + 1, y, 3, _TAB_H),
                )
                # Active tab highlight
                pygame.draw.rect(
                    surface, Colors.SURFACE_5[:3],
                    (x + 4, y, _TAB_STRIP_W - 4, _TAB_H),
                )
                text_color = Colors.TEXT_90
            else:
                text_color = Colors.TEXT_30

            icon_surf = font.render(icon, True, text_color)
            ix = x + 1 + (_TAB_STRIP_W - 1 - icon_surf.get_width()) // 2
            iy = y + (_TAB_H - icon_surf.get_height()) // 2
            surface.blit(icon_surf, (ix, iy))
            y += _TAB_H + _TAB_GAP

    def _render_content(
        self, surface: pygame.Surface, x: int, w: int, h: int,
    ) -> None:
        """Render the content area for the active tab."""
        font = self._font

        # Tab title
        title = self.active_tab.value
        title_surf = font.render(title, True, Colors.TEXT_90)
        surface.blit(title_surf, (x + _CONTENT_PAD, 12))

        # Separator line
        sep_y = 12 + title_surf.get_height() + 6
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (x + _CONTENT_PAD, sep_y),
            (x + w - _CONTENT_PAD, sep_y), 1,
        )

    # ── Input handling ────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle mouse click. Returns True if consumed."""
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Handle mouse movement for hover states."""

    def handle_mouse_up(self) -> None:
        """Handle mouse button release."""
