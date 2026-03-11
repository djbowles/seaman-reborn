"""Modern Minimal HUD — thin top bar + left sidebar tiles.

Top bar (32px): creature name, stage, mood, mic/tts indicators, lineage/settings
buttons, session timer.

Left sidebar (48px): 4 need tiles (hunger, health, comfort, trust) +
divider + 7 action tiles (feed, aerator, temp up/down, clean, drain, fill).
Tooltips on hover, cooldown overlays on action tiles.
"""
from __future__ import annotations

import math
import time

import pygame

from seaman_brain.gui.layout import ScreenLayout
from seaman_brain.gui.theme import (
    VOID_BG,
    Colors,
    Fonts,
    Sizes,
    status_color,
)

# ── Tile Definitions ─────────────────────────────────────────────────

_NEED_DEFS: list[dict[str, str]] = [
    {"key": "hunger", "icon": "H", "label": "Hunger"},
    {"key": "health", "icon": "+", "label": "Health"},
    {"key": "comfort", "icon": "C", "label": "Comfort"},
    {"key": "trust", "icon": "T", "label": "Trust"},
]

_ACTION_DEFS: list[dict[str, str]] = [
    {"key": "feed", "icon": "F", "label": "Feed"},
    {"key": "aerator", "icon": "O", "label": "Toggle Aerator"},
    {"key": "temp_up", "icon": "^", "label": "Raise Temp"},
    {"key": "temp_down", "icon": "v", "label": "Lower Temp"},
    {"key": "clean", "icon": "*", "label": "Clean Tank"},
    {"key": "drain", "icon": "~", "label": "Drain Tank"},
    {"key": "fill", "icon": "o", "label": "Fill Tank"},
]

# Mic/TTS indicator colors
_MIC_ACTIVE = Colors.STATUS_GREEN
_MIC_INACTIVE = Colors.TEXT_20
_TTS_ACTIVE = Colors.STATUS_YELLOW


class HUD:
    """Heads-up display with top bar and sidebar tiles."""

    def __init__(self, layout: ScreenLayout) -> None:
        self._layout = layout
        self._font: pygame.font.Font | None = None

        # Need tiles: icon, key, label, color, value
        self._need_tiles: list[dict] = []
        for d in _NEED_DEFS:
            self._need_tiles.append({
                "key": d["key"],
                "icon": d["icon"],
                "label": d["label"],
                "color": Colors.STATUS_GREEN,
                "value": 1.0,
            })

        # Action tiles: icon, key, label, cooldown_end
        self._action_tiles: list[dict] = []
        for d in _ACTION_DEFS:
            self._action_tiles.append({
                "key": d["key"],
                "icon": d["icon"],
                "label": d["label"],
                "cooldown_end": 0.0,
            })

        # Top bar state
        self._creature_name = "Seaman"
        self._stage_name = "Mushroomer"
        self._mood_name = "neutral"

        # Audio indicators
        self.mic_active: bool = False
        self.tts_active: bool = False
        self.tts_provider_label: str = ""
        self.stt_provider_label: str = ""
        self._mic_pulse_timer: float = 0.0

        # Session timer
        self._session_time: float = 0.0

        # Click-detection rects (set during render)
        self.settings_rect: tuple | None = None
        self.lineage_rect: tuple | None = None
        self.mic_rect: tuple | None = None

        # Tooltip
        self._tooltip: str | None = None

        # Tile rects (computed on render for click/hover detection)
        self._tile_rects: list[tuple[int, int, int, int, str, str]] = []

    @property
    def session_time(self) -> float:
        return self._session_time

    def update(self, dt: float) -> None:
        """Update session timer and mic pulse."""
        self._session_time += dt
        if self.mic_active:
            self._mic_pulse_timer += dt
        else:
            self._mic_pulse_timer = 0.0

    def update_needs(
        self, hunger: float, health: float, comfort: float, trust: float
    ) -> None:
        """Update need tile values and colors."""
        values = [hunger, health, comfort, trust]
        for tile, val in zip(self._need_tiles, values):
            tile["value"] = val
            tile["color"] = status_color(val)

    def update_creature_info(
        self,
        stage: str = "Mushroomer",
        mood: str = "neutral",
        name: str = "Seaman",
    ) -> None:
        """Update top bar creature info."""
        self._stage_name = stage
        self._mood_name = mood
        self._creature_name = name

    def set_cooldown(self, action_key: str, duration: float) -> None:
        """Set a cooldown timer on an action tile."""
        end = time.monotonic() + duration
        for tile in self._action_tiles:
            if tile["key"] == action_key:
                tile["cooldown_end"] = end
                break

    def resize(self, layout: ScreenLayout) -> None:
        """Update layout reference on window resize."""
        self._layout = layout

    def handle_click(self, mx: int, my: int) -> str | None:
        """Check if click hits an action tile. Returns action key or None."""
        for rx, ry, rw, rh, kind, key in self._tile_rects:
            if kind == "action" and rx <= mx < rx + rw and ry <= my < ry + rh:
                return key
        return None

    def handle_hover(self, mx: int, my: int) -> None:
        """Update tooltip based on hover position."""
        for rx, ry, rw, rh, _kind, key in self._tile_rects:
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                # Find matching label
                for d in _NEED_DEFS + _ACTION_DEFS:
                    if d["key"] == key:
                        self._tooltip = d["label"]
                        return
        self._tooltip = None

    def _ensure_font(self) -> None:
        if self._font is None:
            if Fonts.label is not None:
                self._font = Fonts.label
            else:
                for name in ("consolas", "couriernew", "courier"):
                    try:
                        self._font = pygame.font.SysFont(name, 10)
                        return
                    except Exception:
                        continue
                self._font = pygame.font.Font(None, 10)

    def _format_session_time(self) -> str:
        total = int(self._session_time)
        hours = total // 3600
        minutes = (total % 3600) // 60
        seconds = total % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface) -> None:
        """Render top bar and sidebar onto surface."""
        self._ensure_font()
        if self._font is None:
            return

        self._tile_rects.clear()
        self._render_top_bar(surface)
        self._render_sidebar(surface)

        # Tooltip
        if self._tooltip:
            self._render_tooltip(surface)

    def _render_top_bar(self, surface: pygame.Surface) -> None:
        """Render the 32px top bar."""
        w = surface.get_width()
        bar_h = Sizes.TOP_BAR_H

        # Background
        bar_surf = pygame.Surface((w, bar_h), pygame.SRCALPHA)
        bar_surf.fill((*VOID_BG, 220))
        surface.blit(bar_surf, (0, 0))

        # Bottom border line
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (0, bar_h - 1), (w, bar_h - 1), 1,
        )

        font = self._font
        cy = bar_h // 2

        # Left: creature name + stage
        title = f"{self._creature_name} - {self._stage_name}"
        title_surf = font.render(title, True, Colors.TEXT_90)
        surface.blit(
            title_surf, (Sizes.SIDEBAR_W + 8, cy - title_surf.get_height() // 2)
        )

        # Center: mood
        mood_surf = font.render(self._mood_name.capitalize(), True, Colors.TEXT_50)
        mood_x = (w - mood_surf.get_width()) // 2
        surface.blit(mood_surf, (mood_x, cy - mood_surf.get_height() // 2))

        # Right side (flowing left from right edge)
        cursor_x = w - 8

        # Timer
        timer_text = self._format_session_time()
        timer_surf = font.render(timer_text, True, Colors.TEXT_20)
        cursor_x -= timer_surf.get_width()
        surface.blit(timer_surf, (cursor_x, cy - timer_surf.get_height() // 2))
        cursor_x -= 12

        # Settings button
        settings_text = "SET"
        stx_surf = font.render(settings_text, True, Colors.TEXT_50)
        stx_w = stx_surf.get_width() + 8
        cursor_x -= stx_w
        btn_rect = pygame.Rect(cursor_x, 4, stx_w, bar_h - 8)
        pygame.draw.rect(surface, Colors.SURFACE_3[:3], btn_rect)
        pygame.draw.rect(surface, Colors.BORDER[:3], btn_rect, 1)
        surface.blit(
            stx_surf,
            (cursor_x + 4, cy - stx_surf.get_height() // 2),
        )
        self.settings_rect = (btn_rect[0], btn_rect[1], btn_rect[2], btn_rect[3]) \
            if not isinstance(btn_rect, tuple) else btn_rect
        cursor_x -= 8

        # Lineage button
        lin_text = "LIN"
        lin_surf = font.render(lin_text, True, Colors.TEXT_50)
        lin_w = lin_surf.get_width() + 8
        cursor_x -= lin_w
        lin_rect = pygame.Rect(cursor_x, 4, lin_w, bar_h - 8)
        pygame.draw.rect(surface, Colors.SURFACE_3[:3], lin_rect)
        pygame.draw.rect(surface, Colors.BORDER[:3], lin_rect, 1)
        surface.blit(
            lin_surf,
            (cursor_x + 4, cy - lin_surf.get_height() // 2),
        )
        self.lineage_rect = (lin_rect[0], lin_rect[1], lin_rect[2], lin_rect[3]) \
            if not isinstance(lin_rect, tuple) else lin_rect
        cursor_x -= 12

        # TTS dot
        tts_color = _TTS_ACTIVE if self.tts_active else _MIC_INACTIVE
        pygame.draw.circle(surface, tts_color, (cursor_x, cy), 4)
        cursor_x -= 12

        # Mic dot (pulsing when active)
        if self.mic_active:
            pulse = math.sin(self._mic_pulse_timer * 2.0 * math.pi * 1.5)
            t = pulse * 0.5 + 0.5
            mic_color = (
                int(_MIC_INACTIVE[0] + (_MIC_ACTIVE[0] - _MIC_INACTIVE[0]) * t),
                int(_MIC_INACTIVE[1] + (_MIC_ACTIVE[1] - _MIC_INACTIVE[1]) * t),
                int(_MIC_INACTIVE[2] + (_MIC_ACTIVE[2] - _MIC_INACTIVE[2]) * t),
            )
        else:
            mic_color = _MIC_INACTIVE
        pygame.draw.circle(surface, mic_color, (cursor_x, cy), 4)
        self.mic_rect = (cursor_x - 4, cy - 4, 8, 8)

    def _render_sidebar(self, surface: pygame.Surface) -> None:
        """Render the 48px left sidebar with tiles."""
        sb = self._layout.sidebar
        font = self._font

        # Background
        sb_surf = pygame.Surface((sb.w, sb.h), pygame.SRCALPHA)
        sb_surf.fill((*VOID_BG, 200))
        surface.blit(sb_surf, (sb.x, sb.y))

        # Right border
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (sb.x + sb.w - 1, sb.y), (sb.x + sb.w - 1, sb.y + sb.h), 1,
        )

        tile_size = Sizes.TILE
        pad = (sb.w - tile_size) // 2
        y = sb.y + 8

        # Need tiles
        now = time.monotonic()
        for tile in self._need_tiles:
            tx = sb.x + pad
            # Tile background
            pygame.draw.rect(
                surface, Colors.SURFACE_5[:3],
                (tx, y, tile_size, tile_size),
            )
            # Icon
            icon_surf = font.render(tile["icon"], True, tile["color"])
            ix = tx + (tile_size - icon_surf.get_width()) // 2
            iy = y + (tile_size - icon_surf.get_height()) // 2
            surface.blit(icon_surf, (ix, iy))
            self._tile_rects.append(
                (tx, y, tile_size, tile_size, "need", tile["key"])
            )
            y += tile_size + 4

        # Divider line
        y += 4
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (sb.x + 8, y), (sb.x + sb.w - 8, y), 1,
        )
        y += 8

        # Action tiles
        for tile in self._action_tiles:
            tx = sb.x + pad
            pygame.draw.rect(
                surface, Colors.SURFACE_3[:3],
                (tx, y, tile_size, tile_size),
            )
            icon_surf = font.render(tile["icon"], True, Colors.TEXT_50)
            ix = tx + (tile_size - icon_surf.get_width()) // 2
            iy = y + (tile_size - icon_surf.get_height()) // 2
            surface.blit(icon_surf, (ix, iy))

            # Cooldown overlay
            if tile["cooldown_end"] > now:
                # Semi-transparent overlay
                pygame.draw.rect(
                    surface, (*Colors.STATUS_RED, 80),
                    (tx, y, tile_size, tile_size),
                )

            self._tile_rects.append(
                (tx, y, tile_size, tile_size, "action", tile["key"])
            )
            y += tile_size + 4

    def _render_tooltip(self, surface: pygame.Surface) -> None:
        """Render tooltip text near the sidebar."""
        if self._font is None or self._tooltip is None:
            return
        tip_surf = self._font.render(self._tooltip, True, Colors.TEXT_90)
        x = self._layout.sidebar.w + 4
        y = self._layout.sidebar.y + 4
        # Background
        bw = tip_surf.get_width() + 8
        bh = tip_surf.get_height() + 4
        pygame.draw.rect(surface, VOID_BG, (x, y, bw, bh))
        pygame.draw.rect(surface, Colors.BORDER[:3], (x, y, bw, bh), 1)
        surface.blit(tip_surf, (x + 4, y + 2))
