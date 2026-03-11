"""Modern Minimal theme — centralized colors, sizes, fonts, and mood mapping.

All visual constants live here. No other GUI module should define colors or
font sizes. Import from theme.py instead.
"""
from __future__ import annotations

import pygame

# ── Void Background ──────────────────────────────────────────────────

VOID_BG = (8, 8, 15)


class Colors:
    """All UI colors as class attributes."""

    # Surface overlays (RGBA — use with SRCALPHA surfaces)
    SURFACE_3 = (255, 255, 255, 8)     # 3% white opacity
    SURFACE_5 = (255, 255, 255, 13)    # 5% white opacity
    BORDER = (255, 255, 255, 15)       # 6% white opacity
    BORDER_SUBTLE = (255, 255, 255, 10)  # 4% white opacity

    # Text (RGB — alpha handled by surface)
    TEXT_90 = (230, 230, 230)          # 90% white
    TEXT_50 = (128, 128, 128)          # 50% white
    TEXT_30 = (77, 77, 77)             # 30% white
    TEXT_25 = (64, 64, 64)             # 25% white
    TEXT_20 = (51, 51, 51)             # 20% white

    # Status
    STATUS_GREEN = (74, 222, 128)      # #4ade80
    STATUS_YELLOW = (245, 158, 11)     # #f59e0b
    STATUS_RED = (239, 68, 68)         # #ef4444

    # Creature / warm accents
    WARM_BG = (200, 160, 120, 15)      # 6% warm for bubbles
    WARM_BORDER = (200, 160, 120, 25)  # 10% warm for bubble border
    WARM_TEXT = (200, 160, 120)        # Seaman name text
    WARM_BODY = (200, 160, 120, 179)   # 70% warm for body text

    # Chat panel
    CHAT_BG = (8, 8, 8, 217)          # 85% alpha dark overlay

    # Widget accents
    ACCENT = (80, 160, 220)
    ACCENT_HOVER = (100, 180, 240)
    TOGGLE_ON = (74, 222, 128)
    TOGGLE_OFF = (100, 110, 130)


class Sizes:
    """Pixel dimensions for layout constants."""

    TOP_BAR_H = 32
    SIDEBAR_W = 48
    TILE = 24
    TILE_RADIUS = 6
    TILE_GAP = 6
    CHAT_H = 130
    INPUT_H = 28
    INPUT_RADIUS = 14
    DRAWER_WIDTH_RATIO = 0.4  # 40% of screen width


# ── Mood → Glow Color Mapping ───────────────────────────────────────

_MOOD_GLOW: dict[str, tuple[int, int, int]] = {
    "content": (230, 190, 80),
    "amused": (230, 190, 80),
    "curious": (230, 190, 80),
    "neutral": (210, 140, 80),
    "philosophical": (210, 140, 80),
    "sardonic": (210, 160, 60),
    "irritated": (210, 160, 60),
    "hostile": (210, 80, 60),
    "sleeping": (140, 100, 180),
    "sad": (80, 130, 210),
}

_DEFAULT_GLOW = (210, 140, 80)  # amber fallback


def mood_glow_color(mood: str) -> tuple[int, int, int]:
    """Return the glow RGB for a mood string."""
    return _MOOD_GLOW.get(mood, _DEFAULT_GLOW)


def status_color(value: float) -> tuple[int, int, int]:
    """Return green/yellow/red based on a 0-1 value."""
    if value >= 0.5:
        return Colors.STATUS_GREEN
    if value >= 0.25:
        return Colors.STATUS_YELLOW
    return Colors.STATUS_RED


# ── Fonts ────────────────────────────────────────────────────────────

class Fonts:
    """Lazily initialized font instances."""

    label: pygame.font.Font | None = None   # 9-10px, uppercase labels
    body: pygame.font.Font | None = None    # 11px, content text
    header: pygame.font.Font | None = None  # 12px, bold headers

    @classmethod
    def init(cls) -> None:
        """Initialize all fonts. Call after pygame.font.init()."""
        cls.label = _make_font(10)
        cls.body = _make_font(11)
        cls.header = _make_font(12)


def _make_font(size: int) -> pygame.font.Font:
    """Create a font, trying common monospace families first."""
    for name in ("consolas", "couriernew", "courier"):
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue
    return pygame.font.Font(None, size)


def render_spaced_text(
    text: str,
    font: pygame.font.Font,
    color: tuple[int, ...],
    spacing: int = 2,
) -> pygame.Surface:
    """Render text with custom letter-spacing by drawing each glyph."""
    glyphs = [font.render(ch, True, color) for ch in text]
    total_w = sum(g.get_width() for g in glyphs) + spacing * max(0, len(text) - 1)
    h = font.get_linesize()
    surface = pygame.Surface((max(1, total_w), h), pygame.SRCALPHA)
    x = 0
    for g in glyphs:
        surface.blit(g, (x, 0))
        x += g.get_width() + spacing
    return surface
