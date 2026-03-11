"""Render engine — gradient cache, particle system, overlays, notifications.

Provides reusable rendering primitives for the Modern Minimal aesthetic.
"""
from __future__ import annotations

import math
import random

import pygame

from seaman_brain.gui.theme import VOID_BG, Colors


class GradientCache:
    """Pre-computed radial gradient surfaces, regenerated on demand."""

    def __init__(self) -> None:
        self._cache: dict[tuple[tuple[int, int, int], int], pygame.Surface] = {}

    def get(self, color: tuple[int, int, int], radius: int) -> pygame.Surface:
        """Get a cached radial gradient surface, creating if needed."""
        key = (color, radius)
        if key not in self._cache:
            self._cache[key] = self._create_radial(color, radius)
        return self._cache[key]

    def invalidate(self) -> None:
        """Clear the cache (call when mood or size changes)."""
        self._cache.clear()

    @staticmethod
    def _create_radial(
        color: tuple[int, int, int], radius: int
    ) -> pygame.Surface:
        """Draw concentric circles with decreasing alpha."""
        size = radius * 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        r, g, b = color
        cx, cy = radius, radius
        for i in range(radius, 0, -2):
            alpha = int(60 * (i / radius) ** 2)  # quadratic falloff
            pygame.draw.circle(surface, (r, g, b, alpha), (cx, cy), i)
        return surface


class ParticleSystem:
    """Floating particles drifting upward in the void."""

    def __init__(
        self, count: int = 12, bounds: tuple[int, int, int, int] = (0, 0, 800, 600)
    ) -> None:
        self.bounds = bounds  # (x, y, w, h)
        self.particles: list[dict] = []
        for _ in range(count):
            self.particles.append(self._spawn(randomize_y=True))

    def _spawn(self, randomize_y: bool = False) -> dict:
        x0, y0, w, h = self.bounds
        return {
            "x": random.uniform(x0, x0 + w),
            "y": random.uniform(y0, y0 + h) if randomize_y else y0 + h,
            "speed": random.uniform(8, 25),
            "size": random.choice([1, 1, 1, 2]),
            "alpha": random.randint(38, 77),  # 15-30% of 255
        }

    def update(self, dt: float) -> None:
        """Move particles upward, respawn at bottom when off-top."""
        x0, y0, _, h = self.bounds
        for p in self.particles:
            p["y"] -= p["speed"] * dt
            if p["y"] < y0:
                new = self._spawn()
                p.update(new)

    def render(self, surface: pygame.Surface) -> None:
        """Draw all particles."""
        for p in self.particles:
            color = (255, 255, 255, p["alpha"])
            pos = (int(p["x"]), int(p["y"]))
            pygame.draw.circle(surface, color, pos, p["size"])


# ── Overlay Rendering ───────────────────────────────────────────────


def render_game_over(
    surface: pygame.Surface, w: int, h: int, cause: str = "unknown"
) -> None:
    """Full-screen game-over overlay with void aesthetic.

    Args:
        surface: Target surface to draw on.
        w: Surface width.
        h: Surface height.
        cause: Death cause text to display.
    """
    # Dim overlay
    dim = pygame.Surface((w, h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 180))
    surface.blit(dim, (0, 0))

    # Red glow center
    glow_r = min(w, h) // 3
    glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
    for i in range(glow_r, 0, -3):
        alpha = int(40 * (i / glow_r) ** 2)
        pygame.draw.circle(
            glow, (*Colors.STATUS_RED[:3], alpha), (glow_r, glow_r), i,
        )
    surface.blit(glow, (w // 2 - glow_r, h // 2 - glow_r))

    # Death text
    try:
        font_large = pygame.font.SysFont("consolas", 28)
        font_small = pygame.font.SysFont("consolas", 14)
    except Exception:
        font_large = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 14)

    title = font_large.render("YOUR CREATURE HAS DIED", True, Colors.STATUS_RED)
    surface.blit(title, (w // 2 - title.get_width() // 2, h // 2 - 40))

    cause_surf = font_small.render(f"Cause: {cause}", True, Colors.TEXT_50)
    surface.blit(cause_surf, (w // 2 - cause_surf.get_width() // 2, h // 2 + 10))

    hint = font_small.render("Press SPACE to continue", True, Colors.TEXT_20)
    surface.blit(hint, (w // 2 - hint.get_width() // 2, h // 2 + 50))


def render_evolution(
    surface: pygame.Surface,
    w: int,
    h: int,
    stage_name: str = "Unknown",
    progress: float = 0.0,
) -> None:
    """Full-screen evolution overlay with gold pulsing glow.

    Args:
        surface: Target surface to draw on.
        w: Surface width.
        h: Surface height.
        stage_name: Name of the stage being evolved to.
        progress: Animation progress 0.0-1.0 (for pulse intensity).
    """
    # Dim overlay
    dim = pygame.Surface((w, h), pygame.SRCALPHA)
    dim.fill((0, 0, 0, 160))
    surface.blit(dim, (0, 0))

    # Gold glow center (pulse based on progress)
    pulse = 0.6 + 0.4 * math.sin(progress * math.pi * 2)
    glow_r = int(min(w, h) // 3 * pulse)
    if glow_r > 0:
        glow = pygame.Surface((glow_r * 2, glow_r * 2), pygame.SRCALPHA)
        gold = (255, 200, 60)
        for i in range(glow_r, 0, -3):
            alpha = int(50 * (i / glow_r) ** 2)
            pygame.draw.circle(glow, (*gold, alpha), (glow_r, glow_r), i)
        surface.blit(glow, (w // 2 - glow_r, h // 2 - glow_r))

    # Stage text
    try:
        font_large = pygame.font.SysFont("consolas", 32)
        font_small = pygame.font.SysFont("consolas", 14)
    except Exception:
        font_large = pygame.font.Font(None, 32)
        font_small = pygame.font.Font(None, 14)

    title = font_large.render("EVOLVING", True, (255, 200, 60))
    surface.blit(title, (w // 2 - title.get_width() // 2, h // 2 - 40))

    stage_surf = font_small.render(f"→ {stage_name}", True, Colors.TEXT_90)
    surface.blit(stage_surf, (w // 2 - stage_surf.get_width() // 2, h // 2 + 10))


# ── Notification Manager ────────────────────────────────────────────

_NOTIF_MAX = 5
_NOTIF_DURATION = 4.0  # seconds visible
_NOTIF_FADE = 0.5  # seconds for fade in/out


class NotificationManager:
    """Toast notification stack in the bottom-left corner."""

    def __init__(self) -> None:
        self._notifications: list[dict] = []

    def add(self, text: str) -> None:
        """Add a toast notification."""
        self._notifications.append({"text": text, "age": 0.0})
        # Cap at max
        if len(self._notifications) > _NOTIF_MAX:
            self._notifications = self._notifications[-_NOTIF_MAX:]

    def update(self, dt: float) -> None:
        """Age notifications and remove expired ones."""
        for n in self._notifications:
            n["age"] += dt
        self._notifications = [
            n for n in self._notifications if n["age"] < _NOTIF_DURATION
        ]

    def render(self, surface: pygame.Surface, w: int, h: int) -> None:
        """Render notification toasts in the bottom-left."""
        if not self._notifications:
            return

        try:
            font = pygame.font.SysFont("consolas", 12)
        except Exception:
            font = pygame.font.Font(None, 12)

        x = 60  # right of sidebar
        y = h - 20

        for n in reversed(self._notifications):
            age = n["age"]
            # Compute alpha for fade in/out
            if age < _NOTIF_FADE:
                alpha = int(255 * (age / _NOTIF_FADE))
            elif age > _NOTIF_DURATION - _NOTIF_FADE:
                alpha = int(255 * ((_NOTIF_DURATION - age) / _NOTIF_FADE))
            else:
                alpha = 255

            text_surf = font.render(n["text"], True, Colors.TEXT_90)
            tw = text_surf.get_width()
            th = text_surf.get_height()
            pill_w = tw + 16
            pill_h = th + 8

            # Pill background
            pill = pygame.Surface((pill_w, pill_h), pygame.SRCALPHA)
            pill.fill((*VOID_BG, min(alpha, 200)))
            pygame.draw.rect(
                pill, (*Colors.BORDER[:3], min(alpha, 120)),
                (0, 0, pill_w, pill_h), 1,
            )
            surface.blit(pill, (x, y - pill_h))

            # Text
            text_surf.set_alpha(alpha)
            surface.blit(text_surf, (x + 8, y - pill_h + 4))

            y -= pill_h + 4
