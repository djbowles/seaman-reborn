"""Render engine — gradient cache, particle system, draw-order orchestration.

Provides reusable rendering primitives for the Modern Minimal aesthetic.
"""
from __future__ import annotations

import random

import pygame


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
