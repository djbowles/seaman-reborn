"""Void tank renderer — dark background with radial glow and particles.

Replaces the old aquarium/terrarium renderer with the Modern Minimal void.
"""
from __future__ import annotations

import pygame

from seaman_brain.gui.render_engine import GradientCache, ParticleSystem
from seaman_brain.gui.theme import VOID_BG, mood_glow_color


class TankRenderer:
    """Renders the void: background fill, center glow, floating particles."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._gradient_cache = GradientCache()
        self._particles = ParticleSystem(count=12, bounds=(0, 0, width, height))
        self._mood = "neutral"

    def resize(self, width: int, height: int) -> None:
        """Update dimensions on window resize."""
        self._width = width
        self._height = height
        self._particles = ParticleSystem(
            count=12, bounds=(0, 0, width, height)
        )
        self._gradient_cache.invalidate()

    def set_mood(self, mood: str) -> None:
        """Update the creature mood for glow color."""
        if mood != self._mood:
            self._mood = mood
            self._gradient_cache.invalidate()

    def update(self, dt: float) -> None:
        """Advance particle animation."""
        self._particles.update(dt)

    def render(self, surface: pygame.Surface) -> None:
        """Draw the void: background, center glow, particles."""
        surface.fill(VOID_BG)

        # Center glow
        glow_color = mood_glow_color(self._mood)
        radius = min(self._width, self._height) // 3
        if radius > 10:
            glow_surf = self._gradient_cache.get(glow_color, radius)
            cx = self._width // 2 - radius
            cy = self._height // 2 - radius
            surface.blit(glow_surf, (cx, cy))

        # Particles
        self._particles.render(surface)
