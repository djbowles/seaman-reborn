"""Tank environment renderer - water, bubbles, gravel, terrarium.

Renders the creature's habitat as a Pygame surface. Supports two modes:
- Aquarium: animated water surface, rising bubbles, gravel floor, blue-green tint.
- Terrarium: dry substrate, rocks, moisture particles, green tint.

Temperature is shown as a color-graded overlay (blue=cold, green=good, red=hot).
Cleanliness affects water/air clarity (murky green when dirty).
Smooth animated transition between aquarium and terrarium during drain/fill.
All rendering uses procedural Pygame draw calls — no external assets.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from seaman_brain.config import EnvironmentConfig, GUIConfig
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment

# ── Color Palettes ────────────────────────────────────────────────────

_AQUARIUM_BG = (8, 30, 55)
_AQUARIUM_WATER = (15, 60, 100)
_TERRARIUM_BG = (25, 35, 18)
_TERRARIUM_GROUND = (60, 50, 30)
_GRAVEL_BASE = (80, 70, 55)
_GRAVEL_DARK = (55, 48, 35)
_BUBBLE_COLOR = (180, 220, 255, 120)
_WATER_SURFACE_COLOR = (100, 180, 220)
_ROCK_COLOR = (90, 85, 75)
_ROCK_HIGHLIGHT = (120, 115, 105)
_MOISTURE_COLOR = (150, 200, 160, 80)
_MAX_SURFACE_DIM = 8192  # Max pixel dimension for any Surface allocation


@dataclass
class _Bubble:
    """A single animated bubble rising through water."""

    x: float
    y: float
    radius: float
    speed: float
    wobble_phase: float = 0.0
    wobble_amp: float = 0.0


@dataclass
class _MoistureParticle:
    """A single moisture/mist particle floating in terrarium air."""

    x: float
    y: float
    radius: float
    speed: float
    alpha: float = 0.5
    drift_phase: float = 0.0


@dataclass
class _Rock:
    """A decorative rock in the terrarium."""

    x: float
    y: float
    width: float
    height: float
    color: tuple[int, int, int] = (90, 85, 75)


class TankRenderer:
    """Renders the tank environment on a Pygame surface.

    Creates animated aquarium (water + bubbles + gravel) or terrarium
    (substrate + rocks + moisture) visuals. Temperature shown as color
    overlay, cleanliness as murkiness.

    Attributes:
        config: GUI configuration for sizing.
        env_config: Environment config for temperature ranges.
    """

    def __init__(
        self,
        gui_config: GUIConfig | None = None,
        env_config: EnvironmentConfig | None = None,
    ) -> None:
        self.config = gui_config or GUIConfig()
        self.env_config = env_config or EnvironmentConfig()

        # Rendering area (full window minus HUD space at top)
        self._top_margin = 45
        self._render_x = 0
        self._render_y = self._top_margin
        self._render_w = self.config.window_width
        self._render_h = self.config.window_height - self._top_margin

        # Animation state
        self._time = 0.0
        self._bubbles: list[_Bubble] = []
        self._moisture: list[_MoistureParticle] = []
        self._rocks: list[_Rock] = []
        self._gravel_points: list[tuple[int, int, int]] = []  # x, y, shade

        # Transition animation (0.0 = aquarium, 1.0 = terrarium)
        self._transition_progress = 0.0
        self._transition_target = 0.0
        self._transition_speed = 0.3  # units per second

        self._initialized = False

    def _init_decorations(self) -> None:
        """Generate random gravel, rocks, and initial bubbles/moisture."""
        if self._initialized:
            return
        self._initialized = True

        w = self._render_w
        h = self._render_h
        ground_y = self._render_y + int(h * 0.85)

        # Gravel layer (many small dots)
        for _ in range(120):
            gx = random.randint(self._render_x + 5, self._render_x + w - 5)
            gy = random.randint(ground_y, self._render_y + h - 3)
            shade = random.randint(0, 40)
            self._gravel_points.append((gx, gy, shade))

        # Rocks (for terrarium, also visible as underwater stones)
        for _ in range(6):
            rx = random.randint(self._render_x + 30, self._render_x + w - 60)
            ry = random.randint(ground_y - 10, ground_y + 15)
            rw = random.randint(20, 50)
            rh = random.randint(12, 25)
            shade = random.randint(-15, 15)
            color = (
                max(0, min(255, _ROCK_COLOR[0] + shade)),
                max(0, min(255, _ROCK_COLOR[1] + shade)),
                max(0, min(255, _ROCK_COLOR[2] + shade)),
            )
            self._rocks.append(_Rock(x=rx, y=ry, width=rw, height=rh, color=color))

        # Initial bubbles
        self._spawn_bubbles(8)

        # Initial moisture particles
        self._spawn_moisture(10)

    def _spawn_bubbles(self, count: int) -> None:
        """Spawn new bubbles at the bottom of the tank."""
        for _ in range(count):
            bx = random.randint(
                self._render_x + 20, self._render_x + self._render_w - 20
            )
            by = random.uniform(
                self._render_y + self._render_h * 0.5,
                self._render_y + self._render_h * 0.9,
            )
            radius = random.uniform(2.0, 6.0)
            speed = random.uniform(20.0, 50.0)
            wobble_amp = random.uniform(1.0, 4.0)
            wobble_phase = random.uniform(0.0, math.tau)
            self._bubbles.append(
                _Bubble(
                    x=bx,
                    y=by,
                    radius=radius,
                    speed=speed,
                    wobble_amp=wobble_amp,
                    wobble_phase=wobble_phase,
                )
            )

    def _spawn_moisture(self, count: int) -> None:
        """Spawn floating moisture/mist particles in the air."""
        for _ in range(count):
            mx = random.randint(
                self._render_x + 10, self._render_x + self._render_w - 10
            )
            my = random.uniform(
                self._render_y + self._render_h * 0.1,
                self._render_y + self._render_h * 0.6,
            )
            radius = random.uniform(1.5, 4.0)
            speed = random.uniform(5.0, 15.0)
            alpha = random.uniform(0.2, 0.6)
            drift_phase = random.uniform(0.0, math.tau)
            self._moisture.append(
                _MoistureParticle(
                    x=mx,
                    y=my,
                    radius=radius,
                    speed=speed,
                    alpha=alpha,
                    drift_phase=drift_phase,
                )
            )

    def update(self, dt: float, tank: TankEnvironment) -> None:
        """Update animation state.

        Args:
            dt: Delta time in seconds since last frame.
            tank: Current tank environment state.
        """
        self._init_decorations()
        self._time += dt

        # Update transition target based on environment type
        self._transition_target = (
            1.0 if tank.environment_type == EnvironmentType.TERRARIUM else 0.0
        )

        # Smoothly interpolate transition
        if self._transition_progress < self._transition_target:
            self._transition_progress = min(
                self._transition_target,
                self._transition_progress + self._transition_speed * dt,
            )
        elif self._transition_progress > self._transition_target:
            self._transition_progress = max(
                self._transition_target,
                self._transition_progress - self._transition_speed * dt,
            )

        # Update bubbles (visible when transition < 1.0)
        if self._transition_progress < 1.0:
            self._update_bubbles(dt)

        # Update moisture particles (visible when transition > 0.0)
        if self._transition_progress > 0.0:
            self._update_moisture(dt)

    def _update_bubbles(self, dt: float) -> None:
        """Move bubbles upward and respawn ones that reach the surface."""
        water_surface_y = self._render_y + self._render_h * 0.08
        alive: list[_Bubble] = []
        for b in self._bubbles:
            b.y -= b.speed * dt
            b.wobble_phase += dt * 3.0
            if b.y > water_surface_y:
                alive.append(b)
        self._bubbles = alive

        # Respawn to maintain ~8-12 bubbles
        if len(self._bubbles) < 8:
            self._spawn_bubbles(random.randint(1, 3))

    def _update_moisture(self, dt: float) -> None:
        """Float moisture particles gently downward and drift horizontally."""
        alive: list[_MoistureParticle] = []
        for m in self._moisture:
            m.y += m.speed * dt
            m.drift_phase += dt * 1.5
            m.x += math.sin(m.drift_phase) * 0.5

            # Respawn if off screen
            if m.y < self._render_y + self._render_h * 0.8:
                alive.append(m)
        self._moisture = alive

        if len(self._moisture) < 8:
            self._spawn_moisture(random.randint(1, 3))

    def render(self, surface: pygame.Surface, tank: TankEnvironment) -> None:
        """Render the tank environment onto the given surface.

        Args:
            surface: Pygame surface to draw on.
            tank: Current tank environment state for visual parameters.
        """
        self._init_decorations()

        t = self._transition_progress

        # Draw background (blend between aquarium and terrarium)
        self._render_background(surface, t)

        # Draw ground/substrate
        self._render_ground(surface, t)

        # Draw rocks
        self._render_rocks(surface, t)

        # Draw gravel
        self._render_gravel(surface, t)

        # Draw water surface (fades out in terrarium)
        if t < 1.0:
            self._render_water_surface(surface, t)
            self._render_bubbles(surface, t, tank.cleanliness)

        # Draw moisture particles (fades in for terrarium)
        if t > 0.0:
            self._render_moisture_particles(surface, t)

        # Cleanliness overlay (murky green when dirty)
        self._render_cleanliness_overlay(surface, tank.cleanliness, t)

        # Temperature overlay
        self._render_temperature_overlay(surface, tank.temperature)

    def _render_background(
        self, surface: pygame.Surface, transition: float
    ) -> None:
        """Draw the blended background color."""
        bg = _lerp_color(_AQUARIUM_BG, _TERRARIUM_BG, transition)
        area = pygame.Rect(
            self._render_x, self._render_y, self._render_w, self._render_h
        )
        pygame.draw.rect(surface, bg, area)

    def _render_ground(
        self, surface: pygame.Surface, transition: float
    ) -> None:
        """Draw the ground/substrate layer at the bottom."""
        ground_y = self._render_y + int(self._render_h * 0.85)
        ground_h = self._render_h - int(self._render_h * 0.85)
        ground_color = _lerp_color(_GRAVEL_BASE, _TERRARIUM_GROUND, transition)

        area = pygame.Rect(self._render_x, ground_y, self._render_w, ground_h)
        pygame.draw.rect(surface, ground_color, area)

    def _render_gravel(
        self, surface: pygame.Surface, transition: float
    ) -> None:
        """Draw scattered gravel dots on the ground."""
        for gx, gy, shade in self._gravel_points:
            base = _lerp_color(_GRAVEL_BASE, _GRAVEL_DARK, transition)
            color = (
                max(0, min(255, base[0] - shade)),
                max(0, min(255, base[1] - shade)),
                max(0, min(255, base[2] - shade)),
            )
            pygame.draw.circle(surface, color, (gx, gy), 2)

    def _render_rocks(
        self, surface: pygame.Surface, transition: float
    ) -> None:
        """Draw decorative rocks."""
        for rock in self._rocks:
            # Rocks more prominent in terrarium
            alpha_mult = 0.5 + 0.5 * transition
            color = (
                int(rock.color[0] * alpha_mult),
                int(rock.color[1] * alpha_mult),
                int(rock.color[2] * alpha_mult),
            )
            # Main rock body (ellipse)
            rect = pygame.Rect(
                int(rock.x), int(rock.y), int(rock.width), int(rock.height)
            )
            pygame.draw.ellipse(surface, color, rect)
            # Highlight
            highlight = (
                min(255, color[0] + 30),
                min(255, color[1] + 30),
                min(255, color[2] + 30),
            )
            highlight_rect = pygame.Rect(
                int(rock.x + rock.width * 0.2),
                int(rock.y + rock.height * 0.1),
                int(rock.width * 0.4),
                int(rock.height * 0.4),
            )
            pygame.draw.ellipse(surface, highlight, highlight_rect)

    def _render_water_surface(
        self, surface: pygame.Surface, transition: float
    ) -> None:
        """Draw animated water surface ripples."""
        alpha = 1.0 - transition
        if alpha <= 0.0:
            return

        y_base = self._render_y + int(self._render_h * 0.08)
        color = (
            int(_WATER_SURFACE_COLOR[0] * alpha),
            int(_WATER_SURFACE_COLOR[1] * alpha),
            int(_WATER_SURFACE_COLOR[2] * alpha),
        )

        # Draw wavy line
        points: list[tuple[int, int]] = []
        for x in range(self._render_x, self._render_x + self._render_w, 4):
            wave = math.sin(x * 0.02 + self._time * 2.0) * 3.0
            wave += math.sin(x * 0.05 + self._time * 1.3) * 1.5
            points.append((x, int(y_base + wave)))

        if len(points) >= 2:
            pygame.draw.lines(surface, color, False, points, 2)

            # Subtle fill below surface line
            fill_color = (color[0] // 3, color[1] // 3, color[2] // 3)
            fill_points = (
                points
                + [(self._render_x + self._render_w, y_base + 8)]
                + [(self._render_x, y_base + 8)]
            )
            if len(fill_points) >= 3:
                pygame.draw.polygon(surface, fill_color, fill_points)

    def _render_bubbles(
        self,
        surface: pygame.Surface,
        transition: float,
        cleanliness: float,
    ) -> None:
        """Draw rising bubbles."""
        alpha = 1.0 - transition
        if alpha <= 0.0:
            return

        for b in self._bubbles:
            wobble_x = math.sin(b.wobble_phase) * b.wobble_amp
            bx = int(b.x + wobble_x)
            by = int(b.y)
            radius = max(1, int(b.radius))

            # Bubble color affected by cleanliness
            brightness = int(180 * cleanliness + 60)
            color = (
                int(min(255, brightness * 0.7) * alpha),
                int(min(255, brightness * 0.9) * alpha),
                int(min(255, brightness) * alpha),
            )
            pygame.draw.circle(surface, color, (bx, by), radius, 1)

            # Highlight dot
            if radius > 2:
                hl_color = (
                    int(min(255, 220) * alpha),
                    int(min(255, 240) * alpha),
                    int(min(255, 255) * alpha),
                )
                pygame.draw.circle(
                    surface, hl_color, (bx - 1, by - 1), max(1, radius // 3)
                )

    def _render_moisture_particles(
        self, surface: pygame.Surface, transition: float
    ) -> None:
        """Draw floating moisture/mist particles."""
        alpha = transition
        if alpha <= 0.0:
            return

        for m in self._moisture:
            drift_x = math.sin(m.drift_phase) * 3.0
            mx = int(m.x + drift_x)
            my = int(m.y)
            radius = max(1, int(m.radius))

            brightness = int(200 * m.alpha * alpha)
            color = (
                int(brightness * 0.6),
                int(brightness * 0.9),
                int(brightness * 0.65),
            )
            pygame.draw.circle(surface, color, (mx, my), radius)

    def _render_cleanliness_overlay(
        self,
        surface: pygame.Surface,
        cleanliness: float,
        transition: float,
    ) -> None:
        """Draw murky overlay when tank is dirty."""
        # Dirtiness increases overlay opacity (0 at clean, ~80 at filthy)
        dirtiness = 1.0 - max(0.0, min(1.0, cleanliness))
        if dirtiness < 0.05:
            return

        overlay_alpha = int(dirtiness * 80)
        # Greenish murk for aquarium, brownish for terrarium
        r = int(_lerp(20, 40, transition))
        g = int(_lerp(50, 35, transition))
        b = int(_lerp(15, 10, transition))

        ow = max(1, min(self._render_w, _MAX_SURFACE_DIM))
        oh = max(1, min(self._render_h, _MAX_SURFACE_DIM))
        overlay = pygame.Surface((ow, oh), pygame.SRCALPHA)
        overlay.fill((r, g, b, overlay_alpha))
        surface.blit(overlay, (self._render_x, self._render_y))

    def _render_temperature_overlay(
        self, surface: pygame.Surface, temperature: float
    ) -> None:
        """Draw a subtle temperature-based color tint.

        Blue when cold, green when optimal, red when hot.
        Only shows when temperature deviates from optimal.
        """
        cfg = self.env_config
        opt_min = cfg.optimal_temp_min
        opt_max = cfg.optimal_temp_max

        # No overlay if in optimal range
        if opt_min <= temperature <= opt_max:
            return

        if temperature < opt_min:
            # Cold: blue tint, stronger as temperature drops
            severity = min(1.0, (opt_min - temperature) / (opt_min - cfg.lethal_temp_min))
            r, g, b = 30, 60, 180
            alpha = int(severity * 50)
        else:
            # Hot: red tint, stronger as temperature rises
            severity = min(1.0, (temperature - opt_max) / (cfg.lethal_temp_max - opt_max))
            r, g, b = 200, 50, 20
            alpha = int(severity * 50)

        if alpha < 2:
            return

        ow = max(1, min(self._render_w, _MAX_SURFACE_DIM))
        oh = max(1, min(self._render_h, _MAX_SURFACE_DIM))
        overlay = pygame.Surface((ow, oh), pygame.SRCALPHA)
        overlay.fill((r, g, b, alpha))
        surface.blit(overlay, (self._render_x, self._render_y))

    def set_render_area(self, x: int, y: int, w: int, h: int) -> None:
        """Set the rendering area for the tank.

        Args:
            x: Left edge of the render area.
            y: Top edge of the render area.
            w: Width of the render area.
            h: Height of the render area.
        """
        self._render_x = x
        self._render_y = y
        self._render_w = w
        self._render_h = h
        self._initialized = False  # Re-init decorations for new area

    @property
    def transition_progress(self) -> float:
        """Current aquarium->terrarium transition (0.0=aquarium, 1.0=terrarium)."""
        return self._transition_progress

    @property
    def render_area(self) -> tuple[int, int, int, int]:
        """The rendering area as (x, y, width, height)."""
        return (self._render_x, self._render_y, self._render_w, self._render_h)


def _lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation between a and b by t."""
    return a + (b - a) * t


def _lerp_color(
    c1: tuple[int, int, int], c2: tuple[int, int, int], t: float
) -> tuple[int, int, int]:
    """Linear interpolation between two RGB colors."""
    return (
        int(_lerp(c1[0], c2[0], t)),
        int(_lerp(c1[1], c2[1], t)),
        int(_lerp(c1[2], c2[2], t)),
    )
