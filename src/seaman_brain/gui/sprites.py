"""Creature sprite and animation system - procedural art for 5 evolutionary stages.

Each stage has a distinct visual: Mushroomer (small spore with eye), Gillman (fish
with human face), Podfish (fish with legs), Tadman (tadpole humanoid), Frogman
(frog-man). All art is procedural Pygame draw calls — no external assets.

Animation states: IDLE, SWIMMING, TALKING, EATING, SLEEPING.
Smooth position interpolation, mouth sync, face tracking toward mouse cursor.
"""

from __future__ import annotations

import colorsys
import math
import random
from dataclasses import dataclass, field
from enum import Enum

import pygame

from seaman_brain.config import GUIConfig
from seaman_brain.creature.genome import CreatureGenome
from seaman_brain.types import CreatureStage

# ── Enums ─────────────────────────────────────────────────────────────


class AnimationState(Enum):
    """Possible animation states for the creature."""

    IDLE = "idle"
    SWIMMING = "swimming"
    TALKING = "talking"
    EATING = "eating"
    SLEEPING = "sleeping"


# ── Color Palettes Per Stage ─────────────────────────────────────────

_STAGE_COLORS: dict[CreatureStage, dict[str, tuple[int, int, int]]] = {
    CreatureStage.MUSHROOMER: {
        "body": (180, 140, 100),
        "cap": (160, 80, 60),
        "eye": (20, 20, 20),
        "eye_white": (240, 240, 220),
        "highlight": (200, 170, 130),
    },
    CreatureStage.GILLMAN: {
        "body": (60, 140, 120),
        "fin": (40, 110, 100),
        "eye": (20, 20, 20),
        "eye_white": (230, 230, 220),
        "mouth": (120, 60, 60),
        "gill": (80, 50, 50),
    },
    CreatureStage.PODFISH: {
        "body": (80, 130, 160),
        "fin": (60, 100, 140),
        "leg": (100, 80, 70),
        "eye": (20, 20, 20),
        "eye_white": (230, 235, 240),
        "mouth": (130, 70, 70),
    },
    CreatureStage.TADMAN: {
        "body": (70, 100, 70),
        "tail": (50, 80, 50),
        "eye": (20, 20, 20),
        "eye_white": (230, 240, 220),
        "mouth": (110, 60, 50),
        "limb": (80, 110, 80),
    },
    CreatureStage.FROGMAN: {
        "body": (50, 130, 50),
        "belly": (140, 170, 100),
        "eye": (20, 20, 20),
        "eye_white": (220, 240, 200),
        "mouth": (120, 60, 50),
        "limb": (40, 110, 40),
    },
}

# Size multipliers per stage (relative to base 40px)
_STAGE_SIZES: dict[CreatureStage, float] = {
    CreatureStage.MUSHROOMER: 0.6,
    CreatureStage.GILLMAN: 1.0,
    CreatureStage.PODFISH: 1.1,
    CreatureStage.TADMAN: 1.3,
    CreatureStage.FROGMAN: 1.5,
}


# ── Position & Interpolation ─────────────────────────────────────────


@dataclass
class CreaturePosition:
    """Tracks creature position with smooth interpolation."""

    x: float = 0.0
    y: float = 0.0
    target_x: float = 0.0
    target_y: float = 0.0
    speed: float = 60.0  # pixels per second

    def move_toward_target(self, dt: float) -> None:
        """Smoothly interpolate position toward target."""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1.0:
            self.x = self.target_x
            self.y = self.target_y
            return

        move = min(self.speed * dt, dist)
        ratio = move / dist
        self.x += dx * ratio
        self.y += dy * ratio

    def set_target(self, x: float, y: float) -> None:
        """Set a new movement target."""
        self.target_x = x
        self.target_y = y

    def at_target(self) -> bool:
        """Check if creature has reached its target position."""
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return (dx * dx + dy * dy) < 1.0


# ── Creature Renderer ─────────────────────────────────────────────────


@dataclass
class CreatureRenderer:
    """Renders the creature at its current evolutionary stage.

    All art is procedural (Pygame draw calls). Each of the 5 stages has a
    distinct visual representation. Supports animation states (idle, swimming,
    talking, eating, sleeping) and face tracking toward the mouse cursor.

    Attributes:
        stage: Current evolutionary stage for visual selection.
        position: Position with smooth interpolation.
        animation_state: Current animation state.
        gui_config: GUI configuration for sizing.
    """

    stage: CreatureStage = CreatureStage.MUSHROOMER
    position: CreaturePosition = field(default_factory=CreaturePosition)
    animation_state: AnimationState = AnimationState.IDLE
    gui_config: GUIConfig = field(default_factory=GUIConfig)
    genome: CreatureGenome | None = None

    # Animation timers
    _time: float = field(default=0.0, repr=False)
    _mouth_open: float = field(default=0.0, repr=False)  # 0-1 for talking
    _blink_timer: float = field(default=0.0, repr=False)
    _is_blinking: bool = field(default=False, repr=False)
    _idle_wander_timer: float = field(default=0.0, repr=False)
    _mouse_x: float = field(default=0.0, repr=False)
    _mouse_y: float = field(default=0.0, repr=False)

    # Wander bounds (set from render area)
    _bounds_x: int = field(default=0, repr=False)
    _bounds_y: int = field(default=45, repr=False)
    _bounds_w: int = field(default=1024, repr=False)
    _bounds_h: int = field(default=723, repr=False)

    def __post_init__(self) -> None:
        """Initialize bounds from config."""
        self._bounds_w = self.gui_config.window_width
        self._bounds_h = self.gui_config.window_height - 45  # HUD margin
        # Start creature at center
        if self.position.x == 0.0 and self.position.y == 0.0:
            self.position.x = self._bounds_w / 2
            self.position.y = self._bounds_y + self._bounds_h * 0.5
            self.position.target_x = self.position.x
            self.position.target_y = self.position.y

    def set_bounds(self, x: int, y: int, w: int, h: int) -> None:
        """Set the rendering/wander bounds."""
        self._bounds_x = x
        self._bounds_y = y
        self._bounds_w = w
        self._bounds_h = h

    def set_stage(self, stage: CreatureStage) -> None:
        """Update the creature's evolutionary stage."""
        self.stage = stage

    def set_animation(self, state: AnimationState) -> None:
        """Change the animation state."""
        self.animation_state = state
        if state == AnimationState.TALKING:
            self._mouth_open = 0.5

    def set_mouse_position(self, mx: float, my: float) -> None:
        """Update the tracked mouse position for face following."""
        self._mouse_x = mx
        self._mouse_y = my

    def set_genome(self, genome: CreatureGenome | None) -> None:
        """Update the genome used for rendering parameters."""
        self.genome = genome

    # ── Genome Helpers ────────────────────────────────────────────────

    def _genome_scale(
        self, trait: str, lo: float = 0.7, hi: float = 1.3
    ) -> float:
        """Map a genome trait (0-1) to a scale factor.

        When genome is None, returns 1.0. Trait value 0.5 maps to 1.0.
        """
        if self.genome is None:
            return 1.0
        return lo + (hi - lo) * self.genome.traits.get(trait, 0.5)

    def _get_colors(
        self, stage: CreatureStage
    ) -> dict[str, tuple[int, int, int]]:
        """Get stage colors with genome HSV adjustments applied."""
        base_colors = _STAGE_COLORS[stage]
        if self.genome is None:
            return base_colors
        hue_val = self.genome.traits.get("hue", 0.5)
        sat_val = self.genome.traits.get("saturation", 0.5)
        if hue_val == 0.5 and sat_val == 0.5:
            return base_colors
        return {k: self._shift_color(v) for k, v in base_colors.items()}

    def _shift_color(
        self, color: tuple[int, int, int]
    ) -> tuple[int, int, int]:
        """Apply genome hue/saturation shift to an RGB color."""
        r, g, b = color[0] / 255.0, color[1] / 255.0, color[2] / 255.0
        h, s, v = colorsys.rgb_to_hsv(r, g, b)

        hue_val = self.genome.traits.get("hue", 0.5) if self.genome else 0.5
        sat_val = (
            self.genome.traits.get("saturation", 0.5) if self.genome else 0.5
        )

        h = (h + (hue_val - 0.5) * 0.16) % 1.0
        s = max(0.0, min(1.0, s + (sat_val - 0.5) * 0.6))

        r2, g2, b2 = colorsys.hsv_to_rgb(h, s, v)
        return (round(r2 * 255), round(g2 * 255), round(b2 * 255))

    def _render_patterns(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
        body_color: tuple[int, int, int],
    ) -> None:
        """Draw pattern details (spots, stripes) based on pattern_complexity."""
        if self.genome is None:
            return
        complexity = self.genome.traits.get("pattern_complexity", 0.5)
        if complexity <= 0.5:
            return

        # Deterministic seed from genome for consistent patterns
        seed_val = int(sum(self.genome.traits.values()) * 1000)
        rng = random.Random(seed_val)

        spot_color = (
            max(0, body_color[0] - 30),
            max(0, body_color[1] - 30),
            max(0, body_color[2] - 30),
        )

        num_spots = int((complexity - 0.5) * 14)  # 0-7 spots
        spot_r = max(1, int(base * 0.06))
        half_w = max(1, int(base * 0.4))
        half_h = max(1, int(base * 0.25))
        for _ in range(num_spots):
            sx = cx + rng.randint(-half_w, half_w)
            sy = cy + rng.randint(-half_h, half_h)
            pygame.draw.circle(surface, spot_color, (sx, sy), spot_r)

        if complexity > 0.75:
            stripe_color = (
                max(0, body_color[0] - 20),
                max(0, body_color[1] - 20),
                max(0, body_color[2] - 20),
            )
            num_stripes = max(1, int((complexity - 0.75) * 12))
            stripe_spacing = max(1, int(base * 0.12))
            for i in range(num_stripes):
                sy = cy - int(base * 0.15) + i * stripe_spacing
                hw = int(base * 0.3)
                pygame.draw.line(
                    surface, stripe_color,
                    (cx - hw, sy), (cx + hw, sy), 2,
                )

    def update(self, dt: float) -> None:
        """Update animation state and position.

        Args:
            dt: Delta time in seconds since last frame.
        """
        self._time += dt

        # Position interpolation
        self.position.move_toward_target(dt)

        # Blink timer
        self._blink_timer += dt
        if self._is_blinking:
            if self._blink_timer > 0.15:
                self._is_blinking = False
                self._blink_timer = 0.0
        elif self._blink_timer > random.uniform(2.5, 5.0):
            self._is_blinking = True
            self._blink_timer = 0.0

        # Mouth animation for talking (face_expressiveness scales amplitude)
        if self.animation_state == AnimationState.TALKING:
            expr = self._genome_scale("face_expressiveness", 0.5, 1.5)
            self._mouth_open = min(
                1.0, 0.3 + 0.7 * expr * abs(math.sin(self._time * 8.0))
            )
        else:
            self._mouth_open = max(0.0, self._mouth_open - dt * 4.0)

        # Idle wandering
        if self.animation_state in (AnimationState.IDLE, AnimationState.SWIMMING):
            self._idle_wander_timer += dt
            if self.position.at_target() and self._idle_wander_timer > 3.0:
                self._idle_wander_timer = 0.0
                self._pick_wander_target()

    def _pick_wander_target(self) -> None:
        """Choose a new random wander target within bounds."""
        margin = 60
        min_x = self._bounds_x + margin
        max_x = self._bounds_x + self._bounds_w - margin
        min_y = self._bounds_y + int(self._bounds_h * 0.3)
        max_y = self._bounds_y + int(self._bounds_h * 0.75)

        if max_x <= min_x:
            max_x = min_x + 1
        if max_y <= min_y:
            max_y = min_y + 1

        tx = random.uniform(min_x, max_x)
        ty = random.uniform(min_y, max_y)
        self.position.set_target(tx, ty)

    def render(self, surface: pygame.Surface) -> None:
        """Draw the creature on the surface.

        Args:
            surface: Pygame surface to draw on.
        """
        # Calculate eye direction toward mouse
        eye_dx = self._mouse_x - self.position.x
        eye_dy = self._mouse_y - self.position.y
        eye_dist = math.sqrt(eye_dx * eye_dx + eye_dy * eye_dy)
        if eye_dist > 0:
            eye_dir_x = eye_dx / eye_dist
            eye_dir_y = eye_dy / eye_dist
        else:
            eye_dir_x = 0.0
            eye_dir_y = 0.0

        # Base size from stage, scaled by genome body_size
        base = 40 * _STAGE_SIZES.get(self.stage, 1.0) * self._genome_scale(
            "body_size"
        )

        # Sleeping bob
        sleep_y_offset = 0.0
        if self.animation_state == AnimationState.SLEEPING:
            sleep_y_offset = math.sin(self._time * 1.5) * 3.0

        # Swimming bob
        swim_y_offset = 0.0
        if self.animation_state in (AnimationState.IDLE, AnimationState.SWIMMING):
            swim_y_offset = math.sin(self._time * 2.0) * 4.0

        cx = int(self.position.x)
        cy = int(self.position.y + swim_y_offset + sleep_y_offset)

        # Dispatch to stage-specific renderer
        if self.stage == CreatureStage.MUSHROOMER:
            self._render_mushroomer(surface, cx, cy, base, eye_dir_x, eye_dir_y)
        elif self.stage == CreatureStage.GILLMAN:
            self._render_gillman(surface, cx, cy, base, eye_dir_x, eye_dir_y)
        elif self.stage == CreatureStage.PODFISH:
            self._render_podfish(surface, cx, cy, base, eye_dir_x, eye_dir_y)
        elif self.stage == CreatureStage.TADMAN:
            self._render_tadman(surface, cx, cy, base, eye_dir_x, eye_dir_y)
        elif self.stage == CreatureStage.FROGMAN:
            self._render_frogman(surface, cx, cy, base, eye_dir_x, eye_dir_y)

        # Sleeping Z's
        if self.animation_state == AnimationState.SLEEPING:
            self._render_sleep_zs(surface, cx, cy, base)

    # ── Stage Renderers ──────────────────────────────────────────────

    def _render_mushroomer(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
        eye_dx: float,
        eye_dy: float,
    ) -> None:
        """Mushroomer: small spore with a single large eye and mushroom cap."""
        colors = self._get_colors(CreatureStage.MUSHROOMER)
        r = int(base * 0.5)

        # Stem/body — small oval
        body_rect = pygame.Rect(cx - r, cy - int(r * 0.5), r * 2, int(r * 1.5))
        pygame.draw.ellipse(surface, colors["body"], body_rect)

        # Patterns on body
        self._render_patterns(surface, cx, cy, base, colors["body"])

        # Mushroom cap — wider ellipse on top
        cap_w = int(r * 1.8)
        cap_h = int(r * 0.9)
        cap_rect = pygame.Rect(cx - cap_w // 2, cy - r - cap_h // 2, cap_w, cap_h)
        pygame.draw.ellipse(surface, colors["cap"], cap_rect)

        # Highlight on cap
        hl_rect = pygame.Rect(
            cx - cap_w // 4, cy - r - cap_h // 2 + 2, cap_w // 3, cap_h // 3
        )
        pygame.draw.ellipse(surface, colors["highlight"], hl_rect)

        # Single large eye — centered on body (genome eye_size scales radius)
        eye_r = int(r * 0.5 * self._genome_scale("eye_size"))
        self._draw_eye(
            surface, cx, cy - int(r * 0.1), eye_r, eye_dx, eye_dy, colors
        )

    def _render_gillman(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
        eye_dx: float,
        eye_dy: float,
    ) -> None:
        """Gillman: fish body with human-like face, gill slits, dorsal fin."""
        colors = self._get_colors(CreatureStage.GILLMAN)
        hw = int(base * 0.8)  # half-width
        hh = int(base * 0.5)  # half-height
        fin_s = self._genome_scale("fin_length")

        # Main body — elongated oval
        body_rect = pygame.Rect(cx - hw, cy - hh, hw * 2, hh * 2)
        pygame.draw.ellipse(surface, colors["body"], body_rect)

        # Patterns on body
        self._render_patterns(surface, cx, cy, base, colors["body"])

        # Tail fin (genome fin_length scales size)
        tail_pts = [
            (cx + hw - 5, cy),
            (cx + hw + int(base * 0.4 * fin_s), cy - int(base * 0.3 * fin_s)),
            (cx + hw + int(base * 0.4 * fin_s), cy + int(base * 0.3 * fin_s)),
        ]
        pygame.draw.polygon(surface, colors["fin"], tail_pts)

        # Dorsal fin (genome fin_length scales height)
        dorsal_pts = [
            (cx - int(hw * 0.3), cy - hh),
            (cx, cy - hh - int(base * 0.3 * fin_s)),
            (cx + int(hw * 0.3), cy - hh),
        ]
        pygame.draw.polygon(surface, colors["fin"], dorsal_pts)

        # Gill slits (3 short lines on body side)
        for i in range(3):
            gx = cx + int(hw * 0.2) + i * 6
            pygame.draw.line(
                surface, colors["gill"],
                (gx, cy - int(hh * 0.3)),
                (gx, cy + int(hh * 0.3)),
                2,
            )

        # Eyes (two, human-like — genome eye_size scales radius)
        eye_r = int(base * 0.14 * self._genome_scale("eye_size"))
        self._draw_eye(
            surface, cx - int(hw * 0.35), cy - int(hh * 0.15),
            eye_r, eye_dx, eye_dy, colors,
        )
        self._draw_eye(
            surface, cx - int(hw * 0.05), cy - int(hh * 0.15),
            eye_r, eye_dx, eye_dy, colors,
        )

        # Mouth
        mouth_w = int(base * 0.25)
        mouth_h = int(self._mouth_open * base * 0.12)
        mouth_rect = pygame.Rect(
            cx - int(hw * 0.25), cy + int(hh * 0.2), mouth_w, max(2, mouth_h)
        )
        pygame.draw.ellipse(surface, colors["mouth"], mouth_rect)

    def _render_podfish(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
        eye_dx: float,
        eye_dy: float,
    ) -> None:
        """Podfish: fish with stubby legs — transitional form."""
        colors = self._get_colors(CreatureStage.PODFISH)
        hw = int(base * 0.75)
        hh = int(base * 0.45)
        fin_s = self._genome_scale("fin_length")
        limb_s = self._genome_scale("limb_proportion")

        # Body
        body_rect = pygame.Rect(cx - hw, cy - hh, hw * 2, hh * 2)
        pygame.draw.ellipse(surface, colors["body"], body_rect)

        # Patterns on body
        self._render_patterns(surface, cx, cy, base, colors["body"])

        # Tail fin (genome fin_length scales)
        tail_pts = [
            (cx + hw - 4, cy),
            (cx + hw + int(base * 0.3 * fin_s), cy - int(base * 0.25 * fin_s)),
            (cx + hw + int(base * 0.3 * fin_s), cy + int(base * 0.25 * fin_s)),
        ]
        pygame.draw.polygon(surface, colors["fin"], tail_pts)

        # Stubby legs (genome limb_proportion scales height)
        leg_w = int(base * 0.08 * limb_s)
        leg_h = int(base * 0.25 * limb_s)
        for offset in [-int(hw * 0.4), int(hw * 0.2)]:
            lx = cx + offset
            walk_anim = math.sin(self._time * 3.0 + offset) * 3.0
            leg_rect = pygame.Rect(
                lx - max(1, leg_w), cy + hh - 2,
                max(2, leg_w * 2), max(1, leg_h + int(walk_anim)),
            )
            pygame.draw.ellipse(surface, colors["leg"], leg_rect)

        # Pectoral fins (genome fin_length scales)
        fin_pts = [
            (cx - hw + 5, cy),
            (cx - hw - int(base * 0.15 * fin_s), cy + int(base * 0.1)),
            (cx - hw + 5, cy + int(base * 0.15 * fin_s)),
        ]
        pygame.draw.polygon(surface, colors["fin"], fin_pts)

        # Eyes (genome eye_size scales radius)
        eye_r = int(base * 0.12 * self._genome_scale("eye_size"))
        self._draw_eye(
            surface, cx - int(hw * 0.3), cy - int(hh * 0.2),
            eye_r, eye_dx, eye_dy, colors,
        )
        self._draw_eye(
            surface, cx + int(hw * 0.05), cy - int(hh * 0.2),
            eye_r, eye_dx, eye_dy, colors,
        )

        # Mouth
        mouth_w = int(base * 0.2)
        mouth_h = int(self._mouth_open * base * 0.1)
        mouth_rect = pygame.Rect(
            cx - int(hw * 0.15), cy + int(hh * 0.25), mouth_w, max(2, mouth_h)
        )
        pygame.draw.ellipse(surface, colors["mouth"], mouth_rect)

    def _render_tadman(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
        eye_dx: float,
        eye_dy: float,
    ) -> None:
        """Tadman: tadpole humanoid — large head, stubby arms, long tail."""
        colors = self._get_colors(CreatureStage.TADMAN)
        head_r = int(base * 0.45)
        fin_s = self._genome_scale("fin_length")
        limb_s = self._genome_scale("limb_proportion")

        # Tail (behind body — genome fin_length scales)
        tail_pts = [
            (cx, cy + int(base * 0.2)),
            (cx + int(base * 0.8 * fin_s), cy + int(base * 0.5)),
            (cx + int(base * 0.9 * fin_s), cy + int(base * 0.45)),
        ]
        tail_wave = math.sin(self._time * 4.0) * 5.0
        tail_pts_anim = [
            tail_pts[0],
            (tail_pts[1][0], tail_pts[1][1] + int(tail_wave)),
            (tail_pts[2][0], tail_pts[2][1] + int(tail_wave * 0.7)),
        ]
        pygame.draw.polygon(surface, colors["tail"], tail_pts_anim)

        # Body (round head)
        pygame.draw.circle(surface, colors["body"], (cx, cy), head_r)

        # Patterns on body
        self._render_patterns(surface, cx, cy, base, colors["body"])

        # Stubby arms (genome limb_proportion scales length/width)
        arm_len = int(base * 0.3 * limb_s)
        arm_w = int(base * 0.08 * limb_s)
        arm_wave = math.sin(self._time * 2.5) * 4.0
        # Left arm
        pygame.draw.line(
            surface, colors["limb"],
            (cx - head_r + 3, cy + int(base * 0.1)),
            (cx - head_r - arm_len, cy + int(base * 0.15) + int(arm_wave)),
            max(2, arm_w),
        )
        # Right arm
        pygame.draw.line(
            surface, colors["limb"],
            (cx + head_r - 3, cy + int(base * 0.1)),
            (cx + head_r + arm_len, cy + int(base * 0.15) - int(arm_wave)),
            max(2, arm_w),
        )

        # Short legs (genome limb_proportion scales)
        leg_len = int(base * 0.2 * limb_s)
        leg_wave = math.sin(self._time * 3.0) * 3.0
        pygame.draw.line(
            surface, colors["limb"],
            (cx - int(head_r * 0.4), cy + head_r - 2),
            (cx - int(head_r * 0.5), cy + head_r + leg_len + int(leg_wave)),
            max(2, arm_w),
        )
        pygame.draw.line(
            surface, colors["limb"],
            (cx + int(head_r * 0.4), cy + head_r - 2),
            (cx + int(head_r * 0.5), cy + head_r + leg_len - int(leg_wave)),
            max(2, arm_w),
        )

        # Eyes (genome eye_size scales radius)
        eye_r = int(base * 0.12 * self._genome_scale("eye_size"))
        self._draw_eye(
            surface, cx - int(head_r * 0.35), cy - int(head_r * 0.15),
            eye_r, eye_dx, eye_dy, colors,
        )
        self._draw_eye(
            surface, cx + int(head_r * 0.35), cy - int(head_r * 0.15),
            eye_r, eye_dx, eye_dy, colors,
        )

        # Mouth
        mouth_w = int(base * 0.25)
        mouth_h = int(self._mouth_open * base * 0.12)
        mouth_rect = pygame.Rect(
            cx - mouth_w // 2, cy + int(head_r * 0.3), mouth_w, max(2, mouth_h)
        )
        pygame.draw.ellipse(surface, colors["mouth"], mouth_rect)

    def _render_frogman(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
        eye_dx: float,
        eye_dy: float,
    ) -> None:
        """Frogman: frog-man — wide mouth, bulging eyes, strong legs, upright."""
        colors = self._get_colors(CreatureStage.FROGMAN)
        body_w = int(base * 0.6)
        body_h = int(base * 0.7)
        limb_s = self._genome_scale("limb_proportion")

        # Legs (behind body — genome limb_proportion scales)
        leg_w = int(base * 0.1 * limb_s)
        leg_h = int(base * 0.45 * limb_s)
        leg_wave = math.sin(self._time * 2.0) * 4.0
        # Left leg
        pygame.draw.line(
            surface, colors["limb"],
            (cx - int(body_w * 0.4), cy + body_h // 2),
            (cx - int(body_w * 0.6), cy + body_h // 2 + leg_h + int(leg_wave)),
            max(3, leg_w),
        )
        # Right leg
        pygame.draw.line(
            surface, colors["limb"],
            (cx + int(body_w * 0.4), cy + body_h // 2),
            (cx + int(body_w * 0.6), cy + body_h // 2 + leg_h - int(leg_wave)),
            max(3, leg_w),
        )

        # Body (oval, upright)
        body_rect = pygame.Rect(cx - body_w, cy - body_h // 2, body_w * 2, body_h)
        pygame.draw.ellipse(surface, colors["body"], body_rect)

        # Patterns on body
        self._render_patterns(surface, cx, cy, base, colors["body"])

        # Belly
        belly_w = int(body_w * 0.7)
        belly_h = int(body_h * 0.5)
        belly_rect = pygame.Rect(
            cx - belly_w, cy - belly_h // 4, belly_w * 2, belly_h
        )
        pygame.draw.ellipse(surface, colors["belly"], belly_rect)

        # Arms (genome limb_proportion scales)
        arm_w = int(base * 0.08 * limb_s)
        arm_len = int(base * 0.35 * limb_s)
        arm_wave = math.sin(self._time * 1.8) * 5.0
        pygame.draw.line(
            surface, colors["limb"],
            (cx - body_w + 2, cy - int(body_h * 0.1)),
            (cx - body_w - arm_len, cy + int(arm_wave)),
            max(3, arm_w),
        )
        pygame.draw.line(
            surface, colors["limb"],
            (cx + body_w - 2, cy - int(body_h * 0.1)),
            (cx + body_w + arm_len, cy - int(arm_wave)),
            max(3, arm_w),
        )

        # Bulging eyes (on top of head — genome eye_size scales)
        eye_r = int(base * 0.16 * self._genome_scale("eye_size"))
        eye_y = cy - body_h // 2 - int(eye_r * 0.3)
        self._draw_eye(
            surface, cx - int(body_w * 0.45), eye_y,
            eye_r, eye_dx, eye_dy, colors,
        )
        self._draw_eye(
            surface, cx + int(body_w * 0.45), eye_y,
            eye_r, eye_dx, eye_dy, colors,
        )

        # Wide mouth
        mouth_w = int(base * 0.5)
        mouth_h = int(self._mouth_open * base * 0.18) + 3
        mouth_rect = pygame.Rect(
            cx - mouth_w // 2, cy + int(body_h * 0.15), mouth_w, max(3, mouth_h)
        )
        pygame.draw.ellipse(surface, colors["mouth"], mouth_rect)

    # ── Shared Drawing Helpers ───────────────────────────────────────

    def _draw_eye(
        self,
        surface: pygame.Surface,
        ex: int,
        ey: int,
        radius: int,
        dir_x: float,
        dir_y: float,
        colors: dict[str, tuple[int, int, int]],
    ) -> None:
        """Draw an eye with pupil tracking toward (dir_x, dir_y).

        Args:
            surface: Pygame surface.
            ex, ey: Eye center position.
            radius: Eye radius.
            dir_x, dir_y: Normalized direction for pupil offset.
            colors: Color dict with "eye_white" and "eye" keys.
        """
        if self._is_blinking:
            # Draw closed eye (horizontal line)
            pygame.draw.line(
                surface, colors["eye"],
                (ex - radius, ey), (ex + radius, ey), 2,
            )
            return

        # Eye white
        pygame.draw.circle(surface, colors["eye_white"], (ex, ey), radius)

        # Pupil offset (follows mouse)
        pupil_r = max(2, radius // 2)
        max_offset = radius - pupil_r - 1
        px = ex + int(dir_x * max_offset)
        py = ey + int(dir_y * max_offset)
        pygame.draw.circle(surface, colors["eye"], (px, py), pupil_r)

    def _render_sleep_zs(
        self,
        surface: pygame.Surface,
        cx: int,
        cy: int,
        base: float,
    ) -> None:
        """Draw floating 'Z' letters above sleeping creature."""
        z_phase = self._time * 1.2
        for i in range(3):
            offset = i * 0.8
            alpha = max(0.0, min(1.0, 1.0 - ((z_phase + offset) % 3.0) / 3.0))
            if alpha < 0.05:
                continue
            zx = cx + int(base * 0.3) + i * 12
            zy = cy - int(base * 0.7) - int(((z_phase + offset) % 3.0) * 15)
            size = 8 + i * 3
            brightness = int(200 * alpha)
            color = (brightness, brightness, min(255, brightness + 40))

            # Draw a Z shape with lines
            pygame.draw.line(
                surface, color, (zx, zy), (zx + size, zy), 2
            )
            pygame.draw.line(
                surface, color, (zx + size, zy), (zx, zy + size), 2
            )
            pygame.draw.line(
                surface, color, (zx, zy + size), (zx + size, zy + size), 2
            )

    # ── Eating Animation ─────────────────────────────────────────────

    def render_eating_effect(
        self, surface: pygame.Surface, food_x: float, food_y: float
    ) -> None:
        """Draw eating animation: food particles moving toward creature mouth.

        Args:
            surface: Pygame surface.
            food_x, food_y: Position of the food.
        """
        # Simple food particles converging on creature
        phase = (self._time * 5.0) % 1.0
        for i in range(4):
            t = (phase + i * 0.25) % 1.0
            px = int(food_x + (self.position.x - food_x) * t)
            py = int(food_y + (self.position.y - food_y) * t)
            radius = max(1, int(3 * (1.0 - t)))
            color = (200, 160, 80)
            pygame.draw.circle(surface, color, (px, py), radius)
