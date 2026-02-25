"""Interactive elements - feeding, temperature controls, glass tapping.

Handles mouse-based tank interactions: click to feed, temperature up/down
buttons, tap glass for creature reactions, clean tank, and drain/fill
for evolution transitions. Visual feedback via ripple effects and food
drop animations.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from enum import Enum

import pygame

from seaman_brain.config import EnvironmentConfig, GUIConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.care import CareResult, TankCareEngine
from seaman_brain.needs.feeding import FeedingEngine, FeedingResult, FoodType

logger = logging.getLogger(__name__)


# ── Enums ─────────────────────────────────────────────────────────────


class InteractionType(Enum):
    """Types of mouse interactions."""

    FEED = "feed"
    TAP_GLASS = "tap_glass"
    TEMP_UP = "temp_up"
    TEMP_DOWN = "temp_down"
    CLEAN = "clean"
    DRAIN = "drain"


class TapReaction(Enum):
    """Creature reactions to glass tapping."""

    STARTLED = "startled"
    ANNOYED = "annoyed"
    CURIOUS = "curious"


# ── Visual Effects ────────────────────────────────────────────────────


@dataclass
class RippleEffect:
    """A circular ripple expanding from a click point."""

    x: float
    y: float
    radius: float = 5.0
    max_radius: float = 40.0
    speed: float = 60.0
    alpha: float = 1.0
    color: tuple[int, int, int] = (150, 200, 255)

    @property
    def alive(self) -> bool:
        """Whether this ripple is still visible."""
        return self.alpha > 0.01

    def update(self, dt: float) -> None:
        """Expand the ripple and fade it out."""
        self.radius += self.speed * dt
        progress = min(1.0, self.radius / self.max_radius)
        self.alpha = max(0.0, 1.0 - progress)


@dataclass
class FoodDropEffect:
    """A falling food particle animation."""

    x: float
    y: float
    target_y: float
    speed: float = 80.0
    food_type: FoodType = FoodType.PELLET
    alpha: float = 1.0
    _landed: bool = False

    @property
    def alive(self) -> bool:
        """Whether this food drop is still visible."""
        return self.alpha > 0.01

    @property
    def landed(self) -> bool:
        """Whether the food has reached its target."""
        return self._landed

    def update(self, dt: float) -> None:
        """Move food downward toward target, then fade."""
        if not self._landed:
            self.y += self.speed * dt
            if self.y >= self.target_y:
                self.y = self.target_y
                self._landed = True
        else:
            self.alpha = max(0.0, self.alpha - dt * 2.0)


# ── Food Colors ───────────────────────────────────────────────────────

_FOOD_COLORS: dict[FoodType, tuple[int, int, int]] = {
    FoodType.PELLET: (160, 120, 80),
    FoodType.WORM: (180, 100, 100),
    FoodType.INSECT: (80, 120, 60),
    FoodType.NAUTILUS: (200, 180, 140),
}

# ── Button Layout ─────────────────────────────────────────────────────

_BUTTON_WIDTH = 32
_BUTTON_HEIGHT = 24
_BUTTON_MARGIN = 4
_BUTTON_BG = (30, 45, 65)
_BUTTON_BG_HOVER = (50, 70, 100)
_BUTTON_BORDER = (80, 110, 150)
_BUTTON_TEXT_COLOR = (200, 220, 240)
_TEMP_STEP = 1.0  # degrees per click

# Food menu layout
_FOOD_MENU_ITEM_H = 28
_FOOD_MENU_WIDTH = 120
_FOOD_MENU_BG = (20, 35, 55, 220)
_FOOD_MENU_HOVER = (40, 60, 90)
_FOOD_MENU_TEXT = (200, 220, 240)
_FOOD_MENU_DISABLED = (100, 110, 120)


@dataclass
class _Button:
    """A clickable UI button."""

    x: int
    y: int
    width: int
    height: int
    label: str
    action: InteractionType
    hover: bool = False

    def contains(self, mx: int, my: int) -> bool:
        """Check if a point is inside this button."""
        return (
            self.x <= mx <= self.x + self.width
            and self.y <= my <= self.y + self.height
        )


@dataclass
class InteractionResult:
    """Result of a user interaction.

    Fields:
        interaction_type: What kind of interaction occurred.
        message: Human-readable result description.
        tap_reaction: Creature reaction for glass taps (None otherwise).
        feeding_result: Detailed feeding result (None if not a feed).
        care_result: Detailed care result (None if not a care action).
    """

    interaction_type: InteractionType
    message: str
    tap_reaction: TapReaction | None = None
    feeding_result: FeedingResult | None = None
    care_result: CareResult | None = None


# ── Interaction Manager ───────────────────────────────────────────────


class InteractionManager:
    """Handles mouse-based tank interactions.

    Manages feeding (with food selection menu), temperature controls,
    glass tapping, tank cleaning, and drain/fill. Produces visual
    feedback (ripples, food drops) and delegates to FeedingEngine
    and TankCareEngine for game logic.

    Attributes:
        food_menu_open: Whether the food selection menu is visible.
    """

    def __init__(
        self,
        gui_config: GUIConfig | None = None,
        env_config: EnvironmentConfig | None = None,
        needs_config: NeedsConfig | None = None,
    ) -> None:
        self._gui_config = gui_config or GUIConfig()
        self._env_config = env_config or EnvironmentConfig()
        self._needs_config = needs_config or NeedsConfig()

        # Engines
        self._feeding_engine = FeedingEngine(config=self._needs_config)
        self._care_engine = TankCareEngine(
            env_config=self._env_config, needs_config=self._needs_config
        )

        # Visual effects
        self._ripples: list[RippleEffect] = []
        self._food_drops: list[FoodDropEffect] = []

        # Food menu state
        self.food_menu_open: bool = False
        self._food_menu_x: int = 0
        self._food_menu_y: int = 0
        self._food_menu_items: list[FoodType] = []

        # Buttons (built lazily)
        self._buttons: list[_Button] = []
        self._buttons_built = False

        # Tank render area (set externally)
        self._tank_x = 0
        self._tank_y = 45
        self._tank_w = self._gui_config.window_width
        self._tank_h = self._gui_config.window_height - 45

        # Tap tracking for reaction variety
        self._recent_tap_count = 0
        self._tap_cooldown = 0.0

        # Font (lazy-initialized)
        self._font: pygame.font.Font | None = None

    @property
    def feeding_engine(self) -> FeedingEngine:
        """The feeding engine for external access."""
        return self._feeding_engine

    @property
    def care_engine(self) -> TankCareEngine:
        """The tank care engine for external access."""
        return self._care_engine

    @property
    def ripples(self) -> list[RippleEffect]:
        """Active ripple effects."""
        return self._ripples

    @property
    def food_drops(self) -> list[FoodDropEffect]:
        """Active food drop effects."""
        return self._food_drops

    def set_tank_area(self, x: int, y: int, w: int, h: int) -> None:
        """Set the tank rendering area for hit detection.

        Args:
            x: Left edge of the tank area.
            y: Top edge of the tank area.
            w: Width of the tank area.
            h: Height of the tank area.
        """
        self._tank_x = x
        self._tank_y = y
        self._tank_w = w
        self._tank_h = h
        self._buttons_built = False

    def _ensure_font(self) -> None:
        """Initialize font if not yet done."""
        if self._font is None:
            try:
                self._font = pygame.font.SysFont("consolas", 12)
            except Exception:
                self._font = pygame.font.Font(None, 12)

    def _build_buttons(self) -> None:
        """Build the button layout based on current tank area."""
        if self._buttons_built:
            return
        self._buttons_built = True
        self._buttons.clear()

        # Temperature buttons — right edge of tank
        btn_x = self._tank_x + self._tank_w - _BUTTON_WIDTH - _BUTTON_MARGIN
        btn_y = self._tank_y + _BUTTON_MARGIN

        self._buttons.append(_Button(
            x=btn_x, y=btn_y,
            width=_BUTTON_WIDTH, height=_BUTTON_HEIGHT,
            label="+", action=InteractionType.TEMP_UP,
        ))
        self._buttons.append(_Button(
            x=btn_x, y=btn_y + _BUTTON_HEIGHT + _BUTTON_MARGIN,
            width=_BUTTON_WIDTH, height=_BUTTON_HEIGHT,
            label="-", action=InteractionType.TEMP_DOWN,
        ))

        # Clean button — bottom right
        clean_y = self._tank_y + self._tank_h - _BUTTON_HEIGHT - _BUTTON_MARGIN
        clean_w = 50
        self._buttons.append(_Button(
            x=btn_x - (clean_w - _BUTTON_WIDTH), y=clean_y,
            width=clean_w, height=_BUTTON_HEIGHT,
            label="Clean", action=InteractionType.CLEAN,
        ))

        # Drain button — bottom right, above clean
        drain_y = clean_y - _BUTTON_HEIGHT - _BUTTON_MARGIN
        self._buttons.append(_Button(
            x=btn_x - (clean_w - _BUTTON_WIDTH), y=drain_y,
            width=clean_w, height=_BUTTON_HEIGHT,
            label="Drain", action=InteractionType.DRAIN,
        ))

    def update(self, dt: float) -> None:
        """Update visual effects and cooldowns.

        Args:
            dt: Delta time in seconds since last frame.
        """
        # Update ripples
        alive_ripples: list[RippleEffect] = []
        for r in self._ripples:
            r.update(dt)
            if r.alive:
                alive_ripples.append(r)
        self._ripples = alive_ripples

        # Update food drops
        alive_drops: list[FoodDropEffect] = []
        for f in self._food_drops:
            f.update(dt)
            if f.alive:
                alive_drops.append(f)
        self._food_drops = alive_drops

        # Tap cooldown
        if self._tap_cooldown > 0:
            self._tap_cooldown -= dt
            if self._tap_cooldown <= 0:
                self._recent_tap_count = 0

    def handle_click(
        self,
        mx: int,
        my: int,
        creature: CreatureState,
        tank: TankEnvironment,
    ) -> InteractionResult | None:
        """Handle a mouse click event.

        Checks buttons first, then food menu, then tank area for taps/feeding.

        Args:
            mx: Mouse x position.
            my: Mouse y position.
            creature: Current creature state (mutated by feeding).
            tank: Current tank state (mutated by care actions).

        Returns:
            InteractionResult if an interaction occurred, None otherwise.
        """
        self._build_buttons()

        # Check food menu first (if open)
        if self.food_menu_open:
            result = self._handle_food_menu_click(mx, my, creature)
            if result is not None:
                return result
            # Clicking outside menu closes it
            self.food_menu_open = False
            return None

        # Check buttons
        for btn in self._buttons:
            if btn.contains(mx, my):
                return self._handle_button(btn, creature, tank)

        # Check tank area — tap glass or open food menu
        if self._is_in_tank(mx, my):
            return self._handle_tank_click(mx, my, creature, tank)

        return None

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update button hover states.

        Args:
            mx: Mouse x position.
            my: Mouse y position.
        """
        self._build_buttons()
        for btn in self._buttons:
            btn.hover = btn.contains(mx, my)

    def _is_in_tank(self, mx: int, my: int) -> bool:
        """Check if a point is inside the tank area."""
        return (
            self._tank_x <= mx <= self._tank_x + self._tank_w
            and self._tank_y <= my <= self._tank_y + self._tank_h
        )

    def _is_near_wall(self, mx: int, my: int) -> bool:
        """Check if click is near a tank wall (edge region)."""
        margin = 30
        near_left = mx < self._tank_x + margin
        near_right = mx > self._tank_x + self._tank_w - margin
        near_top = my < self._tank_y + margin
        near_bottom = my > self._tank_y + self._tank_h - margin
        return near_left or near_right or near_top or near_bottom

    def _handle_tank_click(
        self,
        mx: int,
        my: int,
        creature: CreatureState,
        tank: TankEnvironment,
    ) -> InteractionResult:
        """Handle a click inside the tank area.

        If clicking near walls, it's a glass tap. Otherwise, open the food menu.
        """
        # Add ripple at click point
        self._ripples.append(RippleEffect(x=float(mx), y=float(my)))

        if self._is_near_wall(mx, my):
            return self._handle_glass_tap(mx, my, creature)

        # Open food selection menu
        available = self._feeding_engine.get_available_foods(creature.stage)
        if available:
            self._food_menu_items = available
            self._food_menu_x = mx
            self._food_menu_y = my
            self.food_menu_open = True
            return InteractionResult(
                interaction_type=InteractionType.FEED,
                message="Choose food to drop...",
            )

        return InteractionResult(
            interaction_type=InteractionType.FEED,
            message="No food available for this stage.",
        )

    def _handle_glass_tap(
        self, mx: int, my: int, creature: CreatureState
    ) -> InteractionResult:
        """Handle a glass tap interaction."""
        self._recent_tap_count += 1
        self._tap_cooldown = 3.0

        # Choose reaction based on recent taps and creature mood
        reaction = self._choose_tap_reaction(creature)

        creature.interaction_count += 1

        messages = {
            TapReaction.STARTLED: f"The {creature.stage.value} flinches!",
            TapReaction.ANNOYED: f"The {creature.stage.value} glares at you.",
            TapReaction.CURIOUS: (
                f"The {creature.stage.value} swims closer to investigate."
            ),
        }

        return InteractionResult(
            interaction_type=InteractionType.TAP_GLASS,
            message=messages[reaction],
            tap_reaction=reaction,
        )

    def _choose_tap_reaction(self, creature: CreatureState) -> TapReaction:
        """Choose creature reaction to glass tap based on mood and tap count."""
        if self._recent_tap_count >= 3:
            return TapReaction.ANNOYED

        if creature.trust_level > 0.6:
            return TapReaction.CURIOUS

        if creature.trust_level < 0.3:
            return TapReaction.STARTLED

        return random.choice([TapReaction.STARTLED, TapReaction.CURIOUS])

    def _handle_food_menu_click(
        self,
        mx: int,
        my: int,
        creature: CreatureState,
    ) -> InteractionResult | None:
        """Handle a click when the food menu is open.

        Returns None if click is outside the menu.
        """
        menu_x = self._food_menu_x
        menu_y = self._food_menu_y
        menu_w = _FOOD_MENU_WIDTH
        menu_h = len(self._food_menu_items) * _FOOD_MENU_ITEM_H

        if not (menu_x <= mx <= menu_x + menu_w and menu_y <= my <= menu_y + menu_h):
            return None

        # Find which food item was clicked
        idx = (my - menu_y) // _FOOD_MENU_ITEM_H
        if 0 <= idx < len(self._food_menu_items):
            food = self._food_menu_items[idx]
            self.food_menu_open = False
            return self._do_feed(food, creature)

        self.food_menu_open = False
        return None

    def _do_feed(
        self, food_type: FoodType, creature: CreatureState
    ) -> InteractionResult:
        """Execute a feeding action and spawn visual effects."""
        result = self._feeding_engine.feed(creature, food_type)

        if result.success:
            # Spawn food drop animation at menu position
            target_y = float(self._tank_y + self._tank_h * 0.7)
            self._food_drops.append(FoodDropEffect(
                x=float(self._food_menu_x),
                y=float(self._food_menu_y),
                target_y=target_y,
                food_type=food_type,
            ))
            creature.interaction_count += 1

        return InteractionResult(
            interaction_type=InteractionType.FEED,
            message=result.message,
            feeding_result=result,
        )

    def _handle_button(
        self,
        btn: _Button,
        creature: CreatureState,
        tank: TankEnvironment,
    ) -> InteractionResult:
        """Handle a button click."""
        if btn.action == InteractionType.TEMP_UP:
            care_result = self._care_engine.adjust_temperature(
                tank, _TEMP_STEP, creature
            )
            return InteractionResult(
                interaction_type=InteractionType.TEMP_UP,
                message=care_result.message,
                care_result=care_result,
            )

        if btn.action == InteractionType.TEMP_DOWN:
            care_result = self._care_engine.adjust_temperature(
                tank, -_TEMP_STEP, creature
            )
            return InteractionResult(
                interaction_type=InteractionType.TEMP_DOWN,
                message=care_result.message,
                care_result=care_result,
            )

        if btn.action == InteractionType.CLEAN:
            care_result = self._care_engine.clean_tank(tank)
            if care_result.success:
                # Add ripple at button center for visual feedback
                bx = btn.x + btn.width // 2
                by = btn.y + btn.height // 2
                self._ripples.append(RippleEffect(
                    x=float(bx), y=float(by), color=(100, 220, 180)
                ))
            return InteractionResult(
                interaction_type=InteractionType.CLEAN,
                message=care_result.message,
                care_result=care_result,
            )

        if btn.action == InteractionType.DRAIN:
            # Toggle drain/fill based on current state
            if tank.water_level > 0.0:
                care_result = self._care_engine.drain_tank(tank, creature)
            else:
                care_result = self._care_engine.fill_tank(tank, creature)
            return InteractionResult(
                interaction_type=InteractionType.DRAIN,
                message=care_result.message,
                care_result=care_result,
            )

        return InteractionResult(
            interaction_type=btn.action,
            message="Unknown button action.",
        )

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface) -> None:
        """Render interactive UI elements and visual effects.

        Args:
            surface: Pygame surface to draw on.
        """
        self._build_buttons()
        self._ensure_font()

        # Draw buttons
        self._render_buttons(surface)

        # Draw ripple effects
        self._render_ripples(surface)

        # Draw food drops
        self._render_food_drops(surface)

        # Draw food menu if open
        if self.food_menu_open:
            self._render_food_menu(surface)

    def _render_buttons(self, surface: pygame.Surface) -> None:
        """Draw interactive buttons."""
        if self._font is None:
            return

        for btn in self._buttons:
            bg = _BUTTON_BG_HOVER if btn.hover else _BUTTON_BG
            rect = pygame.Rect(btn.x, btn.y, btn.width, btn.height)
            pygame.draw.rect(surface, bg, rect)
            pygame.draw.rect(surface, _BUTTON_BORDER, rect, 1)

            text_surf = self._font.render(btn.label, True, _BUTTON_TEXT_COLOR)
            tx = btn.x + (btn.width - text_surf.get_width()) // 2
            ty = btn.y + (btn.height - text_surf.get_height()) // 2
            surface.blit(text_surf, (tx, ty))

    def _render_ripples(self, surface: pygame.Surface) -> None:
        """Draw expanding ripple effects."""
        for r in self._ripples:
            alpha_byte = int(r.alpha * 180)
            if alpha_byte < 2:
                continue
            color = (
                int(r.color[0] * r.alpha),
                int(r.color[1] * r.alpha),
                int(r.color[2] * r.alpha),
            )
            pygame.draw.circle(
                surface, color, (int(r.x), int(r.y)), int(r.radius), 1
            )

    def _render_food_drops(self, surface: pygame.Surface) -> None:
        """Draw falling food particles."""
        for f in self._food_drops:
            color = _FOOD_COLORS.get(f.food_type, (160, 120, 80))
            alpha_color = (
                int(color[0] * f.alpha),
                int(color[1] * f.alpha),
                int(color[2] * f.alpha),
            )
            pygame.draw.circle(surface, alpha_color, (int(f.x), int(f.y)), 4)

    def _render_food_menu(self, surface: pygame.Surface) -> None:
        """Draw the food selection popup menu."""
        if self._font is None:
            return

        x = self._food_menu_x
        y = self._food_menu_y
        h = len(self._food_menu_items) * _FOOD_MENU_ITEM_H

        # Background
        menu_surf = pygame.Surface((_FOOD_MENU_WIDTH, h), pygame.SRCALPHA)
        menu_surf.fill(_FOOD_MENU_BG)
        surface.blit(menu_surf, (x, y))

        # Items
        for i, food in enumerate(self._food_menu_items):
            iy = y + i * _FOOD_MENU_ITEM_H
            label = food.value.capitalize()
            text_surf = self._font.render(label, True, _FOOD_MENU_TEXT)
            surface.blit(text_surf, (x + 8, iy + 6))

        # Border
        pygame.draw.rect(
            surface, _BUTTON_BORDER,
            pygame.Rect(x, y, _FOOD_MENU_WIDTH, h), 1
        )
