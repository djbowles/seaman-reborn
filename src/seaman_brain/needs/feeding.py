"""Feeding mechanics - food types, effects, and validation.

Provides FoodType enum, FeedingResult dataclass, and FeedingEngine class
for managing creature feeding. Different food types have different nutritional
values and mood effects. Stage-appropriate food validation prevents wrong
foods. Overfeeding and cooldown mechanics add depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from seaman_brain.config import NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.types import CreatureStage


class FoodType(Enum):
    """Available food types for the creature."""

    PELLET = "pellet"
    WORM = "worm"
    INSECT = "insect"
    NAUTILUS = "nautilus"


# Nutritional value per food type (how much hunger is reduced, 0-1 scale)
FOOD_NUTRITION: dict[FoodType, float] = {
    FoodType.PELLET: 0.15,
    FoodType.WORM: 0.25,
    FoodType.INSECT: 0.30,
    FoodType.NAUTILUS: 0.20,
}

# Mood effect per food type (positive = happier)
FOOD_MOOD_EFFECT: dict[FoodType, float] = {
    FoodType.PELLET: 0.02,
    FoodType.WORM: 0.05,
    FoodType.INSECT: 0.08,
    FoodType.NAUTILUS: 0.10,
}

# Which food types each stage can eat
STAGE_FOODS: dict[CreatureStage, set[FoodType]] = {
    CreatureStage.MUSHROOMER: {FoodType.NAUTILUS},
    CreatureStage.GILLMAN: {FoodType.PELLET, FoodType.WORM},
    CreatureStage.PODFISH: {FoodType.PELLET, FoodType.WORM, FoodType.INSECT},
    CreatureStage.TADMAN: {FoodType.PELLET, FoodType.WORM, FoodType.INSECT},
    CreatureStage.FROGMAN: {FoodType.WORM, FoodType.INSECT},
}


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


@dataclass
class FeedingResult:
    """Result of a feeding attempt.

    Fields:
        success: Whether the feeding was accepted.
        hunger_change: Change in hunger (negative = less hungry).
        mood_change: Change in mood/trust.
        health_change: Change in health (negative for overfeeding penalty).
        message: Descriptive message about the feeding outcome.
    """

    success: bool
    hunger_change: float
    mood_change: float
    health_change: float
    message: str


class FeedingEngine:
    """Engine that handles creature feeding mechanics.

    Validates food type against creature stage, enforces cooldown between
    feedings, applies overfeeding penalties, and calculates nutritional effects.

    Args:
        config: Needs configuration with feeding_cooldown_seconds.
        now_func: Injectable time function for testing (default: datetime.now(UTC)).
    """

    def __init__(
        self,
        config: NeedsConfig | None = None,
        now_func: callable | None = None,
    ) -> None:
        self._config = config or NeedsConfig()
        self._now_func = now_func or (lambda: datetime.now(UTC))

    @property
    def config(self) -> NeedsConfig:
        """Current needs configuration."""
        return self._config

    def feed(
        self,
        creature_state: CreatureState,
        food_type: FoodType,
    ) -> FeedingResult:
        """Attempt to feed the creature.

        Checks in order:
        1. Cooldown: rejects if too soon since last feeding.
        2. Stage validation: rejects if food is wrong for current stage.
        3. Overfeeding: applies penalty if creature is already full (hunger < 0.1).
        4. Normal feeding: reduces hunger and boosts mood.

        Mutates creature_state on success (hunger, last_fed).

        Args:
            creature_state: Current creature state (mutated on success).
            food_type: Type of food to feed.

        Returns:
            FeedingResult with outcome details.
        """
        now = self._now_func()

        # 1. Check cooldown
        cooldown_result = self._check_cooldown(creature_state, now)
        if cooldown_result is not None:
            return cooldown_result

        # 2. Check stage-appropriate food
        stage_result = self._check_stage_food(creature_state.stage, food_type)
        if stage_result is not None:
            return stage_result

        # 3. Check overfeeding
        if creature_state.hunger < 0.1:
            return self._apply_overfeeding(creature_state, food_type, now)

        # 4. Normal feeding
        return self._apply_normal_feeding(creature_state, food_type, now)

    def get_available_foods(self, stage: CreatureStage) -> list[FoodType]:
        """Return list of food types available for the given stage.

        Args:
            stage: Current creature stage.

        Returns:
            Sorted list of available FoodType values.
        """
        foods = STAGE_FOODS.get(stage, set())
        return sorted(foods, key=lambda f: f.value)

    def is_on_cooldown(self, creature_state: CreatureState) -> bool:
        """Check if feeding is on cooldown.

        Args:
            creature_state: Current creature state.

        Returns:
            True if cooldown is still active.
        """
        now = self._now_func()
        elapsed = (now - creature_state.last_fed).total_seconds()
        return elapsed < self._config.feeding_cooldown_seconds

    def cooldown_remaining(self, creature_state: CreatureState) -> float:
        """Get remaining cooldown time in seconds.

        Args:
            creature_state: Current creature state.

        Returns:
            Seconds remaining (0.0 if cooldown expired).
        """
        now = self._now_func()
        elapsed = (now - creature_state.last_fed).total_seconds()
        remaining = self._config.feeding_cooldown_seconds - elapsed
        return max(0.0, remaining)

    def _check_cooldown(
        self,
        creature_state: CreatureState,
        now: datetime,
    ) -> FeedingResult | None:
        """Check if feeding cooldown has expired.

        Returns FeedingResult if on cooldown, None if OK to proceed.
        """
        elapsed = (now - creature_state.last_fed).total_seconds()
        if elapsed < self._config.feeding_cooldown_seconds:
            remaining = self._config.feeding_cooldown_seconds - elapsed
            return FeedingResult(
                success=False,
                hunger_change=0.0,
                mood_change=0.0,
                health_change=0.0,
                message=f"Too soon! Wait {remaining:.0f} more seconds before feeding again.",
            )
        return None

    def _check_stage_food(
        self,
        stage: CreatureStage,
        food_type: FoodType,
    ) -> FeedingResult | None:
        """Check if food type is appropriate for creature's stage.

        Returns FeedingResult if wrong food, None if OK to proceed.
        """
        allowed = STAGE_FOODS.get(stage, set())
        if food_type not in allowed:
            allowed_names = ", ".join(f.value for f in sorted(allowed, key=lambda f: f.value))
            return FeedingResult(
                success=False,
                hunger_change=0.0,
                mood_change=-0.05,
                health_change=0.0,
                message=(
                    f"The {stage.value} won't eat {food_type.value}! "
                    f"Try: {allowed_names}."
                ),
            )
        return None

    def _apply_overfeeding(
        self,
        creature_state: CreatureState,
        food_type: FoodType,
        now: datetime,
    ) -> FeedingResult:
        """Apply overfeeding penalty when creature is already full."""
        health_penalty = -0.05
        mood_penalty = -0.03

        # Still update last_fed to reset cooldown
        creature_state.last_fed = now
        creature_state.health = _clamp(creature_state.health + health_penalty)

        return FeedingResult(
            success=True,
            hunger_change=0.0,
            mood_change=mood_penalty,
            health_change=health_penalty,
            message=(
                f"The {creature_state.stage.value} is already full! "
                "Overfeeding causes discomfort."
            ),
        )

    def _apply_normal_feeding(
        self,
        creature_state: CreatureState,
        food_type: FoodType,
        now: datetime,
    ) -> FeedingResult:
        """Apply normal feeding effects."""
        nutrition = FOOD_NUTRITION.get(food_type, 0.15)
        mood_boost = FOOD_MOOD_EFFECT.get(food_type, 0.02)

        hunger_change = -nutrition
        new_hunger = _clamp(creature_state.hunger + hunger_change)

        # Apply changes to state
        creature_state.hunger = new_hunger
        creature_state.last_fed = now

        return FeedingResult(
            success=True,
            hunger_change=hunger_change,
            mood_change=mood_boost,
            health_change=0.0,
            message=f"The {creature_state.stage.value} ate the {food_type.value}.",
        )
