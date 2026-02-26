"""Tank care mechanics - temperature control, cleaning, maintenance.

Provides TankCareEngine class coordinating all tank maintenance actions.
Temperature adjustment respects stage-specific optimal ranges. Cleaning
takes time (not instant). Stage-specific requirements like terrarium
sprinklers for Frogman are enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.types import CreatureStage


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


class CareAction(Enum):
    """Types of tank care actions the player can perform."""

    ADJUST_TEMPERATURE = "adjust_temperature"
    CLEAN_TANK = "clean_tank"
    AERATE = "aerate"
    SPRINKLE = "sprinkle"
    DRAIN = "drain"
    FILL = "fill"


# Stage-specific optimal temperature ranges override the global config
# Some stages prefer warmer/cooler temperatures
STAGE_OPTIMAL_TEMPS: dict[CreatureStage, tuple[float, float]] = {
    CreatureStage.MUSHROOMER: (22.0, 26.0),
    CreatureStage.GILLMAN: (20.0, 28.0),
    CreatureStage.PODFISH: (18.0, 26.0),
    CreatureStage.TADMAN: (20.0, 30.0),
    CreatureStage.FROGMAN: (22.0, 32.0),
}

# Cleaning duration per pass in seconds — cleaning is not instant
CLEANING_DURATION_SECONDS: float = 5.0

# How much cleanliness is restored per clean action (partial, not full)
CLEANING_AMOUNT: float = 0.4

# Aerator oxygen boost for aquarium
AERATOR_OXYGEN_BOOST: float = 0.3

# Aerator cooldown in seconds (same rhythm as cleaning)
AERATOR_COOLDOWN_SECONDS: float = 5.0

# Sprinkler moisture boost for terrarium
SPRINKLER_MOISTURE_BOOST: float = 0.3


@dataclass
class CareResult:
    """Result of a tank care action.

    Fields:
        success: Whether the action was accepted.
        action: The care action that was attempted.
        message: Descriptive message about the outcome.
        warnings: Any remaining tank warnings after the action.
    """

    success: bool
    action: CareAction
    message: str
    warnings: list[str]


class TankCareEngine:
    """Engine coordinating all tank maintenance actions.

    Provides higher-level care operations on top of TankEnvironment,
    with stage-specific validation, cleaning cooldowns, and warning
    generation.

    Args:
        env_config: Environment configuration for temperature bounds.
        needs_config: Needs configuration for care thresholds.
        now_func: Injectable time function for testing.
    """

    def __init__(
        self,
        env_config: EnvironmentConfig | None = None,
        needs_config: NeedsConfig | None = None,
        now_func: callable | None = None,
    ) -> None:
        self._env_config = env_config or EnvironmentConfig()
        self._needs_config = needs_config or NeedsConfig()
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._last_clean_time: datetime | None = None
        self._last_aerate_time: datetime | None = None
        self._cleaning_in_progress: bool = False

    @property
    def env_config(self) -> EnvironmentConfig:
        """Current environment configuration."""
        return self._env_config

    @property
    def needs_config(self) -> NeedsConfig:
        """Current needs configuration."""
        return self._needs_config

    def adjust_temperature(
        self,
        tank: TankEnvironment,
        delta: float,
        creature_state: CreatureState,
    ) -> CareResult:
        """Adjust tank temperature by a delta amount.

        Respects stage-specific optimal ranges and provides feedback
        about whether the new temperature is within optimal bounds.

        Args:
            tank: Tank environment to modify (mutated in place).
            delta: Temperature change in Celsius (positive = warmer).
            creature_state: Current creature state for stage-specific ranges.

        Returns:
            CareResult with outcome and any temperature warnings.
        """
        old_temp = tank.temperature
        tank.adjust_temperature(delta, self._env_config)
        new_temp = tank.temperature

        # Check against stage-specific optimal range
        stage_min, stage_max = STAGE_OPTIMAL_TEMPS.get(
            creature_state.stage, (self._env_config.optimal_temp_min,
                                   self._env_config.optimal_temp_max)
        )

        if stage_min <= new_temp <= stage_max:
            direction = "warmer" if delta > 0 else "cooler"
            message = (
                f"Temperature adjusted {direction} to {new_temp:.1f}°C. "
                "Within optimal range."
            )
        elif new_temp < self._env_config.lethal_temp_min:
            message = (
                f"Temperature at {new_temp:.1f}°C — DANGEROUSLY COLD! "
                f"The {creature_state.stage.value} needs {stage_min:.0f}-{stage_max:.0f}°C."
            )
        elif new_temp > self._env_config.lethal_temp_max:
            message = (
                f"Temperature at {new_temp:.1f}°C — DANGEROUSLY HOT! "
                f"The {creature_state.stage.value} needs {stage_min:.0f}-{stage_max:.0f}°C."
            )
        else:
            direction = "warmer" if new_temp > old_temp else "cooler"
            message = (
                f"Temperature adjusted {direction} to {new_temp:.1f}°C. "
                f"The {creature_state.stage.value} prefers {stage_min:.0f}-{stage_max:.0f}°C."
            )

        warnings = self.get_tank_warnings(tank, creature_state)

        return CareResult(
            success=True,
            action=CareAction.ADJUST_TEMPERATURE,
            message=message,
            warnings=warnings,
        )

    def clean_tank(
        self,
        tank: TankEnvironment,
    ) -> CareResult:
        """Clean the tank, partially restoring cleanliness.

        Cleaning is not instant — it restores a portion of cleanliness
        per action. Multiple cleans may be needed for a very dirty tank.
        A cooldown prevents spam-cleaning.

        Args:
            tank: Tank environment to clean (mutated in place).

        Returns:
            CareResult with outcome and remaining warnings.
        """
        now = self._now_func()

        # Check cleaning cooldown
        if self._last_clean_time is not None:
            elapsed = (now - self._last_clean_time).total_seconds()
            if elapsed < CLEANING_DURATION_SECONDS:
                remaining = CLEANING_DURATION_SECONDS - elapsed
                return CareResult(
                    success=False,
                    action=CareAction.CLEAN_TANK,
                    message=(
                        f"Still cleaning! Wait {remaining:.0f} more seconds."
                    ),
                    warnings=self.get_tank_warnings(tank),
                )

        old_cleanliness = tank.cleanliness
        new_cleanliness = _clamp(old_cleanliness + CLEANING_AMOUNT)
        tank.cleanliness = new_cleanliness
        self._last_clean_time = now

        if new_cleanliness >= 1.0:
            message = "Tank is now spotless!"
        elif new_cleanliness >= 0.7:
            message = f"Tank cleaned to {new_cleanliness:.0%}. Looking good."
        else:
            message = (
                f"Tank cleaned to {new_cleanliness:.0%}. "
                "Still needs more cleaning."
            )

        warnings = self.get_tank_warnings(tank)

        return CareResult(
            success=True,
            action=CareAction.CLEAN_TANK,
            message=message,
            warnings=warnings,
        )

    def aerate_tank(
        self,
        tank: TankEnvironment,
    ) -> CareResult:
        """Run the aerator to boost oxygen in the aquarium.

        Only works in aquarium mode. Terrarium creatures should use
        the sprinkler instead. Has a short cooldown to prevent spam.

        Args:
            tank: Tank environment to aerate (mutated in place).

        Returns:
            CareResult with outcome and remaining warnings.
        """
        if tank.environment_type != EnvironmentType.AQUARIUM:
            return CareResult(
                success=False,
                action=CareAction.AERATE,
                message="Aerator only works in aquarium mode! Use sprinkler instead.",
                warnings=self.get_tank_warnings(tank),
            )

        now = self._now_func()

        if self._last_aerate_time is not None:
            elapsed = (now - self._last_aerate_time).total_seconds()
            if elapsed < AERATOR_COOLDOWN_SECONDS:
                remaining = AERATOR_COOLDOWN_SECONDS - elapsed
                return CareResult(
                    success=False,
                    action=CareAction.AERATE,
                    message=f"Aerator cycling! Wait {remaining:.0f} more seconds.",
                    warnings=self.get_tank_warnings(tank),
                )

        old_oxygen = tank.oxygen_level
        tank.oxygen_level = _clamp(old_oxygen + AERATOR_OXYGEN_BOOST)
        self._last_aerate_time = now

        new_oxygen = tank.oxygen_level
        if new_oxygen >= 1.0:
            message = "Oxygen fully restored! Bubbles everywhere."
        elif new_oxygen >= 0.7:
            message = f"Aerator running. Oxygen at {new_oxygen:.0%}."
        else:
            message = (
                f"Aerator running. Oxygen at {new_oxygen:.0%}. "
                "Still needs more aeration."
            )

        warnings = self.get_tank_warnings(tank)

        return CareResult(
            success=True,
            action=CareAction.AERATE,
            message=message,
            warnings=warnings,
        )

    def sprinkle(
        self,
        tank: TankEnvironment,
        creature_state: CreatureState,
    ) -> CareResult:
        """Sprinkle water in the terrarium (Frogman requirement).

        Only works in terrarium mode. Boosts oxygen and cleanliness slightly,
        simulating moisture maintenance for amphibian stages.

        Args:
            tank: Tank environment to sprinkle (mutated in place).
            creature_state: Current creature state for stage validation.

        Returns:
            CareResult with outcome.
        """
        if tank.environment_type != EnvironmentType.TERRARIUM:
            return CareResult(
                success=False,
                action=CareAction.SPRINKLE,
                message="Sprinkler only works in terrarium mode!",
                warnings=self.get_tank_warnings(tank, creature_state),
            )

        # Sprinkler boosts oxygen and cleanliness in terrarium
        tank.oxygen_level = _clamp(tank.oxygen_level + SPRINKLER_MOISTURE_BOOST)
        tank.cleanliness = _clamp(tank.cleanliness + SPRINKLER_MOISTURE_BOOST * 0.5)

        stage = creature_state.stage
        if stage in (CreatureStage.TADMAN, CreatureStage.FROGMAN):
            message = (
                f"The {stage.value} enjoys the moisture! "
                f"Oxygen boosted to {tank.oxygen_level:.0%}."
            )
        else:
            message = f"Sprinkled water. Oxygen at {tank.oxygen_level:.0%}."

        warnings = self.get_tank_warnings(tank, creature_state)

        return CareResult(
            success=True,
            action=CareAction.SPRINKLE,
            message=message,
            warnings=warnings,
        )

    def drain_tank(
        self,
        tank: TankEnvironment,
        creature_state: CreatureState,
    ) -> CareResult:
        """Drain the tank for aquarium-to-terrarium transition.

        Required for Podfish->Tadman evolution.

        Args:
            tank: Tank environment to drain (mutated in place).
            creature_state: Current creature state for validation.

        Returns:
            CareResult with outcome.
        """
        if tank.environment_type == EnvironmentType.TERRARIUM:
            return CareResult(
                success=False,
                action=CareAction.DRAIN,
                message="The tank is already drained!",
                warnings=self.get_tank_warnings(tank, creature_state),
            )

        tank.drain()

        message = (
            "Tank drained! Now in terrarium mode. "
            f"The {creature_state.stage.value} adjusts to the new environment."
        )

        warnings = self.get_tank_warnings(tank, creature_state)

        return CareResult(
            success=True,
            action=CareAction.DRAIN,
            message=message,
            warnings=warnings,
        )

    def fill_tank(
        self,
        tank: TankEnvironment,
        creature_state: CreatureState,
    ) -> CareResult:
        """Fill the tank for terrarium-to-aquarium transition.

        Args:
            tank: Tank environment to fill (mutated in place).
            creature_state: Current creature state for validation.

        Returns:
            CareResult with outcome.
        """
        if tank.environment_type == EnvironmentType.AQUARIUM:
            return CareResult(
                success=False,
                action=CareAction.FILL,
                message="The tank is already filled!",
                warnings=self.get_tank_warnings(tank, creature_state),
            )

        tank.fill()

        message = (
            "Tank filled! Back to aquarium mode. "
            f"The {creature_state.stage.value} slips back into the water."
        )

        warnings = self.get_tank_warnings(tank, creature_state)

        return CareResult(
            success=True,
            action=CareAction.FILL,
            message=message,
            warnings=warnings,
        )

    def get_tank_warnings(
        self,
        tank: TankEnvironment,
        creature_state: CreatureState | None = None,
    ) -> list[str]:
        """Get all tank maintenance warnings including stage-specific alerts.

        Combines base TankEnvironment warnings with stage-specific
        requirements (e.g., Frogman needs sprinkler in terrarium).

        Args:
            tank: Current tank state.
            creature_state: Optional creature state for stage-specific warnings.

        Returns:
            List of warning strings (empty if everything is OK).
        """
        # Base warnings from TankEnvironment
        warnings = tank.get_warnings(self._env_config)

        if creature_state is None:
            return warnings

        stage = creature_state.stage

        # Stage-specific temperature warnings using stage optimal ranges
        stage_min, stage_max = STAGE_OPTIMAL_TEMPS.get(
            stage, (self._env_config.optimal_temp_min,
                    self._env_config.optimal_temp_max)
        )

        # Only add stage-specific temp warning if base didn't already flag critical
        base_has_critical_temp = any("CRITICAL" in w and "Temperature" in w for w in warnings)
        if not base_has_critical_temp:
            if tank.temperature < stage_min:
                stage_msg = (
                    f"Temperature too cold for {stage.value} "
                    f"(needs {stage_min:.0f}-{stage_max:.0f}°C)."
                )
                if stage_msg not in warnings:
                    warnings.append(stage_msg)
            elif tank.temperature > stage_max:
                stage_msg = (
                    f"Temperature too warm for {stage.value} "
                    f"(needs {stage_min:.0f}-{stage_max:.0f}°C)."
                )
                if stage_msg not in warnings:
                    warnings.append(stage_msg)

        # Frogman terrarium requirement
        if stage == CreatureStage.FROGMAN:
            if tank.environment_type != EnvironmentType.TERRARIUM:
                warnings.append(
                    "Frogman needs a terrarium! Drain the tank."
                )
            elif tank.oxygen_level < 0.4:
                warnings.append(
                    "Terrarium needs sprinkling for the Frogman."
                )

        # Tadman terrarium preference
        if stage == CreatureStage.TADMAN:
            if tank.environment_type != EnvironmentType.TERRARIUM:
                warnings.append(
                    "Tadman prefers a terrarium. Consider draining the tank."
                )

        # Aquatic stages in terrarium warning
        if stage in (CreatureStage.MUSHROOMER, CreatureStage.GILLMAN, CreatureStage.PODFISH):
            if tank.environment_type == EnvironmentType.TERRARIUM:
                warnings.append(
                    f"The {stage.value} needs water! Fill the tank."
                )

        return warnings

    def get_stage_optimal_range(
        self,
        stage: CreatureStage,
    ) -> tuple[float, float]:
        """Get the optimal temperature range for a specific stage.

        Args:
            stage: Creature stage to query.

        Returns:
            Tuple of (min_temp, max_temp) in Celsius.
        """
        return STAGE_OPTIMAL_TEMPS.get(
            stage, (self._env_config.optimal_temp_min,
                    self._env_config.optimal_temp_max)
        )

    def is_cleaning_on_cooldown(self) -> bool:
        """Check if tank cleaning is on cooldown.

        Returns:
            True if cleaning cooldown is still active.
        """
        if self._last_clean_time is None:
            return False

        now = self._now_func()
        elapsed = (now - self._last_clean_time).total_seconds()
        return elapsed < CLEANING_DURATION_SECONDS

    def cleaning_cooldown_remaining(self) -> float:
        """Get remaining cleaning cooldown time in seconds.

        Returns:
            Seconds remaining (0.0 if no cooldown active).
        """
        if self._last_clean_time is None:
            return 0.0

        now = self._now_func()
        elapsed = (now - self._last_clean_time).total_seconds()
        remaining = CLEANING_DURATION_SECONDS - elapsed
        return max(0.0, remaining)
