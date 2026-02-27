"""Creature biological needs system.

Manages hunger, comfort, health, and stimulation. Needs degrade over time
based on configurable rates and are affected by tank conditions and player
interaction. Urgent needs trigger warnings that feed into the prompt builder
and mood engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.types import CreatureStage


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# Stage multipliers for hunger rate — higher stages burn energy faster
STAGE_HUNGER_MULTIPLIER: dict[CreatureStage, float] = {
    CreatureStage.MUSHROOMER: 0.5,
    CreatureStage.GILLMAN: 0.8,
    CreatureStage.PODFISH: 1.0,
    CreatureStage.TADMAN: 1.3,
    CreatureStage.FROGMAN: 1.5,
}


@dataclass
class CreatureNeeds:
    """Snapshot of the creature's biological needs.

    Fields:
        hunger: Hunger level (0.0=full, 1.0=starving).
        comfort: Comfort level (0.0=miserable, 1.0=comfortable).
        health: Health level (0.0=dead, 1.0=healthy).
        stimulation: Mental stimulation (0.0=bored, 1.0=engaged).
        last_update: Timestamp of the last needs update.
    """

    hunger: float = 0.0
    comfort: float = 1.0
    health: float = 1.0
    stimulation: float = 1.0
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Clamp all need values to valid ranges."""
        self.hunger = _clamp(self.hunger)
        self.comfort = _clamp(self.comfort)
        self.health = _clamp(self.health)
        self.stimulation = _clamp(self.stimulation)


class NeedsEngine:
    """Engine that updates creature needs over time.

    Hunger increases at a configurable rate (scaled by stage).
    Comfort is derived from tank conditions (temperature + cleanliness).
    Health degrades when hunger or comfort are critically bad.
    Stimulation decays without interaction, restored by conversation.

    Args:
        config: Needs configuration with rates and thresholds.
        env_config: Environment configuration for comfort calculations.
    """

    def __init__(
        self,
        config: NeedsConfig | None = None,
        env_config: EnvironmentConfig | None = None,
    ) -> None:
        self._config = config or NeedsConfig()
        self._env_config = env_config or EnvironmentConfig()

    @property
    def config(self) -> NeedsConfig:
        """Current needs configuration."""
        return self._config

    def update(
        self,
        elapsed_seconds: float,
        creature_state: CreatureState,
        tank: TankEnvironment,
        interaction_count_delta: int = 0,
    ) -> CreatureNeeds:
        """Calculate updated needs based on elapsed time and conditions.

        Args:
            elapsed_seconds: Seconds since last update.
            creature_state: Current creature state (for stage, current values).
            tank: Current tank environment state.
            interaction_count_delta: New interactions since last update.

        Returns:
            Updated CreatureNeeds snapshot.
        """
        if elapsed_seconds <= 0:
            return CreatureNeeds(
                hunger=creature_state.hunger,
                comfort=creature_state.comfort,
                health=creature_state.health,
                stimulation=_clamp(creature_state.comfort),  # fallback
            )

        cfg = self._config

        # --- Hunger: increases over time, faster at higher stages ---
        stage_mult = STAGE_HUNGER_MULTIPLIER.get(creature_state.stage, 1.0)
        hunger_delta = cfg.hunger_rate * stage_mult * elapsed_seconds
        new_hunger = _clamp(creature_state.hunger + hunger_delta)

        # --- Comfort: derived from tank conditions ---
        new_comfort = self._calculate_comfort(tank)

        # --- Stimulation: decays without interaction ---
        stim_decay = cfg.stimulation_decay_rate * elapsed_seconds
        stim_boost = min(1.0, interaction_count_delta * 0.2)
        # Start from creature's current comfort as proxy for stimulation
        # (stimulation isn't stored on CreatureState, so we approximate)
        current_stim = creature_state.comfort  # fallback
        new_stimulation = _clamp(current_stim - stim_decay + stim_boost)

        # --- Health: degrades from critical hunger/comfort, regenerates otherwise ---
        new_health = self._calculate_health(
            elapsed_seconds,
            creature_state.health,
            new_hunger,
            new_comfort,
        )

        return CreatureNeeds(
            hunger=new_hunger,
            comfort=new_comfort,
            health=new_health,
            stimulation=new_stimulation,
        )

    def _calculate_comfort(self, tank: TankEnvironment) -> float:
        """Derive comfort from tank temperature and cleanliness.

        Comfort is the average of temperature comfort and cleanliness,
        with a penalty for non-optimal temperature.
        """
        env_cfg = self._env_config

        # Temperature comfort: 1.0 if optimal, decreasing as it departs
        temp = tank.temperature
        if env_cfg.optimal_temp_min <= temp <= env_cfg.optimal_temp_max:
            temp_comfort = 1.0
        elif temp < env_cfg.lethal_temp_min or temp > env_cfg.lethal_temp_max:
            temp_comfort = 0.0
        else:
            # Linear interpolation between lethal and optimal
            if temp < env_cfg.optimal_temp_min:
                range_size = env_cfg.optimal_temp_min - env_cfg.lethal_temp_min
                if range_size <= 0:
                    temp_comfort = 0.0
                else:
                    temp_comfort = (temp - env_cfg.lethal_temp_min) / range_size
            else:
                range_size = env_cfg.lethal_temp_max - env_cfg.optimal_temp_max
                if range_size <= 0:
                    temp_comfort = 0.0
                else:
                    temp_comfort = (env_cfg.lethal_temp_max - temp) / range_size

        temp_comfort = _clamp(temp_comfort)

        # Overall comfort: weighted combination of temp and cleanliness
        comfort = 0.5 * temp_comfort + 0.5 * tank.cleanliness
        return _clamp(comfort)

    def _calculate_health(
        self,
        elapsed_seconds: float,
        current_health: float,
        hunger: float,
        comfort: float,
    ) -> float:
        """Calculate health based on hunger and comfort levels.

        Health degrades when hunger exceeds critical threshold or comfort
        drops below critical level. Otherwise it slowly regenerates.
        """
        cfg = self._config

        is_starving = hunger >= cfg.critical_hunger_threshold
        is_uncomfortable = comfort < (1.0 - cfg.critical_hunger_threshold)

        if is_starving or is_uncomfortable:
            # Health degrades
            damage = cfg.health_damage_rate * elapsed_seconds
            return _clamp(current_health - damage)
        else:
            # Health slowly regenerates
            regen = cfg.health_regen_rate * elapsed_seconds
            return _clamp(current_health + regen)

    def get_urgent_needs(self, needs: CreatureNeeds) -> list[str]:
        """Return list of needs requiring immediate player attention.

        Args:
            needs: Current creature needs snapshot.

        Returns:
            List of urgent need descriptions (empty if all needs are OK).
        """
        cfg = self._config
        urgent: list[str] = []

        if needs.hunger >= cfg.critical_hunger_threshold:
            urgent.append("The creature is starving! Feed it immediately!")

        if needs.health <= cfg.critical_health_threshold:
            urgent.append("The creature's health is critically low!")

        if needs.comfort < 0.2:
            urgent.append("The creature is very uncomfortable! Check tank conditions.")

        if needs.stimulation < 0.2:
            urgent.append("The creature is bored and unstimulated. Talk to it!")

        return urgent

    def apply_to_state(
        self,
        creature_state: CreatureState,
        needs: CreatureNeeds,
    ) -> None:
        """Apply computed needs back to creature state (mutates state in place).

        Args:
            creature_state: Creature state to update.
            needs: Computed needs to apply.
        """
        creature_state.hunger = needs.hunger
        creature_state.comfort = needs.comfort
        creature_state.health = needs.health
