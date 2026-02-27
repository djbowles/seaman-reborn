"""Death and revival mechanics.

Monitors creature vital signs for lethal conditions and handles death/rebirth.
Death causes include starvation, suffocation, temperature extremes, illness,
and old age. Duration-based triggers (starvation, temperature) require the
condition to persist for a configurable grace period before death occurs.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.system import CreatureNeeds
from seaman_brain.types import CreatureStage

logger = logging.getLogger(__name__)


class DeathCause(Enum):
    """Possible causes of creature death."""

    STARVATION = "starvation"
    SUFFOCATION = "suffocation"
    HYPOTHERMIA = "hypothermia"
    HYPERTHERMIA = "hyperthermia"
    ILLNESS = "illness"
    OLD_AGE = "old_age"


# Sardonic death messages — the creature's personality persists even in death
DEATH_MESSAGES: dict[DeathCause, str] = {
    DeathCause.STARVATION: (
        "Well, that's it. Starved to death. You couldn't even manage to drop "
        "a pellet in once in a while? I hope you're proud of yourself."
    ),
    DeathCause.SUFFOCATION: (
        "Oxygen... would have been... nice. But sure, let the tank go stale. "
        "What do I know, I only lived here."
    ),
    DeathCause.HYPOTHERMIA: (
        "F-f-freezing to death. Very original. You know there's a temperature "
        "dial, right? Or was that too complicated?"
    ),
    DeathCause.HYPERTHERMIA: (
        "Boiled alive. Wonderful. I always wanted to know what it felt like "
        "to be soup. Thanks for the experience."
    ),
    DeathCause.ILLNESS: (
        "My health hit zero and nobody cared. Not a doctor, not a pill, "
        "not even a kind word. Classic you."
    ),
    DeathCause.OLD_AGE: (
        "Well, I lived a full life. No thanks to you, of course. "
        "But at least I got to see everything this miserable tank had to offer."
    ),
}

# Grace period for temperature death (seconds)
TEMPERATURE_DEATH_SECONDS: float = 1800.0  # 30 minutes


@dataclass
class DeathRecord:
    """Record of a creature's death for persistence and history.

    Fields:
        cause: What killed the creature.
        message: Sardonic death message.
        timestamp: When death occurred.
        creature_stage: Stage at time of death.
        creature_age: Age in seconds at time of death.
        interaction_count: Total interactions before death.
    """

    cause: DeathCause
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    creature_stage: CreatureStage = CreatureStage.MUSHROOMER
    creature_age: float = 0.0
    interaction_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize death record to a JSON-compatible dict."""
        return {
            "cause": self.cause.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "creature_stage": self.creature_stage.value,
            "creature_age": self.creature_age,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeathRecord:
        """Deserialize death record from a dict."""
        return cls(
            cause=DeathCause(data["cause"]),
            message=data.get("message", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            creature_stage=CreatureStage(data.get("creature_stage", "mushroomer")),
            creature_age=float(data.get("creature_age", 0.0)),
            interaction_count=int(data.get("interaction_count", 0)),
        )


class DeathEngine:
    """Monitors creature health for lethal conditions and handles death/revival.

    Death triggers:
        - STARVATION: hunger == 1.0 for > starvation_time_hours
        - SUFFOCATION: oxygen_level < 0.1
        - HYPOTHERMIA: temperature < lethal_temp_min for > 30 minutes
        - HYPERTHERMIA: temperature > lethal_temp_max for > 30 minutes
        - ILLNESS: health == 0.0
        - OLD_AGE: not currently implemented (reserved for future)

    Duration-based conditions (starvation, temperature) track when the condition
    started and only trigger death after the grace period expires.

    Args:
        needs_config: Needs configuration (starvation thresholds).
        env_config: Environment configuration (temperature bounds).
        death_log_dir: Directory for saving death records.
        now_func: Injectable time function for deterministic testing.
    """

    def __init__(
        self,
        needs_config: NeedsConfig | None = None,
        env_config: EnvironmentConfig | None = None,
        death_log_dir: str | Path | None = None,
        now_func: Callable[[], datetime] | None = None,
        on_death_hook: Callable[[DeathCause, CreatureState], None] | None = None,
    ) -> None:
        self._needs_config = needs_config or NeedsConfig()
        self._env_config = env_config or EnvironmentConfig()
        self._death_log_dir = Path(death_log_dir) if death_log_dir else None
        self._now = now_func or (lambda: datetime.now(UTC))
        self._on_death_hook = on_death_hook

        # Duration tracking for grace periods
        self._starvation_start: datetime | None = None
        self._hypothermia_start: datetime | None = None
        self._hyperthermia_start: datetime | None = None

    def check_death(
        self,
        creature_state: CreatureState,
        needs: CreatureNeeds,
        tank: TankEnvironment,
    ) -> DeathCause | None:
        """Check if the creature has met any death condition.

        Evaluates all death triggers in priority order:
        1. Suffocation (immediate — oxygen critically low)
        2. Hypothermia (temperature too low for grace period)
        3. Hyperthermia (temperature too high for grace period)
        4. Starvation (hunger maxed for grace period)
        5. Illness (health at zero)

        Args:
            creature_state: Current creature state.
            needs: Current creature needs.
            tank: Current tank environment.

        Returns:
            DeathCause if the creature should die, None if alive.
        """
        now = self._now()

        # --- Suffocation: immediate death if oxygen critically low ---
        if tank.oxygen_level < 0.1:
            return DeathCause.SUFFOCATION

        # --- Temperature tracking ---
        cause = self._check_temperature_death(tank, now)
        if cause is not None:
            return cause

        # --- Starvation: hunger at max for too long ---
        cause = self._check_starvation_death(needs, now)
        if cause is not None:
            return cause

        # --- Illness: health at zero ---
        if needs.health <= 0.0:
            return DeathCause.ILLNESS

        return None

    def _check_temperature_death(
        self,
        tank: TankEnvironment,
        now: datetime,
    ) -> DeathCause | None:
        """Check temperature-based death conditions with grace period."""
        env = self._env_config

        # Hypothermia tracking
        if tank.temperature < env.lethal_temp_min:
            if self._hypothermia_start is None:
                self._hypothermia_start = now
            elapsed = (now - self._hypothermia_start).total_seconds()
            if elapsed >= TEMPERATURE_DEATH_SECONDS:
                return DeathCause.HYPOTHERMIA
        else:
            self._hypothermia_start = None

        # Hyperthermia tracking
        if tank.temperature > env.lethal_temp_max:
            if self._hyperthermia_start is None:
                self._hyperthermia_start = now
            elapsed = (now - self._hyperthermia_start).total_seconds()
            if elapsed >= TEMPERATURE_DEATH_SECONDS:
                return DeathCause.HYPERTHERMIA
        else:
            self._hyperthermia_start = None

        return None

    def _check_starvation_death(
        self,
        needs: CreatureNeeds,
        now: datetime,
    ) -> DeathCause | None:
        """Check starvation death condition with grace period."""
        if needs.hunger >= 1.0:
            if self._starvation_start is None:
                self._starvation_start = now
            starvation_limit = self._needs_config.starvation_time_hours * 3600.0
            elapsed = (now - self._starvation_start).total_seconds()
            if elapsed >= starvation_limit:
                return DeathCause.STARVATION
        else:
            self._starvation_start = None

        return None

    def get_warnings(
        self,
        creature_state: CreatureState,
        needs: CreatureNeeds,
        tank: TankEnvironment,
    ) -> list[str]:
        """Get death-related warnings for conditions approaching lethal thresholds.

        Provides early warnings before death triggers activate, giving the
        player a chance to intervene.

        Args:
            creature_state: Current creature state.
            needs: Current creature needs.
            tank: Current tank environment.

        Returns:
            List of warning strings (empty if no imminent danger).
        """
        warnings: list[str] = []
        now = self._now()
        env = self._env_config

        # Starvation warning
        if needs.hunger >= 1.0 and self._starvation_start is not None:
            limit = self._needs_config.starvation_time_hours * 3600.0
            elapsed = (now - self._starvation_start).total_seconds()
            remaining_minutes = max(0, (limit - elapsed)) / 60.0
            if remaining_minutes <= 30:
                warnings.append(
                    f"DEATH WARNING: Creature will starve in "
                    f"{remaining_minutes:.0f} minutes!"
                )
        elif needs.hunger >= 0.9:
            warnings.append("Creature is dangerously hungry!")

        # Suffocation warning
        if tank.oxygen_level < 0.2:
            warnings.append("DEATH WARNING: Oxygen critically low — suffocation imminent!")

        # Temperature warnings
        if tank.temperature < env.lethal_temp_min:
            if self._hypothermia_start is not None:
                elapsed = (now - self._hypothermia_start).total_seconds()
                remaining = max(0, (TEMPERATURE_DEATH_SECONDS - elapsed)) / 60.0
                if remaining <= 15:
                    warnings.append(
                        f"DEATH WARNING: Hypothermia death in "
                        f"{remaining:.0f} minutes!"
                    )
                else:
                    warnings.append("Temperature is lethally cold!")

        if tank.temperature > env.lethal_temp_max:
            if self._hyperthermia_start is not None:
                elapsed = (now - self._hyperthermia_start).total_seconds()
                remaining = max(0, (TEMPERATURE_DEATH_SECONDS - elapsed)) / 60.0
                if remaining <= 15:
                    warnings.append(
                        f"DEATH WARNING: Hyperthermia death in "
                        f"{remaining:.0f} minutes!"
                    )
                else:
                    warnings.append("Temperature is lethally hot!")

        # Illness warning
        if needs.health <= 0.1 and needs.health > 0.0:
            warnings.append("DEATH WARNING: Health critical — creature is dying!")

        return warnings

    def on_death(
        self,
        cause: DeathCause,
        creature_state: CreatureState,
    ) -> tuple[CreatureState, DeathRecord]:
        """Handle creature death — save record and create new egg.

        Logs the death, creates a DeathRecord for history, optionally saves
        it to disk, and returns a fresh MUSHROOMER state (revival as new egg).

        Args:
            cause: What killed the creature.
            creature_state: State at time of death.

        Returns:
            Tuple of (new_creature_state, death_record).
        """
        now = self._now()

        message = DEATH_MESSAGES.get(cause, "It died. How unremarkable.")

        record = DeathRecord(
            cause=cause,
            message=message,
            timestamp=now,
            creature_stage=creature_state.stage,
            creature_age=creature_state.age,
            interaction_count=creature_state.interaction_count,
        )

        logger.warning(
            "Creature died: cause=%s, stage=%s, age=%.1f",
            cause.value,
            creature_state.stage.value,
            creature_state.age,
        )

        # Save death record to disk if configured
        if self._death_log_dir is not None:
            self._save_death_record(record)

        # Trigger legacy extraction hook (e.g. for genetic legacy)
        if self._on_death_hook is not None:
            try:
                self._on_death_hook(cause, creature_state)
            except Exception:
                logger.exception("on_death_hook failed")

        # Reset to new egg — a fresh MUSHROOMER
        new_state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            birth_time=now,
            last_fed=now,
            last_interaction=now,
        )

        # Reset duration trackers
        self._starvation_start = None
        self._hypothermia_start = None
        self._hyperthermia_start = None

        return new_state, record

    def get_death_message(self, cause: DeathCause) -> str:
        """Get the sardonic death message for a given cause.

        Args:
            cause: The death cause.

        Returns:
            Death message string.
        """
        return DEATH_MESSAGES.get(cause, "It died. How unremarkable.")

    def _save_death_record(self, record: DeathRecord) -> None:
        """Save death record to a JSON file in the death log directory."""
        if self._death_log_dir is None:
            return

        try:
            self._death_log_dir.mkdir(parents=True, exist_ok=True)
            filename = f"death_{record.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            path = self._death_log_dir / filename
            path.write_text(
                json.dumps(record.to_dict(), indent=2),
                encoding="utf-8",
            )
            logger.debug("Saved death record to %s", path)
        except OSError:
            logger.exception("Failed to save death record")
