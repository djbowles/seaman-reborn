"""Creature stage transition logic.

Manages evolution through the 5 stages:
MUSHROOMER -> GILLMAN -> PODFISH -> TADMAN -> FROGMAN.

Evolution is gated by interaction count and trust level thresholds
loaded from CreatureConfig. Stages cannot be skipped or devolved.
"""

from __future__ import annotations

from seaman_brain.config import CreatureConfig, EvolutionThreshold
from seaman_brain.creature.state import CreatureState
from seaman_brain.personality.traits import TraitProfile, get_default_profile
from seaman_brain.types import CreatureStage

# Canonical stage order — index determines progression.
STAGE_ORDER: tuple[CreatureStage, ...] = (
    CreatureStage.MUSHROOMER,
    CreatureStage.GILLMAN,
    CreatureStage.PODFISH,
    CreatureStage.TADMAN,
    CreatureStage.FROGMAN,
)

# Default evolution thresholds used when config has no overrides.
DEFAULT_THRESHOLDS: dict[CreatureStage, EvolutionThreshold] = {
    CreatureStage.GILLMAN: EvolutionThreshold(interactions=20, trust=0.3),
    CreatureStage.PODFISH: EvolutionThreshold(interactions=50, trust=0.5),
    CreatureStage.TADMAN: EvolutionThreshold(interactions=100, trust=0.6),
    CreatureStage.FROGMAN: EvolutionThreshold(interactions=200, trust=0.8),
}


def _stage_index(stage: CreatureStage) -> int:
    """Return the ordinal index of a stage in the progression."""
    return STAGE_ORDER.index(stage)


class EvolutionEngine:
    """Manages creature stage transitions based on config thresholds.

    Thresholds are loaded from CreatureConfig.evolution_thresholds (keyed by
    target stage name) with hardcoded defaults as fallback.
    """

    def __init__(self, config: CreatureConfig | None = None) -> None:
        self._thresholds: dict[CreatureStage, EvolutionThreshold] = {}
        self._load_thresholds(config)

    def _load_thresholds(self, config: CreatureConfig | None) -> None:
        """Populate thresholds from config, falling back to defaults."""
        # Start with hardcoded defaults.
        self._thresholds = dict(DEFAULT_THRESHOLDS)

        if config is None:
            return

        # Override with config values (keyed by stage *name* e.g. "gillman").
        for name, threshold in config.evolution_thresholds.items():
            try:
                stage = CreatureStage(name.lower())
            except ValueError:
                continue  # Ignore unknown stage names in config.
            if stage != CreatureStage.MUSHROOMER:
                self._thresholds[stage] = threshold

    def get_threshold(self, target_stage: CreatureStage) -> EvolutionThreshold | None:
        """Return the threshold for evolving *into* the given stage.

        Returns None for MUSHROOMER (no evolution into the first stage).
        """
        return self._thresholds.get(target_stage)

    def check_evolution(self, state: CreatureState) -> CreatureStage | None:
        """Check whether the creature qualifies for the next stage.

        Evaluates only the *immediate next* stage — never skips stages.

        Args:
            state: Current creature state.

        Returns:
            The next CreatureStage if thresholds are met, or None.
        """
        current_idx = _stage_index(state.stage)

        # Already at max stage.
        if current_idx >= len(STAGE_ORDER) - 1:
            return None

        next_stage = STAGE_ORDER[current_idx + 1]
        threshold = self._thresholds.get(next_stage)

        if threshold is None:
            return None  # No threshold defined — cannot evolve.

        if (
            state.interaction_count >= threshold.interactions
            and state.trust_level >= threshold.trust
        ):
            return next_stage

        return None

    def evolve(
        self,
        state: CreatureState,
        new_stage: CreatureStage,
    ) -> TraitProfile:
        """Apply a stage transition to the creature state.

        Validates that the transition is legal (one step forward only),
        updates the state's stage in-place, and returns the new stage's
        default trait profile.

        Args:
            state: Mutable creature state — its ``stage`` field is updated.
            new_stage: The target stage to evolve into.

        Returns:
            The default TraitProfile for the new stage.

        Raises:
            ValueError: If the transition is invalid (skip, devolve, same, or
                trying to evolve from max stage).
        """
        current_idx = _stage_index(state.stage)
        new_idx = _stage_index(new_stage)

        if new_idx == current_idx:
            raise ValueError(
                f"Already at stage {state.stage.value}, cannot evolve to same stage"
            )

        if new_idx < current_idx:
            raise ValueError(
                f"Cannot devolve from {state.stage.value} to {new_stage.value}"
            )

        if new_idx != current_idx + 1:
            raise ValueError(
                f"Cannot skip stages: {state.stage.value} -> {new_stage.value} "
                f"(expected {STAGE_ORDER[current_idx + 1].value})"
            )

        state.stage = new_stage
        return get_default_profile(new_stage)

    def can_evolve(self, state: CreatureState) -> bool:
        """Convenience check — True if check_evolution would return a stage."""
        return self.check_evolution(state) is not None

    def stages_remaining(self, state: CreatureState) -> int:
        """Return how many evolution steps remain from the current stage."""
        return len(STAGE_ORDER) - 1 - _stage_index(state.stage)
