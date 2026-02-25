"""Tests for creature evolution logic (US-018)."""

from __future__ import annotations

import pytest

from seaman_brain.config import CreatureConfig, EvolutionThreshold
from seaman_brain.creature.evolution import (
    DEFAULT_THRESHOLDS,
    STAGE_ORDER,
    EvolutionEngine,
    _stage_index,
)
from seaman_brain.creature.state import CreatureState
from seaman_brain.personality.traits import STAGE_DEFAULTS, TraitProfile
from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    stage: CreatureStage = CreatureStage.MUSHROOMER,
    interactions: int = 0,
    trust: float = 0.0,
) -> CreatureState:
    """Create a CreatureState with only the fields relevant to evolution."""
    return CreatureState(
        stage=stage,
        interaction_count=interactions,
        trust_level=trust,
    )


def _config_with_thresholds(
    thresholds: dict[str, dict[str, int | float]],
) -> CreatureConfig:
    """Build a CreatureConfig with custom evolution_thresholds."""
    et = {k: EvolutionThreshold(**v) for k, v in thresholds.items()}
    return CreatureConfig(evolution_thresholds=et)


# ===========================================================================
# Stage order & helpers
# ===========================================================================

class TestStageOrder:
    """Tests for STAGE_ORDER and _stage_index."""

    def test_stage_order_length(self) -> None:
        assert len(STAGE_ORDER) == 5

    def test_stage_order_sequence(self) -> None:
        assert STAGE_ORDER == (
            CreatureStage.MUSHROOMER,
            CreatureStage.GILLMAN,
            CreatureStage.PODFISH,
            CreatureStage.TADMAN,
            CreatureStage.FROGMAN,
        )

    def test_stage_index_values(self) -> None:
        for idx, stage in enumerate(STAGE_ORDER):
            assert _stage_index(stage) == idx


# ===========================================================================
# Default thresholds
# ===========================================================================

class TestDefaultThresholds:
    """Tests for DEFAULT_THRESHOLDS mapping."""

    def test_no_threshold_for_mushroomer(self) -> None:
        assert CreatureStage.MUSHROOMER not in DEFAULT_THRESHOLDS

    def test_all_non_first_stages_have_thresholds(self) -> None:
        for stage in STAGE_ORDER[1:]:
            assert stage in DEFAULT_THRESHOLDS

    def test_thresholds_monotonically_increase(self) -> None:
        prev_interactions = 0
        prev_trust = 0.0
        for stage in STAGE_ORDER[1:]:
            t = DEFAULT_THRESHOLDS[stage]
            assert t.interactions >= prev_interactions
            assert t.trust >= prev_trust
            prev_interactions = t.interactions
            prev_trust = t.trust


# ===========================================================================
# EvolutionEngine — initialization
# ===========================================================================

class TestEngineInit:
    """Tests for EvolutionEngine construction and threshold loading."""

    def test_default_thresholds_when_no_config(self) -> None:
        engine = EvolutionEngine()
        for stage in STAGE_ORDER[1:]:
            assert engine.get_threshold(stage) == DEFAULT_THRESHOLDS[stage]

    def test_config_overrides_defaults(self) -> None:
        cfg = _config_with_thresholds({
            "gillman": {"interactions": 99, "trust": 0.9},
        })
        engine = EvolutionEngine(cfg)
        t = engine.get_threshold(CreatureStage.GILLMAN)
        assert t is not None
        assert t.interactions == 99
        assert t.trust == 0.9

    def test_config_preserves_unset_defaults(self) -> None:
        cfg = _config_with_thresholds({
            "gillman": {"interactions": 99, "trust": 0.9},
        })
        engine = EvolutionEngine(cfg)
        # Podfish should still have default thresholds.
        expected = DEFAULT_THRESHOLDS[CreatureStage.PODFISH]
        assert engine.get_threshold(CreatureStage.PODFISH) == expected

    def test_unknown_stage_in_config_ignored(self) -> None:
        cfg = _config_with_thresholds({
            "unknown_stage": {"interactions": 1, "trust": 0.1},
        })
        engine = EvolutionEngine(cfg)
        # Should still have all defaults, no crash.
        for stage in STAGE_ORDER[1:]:
            assert engine.get_threshold(stage) is not None

    def test_mushroomer_threshold_in_config_ignored(self) -> None:
        cfg = _config_with_thresholds({
            "mushroomer": {"interactions": 1, "trust": 0.1},
        })
        engine = EvolutionEngine(cfg)
        assert engine.get_threshold(CreatureStage.MUSHROOMER) is None

    def test_get_threshold_returns_none_for_mushroomer(self) -> None:
        engine = EvolutionEngine()
        assert engine.get_threshold(CreatureStage.MUSHROOMER) is None


# ===========================================================================
# check_evolution — happy path
# ===========================================================================

class TestCheckEvolutionHappy:
    """Tests for check_evolution when thresholds are met."""

    def test_mushroomer_to_gillman(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=20,
            trust=0.3,
        )
        assert engine.check_evolution(state) == CreatureStage.GILLMAN

    def test_gillman_to_podfish(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.GILLMAN,
            interactions=50,
            trust=0.5,
        )
        assert engine.check_evolution(state) == CreatureStage.PODFISH

    def test_podfish_to_tadman(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.PODFISH,
            interactions=100,
            trust=0.6,
        )
        assert engine.check_evolution(state) == CreatureStage.TADMAN

    def test_tadman_to_frogman(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.TADMAN,
            interactions=200,
            trust=0.8,
        )
        assert engine.check_evolution(state) == CreatureStage.FROGMAN

    def test_exceeding_thresholds_still_evolves(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=999,
            trust=1.0,
        )
        assert engine.check_evolution(state) == CreatureStage.GILLMAN


# ===========================================================================
# check_evolution — no evolution
# ===========================================================================

class TestCheckEvolutionNoEvolution:
    """Tests for check_evolution when thresholds are NOT met."""

    def test_no_evolution_zero_stats(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER, interactions=0, trust=0.0)
        assert engine.check_evolution(state) is None

    def test_interactions_met_but_not_trust(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=20,
            trust=0.1,  # Need 0.3.
        )
        assert engine.check_evolution(state) is None

    def test_trust_met_but_not_interactions(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=5,
            trust=0.9,  # Trust is fine, not enough interactions.
        )
        assert engine.check_evolution(state) is None

    def test_max_stage_returns_none(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.FROGMAN,
            interactions=9999,
            trust=1.0,
        )
        assert engine.check_evolution(state) is None

    def test_just_below_thresholds(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=19,
            trust=0.29,
        )
        assert engine.check_evolution(state) is None

    def test_custom_higher_thresholds_block_evolution(self) -> None:
        cfg = _config_with_thresholds({
            "gillman": {"interactions": 100, "trust": 0.9},
        })
        engine = EvolutionEngine(cfg)
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=20,  # Default would pass, but custom is 100.
            trust=0.3,
        )
        assert engine.check_evolution(state) is None


# ===========================================================================
# evolve — happy path
# ===========================================================================

class TestEvolveHappy:
    """Tests for evolve() performing valid stage transitions."""

    def test_evolve_updates_stage(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER)
        engine.evolve(state, CreatureStage.GILLMAN)
        assert state.stage == CreatureStage.GILLMAN

    def test_evolve_returns_trait_profile(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER)
        profile = engine.evolve(state, CreatureStage.GILLMAN)
        assert isinstance(profile, TraitProfile)

    def test_evolve_returns_correct_stage_profile(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.PODFISH)
        profile = engine.evolve(state, CreatureStage.TADMAN)
        expected = STAGE_DEFAULTS[CreatureStage.TADMAN]
        assert profile.to_dict() == expected.to_dict()

    def test_full_evolution_chain(self) -> None:
        """Evolve through all 5 stages sequentially."""
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER)
        for target in STAGE_ORDER[1:]:
            profile = engine.evolve(state, target)
            assert state.stage == target
            assert isinstance(profile, TraitProfile)
        assert state.stage == CreatureStage.FROGMAN


# ===========================================================================
# evolve — error handling
# ===========================================================================

class TestEvolveErrors:
    """Tests for evolve() rejecting invalid transitions."""

    def test_cannot_evolve_to_same_stage(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.GILLMAN)
        with pytest.raises(ValueError, match="same stage"):
            engine.evolve(state, CreatureStage.GILLMAN)

    def test_cannot_devolve(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.PODFISH)
        with pytest.raises(ValueError, match="Cannot devolve"):
            engine.evolve(state, CreatureStage.MUSHROOMER)

    def test_cannot_skip_stages(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER)
        with pytest.raises(ValueError, match="Cannot skip"):
            engine.evolve(state, CreatureStage.PODFISH)

    def test_devolve_by_one_step_rejected(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.GILLMAN)
        with pytest.raises(ValueError, match="Cannot devolve"):
            engine.evolve(state, CreatureStage.MUSHROOMER)

    def test_skip_from_gillman_to_frogman_rejected(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.GILLMAN)
        with pytest.raises(ValueError, match="Cannot skip"):
            engine.evolve(state, CreatureStage.FROGMAN)


# ===========================================================================
# Convenience methods
# ===========================================================================

class TestConvenienceMethods:
    """Tests for can_evolve and stages_remaining."""

    def test_can_evolve_true(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.MUSHROOMER,
            interactions=20,
            trust=0.3,
        )
        assert engine.can_evolve(state) is True

    def test_can_evolve_false(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER)
        assert engine.can_evolve(state) is False

    def test_can_evolve_false_at_max(self) -> None:
        engine = EvolutionEngine()
        state = _state(
            stage=CreatureStage.FROGMAN,
            interactions=9999,
            trust=1.0,
        )
        assert engine.can_evolve(state) is False

    def test_stages_remaining_from_mushroomer(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.MUSHROOMER)
        assert engine.stages_remaining(state) == 4

    def test_stages_remaining_from_tadman(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.TADMAN)
        assert engine.stages_remaining(state) == 1

    def test_stages_remaining_at_max(self) -> None:
        engine = EvolutionEngine()
        state = _state(stage=CreatureStage.FROGMAN)
        assert engine.stages_remaining(state) == 0
