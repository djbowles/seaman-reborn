"""Tests for creature genome and physical trait system (US-047)."""

from __future__ import annotations

import random

import pytest

from seaman_brain.creature.genome import (
    ALL_TRAITS,
    BEHAVIORAL_TRAITS,
    CHROMATIC_TRAITS,
    MORPHOLOGICAL_TRAITS,
    STAGE_DEFAULTS,
    CreatureGenome,
    generate_random_genome,
    mutate_genome,
    update_epigenetic_drift,
)
from seaman_brain.types import CreatureStage

# -----------------------------------------------------------------------
# Trait constants
# -----------------------------------------------------------------------


class TestTraitConstants:
    """Verify trait category definitions."""

    def test_morphological_count(self) -> None:
        assert len(MORPHOLOGICAL_TRAITS) == 5

    def test_chromatic_count(self) -> None:
        assert len(CHROMATIC_TRAITS) == 3

    def test_behavioral_count(self) -> None:
        assert len(BEHAVIORAL_TRAITS) == 4

    def test_all_traits_is_union(self) -> None:
        assert ALL_TRAITS == MORPHOLOGICAL_TRAITS + CHROMATIC_TRAITS + BEHAVIORAL_TRAITS

    def test_total_trait_count(self) -> None:
        assert len(ALL_TRAITS) == 12

    def test_no_duplicate_traits(self) -> None:
        assert len(set(ALL_TRAITS)) == len(ALL_TRAITS)


class TestStageDefaults:
    """Verify stage default lookup tables."""

    def test_all_stages_have_defaults(self) -> None:
        for stage in CreatureStage:
            assert stage in STAGE_DEFAULTS

    def test_defaults_contain_all_traits(self) -> None:
        for stage, defaults in STAGE_DEFAULTS.items():
            for trait in ALL_TRAITS:
                assert trait in defaults, f"{trait} missing for {stage}"

    def test_defaults_within_bounds(self) -> None:
        for stage, defaults in STAGE_DEFAULTS.items():
            for trait, val in defaults.items():
                assert 0.0 <= val <= 1.0, f"{trait}={val} out of range for {stage}"


# -----------------------------------------------------------------------
# CreatureGenome dataclass
# -----------------------------------------------------------------------


class TestCreatureGenome:
    """Tests for the CreatureGenome dataclass."""

    def test_default_genome_has_all_traits(self) -> None:
        g = CreatureGenome()
        assert set(g.traits.keys()) == set(ALL_TRAITS)

    def test_default_values_are_0_5(self) -> None:
        g = CreatureGenome()
        for name in ALL_TRAITS:
            assert g.traits[name] == 0.5

    def test_custom_traits_preserved(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.8, "hue": 0.2})
        assert g.traits["body_size"] == 0.8
        assert g.traits["hue"] == 0.2

    def test_missing_traits_filled_with_default(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.9})
        assert g.traits["voice_pitch"] == 0.5

    def test_traits_clamped_above(self) -> None:
        g = CreatureGenome(traits={"body_size": 1.5})
        assert g.traits["body_size"] == 1.0

    def test_traits_clamped_below(self) -> None:
        g = CreatureGenome(traits={"body_size": -0.3})
        assert g.traits["body_size"] == 0.0

    def test_unknown_traits_in_init_kept_only_if_known(self) -> None:
        # Unknown keys passed to __init__ are NOT removed by __post_init__
        # but from_dict does filter them out
        g = CreatureGenome(traits={"body_size": 0.5, "unknown_trait": 0.9})
        # unknown_trait stays because __post_init__ only ensures ALL_TRAITS exist
        assert "unknown_trait" in g.traits


class TestGenomeSerialization:
    """Tests for to_dict / from_dict round-tripping."""

    def test_to_dict_returns_plain_dict(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.7})
        d = g.to_dict()
        assert isinstance(d, dict)
        assert d["body_size"] == 0.7

    def test_roundtrip(self) -> None:
        original = CreatureGenome(traits={"body_size": 0.3, "hue": 0.8})
        restored = CreatureGenome.from_dict(original.to_dict())
        assert restored.traits == original.traits

    def test_from_dict_ignores_unknown_keys(self) -> None:
        g = CreatureGenome.from_dict({"body_size": 0.6, "alien_power": 99.0})
        assert "alien_power" not in g.traits
        assert g.traits["body_size"] == 0.6

    def test_from_dict_empty(self) -> None:
        g = CreatureGenome.from_dict({})
        # All traits default to 0.5
        for name in ALL_TRAITS:
            assert g.traits[name] == 0.5


# -----------------------------------------------------------------------
# generate_random_genome
# -----------------------------------------------------------------------


class TestGenerateRandomGenome:
    """Tests for random genome generation."""

    def test_generated_genome_has_all_traits(self) -> None:
        g = generate_random_genome()
        assert set(g.traits.keys()) == set(ALL_TRAITS)

    def test_all_traits_within_bounds(self) -> None:
        rng = random.Random(42)
        for _ in range(100):
            g = generate_random_genome(rng=rng)
            for name, val in g.traits.items():
                assert 0.0 <= val <= 1.0, f"{name}={val} out of bounds"

    def test_seeded_generation_is_deterministic(self) -> None:
        g1 = generate_random_genome(rng=random.Random(123))
        g2 = generate_random_genome(rng=random.Random(123))
        assert g1.traits == g2.traits

    def test_different_seeds_produce_different_genomes(self) -> None:
        g1 = generate_random_genome(rng=random.Random(1))
        g2 = generate_random_genome(rng=random.Random(2))
        assert g1.traits != g2.traits

    def test_traits_distributed_around_stage_defaults(self) -> None:
        """With many samples, mean should approach stage defaults."""
        rng = random.Random(42)
        stage = CreatureStage.MUSHROOMER
        n = 500
        sums: dict[str, float] = {t: 0.0 for t in ALL_TRAITS}
        for _ in range(n):
            g = generate_random_genome(stage, rng=rng)
            for t in ALL_TRAITS:
                sums[t] += g.traits[t]
        for t in ALL_TRAITS:
            mean = sums[t] / n
            expected = STAGE_DEFAULTS[stage][t]
            assert abs(mean - expected) < 0.05, (
                f"{t}: mean={mean:.3f} expected≈{expected}"
            )

    @pytest.mark.parametrize("stage", list(CreatureStage))
    def test_all_stages_generate_valid_genomes(self, stage: CreatureStage) -> None:
        g = generate_random_genome(stage, rng=random.Random(0))
        assert set(g.traits.keys()) == set(ALL_TRAITS)
        for val in g.traits.values():
            assert 0.0 <= val <= 1.0

    def test_custom_std_dev(self) -> None:
        """With std_dev=0, all traits should equal stage defaults exactly."""
        g = generate_random_genome(
            CreatureStage.PODFISH, std_dev=0.0, rng=random.Random(0)
        )
        defaults = STAGE_DEFAULTS[CreatureStage.PODFISH]
        for t in ALL_TRAITS:
            assert g.traits[t] == pytest.approx(defaults[t])


# -----------------------------------------------------------------------
# mutate_genome
# -----------------------------------------------------------------------


class TestMutateGenome:
    """Tests for genome mutation."""

    def test_mutation_produces_different_genome(self) -> None:
        parent = generate_random_genome(rng=random.Random(42))
        child = mutate_genome(parent, rng=random.Random(99))
        assert child.traits != parent.traits

    def test_mutation_produces_similar_genome(self) -> None:
        """With default rate, traits should not drift far from parent."""
        rng = random.Random(42)
        parent = generate_random_genome(rng=rng)
        child = mutate_genome(parent, mutation_rate=0.05, rng=random.Random(10))
        for t in ALL_TRAITS:
            diff = abs(child.traits[t] - parent.traits[t])
            assert diff < 0.5, f"{t} drifted too far: {diff}"

    def test_mutation_all_traits_within_bounds(self) -> None:
        parent = CreatureGenome(traits={t: 0.95 for t in ALL_TRAITS})
        rng = random.Random(42)
        for _ in range(50):
            child = mutate_genome(parent, mutation_rate=0.3, rng=rng)
            for name, val in child.traits.items():
                assert 0.0 <= val <= 1.0, f"{name}={val}"

    def test_zero_mutation_rate_returns_copy(self) -> None:
        parent = generate_random_genome(rng=random.Random(42))
        child = mutate_genome(parent, mutation_rate=0.0, rng=random.Random(0))
        assert child.traits == parent.traits
        # But it should be a new object
        assert child is not parent

    def test_high_mutation_rate_produces_large_drift(self) -> None:
        parent = CreatureGenome(traits={t: 0.5 for t in ALL_TRAITS})
        rng = random.Random(42)
        diffs = []
        for _ in range(100):
            child = mutate_genome(parent, mutation_rate=0.5, rng=rng)
            for t in ALL_TRAITS:
                diffs.append(abs(child.traits[t] - parent.traits[t]))
        avg_diff = sum(diffs) / len(diffs)
        assert avg_diff > 0.1  # Significant drift expected

    def test_mutation_seeded_is_deterministic(self) -> None:
        parent = generate_random_genome(rng=random.Random(42))
        c1 = mutate_genome(parent, rng=random.Random(7))
        c2 = mutate_genome(parent, rng=random.Random(7))
        assert c1.traits == c2.traits


# -----------------------------------------------------------------------
# update_epigenetic_drift
# -----------------------------------------------------------------------


class TestEpigeneticDrift:
    """Tests for epigenetic drift during life."""

    def test_zero_days_returns_copy(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=1.0, environment_quality=1.0,
                                          elapsed_days=0)
        assert drifted.traits == g.traits
        assert drifted is not g

    def test_negative_days_returns_copy(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=1.0, environment_quality=1.0,
                                          elapsed_days=-5)
        assert drifted.traits == g.traits

    def test_good_care_increases_body_size(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=1.0, environment_quality=0.5,
                                          elapsed_days=10)
        assert drifted.traits["body_size"] > g.traits["body_size"]

    def test_good_care_decreases_aggression(self) -> None:
        g = CreatureGenome(traits={"aggression_baseline": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=1.0, environment_quality=0.5,
                                          elapsed_days=10)
        assert drifted.traits["aggression_baseline"] < g.traits["aggression_baseline"]

    def test_poor_care_increases_aggression(self) -> None:
        g = CreatureGenome(traits={"aggression_baseline": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=0.0, environment_quality=0.5,
                                          elapsed_days=10)
        assert drifted.traits["aggression_baseline"] > g.traits["aggression_baseline"]

    def test_good_environment_increases_saturation(self) -> None:
        g = CreatureGenome(traits={"saturation": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=0.5, environment_quality=1.0,
                                          elapsed_days=10)
        assert drifted.traits["saturation"] > g.traits["saturation"]

    def test_neutral_conditions_no_drift(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.5, "aggression_baseline": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=0.5, environment_quality=0.5,
                                          elapsed_days=100)
        assert drifted.traits["body_size"] == g.traits["body_size"]
        assert drifted.traits["aggression_baseline"] == g.traits["aggression_baseline"]

    def test_drift_clamped_to_bounds(self) -> None:
        """Even extreme drift should stay in [0, 1]."""
        g = CreatureGenome(traits={"body_size": 0.99})
        drifted = update_epigenetic_drift(g, care_quality=1.0, environment_quality=1.0,
                                          elapsed_days=10000)
        assert drifted.traits["body_size"] <= 1.0

    def test_drift_proportional_to_elapsed_days(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.5})
        short = update_epigenetic_drift(g, care_quality=1.0, environment_quality=0.5,
                                        elapsed_days=1)
        long = update_epigenetic_drift(g, care_quality=1.0, environment_quality=0.5,
                                       elapsed_days=10)
        short_drift = short.traits["body_size"] - g.traits["body_size"]
        long_drift = long.traits["body_size"] - g.traits["body_size"]
        assert long_drift > short_drift

    def test_unaffected_traits_unchanged(self) -> None:
        """Traits without drift rules should not change."""
        g = CreatureGenome(traits={"head_ratio": 0.5, "fin_length": 0.5})
        drifted = update_epigenetic_drift(g, care_quality=1.0, environment_quality=1.0,
                                          elapsed_days=100)
        assert drifted.traits["head_ratio"] == 0.5
        assert drifted.traits["fin_length"] == 0.5

    def test_original_genome_not_mutated(self) -> None:
        g = CreatureGenome(traits={"body_size": 0.5})
        original_val = g.traits["body_size"]
        update_epigenetic_drift(g, care_quality=1.0, environment_quality=1.0,
                                elapsed_days=10)
        assert g.traits["body_size"] == original_val


# -----------------------------------------------------------------------
# CreatureState genome integration
# -----------------------------------------------------------------------


class TestCreatureStateGenomeIntegration:
    """Tests for genome persistence inside CreatureState."""

    def test_state_default_genome_is_none(self) -> None:
        from seaman_brain.creature.state import CreatureState
        s = CreatureState()
        assert s.genome is None

    def test_state_with_genome(self) -> None:
        from seaman_brain.creature.state import CreatureState
        g = CreatureGenome(traits={"body_size": 0.8})
        s = CreatureState(genome=g)
        assert s.genome is not None
        assert s.genome.traits["body_size"] == 0.8

    def test_to_dict_without_genome(self) -> None:
        from seaman_brain.creature.state import CreatureState
        s = CreatureState()
        d = s.to_dict()
        assert "genome" not in d

    def test_to_dict_with_genome(self) -> None:
        from seaman_brain.creature.state import CreatureState
        g = CreatureGenome(traits={"body_size": 0.7})
        s = CreatureState(genome=g)
        d = s.to_dict()
        assert "genome" in d
        assert d["genome"]["body_size"] == 0.7

    def test_from_dict_with_genome_roundtrip(self) -> None:
        from seaman_brain.creature.state import CreatureState
        g = generate_random_genome(rng=random.Random(42))
        s = CreatureState(genome=g)
        d = s.to_dict()
        restored = CreatureState.from_dict(d)
        assert restored.genome is not None
        assert restored.genome.traits == g.traits

    def test_from_dict_without_genome(self) -> None:
        from seaman_brain.creature.state import CreatureState
        s = CreatureState.from_dict({"mood": "happy"})
        assert s.genome is None

    def test_persistence_roundtrip(self, tmp_path: object) -> None:
        """Full save/load cycle with genome included."""
        import json
        from pathlib import Path

        from seaman_brain.creature.state import CreatureState

        g = generate_random_genome(rng=random.Random(42))
        s = CreatureState(genome=g)
        d = s.to_dict()

        save_file = Path(str(tmp_path)) / "creature.json"
        save_file.write_text(json.dumps(d, indent=2), encoding="utf-8")

        loaded_data = json.loads(save_file.read_text(encoding="utf-8"))
        restored = CreatureState.from_dict(loaded_data)
        assert restored.genome is not None
        assert restored.genome.traits == g.traits
