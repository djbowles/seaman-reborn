"""Tests for creature/inheritance.py — US-050 generational inheritance and mutation."""

from __future__ import annotations

import random

import pytest

from seaman_brain.creature.genetics import GeneticLegacy
from seaman_brain.creature.genome import ALL_TRAITS, CreatureGenome, generate_random_genome
from seaman_brain.creature.inheritance import (
    DECAY_PER_GENERATION,
    PERSONALITY_INHERITANCE_WEIGHT,
    VAGUE_PREFIXES,
    AncestralMemory,
    InheritanceEngine,
)
from seaman_brain.personality.traits import STAGE_DEFAULTS
from seaman_brain.types import CreatureStage

# =====================================================================
# AncestralMemory dataclass
# =====================================================================


class TestAncestralMemory:
    """Tests for the AncestralMemory dataclass."""

    def test_default_construction(self):
        """Default AncestralMemory has sensible defaults."""
        mem = AncestralMemory()
        assert mem.original_fact == ""
        assert mem.faded_version == ""
        assert mem.confidence == 1.0
        assert mem.source_generation == 1

    def test_construction_with_data(self):
        """AncestralMemory stores provided data."""
        mem = AncestralMemory(
            original_fact="User likes cats",
            faded_version="I have a dim memory that user likes cats",
            confidence=0.5,
            source_generation=2,
        )
        assert mem.original_fact == "User likes cats"
        assert mem.confidence == 0.5
        assert mem.source_generation == 2
        assert "dim memory" in mem.faded_version


# =====================================================================
# Genome mutation produces valid genome
# =====================================================================


class TestGenomeMutation:
    """Tests that genome mutation through inheritance produces valid genomes."""

    def test_mutated_genome_has_all_traits(self):
        """Child genome from hatch_from_legacy contains all trait keys."""
        rng = random.Random(42)
        parent_genome = generate_random_genome(CreatureStage.PODFISH, rng=rng)
        legacy = GeneticLegacy(
            genome=parent_genome.to_dict(),
            distilled_memories=["User likes cats"],
            generation_number=1,
        )

        engine = InheritanceEngine(mutation_rate=0.1, rng=random.Random(99))
        new_state, child_genome, _ = engine.hatch_from_legacy(legacy)

        for trait in ALL_TRAITS:
            assert trait in child_genome.traits
            assert 0.0 <= child_genome.traits[trait] <= 1.0

    def test_mutated_genome_differs_from_parent(self):
        """Child genome should differ from parent (with non-zero mutation rate)."""
        rng = random.Random(42)
        parent_genome = generate_random_genome(CreatureStage.FROGMAN, rng=rng)
        legacy = GeneticLegacy(
            genome=parent_genome.to_dict(),
            generation_number=3,
        )

        engine = InheritanceEngine(mutation_rate=0.15, rng=random.Random(123))
        _, child_genome, _ = engine.hatch_from_legacy(legacy)

        # At least some traits should differ
        differences = sum(
            1 for t in ALL_TRAITS
            if abs(child_genome.traits[t] - parent_genome.traits[t]) > 0.001
        )
        assert differences > 0, "Child genome should differ from parent"

    def test_zero_mutation_preserves_parent(self):
        """With mutation_rate=0, child genome matches parent exactly."""
        parent_genome = CreatureGenome(traits={t: 0.5 for t in ALL_TRAITS})
        legacy = GeneticLegacy(
            genome=parent_genome.to_dict(),
            generation_number=1,
        )

        engine = InheritanceEngine(mutation_rate=0.0, rng=random.Random(42))
        _, child_genome, _ = engine.hatch_from_legacy(legacy)

        for trait in ALL_TRAITS:
            assert child_genome.traits[trait] == pytest.approx(
                parent_genome.traits[trait], abs=0.001,
            )

    def test_mutation_stays_clamped(self):
        """Even extreme mutation rates keep traits in [0, 1]."""
        # Parent with extreme trait values
        parent_genome = CreatureGenome(
            traits={t: (1.0 if i % 2 == 0 else 0.0) for i, t in enumerate(ALL_TRAITS)},
        )
        legacy = GeneticLegacy(
            genome=parent_genome.to_dict(),
            generation_number=1,
        )

        engine = InheritanceEngine(mutation_rate=0.5, rng=random.Random(42))
        _, child_genome, _ = engine.hatch_from_legacy(legacy)

        for trait in ALL_TRAITS:
            assert 0.0 <= child_genome.traits[trait] <= 1.0


# =====================================================================
# Ancestral memory fading across generations
# =====================================================================


class TestAncestralMemoryFading:
    """Tests for memory confidence decay across multiple generations."""

    def test_direct_child_gets_half_confidence(self):
        """Direct child (gen gap 1) gets confidence = 0.5."""
        legacy = GeneticLegacy(
            distilled_memories=["User likes cats", "User name is Alice"],
            generation_number=1,
        )

        engine = InheritanceEngine(rng=random.Random(42))
        _, _, memories = engine.hatch_from_legacy(legacy)

        assert len(memories) == 2
        for mem in memories:
            assert mem.confidence == pytest.approx(DECAY_PER_GENERATION, abs=0.01)
            assert mem.source_generation == 1

    def test_memories_have_vague_prefixes(self):
        """Faded versions start with one of the defined vague prefixes."""
        legacy = GeneticLegacy(
            distilled_memories=["User is a programmer"],
            generation_number=1,
        )

        engine = InheritanceEngine(rng=random.Random(42))
        _, _, memories = engine.hatch_from_legacy(legacy)

        assert len(memories) == 1
        mem = memories[0]
        assert any(mem.faded_version.startswith(p) for p in VAGUE_PREFIXES)
        assert mem.original_fact == "User is a programmer"

    def test_three_generation_fading(self):
        """Memories fade progressively across 3 generations."""
        rng = random.Random(42)

        # Gen 1 parent dies with memories
        legacy_gen1 = GeneticLegacy(
            distilled_memories=["User loves pizza", "User has a dog named Rex"],
            generation_number=1,
        )

        # Gen 2: hatch from gen 1
        engine = InheritanceEngine(rng=random.Random(rng.randint(0, 9999)))
        _, _, gen2_memories = engine.hatch_from_legacy(legacy_gen1)

        # Gen 2 confidence should be 0.5
        for mem in gen2_memories:
            assert mem.confidence == pytest.approx(0.5, abs=0.01)

        # Gen 3: fade gen 2's memories further
        gen3_memories = InheritanceEngine.fade_across_generations(
            gen2_memories, rng=random.Random(99),
        )

        # Gen 3 confidence should be 0.25
        for mem in gen3_memories:
            assert mem.confidence == pytest.approx(0.25, abs=0.01)

        # Gen 4: fade again
        gen4_memories = InheritanceEngine.fade_across_generations(
            gen3_memories, rng=random.Random(99),
        )

        # Gen 4 confidence should be 0.125
        for mem in gen4_memories:
            assert mem.confidence == pytest.approx(0.125, abs=0.01)

    def test_very_faded_memories_are_dropped(self):
        """Memories below 0.05 confidence are removed."""
        # Create a memory at 0.06 confidence — just above threshold
        memories = [
            AncestralMemory(
                original_fact="Ancient fact",
                faded_version="Something tells me that ancient fact",
                confidence=0.06,
                source_generation=1,
            ),
        ]

        # Fading should drop it (0.06 * 0.5 = 0.03 < 0.05)
        faded = InheritanceEngine.fade_across_generations(
            memories, rng=random.Random(42),
        )
        assert len(faded) == 0

    def test_empty_memories_produce_empty_list(self):
        """Legacy with no memories produces no ancestral memories."""
        legacy = GeneticLegacy(
            distilled_memories=[],
            generation_number=5,
        )

        engine = InheritanceEngine(rng=random.Random(42))
        _, _, memories = engine.hatch_from_legacy(legacy)
        assert memories == []

    def test_blank_memories_are_skipped(self):
        """Blank or whitespace-only facts are filtered out."""
        legacy = GeneticLegacy(
            distilled_memories=["Real fact", "", "  ", "Another fact"],
            generation_number=1,
        )

        engine = InheritanceEngine(rng=random.Random(42))
        _, _, memories = engine.hatch_from_legacy(legacy)
        assert len(memories) == 2
        original_facts = {m.original_fact for m in memories}
        assert "Real fact" in original_facts
        assert "Another fact" in original_facts


# =====================================================================
# Personality inheritance
# =====================================================================


class TestPersonalityInheritance:
    """Tests for personality baseline computation via inheritance."""

    def test_no_drift_gives_mushroomer_defaults(self):
        """With zero personality drift, inherited traits == MUSHROOMER defaults."""
        legacy = GeneticLegacy(
            personality_drift={},
            generation_number=1,
        )

        profile = InheritanceEngine.compute_inherited_traits(legacy)
        mushroomer = STAGE_DEFAULTS[CreatureStage.MUSHROOMER]

        for name in mushroomer.to_dict():
            assert getattr(profile, name) == pytest.approx(
                getattr(mushroomer, name), abs=0.001,
            )

    def test_positive_drift_shifts_traits(self):
        """Parent drift toward higher values shifts child traits upward."""
        # Parent drifted +0.2 in cynicism from default
        legacy = GeneticLegacy(
            personality_drift={"cynicism": 0.2, "wit": 0.3},
            generation_number=1,
        )

        profile = InheritanceEngine.compute_inherited_traits(legacy)
        mushroomer = STAGE_DEFAULTS[CreatureStage.MUSHROOMER]

        # cynicism should be shifted 30% toward (default + 0.2)
        expected_cynicism = (
            mushroomer.cynicism
            + PERSONALITY_INHERITANCE_WEIGHT * 0.2
        )
        assert profile.cynicism == pytest.approx(expected_cynicism, abs=0.01)

        expected_wit = (
            mushroomer.wit
            + PERSONALITY_INHERITANCE_WEIGHT * 0.3
        )
        assert profile.wit == pytest.approx(expected_wit, abs=0.01)

    def test_negative_drift_shifts_traits_down(self):
        """Parent drift toward lower values shifts child traits downward."""
        legacy = GeneticLegacy(
            personality_drift={"patience": -0.3},
            generation_number=1,
        )

        profile = InheritanceEngine.compute_inherited_traits(legacy)
        mushroomer = STAGE_DEFAULTS[CreatureStage.MUSHROOMER]

        expected_patience = (
            mushroomer.patience
            + PERSONALITY_INHERITANCE_WEIGHT * (-0.3)
        )
        assert profile.patience == pytest.approx(expected_patience, abs=0.01)

    def test_inherited_traits_are_clamped(self):
        """Inherited traits stay within [0, 1] even with extreme drift."""
        # Extreme positive drift
        legacy = GeneticLegacy(
            personality_drift={name: 5.0 for name in ("cynicism", "wit", "warmth")},
            generation_number=1,
        )

        profile = InheritanceEngine.compute_inherited_traits(legacy)
        for name in profile.to_dict():
            assert 0.0 <= getattr(profile, name) <= 1.0


# =====================================================================
# Full hatch_from_legacy integration
# =====================================================================


class TestHatchFromLegacy:
    """Integration tests for the full hatching process."""

    def test_new_creature_starts_at_mushroomer(self):
        """Hatched creature always starts at MUSHROOMER stage."""
        parent_genome = generate_random_genome(CreatureStage.FROGMAN, rng=random.Random(42))
        legacy = GeneticLegacy(
            genome=parent_genome.to_dict(),
            distilled_memories=["Some memory"],
            personality_drift={"cynicism": 0.1},
            cause_of_death="old_age",
            generation_number=5,
            stage_reached="frogman",
        )

        engine = InheritanceEngine(rng=random.Random(99))
        new_state, child_genome, memories = engine.hatch_from_legacy(legacy)

        assert new_state.stage == CreatureStage.MUSHROOMER
        assert new_state.age == 0.0
        assert new_state.interaction_count == 0
        assert new_state.trust_level == 0.0

    def test_hatched_creature_has_genome(self):
        """New creature state includes the child genome."""
        legacy = GeneticLegacy(
            genome=generate_random_genome(rng=random.Random(42)).to_dict(),
            generation_number=1,
        )

        engine = InheritanceEngine(rng=random.Random(99))
        new_state, child_genome, _ = engine.hatch_from_legacy(legacy)

        assert new_state.genome is not None
        assert new_state.genome.traits == child_genome.traits

    def test_empty_genome_legacy_still_works(self):
        """Legacy with empty genome dict still produces a valid child."""
        legacy = GeneticLegacy(
            genome={},
            generation_number=1,
        )

        engine = InheritanceEngine(rng=random.Random(42))
        new_state, child_genome, _ = engine.hatch_from_legacy(legacy)

        # CreatureGenome fills missing traits with 0.5
        for trait in ALL_TRAITS:
            assert trait in child_genome.traits
            assert 0.0 <= child_genome.traits[trait] <= 1.0
