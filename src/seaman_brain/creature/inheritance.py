"""Generational inheritance and mutation.

When a creature dies, its GeneticLegacy (from genetics.py) seeds the next
generation. The InheritanceEngine handles:

- Genome mutation: gaussian drift per trait, clamped to [0,1].
- Ancestral memory fading: parent memories become vague recollections
  with confidence that decays across generations.
- Personality baseline: new creature traits shifted 30% toward parent's
  final personality.
- New creature always starts at MUSHROOMER stage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from seaman_brain.creature.genetics import GeneticLegacy
from seaman_brain.creature.genome import (
    CreatureGenome,
    mutate_genome,
)
from seaman_brain.creature.state import CreatureState
from seaman_brain.personality.traits import STAGE_DEFAULTS, TraitProfile
from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Ancestral memory fading templates
# ---------------------------------------------------------------------------

VAGUE_PREFIXES: tuple[str, ...] = (
    "I have a dim memory that",
    "Something tells me that",
    "There's a faint echo in my mind that",
    "I vaguely recall that",
    "A hazy impression suggests that",
)

# Confidence decay factor per generation gap.
# Confidence = base_confidence * (DECAY_PER_GENERATION ** generation_gap)
DECAY_PER_GENERATION: float = 0.5

# Personality inheritance strength: new creature starts with traits
# shifted this fraction toward the parent's final traits.
PERSONALITY_INHERITANCE_WEIGHT: float = 0.3


# ---------------------------------------------------------------------------
# AncestralMemory
# ---------------------------------------------------------------------------


@dataclass
class AncestralMemory:
    """A faded memory inherited from a previous generation.

    Attributes:
        original_fact: The original factual statement from the ancestor.
        faded_version: A vaguer paraphrase suitable for prompt injection.
        confidence: How confident the creature is in this memory (0.0-1.0).
            Decays with each generation gap.
        source_generation: Which generation originally produced this memory.
    """

    original_fact: str = ""
    faded_version: str = ""
    confidence: float = 1.0
    source_generation: int = 1


# ---------------------------------------------------------------------------
# InheritanceEngine
# ---------------------------------------------------------------------------


class InheritanceEngine:
    """Manages the birth of a new creature from a GeneticLegacy.

    The engine takes the dead creature's legacy and produces:
    - A mutated genome for the child.
    - A fresh CreatureState at MUSHROOMER stage.
    - A list of AncestralMemory objects with decayed confidence.

    Args:
        mutation_rate: Standard deviation of gaussian noise per genome trait.
        rng: Optional Random instance for reproducible results.
    """

    def __init__(
        self,
        mutation_rate: float = 0.1,
        rng: random.Random | None = None,
    ) -> None:
        self._mutation_rate = mutation_rate
        self._rng = rng or random.Random()

    def hatch_from_legacy(
        self,
        legacy: GeneticLegacy,
    ) -> tuple[CreatureState, CreatureGenome, list[AncestralMemory]]:
        """Create a new creature from a dead parent's genetic legacy.

        Steps:
        1. Mutate the parent genome to produce a child genome.
        2. Create AncestralMemory objects from the parent's distilled memories,
           with confidence decayed based on generational distance.
        3. Compute a personality-shifted initial trait profile (30% toward parent).
        4. Return a fresh MUSHROOMER-stage CreatureState.

        Args:
            legacy: The parent's GeneticLegacy extracted on death.

        Returns:
            Tuple of (new_creature_state, child_genome, ancestral_memories).
        """
        # 1. Mutate genome
        parent_genome = CreatureGenome.from_dict(legacy.genome)
        child_genome = mutate_genome(
            parent_genome,
            mutation_rate=self._mutation_rate,
            rng=self._rng,
        )

        # 2. Build ancestral memories with fading
        ancestral_memories = self._build_ancestral_memories(legacy)

        # 3. Create new creature state at MUSHROOMER
        new_state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            genome=child_genome,
        )

        return new_state, child_genome, ancestral_memories

    def _build_ancestral_memories(
        self,
        legacy: GeneticLegacy,
    ) -> list[AncestralMemory]:
        """Convert distilled memories into faded ancestral recollections.

        Each memory gets:
        - A vague prefix chosen randomly.
        - Confidence decayed by DECAY_PER_GENERATION for each generation gap.

        Args:
            legacy: The parent's legacy containing distilled memories.

        Returns:
            List of AncestralMemory objects.
        """
        memories: list[AncestralMemory] = []
        parent_gen = legacy.generation_number
        child_gen = parent_gen + 1

        for fact in legacy.distilled_memories:
            if not fact or not fact.strip():
                continue

            prefix = self._rng.choice(VAGUE_PREFIXES)
            # Lowercase the first character of the fact for natural flow
            faded_fact = fact[0].lower() + fact[1:] if len(fact) > 1 else fact.lower()
            faded_version = f"{prefix} {faded_fact}"

            # Confidence decays based on generational distance
            gen_gap = child_gen - parent_gen  # 1 for direct child
            confidence = 1.0 * (DECAY_PER_GENERATION ** gen_gap)

            memories.append(AncestralMemory(
                original_fact=fact,
                faded_version=faded_version,
                confidence=round(confidence, 4),
                source_generation=parent_gen,
            ))

        return memories

    @staticmethod
    def compute_inherited_traits(
        legacy: GeneticLegacy,
    ) -> TraitProfile:
        """Compute the personality baseline for a new creature.

        The new creature starts with MUSHROOMER default traits shifted
        30% toward the parent's final trait values (default + drift).

        Args:
            legacy: The parent's legacy with personality_drift.

        Returns:
            TraitProfile for the new creature.
        """
        mushroomer_defaults = STAGE_DEFAULTS[CreatureStage.MUSHROOMER]
        base = mushroomer_defaults.to_dict()

        # Reconstruct parent's final traits from drift
        parent_final: dict[str, float] = {}
        for name, default_val in base.items():
            drift = legacy.personality_drift.get(name, 0.0)
            parent_final[name] = default_val + drift

        # Shift 30% toward parent's final traits
        inherited: dict[str, float] = {}
        for name in base:
            default_val = base[name]
            parent_val = parent_final[name]
            inherited[name] = (
                default_val + PERSONALITY_INHERITANCE_WEIGHT * (parent_val - default_val)
            )

        return TraitProfile.from_dict(inherited)

    @staticmethod
    def fade_across_generations(
        memories: list[AncestralMemory],
        rng: random.Random | None = None,
    ) -> list[AncestralMemory]:
        """Further decay ancestral memories for the next generation.

        Used when memories are passed down through multiple generations.
        Each memory's confidence is multiplied by DECAY_PER_GENERATION and
        the faded_version gets a new vague prefix.

        Args:
            memories: Existing ancestral memories to further fade.
            rng: Optional Random for reproducibility.

        Returns:
            New list of AncestralMemory with reduced confidence.
            Memories below 0.05 confidence are dropped entirely.
        """
        r = rng or random.Random()
        faded: list[AncestralMemory] = []

        for mem in memories:
            new_confidence = mem.confidence * DECAY_PER_GENERATION
            if new_confidence < 0.05:
                continue  # Too faded to remember

            prefix = r.choice(VAGUE_PREFIXES)
            fact = mem.original_fact
            faded_fact = fact[0].lower() + fact[1:] if len(fact) > 1 else fact.lower()
            new_faded = f"{prefix} {faded_fact}"

            faded.append(AncestralMemory(
                original_fact=mem.original_fact,
                faded_version=new_faded,
                confidence=round(new_confidence, 4),
                source_generation=mem.source_generation,
            ))

        return faded
