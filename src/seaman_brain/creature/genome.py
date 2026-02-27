"""Creature genome and heritable physical trait system.

Each creature has a genome — a dict of float values (0.0-1.0) representing
heritable physical traits. Traits are grouped into morphological, chromatic,
and behavioral categories. Genomes support random generation, mutation for
inheritance, and epigenetic drift during life based on care quality.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Trait name constants grouped by category
# ---------------------------------------------------------------------------

MORPHOLOGICAL_TRAITS: tuple[str, ...] = (
    "body_size",
    "head_ratio",
    "eye_size",
    "fin_length",
    "limb_proportion",
)

CHROMATIC_TRAITS: tuple[str, ...] = (
    "hue",
    "saturation",
    "pattern_complexity",
)

BEHAVIORAL_TRAITS: tuple[str, ...] = (
    "metabolism_rate",
    "voice_pitch",
    "face_expressiveness",
    "aggression_baseline",
)

ALL_TRAITS: tuple[str, ...] = MORPHOLOGICAL_TRAITS + CHROMATIC_TRAITS + BEHAVIORAL_TRAITS

# ---------------------------------------------------------------------------
# Stage defaults — centre points for gaussian generation
# ---------------------------------------------------------------------------

STAGE_DEFAULTS: dict[CreatureStage, dict[str, float]] = {
    CreatureStage.MUSHROOMER: {
        "body_size": 0.3, "head_ratio": 0.6, "eye_size": 0.7,
        "fin_length": 0.2, "limb_proportion": 0.1,
        "hue": 0.5, "saturation": 0.4, "pattern_complexity": 0.2,
        "metabolism_rate": 0.6, "voice_pitch": 0.8,
        "face_expressiveness": 0.3, "aggression_baseline": 0.2,
    },
    CreatureStage.GILLMAN: {
        "body_size": 0.4, "head_ratio": 0.55, "eye_size": 0.6,
        "fin_length": 0.5, "limb_proportion": 0.2,
        "hue": 0.5, "saturation": 0.5, "pattern_complexity": 0.3,
        "metabolism_rate": 0.55, "voice_pitch": 0.7,
        "face_expressiveness": 0.4, "aggression_baseline": 0.3,
    },
    CreatureStage.PODFISH: {
        "body_size": 0.5, "head_ratio": 0.5, "eye_size": 0.5,
        "fin_length": 0.6, "limb_proportion": 0.4,
        "hue": 0.5, "saturation": 0.55, "pattern_complexity": 0.4,
        "metabolism_rate": 0.5, "voice_pitch": 0.6,
        "face_expressiveness": 0.5, "aggression_baseline": 0.35,
    },
    CreatureStage.TADMAN: {
        "body_size": 0.65, "head_ratio": 0.45, "eye_size": 0.45,
        "fin_length": 0.4, "limb_proportion": 0.6,
        "hue": 0.5, "saturation": 0.6, "pattern_complexity": 0.5,
        "metabolism_rate": 0.45, "voice_pitch": 0.5,
        "face_expressiveness": 0.6, "aggression_baseline": 0.4,
    },
    CreatureStage.FROGMAN: {
        "body_size": 0.8, "head_ratio": 0.4, "eye_size": 0.4,
        "fin_length": 0.3, "limb_proportion": 0.8,
        "hue": 0.5, "saturation": 0.65, "pattern_complexity": 0.6,
        "metabolism_rate": 0.4, "voice_pitch": 0.4,
        "face_expressiveness": 0.7, "aggression_baseline": 0.45,
    },
}


def _clamp01(value: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ---------------------------------------------------------------------------
# CreatureGenome
# ---------------------------------------------------------------------------


@dataclass
class CreatureGenome:
    """A creature's heritable physical traits.

    Each trait is a float in [0.0, 1.0]. The traits dict always contains
    all traits defined in ALL_TRAITS — missing entries are filled with 0.5.
    """

    traits: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Ensure all traits exist and are clamped to [0, 1]."""
        for name in ALL_TRAITS:
            self.traits[name] = _clamp01(self.traits.get(name, 0.5))

    def to_dict(self) -> dict[str, float]:
        """Serialize genome traits to a plain dict."""
        return dict(self.traits)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreatureGenome:
        """Deserialize from a dict. Unknown keys are ignored."""
        traits = {k: float(v) for k, v in data.items() if k in ALL_TRAITS}
        return cls(traits=traits)


# ---------------------------------------------------------------------------
# Generation, mutation, and epigenetic drift
# ---------------------------------------------------------------------------


def generate_random_genome(
    stage: CreatureStage = CreatureStage.MUSHROOMER,
    *,
    std_dev: float = 0.1,
    rng: random.Random | None = None,
) -> CreatureGenome:
    """Create a new genome with gaussian-distributed traits around stage defaults.

    Args:
        stage: Evolutionary stage whose defaults form the distribution centres.
        std_dev: Standard deviation for the gaussian noise.
        rng: Optional Random instance for reproducibility.

    Returns:
        A new CreatureGenome with traits sampled around the stage defaults.
    """
    r = rng or random.Random()
    defaults = STAGE_DEFAULTS.get(stage, STAGE_DEFAULTS[CreatureStage.MUSHROOMER])
    traits: dict[str, float] = {}
    for name in ALL_TRAITS:
        centre = defaults.get(name, 0.5)
        traits[name] = _clamp01(r.gauss(centre, std_dev))
    return CreatureGenome(traits=traits)


def mutate_genome(
    parent: CreatureGenome,
    *,
    mutation_rate: float = 0.1,
    rng: random.Random | None = None,
) -> CreatureGenome:
    """Create a child genome by applying random drift to a parent.

    Each trait is shifted by gaussian noise with std_dev = mutation_rate.
    The result is always clamped to [0, 1].

    Args:
        parent: The parent genome to mutate from.
        mutation_rate: Standard deviation of the per-trait mutation noise.
        rng: Optional Random instance for reproducibility.

    Returns:
        A new CreatureGenome similar to but different from the parent.
    """
    r = rng or random.Random()
    child_traits: dict[str, float] = {}
    for name in ALL_TRAITS:
        parent_val = parent.traits.get(name, 0.5)
        child_traits[name] = _clamp01(parent_val + r.gauss(0.0, mutation_rate))
    return CreatureGenome(traits=child_traits)


def update_epigenetic_drift(
    genome: CreatureGenome,
    care_quality: float,
    environment_quality: float,
    elapsed_days: float,
) -> CreatureGenome:
    """Subtly shift genome traits based on life conditions.

    Well-cared creatures drift toward larger size and lower aggression.
    Neglected creatures drift toward higher aggression and lower expressiveness.

    The shift magnitude is proportional to elapsed_days and the deviation of
    care/environment from 0.5 (neutral). Maximum daily drift is ~0.005 per
    trait so changes are gradual.

    Args:
        genome: Current genome (not mutated in place).
        care_quality: Quality of care (0.0=neglected, 1.0=excellent).
        environment_quality: Tank environment quality (0.0=bad, 1.0=pristine).
        elapsed_days: Number of days over which drift accumulates.

    Returns:
        A new CreatureGenome with adjusted traits.
    """
    if elapsed_days <= 0:
        return CreatureGenome(traits=dict(genome.traits))

    # How far care/env deviate from neutral (range: -0.5 to +0.5)
    care_bias = care_quality - 0.5
    env_bias = environment_quality - 0.5

    # Max drift per day per trait
    daily_rate = 0.005

    # Per-trait directional rules
    drift_rules: dict[str, float] = {
        # Good care -> larger body, higher expressiveness
        "body_size": care_bias * daily_rate * elapsed_days,
        "face_expressiveness": care_bias * daily_rate * elapsed_days,
        # Good care -> lower aggression
        "aggression_baseline": -care_bias * daily_rate * elapsed_days,
        # Good environment -> brighter saturation, more pattern complexity
        "saturation": env_bias * daily_rate * elapsed_days,
        "pattern_complexity": env_bias * daily_rate * elapsed_days,
        # Good environment -> slightly larger eyes (alertness)
        "eye_size": env_bias * daily_rate * 0.5 * elapsed_days,
        # Metabolism adjusts to care (well-fed -> slower metabolism)
        "metabolism_rate": -care_bias * daily_rate * 0.5 * elapsed_days,
    }

    new_traits = dict(genome.traits)
    for name, shift in drift_rules.items():
        new_traits[name] = _clamp01(new_traits.get(name, 0.5) + shift)

    return CreatureGenome(traits=new_traits)
