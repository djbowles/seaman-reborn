"""Personality trait dimensions and stage profiles.

Defines 8 personality trait dimensions as floats in [0.0, 1.0] and provides
default TraitProfiles for each CreatureStage. Traits can be loaded from
stage-specific TOML config files via the config module.
"""

from __future__ import annotations

from dataclasses import dataclass

from seaman_brain.config import PersonalityConfig, load_stage_config
from seaman_brain.types import CreatureStage

# The 8 canonical trait dimension names.
TRAIT_NAMES: tuple[str, ...] = (
    "cynicism",
    "wit",
    "patience",
    "curiosity",
    "warmth",
    "verbosity",
    "formality",
    "aggression",
)


def _clamp(value: float) -> float:
    """Clamp a float to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


@dataclass
class TraitProfile:
    """A set of 8 personality trait dimensions, each in [0.0, 1.0].

    Higher values indicate stronger expression of that trait:
    - cynicism: distrust / sardonic worldview
    - wit: clever humor and wordplay
    - patience: tolerance before irritation
    - curiosity: interest in learning / asking questions
    - warmth: affection toward the human
    - verbosity: length and detail in responses
    - formality: politeness and structure in speech
    - aggression: hostility and combativeness
    """

    cynicism: float = 0.5
    wit: float = 0.5
    patience: float = 0.5
    curiosity: float = 0.5
    warmth: float = 0.5
    verbosity: float = 0.5
    formality: float = 0.5
    aggression: float = 0.5

    def __post_init__(self) -> None:
        """Clamp all trait values to [0.0, 1.0]."""
        for name in TRAIT_NAMES:
            setattr(self, name, _clamp(getattr(self, name)))

    def to_dict(self) -> dict[str, float]:
        """Return traits as a dictionary."""
        return {name: getattr(self, name) for name in TRAIT_NAMES}

    @classmethod
    def from_dict(cls, traits: dict[str, float]) -> TraitProfile:
        """Create a TraitProfile from a dict, ignoring unknown keys.

        Values are clamped to [0.0, 1.0] via __post_init__.

        Args:
            traits: Dictionary mapping trait names to float values.

        Returns:
            A new TraitProfile with the specified values.
        """
        known = {k: v for k, v in traits.items() if k in TRAIT_NAMES}
        return cls(**known)


# Default trait profiles for each evolutionary stage.
# These are fallbacks when stage TOML files are unavailable.
STAGE_DEFAULTS: dict[CreatureStage, TraitProfile] = {
    CreatureStage.MUSHROOMER: TraitProfile(
        cynicism=0.5, wit=0.3, patience=0.2, curiosity=0.4,
        warmth=0.1, verbosity=0.2, formality=0.1, aggression=0.6,
    ),
    CreatureStage.GILLMAN: TraitProfile(
        cynicism=0.6, wit=0.5, patience=0.3, curiosity=0.5,
        warmth=0.15, verbosity=0.35, formality=0.15, aggression=0.5,
    ),
    CreatureStage.PODFISH: TraitProfile(
        cynicism=0.8, wit=0.9, patience=0.3, curiosity=0.7,
        warmth=0.2, verbosity=0.5, formality=0.2, aggression=0.4,
    ),
    CreatureStage.TADMAN: TraitProfile(
        cynicism=0.7, wit=0.8, patience=0.5, curiosity=0.8,
        warmth=0.35, verbosity=0.6, formality=0.3, aggression=0.3,
    ),
    CreatureStage.FROGMAN: TraitProfile(
        cynicism=0.6, wit=0.9, patience=0.7, curiosity=0.9,
        warmth=0.5, verbosity=0.7, formality=0.4, aggression=0.2,
    ),
}


def get_default_profile(stage: CreatureStage) -> TraitProfile:
    """Get the hardcoded default TraitProfile for a given stage.

    Args:
        stage: The creature's current evolutionary stage.

    Returns:
        A copy of the default TraitProfile for that stage.
    """
    defaults = STAGE_DEFAULTS[stage]
    return TraitProfile(**defaults.to_dict())


def load_trait_profile(
    stage: CreatureStage,
    config_dir: str = "config",
) -> TraitProfile:
    """Load a TraitProfile for a stage from TOML config files.

    Reads the stage-specific TOML file via load_stage_config(). Falls back
    to the hardcoded STAGE_DEFAULTS if no TOML file exists or if its
    [traits] section is empty.

    Args:
        stage: The creature's current evolutionary stage.
        config_dir: Path to the configuration directory.

    Returns:
        TraitProfile with values from TOML or defaults.
    """
    stage_config = load_stage_config(stage.value, config_dir)

    if stage_config.traits:
        # Start from defaults, then overlay TOML values
        base = STAGE_DEFAULTS[stage].to_dict()
        base.update(stage_config.traits)
        return TraitProfile.from_dict(base)

    return get_default_profile(stage)


def profile_from_config(config: PersonalityConfig) -> TraitProfile:
    """Create a TraitProfile from a PersonalityConfig's base_traits.

    This is useful when working with already-merged config (e.g. from
    load_config_with_stage()) where traits are in the config object.

    Args:
        config: PersonalityConfig with base_traits dict.

    Returns:
        TraitProfile constructed from config's base_traits.
    """
    return TraitProfile.from_dict(config.base_traits)
