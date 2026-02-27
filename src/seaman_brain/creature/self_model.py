"""Creature self-model and body awareness for LLM prompt injection.

The creature maintains a self-model — a natural language description of its
current physical state derived from its genome and evolutionary stage. This
self-model is injected into the LLM system prompt so the creature can reference
its own appearance, react to physical changes during evolution, and have
opinions about how it looks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seaman_brain.creature.genome import (
    BEHAVIORAL_TRAITS,
    CHROMATIC_TRAITS,
    MORPHOLOGICAL_TRAITS,
    CreatureGenome,
)
from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Trait-to-language mappings
# ---------------------------------------------------------------------------

# Each trait maps to (low_description, mid_description, high_description)
# thresholds: <0.33 = low, 0.33-0.66 = mid, >0.66 = high
_MORPHOLOGICAL_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "body_size": ("small and compact", "average-sized", "large and imposing"),
    "head_ratio": (
        "with a proportionally small head",
        "with an average head",
        "with a disproportionately large head",
    ),
    "eye_size": (
        "with beady, squinting eyes",
        "with modest eyes",
        "with huge, bulging eyes",
    ),
    "fin_length": (
        "with stubby fins",
        "with modest fins",
        "with long, flowing fins",
    ),
    "limb_proportion": (
        "with barely visible limb buds",
        "with developing limbs",
        "with prominent, well-formed limbs",
    ),
}

_CHROMATIC_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "hue": ("cool blue-green", "murky green-brown", "warm orange-red"),
    "saturation": ("pale, washed-out", "moderately colored", "vividly colored"),
    "pattern_complexity": (
        "with plain, unmarked skin",
        "with faint patterning",
        "with intricate, striking patterns",
    ),
}

_BEHAVIORAL_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "metabolism_rate": ("sluggish", "steady", "hyperactive"),
    "voice_pitch": ("deep and rumbling", "mid-toned", "high and squeaky"),
    "face_expressiveness": (
        "with a blank, unreadable face",
        "with a somewhat expressive face",
        "with a wildly expressive face",
    ),
    "aggression_baseline": ("docile", "cautious", "aggressive-looking"),
}

# Stage-specific self-awareness depth
_STAGE_AWARENESS: dict[CreatureStage, str] = {
    CreatureStage.MUSHROOMER: "minimal",
    CreatureStage.GILLMAN: "basic",
    CreatureStage.PODFISH: "moderate",
    CreatureStage.TADMAN: "detailed",
    CreatureStage.FROGMAN: "rich",
}

# Stage flavor text for framing the self-perception
_STAGE_FRAME: dict[CreatureStage, str] = {
    CreatureStage.MUSHROOMER: (
        "You are dimly aware of your body. You sense vague shapes and textures "
        "but cannot articulate much about yourself."
    ),
    CreatureStage.GILLMAN: (
        "You are beginning to notice your own body. You can see your reflection "
        "in the tank glass and are starting to form opinions."
    ),
    CreatureStage.PODFISH: (
        "You are keenly aware of your physical form. You study yourself in the "
        "tank glass frequently and have strong opinions about your appearance."
    ),
    CreatureStage.TADMAN: (
        "You have a sophisticated understanding of your body and how it has "
        "changed. You notice subtle shifts and contemplate their meaning."
    ),
    CreatureStage.FROGMAN: (
        "You have complete awareness of your body, its history, and how it "
        "reflects your life experiences. You can articulate your physical self "
        "with nuance and philosophical depth."
    ),
}


def _describe_trait(
    value: float,
    descriptions: tuple[str, str, str],
) -> str:
    """Map a 0.0-1.0 trait value to one of three description levels."""
    if value < 0.33:
        return descriptions[0]
    if value <= 0.66:
        return descriptions[1]
    return descriptions[2]


def _get_trait_level(value: float) -> str:
    """Return 'low', 'mid', or 'high' for a trait value."""
    if value < 0.33:
        return "low"
    if value <= 0.66:
        return "mid"
    return "high"


# ---------------------------------------------------------------------------
# SelfModel
# ---------------------------------------------------------------------------


@dataclass
class SelfModel:
    """The creature's internal representation of its own physical appearance.

    Attributes:
        description: Current natural-language self-description.
        new_changes: List of changes detected since last update.
        previous_levels: Trait levels from the last update, for change detection.
    """

    description: str = ""
    new_changes: list[str] = field(default_factory=list)
    previous_levels: dict[str, str] = field(default_factory=dict)


def build_self_description(
    genome: CreatureGenome,
    stage: CreatureStage,
    previous_description: str | None = None,
    *,
    self_model: SelfModel | None = None,
) -> SelfModel:
    """Generate a natural-language body description from genome + stage.

    The detail level varies by stage — Mushroomer gets minimal awareness,
    Frogman gets rich self-knowledge.

    Args:
        genome: The creature's current genome.
        stage: The creature's current evolutionary stage.
        previous_description: The previous self-description (unused directly,
            kept for API compatibility). Change detection uses self_model.
        self_model: Existing SelfModel to compare against for change detection.
            If None, a fresh model is created with no change history.

    Returns:
        A new SelfModel with updated description and detected changes.
    """
    old_levels = self_model.previous_levels if self_model else {}

    # Build current trait level map
    current_levels: dict[str, str] = {}
    for trait_name in (*MORPHOLOGICAL_TRAITS, *CHROMATIC_TRAITS, *BEHAVIORAL_TRAITS):
        value = genome.traits.get(trait_name, 0.5)
        current_levels[trait_name] = _get_trait_level(value)

    # Detect changes
    new_changes: list[str] = []
    if old_levels:
        new_changes = _detect_changes(old_levels, current_levels, genome)

    # Build the description text with stage-appropriate detail
    description = _build_description_text(genome, stage, current_levels)

    return SelfModel(
        description=description,
        new_changes=new_changes,
        previous_levels=current_levels,
    )


def _detect_changes(
    old_levels: dict[str, str],
    current_levels: dict[str, str],
    genome: CreatureGenome,
) -> list[str]:
    """Detect physical changes between two sets of trait levels."""
    changes: list[str] = []

    change_descriptions: dict[str, dict[str, str]] = {
        "body_size": {"increased": "body has grown larger", "decreased": "body has shrunk"},
        "head_ratio": {
            "increased": "head appears more prominent",
            "decreased": "head looks more proportional",
        },
        "eye_size": {"increased": "eyes have grown larger", "decreased": "eyes have narrowed"},
        "fin_length": {"increased": "fins have grown longer", "decreased": "fins have receded"},
        "limb_proportion": {
            "increased": "limbs have become more developed",
            "decreased": "limbs have receded",
        },
        "hue": {"increased": "coloring has warmed", "decreased": "coloring has cooled"},
        "saturation": {
            "increased": "colors have become more vivid",
            "decreased": "colors have faded",
        },
        "pattern_complexity": {
            "increased": "skin patterns have become more intricate",
            "decreased": "skin patterns have simplified",
        },
        "metabolism_rate": {
            "increased": "metabolism has quickened",
            "decreased": "metabolism has slowed",
        },
        "voice_pitch": {
            "increased": "voice has risen in pitch",
            "decreased": "voice has deepened",
        },
        "face_expressiveness": {
            "increased": "face has become more expressive",
            "decreased": "face has become more stoic",
        },
        "aggression_baseline": {
            "increased": "demeanor has become more aggressive",
            "decreased": "demeanor has become calmer",
        },
    }

    level_order = {"low": 0, "mid": 1, "high": 2}

    for trait_name, descriptions in change_descriptions.items():
        old = old_levels.get(trait_name)
        new = current_levels.get(trait_name)
        if old and new and old != new:
            old_ord = level_order[old]
            new_ord = level_order[new]
            direction = "increased" if new_ord > old_ord else "decreased"
            changes.append(descriptions[direction])

    return changes


def _build_description_text(
    genome: CreatureGenome,
    stage: CreatureStage,
    current_levels: dict[str, str],
) -> str:
    """Build the full description text at the appropriate detail level."""
    awareness = _STAGE_AWARENESS.get(stage, "basic")
    parts: list[str] = []

    # Always include the stage frame
    frame = _STAGE_FRAME.get(stage, _STAGE_FRAME[CreatureStage.MUSHROOMER])
    parts.append(frame)

    # Mushroomer: minimal — just body size and one chromatic trait
    if awareness == "minimal":
        body_size = genome.traits.get("body_size", 0.5)
        desc = _describe_trait(body_size, _MORPHOLOGICAL_DESCRIPTIONS["body_size"])
        parts.append(f"You are {desc}.")
        return " ".join(parts)

    # Basic (Gillman): morphological traits only
    morph_parts = _describe_morphological(genome)
    parts.append(morph_parts)

    if awareness == "basic":
        return " ".join(parts)

    # Moderate (Podfish): add chromatic traits
    chrom_parts = _describe_chromatic(genome)
    parts.append(chrom_parts)

    if awareness == "moderate":
        return " ".join(parts)

    # Detailed (Tadman): add behavioral traits
    behav_parts = _describe_behavioral(genome)
    parts.append(behav_parts)

    if awareness == "detailed":
        return " ".join(parts)

    # Rich (Frogman): everything plus a synthesis
    parts.append(
        "You understand how these traits connect — your body tells the story "
        "of how you were raised and what you have endured."
    )
    return " ".join(parts)


def _describe_morphological(genome: CreatureGenome) -> str:
    """Generate a sentence describing morphological traits."""
    parts: list[str] = []
    for trait_name, descriptions in _MORPHOLOGICAL_DESCRIPTIONS.items():
        value = genome.traits.get(trait_name, 0.5)
        parts.append(_describe_trait(value, descriptions))

    # Combine into a natural sentence
    body = parts[0]  # body_size
    modifiers = [p for p in parts[1:] if p]
    if modifiers:
        return f"You are {body}, {', '.join(modifiers)}."
    return f"You are {body}."


def _describe_chromatic(genome: CreatureGenome) -> str:
    """Generate a sentence describing chromatic traits."""
    hue = _describe_trait(
        genome.traits.get("hue", 0.5),
        _CHROMATIC_DESCRIPTIONS["hue"],
    )
    sat = _describe_trait(
        genome.traits.get("saturation", 0.5),
        _CHROMATIC_DESCRIPTIONS["saturation"],
    )
    pattern = _describe_trait(
        genome.traits.get("pattern_complexity", 0.5),
        _CHROMATIC_DESCRIPTIONS["pattern_complexity"],
    )
    return f"Your skin is {sat} {hue}, {pattern}."


def _describe_behavioral(genome: CreatureGenome) -> str:
    """Generate a sentence describing behavioral traits."""
    meta = _describe_trait(
        genome.traits.get("metabolism_rate", 0.5),
        _BEHAVIORAL_DESCRIPTIONS["metabolism_rate"],
    )
    voice = _describe_trait(
        genome.traits.get("voice_pitch", 0.5),
        _BEHAVIORAL_DESCRIPTIONS["voice_pitch"],
    )
    face = _describe_trait(
        genome.traits.get("face_expressiveness", 0.5),
        _BEHAVIORAL_DESCRIPTIONS["face_expressiveness"],
    )
    agg = _describe_trait(
        genome.traits.get("aggression_baseline", 0.5),
        _BEHAVIORAL_DESCRIPTIONS["aggression_baseline"],
    )
    return f"Your movements are {meta}, your voice is {voice}, {face}, and you appear {agg}."


# ---------------------------------------------------------------------------
# Prompt injection
# ---------------------------------------------------------------------------


def get_prompt_injection(self_model: SelfModel) -> str:
    """Format a self-model as a text block for system prompt insertion.

    Args:
        self_model: The creature's current self-model.

    Returns:
        A formatted text block suitable for inclusion in a system prompt.
        Returns empty string if the self-model has no description.
    """
    if not self_model.description:
        return ""

    parts: list[str] = [
        "[YOUR BODY]",
        self_model.description,
    ]

    if self_model.new_changes:
        parts.append("")
        parts.append("[RECENT PHYSICAL CHANGES]")
        for change in self_model.new_changes:
            parts.append(f"- Your {change}")

    return "\n".join(parts)
