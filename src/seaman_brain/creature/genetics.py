"""Genetic material extraction on death.

When a creature dies, its genetic legacy is extracted — the final genome,
distilled memories, personality drift, behavioral patterns, and death cause.
This legacy is saved to data/saves/lineage/ and becomes the seed for the
next generation (US-050).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from seaman_brain.creature.genome import CreatureGenome
from seaman_brain.creature.state import CreatureState
from seaman_brain.needs.death import DeathCause
from seaman_brain.personality.traits import STAGE_DEFAULTS, TraitProfile

if TYPE_CHECKING:
    from seaman_brain.llm.base import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Memory distillation prompt
# ---------------------------------------------------------------------------

DISTILLATION_PROMPT = """\
You are a memory distillation engine. Below is a list of memories from a \
creature's lifetime. Compress these into the {max_facts} most important core \
facts about the human who cared for this creature. Focus on:

- The human's name, identity, and personal details
- The human's preferences, habits, and personality
- Key emotional moments or relationship milestones
- Recurring topics of conversation

Return ONLY a newline-separated list of short factual statements. \
No numbering, no bullets, no commentary.

Memories:
{memories}"""


# ---------------------------------------------------------------------------
# GeneticLegacy
# ---------------------------------------------------------------------------


@dataclass
class GeneticLegacy:
    """A dead creature's genetic legacy — the seed for the next generation.

    Attributes:
        genome: The creature's final genome (dict of trait floats).
        distilled_memories: Top facts about the human, distilled from
            the creature's semantic memory store.
        personality_drift: Delta between the starting stage-default
            TraitProfile and the creature's final trait values.
        behavioral_patterns: Summary stats about the creature's behavior
            during its life.
        cause_of_death: What killed the creature.
        generation_number: Which generation this creature was (1-based).
        lifespan_days: How many days the creature lived.
        stage_reached: Highest evolutionary stage attained.
        trust_at_death: Trust level at time of death (0.0-1.0).
    """

    genome: dict[str, float] = field(default_factory=dict)
    distilled_memories: list[str] = field(default_factory=list)
    personality_drift: dict[str, float] = field(default_factory=dict)
    behavioral_patterns: dict[str, Any] = field(default_factory=dict)
    cause_of_death: str = "unknown"
    generation_number: int = 1
    lifespan_days: float = 0.0
    stage_reached: str = "mushroomer"
    trust_at_death: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "genome": dict(self.genome),
            "distilled_memories": list(self.distilled_memories),
            "personality_drift": dict(self.personality_drift),
            "behavioral_patterns": dict(self.behavioral_patterns),
            "cause_of_death": self.cause_of_death,
            "generation_number": self.generation_number,
            "lifespan_days": self.lifespan_days,
            "stage_reached": self.stage_reached,
            "trust_at_death": self.trust_at_death,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneticLegacy:
        """Deserialize from a dict. Unknown keys are silently ignored."""
        return cls(
            genome=dict(data.get("genome", {})),
            distilled_memories=list(data.get("distilled_memories", [])),
            personality_drift=dict(data.get("personality_drift", {})),
            behavioral_patterns=dict(data.get("behavioral_patterns", {})),
            cause_of_death=str(data.get("cause_of_death", "unknown")),
            generation_number=int(data.get("generation_number", 1)),
            lifespan_days=float(data.get("lifespan_days", 0.0)),
            stage_reached=str(data.get("stage_reached", "mushroomer")),
            trust_at_death=float(data.get("trust_at_death", 0.0)),
        )


# ---------------------------------------------------------------------------
# Memory distillation
# ---------------------------------------------------------------------------


async def distill_memories_llm(
    memories: list[str],
    llm: LLMProvider,
    max_facts: int = 20,
) -> list[str]:
    """Use an LLM to compress a list of memories into core facts.

    Args:
        memories: Raw memory texts from the creature's lifetime.
        llm: LLM provider for distillation.
        max_facts: Target number of distilled facts.

    Returns:
        List of distilled fact strings.
    """
    from seaman_brain.types import ChatMessage, MessageRole

    if not memories:
        return []

    memories_text = "\n".join(f"- {m}" for m in memories)
    prompt = DISTILLATION_PROMPT.format(
        max_facts=max_facts,
        memories=memories_text,
    )

    try:
        response = await llm.chat([
            ChatMessage(role=MessageRole.USER, content=prompt),
        ])
    except Exception:
        logger.warning("LLM distillation failed, falling back to heuristic")
        return distill_memories_heuristic(memories, max_facts=max_facts)

    if not response or not response.strip():
        return distill_memories_heuristic(memories, max_facts=max_facts)

    facts = [line.strip() for line in response.strip().splitlines()]
    facts = [f for f in facts if f and f.upper() != "NONE"]
    return facts[:max_facts]


def distill_memories_heuristic(
    memories: list[str],
    max_facts: int = 20,
) -> list[str]:
    """Heuristic fallback: pick the longest and most unique memories.

    Prefers longer memories (more info) and deduplicates by checking
    substring overlap. Returns up to max_facts.

    Args:
        memories: Raw memory texts.
        max_facts: Maximum number of facts to return.

    Returns:
        List of selected memory strings.
    """
    if not memories:
        return []

    # Sort by length descending — longer memories tend to be more informative
    sorted_mems = sorted(memories, key=len, reverse=True)

    selected: list[str] = []
    for mem in sorted_mems:
        if len(selected) >= max_facts:
            break
        # Skip near-duplicates
        mem_lower = mem.lower()
        is_dup = any(
            mem_lower in s.lower() or s.lower() in mem_lower
            for s in selected
        )
        if not is_dup:
            selected.append(mem)

    return selected


# ---------------------------------------------------------------------------
# Personality drift computation
# ---------------------------------------------------------------------------


def compute_personality_drift(
    final_traits: TraitProfile,
    stage_reached: str,
) -> dict[str, float]:
    """Compute the delta between stage-default traits and final traits.

    A positive drift means the trait grew during life; negative means
    it shrank.

    Args:
        final_traits: The creature's trait profile at death.
        stage_reached: The creature's stage as a string value.

    Returns:
        Dict mapping trait name to drift (final - default).
    """
    from seaman_brain.types import CreatureStage

    try:
        stage = CreatureStage(stage_reached)
    except ValueError:
        stage = CreatureStage.MUSHROOMER

    defaults = STAGE_DEFAULTS.get(stage, STAGE_DEFAULTS[CreatureStage.MUSHROOMER])
    final_dict = final_traits.to_dict()
    default_dict = defaults.to_dict()

    return {
        name: round(final_dict.get(name, 0.5) - default_dict.get(name, 0.5), 4)
        for name in final_dict
    }


# ---------------------------------------------------------------------------
# Behavioral patterns extraction
# ---------------------------------------------------------------------------


def extract_behavioral_patterns(
    creature_state: CreatureState,
) -> dict[str, Any]:
    """Summarize the creature's behavioral patterns from its state.

    Args:
        creature_state: The creature's state at death.

    Returns:
        Dict of behavioral summary stats.
    """
    now = datetime.now(UTC)
    age_days = creature_state.age / 86400.0

    return {
        "total_interactions": creature_state.interaction_count,
        "interactions_per_day": (
            creature_state.interaction_count / max(age_days, 0.01)
        ),
        "final_mood": creature_state.mood,
        "final_hunger": creature_state.hunger,
        "final_health": creature_state.health,
        "final_comfort": creature_state.comfort,
        "age_days": round(age_days, 2),
        "last_interaction": creature_state.last_interaction.isoformat(),
        "birth_time": creature_state.birth_time.isoformat(),
        "death_time": now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Legacy extraction (main entry point)
# ---------------------------------------------------------------------------


async def extract_legacy(
    creature_state: CreatureState,
    genome: CreatureGenome,
    memories: list[str],
    personality: TraitProfile,
    death_cause: DeathCause,
    *,
    llm: LLMProvider | None = None,
    generation_number: int = 1,
    max_distilled: int = 20,
) -> GeneticLegacy:
    """Extract a full genetic legacy from a dying creature.

    This is the main entry point called on death. It collects the genome,
    distills memories, computes personality drift, and packages everything
    into a GeneticLegacy artifact.

    Args:
        creature_state: The creature's state at time of death.
        genome: The creature's genome.
        memories: List of memory text strings from semantic memory.
        personality: The creature's final trait profile.
        death_cause: What killed the creature.
        llm: Optional LLM provider for memory distillation (falls back
            to heuristic if None or if LLM call fails).
        generation_number: Which generation this creature was.
        max_distilled: Max number of distilled memory facts.

    Returns:
        A complete GeneticLegacy ready for saving.
    """
    # Distill memories
    if llm is not None and memories:
        distilled = await distill_memories_llm(
            memories, llm, max_facts=max_distilled,
        )
    else:
        distilled = distill_memories_heuristic(
            memories, max_facts=max_distilled,
        )

    # Compute personality drift
    drift = compute_personality_drift(
        personality, creature_state.stage.value,
    )

    # Extract behavioral patterns
    patterns = extract_behavioral_patterns(creature_state)

    # Compute lifespan in days
    lifespan_days = creature_state.age / 86400.0

    return GeneticLegacy(
        genome=genome.to_dict(),
        distilled_memories=distilled,
        personality_drift=drift,
        behavioral_patterns=patterns,
        cause_of_death=death_cause.value,
        generation_number=generation_number,
        lifespan_days=round(lifespan_days, 2),
        stage_reached=creature_state.stage.value,
        trust_at_death=round(creature_state.trust_level, 4),
    )


# ---------------------------------------------------------------------------
# Save / load legacy files
# ---------------------------------------------------------------------------


def save_legacy(
    legacy: GeneticLegacy,
    lineage_dir: str | Path,
) -> Path:
    """Save a GeneticLegacy to a JSON file in the lineage directory.

    File is named gen_N.json where N is the generation number.

    Args:
        legacy: The legacy to save.
        lineage_dir: Directory for lineage files (e.g. data/saves/lineage/).

    Returns:
        Path to the saved file.

    Raises:
        OSError: If the file cannot be written.
    """
    dir_path = Path(lineage_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    filename = f"gen_{legacy.generation_number}.json"
    file_path = dir_path / filename

    file_path.write_text(
        json.dumps(legacy.to_dict(), indent=2),
        encoding="utf-8",
    )
    logger.info("Saved genetic legacy to %s", file_path)
    return file_path


def load_legacy(path: str | Path) -> GeneticLegacy:
    """Load a GeneticLegacy from a JSON file.

    Args:
        path: Path to the legacy JSON file.

    Returns:
        Deserialized GeneticLegacy.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    file_path = Path(path)
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return GeneticLegacy.from_dict(data)


def get_latest_legacy(lineage_dir: str | Path) -> GeneticLegacy | None:
    """Find and load the most recent legacy file in the lineage directory.

    Legacy files are named gen_N.json. Returns the one with the highest N.

    Args:
        lineage_dir: Directory containing lineage files.

    Returns:
        The latest GeneticLegacy, or None if no legacy files exist.
    """
    dir_path = Path(lineage_dir)
    if not dir_path.exists():
        return None

    gen_files = sorted(dir_path.glob("gen_*.json"))
    if not gen_files:
        return None

    return load_legacy(gen_files[-1])
