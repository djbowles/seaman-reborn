"""Lineage tracking and family tree across creature generations.

Tracks every generation's genome, lifespan, cause of death, stage reached,
notable memories, and generation number. The lineage feeds into the creature's
self-awareness ("I am the 4th of my line") and can be displayed in the GUI.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from seaman_brain.creature.genetics import GeneticLegacy

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LineageEntry
# ---------------------------------------------------------------------------


@dataclass
class LineageEntry:
    """A single entry in the creature family tree.

    Attributes:
        generation: 1-based generation number.
        genome_snapshot: Dict of trait name -> float at death.
        birth_time: ISO-formatted birth timestamp.
        death_time: ISO-formatted death timestamp.
        death_cause: String describing cause of death.
        stage_reached: Highest evolutionary stage attained (as string).
        trust_at_death: Trust level at time of death (0.0-1.0).
        notable_facts: Key distilled memories from the creature's life.
        personality_summary: Summary of personality drift during life.
    """

    generation: int = 1
    genome_snapshot: dict[str, float] = field(default_factory=dict)
    birth_time: str = ""
    death_time: str = ""
    death_cause: str = "unknown"
    stage_reached: str = "mushroomer"
    trust_at_death: float = 0.0
    notable_facts: list[str] = field(default_factory=list)
    personality_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "generation": self.generation,
            "genome_snapshot": dict(self.genome_snapshot),
            "birth_time": self.birth_time,
            "death_time": self.death_time,
            "death_cause": self.death_cause,
            "stage_reached": self.stage_reached,
            "trust_at_death": self.trust_at_death,
            "notable_facts": list(self.notable_facts),
            "personality_summary": self.personality_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LineageEntry:
        """Deserialize from a dict. Unknown keys are silently ignored."""
        return cls(
            generation=int(data.get("generation", 1)),
            genome_snapshot=dict(data.get("genome_snapshot", {})),
            birth_time=str(data.get("birth_time", "")),
            death_time=str(data.get("death_time", "")),
            death_cause=str(data.get("death_cause", "unknown")),
            stage_reached=str(data.get("stage_reached", "mushroomer")),
            trust_at_death=float(data.get("trust_at_death", 0.0)),
            notable_facts=list(data.get("notable_facts", [])),
            personality_summary=str(data.get("personality_summary", "")),
        )

    @classmethod
    def from_legacy(cls, legacy: GeneticLegacy) -> LineageEntry:
        """Create a LineageEntry from a GeneticLegacy artifact.

        Extracts the relevant fields from the legacy and formats the
        personality summary from the personality drift dict.

        Args:
            legacy: A GeneticLegacy from a dead creature.

        Returns:
            A new LineageEntry populated from the legacy.
        """
        # Build personality summary from drift
        drift = legacy.personality_drift
        summary_parts: list[str] = []
        for trait, delta in sorted(drift.items(), key=lambda x: abs(x[1]), reverse=True):
            if abs(delta) < 0.01:
                continue
            direction = "increased" if delta > 0 else "decreased"
            summary_parts.append(f"{trait} {direction} by {abs(delta):.2f}")

        personality_summary = "; ".join(summary_parts) if summary_parts else "no significant drift"

        # Extract birth/death times from behavioral_patterns if available
        patterns = legacy.behavioral_patterns
        birth_time = str(patterns.get("birth_time", ""))
        death_time = str(patterns.get("death_time", ""))

        return cls(
            generation=legacy.generation_number,
            genome_snapshot=dict(legacy.genome),
            birth_time=birth_time,
            death_time=death_time,
            death_cause=legacy.cause_of_death,
            stage_reached=legacy.stage_reached,
            trust_at_death=legacy.trust_at_death,
            notable_facts=list(legacy.distilled_memories),
            personality_summary=personality_summary,
        )


# ---------------------------------------------------------------------------
# LineageTracker
# ---------------------------------------------------------------------------

DEFAULT_FAMILY_TREE_FILE = "family_tree.json"


class LineageTracker:
    """Manages the persistent family tree across creature generations.

    Reads/writes a JSON file containing a list of LineageEntry dicts.
    The file is stored at ``lineage_dir / family_tree.json``.

    Args:
        lineage_dir: Directory for lineage files (e.g. ``data/saves/lineage/``).
    """

    def __init__(self, lineage_dir: str | Path) -> None:
        self._dir = Path(lineage_dir)
        self._file = self._dir / DEFAULT_FAMILY_TREE_FILE
        self._entries: list[LineageEntry] | None = None  # lazy-loaded cache

    # -- persistence helpers ------------------------------------------------

    def _ensure_dir(self) -> None:
        """Create the lineage directory if it doesn't exist."""
        self._dir.mkdir(parents=True, exist_ok=True)

    def _load(self) -> list[LineageEntry]:
        """Load entries from disk, returning empty list if file missing."""
        if not self._file.exists():
            return []
        try:
            data = json.loads(self._file.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                logger.warning("family_tree.json is not a list, resetting")
                return []
            return [LineageEntry.from_dict(d) for d in data]
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load family tree: %s", exc)
            return []

    def _save(self, entries: list[LineageEntry]) -> None:
        """Write entries to disk."""
        self._ensure_dir()
        self._file.write_text(
            json.dumps([e.to_dict() for e in entries], indent=2),
            encoding="utf-8",
        )

    def _get_entries(self) -> list[LineageEntry]:
        """Return cached entries, loading from disk on first access."""
        if self._entries is None:
            self._entries = self._load()
        return self._entries

    # -- public API ---------------------------------------------------------

    def add_entry(self, legacy: GeneticLegacy) -> LineageEntry:
        """Append a new entry to the family tree from a GeneticLegacy.

        Args:
            legacy: The dead creature's genetic legacy.

        Returns:
            The newly created LineageEntry.
        """
        entry = LineageEntry.from_legacy(legacy)
        entries = self._get_entries()
        entries.append(entry)
        self._save(entries)
        logger.info(
            "Added lineage entry for generation %d (cause: %s)",
            entry.generation,
            entry.death_cause,
        )
        return entry

    def get_lineage(self) -> list[LineageEntry]:
        """Return the full ancestry as a list of LineageEntry, oldest first.

        Returns:
            List of all LineageEntry objects in chronological order.
        """
        return list(self._get_entries())

    def get_generation_count(self) -> int:
        """Return the current generation number.

        If no entries exist, the current creature is generation 1.
        Otherwise, the next creature is one more than the highest recorded
        generation.

        Returns:
            The generation number for the *next* creature.
        """
        entries = self._get_entries()
        if not entries:
            return 1
        return max(e.generation for e in entries) + 1

    def get_lineage_summary(self) -> str:
        """Build a narrative summary for prompt injection.

        Returns a string like:
        "You are the 4th generation. Your predecessor lived 23 days and
        died of starvation. They reached the PODFISH stage and had a trust
        level of 0.72."

        Returns:
            A human-readable lineage summary, or empty string if no lineage.
        """
        entries = self._get_entries()
        if not entries:
            return ""

        gen_count = self.get_generation_count()
        parts: list[str] = [
            f"You are the {_ordinal(gen_count)} of your line.",
        ]

        # Describe the most recent ancestor (last entry)
        prev = entries[-1]
        lifespan = _compute_lifespan_str(prev.birth_time, prev.death_time)
        parts.append(
            f"Your predecessor (generation {prev.generation}) "
            f"{lifespan}"
            f"died of {prev.death_cause}."
        )
        parts.append(
            f"They reached the {prev.stage_reached.upper()} stage "
            f"with a trust level of {prev.trust_at_death:.2f}."
        )

        # Brief mention of older ancestors if any
        if len(entries) > 1:
            parts.append(
                f"Before them, {len(entries) - 1} other "
                f"{'generation has' if len(entries) - 1 == 1 else 'generations have'} "
                f"lived and died."
            )

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ordinal(n: int) -> str:
    """Convert an integer to its ordinal string (1st, 2nd, 3rd, etc.)."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _compute_lifespan_str(birth_iso: str, death_iso: str) -> str:
    """Compute a human-readable lifespan string from ISO timestamps.

    Returns something like "lived 23 days and " or "lived less than a day and "
    or empty string if timestamps are unavailable.
    """
    if not birth_iso or not death_iso:
        return ""
    try:
        birth = datetime.fromisoformat(birth_iso)
        death = datetime.fromisoformat(death_iso)
        delta = death - birth
        days = delta.total_seconds() / 86400.0
        if days < 1:
            return "lived less than a day and "
        return f"lived {int(days)} {'day' if int(days) == 1 else 'days'} and "
    except (ValueError, TypeError):
        return ""
