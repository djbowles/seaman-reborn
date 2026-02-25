"""Tests for creature lineage tracking and family tree (US-051)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from seaman_brain.creature.genetics import GeneticLegacy
from seaman_brain.creature.lineage import (
    LineageEntry,
    LineageTracker,
    _compute_lifespan_str,
    _ordinal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_legacy(
    generation: int = 1,
    cause: str = "starvation",
    stage: str = "podfish",
    trust: float = 0.65,
    memories: list[str] | None = None,
    drift: dict[str, float] | None = None,
    genome: dict[str, float] | None = None,
) -> GeneticLegacy:
    """Helper to build a GeneticLegacy with sensible defaults."""
    birth = datetime(2026, 1, 1, tzinfo=UTC)
    death = datetime(2026, 1, 24, tzinfo=UTC)
    return GeneticLegacy(
        genome={"body_size": 0.5, "hue": 0.6} if genome is None else genome,
        distilled_memories=(
            ["Human likes jazz", "Human name is Dave"]
            if memories is None else memories
        ),
        personality_drift={"cynicism": 0.05, "warmth": -0.12} if drift is None else drift,
        behavioral_patterns={
            "total_interactions": 42,
            "birth_time": birth.isoformat(),
            "death_time": death.isoformat(),
        },
        cause_of_death=cause,
        generation_number=generation,
        lifespan_days=23.0,
        stage_reached=stage,
        trust_at_death=trust,
    )


@pytest.fixture
def lineage_dir(tmp_path):
    """Provide a temporary lineage directory."""
    d = tmp_path / "lineage"
    d.mkdir()
    return d


@pytest.fixture
def tracker(lineage_dir):
    """Provide a LineageTracker pointing at tmp dir."""
    return LineageTracker(lineage_dir)


# ---------------------------------------------------------------------------
# LineageEntry tests
# ---------------------------------------------------------------------------


class TestLineageEntry:
    """Tests for the LineageEntry dataclass."""

    def test_default_values(self):
        """Entry has sensible defaults."""
        entry = LineageEntry()
        assert entry.generation == 1
        assert entry.death_cause == "unknown"
        assert entry.stage_reached == "mushroomer"
        assert entry.trust_at_death == 0.0
        assert entry.notable_facts == []
        assert entry.personality_summary == ""

    def test_roundtrip_serialization(self):
        """to_dict/from_dict preserves all fields."""
        entry = LineageEntry(
            generation=3,
            genome_snapshot={"body_size": 0.7},
            birth_time="2026-01-01T00:00:00+00:00",
            death_time="2026-01-24T00:00:00+00:00",
            death_cause="neglect",
            stage_reached="gillman",
            trust_at_death=0.45,
            notable_facts=["fact one", "fact two"],
            personality_summary="cynicism increased by 0.05",
        )
        restored = LineageEntry.from_dict(entry.to_dict())
        assert restored.generation == 3
        assert restored.genome_snapshot == {"body_size": 0.7}
        assert restored.death_cause == "neglect"
        assert restored.stage_reached == "gillman"
        assert restored.trust_at_death == 0.45
        assert restored.notable_facts == ["fact one", "fact two"]
        assert restored.personality_summary == "cynicism increased by 0.05"

    def test_from_dict_missing_keys(self):
        """from_dict fills defaults for missing keys."""
        entry = LineageEntry.from_dict({})
        assert entry.generation == 1
        assert entry.death_cause == "unknown"
        assert entry.notable_facts == []

    def test_from_legacy(self):
        """from_legacy extracts correct fields from a GeneticLegacy."""
        legacy = _make_legacy(
            generation=2,
            cause="old_age",
            stage="tadman",
            trust=0.88,
            memories=["Knows about cats"],
            drift={"warmth": 0.15, "aggression": -0.03},
        )
        entry = LineageEntry.from_legacy(legacy)
        assert entry.generation == 2
        assert entry.death_cause == "old_age"
        assert entry.stage_reached == "tadman"
        assert entry.trust_at_death == 0.88
        assert entry.notable_facts == ["Knows about cats"]
        assert "warmth increased by 0.15" in entry.personality_summary
        assert "aggression decreased by 0.03" in entry.personality_summary

    def test_from_legacy_no_drift(self):
        """from_legacy with zero drift gives 'no significant drift' summary."""
        legacy = _make_legacy(drift={"warmth": 0.0, "cynicism": 0.005})
        entry = LineageEntry.from_legacy(legacy)
        assert entry.personality_summary == "no significant drift"

    def test_from_legacy_missing_timestamps(self):
        """from_legacy handles missing birth/death in behavioral_patterns."""
        legacy = _make_legacy()
        legacy.behavioral_patterns = {}
        entry = LineageEntry.from_legacy(legacy)
        assert entry.birth_time == ""
        assert entry.death_time == ""


# ---------------------------------------------------------------------------
# LineageTracker - add_entry / get_lineage
# ---------------------------------------------------------------------------


class TestLineageTrackerAddAndRetrieve:
    """Tests for adding entries and retrieving lineage."""

    def test_add_entry_creates_file(self, tracker, lineage_dir):
        """add_entry creates family_tree.json on first call."""
        legacy = _make_legacy(generation=1)
        entry = tracker.add_entry(legacy)

        assert entry.generation == 1
        tree_file = lineage_dir / "family_tree.json"
        assert tree_file.exists()

        data = json.loads(tree_file.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["generation"] == 1

    def test_add_multiple_entries(self, tracker):
        """Multiple add_entry calls accumulate entries."""
        tracker.add_entry(_make_legacy(generation=1, cause="starvation"))
        tracker.add_entry(_make_legacy(generation=2, cause="neglect"))
        tracker.add_entry(_make_legacy(generation=3, cause="old_age"))

        lineage = tracker.get_lineage()
        assert len(lineage) == 3
        assert [e.generation for e in lineage] == [1, 2, 3]

    def test_get_lineage_empty(self, tracker):
        """get_lineage returns empty list when no entries exist."""
        assert tracker.get_lineage() == []

    def test_get_lineage_returns_copy(self, tracker):
        """get_lineage returns a copy — mutating it doesn't affect tracker."""
        tracker.add_entry(_make_legacy(generation=1))
        lineage = tracker.get_lineage()
        lineage.clear()
        assert len(tracker.get_lineage()) == 1

    def test_persistence_across_instances(self, lineage_dir):
        """A new LineageTracker instance loads previously saved data."""
        t1 = LineageTracker(lineage_dir)
        t1.add_entry(_make_legacy(generation=1))
        t1.add_entry(_make_legacy(generation=2))

        t2 = LineageTracker(lineage_dir)
        lineage = t2.get_lineage()
        assert len(lineage) == 2
        assert lineage[0].generation == 1
        assert lineage[1].generation == 2


# ---------------------------------------------------------------------------
# LineageTracker - get_generation_count
# ---------------------------------------------------------------------------


class TestLineageTrackerGenerationCount:
    """Tests for generation counting."""

    def test_generation_count_empty(self, tracker):
        """Empty tracker returns generation 1."""
        assert tracker.get_generation_count() == 1

    def test_generation_count_after_entries(self, tracker):
        """After adding gen 1 and 2, next gen is 3."""
        tracker.add_entry(_make_legacy(generation=1))
        tracker.add_entry(_make_legacy(generation=2))
        assert tracker.get_generation_count() == 3

    def test_generation_count_uses_max(self, tracker):
        """Generation count uses the max generation, not just len."""
        tracker.add_entry(_make_legacy(generation=5))
        assert tracker.get_generation_count() == 6


# ---------------------------------------------------------------------------
# LineageTracker - get_lineage_summary
# ---------------------------------------------------------------------------


class TestLineageTrackerSummary:
    """Tests for get_lineage_summary()."""

    def test_summary_empty(self, tracker):
        """Empty tracker returns empty string."""
        assert tracker.get_lineage_summary() == ""

    def test_summary_single_generation(self, tracker):
        """Summary for one dead ancestor describes them."""
        tracker.add_entry(_make_legacy(
            generation=1,
            cause="starvation",
            stage="podfish",
            trust=0.65,
        ))
        summary = tracker.get_lineage_summary()
        assert "2nd" in summary  # next gen is 2
        assert "starvation" in summary
        assert "PODFISH" in summary
        assert "0.65" in summary

    def test_summary_multiple_generations(self, tracker):
        """Summary for 3 ancestors mentions all older ancestors."""
        tracker.add_entry(_make_legacy(generation=1, cause="neglect"))
        tracker.add_entry(_make_legacy(generation=2, cause="starvation"))
        tracker.add_entry(_make_legacy(generation=3, cause="old_age"))

        summary = tracker.get_lineage_summary()
        assert "4th" in summary  # next gen is 4
        assert "old_age" in summary  # most recent ancestor
        assert "2 other generations have lived" in summary

    def test_summary_lifespan_included(self, tracker):
        """Summary includes lifespan when timestamps available."""
        tracker.add_entry(_make_legacy(generation=1))
        summary = tracker.get_lineage_summary()
        # Birth: Jan 1, Death: Jan 24 = 23 days
        assert "23 days" in summary


# ---------------------------------------------------------------------------
# Error handling / edge cases
# ---------------------------------------------------------------------------


class TestLineageTrackerEdgeCases:
    """Edge cases and error handling."""

    def test_corrupted_json_file(self, lineage_dir):
        """Tracker handles corrupted JSON gracefully."""
        tree_file = lineage_dir / "family_tree.json"
        tree_file.write_text("NOT VALID JSON {{{{", encoding="utf-8")

        tracker = LineageTracker(lineage_dir)
        assert tracker.get_lineage() == []

    def test_json_not_a_list(self, lineage_dir):
        """Tracker handles non-list JSON gracefully."""
        tree_file = lineage_dir / "family_tree.json"
        tree_file.write_text('{"key": "value"}', encoding="utf-8")

        tracker = LineageTracker(lineage_dir)
        assert tracker.get_lineage() == []

    def test_nonexistent_dir_created_on_add(self, tmp_path):
        """add_entry creates the lineage directory if it doesn't exist."""
        new_dir = tmp_path / "deep" / "nested" / "lineage"
        tracker = LineageTracker(new_dir)
        tracker.add_entry(_make_legacy(generation=1))

        assert new_dir.exists()
        assert (new_dir / "family_tree.json").exists()

    def test_legacy_with_empty_genome(self, tracker):
        """add_entry works with an empty genome dict."""
        legacy = _make_legacy(genome={})
        entry = tracker.add_entry(legacy)
        assert entry.genome_snapshot == {}

    def test_legacy_with_empty_memories(self, tracker):
        """add_entry works with empty memories list."""
        legacy = _make_legacy(memories=[])
        entry = tracker.add_entry(legacy)
        assert entry.notable_facts == []


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for module-level helper functions."""

    @pytest.mark.parametrize("n,expected", [
        (1, "1st"),
        (2, "2nd"),
        (3, "3rd"),
        (4, "4th"),
        (11, "11th"),
        (12, "12th"),
        (13, "13th"),
        (21, "21st"),
        (22, "22nd"),
        (23, "23rd"),
        (100, "100th"),
        (101, "101st"),
        (111, "111th"),
    ])
    def test_ordinal(self, n, expected):
        """_ordinal produces correct English ordinals."""
        assert _ordinal(n) == expected

    def test_compute_lifespan_str_valid(self):
        """_compute_lifespan_str computes days correctly."""
        birth = "2026-01-01T00:00:00+00:00"
        death = "2026-01-24T00:00:00+00:00"
        assert _compute_lifespan_str(birth, death) == "lived 23 days and "

    def test_compute_lifespan_str_one_day(self):
        """Single day lifespan uses singular."""
        birth = "2026-01-01T00:00:00+00:00"
        death = "2026-01-02T00:00:00+00:00"
        assert _compute_lifespan_str(birth, death) == "lived 1 day and "

    def test_compute_lifespan_str_less_than_day(self):
        """Sub-day lifespan says 'less than a day'."""
        birth = "2026-01-01T00:00:00+00:00"
        death = "2026-01-01T12:00:00+00:00"
        assert _compute_lifespan_str(birth, death) == "lived less than a day and "

    def test_compute_lifespan_str_empty(self):
        """Empty timestamps return empty string."""
        assert _compute_lifespan_str("", "") == ""

    def test_compute_lifespan_str_invalid(self):
        """Invalid timestamps return empty string."""
        assert _compute_lifespan_str("not-a-date", "also-not-a-date") == ""
