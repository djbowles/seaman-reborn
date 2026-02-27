"""Tests for CreatureState dataclass."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from seaman_brain.creature.state import CreatureState, _clamp
from seaman_brain.types import CreatureStage

# --- Helper: _clamp ---

class TestClamp:
    def test_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_below_min(self) -> None:
        assert _clamp(-0.5) == 0.0

    def test_above_max(self) -> None:
        assert _clamp(1.5) == 1.0

    def test_at_boundaries(self) -> None:
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0

    def test_custom_range(self) -> None:
        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(-1.0, 0.0, 10.0) == 0.0
        assert _clamp(15.0, 0.0, 10.0) == 10.0


# --- Happy path ---

class TestCreatureStateCreation:
    def test_default_state(self) -> None:
        state = CreatureState()
        assert state.stage == CreatureStage.MUSHROOMER
        assert state.age == 0.0
        assert state.interaction_count == 0
        assert state.mood == "neutral"
        assert state.trust_level == 0.0
        assert state.hunger == 0.0
        assert state.health == 1.0
        assert state.comfort == 1.0

    def test_custom_state(self) -> None:
        now = datetime.now(UTC)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            age=3600.0,
            interaction_count=25,
            mood="content",
            trust_level=0.5,
            hunger=0.3,
            health=0.8,
            comfort=0.7,
            last_fed=now,
            last_interaction=now,
            birth_time=now - timedelta(hours=1),
        )
        assert state.stage == CreatureStage.GILLMAN
        assert state.age == 3600.0
        assert state.interaction_count == 25
        assert state.mood == "content"
        assert state.trust_level == 0.5
        assert state.hunger == 0.3

    def test_timestamps_are_utc(self) -> None:
        state = CreatureState()
        assert state.birth_time.tzinfo is not None
        assert state.last_fed.tzinfo is not None
        assert state.last_interaction.tzinfo is not None


# --- Clamping ---

class TestCreatureStateClamping:
    def test_trust_clamped_above(self) -> None:
        state = CreatureState(trust_level=1.5)
        assert state.trust_level == 1.0

    def test_trust_clamped_below(self) -> None:
        state = CreatureState(trust_level=-0.5)
        assert state.trust_level == 0.0

    def test_hunger_clamped(self) -> None:
        state = CreatureState(hunger=2.0)
        assert state.hunger == 1.0

    def test_health_clamped(self) -> None:
        state = CreatureState(health=-0.1)
        assert state.health == 0.0

    def test_comfort_clamped(self) -> None:
        state = CreatureState(comfort=1.5)
        assert state.comfort == 1.0

    def test_negative_age_clamped(self) -> None:
        state = CreatureState(age=-100.0)
        assert state.age == 0.0

    def test_negative_interaction_count_clamped(self) -> None:
        state = CreatureState(interaction_count=-5)
        assert state.interaction_count == 0


# --- Serialization: to_dict ---

class TestToDict:
    def test_roundtrip_fields(self) -> None:
        now = datetime.now(UTC)
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            age=1000.0,
            interaction_count=50,
            mood="curious",
            trust_level=0.6,
            hunger=0.4,
            health=0.9,
            comfort=0.8,
            last_fed=now,
            last_interaction=now,
            birth_time=now,
        )
        d = state.to_dict()
        assert d["stage"] == "podfish"
        assert d["age"] == 1000.0
        assert d["interaction_count"] == 50
        assert d["mood"] == "curious"
        assert d["trust_level"] == 0.6
        assert d["hunger"] == 0.4
        assert d["health"] == 0.9
        assert d["comfort"] == 0.8
        assert d["last_fed"] == now.isoformat()
        assert d["birth_time"] == now.isoformat()

    def test_all_keys_present(self) -> None:
        d = CreatureState().to_dict()
        expected_keys = {
            "stage", "age", "interaction_count", "mood", "trust_level",
            "hunger", "health", "comfort", "last_fed", "last_interaction",
            "birth_time",
        }
        assert set(d.keys()) == expected_keys

    def test_stage_serialized_as_value(self) -> None:
        for stage in CreatureStage:
            state = CreatureState(stage=stage)
            assert state.to_dict()["stage"] == stage.value


# --- Deserialization: from_dict ---

class TestFromDict:
    def test_roundtrip(self) -> None:
        now = datetime.now(UTC)
        original = CreatureState(
            stage=CreatureStage.TADMAN,
            age=5000.0,
            interaction_count=120,
            mood="irritated",
            trust_level=0.7,
            hunger=0.6,
            health=0.5,
            comfort=0.3,
            last_fed=now,
            last_interaction=now,
            birth_time=now,
        )
        restored = CreatureState.from_dict(original.to_dict())
        assert restored.stage == original.stage
        assert restored.age == original.age
        assert restored.interaction_count == original.interaction_count
        assert restored.mood == original.mood
        assert restored.trust_level == original.trust_level
        assert restored.hunger == original.hunger
        assert restored.health == original.health
        assert restored.comfort == original.comfort
        assert restored.birth_time == original.birth_time

    def test_empty_dict_returns_defaults(self) -> None:
        state = CreatureState.from_dict({})
        assert state.stage == CreatureStage.MUSHROOMER
        assert state.mood == "neutral"

    def test_unknown_keys_ignored(self) -> None:
        state = CreatureState.from_dict({"unknown_field": 42, "mood": "happy"})
        assert state.mood == "happy"

    def test_partial_dict(self) -> None:
        state = CreatureState.from_dict({"stage": "frogman", "trust_level": 0.9})
        assert state.stage == CreatureStage.FROGMAN
        assert state.trust_level == 0.9
        assert state.mood == "neutral"  # default

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValueError):
            CreatureState.from_dict({"stage": "invalid_stage"})

    def test_from_dict_clamps_values(self) -> None:
        state = CreatureState.from_dict({"hunger": 5.0, "health": -1.0})
        assert state.hunger == 1.0
        assert state.health == 0.0
