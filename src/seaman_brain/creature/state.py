"""Creature state dataclass with biological needs, mood, and evolution tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from seaman_brain.types import CreatureStage


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


@dataclass
class CreatureState:
    """Full mutable state of the Seaman creature.

    Fields:
        stage: Current evolutionary stage.
        age: Total seconds the creature has existed.
        interaction_count: Total user interactions since birth.
        mood: Named mood string (e.g. "neutral", "irritated", "content").
        trust_level: How much the creature trusts the user (0.0-1.0).
        hunger: Hunger level (0.0=full, 1.0=starving).
        health: Health level (0.0=dead, 1.0=healthy).
        comfort: Comfort level (0.0=miserable, 1.0=happy).
        last_fed: Timestamp of last feeding.
        last_interaction: Timestamp of last user interaction.
        birth_time: Timestamp when the creature was born.
    """

    stage: CreatureStage = CreatureStage.MUSHROOMER
    age: float = 0.0
    interaction_count: int = 0
    mood: str = "neutral"
    trust_level: float = 0.0
    hunger: float = 0.0
    health: float = 1.0
    comfort: float = 1.0
    last_fed: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_interaction: datetime = field(default_factory=lambda: datetime.now(UTC))
    birth_time: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Clamp float fields to valid ranges."""
        self.trust_level = _clamp(self.trust_level)
        self.hunger = _clamp(self.hunger)
        self.health = _clamp(self.health)
        self.comfort = _clamp(self.comfort)
        self.age = max(0.0, self.age)
        self.interaction_count = max(0, self.interaction_count)

    def to_dict(self) -> dict[str, Any]:
        """Serialize state to a JSON-compatible dict."""
        return {
            "stage": self.stage.value,
            "age": self.age,
            "interaction_count": self.interaction_count,
            "mood": self.mood,
            "trust_level": self.trust_level,
            "hunger": self.hunger,
            "health": self.health,
            "comfort": self.comfort,
            "last_fed": self.last_fed.isoformat(),
            "last_interaction": self.last_interaction.isoformat(),
            "birth_time": self.birth_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreatureState:
        """Deserialize state from a dict (e.g. loaded from JSON).

        Unknown keys are silently ignored.
        Missing keys use defaults.
        """
        kwargs: dict[str, Any] = {}

        if "stage" in data:
            kwargs["stage"] = CreatureStage(data["stage"])

        for float_key in ("age", "trust_level", "hunger", "health", "comfort"):
            if float_key in data:
                kwargs[float_key] = float(data[float_key])

        if "interaction_count" in data:
            kwargs["interaction_count"] = int(data["interaction_count"])

        if "mood" in data:
            kwargs["mood"] = str(data["mood"])

        for dt_key in ("last_fed", "last_interaction", "birth_time"):
            if dt_key in data:
                kwargs[dt_key] = datetime.fromisoformat(data[dt_key])

        return cls(**kwargs)
