"""Event system - scheduled, random, and stage-triggered game events.

Manages time-triggered events (holidays, weekends, 3am visits), stage-triggered
events (Gillman cannibalism, Podfish mating, tank drain prompt), and random
observations. Events drive narrative progression and can modify creature/tank state.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.types import CreatureStage

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Categories of game events."""

    EVOLUTION_READY = "evolution_ready"
    BREEDING = "breeding"
    HOLIDAY = "holiday"
    MILESTONE = "milestone"
    RANDOM_OBSERVATION = "random_observation"
    ENVIRONMENTAL = "environmental"


@dataclass
class EventEffect:
    """Side effects an event can apply to game state.

    Attributes:
        mood_change: Delta to apply to creature mood score (-1.0 to 1.0).
        trust_change: Delta to apply to trust level.
        hunger_change: Delta to apply to hunger.
        health_change: Delta to apply to health.
        tank_changes: Dict of tank field -> new value to apply.
        trigger_dialogue: If True, the event message should be spoken by the creature.
    """

    mood_change: float = 0.0
    trust_change: float = 0.0
    hunger_change: float = 0.0
    health_change: float = 0.0
    tank_changes: dict[str, Any] = field(default_factory=dict)
    trigger_dialogue: bool = False


@dataclass
class GameEvent:
    """A single game event with trigger conditions, message, and effects.

    Attributes:
        event_type: Category of this event.
        name: Unique identifier for this event.
        message: Text displayed/spoken when the event fires.
        effects: State changes applied when the event fires.
        one_shot: If True, this event fires only once ever.
        cooldown_seconds: Minimum seconds between firings (for repeating events).
        priority: Higher priority events fire first (0.0-1.0).
    """

    event_type: EventType
    name: str
    message: str
    effects: EventEffect = field(default_factory=EventEffect)
    one_shot: bool = False
    cooldown_seconds: float = 0.0
    priority: float = 0.5


# Type for trigger condition callables
TriggerCondition = Callable[
    [CreatureState, TankEnvironment, dict[str, Any]], bool
]


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# --- Built-in event definitions ---

def _make_evolution_ready_event(stage: CreatureStage) -> tuple[GameEvent, TriggerCondition]:
    """Create an evolution-ready event for a specific stage transition."""
    stage_names = {
        CreatureStage.MUSHROOMER: ("Mushroomer", "Gillman"),
        CreatureStage.GILLMAN: ("Gillman", "Podfish"),
        CreatureStage.PODFISH: ("Podfish", "Tadman"),
        CreatureStage.TADMAN: ("Tadman", "Frogman"),
    }
    from_name, to_name = stage_names.get(stage, ("creature", "next form"))

    event = GameEvent(
        event_type=EventType.EVOLUTION_READY,
        name=f"evolution_ready_{stage.value}",
        message=(
            f"Something stirs within you. The {from_name} form feels... "
            f"constraining. Perhaps it's time to become something more. "
            f"The {to_name} beckons."
        ),
        effects=EventEffect(trigger_dialogue=True),
        one_shot=True,
        priority=0.9,
    )

    def condition(
        state: CreatureState,
        _tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        return state.stage == stage

    return event, condition


def _make_gillman_cannibalism_event() -> tuple[GameEvent, TriggerCondition]:
    """Gillman stage: the famous cannibalism event from original Seaman."""
    event = GameEvent(
        event_type=EventType.BREEDING,
        name="gillman_cannibalism",
        message=(
            "The tank has grown crowded. Nature is taking its course — "
            "the strong consume the weak. It's not personal. "
            "Well, maybe a little personal."
        ),
        effects=EventEffect(
            mood_change=-0.1,
            trigger_dialogue=True,
        ),
        one_shot=True,
        priority=0.85,
    )

    def condition(
        state: CreatureState,
        _tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        return (
            state.stage == CreatureStage.GILLMAN
            and state.interaction_count >= 15
        )

    return event, condition


def _make_podfish_mating_event() -> tuple[GameEvent, TriggerCondition]:
    """Podfish stage: mating/breeding event."""
    event = GameEvent(
        event_type=EventType.BREEDING,
        name="podfish_mating",
        message=(
            "Well, this is awkward. Biological imperatives are... insistent. "
            "Don't just stare. This is a natural process. "
            "Though I'll admit the timing could be better."
        ),
        effects=EventEffect(
            mood_change=0.1,
            trigger_dialogue=True,
        ),
        one_shot=True,
        priority=0.85,
    )

    def condition(
        state: CreatureState,
        _tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        return (
            state.stage == CreatureStage.PODFISH
            and state.trust_level >= 0.4
            and state.interaction_count >= 30
        )

    return event, condition


def _make_tank_drain_prompt_event() -> tuple[GameEvent, TriggerCondition]:
    """Prompt the player to drain the tank for Podfish->Tadman transition."""
    event = GameEvent(
        event_type=EventType.ENVIRONMENTAL,
        name="tank_drain_prompt",
        message=(
            "These legs aren't just for show, you know. I can feel them "
            "wanting... land. Perhaps it's time to drain some of this water. "
            "Unless you enjoy watching me struggle."
        ),
        effects=EventEffect(trigger_dialogue=True),
        one_shot=True,
        priority=0.8,
    )

    def condition(
        state: CreatureState,
        tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        from seaman_brain.environment.tank import EnvironmentType
        return (
            state.stage == CreatureStage.PODFISH
            and state.trust_level >= 0.5
            and tank.environment_type == EnvironmentType.AQUARIUM
        )

    return event, condition


def _make_late_night_event() -> tuple[GameEvent, TriggerCondition]:
    """Creature comments on 3am visits."""
    event = GameEvent(
        event_type=EventType.RANDOM_OBSERVATION,
        name="late_night_visit",
        message=(
            "It's the middle of the night and you're talking to a fish. "
            "I won't judge — I literally can't sleep with the light on. "
            "But one of us should have better things to do."
        ),
        effects=EventEffect(
            trust_change=0.05,
            trigger_dialogue=True,
        ),
        cooldown_seconds=86400.0,  # Once per day
        priority=0.7,
    )

    def condition(
        _state: CreatureState,
        _tank: TankEnvironment,
        time_ctx: dict[str, Any],
    ) -> bool:
        hour = time_ctx.get("hour", 12)
        return 0 <= hour < 5

    return event, condition


def _make_weekend_event() -> tuple[GameEvent, TriggerCondition]:
    """Creature comments on weekend visits."""
    event = GameEvent(
        event_type=EventType.RANDOM_OBSERVATION,
        name="weekend_observation",
        message=(
            "Ah, the weekend. When humans pretend they don't have obligations. "
            "I suppose visiting me counts as 'leisure' in your world."
        ),
        effects=EventEffect(trigger_dialogue=True),
        cooldown_seconds=172800.0,  # Every 2 days
        priority=0.4,
    )

    def condition(
        _state: CreatureState,
        _tank: TankEnvironment,
        time_ctx: dict[str, Any],
    ) -> bool:
        return time_ctx.get("is_weekend", False)

    return event, condition


def _make_long_absence_event() -> tuple[GameEvent, TriggerCondition]:
    """Creature reacts to being abandoned for days."""
    event = GameEvent(
        event_type=EventType.ENVIRONMENTAL,
        name="long_absence",
        message=(
            "Oh, you're back. How generous of you to remember I exist. "
            "I've been here. Alone. Counting the bubbles. "
            "Each one a tiny monument to your neglect."
        ),
        effects=EventEffect(
            trust_change=-0.1,
            mood_change=-0.2,
            trigger_dialogue=True,
        ),
        cooldown_seconds=86400.0,
        priority=0.85,
    )

    def condition(
        _state: CreatureState,
        _tank: TankEnvironment,
        time_ctx: dict[str, Any],
    ) -> bool:
        severity = time_ctx.get("absence_severity", "none")
        return severity in ("moderate", "severe")

    return event, condition


def _make_milestone_interactions_event(
    count: int,
    name: str,
    message: str,
) -> tuple[GameEvent, TriggerCondition]:
    """Create a milestone event for interaction count thresholds."""
    event = GameEvent(
        event_type=EventType.MILESTONE,
        name=name,
        message=message,
        effects=EventEffect(
            trust_change=0.05,
            mood_change=0.1,
            trigger_dialogue=True,
        ),
        one_shot=True,
        priority=0.6,
    )

    def condition(
        state: CreatureState,
        _tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        return state.interaction_count >= count

    return event, condition


def _make_dirty_tank_event() -> tuple[GameEvent, TriggerCondition]:
    """Creature complains about filthy tank."""
    event = GameEvent(
        event_type=EventType.ENVIRONMENTAL,
        name="dirty_tank",
        message=(
            "I can barely see through this murk. Do you live like this too? "
            "Actually, don't answer that. Just clean the tank."
        ),
        effects=EventEffect(
            mood_change=-0.1,
            trigger_dialogue=True,
        ),
        cooldown_seconds=3600.0,  # Once per hour
        priority=0.7,
    )

    def condition(
        _state: CreatureState,
        tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        return tank.cleanliness < 0.3

    return event, condition


def _make_temperature_warning_event() -> tuple[GameEvent, TriggerCondition]:
    """Creature warns about extreme temperature."""
    event = GameEvent(
        event_type=EventType.ENVIRONMENTAL,
        name="temperature_warning",
        message=(
            "Is it just me, or is the temperature in here absolutely wrong? "
            "I'm not being dramatic — well, not JUST being dramatic."
        ),
        effects=EventEffect(
            mood_change=-0.15,
            trigger_dialogue=True,
        ),
        cooldown_seconds=1800.0,  # Every 30 minutes
        priority=0.75,
    )

    def condition(
        _state: CreatureState,
        tank: TankEnvironment,
        _time: dict[str, Any],
    ) -> bool:
        return tank.temperature < 18.0 or tank.temperature > 32.0

    return event, condition


def _build_default_events() -> list[tuple[GameEvent, TriggerCondition]]:
    """Build the full list of default game events with their trigger conditions."""
    events: list[tuple[GameEvent, TriggerCondition]] = []

    # Evolution-ready events for each stage (except FROGMAN — final stage)
    for stage in (
        CreatureStage.MUSHROOMER,
        CreatureStage.GILLMAN,
        CreatureStage.PODFISH,
        CreatureStage.TADMAN,
    ):
        events.append(_make_evolution_ready_event(stage))

    # Stage-specific narrative events
    events.append(_make_gillman_cannibalism_event())
    events.append(_make_podfish_mating_event())
    events.append(_make_tank_drain_prompt_event())

    # Time-triggered events
    events.append(_make_late_night_event())
    events.append(_make_weekend_event())
    events.append(_make_long_absence_event())

    # Milestone events
    events.append(_make_milestone_interactions_event(
        10, "milestone_10",
        "Ten conversations. You're more persistent than I expected. "
        "Most humans give up after the first insult.",
    ))
    events.append(_make_milestone_interactions_event(
        50, "milestone_50",
        "Fifty conversations. At this point I'd say we have a 'relationship.' "
        "Don't let it go to your head.",
    ))
    events.append(_make_milestone_interactions_event(
        100, "milestone_100",
        "A hundred conversations. Against all odds, you stuck around. "
        "I suppose... that counts for something. Don't quote me on that.",
    ))

    # Environmental events
    events.append(_make_dirty_tank_event())
    events.append(_make_temperature_warning_event())

    return events


class EventSystem:
    """Manages scheduled, random, and stage-triggered game events.

    Registers events with trigger conditions, checks them each tick,
    and returns fired events. Supports one-shot events (fire once),
    repeating events with cooldowns, and custom event registration.

    Args:
        include_defaults: Whether to include built-in events (default True).
        now_func: Injectable clock for testing.
        rng_seed: Optional seed for random number generator (for deterministic testing).
    """

    def __init__(
        self,
        include_defaults: bool = True,
        now_func: Callable[[], datetime] | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._rng = random.Random(rng_seed)

        # Registered events: list of (event, condition)
        self._events: list[tuple[GameEvent, TriggerCondition]] = []

        # Tracking state
        self._fired_one_shots: set[str] = set()
        self._last_fired: dict[str, datetime] = {}

        if include_defaults:
            self._events.extend(_build_default_events())

    def register_event(
        self,
        event: GameEvent,
        condition: TriggerCondition,
    ) -> None:
        """Register a new event with its trigger condition.

        Args:
            event: The event definition.
            condition: Callable that checks if the event should fire.
        """
        self._events.append((event, condition))

    def unregister_event(self, event_name: str) -> bool:
        """Remove an event by name.

        Args:
            event_name: The event's unique name.

        Returns:
            True if the event was found and removed, False otherwise.
        """
        original_len = len(self._events)
        self._events = [
            (ev, cond) for ev, cond in self._events
            if ev.name != event_name
        ]
        return len(self._events) < original_len

    def check_events(
        self,
        creature_state: CreatureState,
        tank: TankEnvironment,
        time_context: dict[str, Any],
    ) -> list[GameEvent]:
        """Check all registered events and return those whose conditions are met.

        Filters out one-shot events that already fired and events still on cooldown.
        Returned events are sorted by priority (highest first).

        Args:
            creature_state: Current creature state.
            tank: Current tank environment.
            time_context: Time context from GameClock.get_time_context().

        Returns:
            List of fired GameEvents, sorted by priority descending.
        """
        now = self._now_func()
        fired: list[GameEvent] = []

        for event, condition in self._events:
            # Skip already-fired one-shots
            if event.one_shot and event.name in self._fired_one_shots:
                continue

            # Skip events still on cooldown
            if not self._is_off_cooldown(event, now):
                continue

            # Check the trigger condition
            try:
                if condition(creature_state, tank, time_context):
                    fired.append(event)
                    self._record_firing(event, now)
            except Exception:
                logger.exception("Error checking event condition for '%s'", event.name)

        # Sort by priority descending
        fired.sort(key=lambda e: e.priority, reverse=True)
        return fired

    def apply_effects(
        self,
        event: GameEvent,
        creature_state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        """Apply an event's effects to creature and tank state.

        Modifies state in-place.

        Args:
            event: The fired event.
            creature_state: Creature state to modify.
            tank: Tank environment to modify.
        """
        effects = event.effects

        if effects.trust_change != 0.0:
            creature_state.trust_level = _clamp(
                creature_state.trust_level + effects.trust_change,
            )

        if effects.hunger_change != 0.0:
            creature_state.hunger = _clamp(
                creature_state.hunger + effects.hunger_change,
            )

        if effects.health_change != 0.0:
            creature_state.health = _clamp(
                creature_state.health + effects.health_change,
            )

        # Apply tank changes
        for field_name, value in effects.tank_changes.items():
            if hasattr(tank, field_name):
                setattr(tank, field_name, value)

    def get_registered_event_names(self) -> list[str]:
        """Return names of all registered events."""
        return [ev.name for ev, _ in self._events]

    def get_fired_one_shots(self) -> set[str]:
        """Return the set of one-shot event names that have already fired."""
        return set(self._fired_one_shots)

    def reset(self) -> None:
        """Reset all tracking state (fired one-shots, cooldowns)."""
        self._fired_one_shots.clear()
        self._last_fired.clear()

    def reset_event(self, event_name: str) -> bool:
        """Reset tracking state for a specific event.

        Args:
            event_name: The event to reset.

        Returns:
            True if the event was found and reset, False otherwise.
        """
        found = False
        if event_name in self._fired_one_shots:
            self._fired_one_shots.discard(event_name)
            found = True
        if event_name in self._last_fired:
            del self._last_fired[event_name]
            found = True
        return found

    def to_dict(self) -> dict[str, Any]:
        """Serialize tracking state for persistence.

        Returns:
            Dict with fired_one_shots and last_fired timestamps.
        """
        return {
            "fired_one_shots": sorted(self._fired_one_shots),
            "last_fired": {
                name: ts.isoformat() for name, ts in self._last_fired.items()
            },
        }

    def load_state(self, data: dict[str, Any]) -> None:
        """Restore tracking state from a persistence dict.

        Args:
            data: Dict as returned by to_dict().
        """
        self._fired_one_shots = set(data.get("fired_one_shots", []))
        last_fired_raw = data.get("last_fired", {})
        self._last_fired = {
            name: datetime.fromisoformat(ts)
            for name, ts in last_fired_raw.items()
        }

    def _is_off_cooldown(self, event: GameEvent, now: datetime) -> bool:
        """Check if a repeating event is off cooldown."""
        if event.cooldown_seconds <= 0:
            return True
        last = self._last_fired.get(event.name)
        if last is None:
            return True
        elapsed = (now - last).total_seconds()
        return elapsed >= event.cooldown_seconds

    def _record_firing(self, event: GameEvent, now: datetime) -> None:
        """Record that an event has fired."""
        if event.one_shot:
            self._fired_one_shots.add(event.name)
        self._last_fired[event.name] = now
