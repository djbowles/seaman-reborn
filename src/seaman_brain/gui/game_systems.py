"""Extracted game business logic from GameEngine.

Contains the state machine enum, timing constants, TTS splitting helpers,
interaction situation builders, and the ``GameSystems`` timer-based tick
class that drives creature needs, mood, behavior, events, and evolution.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from seaman_brain.behavior.autonomous import BehaviorEngine, IdleBehavior
from seaman_brain.behavior.events import EventSystem
from seaman_brain.behavior.mood import MoodEngine
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.clock import GameClock
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.death import DeathEngine
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine
from seaman_brain.personality.traits import TraitProfile

logger = logging.getLogger(__name__)

# ── Game state enum ──────────────────────────────────────────────────


class GameState(Enum):
    """Top-level game state for input/update gating."""

    PLAYING = "playing"
    SETTINGS = "settings"
    LINEAGE = "lineage"


# ── Interval constants ───────────────────────────────────────────────

_NEEDS_UPDATE_INTERVAL = 1.0  # seconds between needs ticks
_BEHAVIOR_CHECK_INTERVAL = 15.0  # seconds between behavior checks
_EVENT_CHECK_INTERVAL = 3.0  # seconds between event checks
_VISION_LOOK_TIMEOUT = 30.0  # seconds before Look Now gives up
_STT_DEBOUNCE_SECONDS = 0.2  # wait for speech to settle before submitting
_PENDING_TIMEOUT = 60.0  # seconds before a stuck pending future is force-cancelled
_REACTION_COOLDOWN = 5.0  # seconds between interaction reactions


# ── Tick result ─────────────────────────────────────────────────────


@dataclass
class TickResult:
    """Results from a single game systems tick.

    GameEngine reads these fields to trigger UI responses (chat messages,
    animations, overlays, sound effects) without knowing the tick internals.
    """

    mood_value: str = "neutral"
    behavior: IdleBehavior | None = None
    fired_events: list[Any] = field(default_factory=list)
    new_stage: Any | None = None
    death_cause: Any | None = None


# ── TTS splitting ────────────────────────────────────────────────────

# Sentence boundary for incremental TTS: .!? or newline (always triggers TTS)
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s|$)|\n")

# Clause boundary for early TTS: , ; : (followed by whitespace/end) or em-dash (no space needed)
_CLAUSE_BOUNDARY = re.compile(r"[,;:](?:\s|$)|\u2014")

# Minimum accumulated characters before a clause boundary triggers TTS.
# Short clauses like "Well," shouldn't be spoken on their own.
_MIN_CLAUSE_LENGTH = 40


def find_tts_split(buffer: str) -> int | None:
    """Find the position to split the TTS buffer for incremental speech.

    Sentence boundaries (``.!?\\n``) always trigger a split. Clause boundaries
    (``,;:—``) trigger a split only when the accumulated text before the
    boundary is at least ``_MIN_CLAUSE_LENGTH`` characters long, so that tiny
    fragments like ``"Well,"`` are not spoken on their own.

    Args:
        buffer: The accumulated token buffer.

    Returns:
        The split position (exclusive) if a boundary was found, or ``None``.
    """
    # Sentence boundaries always win — check them first
    sentence_match = _SENTENCE_BOUNDARY.search(buffer)
    clause_match = _CLAUSE_BOUNDARY.search(buffer)

    best: int | None = None

    if sentence_match is not None:
        best = sentence_match.start() + 1  # include the punctuation

    if clause_match is not None:
        clause_pos = clause_match.start() + 1  # include the punctuation
        # Only use clause boundary if buffer up to it is long enough
        if clause_pos >= _MIN_CLAUSE_LENGTH:
            if best is None or clause_pos < best:
                best = clause_pos

    return best


# ── Interaction situations ───────────────────────────────────────────

# Situation prompts for interaction reactions via LLM
_INTERACTION_SITUATIONS: dict[str, str] = {
    "feed": "Your owner just fed you. React to receiving food.",
    "tap_glass": (
        "Your owner just tapped the glass of your tank. React to the disturbance."
    ),
    "clean": "Your owner just cleaned your tank. React to the improved cleanliness.",
    "aerate": "Your owner just aerated your tank water. React to the fresh bubbles.",
    "temp_up": (
        "Your owner just raised your tank temperature. React to the warmth change."
    ),
    "temp_down": (
        "Your owner just lowered your tank temperature. React to the temperature drop."
    ),
    "drain": (
        "Your owner just changed the water level in your tank. React to the change."
    ),
}

# Canned fallback emotes when LLM is busy
_INTERACTION_FALLBACKS: dict[str, str] = {
    "feed": "*munches*",
    "tap_glass": "*startles*",
    "clean": "*looks around at the clean tank*",
    "aerate": "*watches the bubbles*",
    "temp_up": "*stretches in the warmth*",
    "temp_down": "*shivers*",
    "drain": "*blinks at the water level*",
}


def _build_interaction_situation(
    action_key: str,
    creature_state: CreatureState,
    tank: TankEnvironment,
    needs: CreatureNeeds,
) -> str | None:
    """Build a context-enriched interaction situation prompt.

    Takes the base template from ``_INTERACTION_SITUATIONS`` and appends
    dynamic state context so the LLM generates varied, state-aware responses.

    Returns ``None`` if *action_key* is not a known interaction.
    """
    base = _INTERACTION_SITUATIONS.get(action_key)
    if base is None:
        return None

    parts = [base]

    # Action-specific context
    if action_key == "feed":
        hunger_pct = round(needs.hunger * 100)
        parts.append(f"Your hunger is at {hunger_pct}%.")
    elif action_key == "clean":
        clean_pct = round(tank.cleanliness * 100)
        parts.append(f"Tank cleanliness is at {clean_pct}%.")
    elif action_key == "aerate":
        o2_pct = round(tank.oxygen_level * 100)
        parts.append(f"Oxygen level is at {o2_pct}%.")
    elif action_key in ("temp_up", "temp_down"):
        parts.append(f"Current water temperature is {tank.temperature:.1f}\u00b0C.")
    elif action_key == "drain":
        water_pct = round(tank.water_level * 100)
        parts.append(f"Water level is at {water_pct}%.")
    elif action_key == "tap_glass":
        parts.append(f"You're feeling {creature_state.mood}.")

    # Always append current mood
    parts.append(f"Your current mood: {creature_state.mood}.")

    return " ".join(parts)


# ── GameSystems class ────────────────────────────────────────────────


class GameSystems:
    """Timer-based tick logic for creature subsystems.

    Extracted from ``GameEngine._update()`` — drives needs, mood,
    behavior, events, and evolution checks at their respective intervals.

    Args:
        needs_engine: Engine for updating creature needs.
        mood_engine: Engine for calculating creature mood.
        behavior_engine: Engine for checking idle behaviors.
        event_system: System for checking/applying game events.
        evolution_engine: Engine for checking evolution readiness.
        death_engine: Engine for checking death conditions.
        creature_state: Current creature state (mutable reference).
        clock: Game clock for time context.
        tank: Tank environment (mutable reference).
    """

    def __init__(
        self,
        *,
        needs_engine: NeedsEngine,
        mood_engine: MoodEngine,
        behavior_engine: BehaviorEngine,
        event_system: EventSystem,
        evolution_engine: EvolutionEngine,
        death_engine: DeathEngine,
        creature_state: CreatureState,
        clock: GameClock,
        tank: TankEnvironment,
    ) -> None:
        self._needs_engine = needs_engine
        self._mood_engine = mood_engine
        self._behavior_engine = behavior_engine
        self._event_system = event_system
        self._evolution_engine = evolution_engine
        self._death_engine = death_engine
        self._creature_state = creature_state
        self._clock = clock
        self._tank = tank

        # Internal timers
        self._needs_timer = 0.0
        self._behavior_timer = 0.0
        self._event_timer = 0.0

        # Needs update state
        self._needs = CreatureNeeds()
        self._interaction_count_delta = 0

        # Trait accessor (can be overridden)
        self._traits_fn: Any = None

    @property
    def needs(self) -> CreatureNeeds:
        """Current creature needs."""
        return self._needs

    @needs.setter
    def needs(self, value: CreatureNeeds) -> None:
        self._needs = value

    def tick(self, dt: float) -> TickResult | None:
        """Advance all timer-based subsystems by *dt* seconds.

        Returns ``None`` when the creature is dead (no updates run).
        Otherwise returns a :class:`TickResult` with any behaviors,
        events, evolution, or death that occurred this tick.

        Args:
            dt: Delta time in seconds since last tick.
        """
        if not self._creature_state.is_alive:
            return None

        result = TickResult()

        # Accumulate timers
        self._needs_timer += dt
        self._behavior_timer += dt
        self._event_timer += dt

        # ── Needs ─────────────────────────────────────────────────
        if self._needs_timer >= _NEEDS_UPDATE_INTERVAL:
            elapsed = self._needs_timer
            self._needs_timer = 0.0
            self._creature_state.age += elapsed
            try:
                self._update_needs(elapsed)
            except Exception as exc:
                logger.error(
                    "Needs update error (continuing with stale state): %s", exc
                )

        # ── Death ─────────────────────────────────────────────────
        try:
            cause = self._death_engine.check_death(
                self._creature_state, self._needs, self._tank,
            )
            if cause is not None:
                result.death_cause = cause
                return result
        except Exception as exc:
            logger.error("Death check error: %s", exc, exc_info=True)

        # ── Mood ──────────────────────────────────────────────────
        time_context: dict = {}
        try:
            time_context = self._clock.get_time_context()
            traits = self.get_traits()
            mood = self._mood_engine.calculate_mood(
                needs=self._needs,
                trust=self._creature_state.trust_level,
                time_context=time_context,
                recent_interactions=self._creature_state.interaction_count,
                traits=traits,
            )
            self._creature_state.mood = mood.value
            result.mood_value = mood.value
        except Exception as exc:
            logger.error("Mood update error: %s", exc, exc_info=True)

        # ── Behavior ──────────────────────────────────────────────
        try:
            if self._behavior_timer >= _BEHAVIOR_CHECK_INTERVAL:
                self._behavior_timer = 0.0
                traits = self.get_traits()
                creature_dict = {
                    "stage": self._creature_state.stage.value,
                    "mood": self._creature_state.mood,
                    "trust": self._creature_state.trust_level,
                    "hunger": self._creature_state.hunger,
                }
                behavior = self._behavior_engine.get_idle_behavior(
                    creature_state=creature_dict,
                    needs=self._needs,
                    mood=self._mood_engine.current_mood,
                    time_context=time_context,
                    traits=traits,
                )
                result.behavior = behavior
        except Exception as exc:
            logger.error("Behavior check error: %s", exc, exc_info=True)

        # ── Events ────────────────────────────────────────────────
        try:
            if self._event_timer >= _EVENT_CHECK_INTERVAL:
                self._event_timer = 0.0
                fired = self._event_system.check_events(
                    creature_state=self._creature_state,
                    tank=self._tank,
                    time_context=time_context,
                )
                for event in fired:
                    self._event_system.apply_effects(
                        event, self._creature_state, self._tank,
                    )
                result.fired_events = fired
        except Exception as exc:
            logger.error("Event check error: %s", exc, exc_info=True)

        # ── Evolution ─────────────────────────────────────────────
        try:
            new_stage = self._evolution_engine.check_evolution(
                self._creature_state,
            )
            result.new_stage = new_stage
        except Exception as exc:
            logger.error("Evolution check error: %s", exc, exc_info=True)

        return result

    def _update_needs(self, elapsed: float) -> None:
        """Update creature needs based on elapsed time."""
        self._needs = self._needs_engine.update(
            elapsed_seconds=elapsed,
            creature_state=self._creature_state,
            tank=self._tank,
            interaction_count_delta=self._interaction_count_delta,
        )
        self._needs_engine.apply_to_state(self._creature_state, self._needs)
        self._interaction_count_delta = 0

    def get_traits(self) -> TraitProfile:
        """Get the current TraitProfile, using the accessor if set."""
        if self._traits_fn is not None:
            result = self._traits_fn()
            if result is not None:
                return result
        return TraitProfile()
