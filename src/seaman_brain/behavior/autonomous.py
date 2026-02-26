"""Autonomous behavior system - idle actions, unprompted speech, creature initiative.

The creature performs actions unprompted: swimming around, tapping the glass,
making observations about the time, complaining about hunger, reacting to the
player's presence. These idle behaviors make the creature feel alive between
conversations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from seaman_brain.behavior.mood import CreatureMood
from seaman_brain.needs.system import CreatureNeeds
from seaman_brain.personality.traits import TraitProfile


class BehaviorType(Enum):
    """Types of autonomous creature behavior."""

    IDLE_SWIM = "idle_swim"
    TAP_GLASS = "tap_glass"
    COMPLAIN = "complain"
    OBSERVE = "observe"
    SLEEP = "sleep"
    EAT = "eat"


# Verbal behaviors that benefit from LLM-generated speech
VERBAL_BEHAVIORS: frozenset[BehaviorType] = frozenset({
    BehaviorType.COMPLAIN,
    BehaviorType.OBSERVE,
})


@dataclass
class IdleBehavior:
    """An autonomous behavior the creature can perform unprompted.

    Attributes:
        action_type: The kind of behavior.
        message: Text the creature says/thinks during this behavior.
        animation_hint: Suggested animation for the GUI layer.
        priority: Higher priority behaviors are more likely to be selected (0.0-1.0).
        needs_llm: Whether this behavior should be routed through the LLM for speech.
    """

    action_type: BehaviorType
    message: str
    animation_hint: str = ""
    priority: float = 0.5
    needs_llm: bool = False


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


# Default messages per behavior type, keyed by mood category.
# "negative" = HOSTILE/IRRITATED, "neutral" = SARDONIC/NEUTRAL,
# "positive" = CURIOUS/AMUSED/PHILOSOPHICAL/CONTENT
_BEHAVIOR_MESSAGES: dict[BehaviorType, dict[str, list[str]]] = {
    BehaviorType.IDLE_SWIM: {
        "negative": [
            "...",
            "*drifts aimlessly*",
            "*circles the tank with visible agitation*",
        ],
        "neutral": [
            "*swims in a slow circle*",
            "*drifts to the other side of the tank*",
            "*floats, watching nothing in particular*",
        ],
        "positive": [
            "*glides gracefully through the water*",
            "*does a little loop*",
            "*swims up to the glass and peers out*",
        ],
    },
    BehaviorType.TAP_GLASS: {
        "negative": [
            "*slams against the glass*",
            "*headbutts the tank wall*",
        ],
        "neutral": [
            "*taps the glass experimentally*",
            "*presses face against the glass*",
        ],
        "positive": [
            "*taps the glass playfully*",
            "*nudges the glass to get your attention*",
        ],
    },
    BehaviorType.COMPLAIN: {
        "negative": [
            "This is what passes for care? Pathetic.",
            "I'm wasting away in here.",
            "Would it kill you to do something useful?",
        ],
        "neutral": [
            "You know, I have needs.",
            "Just putting it out there... I exist.",
            "Remember me? Your responsibility?",
        ],
        "positive": [
            "Not that I'd ever ask for anything, but...",
            "I suppose I could use some attention.",
        ],
    },
    BehaviorType.OBSERVE: {
        "negative": [
            "Another wasted moment of existence.",
            "The futility of it all is... remarkable.",
        ],
        "neutral": [
            "Interesting... the world continues to turn.",
            "I see you're still here. How persistent.",
            "The passage of time is a curious thing.",
        ],
        "positive": [
            "You know, from a certain angle this tank isn't half bad.",
            "I've been thinking about something...",
            "Funny how the light hits the water at this hour.",
        ],
    },
    BehaviorType.SLEEP: {
        "negative": [
            "*pretends to sleep to avoid you*",
            "Zzz... go away... zzz...",
        ],
        "neutral": [
            "*dozes off briefly*",
            "*closes eyes and rests*",
            "*settles to the bottom and naps*",
        ],
        "positive": [
            "*yawns contentedly*",
            "*drifts into a peaceful nap*",
        ],
    },
    BehaviorType.EAT: {
        "negative": [
            "*chews angrily on a pebble*",
            "*gnaws on nothing out of frustration*",
        ],
        "neutral": [
            "*nibbles at some algae*",
            "*pecks at the gravel*",
        ],
        "positive": [
            "*happily munches on something*",
            "*savors a morsel*",
        ],
    },
}

# Animation hints per behavior type
_ANIMATION_HINTS: dict[BehaviorType, str] = {
    BehaviorType.IDLE_SWIM: "swimming",
    BehaviorType.TAP_GLASS: "tap",
    BehaviorType.COMPLAIN: "talking",
    BehaviorType.OBSERVE: "idle",
    BehaviorType.SLEEP: "sleeping",
    BehaviorType.EAT: "eating",
}

# Negative moods set
_NEGATIVE_MOODS: set[CreatureMood] = {CreatureMood.HOSTILE, CreatureMood.IRRITATED}
_POSITIVE_MOODS: set[CreatureMood] = {
    CreatureMood.CURIOUS,
    CreatureMood.AMUSED,
    CreatureMood.PHILOSOPHICAL,
    CreatureMood.CONTENT,
}


def _mood_category(mood: CreatureMood) -> str:
    """Map a mood to a message category: negative, neutral, or positive."""
    if mood in _NEGATIVE_MOODS:
        return "negative"
    if mood in _POSITIVE_MOODS:
        return "positive"
    return "neutral"


@dataclass
class _CooldownEntry:
    """Tracks when a behavior type was last triggered."""

    behavior: BehaviorType
    last_triggered: datetime
    cooldown_seconds: float


class BehaviorEngine:
    """Manages autonomous creature actions between conversations.

    Selects behaviors based on creature mood, needs, time context, and
    personality traits. Enforces cooldowns to prevent repetitive actions.

    Args:
        base_cooldown: Default cooldown between any behavior (seconds).
        behavior_cooldowns: Per-behavior-type cooldown overrides.
        now_func: Injectable clock for testing.
    """

    def __init__(
        self,
        base_cooldown: float = 30.0,
        behavior_cooldowns: dict[BehaviorType, float] | None = None,
        now_func: Any = None,
    ) -> None:
        self._base_cooldown = max(0.0, base_cooldown)
        self._cooldowns: dict[BehaviorType, float] = {
            BehaviorType.IDLE_SWIM: 15.0,
            BehaviorType.TAP_GLASS: 60.0,
            BehaviorType.COMPLAIN: 45.0,
            BehaviorType.OBSERVE: 40.0,
            BehaviorType.SLEEP: 120.0,
            BehaviorType.EAT: 90.0,
        }
        if behavior_cooldowns:
            self._cooldowns.update(behavior_cooldowns)
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._last_triggered: dict[BehaviorType, datetime] = {}
        self._message_index: dict[BehaviorType, int] = {}

    def get_idle_behavior(
        self,
        creature_state: dict[str, Any],
        needs: CreatureNeeds,
        mood: CreatureMood,
        time_context: dict[str, Any],
        traits: TraitProfile | None = None,
    ) -> IdleBehavior | None:
        """Select an autonomous behavior based on current state.

        Evaluates candidate behaviors, scores them by relevance to current
        needs/mood/time, filters by cooldown, and returns the highest priority
        one — or None if all are on cooldown.

        Args:
            creature_state: Creature state dict (stage, hunger, health, etc.).
            needs: Current creature needs.
            mood: Current creature mood.
            time_context: Time context from GameClock.get_time_context().
            traits: Optional personality traits for behavior weighting.

        Returns:
            The selected IdleBehavior, or None if nothing is available.
        """
        now = self._now_func()
        candidates: list[tuple[float, BehaviorType]] = []

        for btype in BehaviorType:
            if not self._is_off_cooldown(btype, now):
                continue
            priority = self._score_behavior(btype, needs, mood, time_context, traits)
            if priority > 0.0:
                candidates.append((priority, btype))

        if not candidates:
            return None

        # Sort by priority descending, pick the best
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_priority, best_type = candidates[0]

        # Build the behavior
        category = _mood_category(mood)
        message = self._pick_message(best_type, category)
        animation = _ANIMATION_HINTS.get(best_type, "idle")

        # Record trigger time
        self._last_triggered[best_type] = now

        return IdleBehavior(
            action_type=best_type,
            message=message,
            animation_hint=animation,
            priority=_clamp(best_priority),
            needs_llm=best_type in VERBAL_BEHAVIORS,
        )

    async def generate_idle_comment(
        self,
        behavior: IdleBehavior,
        llm_provider: Any,
    ) -> str | None:
        """Optionally generate an LLM-powered idle comment for a behavior.

        Args:
            behavior: The selected idle behavior.
            llm_provider: An LLMProvider instance for generating speech.

        Returns:
            Generated comment string, or None on failure.
        """
        if llm_provider is None:
            return None

        from seaman_brain.types import ChatMessage, MessageRole

        prompt = (
            f"You are Seaman, a sardonic aquatic creature. "
            f"You are currently performing: {behavior.action_type.value}. "
            f"Generate a single brief in-character remark (1 sentence max). "
            f"Be sardonic and darkly witty. No AI assistant language."
        )
        messages = [ChatMessage(role=MessageRole.SYSTEM, content=prompt)]

        try:
            result = await llm_provider.chat(messages)
            return result.strip() if result else None
        except Exception:
            return None

    def reset_cooldowns(self) -> None:
        """Clear all cooldown timers."""
        self._last_triggered.clear()

    def get_cooldown_remaining(self, behavior_type: BehaviorType) -> float:
        """Get remaining cooldown seconds for a behavior type.

        Returns:
            Seconds remaining, or 0.0 if off cooldown.
        """
        last = self._last_triggered.get(behavior_type)
        if last is None:
            return 0.0
        now = self._now_func()
        cd = self._cooldowns.get(behavior_type, self._base_cooldown)
        elapsed = (now - last).total_seconds()
        remaining = cd - elapsed
        return max(0.0, remaining)

    def _is_off_cooldown(self, behavior_type: BehaviorType, now: datetime) -> bool:
        """Check if a behavior type is off cooldown."""
        last = self._last_triggered.get(behavior_type)
        if last is None:
            return True
        cd = self._cooldowns.get(behavior_type, self._base_cooldown)
        return (now - last).total_seconds() >= cd

    def _score_behavior(
        self,
        btype: BehaviorType,
        needs: CreatureNeeds,
        mood: CreatureMood,
        time_context: dict[str, Any],
        traits: TraitProfile | None,
    ) -> float:
        """Score a behavior's priority based on current conditions.

        Returns a 0.0-1.0 score; higher means more appropriate right now.
        """
        score = 0.3  # base score

        if btype == BehaviorType.COMPLAIN:
            score = self._score_complain(needs, mood)
        elif btype == BehaviorType.SLEEP:
            score = self._score_sleep(needs, mood, time_context)
        elif btype == BehaviorType.EAT:
            score = self._score_eat(needs)
        elif btype == BehaviorType.OBSERVE:
            score = self._score_observe(mood, traits)
        elif btype == BehaviorType.TAP_GLASS:
            score = self._score_tap_glass(needs, mood)
        elif btype == BehaviorType.IDLE_SWIM:
            score = self._score_idle_swim(mood)

        return _clamp(score)

    @staticmethod
    def _score_complain(needs: CreatureNeeds, mood: CreatureMood) -> float:
        """Hungry or uncomfortable creatures complain more."""
        score = 0.2
        # Hunger drives complaints strongly
        score += needs.hunger * 0.5
        # Low comfort also drives complaints
        score += (1.0 - needs.comfort) * 0.3
        # Negative mood amplifies
        if mood in _NEGATIVE_MOODS:
            score += 0.2
        # Low stimulation makes creature more vocal
        score += (1.0 - needs.stimulation) * 0.15
        return _clamp(score)

    @staticmethod
    def _score_sleep(
        needs: CreatureNeeds,
        mood: CreatureMood,
        time_context: dict[str, Any],
    ) -> float:
        """Tired, content, or nighttime creatures sleep more."""
        score = 0.1
        # Nighttime strongly encourages sleep
        if time_context.get("time_of_day") == "night":
            score += 0.5
        elif time_context.get("time_of_day") == "evening":
            score += 0.2
        # Low stimulation = drowsy
        score += (1.0 - needs.stimulation) * 0.2
        # Content mood encourages rest
        if mood == CreatureMood.CONTENT:
            score += 0.15
        # Poor health makes creature rest
        score += (1.0 - needs.health) * 0.2
        return _clamp(score)

    @staticmethod
    def _score_eat(needs: CreatureNeeds) -> float:
        """Hungry creatures forage/nibble more."""
        score = 0.1
        score += needs.hunger * 0.6
        # Not eating when full
        if needs.hunger < 0.2:
            score = 0.05
        return _clamp(score)

    @staticmethod
    def _score_observe(
        mood: CreatureMood,
        traits: TraitProfile | None,
    ) -> float:
        """Curious or philosophical creatures make observations."""
        score = 0.25
        if mood in {CreatureMood.CURIOUS, CreatureMood.PHILOSOPHICAL}:
            score += 0.35
        if mood == CreatureMood.AMUSED:
            score += 0.2
        # Curiosity trait boosts observations
        if traits is not None:
            score += traits.curiosity * 0.2
            score += traits.wit * 0.1
        return _clamp(score)

    @staticmethod
    def _score_tap_glass(needs: CreatureNeeds, mood: CreatureMood) -> float:
        """Bored or irritated creatures tap the glass for attention."""
        score = 0.15
        # Low stimulation = attention seeking
        score += (1.0 - needs.stimulation) * 0.4
        # Irritated creatures tap angrily
        if mood == CreatureMood.IRRITATED:
            score += 0.3
        elif mood == CreatureMood.HOSTILE:
            score += 0.2
        return _clamp(score)

    @staticmethod
    def _score_idle_swim(mood: CreatureMood) -> float:
        """Default swimming behavior — always somewhat likely."""
        score = 0.35
        if mood in _POSITIVE_MOODS:
            score += 0.1
        return _clamp(score)

    def _pick_message(self, btype: BehaviorType, category: str) -> str:
        """Pick the next message for a behavior type, cycling through options."""
        messages_by_cat = _BEHAVIOR_MESSAGES.get(btype, {})
        messages = messages_by_cat.get(category, messages_by_cat.get("neutral", ["..."]))
        if not messages:
            return "..."

        idx = self._message_index.get(btype, 0)
        msg = messages[idx % len(messages)]
        self._message_index[btype] = idx + 1
        return msg


# ── Situation prompts for LLM-powered verbal behaviors ───────────────

# Mood-keyed situation templates for verbal behavior types.
# Each template describes the creature's current emotional context.
_BEHAVIOR_SITUATIONS: dict[BehaviorType, dict[str, str]] = {
    BehaviorType.COMPLAIN: {
        "negative": "You are deeply unhappy and want to voice your displeasure.",
        "neutral": "You feel neglected and want to remind your owner you exist.",
        "positive": "You're in a decent mood but still have something to grumble about.",
    },
    BehaviorType.OBSERVE: {
        "negative": "You are brooding and notice something about your bleak situation.",
        "neutral": "You are observing your surroundings with sardonic detachment.",
        "positive": "You are in a reflective mood, noticing something interesting.",
    },
}


def get_behavior_situation(
    behavior_type: BehaviorType,
    mood: CreatureMood,
    needs: CreatureNeeds,
) -> str | None:
    """Build an enriched situation string for a verbal behavior.

    Combines a mood-keyed situation template with concrete needs context
    so the LLM can produce a remark that reflects the creature's actual state.

    Args:
        behavior_type: The type of verbal behavior (COMPLAIN or OBSERVE).
        mood: The creature's current mood.
        needs: The creature's current needs.

    Returns:
        A situation string for the LLM, or None if the behavior type
        has no situation templates.
    """
    templates = _BEHAVIOR_SITUATIONS.get(behavior_type)
    if templates is None:
        return None

    category = _mood_category(mood)
    situation = templates.get(category, templates.get("neutral", ""))

    # Append concrete needs context
    parts: list[str] = [situation]
    if needs.hunger > 0.6:
        parts.append(f"You are very hungry (hunger: {needs.hunger:.0%}).")
    elif needs.hunger > 0.3:
        parts.append(f"You are somewhat hungry (hunger: {needs.hunger:.0%}).")
    if needs.comfort < 0.4:
        parts.append(f"Your comfort is low ({needs.comfort:.0%}).")
    if needs.stimulation < 0.3:
        parts.append(f"You are bored (stimulation: {needs.stimulation:.0%}).")
    if needs.health < 0.5:
        parts.append(f"You are unwell (health: {needs.health:.0%}).")

    return " ".join(parts)
