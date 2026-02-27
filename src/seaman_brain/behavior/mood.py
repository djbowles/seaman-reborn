"""Dynamic mood engine - emergent mood from needs, trust, time, and traits.

Mood is a dynamic emergent property of the creature's needs satisfaction,
trust level, recent interactions, time of day, and personality traits.
It affects response tone via prompt modifiers and is displayed in the GUI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from seaman_brain.needs.system import CreatureNeeds
from seaman_brain.personality.traits import TraitProfile


class CreatureMood(Enum):
    """Named mood states ordered from most negative to most positive.

    The numeric ordering (0=HOSTILE, 7=CONTENT) is used internally for
    mood scoring and transition clamping.
    """

    HOSTILE = "hostile"
    IRRITATED = "irritated"
    SARDONIC = "sardonic"
    NEUTRAL = "neutral"
    CURIOUS = "curious"
    AMUSED = "amused"
    PHILOSOPHICAL = "philosophical"
    CONTENT = "content"


# Ordered list for index-based operations.
_MOOD_ORDER: tuple[CreatureMood, ...] = (
    CreatureMood.HOSTILE,
    CreatureMood.IRRITATED,
    CreatureMood.SARDONIC,
    CreatureMood.NEUTRAL,
    CreatureMood.CURIOUS,
    CreatureMood.AMUSED,
    CreatureMood.PHILOSOPHICAL,
    CreatureMood.CONTENT,
)

# Prompt modifier text for each mood.
_MOOD_MODIFIERS: dict[CreatureMood, dict[str, str]] = {
    CreatureMood.HOSTILE: {
        "tone": "aggressive and confrontational",
        "instruction": "Snap at the human. Be aggressive and hostile.",
        "emoji_hint": "angry",
    },
    CreatureMood.IRRITATED: {
        "tone": "annoyed and impatient",
        "instruction": "Everything the human says annoys you. Be short and curt.",
        "emoji_hint": "annoyed",
    },
    CreatureMood.SARDONIC: {
        "tone": "biting sarcasm and dark wit",
        "instruction": "Deploy peak sarcasm and cutting observations.",
        "emoji_hint": "smirk",
    },
    CreatureMood.NEUTRAL: {
        "tone": "baseline personality",
        "instruction": "Respond with your default personality traits.",
        "emoji_hint": "neutral",
    },
    CreatureMood.CURIOUS: {
        "tone": "inquisitive and probing",
        "instruction": "Ask questions. Probe the human's life and motives.",
        "emoji_hint": "thinking",
    },
    CreatureMood.AMUSED: {
        "tone": "darkly entertained",
        "instruction": "Something has tickled your dark sense of humor. Be witty.",
        "emoji_hint": "amused",
    },
    CreatureMood.PHILOSOPHICAL: {
        "tone": "contemplative and existential",
        "instruction": "Ponder existence. Mix cynicism with genuine insight.",
        "emoji_hint": "thoughtful",
    },
    CreatureMood.CONTENT: {
        "tone": "unusually relaxed and mellow",
        "instruction": "You are unusually content. Still sarcastic, but less biting.",
        "emoji_hint": "relaxed",
    },
}


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def _mood_index(mood: CreatureMood) -> int:
    """Get the ordinal index of a mood in _MOOD_ORDER."""
    return _MOOD_ORDER.index(mood)


def _mood_from_index(idx: int) -> CreatureMood:
    """Get mood from a clamped ordinal index."""
    clamped = max(0, min(len(_MOOD_ORDER) - 1, idx))
    return _MOOD_ORDER[clamped]


class MoodEngine:
    """Calculates creature mood as a weighted combination of multiple factors.

    Mood is computed as a 0.0-1.0 score from:
    - Needs satisfaction (30%): how well-fed, comfortable, healthy the creature is
    - Trust level (20%): how much the creature trusts the user
    - Interaction quality (20%): recent positive interactions boost mood
    - Time factors (15%): time of day and session context
    - Personality baseline (15%): trait-driven mood tendency

    The score maps to one of 8 discrete moods (HOSTILE to CONTENT).
    Transitions are smoothed to prevent jarring mood jumps.

    Args:
        max_transition_steps: Maximum mood levels that can change per update
            (default 2). Prevents instant jumps from HOSTILE to CONTENT.
    """

    def __init__(self, max_transition_steps: int = 2) -> None:
        self._current_mood: CreatureMood = CreatureMood.NEUTRAL
        self._current_score: float = 0.5
        self._max_steps = max(1, max_transition_steps)

    @property
    def current_mood(self) -> CreatureMood:
        """The most recently calculated mood."""
        return self._current_mood

    @property
    def current_score(self) -> float:
        """The raw mood score (0.0=hostile, 1.0=content)."""
        return self._current_score

    def calculate_mood(
        self,
        needs: CreatureNeeds,
        trust: float,
        time_context: dict[str, Any],
        recent_interactions: int,
        traits: TraitProfile,
    ) -> CreatureMood:
        """Calculate the creature's mood from all contributing factors.

        Args:
            needs: Current creature needs snapshot.
            trust: Current trust level (0.0-1.0).
            time_context: Time context dict from GameClock.get_time_context().
            recent_interactions: Number of interactions in the current session.
            traits: Current personality trait profile.

        Returns:
            The computed CreatureMood after transition smoothing.
        """
        # Component scores (each 0.0 to 1.0, higher = happier)
        needs_score = self._score_needs(needs)
        trust_score = _clamp(trust)
        interaction_score = self._score_interactions(recent_interactions)
        time_score = self._score_time(time_context)
        personality_score = self._score_personality(traits)

        # Weighted combination
        raw_score = (
            0.30 * needs_score
            + 0.20 * trust_score
            + 0.20 * interaction_score
            + 0.15 * time_score
            + 0.15 * personality_score
        )
        raw_score = _clamp(raw_score)

        # Map score to target mood index
        target_idx = self._score_to_mood_index(raw_score)
        target_mood = _mood_from_index(target_idx)

        # Apply transition smoothing
        smoothed_mood = self._smooth_transition(target_mood)

        # Update internal state
        self._current_score = raw_score
        self._current_mood = smoothed_mood
        return smoothed_mood

    def get_mood_modifiers(self) -> dict[str, str]:
        """Return prompt modifier dict for the current mood.

        Returns:
            Dict with keys: tone, instruction, emoji_hint.
        """
        return dict(_MOOD_MODIFIERS[self._current_mood])

    def set_mood(self, mood: CreatureMood) -> None:
        """Override the current mood directly (e.g., for events or testing).

        Args:
            mood: The mood to set.
        """
        self._current_mood = mood
        self._current_score = _mood_index(mood) / max(1, len(_MOOD_ORDER) - 1)

    @staticmethod
    def _score_needs(needs: CreatureNeeds) -> float:
        """Score needs satisfaction from 0.0 (terrible) to 1.0 (all satisfied).

        Components:
        - Inverse hunger (0=starving => 0.0, 0=full => 1.0)
        - Comfort directly
        - Health directly
        - Stimulation directly
        """
        inv_hunger = 1.0 - needs.hunger
        avg = (inv_hunger + needs.comfort + needs.health + needs.stimulation) / 4.0
        return _clamp(avg)

    @staticmethod
    def _score_interactions(recent_interactions: int) -> float:
        """Score interaction quality from 0.0 (none) to 1.0 (well-engaged).

        Diminishing returns: first few interactions matter most.
        Score = min(1.0, interactions / 10).
        """
        if recent_interactions <= 0:
            return 0.0
        return _clamp(recent_interactions / 10.0)

    @staticmethod
    def _score_time(time_context: dict[str, Any]) -> float:
        """Score time factors from 0.0 (worst) to 1.0 (best mood time).

        Factors:
        - Time of day: morning/afternoon are best (0.7), evening ok (0.5), night worst (0.3)
        - Long absence: severe absence penalizes mood
        - Weekend bonus: slight mood boost on weekends
        """
        base = 0.5

        time_of_day = time_context.get("time_of_day", "")
        tod_map: dict[str, float] = {
            "morning": 0.7,
            "afternoon": 0.7,
            "evening": 0.5,
            "night": 0.3,
        }
        base = tod_map.get(time_of_day, 0.5)

        # Absence penalty
        absence = time_context.get("absence_severity", "none")
        absence_penalty: dict[str, float] = {
            "none": 0.0,
            "mild": -0.1,
            "moderate": -0.2,
            "severe": -0.3,
        }
        base += absence_penalty.get(absence, 0.0)

        # Weekend bonus
        if time_context.get("is_weekend", False):
            base += 0.1

        return _clamp(base)

    @staticmethod
    def _score_personality(traits: TraitProfile) -> float:
        """Score personality's baseline mood tendency.

        Positive factors: warmth, patience, curiosity (push mood up).
        Negative factors: aggression, cynicism (push mood down).
        """
        positive = (traits.warmth + traits.patience + traits.curiosity) / 3.0
        negative = (traits.aggression + traits.cynicism) / 2.0
        score = 0.5 + 0.3 * (positive - negative)
        return _clamp(score)

    @staticmethod
    def _score_to_mood_index(score: float) -> int:
        """Map a 0.0-1.0 score to a mood index (0-7).

        Uses evenly spaced thresholds:
        0.000-0.125 -> HOSTILE (0)
        0.125-0.250 -> IRRITATED (1)
        0.250-0.375 -> SARDONIC (2)
        0.375-0.500 -> NEUTRAL (3)
        0.500-0.625 -> CURIOUS (4)
        0.625-0.750 -> AMUSED (5)
        0.750-0.875 -> PHILOSOPHICAL (6)
        0.875-1.000 -> CONTENT (7)
        """
        num_moods = len(_MOOD_ORDER)
        idx = int(score * num_moods)
        return max(0, min(num_moods - 1, idx))

    def _smooth_transition(self, target: CreatureMood) -> CreatureMood:
        """Apply transition smoothing so mood doesn't jump too far.

        Limits the change to max_transition_steps per update.

        Args:
            target: The desired new mood.

        Returns:
            The actual mood after clamping the transition distance.
        """
        current_idx = _mood_index(self._current_mood)
        target_idx = _mood_index(target)
        delta = target_idx - current_idx

        if abs(delta) <= self._max_steps:
            return target

        # Clamp the jump
        if delta > 0:
            new_idx = current_idx + self._max_steps
        else:
            new_idx = current_idx - self._max_steps

        return _mood_from_index(new_idx)
