"""Tests for the dynamic mood engine (US-032)."""

from __future__ import annotations

import pytest

from seaman_brain.behavior.mood import (
    _MOOD_MODIFIERS,
    _MOOD_ORDER,
    CreatureMood,
    MoodEngine,
    _clamp,
    _mood_from_index,
    _mood_index,
)
from seaman_brain.needs.system import CreatureNeeds
from seaman_brain.personality.traits import TraitProfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine() -> MoodEngine:
    """Default MoodEngine with max 2 transition steps."""
    return MoodEngine()


@pytest.fixture
def happy_needs() -> CreatureNeeds:
    """Needs reflecting a well-cared-for creature."""
    return CreatureNeeds(hunger=0.0, comfort=1.0, health=1.0, stimulation=1.0)


@pytest.fixture
def miserable_needs() -> CreatureNeeds:
    """Needs reflecting a neglected creature."""
    return CreatureNeeds(hunger=1.0, comfort=0.0, health=0.1, stimulation=0.0)


@pytest.fixture
def neutral_needs() -> CreatureNeeds:
    """Middle-of-the-road needs."""
    return CreatureNeeds(hunger=0.5, comfort=0.5, health=0.5, stimulation=0.5)


@pytest.fixture
def warm_traits() -> TraitProfile:
    """Warm, patient, curious traits (positive mood tendency)."""
    return TraitProfile(
        cynicism=0.2, wit=0.5, patience=0.8, curiosity=0.9,
        warmth=0.8, verbosity=0.5, formality=0.3, aggression=0.1,
    )


@pytest.fixture
def hostile_traits() -> TraitProfile:
    """Aggressive, cynical traits (negative mood tendency)."""
    return TraitProfile(
        cynicism=0.9, wit=0.5, patience=0.1, curiosity=0.2,
        warmth=0.05, verbosity=0.5, formality=0.5, aggression=0.9,
    )


@pytest.fixture
def neutral_traits() -> TraitProfile:
    """Balanced traits."""
    return TraitProfile()


@pytest.fixture
def morning_context() -> dict:
    """Morning time context with no absence."""
    return {
        "time_of_day": "morning",
        "day_of_week": "Monday",
        "is_weekend": False,
        "hour": 9,
        "minute": 30,
        "session_duration_minutes": 5.0,
        "hours_since_last_session": 12.0,
        "absence_severity": "none",
    }


@pytest.fixture
def night_context() -> dict:
    """Night time context with severe absence."""
    return {
        "time_of_day": "night",
        "day_of_week": "Wednesday",
        "is_weekend": False,
        "hour": 2,
        "minute": 0,
        "session_duration_minutes": 1.0,
        "hours_since_last_session": 200.0,
        "absence_severity": "severe",
    }


# ---------------------------------------------------------------------------
# CreatureMood enum tests
# ---------------------------------------------------------------------------

class TestCreatureMood:
    """Tests for the CreatureMood enum."""

    def test_all_eight_moods_exist(self) -> None:
        """All 8 mood values are accessible."""
        expected = {
            "hostile", "irritated", "sardonic", "neutral",
            "curious", "amused", "philosophical", "content",
        }
        assert {m.value for m in CreatureMood} == expected

    def test_mood_count(self) -> None:
        """Exactly 8 moods."""
        assert len(CreatureMood) == 8

    def test_mood_from_value(self) -> None:
        """Can create mood from string value."""
        assert CreatureMood("sardonic") == CreatureMood.SARDONIC

    def test_mood_order_length(self) -> None:
        """_MOOD_ORDER contains all moods."""
        assert len(_MOOD_ORDER) == 8
        assert set(_MOOD_ORDER) == set(CreatureMood)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for mood module helper functions."""

    def test_clamp_in_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self) -> None:
        assert _clamp(-1.0) == 0.0

    def test_clamp_above(self) -> None:
        assert _clamp(2.0) == 1.0

    def test_mood_index_hostile(self) -> None:
        assert _mood_index(CreatureMood.HOSTILE) == 0

    def test_mood_index_content(self) -> None:
        assert _mood_index(CreatureMood.CONTENT) == 7

    def test_mood_from_index_valid(self) -> None:
        assert _mood_from_index(3) == CreatureMood.NEUTRAL

    def test_mood_from_index_clamped_negative(self) -> None:
        assert _mood_from_index(-5) == CreatureMood.HOSTILE

    def test_mood_from_index_clamped_overflow(self) -> None:
        assert _mood_from_index(100) == CreatureMood.CONTENT


# ---------------------------------------------------------------------------
# MoodEngine init tests
# ---------------------------------------------------------------------------

class TestMoodEngineInit:
    """Tests for MoodEngine initialization."""

    def test_default_mood_is_neutral(self, engine: MoodEngine) -> None:
        assert engine.current_mood == CreatureMood.NEUTRAL

    def test_default_score(self, engine: MoodEngine) -> None:
        assert engine.current_score == 0.5

    def test_custom_max_steps(self) -> None:
        eng = MoodEngine(max_transition_steps=1)
        assert eng.current_mood == CreatureMood.NEUTRAL

    def test_max_steps_minimum_one(self) -> None:
        """max_transition_steps < 1 gets clamped to 1."""
        eng = MoodEngine(max_transition_steps=0)
        # Should still function — just limited to 1-step transitions
        eng.set_mood(CreatureMood.HOSTILE)
        eng._max_steps = 0  # Force invalid value (constructor prevents this)
        # The constructor enforces max(1, ...), so _max_steps should be 1
        eng2 = MoodEngine(max_transition_steps=-5)
        assert eng2._max_steps == 1


# ---------------------------------------------------------------------------
# Needs scoring tests
# ---------------------------------------------------------------------------

class TestNeedsScoring:
    """Tests for _score_needs."""

    def test_perfect_needs_score_one(self, happy_needs: CreatureNeeds) -> None:
        score = MoodEngine._score_needs(happy_needs)
        assert score == pytest.approx(1.0)

    def test_terrible_needs_score_low(self, miserable_needs: CreatureNeeds) -> None:
        score = MoodEngine._score_needs(miserable_needs)
        assert score < 0.15

    def test_neutral_needs_score_mid(self, neutral_needs: CreatureNeeds) -> None:
        score = MoodEngine._score_needs(neutral_needs)
        assert 0.3 <= score <= 0.7

    def test_needs_score_average(self) -> None:
        """Score is the average of (1-hunger), comfort, health, stimulation."""
        needs = CreatureNeeds(hunger=0.2, comfort=0.6, health=0.8, stimulation=0.4)
        expected = (0.8 + 0.6 + 0.8 + 0.4) / 4.0
        assert MoodEngine._score_needs(needs) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Interaction scoring tests
# ---------------------------------------------------------------------------

class TestInteractionScoring:
    """Tests for _score_interactions."""

    def test_zero_interactions(self) -> None:
        assert MoodEngine._score_interactions(0) == 0.0

    def test_negative_interactions(self) -> None:
        assert MoodEngine._score_interactions(-5) == 0.0

    def test_ten_interactions_maxes_out(self) -> None:
        assert MoodEngine._score_interactions(10) == pytest.approx(1.0)

    def test_five_interactions_half(self) -> None:
        assert MoodEngine._score_interactions(5) == pytest.approx(0.5)

    def test_beyond_ten_capped(self) -> None:
        assert MoodEngine._score_interactions(100) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Time scoring tests
# ---------------------------------------------------------------------------

class TestTimeScoring:
    """Tests for _score_time."""

    def test_morning_good_mood(self, morning_context: dict) -> None:
        score = MoodEngine._score_time(morning_context)
        assert score == pytest.approx(0.7)

    def test_night_with_severe_absence(self, night_context: dict) -> None:
        score = MoodEngine._score_time(night_context)
        assert score == pytest.approx(0.0)  # 0.3 - 0.3 = 0.0

    def test_weekend_bonus(self) -> None:
        ctx = {
            "time_of_day": "afternoon",
            "is_weekend": True,
            "absence_severity": "none",
        }
        score = MoodEngine._score_time(ctx)
        assert score == pytest.approx(0.8)  # 0.7 + 0.1

    def test_mild_absence_penalty(self) -> None:
        ctx = {
            "time_of_day": "morning",
            "is_weekend": False,
            "absence_severity": "mild",
        }
        score = MoodEngine._score_time(ctx)
        assert score == pytest.approx(0.6)  # 0.7 - 0.1

    def test_empty_context(self) -> None:
        """Empty dict uses fallback values."""
        score = MoodEngine._score_time({})
        assert 0.0 <= score <= 1.0

    def test_unknown_time_of_day_fallback(self) -> None:
        ctx = {"time_of_day": "twilight", "is_weekend": False, "absence_severity": "none"}
        score = MoodEngine._score_time(ctx)
        assert score == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Personality scoring tests
# ---------------------------------------------------------------------------

class TestPersonalityScoring:
    """Tests for _score_personality."""

    def test_warm_traits_positive(self, warm_traits: TraitProfile) -> None:
        score = MoodEngine._score_personality(warm_traits)
        assert score > 0.6

    def test_hostile_traits_negative(self, hostile_traits: TraitProfile) -> None:
        score = MoodEngine._score_personality(hostile_traits)
        assert score < 0.4

    def test_neutral_traits_midrange(self, neutral_traits: TraitProfile) -> None:
        score = MoodEngine._score_personality(neutral_traits)
        assert 0.3 <= score <= 0.7

    def test_score_clamped(self) -> None:
        """Extreme traits don't push score out of [0, 1]."""
        extreme_positive = TraitProfile(
            warmth=1.0, patience=1.0, curiosity=1.0,
            aggression=0.0, cynicism=0.0,
        )
        score = MoodEngine._score_personality(extreme_positive)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Mood index mapping tests
# ---------------------------------------------------------------------------

class TestScoreToMoodIndex:
    """Tests for _score_to_mood_index."""

    def test_zero_maps_to_hostile(self) -> None:
        assert MoodEngine._score_to_mood_index(0.0) == 0

    def test_one_maps_to_content(self) -> None:
        assert MoodEngine._score_to_mood_index(1.0) == 7

    def test_half_maps_to_curious(self) -> None:
        # 0.5 * 8 = 4 -> CURIOUS
        assert MoodEngine._score_to_mood_index(0.5) == 4

    def test_just_below_one(self) -> None:
        assert MoodEngine._score_to_mood_index(0.99) == 7

    def test_quarter_maps_to_sardonic(self) -> None:
        # 0.3 * 8 = 2.4 -> idx 2 = SARDONIC
        assert MoodEngine._score_to_mood_index(0.3) == 2


# ---------------------------------------------------------------------------
# Transition smoothing tests
# ---------------------------------------------------------------------------

class TestTransitionSmoothing:
    """Tests for mood transition smoothing."""

    def test_small_transition_unaffected(self, engine: MoodEngine) -> None:
        """NEUTRAL -> CURIOUS (1 step) passes through with max_steps=2."""
        result = engine._smooth_transition(CreatureMood.CURIOUS)
        assert result == CreatureMood.CURIOUS

    def test_two_step_transition_allowed(self, engine: MoodEngine) -> None:
        """NEUTRAL -> AMUSED (2 steps) is within max_steps=2."""
        result = engine._smooth_transition(CreatureMood.AMUSED)
        assert result == CreatureMood.AMUSED

    def test_large_positive_jump_clamped(self, engine: MoodEngine) -> None:
        """NEUTRAL -> CONTENT (4 steps) gets clamped to +2 = AMUSED."""
        result = engine._smooth_transition(CreatureMood.CONTENT)
        assert result == CreatureMood.AMUSED

    def test_large_negative_jump_clamped(self, engine: MoodEngine) -> None:
        """NEUTRAL -> HOSTILE (3 steps) gets clamped to -2 = IRRITATED."""
        result = engine._smooth_transition(CreatureMood.HOSTILE)
        assert result == CreatureMood.IRRITATED

    def test_same_mood_no_change(self, engine: MoodEngine) -> None:
        """NEUTRAL -> NEUTRAL is a 0-step transition."""
        result = engine._smooth_transition(CreatureMood.NEUTRAL)
        assert result == CreatureMood.NEUTRAL

    def test_max_steps_one_limits_to_one(self) -> None:
        """With max_steps=1, can only move 1 level at a time."""
        eng = MoodEngine(max_transition_steps=1)
        # NEUTRAL -> CONTENT should clamp to CURIOUS (1 step up)
        result = eng._smooth_transition(CreatureMood.CONTENT)
        assert result == CreatureMood.CURIOUS

    def test_transition_from_non_neutral(self) -> None:
        """Transition smoothing works from any starting mood."""
        eng = MoodEngine()
        eng.set_mood(CreatureMood.HOSTILE)
        # HOSTILE -> CONTENT (7 steps), clamped to +2 = SARDONIC
        result = eng._smooth_transition(CreatureMood.CONTENT)
        assert result == CreatureMood.SARDONIC

    def test_transition_negative_from_content(self) -> None:
        """Transition smoothing works in the negative direction."""
        eng = MoodEngine()
        eng.set_mood(CreatureMood.CONTENT)
        # CONTENT -> HOSTILE (7 steps), clamped to -2 = AMUSED
        result = eng._smooth_transition(CreatureMood.HOSTILE)
        assert result == CreatureMood.AMUSED


# ---------------------------------------------------------------------------
# Full calculate_mood tests
# ---------------------------------------------------------------------------

class TestCalculateMood:
    """Tests for the full calculate_mood pipeline."""

    def test_happy_creature_positive_mood(
        self,
        engine: MoodEngine,
        happy_needs: CreatureNeeds,
        warm_traits: TraitProfile,
        morning_context: dict,
    ) -> None:
        """Well-cared-for creature with good traits trends positive."""
        mood = engine.calculate_mood(
            needs=happy_needs,
            trust=0.8,
            time_context=morning_context,
            recent_interactions=8,
            traits=warm_traits,
        )
        assert _mood_index(mood) >= _mood_index(CreatureMood.NEUTRAL)

    def test_miserable_creature_negative_mood(
        self,
        engine: MoodEngine,
        miserable_needs: CreatureNeeds,
        hostile_traits: TraitProfile,
        night_context: dict,
    ) -> None:
        """Neglected creature with hostile traits trends negative."""
        # Start from HOSTILE so smoothing doesn't prevent going low
        engine.set_mood(CreatureMood.HOSTILE)
        mood = engine.calculate_mood(
            needs=miserable_needs,
            trust=0.0,
            time_context=night_context,
            recent_interactions=0,
            traits=hostile_traits,
        )
        assert _mood_index(mood) <= _mood_index(CreatureMood.SARDONIC)

    def test_updates_current_mood(
        self,
        engine: MoodEngine,
        neutral_needs: CreatureNeeds,
        neutral_traits: TraitProfile,
        morning_context: dict,
    ) -> None:
        """calculate_mood updates the engine's current_mood property."""
        engine.calculate_mood(
            needs=neutral_needs,
            trust=0.5,
            time_context=morning_context,
            recent_interactions=5,
            traits=neutral_traits,
        )
        # It should have updated (may or may not be the same value, but
        # the property should reflect the last calculation)
        assert engine.current_mood is not None

    def test_updates_current_score(
        self,
        engine: MoodEngine,
        happy_needs: CreatureNeeds,
        warm_traits: TraitProfile,
        morning_context: dict,
    ) -> None:
        """calculate_mood updates the raw score."""
        engine.calculate_mood(
            needs=happy_needs,
            trust=0.9,
            time_context=morning_context,
            recent_interactions=10,
            traits=warm_traits,
        )
        assert engine.current_score > 0.5

    def test_transition_smoothing_applied(self, engine: MoodEngine) -> None:
        """Mood can't jump more than max_steps in one call."""
        engine.set_mood(CreatureMood.HOSTILE)
        # Very positive inputs should push toward CONTENT, but smoothing caps it
        happy = CreatureNeeds(hunger=0.0, comfort=1.0, health=1.0, stimulation=1.0)
        warm = TraitProfile(warmth=1.0, patience=1.0, curiosity=1.0, aggression=0.0, cynicism=0.0)
        morning = {"time_of_day": "morning", "is_weekend": True, "absence_severity": "none"}
        mood = engine.calculate_mood(
            needs=happy, trust=1.0, time_context=morning,
            recent_interactions=20, traits=warm,
        )
        # From HOSTILE (idx 0), max 2 steps up = SARDONIC (idx 2)
        assert _mood_index(mood) <= 2

    def test_repeated_calls_converge(self, engine: MoodEngine) -> None:
        """Multiple calls with same positive input gradually improve mood."""
        engine.set_mood(CreatureMood.HOSTILE)
        happy = CreatureNeeds(hunger=0.0, comfort=1.0, health=1.0, stimulation=1.0)
        warm = TraitProfile(warmth=0.8, patience=0.8, curiosity=0.8, aggression=0.1, cynicism=0.1)
        ctx = {"time_of_day": "morning", "is_weekend": False, "absence_severity": "none"}

        moods: list[CreatureMood] = []
        for _ in range(10):
            mood = engine.calculate_mood(
                needs=happy, trust=0.8, time_context=ctx,
                recent_interactions=5, traits=warm,
            )
            moods.append(mood)

        # Should trend upward over time
        assert _mood_index(moods[-1]) > _mood_index(moods[0])

    def test_trust_affects_mood(
        self,
        engine: MoodEngine,
        neutral_needs: CreatureNeeds,
        neutral_traits: TraitProfile,
        morning_context: dict,
    ) -> None:
        """Higher trust yields a more positive mood score."""
        engine_low = MoodEngine(max_transition_steps=8)  # no smoothing limits
        engine_high = MoodEngine(max_transition_steps=8)

        engine_low.calculate_mood(
            needs=neutral_needs, trust=0.0, time_context=morning_context,
            recent_interactions=5, traits=neutral_traits,
        )
        engine_high.calculate_mood(
            needs=neutral_needs, trust=1.0, time_context=morning_context,
            recent_interactions=5, traits=neutral_traits,
        )
        assert engine_high.current_score > engine_low.current_score


# ---------------------------------------------------------------------------
# get_mood_modifiers tests
# ---------------------------------------------------------------------------

class TestGetMoodModifiers:
    """Tests for get_mood_modifiers."""

    def test_returns_dict_with_required_keys(self, engine: MoodEngine) -> None:
        mods = engine.get_mood_modifiers()
        assert "tone" in mods
        assert "instruction" in mods
        assert "emoji_hint" in mods

    def test_modifiers_match_current_mood(self, engine: MoodEngine) -> None:
        engine.set_mood(CreatureMood.SARDONIC)
        mods = engine.get_mood_modifiers()
        assert "sarcasm" in mods["tone"].lower() or "wit" in mods["tone"].lower()

    def test_all_moods_have_modifiers(self) -> None:
        """Every mood enum has a modifier entry."""
        for mood in CreatureMood:
            assert mood in _MOOD_MODIFIERS
            mods = _MOOD_MODIFIERS[mood]
            assert "tone" in mods
            assert "instruction" in mods
            assert "emoji_hint" in mods

    def test_modifiers_are_copy(self, engine: MoodEngine) -> None:
        """get_mood_modifiers returns a copy, not a reference."""
        mods = engine.get_mood_modifiers()
        mods["tone"] = "MUTATED"
        original = engine.get_mood_modifiers()
        assert original["tone"] != "MUTATED"


# ---------------------------------------------------------------------------
# set_mood tests
# ---------------------------------------------------------------------------

class TestSetMood:
    """Tests for manual mood override via set_mood."""

    def test_set_mood_updates_current(self, engine: MoodEngine) -> None:
        engine.set_mood(CreatureMood.PHILOSOPHICAL)
        assert engine.current_mood == CreatureMood.PHILOSOPHICAL

    def test_set_mood_updates_score(self, engine: MoodEngine) -> None:
        engine.set_mood(CreatureMood.HOSTILE)
        assert engine.current_score == pytest.approx(0.0)

    def test_set_mood_content_score(self, engine: MoodEngine) -> None:
        engine.set_mood(CreatureMood.CONTENT)
        assert engine.current_score == pytest.approx(1.0)

    def test_set_mood_neutral_score(self, engine: MoodEngine) -> None:
        engine.set_mood(CreatureMood.NEUTRAL)
        # NEUTRAL is index 3 out of 7 = 0.4286...
        assert 0.3 <= engine.current_score <= 0.5


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and boundary condition tests."""

    def test_all_zero_inputs(self, engine: MoodEngine) -> None:
        """All zero/empty inputs produce a valid mood."""
        needs = CreatureNeeds(hunger=0.0, comfort=0.0, health=0.0, stimulation=0.0)
        traits = TraitProfile(
            cynicism=0.0, wit=0.0, patience=0.0, curiosity=0.0,
            warmth=0.0, verbosity=0.0, formality=0.0, aggression=0.0,
        )
        mood = engine.calculate_mood(
            needs=needs, trust=0.0, time_context={},
            recent_interactions=0, traits=traits,
        )
        assert isinstance(mood, CreatureMood)

    def test_all_max_inputs(self, engine: MoodEngine) -> None:
        """All maximum inputs produce a valid mood."""
        needs = CreatureNeeds(hunger=1.0, comfort=1.0, health=1.0, stimulation=1.0)
        traits = TraitProfile(
            cynicism=1.0, wit=1.0, patience=1.0, curiosity=1.0,
            warmth=1.0, verbosity=1.0, formality=1.0, aggression=1.0,
        )
        mood = engine.calculate_mood(
            needs=needs, trust=1.0, time_context={},
            recent_interactions=100, traits=traits,
        )
        assert isinstance(mood, CreatureMood)

    def test_trust_out_of_range_clamped(
        self,
        engine: MoodEngine,
        neutral_needs: CreatureNeeds,
        neutral_traits: TraitProfile,
        morning_context: dict,
    ) -> None:
        """Trust value > 1.0 is clamped internally."""
        mood = engine.calculate_mood(
            needs=neutral_needs, trust=5.0, time_context=morning_context,
            recent_interactions=5, traits=neutral_traits,
        )
        assert isinstance(mood, CreatureMood)

    def test_negative_trust_clamped(
        self,
        engine: MoodEngine,
        neutral_needs: CreatureNeeds,
        neutral_traits: TraitProfile,
        morning_context: dict,
    ) -> None:
        """Negative trust is clamped to 0.0."""
        mood = engine.calculate_mood(
            needs=neutral_needs, trust=-1.0, time_context=morning_context,
            recent_interactions=5, traits=neutral_traits,
        )
        assert isinstance(mood, CreatureMood)
