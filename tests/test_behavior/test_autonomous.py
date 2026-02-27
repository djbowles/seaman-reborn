"""Tests for the autonomous behavior system."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from seaman_brain.behavior.autonomous import (
    VERBAL_BEHAVIORS,
    BehaviorEngine,
    BehaviorType,
    IdleBehavior,
    _mood_category,
    get_behavior_situation,
)
from seaman_brain.behavior.mood import CreatureMood
from seaman_brain.needs.system import CreatureNeeds
from seaman_brain.personality.traits import TraitProfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_needs() -> CreatureNeeds:
    """Neutral needs — everything moderate."""
    return CreatureNeeds(hunger=0.3, comfort=0.7, health=0.8, stimulation=0.5)


@pytest.fixture
def hungry_needs() -> CreatureNeeds:
    """Very hungry creature."""
    return CreatureNeeds(hunger=0.9, comfort=0.7, health=0.8, stimulation=0.5)


@pytest.fixture
def bored_needs() -> CreatureNeeds:
    """Bored creature with low stimulation."""
    return CreatureNeeds(hunger=0.2, comfort=0.8, health=0.9, stimulation=0.1)


@pytest.fixture
def healthy_needs() -> CreatureNeeds:
    """Well-fed, comfortable creature."""
    return CreatureNeeds(hunger=0.05, comfort=0.95, health=1.0, stimulation=0.8)


@pytest.fixture
def default_traits() -> TraitProfile:
    return TraitProfile()


@pytest.fixture
def curious_traits() -> TraitProfile:
    return TraitProfile(curiosity=0.9, wit=0.8)


@pytest.fixture
def base_time_context() -> dict:
    return {"time_of_day": "afternoon", "is_weekend": False, "absence_severity": "none"}


@pytest.fixture
def night_time_context() -> dict:
    return {"time_of_day": "night", "is_weekend": False, "absence_severity": "none"}


@pytest.fixture
def creature_state() -> dict:
    return {
        "stage": "mushroomer",
        "hunger": 0.3,
        "health": 0.8,
        "interaction_count": 5,
    }


def _make_engine(
    base_cooldown: float = 30.0,
    cooldowns: dict | None = None,
    now: datetime | None = None,
) -> BehaviorEngine:
    """Create a BehaviorEngine with a fixed clock."""
    fixed_now = now or datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC)
    return BehaviorEngine(
        base_cooldown=base_cooldown,
        behavior_cooldowns=cooldowns,
        now_func=lambda: fixed_now,
    )


# ---------------------------------------------------------------------------
# BehaviorType and IdleBehavior tests
# ---------------------------------------------------------------------------

class TestBehaviorType:
    """Tests for the BehaviorType enum."""

    def test_all_types_exist(self) -> None:
        assert len(BehaviorType) == 6
        expected = {"idle_swim", "tap_glass", "complain", "observe", "sleep", "eat"}
        assert {bt.value for bt in BehaviorType} == expected

    def test_enum_values_are_strings(self) -> None:
        for bt in BehaviorType:
            assert isinstance(bt.value, str)


class TestIdleBehavior:
    """Tests for the IdleBehavior dataclass."""

    def test_creation_with_defaults(self) -> None:
        b = IdleBehavior(action_type=BehaviorType.IDLE_SWIM, message="*swims*")
        assert b.action_type == BehaviorType.IDLE_SWIM
        assert b.message == "*swims*"
        assert b.animation_hint == ""
        assert b.priority == 0.5

    def test_creation_with_all_fields(self) -> None:
        b = IdleBehavior(
            action_type=BehaviorType.COMPLAIN,
            message="Feed me!",
            animation_hint="talking",
            priority=0.9,
        )
        assert b.action_type == BehaviorType.COMPLAIN
        assert b.message == "Feed me!"
        assert b.animation_hint == "talking"
        assert b.priority == 0.9


# ---------------------------------------------------------------------------
# Mood category helper tests
# ---------------------------------------------------------------------------

class TestMoodCategory:
    """Tests for _mood_category helper."""

    def test_negative_moods(self) -> None:
        assert _mood_category(CreatureMood.HOSTILE) == "negative"
        assert _mood_category(CreatureMood.IRRITATED) == "negative"

    def test_neutral_moods(self) -> None:
        assert _mood_category(CreatureMood.SARDONIC) == "neutral"
        assert _mood_category(CreatureMood.NEUTRAL) == "neutral"

    def test_positive_moods(self) -> None:
        assert _mood_category(CreatureMood.CURIOUS) == "positive"
        assert _mood_category(CreatureMood.AMUSED) == "positive"
        assert _mood_category(CreatureMood.PHILOSOPHICAL) == "positive"
        assert _mood_category(CreatureMood.CONTENT) == "positive"


# ---------------------------------------------------------------------------
# BehaviorEngine - behavior selection tests
# ---------------------------------------------------------------------------

class TestBehaviorSelection:
    """Tests for get_idle_behavior selection logic."""

    def test_returns_behavior_with_moderate_needs(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is not None
        assert isinstance(result, IdleBehavior)
        assert isinstance(result.action_type, BehaviorType)
        assert result.message != ""

    def test_hungry_creature_prefers_complain_or_eat(
        self, hungry_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, hungry_needs, CreatureMood.IRRITATED, base_time_context
        )
        assert result is not None
        assert result.action_type in {BehaviorType.COMPLAIN, BehaviorType.EAT}

    def test_bored_creature_taps_glass(
        self, bored_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, bored_needs, CreatureMood.IRRITATED, base_time_context
        )
        assert result is not None
        # Bored + irritated should prioritize tap_glass or complain
        assert result.action_type in {
            BehaviorType.TAP_GLASS, BehaviorType.COMPLAIN, BehaviorType.OBSERVE
        }

    def test_nighttime_encourages_sleep(
        self, healthy_needs, night_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, healthy_needs, CreatureMood.CONTENT, night_time_context
        )
        assert result is not None
        assert result.action_type == BehaviorType.SLEEP

    def test_curious_creature_observes(
        self, healthy_needs, base_time_context, curious_traits, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state,
            healthy_needs,
            CreatureMood.CURIOUS,
            base_time_context,
            traits=curious_traits,
        )
        assert result is not None
        assert result.action_type == BehaviorType.OBSERVE

    def test_behavior_has_animation_hint(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is not None
        assert result.animation_hint != ""

    def test_priority_is_clamped(
        self, hungry_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, hungry_needs, CreatureMood.HOSTILE, base_time_context
        )
        assert result is not None
        assert 0.0 <= result.priority <= 1.0


# ---------------------------------------------------------------------------
# BehaviorEngine - cooldown tests
# ---------------------------------------------------------------------------

class TestCooldownSystem:
    """Tests for cooldown enforcement."""

    def test_first_call_never_on_cooldown(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is not None

    def test_same_behavior_on_cooldown_after_trigger(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        """After a behavior fires, it should be on cooldown."""
        now = datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC)
        engine = BehaviorEngine(
            base_cooldown=30.0,
            now_func=lambda: now,
        )
        result1 = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result1 is not None
        triggered_type = result1.action_type

        remaining = engine.get_cooldown_remaining(triggered_type)
        assert remaining > 0.0

    def test_cooldown_expires_after_enough_time(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        """After cooldown expires, behavior should be available again."""
        times = [
            datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC),
            datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC),
            datetime(2026, 2, 25, 14, 5, 0, tzinfo=UTC),  # 5 min later
            datetime(2026, 2, 25, 14, 5, 0, tzinfo=UTC),
        ]
        time_iter = iter(times)
        engine = BehaviorEngine(
            base_cooldown=30.0,
            now_func=lambda: next(time_iter),
        )
        result1 = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result1 is not None
        # 5 minutes later, all cooldowns should be expired
        result2 = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result2 is not None

    def test_all_on_cooldown_returns_none(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        """When all behaviors are on cooldown, returns None."""
        now = datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC)
        # Set very long cooldowns so nothing resets
        long_cooldowns = {bt: 9999.0 for bt in BehaviorType}
        engine = BehaviorEngine(
            base_cooldown=9999.0,
            behavior_cooldowns=long_cooldowns,
            now_func=lambda: now,
        )
        # Trigger all behavior types by calling many times — but they'll only
        # pick the top one each time. Instead, manually set cooldowns.
        for bt in BehaviorType:
            engine._last_triggered[bt] = now

        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is None

    def test_reset_cooldowns_clears_all(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        now = datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC)
        engine = BehaviorEngine(now_func=lambda: now)
        # Trigger a behavior
        engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert len(engine._last_triggered) > 0

        engine.reset_cooldowns()
        assert len(engine._last_triggered) == 0

    def test_get_cooldown_remaining_untriggered(self) -> None:
        engine = _make_engine()
        assert engine.get_cooldown_remaining(BehaviorType.IDLE_SWIM) == 0.0

    def test_get_cooldown_remaining_triggered(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        now = datetime(2026, 2, 25, 14, 0, 0, tzinfo=UTC)
        engine = BehaviorEngine(now_func=lambda: now)
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is not None
        remaining = engine.get_cooldown_remaining(result.action_type)
        assert remaining > 0.0

    def test_custom_cooldown_override(self) -> None:
        engine = _make_engine(cooldowns={BehaviorType.IDLE_SWIM: 999.0})
        assert engine._cooldowns[BehaviorType.IDLE_SWIM] == 999.0
        # Other defaults remain
        assert engine._cooldowns[BehaviorType.TAP_GLASS] == 60.0


# ---------------------------------------------------------------------------
# BehaviorEngine - need-driven priority tests
# ---------------------------------------------------------------------------

class TestNeedDrivenPriorities:
    """Tests for need-driven behavior scoring."""

    def test_high_hunger_boosts_complain_score(self) -> None:
        engine = _make_engine()
        hungry = CreatureNeeds(hunger=0.95, comfort=0.5, health=0.5, stimulation=0.5)
        full = CreatureNeeds(hunger=0.05, comfort=0.5, health=0.5, stimulation=0.5)

        score_hungry = engine._score_complain(hungry, CreatureMood.NEUTRAL)
        score_full = engine._score_complain(full, CreatureMood.NEUTRAL)
        assert score_hungry > score_full

    def test_negative_mood_amplifies_complain(self) -> None:
        engine = _make_engine()
        needs = CreatureNeeds(hunger=0.5, comfort=0.5, health=0.5, stimulation=0.5)
        score_hostile = engine._score_complain(needs, CreatureMood.HOSTILE)
        score_neutral = engine._score_complain(needs, CreatureMood.NEUTRAL)
        assert score_hostile > score_neutral

    def test_low_stimulation_boosts_tap_glass(self) -> None:
        engine = _make_engine()
        bored = CreatureNeeds(hunger=0.2, comfort=0.8, health=0.9, stimulation=0.1)
        engaged = CreatureNeeds(hunger=0.2, comfort=0.8, health=0.9, stimulation=0.9)

        score_bored = engine._score_tap_glass(bored, CreatureMood.NEUTRAL)
        score_engaged = engine._score_tap_glass(engaged, CreatureMood.NEUTRAL)
        assert score_bored > score_engaged

    def test_nighttime_boosts_sleep(self) -> None:
        engine = _make_engine()
        needs = CreatureNeeds(hunger=0.2, comfort=0.8, health=0.9, stimulation=0.3)
        night = {"time_of_day": "night"}
        afternoon = {"time_of_day": "afternoon"}

        score_night = engine._score_sleep(needs, CreatureMood.NEUTRAL, night)
        score_afternoon = engine._score_sleep(needs, CreatureMood.NEUTRAL, afternoon)
        assert score_night > score_afternoon

    def test_hunger_boosts_eat(self) -> None:
        engine = _make_engine()
        hungry = CreatureNeeds(hunger=0.8, comfort=0.5, health=0.5, stimulation=0.5)
        full = CreatureNeeds(hunger=0.05, comfort=0.5, health=0.5, stimulation=0.5)

        score_hungry = engine._score_eat(hungry)
        score_full = engine._score_eat(full)
        assert score_hungry > score_full

    def test_curiosity_trait_boosts_observe(self) -> None:
        engine = _make_engine()
        curious = TraitProfile(curiosity=0.9, wit=0.8)
        dull = TraitProfile(curiosity=0.1, wit=0.1)

        score_curious = engine._score_observe(CreatureMood.CURIOUS, curious)
        score_dull = engine._score_observe(CreatureMood.NEUTRAL, dull)
        assert score_curious > score_dull

    def test_full_creature_low_eat_score(self) -> None:
        engine = _make_engine()
        full = CreatureNeeds(hunger=0.05, comfort=0.9, health=1.0, stimulation=0.8)
        score = engine._score_eat(full)
        assert score < 0.15

    def test_idle_swim_always_has_base_score(self) -> None:
        engine = _make_engine()
        score = engine._score_idle_swim(CreatureMood.NEUTRAL)
        assert score >= 0.3

    def test_idle_swim_positive_mood_boost(self) -> None:
        engine = _make_engine()
        positive = engine._score_idle_swim(CreatureMood.CONTENT)
        neutral = engine._score_idle_swim(CreatureMood.NEUTRAL)
        assert positive > neutral


# ---------------------------------------------------------------------------
# BehaviorEngine - message selection tests
# ---------------------------------------------------------------------------

class TestMessageSelection:
    """Tests for message picking and cycling."""

    def test_message_matches_mood_category(
        self, hungry_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, hungry_needs, CreatureMood.HOSTILE, base_time_context
        )
        assert result is not None
        assert result.message != ""
        assert result.message != "..."

    def test_messages_cycle_on_repeated_calls(self) -> None:
        """Repeated calls should cycle through available messages."""
        engine = _make_engine()
        msgs: list[str] = []
        # Manually pick messages
        for _ in range(6):
            msg = engine._pick_message(BehaviorType.IDLE_SWIM, "neutral")
            msgs.append(msg)
        # Should cycle — after 3 messages it wraps
        assert msgs[0] == msgs[3]
        assert msgs[1] == msgs[4]

    def test_unknown_category_falls_back_to_neutral(self) -> None:
        engine = _make_engine()
        msg = engine._pick_message(BehaviorType.IDLE_SWIM, "unknown_category")
        assert msg != ""


# ---------------------------------------------------------------------------
# BehaviorEngine - generate_idle_comment tests
# ---------------------------------------------------------------------------

class TestGenerateIdleComment:
    """Tests for LLM-powered idle comment generation."""

    @pytest.mark.asyncio
    async def test_generates_comment_with_llm(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = "  How tedious.  "

        engine = _make_engine()
        behavior = IdleBehavior(
            action_type=BehaviorType.OBSERVE, message="*looks around*"
        )
        result = await engine.generate_idle_comment(behavior, mock_llm)
        assert result == "How tedious."
        mock_llm.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_when_no_llm(self) -> None:
        engine = _make_engine()
        behavior = IdleBehavior(
            action_type=BehaviorType.OBSERVE, message="*looks around*"
        )
        result = await engine.generate_idle_comment(behavior, None)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_llm_error(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.side_effect = ConnectionError("LLM unavailable")

        engine = _make_engine()
        behavior = IdleBehavior(
            action_type=BehaviorType.IDLE_SWIM, message="*swims*"
        )
        result = await engine.generate_idle_comment(behavior, mock_llm)
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_response(self) -> None:
        mock_llm = AsyncMock()
        mock_llm.chat.return_value = ""

        engine = _make_engine()
        behavior = IdleBehavior(
            action_type=BehaviorType.IDLE_SWIM, message="*swims*"
        )
        result = await engine.generate_idle_comment(behavior, mock_llm)
        assert result is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_time_context(
        self, base_needs, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, {}
        )
        assert result is not None

    def test_none_traits(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context, traits=None
        )
        assert result is not None

    def test_zero_base_cooldown(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        engine = _make_engine(base_cooldown=0.0)
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is not None

    def test_negative_base_cooldown_clamped(self) -> None:
        engine = BehaviorEngine(base_cooldown=-10.0)
        assert engine._base_cooldown == 0.0

    def test_extreme_needs_values(
        self, base_time_context, creature_state
    ) -> None:
        """Extreme needs should not crash."""
        extreme = CreatureNeeds(hunger=1.0, comfort=0.0, health=0.0, stimulation=0.0)
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, extreme, CreatureMood.HOSTILE, base_time_context
        )
        assert result is not None
        assert 0.0 <= result.priority <= 1.0

    def test_all_needs_perfect(
        self, base_time_context, creature_state
    ) -> None:
        """Perfect needs should still yield behavior."""
        perfect = CreatureNeeds(hunger=0.0, comfort=1.0, health=1.0, stimulation=1.0)
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, perfect, CreatureMood.CONTENT, base_time_context
        )
        assert result is not None

    def test_all_moods_produce_valid_category(self) -> None:
        for mood in CreatureMood:
            cat = _mood_category(mood)
            assert cat in {"negative", "neutral", "positive"}


# ---------------------------------------------------------------------------
# VERBAL_BEHAVIORS and needs_llm tests
# ---------------------------------------------------------------------------

class TestVerbalBehaviors:
    """Tests for VERBAL_BEHAVIORS constant and needs_llm flag."""

    def test_verbal_behaviors_contains_complain_and_observe(self) -> None:
        assert BehaviorType.COMPLAIN in VERBAL_BEHAVIORS
        assert BehaviorType.OBSERVE in VERBAL_BEHAVIORS

    def test_verbal_behaviors_excludes_non_verbal(self) -> None:
        assert BehaviorType.IDLE_SWIM not in VERBAL_BEHAVIORS
        assert BehaviorType.TAP_GLASS not in VERBAL_BEHAVIORS
        assert BehaviorType.SLEEP not in VERBAL_BEHAVIORS
        assert BehaviorType.EAT not in VERBAL_BEHAVIORS

    def test_verbal_behaviors_is_frozenset(self) -> None:
        assert isinstance(VERBAL_BEHAVIORS, frozenset)

    def test_needs_llm_set_for_complain(
        self, hungry_needs, base_time_context, creature_state
    ) -> None:
        """COMPLAIN behavior gets needs_llm=True."""
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, hungry_needs, CreatureMood.HOSTILE, base_time_context
        )
        assert result is not None
        if result.action_type == BehaviorType.COMPLAIN:
            assert result.needs_llm is True

    def test_needs_llm_set_for_observe(
        self, healthy_needs, base_time_context, curious_traits, creature_state
    ) -> None:
        """OBSERVE behavior gets needs_llm=True."""
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state,
            healthy_needs,
            CreatureMood.CURIOUS,
            base_time_context,
            traits=curious_traits,
        )
        assert result is not None
        if result.action_type == BehaviorType.OBSERVE:
            assert result.needs_llm is True

    def test_needs_llm_false_for_idle_swim(
        self, base_needs, base_time_context, creature_state
    ) -> None:
        """IDLE_SWIM behavior gets needs_llm=False."""
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, base_needs, CreatureMood.NEUTRAL, base_time_context
        )
        assert result is not None
        if result.action_type == BehaviorType.IDLE_SWIM:
            assert result.needs_llm is False

    def test_needs_llm_false_for_sleep(
        self, healthy_needs, night_time_context, creature_state
    ) -> None:
        """SLEEP behavior gets needs_llm=False."""
        engine = _make_engine()
        result = engine.get_idle_behavior(
            creature_state, healthy_needs, CreatureMood.CONTENT, night_time_context
        )
        assert result is not None
        if result.action_type == BehaviorType.SLEEP:
            assert result.needs_llm is False

    def test_idle_behavior_needs_llm_default_false(self) -> None:
        """IdleBehavior.needs_llm defaults to False."""
        b = IdleBehavior(action_type=BehaviorType.IDLE_SWIM, message="*swims*")
        assert b.needs_llm is False


# ---------------------------------------------------------------------------
# get_behavior_situation tests
# ---------------------------------------------------------------------------

class TestGetBehaviorSituation:
    """Tests for get_behavior_situation() helper."""

    def test_complain_negative_mood(self, hungry_needs) -> None:
        result = get_behavior_situation(
            BehaviorType.COMPLAIN, CreatureMood.HOSTILE, hungry_needs
        )
        assert result is not None
        assert "unhappy" in result.lower() or "displeasure" in result.lower()

    def test_observe_positive_mood(self, healthy_needs) -> None:
        result = get_behavior_situation(
            BehaviorType.OBSERVE, CreatureMood.CURIOUS, healthy_needs
        )
        assert result is not None
        assert "reflective" in result.lower() or "interesting" in result.lower()

    def test_observe_neutral_mood(self, base_needs) -> None:
        result = get_behavior_situation(
            BehaviorType.OBSERVE, CreatureMood.NEUTRAL, base_needs
        )
        assert result is not None
        assert "sardonic" in result.lower() or "observing" in result.lower()

    def test_non_verbal_returns_none(self, base_needs) -> None:
        """Non-verbal behavior types return None."""
        result = get_behavior_situation(
            BehaviorType.IDLE_SWIM, CreatureMood.NEUTRAL, base_needs
        )
        assert result is None

    def test_hunger_context_appended(self) -> None:
        """High hunger is mentioned in the situation string."""
        needs = CreatureNeeds(hunger=0.8, comfort=0.7, health=0.9, stimulation=0.5)
        result = get_behavior_situation(
            BehaviorType.COMPLAIN, CreatureMood.IRRITATED, needs
        )
        assert result is not None
        assert "hungry" in result.lower()

    def test_low_comfort_context_appended(self) -> None:
        """Low comfort is mentioned in the situation string."""
        needs = CreatureNeeds(hunger=0.2, comfort=0.2, health=0.9, stimulation=0.5)
        result = get_behavior_situation(
            BehaviorType.COMPLAIN, CreatureMood.NEUTRAL, needs
        )
        assert result is not None
        assert "comfort" in result.lower()

    def test_low_stimulation_context_appended(self) -> None:
        """Low stimulation is mentioned in the situation string."""
        needs = CreatureNeeds(hunger=0.2, comfort=0.7, health=0.9, stimulation=0.1)
        result = get_behavior_situation(
            BehaviorType.OBSERVE, CreatureMood.NEUTRAL, needs
        )
        assert result is not None
        assert "bored" in result.lower()

    def test_low_health_context_appended(self) -> None:
        """Low health is mentioned in the situation string."""
        needs = CreatureNeeds(hunger=0.2, comfort=0.7, health=0.3, stimulation=0.5)
        result = get_behavior_situation(
            BehaviorType.COMPLAIN, CreatureMood.NEUTRAL, needs
        )
        assert result is not None
        assert "unwell" in result.lower()

    def test_no_extra_context_when_needs_fine(self, healthy_needs) -> None:
        """When all needs are fine, no extra context is appended."""
        result = get_behavior_situation(
            BehaviorType.OBSERVE, CreatureMood.CONTENT, healthy_needs
        )
        assert result is not None
        assert "hungry" not in result.lower()
        assert "bored" not in result.lower()
        assert "unwell" not in result.lower()
