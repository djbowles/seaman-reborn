"""Tests for the event system — scheduled, random, and stage-triggered events."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from seaman_brain.behavior.events import (
    EventEffect,
    EventSystem,
    EventType,
    GameEvent,
    _build_default_events,
    _clamp,
)
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def now() -> datetime:
    return datetime(2026, 2, 25, 14, 30, 0, tzinfo=UTC)


@pytest.fixture
def state() -> CreatureState:
    return CreatureState(
        stage=CreatureStage.MUSHROOMER,
        interaction_count=5,
        trust_level=0.3,
        hunger=0.2,
        health=0.9,
        comfort=0.8,
    )


@pytest.fixture
def tank() -> TankEnvironment:
    return TankEnvironment(
        temperature=24.0,
        cleanliness=0.9,
        oxygen_level=0.8,
        water_level=1.0,
    )


@pytest.fixture
def time_context() -> dict:
    return {
        "hour": 14,
        "time_of_day": "afternoon",
        "day_of_week": "Tuesday",
        "is_weekend": False,
        "absence_severity": "none",
        "hours_since_last_session": 2.0,
        "session_duration_minutes": 10.0,
    }


@pytest.fixture
def system(now: datetime) -> EventSystem:
    return EventSystem(
        include_defaults=False,
        now_func=lambda: now,
    )


@pytest.fixture
def default_system(now: datetime) -> EventSystem:
    return EventSystem(
        include_defaults=True,
        now_func=lambda: now,
    )


def _always_true(
    _state: CreatureState,
    _tank: TankEnvironment,
    _time: dict,
) -> bool:
    return True


def _always_false(
    _state: CreatureState,
    _tank: TankEnvironment,
    _time: dict,
) -> bool:
    return False


# ---------------------------------------------------------------------------
# EventType enum tests
# ---------------------------------------------------------------------------

class TestEventType:
    def test_all_values(self) -> None:
        assert EventType.EVOLUTION_READY.value == "evolution_ready"
        assert EventType.BREEDING.value == "breeding"
        assert EventType.HOLIDAY.value == "holiday"
        assert EventType.MILESTONE.value == "milestone"
        assert EventType.RANDOM_OBSERVATION.value == "random_observation"
        assert EventType.ENVIRONMENTAL.value == "environmental"

    def test_count(self) -> None:
        assert len(EventType) == 6


# ---------------------------------------------------------------------------
# EventEffect tests
# ---------------------------------------------------------------------------

class TestEventEffect:
    def test_defaults(self) -> None:
        effect = EventEffect()
        assert effect.mood_change == 0.0
        assert effect.trust_change == 0.0
        assert effect.hunger_change == 0.0
        assert effect.health_change == 0.0
        assert effect.tank_changes == {}
        assert effect.trigger_dialogue is False

    def test_custom_values(self) -> None:
        effect = EventEffect(
            mood_change=-0.2,
            trust_change=0.1,
            hunger_change=0.05,
            health_change=-0.1,
            tank_changes={"cleanliness": 0.5},
            trigger_dialogue=True,
        )
        assert effect.mood_change == -0.2
        assert effect.trust_change == 0.1
        assert effect.tank_changes == {"cleanliness": 0.5}
        assert effect.trigger_dialogue is True


# ---------------------------------------------------------------------------
# GameEvent tests
# ---------------------------------------------------------------------------

class TestGameEvent:
    def test_defaults(self) -> None:
        event = GameEvent(
            event_type=EventType.MILESTONE,
            name="test",
            message="hello",
        )
        assert event.one_shot is False
        assert event.cooldown_seconds == 0.0
        assert event.priority == 0.5
        assert isinstance(event.effects, EventEffect)

    def test_custom_event(self) -> None:
        event = GameEvent(
            event_type=EventType.BREEDING,
            name="breed_event",
            message="Life finds a way.",
            effects=EventEffect(mood_change=0.1),
            one_shot=True,
            cooldown_seconds=3600.0,
            priority=0.9,
        )
        assert event.one_shot is True
        assert event.cooldown_seconds == 3600.0
        assert event.priority == 0.9


# ---------------------------------------------------------------------------
# _clamp helper
# ---------------------------------------------------------------------------

class TestClamp:
    def test_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_below_min(self) -> None:
        assert _clamp(-0.5) == 0.0

    def test_above_max(self) -> None:
        assert _clamp(1.5) == 1.0

    def test_custom_range(self) -> None:
        assert _clamp(15.0, 0.0, 10.0) == 10.0


# ---------------------------------------------------------------------------
# EventSystem init
# ---------------------------------------------------------------------------

class TestEventSystemInit:
    def test_empty_no_defaults(self) -> None:
        system = EventSystem(include_defaults=False)
        assert system.get_registered_event_names() == []
        assert system.get_fired_one_shots() == set()

    def test_with_defaults(self) -> None:
        system = EventSystem(include_defaults=True)
        names = system.get_registered_event_names()
        assert len(names) > 0
        assert "late_night_visit" in names
        assert "dirty_tank" in names
        assert "milestone_10" in names

    def test_custom_now_func(self, now: datetime) -> None:
        system = EventSystem(
            include_defaults=False,
            now_func=lambda: now,
        )
        # The system accepts the now_func without error
        assert system.get_registered_event_names() == []

    def test_rng_seed_deterministic(self) -> None:
        s1 = EventSystem(include_defaults=False, rng_seed=42)
        s2 = EventSystem(include_defaults=False, rng_seed=42)
        assert s1._rng.random() == s2._rng.random()


# ---------------------------------------------------------------------------
# register_event / unregister_event
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_event(
        self, system: EventSystem,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "test_ev", "msg")
        system.register_event(event, _always_true)
        assert "test_ev" in system.get_registered_event_names()

    def test_register_multiple(self, system: EventSystem) -> None:
        for i in range(3):
            system.register_event(
                GameEvent(EventType.MILESTONE, f"ev_{i}", f"msg_{i}"),
                _always_true,
            )
        assert len(system.get_registered_event_names()) == 3

    def test_unregister_existing(self, system: EventSystem) -> None:
        event = GameEvent(EventType.MILESTONE, "to_remove", "msg")
        system.register_event(event, _always_true)
        assert system.unregister_event("to_remove") is True
        assert "to_remove" not in system.get_registered_event_names()

    def test_unregister_nonexistent(self, system: EventSystem) -> None:
        assert system.unregister_event("nope") is False

    def test_unregister_preserves_others(self, system: EventSystem) -> None:
        system.register_event(
            GameEvent(EventType.MILESTONE, "keep", "msg"), _always_true,
        )
        system.register_event(
            GameEvent(EventType.MILESTONE, "remove", "msg"), _always_true,
        )
        system.unregister_event("remove")
        assert system.get_registered_event_names() == ["keep"]


# ---------------------------------------------------------------------------
# check_events — basic
# ---------------------------------------------------------------------------

class TestCheckEvents:
    def test_fires_matching_event(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "basic", "basic msg")
        system.register_event(event, _always_true)
        fired = system.check_events(state, tank, time_context)
        assert len(fired) == 1
        assert fired[0].name == "basic"

    def test_skips_false_condition(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "nope", "nope")
        system.register_event(event, _always_false)
        fired = system.check_events(state, tank, time_context)
        assert fired == []

    def test_sorted_by_priority(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        system.register_event(
            GameEvent(EventType.MILESTONE, "low", "low", priority=0.2),
            _always_true,
        )
        system.register_event(
            GameEvent(EventType.MILESTONE, "high", "high", priority=0.9),
            _always_true,
        )
        system.register_event(
            GameEvent(EventType.MILESTONE, "mid", "mid", priority=0.5),
            _always_true,
        )
        fired = system.check_events(state, tank, time_context)
        assert [e.name for e in fired] == ["high", "mid", "low"]

    def test_condition_exception_logged(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        def bad_cond(_s, _t, _tc):
            raise ValueError("boom")

        system.register_event(
            GameEvent(EventType.MILESTONE, "bad", "bad"),
            bad_cond,
        )
        # Should not raise, just skip
        fired = system.check_events(state, tank, time_context)
        assert fired == []

    def test_empty_system_returns_empty(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        fired = system.check_events(state, tank, time_context)
        assert fired == []


# ---------------------------------------------------------------------------
# One-shot events
# ---------------------------------------------------------------------------

class TestOneShot:
    def test_one_shot_fires_once(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "once", "once", one_shot=True)
        system.register_event(event, _always_true)

        fired_1 = system.check_events(state, tank, time_context)
        assert len(fired_1) == 1

        fired_2 = system.check_events(state, tank, time_context)
        assert fired_2 == []

    def test_one_shot_tracked_in_fired_set(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "tracked", "msg", one_shot=True)
        system.register_event(event, _always_true)
        system.check_events(state, tank, time_context)
        assert "tracked" in system.get_fired_one_shots()

    def test_non_one_shot_fires_repeatedly(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "repeat", "msg", one_shot=False)
        system.register_event(event, _always_true)

        fired_1 = system.check_events(state, tank, time_context)
        fired_2 = system.check_events(state, tank, time_context)
        assert len(fired_1) == 1
        assert len(fired_2) == 1


# ---------------------------------------------------------------------------
# Cooldown events
# ---------------------------------------------------------------------------

class TestCooldown:
    def test_cooldown_blocks_second_fire(
        self,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
        now: datetime,
    ) -> None:
        time = [now]
        system = EventSystem(
            include_defaults=False,
            now_func=lambda: time[0],
        )
        event = GameEvent(
            EventType.MILESTONE, "cd", "msg", cooldown_seconds=60.0,
        )
        system.register_event(event, _always_true)

        # First fire
        fired_1 = system.check_events(state, tank, time_context)
        assert len(fired_1) == 1

        # Still within cooldown (30s later)
        time[0] = now + timedelta(seconds=30)
        fired_2 = system.check_events(state, tank, time_context)
        assert fired_2 == []

    def test_cooldown_expires(
        self,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
        now: datetime,
    ) -> None:
        time = [now]
        system = EventSystem(
            include_defaults=False,
            now_func=lambda: time[0],
        )
        event = GameEvent(
            EventType.MILESTONE, "cd_exp", "msg", cooldown_seconds=60.0,
        )
        system.register_event(event, _always_true)

        system.check_events(state, tank, time_context)

        # Advance past cooldown
        time[0] = now + timedelta(seconds=61)
        fired = system.check_events(state, tank, time_context)
        assert len(fired) == 1

    def test_zero_cooldown_always_fires(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(
            EventType.MILESTONE, "no_cd", "msg", cooldown_seconds=0.0,
        )
        system.register_event(event, _always_true)
        for _ in range(3):
            fired = system.check_events(state, tank, time_context)
            assert len(fired) == 1


# ---------------------------------------------------------------------------
# apply_effects
# ---------------------------------------------------------------------------

class TestApplyEffects:
    def test_trust_change(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        event = GameEvent(
            EventType.MILESTONE, "trust", "msg",
            effects=EventEffect(trust_change=0.1),
        )
        old_trust = state.trust_level
        system.apply_effects(event, state, tank)
        assert state.trust_level == pytest.approx(old_trust + 0.1)

    def test_hunger_change(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        event = GameEvent(
            EventType.MILESTONE, "hunger", "msg",
            effects=EventEffect(hunger_change=0.3),
        )
        old_hunger = state.hunger
        system.apply_effects(event, state, tank)
        assert state.hunger == pytest.approx(old_hunger + 0.3)

    def test_health_change(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        event = GameEvent(
            EventType.MILESTONE, "health", "msg",
            effects=EventEffect(health_change=-0.2),
        )
        old_health = state.health
        system.apply_effects(event, state, tank)
        assert state.health == pytest.approx(old_health - 0.2)

    def test_tank_changes(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        event = GameEvent(
            EventType.MILESTONE, "tank", "msg",
            effects=EventEffect(tank_changes={"cleanliness": 0.3}),
        )
        system.apply_effects(event, state, tank)
        assert tank.cleanliness == 0.3

    def test_clamped_trust(
        self,
        system: EventSystem,
        tank: TankEnvironment,
    ) -> None:
        state = CreatureState(trust_level=0.95)
        event = GameEvent(
            EventType.MILESTONE, "clamp", "msg",
            effects=EventEffect(trust_change=0.2),
        )
        system.apply_effects(event, state, tank)
        assert state.trust_level == 1.0

    def test_clamped_hunger_below_zero(
        self,
        system: EventSystem,
        tank: TankEnvironment,
    ) -> None:
        state = CreatureState(hunger=0.05)
        event = GameEvent(
            EventType.MILESTONE, "clamp_h", "msg",
            effects=EventEffect(hunger_change=-0.5),
        )
        system.apply_effects(event, state, tank)
        assert state.hunger == 0.0

    def test_no_effects(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "noop", "msg")
        old_trust = state.trust_level
        old_hunger = state.hunger
        system.apply_effects(event, state, tank)
        assert state.trust_level == old_trust
        assert state.hunger == old_hunger

    def test_tank_changes_unknown_field_ignored(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        event = GameEvent(
            EventType.MILESTONE, "bad_field", "msg",
            effects=EventEffect(tank_changes={"nonexistent_field": 42}),
        )
        # Should not raise
        system.apply_effects(event, state, tank)


# ---------------------------------------------------------------------------
# reset / reset_event
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_one_shots(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "once", "msg", one_shot=True)
        system.register_event(event, _always_true)
        system.check_events(state, tank, time_context)
        assert "once" in system.get_fired_one_shots()

        system.reset()
        assert system.get_fired_one_shots() == set()

        # Can fire again
        fired = system.check_events(state, tank, time_context)
        assert len(fired) == 1

    def test_reset_clears_cooldowns(
        self,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
        now: datetime,
    ) -> None:
        system = EventSystem(
            include_defaults=False,
            now_func=lambda: now,
        )
        event = GameEvent(
            EventType.MILESTONE, "cd", "msg", cooldown_seconds=9999.0,
        )
        system.register_event(event, _always_true)
        system.check_events(state, tank, time_context)

        # On cooldown
        fired = system.check_events(state, tank, time_context)
        assert fired == []

        system.reset()
        fired = system.check_events(state, tank, time_context)
        assert len(fired) == 1

    def test_reset_event_specific(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
    ) -> None:
        ev1 = GameEvent(EventType.MILESTONE, "ev1", "msg1", one_shot=True)
        ev2 = GameEvent(EventType.MILESTONE, "ev2", "msg2", one_shot=True)
        system.register_event(ev1, _always_true)
        system.register_event(ev2, _always_true)
        system.check_events(state, tank, time_context)

        result = system.reset_event("ev1")
        assert result is True
        assert "ev1" not in system.get_fired_one_shots()
        assert "ev2" in system.get_fired_one_shots()

    def test_reset_event_nonexistent(self, system: EventSystem) -> None:
        assert system.reset_event("nope") is False


# ---------------------------------------------------------------------------
# Persistence — to_dict / load_state
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_to_dict_empty(self, system: EventSystem) -> None:
        d = system.to_dict()
        assert d == {"fired_one_shots": [], "last_fired": {}}

    def test_to_dict_with_state(
        self,
        system: EventSystem,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
        now: datetime,
    ) -> None:
        event = GameEvent(EventType.MILESTONE, "persist", "msg", one_shot=True)
        system.register_event(event, _always_true)
        system.check_events(state, tank, time_context)

        d = system.to_dict()
        assert "persist" in d["fired_one_shots"]
        assert "persist" in d["last_fired"]
        assert d["last_fired"]["persist"] == now.isoformat()

    def test_load_state_roundtrip(
        self,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
        now: datetime,
    ) -> None:
        system1 = EventSystem(
            include_defaults=False,
            now_func=lambda: now,
        )
        event = GameEvent(EventType.MILESTONE, "round", "msg", one_shot=True)
        system1.register_event(event, _always_true)
        system1.check_events(state, tank, time_context)
        data = system1.to_dict()

        system2 = EventSystem(
            include_defaults=False,
            now_func=lambda: now,
        )
        system2.register_event(event, _always_true)
        system2.load_state(data)

        # One-shot should be blocked
        fired = system2.check_events(state, tank, time_context)
        assert fired == []
        assert "round" in system2.get_fired_one_shots()

    def test_load_state_empty(self, system: EventSystem) -> None:
        system.load_state({})
        assert system.get_fired_one_shots() == set()

    def test_load_state_partial(self, system: EventSystem) -> None:
        system.load_state({"fired_one_shots": ["x", "y"]})
        assert system.get_fired_one_shots() == {"x", "y"}


# ---------------------------------------------------------------------------
# Built-in event factories
# ---------------------------------------------------------------------------

class TestBuiltInEvents:
    def test_default_events_count(self) -> None:
        events = _build_default_events()
        # 4 evolution + 3 stage + 3 time + 3 milestones + 2 environmental = 15
        assert len(events) == 15

    def test_all_events_have_unique_names(self) -> None:
        events = _build_default_events()
        names = [ev.name for ev, _ in events]
        assert len(names) == len(set(names))

    def test_evolution_ready_events_present(self) -> None:
        events = _build_default_events()
        names = [ev.name for ev, _ in events]
        assert "evolution_ready_mushroomer" in names
        assert "evolution_ready_gillman" in names
        assert "evolution_ready_podfish" in names
        assert "evolution_ready_tadman" in names

    def test_evolution_ready_are_one_shot(self) -> None:
        events = _build_default_events()
        evo_events = [
            ev for ev, _ in events
            if ev.name.startswith("evolution_ready_")
        ]
        for ev in evo_events:
            assert ev.one_shot is True

    def test_milestone_events_present(self) -> None:
        events = _build_default_events()
        names = [ev.name for ev, _ in events]
        assert "milestone_10" in names
        assert "milestone_50" in names
        assert "milestone_100" in names


# ---------------------------------------------------------------------------
# Built-in event trigger conditions
# ---------------------------------------------------------------------------

class TestBuiltInTriggers:
    def test_late_night_fires_at_3am(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        late_night = [
            (ev, cond) for ev, cond in events if ev.name == "late_night_visit"
        ][0]
        _, condition = late_night

        night_ctx = {"hour": 3, "is_weekend": False, "absence_severity": "none"}
        assert condition(state, tank, night_ctx) is True

    def test_late_night_does_not_fire_at_noon(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        late_night = [
            (ev, cond) for ev, cond in events if ev.name == "late_night_visit"
        ][0]
        _, condition = late_night

        noon_ctx = {"hour": 12, "is_weekend": False, "absence_severity": "none"}
        assert condition(state, tank, noon_ctx) is False

    def test_weekend_fires_on_weekend(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        weekend = [
            (ev, cond) for ev, cond in events if ev.name == "weekend_observation"
        ][0]
        _, condition = weekend
        ctx = {"is_weekend": True}
        assert condition(state, tank, ctx) is True

    def test_weekend_no_fire_weekday(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        weekend = [
            (ev, cond) for ev, cond in events if ev.name == "weekend_observation"
        ][0]
        _, condition = weekend
        ctx = {"is_weekend": False}
        assert condition(state, tank, ctx) is False

    def test_long_absence_moderate(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        absence = [
            (ev, cond) for ev, cond in events if ev.name == "long_absence"
        ][0]
        _, condition = absence
        ctx = {"absence_severity": "moderate"}
        assert condition(state, tank, ctx) is True

    def test_long_absence_none(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        absence = [
            (ev, cond) for ev, cond in events if ev.name == "long_absence"
        ][0]
        _, condition = absence
        ctx = {"absence_severity": "none"}
        assert condition(state, tank, ctx) is False

    def test_dirty_tank_fires_when_dirty(
        self,
        state: CreatureState,
    ) -> None:
        dirty_tank = TankEnvironment(cleanliness=0.2)
        events = _build_default_events()
        dirt = [
            (ev, cond) for ev, cond in events if ev.name == "dirty_tank"
        ][0]
        _, condition = dirt
        assert condition(state, dirty_tank, {}) is True

    def test_dirty_tank_no_fire_when_clean(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        dirt = [
            (ev, cond) for ev, cond in events if ev.name == "dirty_tank"
        ][0]
        _, condition = dirt
        assert condition(state, tank, {}) is False

    def test_temperature_warning_fires_hot(
        self,
        state: CreatureState,
    ) -> None:
        hot_tank = TankEnvironment(temperature=35.0)
        events = _build_default_events()
        temp = [
            (ev, cond) for ev, cond in events if ev.name == "temperature_warning"
        ][0]
        _, condition = temp
        assert condition(state, hot_tank, {}) is True

    def test_temperature_warning_fires_cold(
        self,
        state: CreatureState,
    ) -> None:
        cold_tank = TankEnvironment(temperature=15.0)
        events = _build_default_events()
        temp = [
            (ev, cond) for ev, cond in events if ev.name == "temperature_warning"
        ][0]
        _, condition = temp
        assert condition(state, cold_tank, {}) is True

    def test_temperature_warning_no_fire_normal(
        self,
        state: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        temp = [
            (ev, cond) for ev, cond in events if ev.name == "temperature_warning"
        ][0]
        _, condition = temp
        assert condition(state, tank, {}) is False

    def test_gillman_cannibalism_condition(
        self,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        cann = [
            (ev, cond) for ev, cond in events if ev.name == "gillman_cannibalism"
        ][0]
        _, condition = cann

        gillman_state = CreatureState(
            stage=CreatureStage.GILLMAN,
            interaction_count=20,
        )
        assert condition(gillman_state, tank, {}) is True

        mushroomer_state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=20,
        )
        assert condition(mushroomer_state, tank, {}) is False

    def test_podfish_mating_condition(
        self,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        mate = [
            (ev, cond) for ev, cond in events if ev.name == "podfish_mating"
        ][0]
        _, condition = mate

        ready = CreatureState(
            stage=CreatureStage.PODFISH,
            trust_level=0.5,
            interaction_count=35,
        )
        assert condition(ready, tank, {}) is True

        not_ready = CreatureState(
            stage=CreatureStage.PODFISH,
            trust_level=0.1,
            interaction_count=5,
        )
        assert condition(not_ready, tank, {}) is False

    def test_tank_drain_prompt_condition(self) -> None:
        events = _build_default_events()
        drain = [
            (ev, cond) for ev, cond in events if ev.name == "tank_drain_prompt"
        ][0]
        _, condition = drain

        podfish = CreatureState(
            stage=CreatureStage.PODFISH,
            trust_level=0.6,
        )
        aquarium = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)
        assert condition(podfish, aquarium, {}) is True

        terrarium = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)
        assert condition(podfish, terrarium, {}) is False

    def test_milestone_10_condition(
        self,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        m10 = [
            (ev, cond) for ev, cond in events if ev.name == "milestone_10"
        ][0]
        _, condition = m10

        enough = CreatureState(interaction_count=10)
        assert condition(enough, tank, {}) is True

        not_enough = CreatureState(interaction_count=9)
        assert condition(not_enough, tank, {}) is False

    def test_evolution_ready_mushroomer(
        self,
        tank: TankEnvironment,
    ) -> None:
        events = _build_default_events()
        evo = [
            (ev, cond) for ev, cond in events
            if ev.name == "evolution_ready_mushroomer"
        ][0]
        _, condition = evo

        mushroomer = CreatureState(stage=CreatureStage.MUSHROOMER)
        assert condition(mushroomer, tank, {}) is True

        gillman = CreatureState(stage=CreatureStage.GILLMAN)
        assert condition(gillman, tank, {}) is False


# ---------------------------------------------------------------------------
# Integration: default system with real conditions
# ---------------------------------------------------------------------------

class TestDefaultSystemIntegration:
    def test_new_mushroomer_events(
        self,
        default_system: EventSystem,
        time_context: dict,
    ) -> None:
        """A brand new mushroomer at 2pm should get evolution_ready event."""
        state = CreatureState(stage=CreatureStage.MUSHROOMER)
        tank = TankEnvironment()
        fired = default_system.check_events(state, tank, time_context)
        names = [e.name for e in fired]
        assert "evolution_ready_mushroomer" in names

    def test_dirty_cold_tank_events(
        self,
        default_system: EventSystem,
        time_context: dict,
    ) -> None:
        """Dirty + cold tank triggers environmental events."""
        state = CreatureState()
        tank = TankEnvironment(cleanliness=0.1, temperature=10.0)
        fired = default_system.check_events(state, tank, time_context)
        names = [e.name for e in fired]
        assert "dirty_tank" in names
        assert "temperature_warning" in names

    def test_late_night_gillman(
        self,
        default_system: EventSystem,
    ) -> None:
        """Gillman at 3am with 20 interactions triggers multiple events."""
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            interaction_count=20,
        )
        tank = TankEnvironment()
        ctx = {
            "hour": 3,
            "is_weekend": False,
            "absence_severity": "none",
        }
        fired = default_system.check_events(state, tank, ctx)
        names = [e.name for e in fired]
        assert "late_night_visit" in names
        assert "gillman_cannibalism" in names
        assert "milestone_10" in names

    def test_one_shot_events_dont_refire(
        self,
        default_system: EventSystem,
        time_context: dict,
    ) -> None:
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=15,
        )
        tank = TankEnvironment()
        fired_1 = default_system.check_events(state, tank, time_context)
        one_shot_names = [e.name for e in fired_1 if e.one_shot]
        assert len(one_shot_names) > 0

        fired_2 = default_system.check_events(state, tank, time_context)
        refired = [e.name for e in fired_2 if e.name in one_shot_names]
        assert refired == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_multiple_effects_combined(
        self,
        system: EventSystem,
    ) -> None:
        state = CreatureState(trust_level=0.5, hunger=0.3, health=0.8)
        tank = TankEnvironment(cleanliness=0.5)
        event = GameEvent(
            EventType.MILESTONE, "multi", "msg",
            effects=EventEffect(
                trust_change=0.1,
                hunger_change=0.1,
                health_change=-0.1,
                tank_changes={"cleanliness": 0.2},
            ),
        )
        system.apply_effects(event, state, tank)
        assert state.trust_level == pytest.approx(0.6)
        assert state.hunger == pytest.approx(0.4)
        assert state.health == pytest.approx(0.7)
        assert tank.cleanliness == 0.2

    def test_get_registered_event_names_order(
        self,
        system: EventSystem,
    ) -> None:
        system.register_event(
            GameEvent(EventType.MILESTONE, "a", "msg"), _always_true,
        )
        system.register_event(
            GameEvent(EventType.MILESTONE, "b", "msg"), _always_true,
        )
        assert system.get_registered_event_names() == ["a", "b"]

    def test_concurrent_one_shot_and_cooldown_events(
        self,
        state: CreatureState,
        tank: TankEnvironment,
        time_context: dict,
        now: datetime,
    ) -> None:
        time = [now]
        system = EventSystem(
            include_defaults=False,
            now_func=lambda: time[0],
        )

        one_shot = GameEvent(
            EventType.MILESTONE, "os", "msg", one_shot=True, priority=0.9,
        )
        cooldown = GameEvent(
            EventType.MILESTONE, "cd", "msg",
            cooldown_seconds=60.0, priority=0.5,
        )
        system.register_event(one_shot, _always_true)
        system.register_event(cooldown, _always_true)

        fired = system.check_events(state, tank, time_context)
        assert len(fired) == 2
        assert fired[0].name == "os"  # Higher priority first

        # Next check: one-shot gone, cooldown blocked
        fired2 = system.check_events(state, tank, time_context)
        assert fired2 == []

        # Advance past cooldown
        time[0] = now + timedelta(seconds=61)
        fired3 = system.check_events(state, tank, time_context)
        assert len(fired3) == 1
        assert fired3[0].name == "cd"

    def test_to_dict_sorted_one_shots(self, system: EventSystem) -> None:
        system._fired_one_shots = {"z", "a", "m"}
        d = system.to_dict()
        assert d["fired_one_shots"] == ["a", "m", "z"]
