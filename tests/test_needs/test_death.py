"""Tests for death and revival mechanics (US-031)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.death import (
    DEATH_MESSAGES,
    DeathCause,
    DeathEngine,
    DeathRecord,
)
from seaman_brain.needs.system import CreatureNeeds
from seaman_brain.types import CreatureStage

# ── DeathCause enum ──────────────────────────────────────────────

class TestDeathCause:
    """Tests for DeathCause enum."""

    def test_all_causes_defined(self) -> None:
        causes = list(DeathCause)
        assert len(causes) == 6

    def test_cause_values(self) -> None:
        assert DeathCause.STARVATION.value == "starvation"
        assert DeathCause.SUFFOCATION.value == "suffocation"
        assert DeathCause.HYPOTHERMIA.value == "hypothermia"
        assert DeathCause.HYPERTHERMIA.value == "hyperthermia"
        assert DeathCause.ILLNESS.value == "illness"
        assert DeathCause.OLD_AGE.value == "old_age"

    def test_from_string(self) -> None:
        assert DeathCause("starvation") == DeathCause.STARVATION


# ── DeathRecord ──────────────────────────────────────────────────

class TestDeathRecord:
    """Tests for DeathRecord dataclass."""

    def test_creation(self) -> None:
        now = datetime.now(UTC)
        record = DeathRecord(
            cause=DeathCause.STARVATION,
            message="Starved.",
            timestamp=now,
            creature_stage=CreatureStage.GILLMAN,
            creature_age=5000.0,
            interaction_count=42,
        )
        assert record.cause == DeathCause.STARVATION
        assert record.creature_stage == CreatureStage.GILLMAN
        assert record.creature_age == 5000.0

    def test_to_dict(self) -> None:
        now = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)
        record = DeathRecord(
            cause=DeathCause.SUFFOCATION,
            message="No air.",
            timestamp=now,
            creature_stage=CreatureStage.PODFISH,
            creature_age=1000.0,
            interaction_count=10,
        )
        d = record.to_dict()
        assert d["cause"] == "suffocation"
        assert d["creature_stage"] == "podfish"
        assert d["creature_age"] == 1000.0
        assert d["interaction_count"] == 10
        assert "timestamp" in d

    def test_from_dict_roundtrip(self) -> None:
        now = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)
        original = DeathRecord(
            cause=DeathCause.HYPOTHERMIA,
            message="Frozen.",
            timestamp=now,
            creature_stage=CreatureStage.TADMAN,
            creature_age=3000.0,
            interaction_count=25,
        )
        restored = DeathRecord.from_dict(original.to_dict())
        assert restored.cause == original.cause
        assert restored.message == original.message
        assert restored.creature_stage == original.creature_stage
        assert restored.creature_age == original.creature_age

    def test_from_dict_defaults(self) -> None:
        d = {
            "cause": "illness",
            "timestamp": datetime.now(UTC).isoformat(),
        }
        record = DeathRecord.from_dict(d)
        assert record.cause == DeathCause.ILLNESS
        assert record.message == ""
        assert record.creature_age == 0.0


# ── DeathEngine init ─────────────────────────────────────────────

class TestDeathEngineInit:
    """Tests for DeathEngine initialization."""

    def test_default_init(self) -> None:
        engine = DeathEngine()
        assert engine._starvation_start is None
        assert engine._hypothermia_start is None
        assert engine._hyperthermia_start is None

    def test_custom_config(self) -> None:
        cfg = NeedsConfig(starvation_time_hours=2.0)
        env_cfg = EnvironmentConfig(lethal_temp_min=5.0, lethal_temp_max=40.0)
        engine = DeathEngine(needs_config=cfg, env_config=env_cfg)
        assert engine._needs_config.starvation_time_hours == 2.0
        assert engine._env_config.lethal_temp_min == 5.0

    def test_injectable_now_func(self) -> None:
        fixed = datetime(2026, 1, 1, tzinfo=UTC)
        engine = DeathEngine(now_func=lambda: fixed)
        assert engine._now() == fixed


# ── Suffocation (immediate) ──────────────────────────────────────

class TestSuffocationDeath:
    """Tests for suffocation death (oxygen < 0.1)."""

    def test_suffocation_low_oxygen(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=1.0, hunger=0.0)
        tank = TankEnvironment(oxygen_level=0.05)
        assert engine.check_death(state, needs, tank) == DeathCause.SUFFOCATION

    def test_no_suffocation_at_threshold(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=1.0, hunger=0.0)
        tank = TankEnvironment(oxygen_level=0.1)
        assert engine.check_death(state, needs, tank) is None

    def test_suffocation_overrides_other_causes(self) -> None:
        """Suffocation is checked first — it overrides starvation/illness."""
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=0.0, hunger=1.0)
        tank = TankEnvironment(oxygen_level=0.05)
        assert engine.check_death(state, needs, tank) == DeathCause.SUFFOCATION


# ── Starvation (duration-based) ──────────────────────────────────

class TestStarvationDeath:
    """Tests for starvation death (hunger=1.0 for starvation_time_hours)."""

    def test_starvation_after_grace_period(self) -> None:
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(hunger=1.0, health=0.5)
        tank = TankEnvironment()

        # First check starts the timer
        result = engine.check_death(state, needs, tank)
        assert result is None  # still in grace period

        # Advance past 1 hour
        time[0] += timedelta(hours=1, seconds=1)
        result = engine.check_death(state, needs, tank)
        assert result == DeathCause.STARVATION

    def test_starvation_resets_if_fed(self) -> None:
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(now_func=lambda: time[0])
        state = CreatureState()
        needs_starving = CreatureNeeds(hunger=1.0, health=0.5)
        needs_fed = CreatureNeeds(hunger=0.5, health=0.5)
        tank = TankEnvironment()

        # Start starvation timer
        engine.check_death(state, needs_starving, tank)
        time[0] += timedelta(minutes=30)

        # Feed the creature — resets timer
        engine.check_death(state, needs_fed, tank)
        assert engine._starvation_start is None

        # Re-starve — should need another full hour
        time[0] += timedelta(minutes=10)
        engine.check_death(state, needs_starving, tank)
        time[0] += timedelta(minutes=50)
        assert engine.check_death(state, needs_starving, tank) is None

        time[0] += timedelta(minutes=11)
        assert engine.check_death(state, needs_starving, tank) == DeathCause.STARVATION

    def test_starvation_custom_hours(self) -> None:
        cfg = NeedsConfig(starvation_time_hours=0.5)  # 30 minutes
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(needs_config=cfg, now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(hunger=1.0, health=0.5)
        tank = TankEnvironment()

        engine.check_death(state, needs, tank)
        time[0] += timedelta(minutes=31)
        assert engine.check_death(state, needs, tank) == DeathCause.STARVATION

    def test_no_starvation_below_max_hunger(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(hunger=0.99, health=0.5)
        tank = TankEnvironment()
        assert engine.check_death(state, needs, tank) is None


# ── Temperature death (duration-based) ───────────────────────────

class TestTemperatureDeath:
    """Tests for hypothermia/hyperthermia death with grace period."""

    def test_hypothermia_after_grace_period(self) -> None:
        env_cfg = EnvironmentConfig(lethal_temp_min=10.0)
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(env_config=env_cfg, now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(health=0.5)
        tank = TankEnvironment(temperature=5.0, oxygen_level=0.5)

        # First check starts timer
        assert engine.check_death(state, needs, tank) is None

        # Just before 30 minutes
        time[0] += timedelta(minutes=29, seconds=59)
        assert engine.check_death(state, needs, tank) is None

        # After 30 minutes
        time[0] += timedelta(seconds=2)
        assert engine.check_death(state, needs, tank) == DeathCause.HYPOTHERMIA

    def test_hyperthermia_after_grace_period(self) -> None:
        env_cfg = EnvironmentConfig(lethal_temp_max=35.0)
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(env_config=env_cfg, now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(health=0.5)
        tank = TankEnvironment(temperature=40.0, oxygen_level=0.5)

        engine.check_death(state, needs, tank)
        time[0] += timedelta(minutes=31)
        assert engine.check_death(state, needs, tank) == DeathCause.HYPERTHERMIA

    def test_temperature_resets_when_corrected(self) -> None:
        env_cfg = EnvironmentConfig(lethal_temp_min=10.0)
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(env_config=env_cfg, now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(health=0.5)
        cold_tank = TankEnvironment(temperature=5.0, oxygen_level=0.5)
        warm_tank = TankEnvironment(temperature=20.0, oxygen_level=0.5)

        # Start hypothermia timer
        engine.check_death(state, needs, cold_tank)
        time[0] += timedelta(minutes=20)

        # Fix temperature — resets timer
        engine.check_death(state, needs, warm_tank)
        assert engine._hypothermia_start is None

    def test_no_death_at_lethal_boundary(self) -> None:
        """At exactly lethal_temp_min, creature is alive (not below it)."""
        env_cfg = EnvironmentConfig(lethal_temp_min=10.0)
        engine = DeathEngine(env_config=env_cfg)
        state = CreatureState()
        needs = CreatureNeeds(health=0.5)
        tank = TankEnvironment(temperature=10.0, oxygen_level=0.5)
        assert engine.check_death(state, needs, tank) is None


# ── Illness death (immediate) ────────────────────────────────────

class TestIllnessDeath:
    """Tests for illness death (health=0)."""

    def test_illness_at_zero_health(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=0.0)
        tank = TankEnvironment()
        assert engine.check_death(state, needs, tank) == DeathCause.ILLNESS

    def test_no_illness_above_zero(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=0.01)
        tank = TankEnvironment()
        assert engine.check_death(state, needs, tank) is None


# ── Alive (no death) ────────────────────────────────────────────

class TestNoDeathCondition:
    """Tests for healthy creatures that should NOT die."""

    def test_healthy_creature_survives(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(hunger=0.5, health=0.8)
        tank = TankEnvironment(temperature=24.0, oxygen_level=0.8)
        assert engine.check_death(state, needs, tank) is None

    def test_hungry_but_not_max(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(hunger=0.95, health=0.5)
        tank = TankEnvironment()
        assert engine.check_death(state, needs, tank) is None


# ── Revival (on_death) ──────────────────────────────────────────

class TestOnDeath:
    """Tests for death handling and revival to new egg."""

    def test_revival_creates_new_mushroomer(self) -> None:
        now = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)
        engine = DeathEngine(now_func=lambda: now)
        old_state = CreatureState(
            stage=CreatureStage.FROGMAN,
            age=50000.0,
            interaction_count=200,
            trust_level=0.8,
            hunger=1.0,
            health=0.0,
        )

        new_state, record = engine.on_death(DeathCause.STARVATION, old_state)

        # New egg state
        assert new_state.stage == CreatureStage.MUSHROOMER
        assert new_state.age == 0.0
        assert new_state.interaction_count == 0
        assert new_state.trust_level == 0.0
        assert new_state.hunger == 0.0
        assert new_state.health == 1.0
        assert new_state.birth_time == now

    def test_death_record_captures_old_state(self) -> None:
        now = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)
        engine = DeathEngine(now_func=lambda: now)
        old_state = CreatureState(
            stage=CreatureStage.GILLMAN,
            age=10000.0,
            interaction_count=50,
        )

        _, record = engine.on_death(DeathCause.SUFFOCATION, old_state)

        assert record.cause == DeathCause.SUFFOCATION
        assert record.creature_stage == CreatureStage.GILLMAN
        assert record.creature_age == 10000.0
        assert record.interaction_count == 50
        assert record.timestamp == now

    def test_sardonic_death_message(self) -> None:
        engine = DeathEngine()
        _, record = engine.on_death(DeathCause.STARVATION, CreatureState())
        assert "starved" in record.message.lower() or "pellet" in record.message.lower()

    def test_all_causes_have_messages(self) -> None:
        for cause in DeathCause:
            assert cause in DEATH_MESSAGES

    def test_on_death_resets_duration_trackers(self) -> None:
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(now_func=lambda: time[0])

        # Start some trackers
        engine._starvation_start = time[0]
        engine._hypothermia_start = time[0]
        engine._hyperthermia_start = time[0]

        engine.on_death(DeathCause.ILLNESS, CreatureState())

        assert engine._starvation_start is None
        assert engine._hypothermia_start is None
        assert engine._hyperthermia_start is None


# ── Death record persistence ────────────────────────────────────

class TestDeathRecordPersistence:
    """Tests for death record saving to disk."""

    def test_saves_death_record_to_dir(self, tmp_path: str) -> None:
        now = datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)
        engine = DeathEngine(death_log_dir=tmp_path, now_func=lambda: now)
        state = CreatureState(stage=CreatureStage.PODFISH, age=5000.0)

        engine.on_death(DeathCause.HYPOTHERMIA, state)

        import json
        from pathlib import Path
        files = list(Path(tmp_path).glob("death_*.json"))
        assert len(files) == 1

        data = json.loads(files[0].read_text())
        assert data["cause"] == "hypothermia"
        assert data["creature_stage"] == "podfish"

    def test_no_save_without_log_dir(self) -> None:
        engine = DeathEngine()  # no death_log_dir
        # Should not raise
        engine.on_death(DeathCause.ILLNESS, CreatureState())

    def test_handles_save_error_gracefully(self, tmp_path: str) -> None:
        from pathlib import Path
        from unittest.mock import patch
        engine = DeathEngine(death_log_dir=tmp_path)

        with patch.object(Path, "write_text", side_effect=OSError("disk full")):
            # Should not raise — error is logged
            new_state, record = engine.on_death(DeathCause.STARVATION, CreatureState())
            assert new_state.stage == CreatureStage.MUSHROOMER
            assert record.cause == DeathCause.STARVATION


# ── Warnings ─────────────────────────────────────────────────────

class TestDeathWarnings:
    """Tests for pre-death warning system."""

    def test_no_warnings_when_healthy(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(hunger=0.3, health=0.8)
        tank = TankEnvironment(oxygen_level=0.8)
        assert engine.get_warnings(state, needs, tank) == []

    def test_hungry_warning(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(hunger=0.95, health=0.5)
        tank = TankEnvironment()
        warnings = engine.get_warnings(state, needs, tank)
        assert any("hungry" in w.lower() for w in warnings)

    def test_oxygen_warning(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=0.5)
        tank = TankEnvironment(oxygen_level=0.15)
        warnings = engine.get_warnings(state, needs, tank)
        assert any("oxygen" in w.lower() for w in warnings)

    def test_health_critical_warning(self) -> None:
        engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=0.05)
        tank = TankEnvironment()
        warnings = engine.get_warnings(state, needs, tank)
        assert any("health" in w.lower() for w in warnings)

    def test_starvation_countdown_warning(self) -> None:
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(hunger=1.0, health=0.5)
        tank = TankEnvironment()

        # Start starvation timer via check_death
        engine.check_death(state, needs, tank)

        # Advance to 40 minutes (within 30-min warning window of 1hr limit)
        time[0] += timedelta(minutes=40)
        warnings = engine.get_warnings(state, needs, tank)
        assert any("starve" in w.lower() for w in warnings)

    def test_temperature_warning_with_countdown(self) -> None:
        env_cfg = EnvironmentConfig(lethal_temp_min=10.0)
        time = [datetime(2026, 1, 1, tzinfo=UTC)]
        engine = DeathEngine(env_config=env_cfg, now_func=lambda: time[0])
        state = CreatureState()
        needs = CreatureNeeds(health=0.5)
        tank = TankEnvironment(temperature=5.0, oxygen_level=0.5)

        # Start hypothermia timer
        engine.check_death(state, needs, tank)

        # Advance to 20 minutes (within 15-min warning window)
        time[0] += timedelta(minutes=20)
        warnings = engine.get_warnings(state, needs, tank)
        assert any("hypothermia" in w.lower() for w in warnings)


# ── Death message helper ─────────────────────────────────────────

class TestGetDeathMessage:
    """Tests for the get_death_message helper."""

    def test_known_cause(self) -> None:
        engine = DeathEngine()
        msg = engine.get_death_message(DeathCause.SUFFOCATION)
        assert "oxygen" in msg.lower() or "stale" in msg.lower()

    def test_all_messages_non_empty(self) -> None:
        engine = DeathEngine()
        for cause in DeathCause:
            msg = engine.get_death_message(cause)
            assert len(msg) > 10
