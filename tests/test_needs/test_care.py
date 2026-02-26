"""Tests for tank care mechanics (US-030)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.needs.care import (
    AERATOR_COOLDOWN_SECONDS,
    AERATOR_OXYGEN_BOOST,
    CLEANING_AMOUNT,
    CLEANING_DURATION_SECONDS,
    SPRINKLER_MOISTURE_BOOST,
    STAGE_OPTIMAL_TEMPS,
    CareAction,
    CareResult,
    TankCareEngine,
)
from seaman_brain.types import CreatureStage

# --- Fixtures ---


@pytest.fixture
def env_config() -> EnvironmentConfig:
    """Standard environment config for testing."""
    return EnvironmentConfig()


@pytest.fixture
def needs_config() -> NeedsConfig:
    """Standard needs config for testing."""
    return NeedsConfig()


@pytest.fixture
def tank() -> TankEnvironment:
    """Default tank in good condition."""
    return TankEnvironment(
        temperature=24.0,
        cleanliness=1.0,
        oxygen_level=1.0,
        water_level=1.0,
        environment_type=EnvironmentType.AQUARIUM,
    )


@pytest.fixture
def creature() -> CreatureState:
    """Default creature at Gillman stage."""
    return CreatureState(stage=CreatureStage.GILLMAN)


@pytest.fixture
def now() -> datetime:
    """Fixed time for deterministic tests."""
    return datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def engine(env_config: EnvironmentConfig, needs_config: NeedsConfig,
           now: datetime) -> TankCareEngine:
    """TankCareEngine with fixed time."""
    return TankCareEngine(
        env_config=env_config,
        needs_config=needs_config,
        now_func=lambda: now,
    )


# --- CareResult Tests ---


class TestCareResult:
    """Tests for CareResult dataclass."""

    def test_create_success_result(self) -> None:
        result = CareResult(
            success=True,
            action=CareAction.CLEAN_TANK,
            message="Tank cleaned!",
            warnings=[],
        )
        assert result.success is True
        assert result.action == CareAction.CLEAN_TANK
        assert result.message == "Tank cleaned!"
        assert result.warnings == []

    def test_create_failure_result(self) -> None:
        result = CareResult(
            success=False,
            action=CareAction.SPRINKLE,
            message="Can't sprinkle in aquarium",
            warnings=["Low oxygen"],
        )
        assert result.success is False
        assert result.warnings == ["Low oxygen"]


# --- CareAction Enum Tests ---


class TestCareAction:
    """Tests for CareAction enum."""

    def test_all_actions_defined(self) -> None:
        assert len(CareAction) == 6
        assert CareAction.ADJUST_TEMPERATURE.value == "adjust_temperature"
        assert CareAction.CLEAN_TANK.value == "clean_tank"
        assert CareAction.AERATE.value == "aerate"
        assert CareAction.SPRINKLE.value == "sprinkle"
        assert CareAction.DRAIN.value == "drain"
        assert CareAction.FILL.value == "fill"


# --- TankCareEngine Init Tests ---


class TestTankCareEngineInit:
    """Tests for TankCareEngine initialization."""

    def test_default_init(self) -> None:
        engine = TankCareEngine()
        assert engine.env_config is not None
        assert engine.needs_config is not None

    def test_custom_config(self, env_config: EnvironmentConfig,
                           needs_config: NeedsConfig) -> None:
        engine = TankCareEngine(env_config=env_config, needs_config=needs_config)
        assert engine.env_config is env_config
        assert engine.needs_config is needs_config

    def test_injectable_now_func(self) -> None:
        fixed = datetime(2026, 1, 1, tzinfo=UTC)
        engine = TankCareEngine(now_func=lambda: fixed)
        assert not engine.is_cleaning_on_cooldown()


# --- Temperature Adjustment Tests ---


class TestAdjustTemperature:
    """Tests for temperature adjustment."""

    def test_warm_up_within_optimal(self, engine: TankCareEngine, tank: TankEnvironment,
                                    creature: CreatureState) -> None:
        result = engine.adjust_temperature(tank, 2.0, creature)
        assert result.success is True
        assert result.action == CareAction.ADJUST_TEMPERATURE
        assert tank.temperature == 26.0
        assert "optimal" in result.message.lower()

    def test_cool_down_within_optimal(self, engine: TankCareEngine, tank: TankEnvironment,
                                      creature: CreatureState) -> None:
        result = engine.adjust_temperature(tank, -2.0, creature)
        assert result.success is True
        assert tank.temperature == 22.0
        assert "optimal" in result.message.lower()

    def test_temperature_above_stage_optimal(self, engine: TankCareEngine,
                                              tank: TankEnvironment) -> None:
        creature = CreatureState(stage=CreatureStage.MUSHROOMER)
        tank.temperature = 25.0
        result = engine.adjust_temperature(tank, 3.0, creature)
        assert result.success is True
        assert tank.temperature == 28.0
        # Mushroomer optimal is 22-26, so 28 is above optimal
        assert "prefers" in result.message.lower()

    def test_temperature_below_stage_optimal(self, engine: TankCareEngine,
                                              tank: TankEnvironment) -> None:
        creature = CreatureState(stage=CreatureStage.FROGMAN)
        tank.temperature = 24.0
        result = engine.adjust_temperature(tank, -3.0, creature)
        assert result.success is True
        assert tank.temperature == 21.0
        # Frogman optimal is 22-32
        assert "prefers" in result.message.lower()

    def test_temperature_dangerously_low(self, engine: TankCareEngine,
                                         tank: TankEnvironment,
                                         creature: CreatureState) -> None:
        tank.temperature = 12.0
        result = engine.adjust_temperature(tank, -5.0, creature)
        assert result.success is True
        assert tank.temperature < 10.0
        assert "DANGEROUSLY COLD" in result.message

    def test_temperature_dangerously_high(self, engine: TankCareEngine,
                                          tank: TankEnvironment,
                                          creature: CreatureState) -> None:
        tank.temperature = 36.0
        result = engine.adjust_temperature(tank, 5.0, creature)
        assert result.success is True
        assert tank.temperature > 38.0
        assert "DANGEROUSLY HOT" in result.message

    def test_temperature_clamped_to_bounds(self, engine: TankCareEngine,
                                            tank: TankEnvironment,
                                            creature: CreatureState) -> None:
        result = engine.adjust_temperature(tank, 100.0, creature)
        assert result.success is True
        # Max is lethal_max + 5 = 43
        assert tank.temperature <= 43.0

    def test_zero_delta(self, engine: TankCareEngine, tank: TankEnvironment,
                        creature: CreatureState) -> None:
        old = tank.temperature
        result = engine.adjust_temperature(tank, 0.0, creature)
        assert result.success is True
        assert tank.temperature == old

    def test_warnings_included(self, engine: TankCareEngine, tank: TankEnvironment,
                               creature: CreatureState) -> None:
        tank.cleanliness = 0.1
        result = engine.adjust_temperature(tank, 1.0, creature)
        assert result.success is True
        assert len(result.warnings) > 0


# --- Clean Tank Tests ---


class TestCleanTank:
    """Tests for tank cleaning mechanics."""

    def test_clean_dirty_tank(self, engine: TankCareEngine,
                              tank: TankEnvironment) -> None:
        tank.cleanliness = 0.3
        result = engine.clean_tank(tank)
        assert result.success is True
        assert result.action == CareAction.CLEAN_TANK
        assert tank.cleanliness == pytest.approx(0.3 + CLEANING_AMOUNT)

    def test_clean_restores_partial(self, engine: TankCareEngine,
                                    tank: TankEnvironment) -> None:
        tank.cleanliness = 0.0
        result = engine.clean_tank(tank)
        assert result.success is True
        assert tank.cleanliness == pytest.approx(CLEANING_AMOUNT)
        assert "more cleaning" in result.message.lower()

    def test_clean_caps_at_one(self, engine: TankCareEngine,
                               tank: TankEnvironment) -> None:
        tank.cleanliness = 0.9
        result = engine.clean_tank(tank)
        assert result.success is True
        assert tank.cleanliness == 1.0
        assert "spotless" in result.message.lower()

    def test_clean_cooldown_enforced(self, now: datetime) -> None:
        times = [now, now + timedelta(seconds=2)]
        idx = [0]

        def advancing_clock() -> datetime:
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        engine = TankCareEngine(now_func=advancing_clock)
        tank = TankEnvironment(cleanliness=0.5)

        result1 = engine.clean_tank(tank)
        assert result1.success is True

        result2 = engine.clean_tank(tank)
        assert result2.success is False
        assert "wait" in result2.message.lower()

    def test_clean_after_cooldown_expires(self, now: datetime) -> None:
        times = [now, now + timedelta(seconds=CLEANING_DURATION_SECONDS + 1)]
        idx = [0]

        def advancing_clock() -> datetime:
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        engine = TankCareEngine(now_func=advancing_clock)
        tank = TankEnvironment(cleanliness=0.3)

        result1 = engine.clean_tank(tank)
        assert result1.success is True

        result2 = engine.clean_tank(tank)
        assert result2.success is True

    def test_clean_medium_cleanliness_message(self, engine: TankCareEngine,
                                              tank: TankEnvironment) -> None:
        tank.cleanliness = 0.5
        result = engine.clean_tank(tank)
        assert result.success is True
        assert "looking good" in result.message.lower()

    def test_clean_very_dirty_needs_more(self, engine: TankCareEngine,
                                         tank: TankEnvironment) -> None:
        tank.cleanliness = 0.1
        result = engine.clean_tank(tank)
        assert result.success is True
        assert tank.cleanliness == pytest.approx(0.1 + CLEANING_AMOUNT)


# --- Sprinkle Tests ---


class TestSprinkle:
    """Tests for terrarium sprinkler mechanics."""

    def test_sprinkle_in_terrarium(self, engine: TankCareEngine,
                                   creature: CreatureState) -> None:
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            oxygen_level=0.5,
        )
        result = engine.sprinkle(tank, creature)
        assert result.success is True
        assert result.action == CareAction.SPRINKLE
        assert tank.oxygen_level == pytest.approx(0.5 + SPRINKLER_MOISTURE_BOOST)

    def test_sprinkle_boosts_cleanliness(self, engine: TankCareEngine,
                                         creature: CreatureState) -> None:
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            cleanliness=0.5,
            oxygen_level=0.5,
        )
        engine.sprinkle(tank, creature)
        assert tank.cleanliness == pytest.approx(0.5 + SPRINKLER_MOISTURE_BOOST * 0.5)

    def test_sprinkle_fails_in_aquarium(self, engine: TankCareEngine,
                                        tank: TankEnvironment,
                                        creature: CreatureState) -> None:
        result = engine.sprinkle(tank, creature)
        assert result.success is False
        assert "terrarium" in result.message.lower()

    def test_sprinkle_frogman_message(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.FROGMAN)
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            oxygen_level=0.5,
        )
        result = engine.sprinkle(tank, creature)
        assert result.success is True
        assert "enjoys the moisture" in result.message.lower()

    def test_sprinkle_tadman_message(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.TADMAN)
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            oxygen_level=0.5,
        )
        result = engine.sprinkle(tank, creature)
        assert result.success is True
        assert "enjoys the moisture" in result.message.lower()

    def test_sprinkle_capped_at_one(self, engine: TankCareEngine,
                                     creature: CreatureState) -> None:
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            oxygen_level=0.9,
            cleanliness=0.95,
        )
        engine.sprinkle(tank, creature)
        assert tank.oxygen_level <= 1.0
        assert tank.cleanliness <= 1.0


# --- Aerate Tests ---


class TestAerate:
    """Tests for aquarium aerator mechanics."""

    def test_aerate_in_aquarium(self, engine: TankCareEngine,
                                tank: TankEnvironment) -> None:
        tank.oxygen_level = 0.5
        result = engine.aerate_tank(tank)
        assert result.success is True
        assert result.action == CareAction.AERATE
        assert tank.oxygen_level == pytest.approx(0.5 + AERATOR_OXYGEN_BOOST)

    def test_aerate_fails_in_terrarium(self, engine: TankCareEngine) -> None:
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            oxygen_level=0.5,
        )
        result = engine.aerate_tank(tank)
        assert result.success is False
        assert "aquarium" in result.message.lower()

    def test_aerate_capped_at_one(self, engine: TankCareEngine,
                                   tank: TankEnvironment) -> None:
        tank.oxygen_level = 0.9
        engine.aerate_tank(tank)
        assert tank.oxygen_level <= 1.0

    def test_aerate_full_message(self, engine: TankCareEngine,
                                  tank: TankEnvironment) -> None:
        tank.oxygen_level = 0.9
        result = engine.aerate_tank(tank)
        assert result.success is True
        assert "fully restored" in result.message.lower()

    def test_aerate_low_oxygen_message(self, engine: TankCareEngine,
                                        tank: TankEnvironment) -> None:
        tank.oxygen_level = 0.1
        result = engine.aerate_tank(tank)
        assert result.success is True
        assert "still needs more" in result.message.lower()

    def test_aerate_cooldown(self, now: datetime) -> None:
        t = now
        engine = TankCareEngine(now_func=lambda: t)
        tank = TankEnvironment(oxygen_level=0.5)
        engine.aerate_tank(tank)

        # Immediate second attempt should fail
        result = engine.aerate_tank(tank)
        assert result.success is False
        assert "cycling" in result.message.lower()

    def test_aerate_cooldown_expires(self, now: datetime) -> None:
        t = now
        engine = TankCareEngine(now_func=lambda: t)
        tank = TankEnvironment(oxygen_level=0.5)
        engine.aerate_tank(tank)

        # Advance past cooldown
        t = now + timedelta(seconds=AERATOR_COOLDOWN_SECONDS + 1)
        engine._now_func = lambda: t
        result = engine.aerate_tank(tank)
        assert result.success is True


# --- Drain/Fill Tests ---


class TestDrainFill:
    """Tests for tank drain and fill operations."""

    def test_drain_aquarium(self, engine: TankCareEngine, tank: TankEnvironment,
                            creature: CreatureState) -> None:
        result = engine.drain_tank(tank, creature)
        assert result.success is True
        assert result.action == CareAction.DRAIN
        assert tank.environment_type == EnvironmentType.TERRARIUM
        assert tank.water_level == 0.0

    def test_drain_already_terrarium(self, engine: TankCareEngine,
                                     creature: CreatureState) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        result = engine.drain_tank(tank, creature)
        assert result.success is False
        assert "already drained" in result.message.lower()

    def test_fill_terrarium(self, engine: TankCareEngine,
                            creature: CreatureState) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        result = engine.fill_tank(tank, creature)
        assert result.success is True
        assert result.action == CareAction.FILL
        assert tank.environment_type == EnvironmentType.AQUARIUM
        assert tank.water_level == 1.0

    def test_fill_already_aquarium(self, engine: TankCareEngine, tank: TankEnvironment,
                                    creature: CreatureState) -> None:
        result = engine.fill_tank(tank, creature)
        assert result.success is False
        assert "already filled" in result.message.lower()


# --- Warning Generation Tests ---


class TestGetTankWarnings:
    """Tests for tank warning generation."""

    def test_no_warnings_healthy_tank(self, engine: TankCareEngine,
                                      tank: TankEnvironment,
                                      creature: CreatureState) -> None:
        warnings = engine.get_tank_warnings(tank, creature)
        assert warnings == []

    def test_low_oxygen_warning(self, engine: TankCareEngine,
                                creature: CreatureState) -> None:
        tank = TankEnvironment(oxygen_level=0.2)
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("oxygen" in w.lower() for w in warnings)

    def test_critical_oxygen_warning(self, engine: TankCareEngine,
                                     creature: CreatureState) -> None:
        tank = TankEnvironment(oxygen_level=0.05)
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("CRITICAL" in w and "Oxygen" in w for w in warnings)

    def test_dirty_tank_warning(self, engine: TankCareEngine,
                                creature: CreatureState) -> None:
        tank = TankEnvironment(cleanliness=0.1)
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("cleaning" in w.lower() or "clean" in w.lower() for w in warnings)

    def test_stage_specific_temp_warning(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.MUSHROOMER)
        # Mushroomer optimal is 22-26, set to 27 (suboptimal for mushroomer but not config)
        tank = TankEnvironment(temperature=27.0)
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("mushroomer" in w.lower() for w in warnings)

    def test_frogman_needs_terrarium(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.FROGMAN)
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("frogman" in w.lower() and "terrarium" in w.lower() for w in warnings)

    def test_frogman_low_oxygen_sprinkle(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.FROGMAN)
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
            oxygen_level=0.3,
        )
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("sprinkl" in w.lower() for w in warnings)

    def test_tadman_prefers_terrarium(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.TADMAN)
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("tadman" in w.lower() and "terrarium" in w.lower() for w in warnings)

    def test_aquatic_stage_in_terrarium(self, engine: TankCareEngine) -> None:
        creature = CreatureState(stage=CreatureStage.GILLMAN)
        tank = TankEnvironment(
            environment_type=EnvironmentType.TERRARIUM,
            water_level=0.0,
        )
        warnings = engine.get_tank_warnings(tank, creature)
        assert any("water" in w.lower() and "fill" in w.lower() for w in warnings)

    def test_warnings_without_creature_state(self, engine: TankCareEngine) -> None:
        tank = TankEnvironment(oxygen_level=0.05)
        warnings = engine.get_tank_warnings(tank)
        assert any("CRITICAL" in w for w in warnings)
        # No stage-specific warnings without creature_state
        assert not any("mushroomer" in w.lower() for w in warnings)


# --- Stage Optimal Range Tests ---


class TestStageOptimalRange:
    """Tests for stage-specific temperature ranges."""

    def test_all_stages_have_ranges(self) -> None:
        for stage in CreatureStage:
            assert stage in STAGE_OPTIMAL_TEMPS

    def test_mushroomer_range(self, engine: TankCareEngine) -> None:
        min_t, max_t = engine.get_stage_optimal_range(CreatureStage.MUSHROOMER)
        assert min_t == 22.0
        assert max_t == 26.0

    def test_frogman_range(self, engine: TankCareEngine) -> None:
        min_t, max_t = engine.get_stage_optimal_range(CreatureStage.FROGMAN)
        assert min_t == 22.0
        assert max_t == 32.0

    def test_ranges_widen_with_evolution(self) -> None:
        # Higher stages generally tolerate wider or warmer ranges
        mushroomer_range = STAGE_OPTIMAL_TEMPS[CreatureStage.MUSHROOMER]
        frogman_range = STAGE_OPTIMAL_TEMPS[CreatureStage.FROGMAN]
        mushroomer_width = mushroomer_range[1] - mushroomer_range[0]
        frogman_width = frogman_range[1] - frogman_range[0]
        assert frogman_width >= mushroomer_width


# --- Cleaning Cooldown Tests ---


class TestCleaningCooldown:
    """Tests for cleaning cooldown helpers."""

    def test_no_cooldown_initially(self, engine: TankCareEngine) -> None:
        assert not engine.is_cleaning_on_cooldown()
        assert engine.cleaning_cooldown_remaining() == 0.0

    def test_cooldown_after_clean(self, now: datetime) -> None:
        engine = TankCareEngine(now_func=lambda: now)
        tank = TankEnvironment(cleanliness=0.5)
        engine.clean_tank(tank)
        assert engine.is_cleaning_on_cooldown()
        assert engine.cleaning_cooldown_remaining() > 0.0

    def test_cooldown_expires(self, now: datetime) -> None:
        times = [now, now + timedelta(seconds=CLEANING_DURATION_SECONDS + 1)]
        idx = [0]

        def clock() -> datetime:
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        engine = TankCareEngine(now_func=clock)
        tank = TankEnvironment(cleanliness=0.5)
        engine.clean_tank(tank)

        assert not engine.is_cleaning_on_cooldown()
        assert engine.cleaning_cooldown_remaining() == 0.0


# --- Integration Tests ---


class TestCareIntegration:
    """Integration tests combining multiple care actions."""

    def test_full_maintenance_routine(self, now: datetime) -> None:
        """Test a complete tank maintenance cycle."""
        times = [now + timedelta(seconds=i * 10) for i in range(5)]
        idx = [0]

        def clock() -> datetime:
            t = times[min(idx[0], len(times) - 1)]
            idx[0] += 1
            return t

        engine = TankCareEngine(now_func=clock)
        creature = CreatureState(stage=CreatureStage.GILLMAN)
        tank = TankEnvironment(
            temperature=18.0,
            cleanliness=0.2,
            oxygen_level=0.5,
        )

        # 1. Adjust temperature
        result = engine.adjust_temperature(tank, 4.0, creature)
        assert result.success is True
        assert tank.temperature == 22.0

        # 2. Clean tank
        result = engine.clean_tank(tank)
        assert result.success is True
        assert tank.cleanliness > 0.2

        # 3. Clean again after cooldown
        result = engine.clean_tank(tank)
        assert result.success is True

    def test_drain_then_sprinkle_for_tadman(self, engine: TankCareEngine) -> None:
        """Test draining for Tadman evolution then sprinkling."""
        creature = CreatureState(stage=CreatureStage.TADMAN)
        tank = TankEnvironment(
            environment_type=EnvironmentType.AQUARIUM,
            oxygen_level=0.5,
        )

        # Drain for terrarium
        result = engine.drain_tank(tank, creature)
        assert result.success is True
        assert tank.environment_type == EnvironmentType.TERRARIUM

        # Sprinkle to maintain moisture
        result = engine.sprinkle(tank, creature)
        assert result.success is True
        assert tank.oxygen_level > 0.5

    def test_warnings_change_after_care(self, engine: TankCareEngine,
                                        creature: CreatureState) -> None:
        """Test that warnings update after care actions."""
        tank = TankEnvironment(cleanliness=0.1)
        warnings_before = engine.get_tank_warnings(tank, creature)
        assert len(warnings_before) > 0

        engine.clean_tank(tank)
        warnings_after = engine.get_tank_warnings(tank, creature)
        # Cleaning should reduce/eliminate cleanliness warnings
        clean_warnings_before = [w for w in warnings_before if "clean" in w.lower()]
        clean_warnings_after = [w for w in warnings_after if "clean" in w.lower()]
        assert len(clean_warnings_after) <= len(clean_warnings_before)

    def test_stage_mismatch_environment_warning(self, engine: TankCareEngine) -> None:
        """Test that aquatic creatures in terrarium get warnings."""
        for stage in (CreatureStage.MUSHROOMER, CreatureStage.GILLMAN, CreatureStage.PODFISH):
            creature = CreatureState(stage=stage)
            tank = TankEnvironment(
                environment_type=EnvironmentType.TERRARIUM,
                water_level=0.0,
            )
            warnings = engine.get_tank_warnings(tank, creature)
            assert any("water" in w.lower() or "fill" in w.lower() for w in warnings), (
                f"No water warning for {stage.value} in terrarium"
            )


# --- Aerator Cooldown Methods ---


class TestAeratorCooldownMethods:
    """Tests for is_aerating_on_cooldown() and aerating_cooldown_remaining()."""

    def test_no_cooldown_initially(self, engine: TankCareEngine) -> None:
        """No cooldown before any aeration."""
        assert engine.is_aerating_on_cooldown() is False
        assert engine.aerating_cooldown_remaining() == 0.0

    def test_cooldown_active_after_aerate(self, now: datetime) -> None:
        """Cooldown is active immediately after aerating."""
        current = now
        eng = TankCareEngine(now_func=lambda: current)
        tank = TankEnvironment(
            oxygen_level=0.5,
            environment_type=EnvironmentType.AQUARIUM,
        )
        eng.aerate_tank(tank)
        assert eng.is_aerating_on_cooldown() is True
        assert eng.aerating_cooldown_remaining() == pytest.approx(AERATOR_COOLDOWN_SECONDS)

    def test_cooldown_decreases_over_time(self, now: datetime) -> None:
        """Cooldown remaining decreases as time passes."""
        current = now
        eng = TankCareEngine(now_func=lambda: current)
        tank = TankEnvironment(
            oxygen_level=0.5,
            environment_type=EnvironmentType.AQUARIUM,
        )
        eng.aerate_tank(tank)

        # Advance time by 2 seconds
        current = now + timedelta(seconds=2)
        eng._now_func = lambda: current
        remaining = eng.aerating_cooldown_remaining()
        assert remaining == pytest.approx(AERATOR_COOLDOWN_SECONDS - 2.0)
        assert eng.is_aerating_on_cooldown() is True

    def test_cooldown_expired(self, now: datetime) -> None:
        """Cooldown is no longer active after full duration."""
        current = now
        eng = TankCareEngine(now_func=lambda: current)
        tank = TankEnvironment(
            oxygen_level=0.5,
            environment_type=EnvironmentType.AQUARIUM,
        )
        eng.aerate_tank(tank)

        # Advance time past cooldown
        current = now + timedelta(seconds=AERATOR_COOLDOWN_SECONDS + 1.0)
        eng._now_func = lambda: current
        assert eng.is_aerating_on_cooldown() is False
        assert eng.aerating_cooldown_remaining() == 0.0
