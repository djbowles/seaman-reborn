"""Tests for creature biological needs system (US-028)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.system import (
    STAGE_HUNGER_MULTIPLIER,
    CreatureNeeds,
    NeedsEngine,
    _clamp,
)
from seaman_brain.types import CreatureStage

# ── Helper fixtures ──────────────────────────────────────────────────


@pytest.fixture
def default_config() -> NeedsConfig:
    return NeedsConfig()


@pytest.fixture
def env_config() -> EnvironmentConfig:
    return EnvironmentConfig()


@pytest.fixture
def engine(default_config: NeedsConfig, env_config: EnvironmentConfig) -> NeedsEngine:
    return NeedsEngine(config=default_config, env_config=env_config)


@pytest.fixture
def default_state() -> CreatureState:
    return CreatureState()


@pytest.fixture
def optimal_tank() -> TankEnvironment:
    """Tank with perfect conditions."""
    return TankEnvironment(temperature=24.0, cleanliness=1.0, oxygen_level=1.0)


@pytest.fixture
def bad_tank() -> TankEnvironment:
    """Tank with poor conditions."""
    return TankEnvironment(temperature=12.0, cleanliness=0.1, oxygen_level=0.5)


# ── CreatureNeeds dataclass tests ────────────────────────────────────


class TestCreatureNeeds:
    """Tests for CreatureNeeds dataclass."""

    def test_default_values(self) -> None:
        needs = CreatureNeeds()
        assert needs.hunger == 0.0
        assert needs.comfort == 1.0
        assert needs.health == 1.0
        assert needs.stimulation == 1.0

    def test_custom_values(self) -> None:
        needs = CreatureNeeds(hunger=0.5, comfort=0.7, health=0.9, stimulation=0.3)
        assert needs.hunger == 0.5
        assert needs.comfort == 0.7
        assert needs.health == 0.9
        assert needs.stimulation == 0.3

    def test_clamping_above_max(self) -> None:
        needs = CreatureNeeds(hunger=1.5, comfort=2.0, health=3.0, stimulation=10.0)
        assert needs.hunger == 1.0
        assert needs.comfort == 1.0
        assert needs.health == 1.0
        assert needs.stimulation == 1.0

    def test_clamping_below_min(self) -> None:
        needs = CreatureNeeds(hunger=-0.5, comfort=-1.0, health=-2.0, stimulation=-0.1)
        assert needs.hunger == 0.0
        assert needs.comfort == 0.0
        assert needs.health == 0.0
        assert needs.stimulation == 0.0

    def test_timestamp_set_automatically(self) -> None:
        before = datetime.now(UTC)
        needs = CreatureNeeds()
        after = datetime.now(UTC)
        assert before <= needs.last_update <= after


# ── _clamp helper tests ─────────────────────────────────────────────


class TestClamp:
    """Tests for the _clamp utility."""

    def test_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_above_max(self) -> None:
        assert _clamp(1.5) == 1.0

    def test_below_min(self) -> None:
        assert _clamp(-0.5) == 0.0

    def test_custom_range(self) -> None:
        assert _clamp(15.0, 0.0, 10.0) == 10.0
        assert _clamp(-5.0, 0.0, 10.0) == 0.0

    def test_boundary_values(self) -> None:
        assert _clamp(0.0) == 0.0
        assert _clamp(1.0) == 1.0


# ── NeedsEngine construction tests ──────────────────────────────────


class TestNeedsEngineInit:
    """Tests for NeedsEngine initialization."""

    def test_default_config(self) -> None:
        engine = NeedsEngine()
        assert engine.config.hunger_rate == 0.02

    def test_custom_config(self) -> None:
        cfg = NeedsConfig(hunger_rate=0.05)
        engine = NeedsEngine(config=cfg)
        assert engine.config.hunger_rate == 0.05


# ── Hunger growth tests ─────────────────────────────────────────────


class TestHungerGrowth:
    """Tests for hunger increasing over time."""

    def test_hunger_increases_over_time(
        self, engine: NeedsEngine, default_state: CreatureState, optimal_tank: TankEnvironment
    ) -> None:
        result = engine.update(60.0, default_state, optimal_tank)
        assert result.hunger > 0.0

    def test_hunger_rate_scales_with_stage(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        mushroomer = CreatureState(stage=CreatureStage.MUSHROOMER, hunger=0.0)
        frogman = CreatureState(stage=CreatureStage.FROGMAN, hunger=0.0)

        result_mush = engine.update(10.0, mushroomer, optimal_tank)
        result_frog = engine.update(10.0, frogman, optimal_tank)

        assert result_frog.hunger > result_mush.hunger

    def test_stage_hunger_multipliers_exist_for_all_stages(self) -> None:
        for stage in CreatureStage:
            assert stage in STAGE_HUNGER_MULTIPLIER

    def test_hunger_capped_at_one(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        starving = CreatureState(hunger=0.99)
        result = engine.update(10000.0, starving, optimal_tank)
        assert result.hunger == 1.0

    def test_zero_elapsed_returns_current_values(
        self, engine: NeedsEngine, default_state: CreatureState, optimal_tank: TankEnvironment
    ) -> None:
        result = engine.update(0.0, default_state, optimal_tank)
        assert result.hunger == default_state.hunger

    def test_negative_elapsed_returns_current_values(
        self, engine: NeedsEngine, default_state: CreatureState, optimal_tank: TankEnvironment
    ) -> None:
        result = engine.update(-10.0, default_state, optimal_tank)
        assert result.hunger == default_state.hunger


# ── Comfort from tank conditions ─────────────────────────────────────


class TestComfortFromTank:
    """Tests for comfort derived from tank conditions."""

    def test_optimal_tank_gives_high_comfort(
        self, engine: NeedsEngine, default_state: CreatureState, optimal_tank: TankEnvironment
    ) -> None:
        result = engine.update(1.0, default_state, optimal_tank)
        assert result.comfort == 1.0

    def test_dirty_tank_reduces_comfort(
        self, engine: NeedsEngine, default_state: CreatureState
    ) -> None:
        dirty_tank = TankEnvironment(temperature=24.0, cleanliness=0.2, oxygen_level=1.0)
        result = engine.update(1.0, default_state, dirty_tank)
        assert result.comfort < 1.0

    def test_cold_tank_reduces_comfort(
        self, engine: NeedsEngine, default_state: CreatureState
    ) -> None:
        cold_tank = TankEnvironment(temperature=15.0, cleanliness=1.0, oxygen_level=1.0)
        result = engine.update(1.0, default_state, cold_tank)
        assert result.comfort < 1.0

    def test_hot_tank_reduces_comfort(
        self, engine: NeedsEngine, default_state: CreatureState
    ) -> None:
        hot_tank = TankEnvironment(temperature=35.0, cleanliness=1.0, oxygen_level=1.0)
        result = engine.update(1.0, default_state, hot_tank)
        assert result.comfort < 1.0

    def test_lethal_temp_gives_zero_temp_comfort(
        self, engine: NeedsEngine, default_state: CreatureState
    ) -> None:
        lethal_tank = TankEnvironment(temperature=5.0, cleanliness=1.0, oxygen_level=1.0)
        result = engine.update(1.0, default_state, lethal_tank)
        # Comfort = 0.5 * 0.0 (lethal temp) + 0.5 * 1.0 (clean) = 0.5
        assert result.comfort == 0.5

    def test_both_bad_conditions_give_low_comfort(
        self, engine: NeedsEngine, default_state: CreatureState, bad_tank: TankEnvironment
    ) -> None:
        result = engine.update(1.0, default_state, bad_tank)
        assert result.comfort < 0.5

    def test_comfort_clamped_to_valid_range(
        self, engine: NeedsEngine, default_state: CreatureState
    ) -> None:
        extreme_tank = TankEnvironment(temperature=0.0, cleanliness=0.0, oxygen_level=0.0)
        result = engine.update(1.0, default_state, extreme_tank)
        assert 0.0 <= result.comfort <= 1.0


# ── Health degradation tests ─────────────────────────────────────────


class TestHealthDegradation:
    """Tests for health degrading from unmet needs."""

    def test_health_degrades_when_starving(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        starving = CreatureState(hunger=0.9, health=1.0)
        result = engine.update(100.0, starving, optimal_tank)
        assert result.health < 1.0

    def test_health_regenerates_when_needs_met(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        hurt = CreatureState(hunger=0.0, health=0.5)
        result = engine.update(10.0, hurt, optimal_tank)
        assert result.health > 0.5

    def test_health_capped_at_one(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        almost_full = CreatureState(hunger=0.0, health=0.99)
        result = engine.update(10.0, almost_full, optimal_tank)
        assert result.health == 1.0

    def test_health_floors_at_zero(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        dying = CreatureState(hunger=1.0, health=0.01)
        result = engine.update(10000.0, dying, optimal_tank)
        assert result.health == 0.0

    def test_health_degrades_with_bad_comfort(
        self, engine: NeedsEngine, bad_tank: TankEnvironment
    ) -> None:
        state = CreatureState(hunger=0.0, health=1.0, comfort=1.0)
        result = engine.update(100.0, state, bad_tank)
        # Bad tank => low comfort => health damage
        assert result.health < 1.0


# ── Stimulation tests ────────────────────────────────────────────────


class TestStimulation:
    """Tests for stimulation decay and interaction boost."""

    def test_stimulation_decays_over_time(
        self, engine: NeedsEngine, default_state: CreatureState, optimal_tank: TankEnvironment
    ) -> None:
        result = engine.update(60.0, default_state, optimal_tank)
        assert result.stimulation < 1.0

    def test_interaction_boosts_stimulation(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        bored = CreatureState(comfort=0.3)
        result_no_interact = engine.update(10.0, bored, optimal_tank, interaction_count_delta=0)
        result_interact = engine.update(10.0, bored, optimal_tank, interaction_count_delta=3)
        assert result_interact.stimulation > result_no_interact.stimulation

    def test_stimulation_clamped(
        self, engine: NeedsEngine, optimal_tank: TankEnvironment
    ) -> None:
        state = CreatureState(comfort=0.5)
        result = engine.update(1.0, state, optimal_tank, interaction_count_delta=100)
        assert result.stimulation <= 1.0


# ── Urgent needs tests ───────────────────────────────────────────────


class TestUrgentNeeds:
    """Tests for get_urgent_needs() warnings."""

    def test_no_urgent_needs_when_healthy(self, engine: NeedsEngine) -> None:
        needs = CreatureNeeds(hunger=0.0, comfort=1.0, health=1.0, stimulation=1.0)
        assert engine.get_urgent_needs(needs) == []

    def test_starving_triggers_urgent(self, engine: NeedsEngine) -> None:
        needs = CreatureNeeds(hunger=0.9, comfort=1.0, health=1.0, stimulation=1.0)
        urgent = engine.get_urgent_needs(needs)
        assert any("starving" in msg.lower() for msg in urgent)

    def test_low_health_triggers_urgent(self, engine: NeedsEngine) -> None:
        needs = CreatureNeeds(hunger=0.0, comfort=1.0, health=0.1, stimulation=1.0)
        urgent = engine.get_urgent_needs(needs)
        assert any("health" in msg.lower() for msg in urgent)

    def test_low_comfort_triggers_urgent(self, engine: NeedsEngine) -> None:
        needs = CreatureNeeds(hunger=0.0, comfort=0.1, health=1.0, stimulation=1.0)
        urgent = engine.get_urgent_needs(needs)
        assert any("uncomfortable" in msg.lower() for msg in urgent)

    def test_low_stimulation_triggers_urgent(self, engine: NeedsEngine) -> None:
        needs = CreatureNeeds(hunger=0.0, comfort=1.0, health=1.0, stimulation=0.1)
        urgent = engine.get_urgent_needs(needs)
        assert any("bored" in msg.lower() or "unstimulated" in msg.lower() for msg in urgent)

    def test_multiple_urgent_needs(self, engine: NeedsEngine) -> None:
        needs = CreatureNeeds(hunger=0.95, comfort=0.1, health=0.1, stimulation=0.05)
        urgent = engine.get_urgent_needs(needs)
        assert len(urgent) == 4

    def test_threshold_boundary_not_triggered(self, engine: NeedsEngine) -> None:
        # Hunger at 0.79 (just below 0.8 threshold)
        needs = CreatureNeeds(hunger=0.79, comfort=0.5, health=0.5, stimulation=0.5)
        urgent = engine.get_urgent_needs(needs)
        assert not any("starving" in msg.lower() for msg in urgent)

    def test_threshold_boundary_triggered(self, engine: NeedsEngine) -> None:
        # Hunger at exactly 0.8 (at threshold)
        needs = CreatureNeeds(hunger=0.8, comfort=0.5, health=0.5, stimulation=0.5)
        urgent = engine.get_urgent_needs(needs)
        assert any("starving" in msg.lower() for msg in urgent)


# ── apply_to_state tests ─────────────────────────────────────────────


class TestApplyToState:
    """Tests for applying needs back to creature state."""

    def test_applies_hunger(self, engine: NeedsEngine) -> None:
        state = CreatureState(hunger=0.0)
        needs = CreatureNeeds(hunger=0.5, comfort=0.7, health=0.8)
        engine.apply_to_state(state, needs)
        assert state.hunger == 0.5

    def test_applies_comfort(self, engine: NeedsEngine) -> None:
        state = CreatureState(comfort=1.0)
        needs = CreatureNeeds(hunger=0.0, comfort=0.3, health=1.0)
        engine.apply_to_state(state, needs)
        assert state.comfort == 0.3

    def test_applies_health(self, engine: NeedsEngine) -> None:
        state = CreatureState(health=1.0)
        needs = CreatureNeeds(hunger=0.0, comfort=1.0, health=0.4)
        engine.apply_to_state(state, needs)
        assert state.health == 0.4

    def test_does_not_modify_other_fields(self, engine: NeedsEngine) -> None:
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            mood="irritated",
            trust_level=0.5,
            interaction_count=10,
        )
        needs = CreatureNeeds(hunger=0.5, comfort=0.5, health=0.5)
        engine.apply_to_state(state, needs)
        assert state.stage == CreatureStage.GILLMAN
        assert state.mood == "irritated"
        assert state.trust_level == 0.5
        assert state.interaction_count == 10


# ── Integration-style tests ──────────────────────────────────────────


class TestNeedsIntegration:
    """Integration tests combining multiple aspects."""

    def test_full_cycle_update_and_apply(self) -> None:
        engine = NeedsEngine()
        state = CreatureState(hunger=0.0, health=1.0, comfort=1.0)
        tank = TankEnvironment(temperature=24.0, cleanliness=1.0, oxygen_level=1.0)

        needs = engine.update(300.0, state, tank)  # 5 minutes
        engine.apply_to_state(state, needs)

        assert state.hunger > 0.0
        assert state.health <= 1.0
        assert state.comfort <= 1.0

    def test_neglected_creature_deteriorates(self) -> None:
        engine = NeedsEngine()
        state = CreatureState(hunger=0.5, health=0.8, comfort=0.5)
        bad_tank = TankEnvironment(temperature=12.0, cleanliness=0.1, oxygen_level=0.5)

        # Simulate a long period of neglect
        needs = engine.update(3600.0, state, bad_tank)  # 1 hour
        engine.apply_to_state(state, needs)

        assert state.hunger > 0.5
        assert state.health < 0.8

    def test_well_cared_creature_thrives(self) -> None:
        engine = NeedsEngine()
        state = CreatureState(hunger=0.0, health=0.7, comfort=1.0)
        tank = TankEnvironment(temperature=24.0, cleanliness=1.0, oxygen_level=1.0)

        # Short period, good conditions
        needs = engine.update(60.0, state, tank, interaction_count_delta=2)
        engine.apply_to_state(state, needs)

        # Health should regenerate
        assert state.health > 0.7
        assert state.comfort == 1.0

    def test_different_configs_change_behavior(self) -> None:
        fast_config = NeedsConfig(hunger_rate=0.1)
        slow_config = NeedsConfig(hunger_rate=0.001)
        tank = TankEnvironment(temperature=24.0, cleanliness=1.0, oxygen_level=1.0)
        state = CreatureState(hunger=0.0)

        fast_engine = NeedsEngine(config=fast_config)
        slow_engine = NeedsEngine(config=slow_config)

        fast_result = fast_engine.update(100.0, state, tank)
        slow_result = slow_engine.update(100.0, state, tank)

        assert fast_result.hunger > slow_result.hunger
