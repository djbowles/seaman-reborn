"""Tests for api.actions — ActionDispatcher routing and ActionResult."""

from __future__ import annotations

import pytest

from seaman_brain.api.actions import ActionDispatcher, ActionResult
from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.types import CreatureStage


@pytest.fixture()
def creature() -> CreatureState:
    """Default MUSHROOMER creature with some hunger so feeding succeeds."""
    return CreatureState(stage=CreatureStage.MUSHROOMER, hunger=0.5)


@pytest.fixture()
def tank() -> TankEnvironment:
    """Default aquarium tank."""
    return TankEnvironment()


@pytest.fixture()
def needs_config() -> NeedsConfig:
    """Needs config with zero feeding cooldown so tests don't hit cooldown."""
    return NeedsConfig(feeding_cooldown_seconds=0)


@pytest.fixture()
def env_config() -> EnvironmentConfig:
    return EnvironmentConfig()


@pytest.fixture()
def dispatcher(
    creature: CreatureState,
    tank: TankEnvironment,
    needs_config: NeedsConfig,
    env_config: EnvironmentConfig,
) -> ActionDispatcher:
    return ActionDispatcher(creature, tank, needs_config, env_config)


# ---------------------------------------------------------------------------
# TestActionDispatcherInit
# ---------------------------------------------------------------------------


class TestActionDispatcherInit:
    """Construction and internal wiring."""

    def test_stores_creature_state(
        self, creature: CreatureState, tank: TankEnvironment
    ) -> None:
        d = ActionDispatcher(creature, tank)
        assert d.creature_state is creature

    def test_accepts_none_configs(
        self, creature: CreatureState, tank: TankEnvironment
    ) -> None:
        """None configs fall back to defaults without raising."""
        d = ActionDispatcher(creature, tank, None, None)
        assert d.creature_state is creature

    def test_stores_explicit_configs(
        self,
        creature: CreatureState,
        tank: TankEnvironment,
        needs_config: NeedsConfig,
        env_config: EnvironmentConfig,
    ) -> None:
        d = ActionDispatcher(creature, tank, needs_config, env_config)
        assert d.creature_state is creature


# ---------------------------------------------------------------------------
# TestFeedAction
# ---------------------------------------------------------------------------


class TestFeedAction:
    """Feeding dispatches — valid food, wrong stage, auto-pick, bad string."""

    def test_feed_valid_food_nautilus_for_mushroomer(
        self, dispatcher: ActionDispatcher
    ) -> None:
        result = dispatcher.dispatch("feed", {"food_type": "nautilus"})
        assert isinstance(result, ActionResult)
        assert result.success is True
        assert result.action == "feed"
        assert "nautilus" in result.message.lower()

    def test_feed_wrong_stage_food(self, dispatcher: ActionDispatcher) -> None:
        """MUSHROOMER cannot eat pellet — should fail."""
        result = dispatcher.dispatch("feed", {"food_type": "pellet"})
        assert result.success is False
        assert result.action == "feed"
        assert "won't eat" in result.message.lower() or "nautilus" in result.message.lower()

    def test_feed_auto_pick_first_available(
        self, dispatcher: ActionDispatcher
    ) -> None:
        """No food_type param auto-selects first available for the stage."""
        result = dispatcher.dispatch("feed")
        assert result.success is True
        assert result.action == "feed"

    def test_feed_invalid_food_type_string(
        self, dispatcher: ActionDispatcher
    ) -> None:
        result = dispatcher.dispatch("feed", {"food_type": "pizza"})
        assert result.success is False
        assert "invalid" in result.message.lower() or "valid" in result.message.lower()
        assert result.action == "feed"

    def test_feed_gillman_pellet(self, tank: TankEnvironment) -> None:
        """GILLMAN should accept pellet."""
        cs = CreatureState(stage=CreatureStage.GILLMAN, hunger=0.5)
        d = ActionDispatcher(cs, tank, NeedsConfig(feeding_cooldown_seconds=0))
        result = d.dispatch("feed", {"food_type": "pellet"})
        assert result.success is True

    def test_feed_gillman_rejects_nautilus(self, tank: TankEnvironment) -> None:
        """GILLMAN cannot eat nautilus."""
        cs = CreatureState(stage=CreatureStage.GILLMAN, hunger=0.5)
        d = ActionDispatcher(cs, tank, NeedsConfig(feeding_cooldown_seconds=0))
        result = d.dispatch("feed", {"food_type": "nautilus"})
        assert result.success is False


# ---------------------------------------------------------------------------
# TestTapGlassAction
# ---------------------------------------------------------------------------


class TestTapGlassAction:
    """Tapping the glass increments interaction_count."""

    def test_tap_glass_increments_count(
        self, dispatcher: ActionDispatcher, creature: CreatureState
    ) -> None:
        assert creature.interaction_count == 0
        result = dispatcher.dispatch("tap_glass")
        assert result.success is True
        assert result.action == "tap_glass"
        assert creature.interaction_count == 1

    def test_tap_glass_multiple_times(
        self, dispatcher: ActionDispatcher, creature: CreatureState
    ) -> None:
        for _ in range(3):
            dispatcher.dispatch("tap_glass")
        assert creature.interaction_count == 3


# ---------------------------------------------------------------------------
# TestAdjustTemperatureAction
# ---------------------------------------------------------------------------


class TestAdjustTemperatureAction:
    """Temperature adjustment with explicit delta and default delta."""

    def test_adjust_temperature_with_delta(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        original = tank.temperature
        result = dispatcher.dispatch("adjust_temperature", {"delta": 2.0})
        assert result.success is True
        assert result.action == "adjust_temperature"
        assert tank.temperature == pytest.approx(original + 2.0)

    def test_adjust_temperature_default_delta(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        """Omitting delta defaults to +1.0."""
        original = tank.temperature
        result = dispatcher.dispatch("adjust_temperature")
        assert result.success is True
        assert tank.temperature == pytest.approx(original + 1.0)

    def test_adjust_temperature_negative_delta(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        original = tank.temperature
        result = dispatcher.dispatch("adjust_temperature", {"delta": -3.0})
        assert result.success is True
        assert tank.temperature == pytest.approx(original - 3.0)


# ---------------------------------------------------------------------------
# TestCleanAction
# ---------------------------------------------------------------------------


class TestCleanAction:
    """Tank cleaning — success and cooldown."""

    def test_clean_success(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        tank.cleanliness = 0.5
        result = dispatcher.dispatch("clean")
        assert result.success is True
        assert result.action == "clean"
        assert tank.cleanliness > 0.5

    def test_clean_cooldown_blocks_immediate_retry(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        tank.cleanliness = 0.3
        first = dispatcher.dispatch("clean")
        assert first.success is True
        second = dispatcher.dispatch("clean")
        assert second.success is False
        assert "wait" in second.message.lower() or "still" in second.message.lower()


# ---------------------------------------------------------------------------
# TestAerateAction
# ---------------------------------------------------------------------------


class TestAerateAction:
    """Aeration in aquarium vs terrarium auto-switch to sprinkle."""

    def test_aerate_aquarium(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        tank.oxygen_level = 0.5
        result = dispatcher.dispatch("aerate")
        assert result.success is True
        assert result.action == "aerate"
        assert tank.oxygen_level > 0.5

    def test_aerate_terrarium_auto_switches_to_sprinkle(
        self,
        creature: CreatureState,
        needs_config: NeedsConfig,
        env_config: EnvironmentConfig,
    ) -> None:
        """In TERRARIUM mode, aerate auto-delegates to sprinkle()."""
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        tank.oxygen_level = 0.5
        d = ActionDispatcher(creature, tank, needs_config, env_config)
        result = d.dispatch("aerate")
        assert result.success is True
        assert result.action == "aerate"
        assert tank.oxygen_level > 0.5


# ---------------------------------------------------------------------------
# TestDrainFillAction
# ---------------------------------------------------------------------------


class TestDrainFillAction:
    """Drain and fill transition the tank between aquarium and terrarium."""

    def test_drain_aquarium(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        assert tank.environment_type == EnvironmentType.AQUARIUM
        result = dispatcher.dispatch("drain")
        assert result.success is True
        assert result.action == "drain"
        assert tank.environment_type == EnvironmentType.TERRARIUM
        assert tank.water_level == pytest.approx(0.0)

    def test_drain_already_terrarium_fails(
        self,
        creature: CreatureState,
        needs_config: NeedsConfig,
        env_config: EnvironmentConfig,
    ) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        d = ActionDispatcher(creature, tank, needs_config, env_config)
        result = d.dispatch("drain")
        assert result.success is False
        assert "already" in result.message.lower()

    def test_fill_terrarium(
        self,
        creature: CreatureState,
        needs_config: NeedsConfig,
        env_config: EnvironmentConfig,
    ) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        d = ActionDispatcher(creature, tank, needs_config, env_config)
        result = d.dispatch("fill")
        assert result.success is True
        assert result.action == "fill"
        assert tank.environment_type == EnvironmentType.AQUARIUM
        assert tank.water_level == pytest.approx(1.0)

    def test_fill_already_aquarium_fails(
        self, dispatcher: ActionDispatcher, tank: TankEnvironment
    ) -> None:
        assert tank.environment_type == EnvironmentType.AQUARIUM
        result = dispatcher.dispatch("fill")
        assert result.success is False
        assert "already" in result.message.lower()


# ---------------------------------------------------------------------------
# TestUnknownAction
# ---------------------------------------------------------------------------


class TestUnknownAction:
    """Unknown action names return a failure result."""

    def test_unknown_action_returns_failure(
        self, dispatcher: ActionDispatcher
    ) -> None:
        result = dispatcher.dispatch("dance")
        assert result.success is False
        assert result.action == "dance"
        assert "unknown" in result.message.lower()

    def test_empty_action_string(self, dispatcher: ActionDispatcher) -> None:
        result = dispatcher.dispatch("")
        assert result.success is False


# ---------------------------------------------------------------------------
# TestCreatureStateProperty
# ---------------------------------------------------------------------------


class TestCreatureStateProperty:
    """Getter and setter for creature_state."""

    def test_getter_returns_initial(
        self, dispatcher: ActionDispatcher, creature: CreatureState
    ) -> None:
        assert dispatcher.creature_state is creature

    def test_setter_replaces_state(
        self, dispatcher: ActionDispatcher
    ) -> None:
        new_cs = CreatureState(stage=CreatureStage.FROGMAN)
        dispatcher.creature_state = new_cs
        assert dispatcher.creature_state is new_cs
        assert dispatcher.creature_state.stage == CreatureStage.FROGMAN

    def test_actions_use_updated_state(
        self, dispatcher: ActionDispatcher
    ) -> None:
        """After setter, dispatch uses the new creature state."""
        new_cs = CreatureState(stage=CreatureStage.GILLMAN, hunger=0.5)
        dispatcher.creature_state = new_cs
        result = dispatcher.dispatch("tap_glass")
        assert result.success is True
        assert new_cs.interaction_count == 1
