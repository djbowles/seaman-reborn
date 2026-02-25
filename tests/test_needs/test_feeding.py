"""Tests for the feeding mechanics module (US-029)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from seaman_brain.config import NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.needs.feeding import (
    FOOD_MOOD_EFFECT,
    FOOD_NUTRITION,
    STAGE_FOODS,
    FeedingEngine,
    FeedingResult,
    FoodType,
)
from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> NeedsConfig:
    """Default needs config with 30s cooldown."""
    return NeedsConfig(feeding_cooldown_seconds=30)


@pytest.fixture
def now() -> datetime:
    """Fixed reference time for testing."""
    return datetime(2026, 2, 25, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def engine(config: NeedsConfig, now: datetime) -> FeedingEngine:
    """FeedingEngine with injectable clock."""
    return FeedingEngine(config=config, now_func=lambda: now)


@pytest.fixture
def hungry_mushroomer(now: datetime) -> CreatureState:
    """A hungry mushroomer ready to eat (past cooldown)."""
    return CreatureState(
        stage=CreatureStage.MUSHROOMER,
        hunger=0.5,
        health=1.0,
        last_fed=now - timedelta(seconds=60),  # 60s ago, past 30s cooldown
    )


@pytest.fixture
def hungry_gillman(now: datetime) -> CreatureState:
    """A hungry gillman ready to eat."""
    return CreatureState(
        stage=CreatureStage.GILLMAN,
        hunger=0.6,
        health=1.0,
        last_fed=now - timedelta(seconds=60),
    )


@pytest.fixture
def full_creature(now: datetime) -> CreatureState:
    """A well-fed creature (hunger < 0.1)."""
    return CreatureState(
        stage=CreatureStage.PODFISH,
        hunger=0.05,
        health=1.0,
        last_fed=now - timedelta(seconds=60),
    )


# ---------------------------------------------------------------------------
# FoodType enum tests
# ---------------------------------------------------------------------------

class TestFoodType:
    """Tests for the FoodType enum."""

    def test_food_type_values(self) -> None:
        assert FoodType.PELLET.value == "pellet"
        assert FoodType.WORM.value == "worm"
        assert FoodType.INSECT.value == "insect"
        assert FoodType.NAUTILUS.value == "nautilus"

    def test_food_type_count(self) -> None:
        assert len(FoodType) == 4

    def test_food_type_from_value(self) -> None:
        assert FoodType("pellet") is FoodType.PELLET


# ---------------------------------------------------------------------------
# FeedingResult tests
# ---------------------------------------------------------------------------

class TestFeedingResult:
    """Tests for the FeedingResult dataclass."""

    def test_success_result(self) -> None:
        result = FeedingResult(
            success=True,
            hunger_change=-0.2,
            mood_change=0.05,
            health_change=0.0,
            message="Yum!",
        )
        assert result.success is True
        assert result.hunger_change == -0.2
        assert result.mood_change == 0.05
        assert result.health_change == 0.0
        assert result.message == "Yum!"

    def test_failure_result(self) -> None:
        result = FeedingResult(
            success=False,
            hunger_change=0.0,
            mood_change=0.0,
            health_change=0.0,
            message="Can't eat that!",
        )
        assert result.success is False


# ---------------------------------------------------------------------------
# Stage food validation tests
# ---------------------------------------------------------------------------

class TestStageFoods:
    """Tests for stage-specific food validation."""

    def test_mushroomer_only_eats_nautilus(self) -> None:
        assert STAGE_FOODS[CreatureStage.MUSHROOMER] == {FoodType.NAUTILUS}

    def test_gillman_eats_pellet_and_worm(self) -> None:
        assert STAGE_FOODS[CreatureStage.GILLMAN] == {FoodType.PELLET, FoodType.WORM}

    def test_podfish_eats_pellet_worm_insect(self) -> None:
        assert STAGE_FOODS[CreatureStage.PODFISH] == {
            FoodType.PELLET, FoodType.WORM, FoodType.INSECT,
        }

    def test_tadman_eats_pellet_worm_insect(self) -> None:
        assert STAGE_FOODS[CreatureStage.TADMAN] == {
            FoodType.PELLET, FoodType.WORM, FoodType.INSECT,
        }

    def test_frogman_eats_worm_and_insect(self) -> None:
        assert STAGE_FOODS[CreatureStage.FROGMAN] == {FoodType.WORM, FoodType.INSECT}

    def test_all_stages_have_foods(self) -> None:
        for stage in CreatureStage:
            assert stage in STAGE_FOODS, f"Missing foods for {stage}"
            assert len(STAGE_FOODS[stage]) > 0

    def test_nutrition_defined_for_all_foods(self) -> None:
        for food in FoodType:
            assert food in FOOD_NUTRITION

    def test_mood_effect_defined_for_all_foods(self) -> None:
        for food in FoodType:
            assert food in FOOD_MOOD_EFFECT


# ---------------------------------------------------------------------------
# FeedingEngine initialization tests
# ---------------------------------------------------------------------------

class TestFeedingEngineInit:
    """Tests for FeedingEngine construction."""

    def test_default_config(self) -> None:
        engine = FeedingEngine()
        assert engine.config.feeding_cooldown_seconds == 30

    def test_custom_config(self, config: NeedsConfig) -> None:
        engine = FeedingEngine(config=config)
        assert engine.config is config

    def test_custom_now_func(self, now: datetime) -> None:
        engine = FeedingEngine(now_func=lambda: now)
        # Engine should use the injected clock
        assert engine.config is not None


# ---------------------------------------------------------------------------
# Successful feeding tests
# ---------------------------------------------------------------------------

class TestSuccessfulFeeding:
    """Tests for happy-path feeding."""

    def test_feed_mushroomer_nautilus(
        self, engine: FeedingEngine, hungry_mushroomer: CreatureState, now: datetime,
    ) -> None:
        result = engine.feed(hungry_mushroomer, FoodType.NAUTILUS)

        assert result.success is True
        assert result.hunger_change == pytest.approx(-FOOD_NUTRITION[FoodType.NAUTILUS])
        assert result.mood_change == pytest.approx(FOOD_MOOD_EFFECT[FoodType.NAUTILUS])
        assert result.health_change == 0.0
        assert "nautilus" in result.message
        # State should be mutated
        expected_hunger = 0.5 - FOOD_NUTRITION[FoodType.NAUTILUS]
        assert hungry_mushroomer.hunger == pytest.approx(expected_hunger)
        assert hungry_mushroomer.last_fed == now

    def test_feed_gillman_worm(
        self, engine: FeedingEngine, hungry_gillman: CreatureState, now: datetime,
    ) -> None:
        result = engine.feed(hungry_gillman, FoodType.WORM)

        assert result.success is True
        assert result.hunger_change == pytest.approx(-FOOD_NUTRITION[FoodType.WORM])
        expected_hunger = 0.6 - FOOD_NUTRITION[FoodType.WORM]
        assert hungry_gillman.hunger == pytest.approx(expected_hunger)

    def test_feed_podfish_insect(
        self, engine: FeedingEngine, now: datetime,
    ) -> None:
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=0.7,
            last_fed=now - timedelta(seconds=60),
        )
        result = engine.feed(state, FoodType.INSECT)

        assert result.success is True
        assert result.hunger_change == pytest.approx(-FOOD_NUTRITION[FoodType.INSECT])

    def test_feed_reduces_hunger_not_below_zero(
        self, engine: FeedingEngine, now: datetime,
    ) -> None:
        """Feeding when hunger is low but >= 0.1 should clamp to 0."""
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=0.12,  # above 0.1 threshold
            last_fed=now - timedelta(seconds=60),
        )
        result = engine.feed(state, FoodType.INSECT)

        assert result.success is True
        assert state.hunger >= 0.0  # clamped, not negative

    def test_feed_updates_last_fed(
        self, engine: FeedingEngine, hungry_mushroomer: CreatureState, now: datetime,
    ) -> None:
        old_fed = hungry_mushroomer.last_fed
        engine.feed(hungry_mushroomer, FoodType.NAUTILUS)
        assert hungry_mushroomer.last_fed == now
        assert hungry_mushroomer.last_fed != old_fed

    def test_feed_all_stages_with_valid_food(
        self, now: datetime, config: NeedsConfig,
    ) -> None:
        """Each stage should accept at least one food type."""
        engine = FeedingEngine(config=config, now_func=lambda: now)
        for stage in CreatureStage:
            foods = STAGE_FOODS[stage]
            food = next(iter(foods))
            state = CreatureState(
                stage=stage,
                hunger=0.5,
                last_fed=now - timedelta(seconds=60),
            )
            result = engine.feed(state, food)
            assert result.success is True, f"Failed feeding {stage.value} with {food.value}"


# ---------------------------------------------------------------------------
# Wrong food for stage tests
# ---------------------------------------------------------------------------

class TestWrongFoodForStage:
    """Tests for stage-food validation failures."""

    def test_mushroomer_rejects_pellet(
        self, engine: FeedingEngine, hungry_mushroomer: CreatureState,
    ) -> None:
        result = engine.feed(hungry_mushroomer, FoodType.PELLET)

        assert result.success is False
        assert result.hunger_change == 0.0
        assert result.mood_change < 0  # negative mood from rejection
        assert "won't eat" in result.message
        assert "nautilus" in result.message  # suggest correct food

    def test_mushroomer_rejects_worm(
        self, engine: FeedingEngine, hungry_mushroomer: CreatureState,
    ) -> None:
        result = engine.feed(hungry_mushroomer, FoodType.WORM)
        assert result.success is False

    def test_mushroomer_rejects_insect(
        self, engine: FeedingEngine, hungry_mushroomer: CreatureState,
    ) -> None:
        result = engine.feed(hungry_mushroomer, FoodType.INSECT)
        assert result.success is False

    def test_gillman_rejects_nautilus(
        self, engine: FeedingEngine, hungry_gillman: CreatureState,
    ) -> None:
        result = engine.feed(hungry_gillman, FoodType.NAUTILUS)
        assert result.success is False
        assert "won't eat" in result.message

    def test_gillman_rejects_insect(
        self, engine: FeedingEngine, hungry_gillman: CreatureState,
    ) -> None:
        result = engine.feed(hungry_gillman, FoodType.INSECT)
        assert result.success is False

    def test_frogman_rejects_pellet(
        self, engine: FeedingEngine, now: datetime,
    ) -> None:
        state = CreatureState(
            stage=CreatureStage.FROGMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=60),
        )
        result = engine.feed(state, FoodType.PELLET)
        assert result.success is False
        assert "won't eat" in result.message

    def test_wrong_food_does_not_mutate_state(
        self, engine: FeedingEngine, hungry_mushroomer: CreatureState,
    ) -> None:
        original_hunger = hungry_mushroomer.hunger
        original_last_fed = hungry_mushroomer.last_fed
        engine.feed(hungry_mushroomer, FoodType.PELLET)
        assert hungry_mushroomer.hunger == original_hunger
        assert hungry_mushroomer.last_fed == original_last_fed


# ---------------------------------------------------------------------------
# Overfeeding tests
# ---------------------------------------------------------------------------

class TestOverfeeding:
    """Tests for overfeeding penalty when creature is full."""

    def test_overfeeding_when_hunger_below_threshold(
        self, engine: FeedingEngine, full_creature: CreatureState,
    ) -> None:
        result = engine.feed(full_creature, FoodType.PELLET)

        assert result.success is True  # feeding "works" but with penalty
        assert result.hunger_change == 0.0  # no hunger reduction
        assert result.mood_change < 0  # mood penalty
        assert result.health_change < 0  # health penalty
        assert "already full" in result.message
        assert "discomfort" in result.message.lower()

    def test_overfeeding_health_penalty(
        self, engine: FeedingEngine, full_creature: CreatureState,
    ) -> None:
        original_health = full_creature.health
        engine.feed(full_creature, FoodType.PELLET)
        assert full_creature.health < original_health
        assert full_creature.health == pytest.approx(original_health - 0.05)

    def test_overfeeding_updates_last_fed(
        self, engine: FeedingEngine, full_creature: CreatureState, now: datetime,
    ) -> None:
        engine.feed(full_creature, FoodType.PELLET)
        assert full_creature.last_fed == now

    def test_overfeeding_at_exact_threshold(
        self, engine: FeedingEngine, now: datetime,
    ) -> None:
        """Hunger exactly 0.1 should NOT trigger overfeeding."""
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=0.1,
            last_fed=now - timedelta(seconds=60),
        )
        result = engine.feed(state, FoodType.PELLET)
        assert result.success is True
        assert "already full" not in result.message  # normal feed, not overfeeding

    def test_overfeeding_health_clamps_at_zero(
        self, engine: FeedingEngine, now: datetime,
    ) -> None:
        """Repeated overfeeding shouldn't push health below 0."""
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=0.0,
            health=0.02,
            last_fed=now - timedelta(seconds=60),
        )
        engine.feed(state, FoodType.PELLET)
        assert state.health >= 0.0


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------

class TestFeedingCooldown:
    """Tests for feeding cooldown enforcement."""

    def test_cooldown_rejects_feeding(
        self, config: NeedsConfig, now: datetime,
    ) -> None:
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=10),  # only 10s ago, cooldown is 30s
        )
        result = engine.feed(state, FoodType.PELLET)

        assert result.success is False
        assert result.hunger_change == 0.0
        assert "Too soon" in result.message
        assert "20" in result.message  # ~20 seconds remaining

    def test_cooldown_does_not_mutate_state(
        self, config: NeedsConfig, now: datetime,
    ) -> None:
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=10),
        )
        original_hunger = state.hunger
        original_last_fed = state.last_fed
        engine.feed(state, FoodType.PELLET)
        assert state.hunger == original_hunger
        assert state.last_fed == original_last_fed

    def test_cooldown_expires_allows_feeding(
        self, config: NeedsConfig, now: datetime,
    ) -> None:
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=31),  # past 30s cooldown
        )
        result = engine.feed(state, FoodType.PELLET)
        assert result.success is True

    def test_cooldown_exact_boundary(
        self, config: NeedsConfig, now: datetime,
    ) -> None:
        """Feeding exactly at cooldown boundary should still be rejected."""
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=29),  # just under 30s
        )
        result = engine.feed(state, FoodType.PELLET)
        assert result.success is False

    def test_cooldown_at_exact_seconds(
        self, config: NeedsConfig, now: datetime,
    ) -> None:
        """Feeding exactly at cooldown time should succeed."""
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=30),  # exactly 30s
        )
        result = engine.feed(state, FoodType.PELLET)
        assert result.success is True

    def test_custom_cooldown(self, now: datetime) -> None:
        """Custom cooldown duration should be respected."""
        config = NeedsConfig(feeding_cooldown_seconds=5)
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.5,
            last_fed=now - timedelta(seconds=6),
        )
        result = engine.feed(state, FoodType.PELLET)
        assert result.success is True


# ---------------------------------------------------------------------------
# is_on_cooldown / cooldown_remaining tests
# ---------------------------------------------------------------------------

class TestCooldownHelpers:
    """Tests for is_on_cooldown and cooldown_remaining."""

    def test_is_on_cooldown_true(self, engine: FeedingEngine, now: datetime) -> None:
        state = CreatureState(
            last_fed=now - timedelta(seconds=10),
        )
        assert engine.is_on_cooldown(state) is True

    def test_is_on_cooldown_false(self, engine: FeedingEngine, now: datetime) -> None:
        state = CreatureState(
            last_fed=now - timedelta(seconds=60),
        )
        assert engine.is_on_cooldown(state) is False

    def test_cooldown_remaining_active(self, engine: FeedingEngine, now: datetime) -> None:
        state = CreatureState(
            last_fed=now - timedelta(seconds=10),
        )
        remaining = engine.cooldown_remaining(state)
        assert remaining == pytest.approx(20.0)

    def test_cooldown_remaining_expired(self, engine: FeedingEngine, now: datetime) -> None:
        state = CreatureState(
            last_fed=now - timedelta(seconds=60),
        )
        remaining = engine.cooldown_remaining(state)
        assert remaining == 0.0


# ---------------------------------------------------------------------------
# get_available_foods tests
# ---------------------------------------------------------------------------

class TestGetAvailableFoods:
    """Tests for available food lookup."""

    def test_mushroomer_available_foods(self, engine: FeedingEngine) -> None:
        foods = engine.get_available_foods(CreatureStage.MUSHROOMER)
        assert foods == [FoodType.NAUTILUS]

    def test_podfish_available_foods(self, engine: FeedingEngine) -> None:
        foods = engine.get_available_foods(CreatureStage.PODFISH)
        assert set(foods) == {FoodType.PELLET, FoodType.WORM, FoodType.INSECT}
        assert len(foods) == 3

    def test_frogman_available_foods(self, engine: FeedingEngine) -> None:
        foods = engine.get_available_foods(CreatureStage.FROGMAN)
        assert set(foods) == {FoodType.WORM, FoodType.INSECT}


# ---------------------------------------------------------------------------
# Priority order tests (cooldown checked before stage food)
# ---------------------------------------------------------------------------

class TestPriorityOrder:
    """Tests for check priority: cooldown -> stage food -> overfeeding -> normal."""

    def test_cooldown_checked_before_stage_food(
        self, config: NeedsConfig, now: datetime,
    ) -> None:
        """Even if food is wrong, cooldown rejection should happen first."""
        engine = FeedingEngine(config=config, now_func=lambda: now)
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            hunger=0.5,
            last_fed=now - timedelta(seconds=10),  # on cooldown
        )
        result = engine.feed(state, FoodType.PELLET)  # wrong food
        assert result.success is False
        assert "Too soon" in result.message  # cooldown message, not wrong-food

    def test_stage_food_checked_before_overfeeding(
        self, engine: FeedingEngine, now: datetime,
    ) -> None:
        """Wrong food rejection should happen before overfeeding check."""
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            hunger=0.05,  # full
            last_fed=now - timedelta(seconds=60),
        )
        result = engine.feed(state, FoodType.PELLET)  # wrong food
        assert result.success is False
        assert "won't eat" in result.message  # wrong-food, not overfeeding


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge case tests for feeding mechanics."""

    def test_feed_at_max_hunger(self, engine: FeedingEngine, now: datetime) -> None:
        """Feeding when starving should reduce hunger significantly."""
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=1.0,
            last_fed=now - timedelta(seconds=60),
        )
        result = engine.feed(state, FoodType.INSECT)
        assert result.success is True
        assert state.hunger < 1.0
        assert state.hunger == pytest.approx(1.0 - FOOD_NUTRITION[FoodType.INSECT])

    def test_repeated_feeding_reduces_hunger_incrementally(
        self, now: datetime,
    ) -> None:
        """Multiple feedings with advancing clock should accumulate."""
        time = [now]
        config = NeedsConfig(feeding_cooldown_seconds=1)
        engine = FeedingEngine(config=config, now_func=lambda: time[0])

        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=0.8,
            last_fed=now - timedelta(seconds=10),
        )

        # First feed
        r1 = engine.feed(state, FoodType.PELLET)
        assert r1.success is True
        hunger_after_1 = state.hunger

        # Advance clock past cooldown
        time[0] = now + timedelta(seconds=2)
        r2 = engine.feed(state, FoodType.PELLET)
        assert r2.success is True
        assert state.hunger < hunger_after_1

    def test_consecutive_overfeeding_stacks_penalty(
        self, now: datetime,
    ) -> None:
        """Multiple overfeedings should stack health penalties."""
        time = [now]
        config = NeedsConfig(feeding_cooldown_seconds=1)
        engine = FeedingEngine(config=config, now_func=lambda: time[0])

        state = CreatureState(
            stage=CreatureStage.PODFISH,
            hunger=0.0,
            health=1.0,
            last_fed=now - timedelta(seconds=10),
        )

        engine.feed(state, FoodType.PELLET)
        health_after_1 = state.health

        time[0] = now + timedelta(seconds=2)
        engine.feed(state, FoodType.PELLET)
        assert state.health < health_after_1
