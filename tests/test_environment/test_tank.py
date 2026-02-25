"""Tests for TankEnvironment - tank state, degradation, maintenance, persistence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from seaman_brain.config import EnvironmentConfig
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment, _clamp

# ── Helpers ──────────────────────────────────────────────────────────────────


class TestClamp:
    """Tests for the _clamp helper."""

    def test_within_range(self) -> None:
        assert _clamp(0.5) == 0.5

    def test_below_min(self) -> None:
        assert _clamp(-0.5) == 0.0

    def test_above_max(self) -> None:
        assert _clamp(1.5) == 1.0

    def test_custom_range(self) -> None:
        assert _clamp(5.0, 0.0, 10.0) == 5.0
        assert _clamp(-1.0, 0.0, 10.0) == 0.0
        assert _clamp(15.0, 0.0, 10.0) == 10.0


# ── EnvironmentType Enum ────────────────────────────────────────────────────


class TestEnvironmentType:
    """Tests for EnvironmentType enum."""

    def test_values(self) -> None:
        assert EnvironmentType.AQUARIUM.value == "aquarium"
        assert EnvironmentType.TERRARIUM.value == "terrarium"

    def test_from_string(self) -> None:
        assert EnvironmentType("aquarium") == EnvironmentType.AQUARIUM
        assert EnvironmentType("terrarium") == EnvironmentType.TERRARIUM

    def test_invalid_value(self) -> None:
        with pytest.raises(ValueError):
            EnvironmentType("swamp")


# ── TankEnvironment Creation ────────────────────────────────────────────────


class TestTankCreation:
    """Tests for TankEnvironment creation and defaults."""

    def test_defaults(self) -> None:
        tank = TankEnvironment()
        assert tank.temperature == 24.0
        assert tank.cleanliness == 1.0
        assert tank.oxygen_level == 1.0
        assert tank.water_level == 1.0
        assert tank.environment_type == EnvironmentType.AQUARIUM
        assert isinstance(tank.last_update, datetime)

    def test_custom_values(self) -> None:
        tank = TankEnvironment(
            temperature=22.0,
            cleanliness=0.8,
            oxygen_level=0.9,
            water_level=0.5,
            environment_type=EnvironmentType.TERRARIUM,
        )
        assert tank.temperature == 22.0
        assert tank.cleanliness == 0.8
        assert tank.oxygen_level == 0.9
        assert tank.water_level == 0.5
        assert tank.environment_type == EnvironmentType.TERRARIUM

    def test_clamping(self) -> None:
        tank = TankEnvironment(cleanliness=-0.5, oxygen_level=1.5, water_level=2.0)
        assert tank.cleanliness == 0.0
        assert tank.oxygen_level == 1.0
        assert tank.water_level == 1.0

    def test_string_environment_type(self) -> None:
        """String environment_type is auto-converted to enum."""
        tank = TankEnvironment(environment_type="terrarium")  # type: ignore[arg-type]
        assert tank.environment_type == EnvironmentType.TERRARIUM

    def test_from_config(self) -> None:
        config = EnvironmentConfig(
            initial_temperature=26.0,
            initial_environment="terrarium",
        )
        tank = TankEnvironment.from_config(config)
        assert tank.temperature == 26.0
        assert tank.environment_type == EnvironmentType.TERRARIUM
        assert tank.cleanliness == 1.0
        assert tank.oxygen_level == 1.0

    def test_from_config_defaults(self) -> None:
        tank = TankEnvironment.from_config(EnvironmentConfig())
        assert tank.temperature == 24.0
        assert tank.environment_type == EnvironmentType.AQUARIUM


# ── Update / Degradation ────────────────────────────────────────────────────


class TestUpdate:
    """Tests for tank degradation over time."""

    def test_cleanliness_degrades(self) -> None:
        tank = TankEnvironment(cleanliness=1.0, oxygen_level=1.0)
        config = EnvironmentConfig(cleanliness_decay_rate=0.01, oxygen_decay_rate=0.0)
        tank.update(10.0, config)
        assert tank.cleanliness == pytest.approx(0.9)
        assert tank.oxygen_level == 1.0  # No oxygen decay

    def test_oxygen_degrades(self) -> None:
        tank = TankEnvironment(cleanliness=1.0, oxygen_level=1.0)
        config = EnvironmentConfig(cleanliness_decay_rate=0.0, oxygen_decay_rate=0.005)
        tank.update(20.0, config)
        assert tank.oxygen_level == pytest.approx(0.9)
        assert tank.cleanliness == 1.0  # No cleanliness decay

    def test_both_degrade(self) -> None:
        tank = TankEnvironment(cleanliness=1.0, oxygen_level=1.0)
        config = EnvironmentConfig(cleanliness_decay_rate=0.01, oxygen_decay_rate=0.005)
        tank.update(10.0, config)
        assert tank.cleanliness == pytest.approx(0.9)
        assert tank.oxygen_level == pytest.approx(0.95)

    def test_degradation_clamps_to_zero(self) -> None:
        tank = TankEnvironment(cleanliness=0.05, oxygen_level=0.02)
        config = EnvironmentConfig(cleanliness_decay_rate=0.1, oxygen_decay_rate=0.1)
        tank.update(10.0, config)
        assert tank.cleanliness == 0.0
        assert tank.oxygen_level == 0.0

    def test_negative_elapsed_ignored(self) -> None:
        tank = TankEnvironment(cleanliness=0.5)
        tank.update(-5.0)
        assert tank.cleanliness == 0.5

    def test_zero_elapsed_ignored(self) -> None:
        tank = TankEnvironment(cleanliness=0.5)
        tank.update(0.0)
        assert tank.cleanliness == 0.5

    def test_terrarium_halves_oxygen_decay(self) -> None:
        aquarium = TankEnvironment(oxygen_level=1.0, environment_type=EnvironmentType.AQUARIUM)
        terrarium = TankEnvironment(oxygen_level=1.0, environment_type=EnvironmentType.TERRARIUM)
        config = EnvironmentConfig(oxygen_decay_rate=0.01, cleanliness_decay_rate=0.0)

        aquarium.update(10.0, config)
        terrarium.update(10.0, config)

        assert aquarium.oxygen_level == pytest.approx(0.9)
        assert terrarium.oxygen_level == pytest.approx(0.95)

    def test_update_uses_default_config(self) -> None:
        tank = TankEnvironment(cleanliness=1.0, oxygen_level=1.0)
        tank.update(10.0)  # No config — uses EnvironmentConfig defaults
        # Default rates: cleanliness=0.01, oxygen=0.005
        assert tank.cleanliness == pytest.approx(0.9)
        assert tank.oxygen_level == pytest.approx(0.95)

    def test_update_updates_last_update(self) -> None:
        before = datetime.now(UTC)
        tank = TankEnvironment()
        tank.update(1.0)
        assert tank.last_update >= before


# ── Temperature ─────────────────────────────────────────────────────────────


class TestTemperature:
    """Tests for temperature control."""

    def test_set_temperature_normal(self) -> None:
        tank = TankEnvironment()
        tank.set_temperature(26.0)
        assert tank.temperature == 26.0

    def test_set_temperature_clamped_low(self) -> None:
        tank = TankEnvironment()
        config = EnvironmentConfig(lethal_temp_min=10.0)
        tank.set_temperature(-50.0, config)
        assert tank.temperature == 5.0  # lethal_min - 5

    def test_set_temperature_clamped_high(self) -> None:
        tank = TankEnvironment()
        config = EnvironmentConfig(lethal_temp_max=38.0)
        tank.set_temperature(100.0, config)
        assert tank.temperature == 43.0  # lethal_max + 5

    def test_set_temperature_default_config(self) -> None:
        tank = TankEnvironment()
        tank.set_temperature(25.0)
        assert tank.temperature == 25.0

    def test_adjust_temperature_positive(self) -> None:
        tank = TankEnvironment(temperature=24.0)
        tank.adjust_temperature(2.0)
        assert tank.temperature == 26.0

    def test_adjust_temperature_negative(self) -> None:
        tank = TankEnvironment(temperature=24.0)
        tank.adjust_temperature(-4.0)
        assert tank.temperature == 20.0

    def test_adjust_temperature_clamped(self) -> None:
        tank = TankEnvironment(temperature=24.0)
        config = EnvironmentConfig(lethal_temp_max=38.0)
        tank.adjust_temperature(100.0, config)
        assert tank.temperature == 43.0  # lethal_max + 5


# ── Clean ────────────────────────────────────────────────────────────────────


class TestClean:
    """Tests for tank cleaning."""

    def test_clean_restores_full(self) -> None:
        tank = TankEnvironment(cleanliness=0.3)
        tank.clean()
        assert tank.cleanliness == 1.0

    def test_clean_already_clean(self) -> None:
        tank = TankEnvironment(cleanliness=1.0)
        tank.clean()
        assert tank.cleanliness == 1.0

    def test_clean_from_zero(self) -> None:
        tank = TankEnvironment(cleanliness=0.0)
        tank.clean()
        assert tank.cleanliness == 1.0


# ── Drain / Fill ─────────────────────────────────────────────────────────────


class TestDrainFill:
    """Tests for aquarium<->terrarium transitions."""

    def test_drain_success(self) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)
        result = tank.drain()
        assert result is True
        assert tank.environment_type == EnvironmentType.TERRARIUM
        assert tank.water_level == 0.0

    def test_drain_already_terrarium(self) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        result = tank.drain()
        assert result is False
        assert tank.environment_type == EnvironmentType.TERRARIUM

    def test_fill_success(self) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM, water_level=0.0)
        result = tank.fill()
        assert result is True
        assert tank.environment_type == EnvironmentType.AQUARIUM
        assert tank.water_level == 1.0

    def test_fill_already_aquarium(self) -> None:
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)
        result = tank.fill()
        assert result is False
        assert tank.environment_type == EnvironmentType.AQUARIUM


# ── Habitability ─────────────────────────────────────────────────────────────


class TestHabitability:
    """Tests for is_habitable check."""

    def test_habitable_defaults(self) -> None:
        tank = TankEnvironment()
        assert tank.is_habitable() is True

    def test_not_habitable_temp_too_low(self) -> None:
        tank = TankEnvironment(temperature=5.0)
        assert tank.is_habitable() is False

    def test_not_habitable_temp_too_high(self) -> None:
        tank = TankEnvironment(temperature=42.0)
        assert tank.is_habitable() is False

    def test_not_habitable_no_oxygen(self) -> None:
        tank = TankEnvironment(oxygen_level=0.05)
        assert tank.is_habitable() is False

    def test_not_habitable_too_dirty(self) -> None:
        tank = TankEnvironment(cleanliness=0.03)
        assert tank.is_habitable() is False

    def test_habitable_at_boundaries(self) -> None:
        """Values at exact boundary should still be habitable."""
        config = EnvironmentConfig(lethal_temp_min=10.0, lethal_temp_max=38.0)
        tank = TankEnvironment(temperature=10.0, oxygen_level=0.1, cleanliness=0.05)
        assert tank.is_habitable(config) is True

    def test_not_habitable_custom_config(self) -> None:
        config = EnvironmentConfig(lethal_temp_min=20.0, lethal_temp_max=30.0)
        tank = TankEnvironment(temperature=19.0)
        assert tank.is_habitable(config) is False


# ── Optimal Temperature ──────────────────────────────────────────────────────


class TestOptimalTemperature:
    """Tests for is_temperature_optimal."""

    def test_optimal(self) -> None:
        tank = TankEnvironment(temperature=24.0)
        assert tank.is_temperature_optimal() is True

    def test_below_optimal(self) -> None:
        tank = TankEnvironment(temperature=15.0)
        assert tank.is_temperature_optimal() is False

    def test_above_optimal(self) -> None:
        tank = TankEnvironment(temperature=35.0)
        assert tank.is_temperature_optimal() is False

    def test_at_optimal_boundaries(self) -> None:
        config = EnvironmentConfig(optimal_temp_min=20.0, optimal_temp_max=28.0)
        assert TankEnvironment(temperature=20.0).is_temperature_optimal(config) is True
        assert TankEnvironment(temperature=28.0).is_temperature_optimal(config) is True


# ── Warnings ─────────────────────────────────────────────────────────────────


class TestWarnings:
    """Tests for tank warning generation."""

    def test_no_warnings_good_tank(self) -> None:
        tank = TankEnvironment()
        assert tank.get_warnings() == []

    def test_critical_temp_low(self) -> None:
        tank = TankEnvironment(temperature=5.0)
        warnings = tank.get_warnings()
        assert any("CRITICAL" in w and "low" in w.lower() for w in warnings)

    def test_critical_temp_high(self) -> None:
        tank = TankEnvironment(temperature=42.0)
        warnings = tank.get_warnings()
        assert any("CRITICAL" in w and "high" in w.lower() for w in warnings)

    def test_suboptimal_temp_low(self) -> None:
        tank = TankEnvironment(temperature=18.0)
        warnings = tank.get_warnings()
        assert any("below optimal" in w.lower() for w in warnings)

    def test_suboptimal_temp_high(self) -> None:
        tank = TankEnvironment(temperature=30.0)
        warnings = tank.get_warnings()
        assert any("above optimal" in w.lower() for w in warnings)

    def test_critical_oxygen(self) -> None:
        tank = TankEnvironment(oxygen_level=0.05)
        warnings = tank.get_warnings()
        assert any("CRITICAL" in w and "oxygen" in w.lower() for w in warnings)

    def test_low_oxygen(self) -> None:
        tank = TankEnvironment(oxygen_level=0.2)
        warnings = tank.get_warnings()
        assert any("oxygen" in w.lower() and "low" in w.lower() for w in warnings)

    def test_critical_cleanliness(self) -> None:
        tank = TankEnvironment(cleanliness=0.03)
        warnings = tank.get_warnings()
        assert any("CRITICAL" in w and "filthy" in w.lower() for w in warnings)

    def test_needs_cleaning(self) -> None:
        tank = TankEnvironment(cleanliness=0.15)
        warnings = tank.get_warnings()
        assert any("cleaning" in w.lower() for w in warnings)

    def test_multiple_warnings(self) -> None:
        tank = TankEnvironment(temperature=5.0, oxygen_level=0.05, cleanliness=0.03)
        warnings = tank.get_warnings()
        assert len(warnings) == 3  # All critical


# ── Serialization ────────────────────────────────────────────────────────────


class TestSerialization:
    """Tests for to_dict / from_dict persistence."""

    def test_to_dict(self) -> None:
        tank = TankEnvironment(
            temperature=22.0,
            cleanliness=0.8,
            oxygen_level=0.9,
            water_level=0.5,
            environment_type=EnvironmentType.TERRARIUM,
        )
        d = tank.to_dict()
        assert d["temperature"] == 22.0
        assert d["cleanliness"] == 0.8
        assert d["oxygen_level"] == 0.9
        assert d["water_level"] == 0.5
        assert d["environment_type"] == "terrarium"
        assert "last_update" in d

    def test_from_dict(self) -> None:
        data = {
            "temperature": 22.0,
            "cleanliness": 0.8,
            "oxygen_level": 0.9,
            "water_level": 0.5,
            "environment_type": "terrarium",
            "last_update": "2026-02-25T12:00:00+00:00",
        }
        tank = TankEnvironment.from_dict(data)
        assert tank.temperature == 22.0
        assert tank.cleanliness == 0.8
        assert tank.oxygen_level == 0.9
        assert tank.water_level == 0.5
        assert tank.environment_type == EnvironmentType.TERRARIUM

    def test_roundtrip(self) -> None:
        original = TankEnvironment(
            temperature=26.5,
            cleanliness=0.7,
            oxygen_level=0.85,
            water_level=0.0,
            environment_type=EnvironmentType.TERRARIUM,
        )
        restored = TankEnvironment.from_dict(original.to_dict())
        assert restored.temperature == original.temperature
        assert restored.cleanliness == original.cleanliness
        assert restored.oxygen_level == original.oxygen_level
        assert restored.water_level == original.water_level
        assert restored.environment_type == original.environment_type

    def test_from_dict_missing_keys(self) -> None:
        """Missing keys use defaults."""
        tank = TankEnvironment.from_dict({})
        assert tank.temperature == 24.0
        assert tank.environment_type == EnvironmentType.AQUARIUM

    def test_from_dict_unknown_keys_ignored(self) -> None:
        tank = TankEnvironment.from_dict({"temperature": 25.0, "unknown_key": "value"})
        assert tank.temperature == 25.0

    def test_from_dict_partial(self) -> None:
        tank = TankEnvironment.from_dict({"temperature": 30.0, "cleanliness": 0.5})
        assert tank.temperature == 30.0
        assert tank.cleanliness == 0.5
        assert tank.oxygen_level == 1.0  # default
