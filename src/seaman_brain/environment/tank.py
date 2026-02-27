"""Tank environment state - temperature, cleanliness, oxygen, water level.

Manages the creature's habitat conditions. The tank degrades over time (cleanliness
and oxygen decay), requiring player maintenance. Supports two environment types:
aquarium (water) and terrarium (land), with drain() transitioning between them
during the Podfish->Tadman evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from seaman_brain.config import EnvironmentConfig


class EnvironmentType(Enum):
    """Type of creature habitat."""

    AQUARIUM = "aquarium"
    TERRARIUM = "terrarium"


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a float to [min_val, max_val]."""
    return max(min_val, min(max_val, value))


@dataclass
class TankEnvironment:
    """Mutable state of the creature's tank/habitat.

    Fields:
        temperature: Current temperature in degrees Celsius.
        cleanliness: Cleanliness level (0.0=filthy, 1.0=spotless).
        oxygen_level: Oxygen level (0.0=none, 1.0=saturated).
        water_level: Water level (0.0=empty, 1.0=full). Drops to 0 in terrarium.
        environment_type: Current habitat type (aquarium or terrarium).
        last_update: Timestamp of last update() call.
    """

    temperature: float = 24.0
    cleanliness: float = 1.0
    oxygen_level: float = 1.0
    water_level: float = 1.0
    environment_type: EnvironmentType = EnvironmentType.AQUARIUM
    last_update: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        """Clamp fields to valid ranges."""
        self.cleanliness = _clamp(self.cleanliness)
        self.oxygen_level = _clamp(self.oxygen_level)
        self.water_level = _clamp(self.water_level)
        if isinstance(self.environment_type, str):
            self.environment_type = EnvironmentType(self.environment_type)

    def update(
        self,
        elapsed_seconds: float,
        config: EnvironmentConfig | None = None,
    ) -> None:
        """Degrade tank conditions over time.

        Cleanliness and oxygen decay at configurable rates per second.
        In terrarium mode, oxygen decays slower (natural air).

        Args:
            elapsed_seconds: Seconds since last update.
            config: Environment config for decay rates. Uses defaults if None.
        """
        if elapsed_seconds <= 0:
            return

        cfg = config or EnvironmentConfig()

        cleanliness_rate = cfg.cleanliness_decay_rate
        oxygen_rate = cfg.oxygen_decay_rate

        # Terrarium has natural air circulation — oxygen decays at half rate
        if self.environment_type == EnvironmentType.TERRARIUM:
            oxygen_rate *= 0.5

        self.cleanliness = _clamp(self.cleanliness - cleanliness_rate * elapsed_seconds)
        self.oxygen_level = _clamp(self.oxygen_level - oxygen_rate * elapsed_seconds)
        self.last_update = datetime.now(UTC)

    def set_temperature(
        self,
        value: float,
        config: EnvironmentConfig | None = None,
    ) -> None:
        """Set tank temperature within bounds.

        Temperature is clamped to [lethal_min - 5, lethal_max + 5] to allow
        reaching dangerous-but-not-impossible levels.

        Args:
            value: Desired temperature in Celsius.
            config: Environment config for temp bounds. Uses defaults if None.
        """
        cfg = config or EnvironmentConfig()
        min_temp = cfg.lethal_temp_min - 5.0
        max_temp = cfg.lethal_temp_max + 5.0
        self.temperature = max(min_temp, min(max_temp, value))

    def adjust_temperature(
        self,
        delta: float,
        config: EnvironmentConfig | None = None,
    ) -> None:
        """Adjust temperature by a delta amount.

        Args:
            delta: Temperature change in Celsius (positive = warmer).
            config: Environment config for temp bounds. Uses defaults if None.
        """
        self.set_temperature(self.temperature + delta, config)

    def clean(self) -> None:
        """Clean the tank, restoring cleanliness to 1.0."""
        self.cleanliness = 1.0

    def drain(self) -> bool:
        """Drain the tank, transitioning from aquarium to terrarium.

        Required for the Podfish->Tadman evolution stage transition.

        Returns:
            True if drain succeeded, False if already a terrarium.
        """
        if self.environment_type == EnvironmentType.TERRARIUM:
            return False

        self.environment_type = EnvironmentType.TERRARIUM
        self.water_level = 0.0
        return True

    def fill(self) -> bool:
        """Fill the tank, transitioning from terrarium to aquarium.

        Returns:
            True if fill succeeded, False if already an aquarium.
        """
        if self.environment_type == EnvironmentType.AQUARIUM:
            return False

        self.environment_type = EnvironmentType.AQUARIUM
        self.water_level = 1.0
        return True

    def is_habitable(self, config: EnvironmentConfig | None = None) -> bool:
        """Check if the tank conditions are survivable.

        Conditions must all be met:
        - Temperature within lethal bounds
        - Oxygen above 0.1
        - Cleanliness above 0.05

        Args:
            config: Environment config for temp bounds. Uses defaults if None.

        Returns:
            True if the tank is habitable.
        """
        cfg = config or EnvironmentConfig()

        if self.temperature < cfg.lethal_temp_min or self.temperature > cfg.lethal_temp_max:
            return False

        if self.oxygen_level < 0.1:
            return False

        if self.cleanliness < 0.05:
            return False

        return True

    def is_temperature_optimal(self, config: EnvironmentConfig | None = None) -> bool:
        """Check if temperature is in the optimal range.

        Args:
            config: Environment config for optimal temp range. Uses defaults if None.

        Returns:
            True if temperature is in optimal range.
        """
        cfg = config or EnvironmentConfig()
        return cfg.optimal_temp_min <= self.temperature <= cfg.optimal_temp_max

    def get_warnings(self, config: EnvironmentConfig | None = None) -> list[str]:
        """Get list of urgent tank maintenance warnings.

        Args:
            config: Environment config for thresholds. Uses defaults if None.

        Returns:
            List of warning strings (empty if all conditions OK).
        """
        cfg = config or EnvironmentConfig()
        warnings: list[str] = []

        # Temperature warnings
        if self.temperature < cfg.lethal_temp_min:
            warnings.append("CRITICAL: Temperature dangerously low!")
        elif self.temperature < cfg.optimal_temp_min:
            warnings.append("Temperature below optimal range.")

        if self.temperature > cfg.lethal_temp_max:
            warnings.append("CRITICAL: Temperature dangerously high!")
        elif self.temperature > cfg.optimal_temp_max:
            warnings.append("Temperature above optimal range.")

        # Oxygen warnings
        if self.oxygen_level < 0.1:
            warnings.append("CRITICAL: Oxygen level critically low!")
        elif self.oxygen_level < 0.3:
            warnings.append("Oxygen level is getting low.")

        # Cleanliness warnings
        if self.cleanliness < 0.05:
            warnings.append("CRITICAL: Tank is dangerously filthy!")
        elif self.cleanliness < 0.2:
            warnings.append("Tank needs cleaning.")

        return warnings

    def to_dict(self) -> dict[str, Any]:
        """Serialize tank state to a JSON-compatible dict."""
        return {
            "temperature": self.temperature,
            "cleanliness": self.cleanliness,
            "oxygen_level": self.oxygen_level,
            "water_level": self.water_level,
            "environment_type": self.environment_type.value,
            "last_update": self.last_update.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TankEnvironment:
        """Deserialize tank state from a dict.

        Unknown keys are silently ignored.
        Missing keys use defaults.
        """
        kwargs: dict[str, Any] = {}

        for float_key in ("temperature", "cleanliness", "oxygen_level", "water_level"):
            if float_key in data:
                kwargs[float_key] = float(data[float_key])

        if "environment_type" in data:
            kwargs["environment_type"] = EnvironmentType(data["environment_type"])

        if "last_update" in data:
            kwargs["last_update"] = datetime.fromisoformat(data["last_update"])

        return cls(**kwargs)

    @classmethod
    def from_config(cls, config: EnvironmentConfig) -> TankEnvironment:
        """Create a TankEnvironment from config defaults.

        Args:
            config: Environment configuration.

        Returns:
            TankEnvironment initialized from config values.
        """
        return cls(
            temperature=config.initial_temperature,
            environment_type=EnvironmentType(config.initial_environment),
        )
