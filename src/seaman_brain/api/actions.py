"""Action dispatcher — routes player actions to game engines.

Translates ``action`` WebSocket messages into calls to the existing
:class:`FeedingEngine` and :class:`TankCareEngine`, keeping the
:class:`BrainServer` as a thin transport layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from seaman_brain.config import EnvironmentConfig, NeedsConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.needs.care import TankCareEngine
from seaman_brain.needs.feeding import FeedingEngine, FoodType

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Outcome of a dispatched action.

    Attributes:
        success: Whether the action was accepted.
        message: Human-readable outcome description.
        action: The action name that was dispatched.
    """

    success: bool
    message: str
    action: str


class ActionDispatcher:
    """Routes player actions to the appropriate game engine.

    Supported actions: ``feed``, ``tap_glass``, ``adjust_temperature``,
    ``clean``, ``aerate``, ``drain``, ``fill``.

    Args:
        creature_state: Shared creature state reference.
        tank: Shared tank environment reference.
        needs_config: Configuration for feeding cooldowns / thresholds.
        env_config: Configuration for temperature bounds.
    """

    def __init__(
        self,
        creature_state: CreatureState,
        tank: TankEnvironment,
        needs_config: NeedsConfig | None = None,
        env_config: EnvironmentConfig | None = None,
    ) -> None:
        self._creature_state = creature_state
        self._tank = tank
        self._feeding = FeedingEngine(config=needs_config)
        self._care = TankCareEngine(
            env_config=env_config, needs_config=needs_config
        )

    @property
    def creature_state(self) -> CreatureState:
        """Current creature state reference."""
        return self._creature_state

    @creature_state.setter
    def creature_state(self, value: CreatureState) -> None:
        """Replace creature state (e.g. after death → rebirth)."""
        self._creature_state = value

    def dispatch(self, action: str, params: dict[str, Any] | None = None) -> ActionResult:
        """Dispatch an action to the appropriate handler.

        Args:
            action: Action name string.
            params: Optional parameter dict for the action.

        Returns:
            ActionResult with success/failure and descriptive message.
        """
        params = params or {}
        handlers = {
            "feed": self._handle_feed,
            "tap_glass": self._handle_tap_glass,
            "adjust_temperature": self._handle_adjust_temperature,
            "clean": self._handle_clean,
            "aerate": self._handle_aerate,
            "drain": self._handle_drain,
            "fill": self._handle_fill,
        }
        handler = handlers.get(action)
        if handler is None:
            return ActionResult(
                success=False,
                message=f"Unknown action: {action!r}",
                action=action,
            )
        try:
            return handler(params)
        except Exception as exc:
            logger.exception("Action '%s' failed", action)
            return ActionResult(
                success=False,
                message=f"Action failed: {exc}",
                action=action,
            )

    # -- Individual action handlers ------------------------------------------

    def _handle_feed(self, params: dict[str, Any]) -> ActionResult:
        raw_food = params.get("food_type", "")
        if raw_food:
            try:
                food = FoodType(raw_food)
            except ValueError:
                valid = [f.value for f in FoodType]
                return ActionResult(
                    success=False,
                    message=f"Invalid food_type {raw_food!r}. Valid: {valid}",
                    action="feed",
                )
        else:
            # Auto-pick first available food for the stage
            available = self._feeding.get_available_foods(
                self._creature_state.stage
            )
            if not available:
                return ActionResult(
                    success=False,
                    message="No food available for this stage.",
                    action="feed",
                )
            food = available[0]

        result = self._feeding.feed(self._creature_state, food)
        return ActionResult(
            success=result.success,
            message=result.message,
            action="feed",
        )

    def _handle_tap_glass(self, params: dict[str, Any]) -> ActionResult:
        self._creature_state.interaction_count += 1
        return ActionResult(
            success=True,
            message="You tap the glass. The creature glares at you.",
            action="tap_glass",
        )

    def _handle_adjust_temperature(self, params: dict[str, Any]) -> ActionResult:
        delta = float(params.get("delta", 1.0))
        result = self._care.adjust_temperature(
            self._tank, delta, self._creature_state
        )
        return ActionResult(
            success=result.success,
            message=result.message,
            action="adjust_temperature",
        )

    def _handle_clean(self, params: dict[str, Any]) -> ActionResult:
        result = self._care.clean_tank(self._tank)
        return ActionResult(
            success=result.success,
            message=result.message,
            action="clean",
        )

    def _handle_aerate(self, params: dict[str, Any]) -> ActionResult:
        # Auto-switch to sprinkle in terrarium mode
        if self._tank.environment_type == EnvironmentType.TERRARIUM:
            result = self._care.sprinkle(self._tank, self._creature_state)
        else:
            result = self._care.aerate_tank(self._tank)
        return ActionResult(
            success=result.success,
            message=result.message,
            action="aerate",
        )

    def _handle_drain(self, params: dict[str, Any]) -> ActionResult:
        result = self._care.drain_tank(self._tank, self._creature_state)
        return ActionResult(
            success=result.success,
            message=result.message,
            action="drain",
        )

    def _handle_fill(self, params: dict[str, Any]) -> ActionResult:
        result = self._care.fill_tank(self._tank, self._creature_state)
        return ActionResult(
            success=result.success,
            message=result.message,
            action="fill",
        )
