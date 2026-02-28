"""FastAPI WebSocket server for UE5 and external client communication.

Provides:
- WebSocket endpoint ``/ws/brain`` for real-time bidirectional messaging.
- REST endpoints: ``GET /api/state``, ``GET /api/health``, ``POST /api/reset``.
- CORS middleware configured for local UE5 development.
- Per-channel event subscriptions and streaming LLM responses.
- Background simulation loop for headless ``--api`` mode.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from seaman_brain.api.actions import ActionDispatcher
from seaman_brain.api.protocol import (
    ActionResultMessage,
    BrainStateSnapshot,
    CreatureStateSnapshot,
    EventSeverity,
    NeedsSnapshot,
    StreamEndMessage,
    StreamStartMessage,
    StreamTokenMessage,
    SubscribedMessage,
    TankSnapshot,
    TraitsSnapshot,
    serialize_response,
)
from seaman_brain.api.streaming import (
    ALL_CHANNELS,
    EventBroadcaster,
    EventChannel,
)
from seaman_brain.behavior.events import EventSystem
from seaman_brain.behavior.mood import MoodEngine
from seaman_brain.config import SeamanConfig
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.clock import GameClock
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.death import DeathEngine
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine
from seaman_brain.personality.traits import TraitProfile, get_default_profile

logger = logging.getLogger(__name__)

_STREAM_OVERALL_TIMEOUT = 300.0  # max seconds for entire streaming response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_state_snapshot(manager: ConversationManager) -> dict[str, Any]:
    """Build a JSON-serialisable state snapshot from the conversation manager."""
    state = manager.creature_state
    if state is None:
        return {}
    return state.to_dict()


# ---------------------------------------------------------------------------
# BrainServer
# ---------------------------------------------------------------------------

class BrainServer:
    """Manages the FastAPI app, WebSocket clients, and background brain loop."""

    def __init__(
        self,
        config: SeamanConfig | None = None,
        manager: ConversationManager | None = None,
    ) -> None:
        self._config = config or SeamanConfig()
        self._manager = manager or ConversationManager(config=self._config)

        # EventBroadcaster replaces the old _clients list
        self._broadcaster = EventBroadcaster(
            broadcast_interval_ms=self._config.api.broadcast_interval_ms
        )

        # Request counter for stream correlation IDs
        self._request_counter = 0

        # Simulation engines — created during lifespan startup
        self._creature_state: CreatureState | None = None
        self._tank: TankEnvironment | None = None
        self._needs: CreatureNeeds | None = None
        self._needs_engine: NeedsEngine | None = None
        self._mood_engine: MoodEngine | None = None
        self._event_system: EventSystem | None = None
        self._evolution_engine: EvolutionEngine | None = None
        self._death_engine: DeathEngine | None = None
        self._clock: GameClock | None = None
        self._dispatcher: ActionDispatcher | None = None

        # Anti-spam tracking for simulation loop
        self._last_mood: str = ""
        self._last_urgent_warnings: set[str] = set()

        # Simulation background task
        self._simulation_task: asyncio.Task[None] | None = None

        self.app = self._create_app()

    @property
    def manager(self) -> ConversationManager:
        """The underlying ConversationManager."""
        return self._manager

    # -- FastAPI app creation -----------------------------------------------

    def _create_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):  # noqa: ARG001
            await self._manager.initialize()
            self._init_simulation_engines()
            self._broadcaster.start(self._build_brain_snapshot)
            self._simulation_task = asyncio.create_task(self._simulation_loop())
            logger.info("BrainServer started")
            yield
            # Shutdown
            if self._simulation_task is not None:
                self._simulation_task.cancel()
                try:
                    await self._simulation_task
                except asyncio.CancelledError:
                    pass
            self._broadcaster.stop()
            await self._manager.shutdown()
            logger.info("BrainServer stopped")

        app = FastAPI(
            title="Seaman Brain API",
            version="1.0.0",
            lifespan=lifespan,
        )

        api_cfg = self._config.api
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_cfg.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # -- REST endpoints -------------------------------------------------

        @app.get("/api/health")
        async def health() -> dict[str, Any]:
            return {
                "status": "ok",
                "initialized": self._manager.is_initialized,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        @app.get("/api/state")
        async def get_state() -> dict[str, Any]:
            return {
                "type": "state_update",
                "state": _build_state_snapshot(self._manager),
                "timestamp": datetime.now(UTC).isoformat(),
            }

        @app.post("/api/reset")
        async def reset_state() -> dict[str, Any]:
            if self._manager.creature_state is not None:
                new_state = CreatureState()
                self._manager._creature_state = new_state  # noqa: SLF001
                self._creature_state = new_state
                if self._dispatcher is not None:
                    self._dispatcher.creature_state = new_state
            return {
                "type": "state_update",
                "state": _build_state_snapshot(self._manager),
                "message": "Creature state reset to defaults.",
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # -- WebSocket endpoint ---------------------------------------------

        @app.websocket("/ws/brain")
        async def ws_brain(ws: WebSocket) -> None:
            await ws.accept()
            self._broadcaster.add_client(ws)
            logger.info(
                "WebSocket client connected (%d total)",
                self._broadcaster.client_count,
            )
            try:
                while True:
                    raw = await ws.receive_text()
                    await self._handle_ws_message(ws, raw)
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception:
                logger.exception("WebSocket error")
            finally:
                self._broadcaster.remove_client(ws)

        return app

    # -- Simulation engine initialization -----------------------------------

    def _init_simulation_engines(self) -> None:
        """Create simulation engines from config and manager state."""
        cfg = self._config
        self._creature_state = self._manager.creature_state
        self._tank = TankEnvironment.from_config(cfg.environment)
        self._needs = CreatureNeeds()
        self._needs_engine = NeedsEngine(
            config=cfg.needs, env_config=cfg.environment
        )
        self._mood_engine = MoodEngine()
        self._event_system = EventSystem()
        self._evolution_engine = EvolutionEngine(cfg.creature)
        self._death_engine = DeathEngine(
            needs_config=cfg.needs, env_config=cfg.environment
        )
        self._clock = GameClock()

        if self._creature_state is not None:
            self._dispatcher = ActionDispatcher(
                creature_state=self._creature_state,
                tank=self._tank,
                needs_config=cfg.needs,
                env_config=cfg.environment,
            )
            self._last_mood = self._creature_state.mood

    # -- BrainStateSnapshot builder -----------------------------------------

    def _build_brain_snapshot(self) -> BrainStateSnapshot | None:
        """Build a full BrainStateSnapshot from current simulation state."""
        cs = self._creature_state
        if cs is None:
            return None

        traits = self._manager.traits or TraitProfile()

        return BrainStateSnapshot(
            creature_state=CreatureStateSnapshot(
                stage=cs.stage.value,
                age=cs.age,
                interaction_count=cs.interaction_count,
                mood=cs.mood,
                trust_level=cs.trust_level,
                hunger=cs.hunger,
                health=cs.health,
                comfort=cs.comfort,
                last_fed=cs.last_fed.isoformat() if cs.last_fed else "",
                last_interaction=(
                    cs.last_interaction.isoformat() if cs.last_interaction else ""
                ),
                birth_time=(
                    cs.birth_time.isoformat() if cs.birth_time else ""
                ),
            ),
            needs=NeedsSnapshot(
                hunger=cs.hunger,
                comfort=cs.comfort,
                health=cs.health,
                stimulation=cs.comfort,  # proxy
            ),
            tank=TankSnapshot(
                temperature=self._tank.temperature if self._tank else 24.0,
                cleanliness=self._tank.cleanliness if self._tank else 1.0,
                oxygen_level=self._tank.oxygen_level if self._tank else 1.0,
                water_level=self._tank.water_level if self._tank else 1.0,
                environment_type=(
                    self._tank.environment_type.value if self._tank else "aquarium"
                ),
            ),
            mood=cs.mood,
            active_traits=TraitsSnapshot(
                cynicism=traits.cynicism,
                wit=traits.wit,
                patience=traits.patience,
                curiosity=traits.curiosity,
                warmth=traits.warmth,
                verbosity=traits.verbosity,
                formality=traits.formality,
                aggression=traits.aggression,
            ),
            current_stage=cs.stage.value,
        )

    # -- WebSocket message handling -----------------------------------------

    async def _handle_ws_message(self, ws: WebSocket, raw: str) -> None:
        """Parse an incoming JSON message and dispatch."""
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send_json({"type": "error", "message": "Invalid JSON"})
            return

        msg_type = msg.get("type")
        if msg_type == "input":
            await self._handle_input(ws, msg)
        elif msg_type == "subscribe":
            await self._handle_subscribe(ws, msg)
        elif msg_type == "unsubscribe":
            await self._handle_unsubscribe(ws, msg)
        elif msg_type == "action":
            await self._handle_action(ws, msg)
        else:
            await ws.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            })

    async def _handle_input(self, ws: WebSocket, msg: dict[str, Any]) -> None:
        """Handle an input message with streaming response."""
        text = msg.get("text", "")
        if not text:
            await ws.send_json(
                {"type": "error", "message": "Missing 'text' field"}
            )
            return

        request_id = str(self._request_counter)
        self._request_counter += 1

        # Send stream_start
        start_msg = StreamStartMessage(request_id=request_id)
        try:
            await ws.send_text(serialize_response(start_msg))
        except Exception:
            return

        # Stream tokens (with overall timeout as defense-in-depth)
        accumulated: list[str] = []
        ws_failed = False
        try:
            async with asyncio.timeout(_STREAM_OVERALL_TIMEOUT):
                async for token in self._manager.process_input_stream(text):
                    accumulated.append(token)
                    if not ws_failed:
                        token_msg = StreamTokenMessage(
                            token=token, request_id=request_id
                        )
                        try:
                            await ws.send_text(serialize_response(token_msg))
                        except Exception:
                            # Client disconnected mid-stream — keep consuming
                            # so post-stream steps (memory, save) still run
                            ws_failed = True
        except TimeoutError:
            logger.warning(
                "Stream overall timeout (%.0fs) for request %s",
                _STREAM_OVERALL_TIMEOUT,
                request_id,
            )
            try:
                await ws.send_json({
                    "type": "error",
                    "message": "Response generation timed out",
                    "request_id": request_id,
                })
            except Exception:
                pass
            return
        except RuntimeError as exc:
            try:
                await ws.send_json({"type": "error", "message": str(exc)})
            except Exception:
                pass
            return

        if ws_failed:
            return

        # Send stream_end with full text and state snapshot
        full_text = "".join(accumulated)
        snapshot = self._build_brain_snapshot() or BrainStateSnapshot()
        end_msg = StreamEndMessage(
            text=full_text,
            request_id=request_id,
            state=snapshot,
        )
        try:
            await ws.send_text(serialize_response(end_msg))
        except Exception:
            pass

    async def _handle_subscribe(
        self, ws: WebSocket, msg: dict[str, Any]
    ) -> None:
        """Handle a subscribe message."""
        channels = msg.get("channels", [])
        if not isinstance(channels, list) or not channels:
            await ws.send_json({
                "type": "error",
                "message": "Missing or empty 'channels' list",
            })
            return

        # Validate channel names
        invalid = [ch for ch in channels if ch not in ALL_CHANNELS]
        if invalid:
            await ws.send_json({
                "type": "error",
                "message": f"Unknown channels: {invalid}",
            })
            return

        sub = self._broadcaster.get_subscription(ws)
        if sub is None:
            await ws.send_json({
                "type": "error", "message": "Client not registered"
            })
            return

        for ch in channels:
            sub.subscribe(ch)

        confirmed = SubscribedMessage(channels=sorted(sub.channels))
        await ws.send_text(serialize_response(confirmed))

    async def _handle_unsubscribe(
        self, ws: WebSocket, msg: dict[str, Any]
    ) -> None:
        """Handle an unsubscribe message."""
        channels = msg.get("channels", [])
        if not isinstance(channels, list) or not channels:
            await ws.send_json({
                "type": "error",
                "message": "Missing or empty 'channels' list",
            })
            return

        sub = self._broadcaster.get_subscription(ws)
        if sub is None:
            await ws.send_json({
                "type": "error", "message": "Client not registered"
            })
            return

        for ch in channels:
            sub.unsubscribe(ch)

        confirmed = SubscribedMessage(channels=sorted(sub.channels))
        await ws.send_text(serialize_response(confirmed))

    async def _handle_action(
        self, ws: WebSocket, msg: dict[str, Any]
    ) -> None:
        """Handle an action message."""
        action = msg.get("action", "")
        if not action:
            await ws.send_json({
                "type": "error", "message": "Missing 'action' field"
            })
            return

        params = msg.get("params", {})
        request_id = msg.get("request_id", "")

        if self._dispatcher is None:
            await ws.send_json({
                "type": "error",
                "message": "Server not initialized for actions",
            })
            return

        result = self._dispatcher.dispatch(action, params)
        snapshot = self._build_brain_snapshot() or BrainStateSnapshot()

        result_msg = ActionResultMessage(
            action=result.action,
            success=result.success,
            message=result.message,
            request_id=request_id,
            state=snapshot,
        )
        await ws.send_text(serialize_response(result_msg))

    # -- Background simulation loop -----------------------------------------

    async def _simulation_loop(self) -> None:
        """Periodic simulation tick mirroring GUI game_loop for headless mode."""
        interval = 1.0  # 1 second ticks
        while True:
            await asyncio.sleep(interval)
            try:
                await self._simulation_tick(interval)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in simulation loop")

    async def _simulation_tick(self, elapsed: float) -> None:
        """Execute a single simulation tick."""
        if self._creature_state is None or self._tank is None:
            return
        if self._needs_engine is None or self._death_engine is None:
            return

        # 1. Tank degradation
        self._tank.update(elapsed, self._config.environment)

        # 2. Needs update
        needs = self._needs_engine.update(
            elapsed, self._creature_state, self._tank
        )
        self._needs_engine.apply_to_state(self._creature_state, needs)
        self._needs = needs

        # 3. Age increment
        self._creature_state.age += elapsed

        # 4. Death check
        cause = self._death_engine.check_death(
            self._creature_state, needs, self._tank
        )
        if cause is not None:
            new_state, record = self._death_engine.on_death(
                cause, self._creature_state
            )
            self._creature_state = new_state
            self._manager._creature_state = new_state  # noqa: SLF001
            if self._dispatcher is not None:
                self._dispatcher.creature_state = new_state
            await self._broadcaster.broadcast_event(
                event_type="death",
                message=record.message,
                severity=EventSeverity.CRITICAL,
                effects={"cause": cause.value},
                channel=EventChannel.DEATH,
            )
            return  # Skip remaining checks after death

        # 5. Mood calculation
        if self._mood_engine is not None and self._clock is not None:
            traits = self._manager.traits or get_default_profile(
                self._creature_state.stage
            )
            mood = self._mood_engine.calculate_mood(
                needs=needs,
                trust=self._creature_state.trust_level,
                time_context=self._clock.get_time_context(),
                recent_interactions=self._creature_state.interaction_count,
                traits=traits,
            )
            mood_str = mood.value
            if mood_str != self._last_mood:
                self._creature_state.mood = mood_str
                self._last_mood = mood_str
                await self._broadcaster.broadcast_event(
                    event_type="mood_change",
                    message=f"Mood shifted to {mood_str}",
                    severity=EventSeverity.INFO,
                    channel=EventChannel.MOOD,
                )

        # 6. Urgent needs
        if self._needs_engine is not None:
            warnings = self._needs_engine.get_urgent_needs(needs)
            new_warnings = set(warnings) - self._last_urgent_warnings
            for warning in new_warnings:
                await self._broadcaster.broadcast_event(
                    event_type="urgent_need",
                    message=warning,
                    severity=EventSeverity.WARNING,
                    channel=EventChannel.NEEDS,
                )
            self._last_urgent_warnings = set(warnings)

        # 7. Event system checks
        if self._event_system is not None and self._clock is not None:
            time_ctx = self._clock.get_time_context()
            fired = self._event_system.check_events(
                self._creature_state, self._tank, time_ctx
            )
            for event in fired:
                severity = EventSeverity.INFO
                if event.priority >= 0.8:
                    severity = EventSeverity.WARNING
                if event.priority >= 0.9:
                    severity = EventSeverity.CRITICAL
                self._event_system.apply_effects(
                    event, self._creature_state, self._tank
                )
                await self._broadcaster.broadcast_event(
                    event_type=event.event_type.value,
                    message=event.message,
                    severity=severity,
                    effects={
                        "mood_change": event.effects.mood_change,
                        "trust_change": event.effects.trust_change,
                        "trigger_dialogue": event.effects.trigger_dialogue,
                    },
                )

        # 8. Evolution check
        if self._evolution_engine is not None:
            new_stage = self._evolution_engine.check_evolution(
                self._creature_state
            )
            if new_stage is not None:
                try:
                    self._evolution_engine.evolve(
                        self._creature_state, new_stage
                    )
                    await self._broadcaster.broadcast_event(
                        event_type="evolution",
                        message=f"Evolved to {new_stage.value}!",
                        severity=EventSeverity.WARNING,
                        effects={"new_stage": new_stage.value},
                        channel=EventChannel.EVOLUTION,
                    )
                except ValueError as exc:
                    logger.warning("Evolution failed in sim loop: %s", exc)

    # -- Convenience runner -------------------------------------------------

    def run(self) -> None:
        """Start the server with uvicorn (blocking)."""
        import uvicorn

        uvicorn.run(
            self.app,
            host=self._config.api.host,
            port=self._config.api.port,
            log_level="info",
        )
