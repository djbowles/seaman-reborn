"""FastAPI WebSocket server for UE5 and external client communication.

Provides:
- WebSocket endpoint ``/ws/brain`` for real-time bidirectional messaging.
- REST endpoints: ``GET /api/state``, ``GET /api/health``, ``POST /api/reset``.
- CORS middleware configured for local UE5 development.
- Unsolicited state-update pushes on mood/need/evolution changes.
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

from seaman_brain.config import SeamanConfig
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.state import CreatureState

logger = logging.getLogger(__name__)


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
        self._clients: list[WebSocket] = []
        self._last_snapshot: dict[str, Any] = {}
        self._broadcast_task: asyncio.Task[None] | None = None
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
            self._last_snapshot = _build_state_snapshot(self._manager)
            self._broadcast_task = asyncio.create_task(self._broadcast_loop())
            logger.info("BrainServer started")
            yield
            if self._broadcast_task is not None:
                self._broadcast_task.cancel()
                try:
                    await self._broadcast_task
                except asyncio.CancelledError:
                    pass
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
            self._clients.append(ws)
            logger.info(
                "WebSocket client connected (%d total)", len(self._clients)
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
                if ws in self._clients:
                    self._clients.remove(ws)

        return app

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
            text = msg.get("text", "")
            if not text:
                await ws.send_json(
                    {"type": "error", "message": "Missing 'text' field"}
                )
                return
            try:
                response = await self._manager.process_input(text)
            except RuntimeError as exc:
                await ws.send_json({"type": "error", "message": str(exc)})
                return

            await ws.send_json({
                "type": "response",
                "text": response,
                "state": _build_state_snapshot(self._manager),
                "timestamp": datetime.now(UTC).isoformat(),
            })
        else:
            await ws.send_json({
                "type": "error",
                "message": f"Unknown message type: {msg_type}",
            })

    # -- Background state broadcasting --------------------------------------

    async def _broadcast_loop(self) -> None:
        """Periodically push state diffs to all connected WebSocket clients."""
        interval = self._config.api.broadcast_interval_ms / 1000.0
        while True:
            await asyncio.sleep(interval)
            snapshot = _build_state_snapshot(self._manager)
            if snapshot != self._last_snapshot and self._clients:
                self._last_snapshot = snapshot
                payload = json.dumps({
                    "type": "state_update",
                    "state": snapshot,
                    "timestamp": datetime.now(UTC).isoformat(),
                })
                disconnected: list[WebSocket] = []
                for client in self._clients:
                    try:
                        await client.send_text(payload)
                    except Exception:
                        disconnected.append(client)
                for client in disconnected:
                    if client in self._clients:
                        self._clients.remove(client)

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
