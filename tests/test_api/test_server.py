"""Tests for the FastAPI WebSocket server (US-043).

Covers:
- REST endpoints: /api/health, /api/state, /api/reset
- WebSocket connection and message exchange
- Error handling for invalid messages and missing fields
- State update broadcasting
- Graceful client disconnection
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from seaman_brain.api.server import BrainServer, _build_state_snapshot
from seaman_brain.config import SeamanConfig
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.state import CreatureState
from seaman_brain.types import ChatMessage, CreatureStage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """Minimal LLM mock for test conversations."""

    def __init__(self, response: str = "Whatever, human.") -> None:
        self._response = response
        self.chat = AsyncMock(side_effect=self._chat)

    async def _chat(self, messages: list[ChatMessage]) -> str:
        return self._response

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        yield self._response


def _make_server(
    *,
    creature_state: CreatureState | None = None,
    llm_response: str = "Whatever, human.",
) -> BrainServer:
    """Build a BrainServer with a fully-mocked ConversationManager."""
    config = SeamanConfig()
    state = creature_state or CreatureState()
    llm = MockLLM(llm_response)
    manager = ConversationManager(config=config, llm=llm, creature_state=state)
    return BrainServer(config=config, manager=manager)


# ---------------------------------------------------------------------------
# REST endpoint tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    """GET /api/health."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self) -> None:
        server = _make_server()
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_health_reports_initialized_false_before_lifespan(self) -> None:
        """Before the lifespan runs, initialized should be False."""
        server = _make_server()
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/health")
        body = resp.json()
        # Manager is not initialized without lifespan context
        assert body["initialized"] is False


class TestStateEndpoint:
    """GET /api/state."""

    @pytest.mark.asyncio
    async def test_state_returns_creature_state(self) -> None:
        state = CreatureState(stage=CreatureStage.GILLMAN, mood="sardonic")
        server = _make_server(creature_state=state)
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["type"] == "state_update"
        assert body["state"]["stage"] == "gillman"
        assert body["state"]["mood"] == "sardonic"
        assert "timestamp" in body

    @pytest.mark.asyncio
    async def test_state_empty_when_no_creature(self) -> None:
        """If creature_state is None, state dict is empty."""
        server = _make_server()
        # Manually set manager's creature_state to None
        server._manager._creature_state = None  # noqa: SLF001
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.get("/api/state")
        assert resp.status_code == 200
        assert resp.json()["state"] == {}


class TestResetEndpoint:
    """POST /api/reset."""

    @pytest.mark.asyncio
    async def test_reset_restores_defaults(self) -> None:
        state = CreatureState(
            stage=CreatureStage.FROGMAN,
            trust_level=0.9,
            interaction_count=500,
        )
        server = _make_server(creature_state=state)
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/reset")
        assert resp.status_code == 200
        body = resp.json()
        assert body["state"]["stage"] == "mushroomer"
        assert body["state"]["trust_level"] == 0.0
        assert body["state"]["interaction_count"] == 0
        assert body["message"] == "Creature state reset to defaults."

    @pytest.mark.asyncio
    async def test_reset_when_no_creature(self) -> None:
        server = _make_server()
        server._manager._creature_state = None  # noqa: SLF001
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.post("/api/reset")
        assert resp.status_code == 200
        assert resp.json()["state"] == {}


# ---------------------------------------------------------------------------
# WebSocket tests
# ---------------------------------------------------------------------------

class TestWebSocketBrain:
    """WebSocket /ws/brain endpoint."""

    def test_websocket_input_response(self) -> None:
        """Send input message, receive response with state."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        server._manager.process_input = AsyncMock(return_value="Go away.")
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input", "text": "Hello"}))
                data = ws.receive_json()
                assert data["type"] == "response"
                assert data["text"] == "Go away."
                assert "state" in data
                assert "timestamp" in data

    def test_websocket_invalid_json(self) -> None:
        """Non-JSON text gets an error response."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text("not json")
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Invalid JSON" in data["message"]

    def test_websocket_unknown_type(self) -> None:
        """Unknown message type gets an error response."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "ping"}))
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Unknown message type" in data["message"]

    def test_websocket_missing_text_field(self) -> None:
        """Input message without text field gets an error."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input"}))
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Missing 'text' field" in data["message"]

    def test_websocket_empty_text(self) -> None:
        """Empty text string in input message gets an error."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input", "text": ""}))
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Missing 'text' field" in data["message"]

    def test_websocket_process_input_error(self) -> None:
        """RuntimeError from process_input is forwarded as error message."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        server._manager.process_input = AsyncMock(
            side_effect=RuntimeError("No LLM available")
        )
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input", "text": "Hello"}))
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "No LLM available" in data["message"]


# ---------------------------------------------------------------------------
# Helper / utility tests
# ---------------------------------------------------------------------------

class TestBuildStateSnapshot:
    """_build_state_snapshot helper."""

    def test_snapshot_from_manager_with_state(self) -> None:
        state = CreatureState(stage=CreatureStage.PODFISH, hunger=0.5)
        server = _make_server(creature_state=state)
        snap = _build_state_snapshot(server.manager)
        assert snap["stage"] == "podfish"
        assert snap["hunger"] == 0.5

    def test_snapshot_empty_when_no_state(self) -> None:
        server = _make_server()
        server._manager._creature_state = None  # noqa: SLF001
        snap = _build_state_snapshot(server.manager)
        assert snap == {}


class TestBrainServerInit:
    """BrainServer construction and properties."""

    def test_default_construction(self) -> None:
        server = BrainServer()
        assert server.app is not None
        assert server.manager is not None

    def test_custom_config(self) -> None:
        cfg = SeamanConfig()
        server = BrainServer(config=cfg)
        assert server._config is cfg  # noqa: SLF001

    def test_custom_manager(self) -> None:
        mgr = ConversationManager()
        server = BrainServer(manager=mgr)
        assert server.manager is mgr


# ---------------------------------------------------------------------------
# CORS configuration test
# ---------------------------------------------------------------------------

class TestCORSConfiguration:
    """Verify CORS middleware is applied."""

    @pytest.mark.asyncio
    async def test_cors_headers_on_preflight(self) -> None:
        server = _make_server()
        transport = ASGITransport(app=server.app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            resp = await ac.options(
                "/api/health",
                headers={
                    "origin": "http://localhost:3000",
                    "access-control-request-method": "GET",
                },
            )
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers


# ---------------------------------------------------------------------------
# Client tracking tests
# ---------------------------------------------------------------------------

class TestClientTracking:
    """Verify WebSocket client list management."""

    def test_client_added_and_removed(self) -> None:
        """Client is tracked on connect and removed on disconnect."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        assert len(server._clients) == 0
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain"):
                assert len(server._clients) == 1
        # After disconnect the client is cleaned up
        assert len(server._clients) == 0

    def test_multiple_clients(self) -> None:
        """Multiple WebSocket clients are tracked concurrently."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain"):
                assert len(server._clients) == 1
                with client.websocket_connect("/ws/brain"):
                    assert len(server._clients) == 2
                # Inner client disconnected
                assert len(server._clients) == 1
        assert len(server._clients) == 0
