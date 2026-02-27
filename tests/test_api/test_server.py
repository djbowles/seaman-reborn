"""Tests for the FastAPI WebSocket server (US-043).

Covers:
- REST endpoints: /api/health, /api/state, /api/reset
- WebSocket connection and message exchange
- Streaming input: stream_start -> stream_token -> stream_end
- Subscribe/unsubscribe messages and confirmation
- Action messages for each action type
- Error handling for invalid messages and missing fields
- Client tracking via EventBroadcaster
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


def _mock_process_input_stream(response: str):
    """Create a mock async generator for process_input_stream."""
    async def _stream(text: str) -> AsyncIterator[str]:
        yield response
    return _stream


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
# WebSocket tests — basic input (now streaming)
# ---------------------------------------------------------------------------

class TestWebSocketStreamingInput:
    """WebSocket /ws/brain streaming input handler."""

    def test_streaming_input_sequence(self) -> None:
        """Send input message, receive stream_start → stream_token → stream_end."""
        server = _make_server(llm_response="Go away.")
        server._manager._initialized = True  # noqa: SLF001
        server._manager.process_input_stream = _mock_process_input_stream("Go away.")

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input", "text": "Hello"}))

                # stream_start
                data = ws.receive_json()
                assert data["type"] == "stream_start"
                assert "request_id" in data

                # stream_token (mock yields one token)
                data = ws.receive_json()
                assert data["type"] == "stream_token"
                assert data["token"] == "Go away."

                # stream_end
                data = ws.receive_json()
                assert data["type"] == "stream_end"
                assert data["text"] == "Go away."
                assert "state" in data

    def test_streaming_request_id_consistent(self) -> None:
        """All stream messages share the same request_id."""
        server = _make_server(llm_response="Fine.")
        server._manager._initialized = True  # noqa: SLF001
        server._manager.process_input_stream = _mock_process_input_stream("Fine.")

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input", "text": "Hey"}))

                start = ws.receive_json()
                token = ws.receive_json()
                end = ws.receive_json()

                rid = start["request_id"]
                assert token["request_id"] == rid
                assert end["request_id"] == rid


class TestWebSocketErrors:
    """WebSocket error handling."""

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
        """RuntimeError from process_input_stream is forwarded as error."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001

        async def _failing_stream(text):
            raise RuntimeError("No LLM available")
            yield  # pragma: no cover — make it an async generator

        server._manager.process_input_stream = _failing_stream
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "input", "text": "Hello"}))
                # stream_start is sent before the generator is consumed
                data1 = ws.receive_json()
                # The error comes after stream_start (RuntimeError during iteration)
                data2 = ws.receive_json()
                types = {data1["type"], data2["type"]}
                assert "error" in types


# ---------------------------------------------------------------------------
# Subscribe / Unsubscribe tests
# ---------------------------------------------------------------------------

class TestWebSocketSubscribe:
    """WebSocket subscribe/unsubscribe handlers."""

    def test_subscribe_returns_confirmation(self) -> None:
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "subscribe",
                    "channels": ["mood", "evolution"],
                }))
                data = ws.receive_json()
                assert data["type"] == "subscribed"
                assert "mood" in data["channels"]
                assert "evolution" in data["channels"]

    def test_unsubscribe_returns_updated_list(self) -> None:
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                # First unsubscribe from some channels
                ws.send_text(json.dumps({
                    "type": "unsubscribe",
                    "channels": ["mood", "death"],
                }))
                data = ws.receive_json()
                assert data["type"] == "subscribed"
                # mood and death should no longer be in the list
                assert "mood" not in data["channels"]
                assert "death" not in data["channels"]

    def test_subscribe_invalid_channel_error(self) -> None:
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "subscribe",
                    "channels": ["unicorn"],
                }))
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Unknown channels" in data["message"]

    def test_subscribe_empty_channels_error(self) -> None:
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "subscribe",
                    "channels": [],
                }))
                data = ws.receive_json()
                assert data["type"] == "error"


# ---------------------------------------------------------------------------
# Action message tests
# ---------------------------------------------------------------------------

class TestWebSocketAction:
    """WebSocket action handler."""

    def _make_action_server(self) -> BrainServer:
        """Build a server with dispatcher initialized."""
        state = CreatureState(hunger=0.5)
        server = _make_server(creature_state=state)
        server._manager._initialized = True  # noqa: SLF001
        # Initialize simulation engines so dispatcher exists
        server._creature_state = state
        server._init_simulation_engines()
        return server

    def test_action_feed(self) -> None:
        server = self._make_action_server()
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "action",
                    "action": "feed",
                    "params": {"food_type": "nautilus"},
                    "request_id": "r1",
                }))
                data = ws.receive_json()
                assert data["type"] == "action_result"
                assert data["action"] == "feed"
                assert data["request_id"] == "r1"
                assert isinstance(data["success"], bool)

    def test_action_tap_glass(self) -> None:
        server = self._make_action_server()
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "action",
                    "action": "tap_glass",
                }))
                data = ws.receive_json()
                assert data["type"] == "action_result"
                assert data["success"] is True
                assert data["action"] == "tap_glass"

    def test_action_clean(self) -> None:
        server = self._make_action_server()
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "action",
                    "action": "clean",
                }))
                data = ws.receive_json()
                assert data["type"] == "action_result"
                assert data["success"] is True

    def test_action_adjust_temperature(self) -> None:
        server = self._make_action_server()
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "action",
                    "action": "adjust_temperature",
                    "params": {"delta": 2.0},
                }))
                data = ws.receive_json()
                assert data["type"] == "action_result"
                assert data["success"] is True

    def test_action_missing_action_field(self) -> None:
        server = self._make_action_server()
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({"type": "action"}))
                data = ws.receive_json()
                assert data["type"] == "error"
                assert "Missing 'action'" in data["message"]

    def test_action_unknown_returns_failure(self) -> None:
        server = self._make_action_server()
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "action",
                    "action": "explode",
                }))
                data = ws.receive_json()
                assert data["type"] == "action_result"
                assert data["success"] is False

    def test_action_without_dispatcher_error(self) -> None:
        """Action with dispatcher removed returns error."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            # Lifespan has run; now remove dispatcher to simulate uninitialized
            server._dispatcher = None
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_text(json.dumps({
                    "type": "action",
                    "action": "feed",
                }))
                data = ws.receive_json()
                assert data["type"] == "error"


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
# Client tracking tests via EventBroadcaster
# ---------------------------------------------------------------------------

class TestClientTracking:
    """Verify WebSocket client tracking via EventBroadcaster."""

    def test_client_added_and_removed(self) -> None:
        """Client is tracked on connect and removed on disconnect."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        assert server._broadcaster.client_count == 0
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain"):
                assert server._broadcaster.client_count == 1
        # After disconnect the client is cleaned up
        assert server._broadcaster.client_count == 0

    def test_multiple_clients(self) -> None:
        """Multiple WebSocket clients are tracked concurrently."""
        server = _make_server()
        server._manager._initialized = True  # noqa: SLF001
        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain"):
                assert server._broadcaster.client_count == 1
                with client.websocket_connect("/ws/brain"):
                    assert server._broadcaster.client_count == 2
                # Inner client disconnected
                assert server._broadcaster.client_count == 1
        assert server._broadcaster.client_count == 0


# ---------------------------------------------------------------------------
# BrainStateSnapshot builder test
# ---------------------------------------------------------------------------

class TestBuildBrainSnapshot:
    """Tests for _build_brain_snapshot."""

    def test_returns_none_without_creature(self) -> None:
        server = _make_server()
        # creature_state is not set on server (only on manager)
        assert server._build_brain_snapshot() is None

    def test_returns_snapshot_after_init(self) -> None:
        server = _make_server()
        server._creature_state = server._manager._creature_state  # noqa: SLF001
        server._init_simulation_engines()
        snap = server._build_brain_snapshot()
        assert snap is not None
        assert snap.creature_state.stage == "mushroomer"
        assert snap.tank.temperature == 24.0
