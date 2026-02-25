"""Full game integration tests (US-046).

Comprehensive end-to-end tests exercising the entire system: brain + needs +
events + evolution + death + API + save/load. Only external services (LLM,
Ollama embeddings) are mocked at the boundary.

Run with:  pytest -m integration tests/test_integration/test_full_game.py
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pytest
from starlette.testclient import TestClient

from seaman_brain.behavior.events import EventSystem, EventType
from seaman_brain.behavior.mood import CreatureMood, MoodEngine
from seaman_brain.config import (
    CreatureConfig,
    EnvironmentConfig,
    EvolutionThreshold,
    MemoryConfig,
    NeedsConfig,
    SeamanConfig,
)
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.persistence import StatePersistence
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.clock import GameClock
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.needs.death import DeathCause, DeathEngine
from seaman_brain.needs.feeding import FeedingEngine, FoodType
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine
from seaman_brain.types import ChatMessage, CreatureStage

# All tests in this file use the integration marker.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """Mock LLM provider returning canned responses."""

    def __init__(self, *responses: str) -> None:
        self._responses = list(responses) or ["Whatever."]
        self._index = 0
        self.chat = AsyncMock(side_effect=self._next_response)

    async def _next_response(self, messages: list[ChatMessage]) -> str:
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        yield response


def _fake_embed(text: str) -> list[float]:
    """Deterministic fake embedding based on hash of text."""
    rng = np.random.default_rng(hash(text) % (2**31))
    return rng.standard_normal(384).astype(np.float32).tolist()


def _make_config(
    save_path: str,
    *,
    evolution_thresholds: dict[str, EvolutionThreshold] | None = None,
    needs_config: NeedsConfig | None = None,
    env_config: EnvironmentConfig | None = None,
) -> SeamanConfig:
    """Build a SeamanConfig suitable for integration tests."""
    creature_cfg = CreatureConfig(
        save_path=save_path,
        evolution_thresholds=evolution_thresholds or {},
    )
    return SeamanConfig(
        creature=creature_cfg,
        memory=MemoryConfig(extraction_interval=100),
        needs=needs_config or NeedsConfig(),
        environment=env_config or EnvironmentConfig(),
    )


# ---------------------------------------------------------------------------
# 1. Full conversation with mocked LLM through GUI bridge (API WebSocket)
# ---------------------------------------------------------------------------

class TestConversationThroughGUIBridge:
    """Test full conversation pipeline via the API WebSocket bridge,
    simulating how UE5 or a GUI client would interact with the brain.
    """

    async def test_websocket_conversation_round_trip(self, tmp_path):
        """Client sends input via WebSocket, receives response with state."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("The water is cold and I don't care.")
        state = CreatureState()
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_json({"type": "input", "text": "Hello creature!"})
                resp = ws.receive_json()

                assert resp["type"] == "response"
                assert isinstance(resp["text"], str)
                assert len(resp["text"]) > 0
                assert "state" in resp
                assert resp["state"]["interaction_count"] == 1

    async def test_websocket_multiple_turns_state_updates(self, tmp_path):
        """Multiple WebSocket messages accumulate state correctly."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Response 1.", "Response 2.", "Response 3.")
        state = CreatureState()
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                for i in range(3):
                    ws.send_json({"type": "input", "text": f"Message {i + 1}"})
                    resp = ws.receive_json()
                    assert resp["type"] == "response"
                    assert resp["state"]["interaction_count"] == i + 1

                # Trust should have increased over 3 interactions
                assert resp["state"]["trust_level"] > 0.0

    async def test_rest_state_endpoint_reflects_conversation(self, tmp_path):
        """GET /api/state reflects state after WebSocket conversation."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Sure thing.")
        state = CreatureState(stage=CreatureStage.GILLMAN)
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            # Send a message via WebSocket
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_json({"type": "input", "text": "Hello"})
                ws.receive_json()

            # REST state should reflect the interaction
            resp = client.get("/api/state")
            body = resp.json()
            assert body["state"]["stage"] == "gillman"
            assert body["state"]["interaction_count"] == 1


# ---------------------------------------------------------------------------
# 2. Creature needs degrade over simulated time, feeding restores them
# ---------------------------------------------------------------------------

class TestNeedsDegradationAndFeeding:
    """Needs degrade over time, urgent warnings fire, and feeding restores."""

    def test_needs_degrade_over_simulated_time(self):
        """Hunger increases and comfort degrades over elapsed time."""
        engine = NeedsEngine()
        state = CreatureState(hunger=0.0, comfort=1.0, health=1.0)
        tank = TankEnvironment()

        # Simulate 1 hour (3600 seconds) elapsed
        needs = engine.update(3600.0, state, tank)

        # Hunger should increase: 0.02 rate * 0.5 (MUSHROOMER mult) * 3600 = 36.0 → clamped to 1.0
        assert needs.hunger > 0.0
        # Health may start degrading if hunger exceeds critical threshold
        engine.apply_to_state(state, needs)
        assert state.hunger > 0.0

    def test_feeding_restores_hunger(self):
        """Feeding reduces hunger and produces a success result."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        # Set last_fed far enough in the past to avoid cooldown
        state = CreatureState(
            hunger=0.5,
            stage=CreatureStage.MUSHROOMER,
            last_fed=now - timedelta(minutes=5),
        )
        feeder = FeedingEngine(now_func=lambda: now)

        result = feeder.feed(state, FoodType.NAUTILUS)

        assert result.success is True
        assert result.hunger_change < 0  # hunger decreased
        assert state.hunger < 0.5  # state mutated

    def test_full_needs_cycle_degrade_then_feed(self):
        """Complete cycle: needs degrade → urgent warnings → feeding restores."""
        needs_engine = NeedsEngine(config=NeedsConfig(hunger_rate=0.1))
        tank = TankEnvironment()

        # Start with a fed, healthy creature
        state = CreatureState(
            hunger=0.0, health=1.0, comfort=1.0,
            stage=CreatureStage.GILLMAN,
            last_fed=datetime(2026, 1, 1, 10, 0, 0, tzinfo=UTC),
        )

        # Simulate time passing — creature gets hungry
        needs = needs_engine.update(600.0, state, tank)  # 10 minutes
        needs_engine.apply_to_state(state, needs)
        mid_hunger = state.hunger
        assert mid_hunger > 0.0

        # More time — should get critical
        needs = needs_engine.update(6000.0, state, tank)  # ~100 more min
        needs_engine.apply_to_state(state, needs)

        urgents = needs_engine.get_urgent_needs(needs)
        # At very high hunger, expect starvation warnings
        assert state.hunger >= 0.8 or len(urgents) > 0

        # Now feed the creature
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        feeder = FeedingEngine(now_func=lambda: now)
        state.last_fed = now - timedelta(minutes=5)  # clear cooldown
        result = feeder.feed(state, FoodType.WORM)  # GILLMAN can eat worms

        assert result.success is True
        assert state.hunger < 1.0  # hunger reduced from feeding

    def test_wrong_food_rejected(self):
        """Creature rejects food inappropriate for its stage."""
        now = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        state = CreatureState(
            hunger=0.5,
            stage=CreatureStage.MUSHROOMER,
            last_fed=now - timedelta(minutes=5),
        )
        feeder = FeedingEngine(now_func=lambda: now)

        # MUSHROOMER can only eat NAUTILUS, not WORM
        result = feeder.feed(state, FoodType.WORM)

        assert result.success is False
        assert "won't eat" in result.message


# ---------------------------------------------------------------------------
# 3. Evolution triggers, stage changes propagate to all subsystems
# ---------------------------------------------------------------------------

class TestEvolutionPropagation:
    """Evolution triggers at threshold and propagates to mood, traits, events."""

    async def test_evolution_updates_traits_and_mood(self, tmp_path):
        """After evolution, traits change and mood engine uses new traits."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=2, trust=0.0),
            },
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=1,
            trust_level=0.5,
        )
        llm = MockLLM("Evolving now.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        old_traits = mgr.traits
        assert old_traits is not None

        await mgr.process_input("Grow!")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN

        new_traits = mgr.traits
        assert new_traits is not None
        assert new_traits is not old_traits

        # Mood engine can compute mood with the new traits
        mood_engine = MoodEngine()
        clock = GameClock()
        needs = CreatureNeeds()
        mood = mood_engine.calculate_mood(
            needs, mgr.creature_state.trust_level,
            clock.get_time_context(), 2, new_traits,
        )
        assert isinstance(mood, CreatureMood)

    def test_evolution_triggers_event_system(self):
        """EventSystem fires evolution_ready events at the right stage."""
        events = EventSystem(rng_seed=42)
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=5,
            trust_level=0.5,
        )
        tank = TankEnvironment()
        clock = GameClock()
        time_ctx = clock.get_time_context()

        fired = events.check_events(state, tank, time_ctx)
        evolution_events = [e for e in fired if e.event_type == EventType.EVOLUTION_READY]

        # MUSHROOMER should fire evolution_ready_mushroomer (one-shot)
        assert len(evolution_events) >= 1
        assert evolution_events[0].name == "evolution_ready_mushroomer"

        # Apply the effects
        for event in fired:
            events.apply_effects(event, state, tank)

    async def test_evolution_propagates_to_system_prompt(self, tmp_path):
        """After evolution, the LLM system prompt reflects the new stage."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=1, trust=0.0),
            },
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.5,
        )
        llm = MockLLM("Pre-evolution.", "Post-evolution.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        await mgr.process_input("Evolve!")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN

        # Second call should have GILLMAN system prompt
        await mgr.process_input("What are you now?")
        call_args = llm.chat.call_args[0][0]
        system_content = call_args[0].content.lower()
        assert "gillman" in system_content or "fish-like" in system_content


# ---------------------------------------------------------------------------
# 4. Death triggers from neglect, game-over state reached
# ---------------------------------------------------------------------------

class TestDeathFromNeglect:
    """Creature dies when neglected — needs hit lethal thresholds."""

    def test_starvation_death_after_grace_period(self):
        """Creature starves to death when hunger maxed for too long."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        current_time = base_time
        death_engine = DeathEngine(
            needs_config=NeedsConfig(starvation_time_hours=0.5),  # 30 min
            now_func=lambda: current_time,
        )

        state = CreatureState(stage=CreatureStage.GILLMAN, hunger=1.0)
        needs = CreatureNeeds(hunger=1.0, health=0.5)
        tank = TankEnvironment()

        # First check — starts grace period
        cause = death_engine.check_death(state, needs, tank)
        assert cause is None  # not dead yet — grace period started

        # Advance past starvation grace period (30 min = 1800 sec)
        current_time = base_time + timedelta(minutes=31)
        cause = death_engine.check_death(state, needs, tank)
        assert cause == DeathCause.STARVATION

    def test_suffocation_death_immediate(self):
        """Suffocation is immediate when oxygen drops below 0.1."""
        death_engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds()
        tank = TankEnvironment(oxygen_level=0.05)

        cause = death_engine.check_death(state, needs, tank)
        assert cause == DeathCause.SUFFOCATION

    def test_illness_death_from_health_zero(self):
        """Health reaching zero triggers illness death."""
        death_engine = DeathEngine()
        state = CreatureState()
        needs = CreatureNeeds(health=0.0)
        tank = TankEnvironment()

        cause = death_engine.check_death(state, needs, tank)
        assert cause == DeathCause.ILLNESS

    def test_death_resets_to_new_egg(self):
        """on_death creates a new MUSHROOMER (fresh egg) and a death record."""
        death_engine = DeathEngine(death_log_dir=None)
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            age=10000.0,
            interaction_count=42,
        )

        new_state, record = death_engine.on_death(DeathCause.STARVATION, state)

        assert new_state.stage == CreatureStage.MUSHROOMER
        assert new_state.interaction_count == 0
        assert record.cause == DeathCause.STARVATION
        assert record.creature_stage == CreatureStage.PODFISH
        assert "starved" in record.message.lower()

    def test_full_neglect_cycle_degrades_to_death(self):
        """Full cycle: tank degrades → needs worsen → death triggers."""
        base_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
        current_time = base_time
        env_config = EnvironmentConfig(
            oxygen_decay_rate=0.1,  # fast decay for test
        )
        needs_engine = NeedsEngine(
            config=NeedsConfig(hunger_rate=0.5),  # fast hunger
            env_config=env_config,
        )
        death_engine = DeathEngine(
            env_config=env_config,
            now_func=lambda: current_time,
        )
        tank = TankEnvironment()
        state = CreatureState(hunger=0.0, health=1.0, comfort=1.0)

        # Simulate 15 seconds of neglect with fast decay
        tank.update(15.0, env_config)

        # Oxygen should be very low now: 1.0 - 0.1 * 15 = -0.5 → clamped to 0.0
        assert tank.oxygen_level < 0.1

        needs = needs_engine.update(15.0, state, tank)
        needs_engine.apply_to_state(state, needs)

        cause = death_engine.check_death(state, needs, tank)
        assert cause == DeathCause.SUFFOCATION

        new_state, record = death_engine.on_death(cause, state)
        assert new_state.stage == CreatureStage.MUSHROOMER
        assert record.cause == DeathCause.SUFFOCATION

    def test_death_record_saved_to_disk(self, tmp_path):
        """Death record is persisted as JSON when death_log_dir is set."""
        log_dir = tmp_path / "death_logs"
        death_engine = DeathEngine(death_log_dir=str(log_dir))
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            age=5000.0,
            interaction_count=20,
        )

        new_state, record = death_engine.on_death(DeathCause.ILLNESS, state)

        assert log_dir.exists()
        log_files = list(log_dir.glob("death_*.json"))
        assert len(log_files) == 1

        data = json.loads(log_files[0].read_text())
        assert data["cause"] == "illness"
        assert data["creature_stage"] == "gillman"


# ---------------------------------------------------------------------------
# 5. API WebSocket client receives state updates and can send input
# ---------------------------------------------------------------------------

class TestAPIWebSocketIntegration:
    """Full API integration — WebSocket messaging and REST state queries."""

    async def test_websocket_input_and_state_response(self, tmp_path):
        """WebSocket client sends input and receives response with full state."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("The tank is murky today.")
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            hunger=0.3,
            trust_level=0.2,
        )
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_json({"type": "input", "text": "How's the water?"})
                resp = ws.receive_json()

                assert resp["type"] == "response"
                assert resp["text"]  # non-empty response
                assert resp["state"]["stage"] == "gillman"
                assert resp["state"]["hunger"] == pytest.approx(0.3, abs=0.01)
                assert resp["state"]["interaction_count"] == 1
                assert "timestamp" in resp

    async def test_websocket_error_on_invalid_message(self, tmp_path):
        """WebSocket returns error for unknown message types."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Whatever.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_json({"type": "bogus"})
                resp = ws.receive_json()
                assert resp["type"] == "error"

    async def test_websocket_error_on_empty_text(self, tmp_path):
        """WebSocket returns error when input text is empty."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Whatever.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            with client.websocket_connect("/ws/brain") as ws:
                ws.send_json({"type": "input", "text": ""})
                resp = ws.receive_json()
                assert resp["type"] == "error"
                assert "text" in resp["message"].lower() or "missing" in resp["message"].lower()

    async def test_rest_health_and_reset(self, tmp_path):
        """REST endpoints work: health check, state query, reset."""
        from seaman_brain.api.server import BrainServer

        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Fine.")
        state = CreatureState(interaction_count=5, trust_level=0.3)
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        server = BrainServer(config=cfg, manager=mgr)

        with TestClient(server.app) as client:
            # Health check
            resp = client.get("/api/health")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"

            # State query
            resp = client.get("/api/state")
            assert resp.status_code == 200
            assert resp.json()["state"]["interaction_count"] == 5

            # Reset
            resp = client.post("/api/reset")
            assert resp.status_code == 200
            assert resp.json()["state"]["interaction_count"] == 0


# ---------------------------------------------------------------------------
# 6. Save/load preserves creature state, tank state, and memories
# ---------------------------------------------------------------------------

class TestSaveLoadFullState:
    """Save/load cycle preserves creature state, tank state, and memory."""

    async def test_creature_state_round_trip(self, tmp_path):
        """Creature state survives save → load cycle with all fields."""
        save_dir = tmp_path / "saves"
        persistence = StatePersistence(str(save_dir))

        original = CreatureState(
            stage=CreatureStage.PODFISH,
            age=8000.0,
            interaction_count=42,
            trust_level=0.65,
            hunger=0.3,
            health=0.8,
            comfort=0.7,
            mood="sardonic",
        )

        persistence.save(original)
        loaded = persistence.load()

        assert loaded.stage == CreatureStage.PODFISH
        assert loaded.interaction_count == 42
        assert loaded.trust_level == pytest.approx(0.65, abs=0.01)
        assert loaded.hunger == pytest.approx(0.3, abs=0.01)
        assert loaded.health == pytest.approx(0.8, abs=0.01)

    def test_tank_state_serialization_round_trip(self):
        """TankEnvironment survives to_dict → from_dict cycle."""
        from seaman_brain.environment.tank import EnvironmentType

        original = TankEnvironment(
            temperature=22.5,
            cleanliness=0.6,
            oxygen_level=0.8,
            water_level=0.0,
            environment_type=EnvironmentType.TERRARIUM,
        )

        data = original.to_dict()
        loaded = TankEnvironment.from_dict(data)

        assert loaded.temperature == pytest.approx(22.5)
        assert loaded.cleanliness == pytest.approx(0.6)
        assert loaded.oxygen_level == pytest.approx(0.8)
        assert loaded.water_level == pytest.approx(0.0)
        assert loaded.environment_type == EnvironmentType.TERRARIUM

    async def test_full_session_save_and_reload(self, tmp_path):
        """Full session: conversation → save → new session → verify state."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(save_path=save_dir)

        # === Session 1: interact and build state ===
        llm1 = MockLLM("Session 1 reply.")
        state1 = CreatureState(stage=CreatureStage.MUSHROOMER)
        mgr1 = ConversationManager(config=cfg, llm=llm1, creature_state=state1)
        await mgr1.initialize()

        await mgr1.process_input("Hello")
        await mgr1.process_input("How are you?")
        await mgr1.process_input("Tell me about yourself")
        await mgr1.shutdown()

        # Verify save file
        save_file = tmp_path / "saves" / "creature.json"
        assert save_file.exists()
        saved_data = json.loads(save_file.read_text())
        assert saved_data["interaction_count"] == 3
        assert saved_data["stage"] == "mushroomer"

        # === Session 2: load and verify ===
        llm2 = MockLLM("Session 2 reply.")
        mgr2 = ConversationManager(config=cfg, llm=llm2)
        await mgr2.initialize()

        assert mgr2.creature_state.interaction_count == 3
        assert mgr2.creature_state.trust_level > 0.0
        assert mgr2.creature_state.stage == CreatureStage.MUSHROOMER

        # Continue interaction in session 2
        await mgr2.process_input("I'm back!")
        assert mgr2.creature_state.interaction_count == 4

    async def test_evolution_persists_across_save_load(self, tmp_path):
        """Evolved stage persists through save/load cycle."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(
            save_path=save_dir,
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=1, trust=0.0),
            },
        )

        # Session 1: evolve
        llm1 = MockLLM("Evolved!")
        state1 = CreatureState(trust_level=0.5)
        mgr1 = ConversationManager(config=cfg, llm=llm1, creature_state=state1)
        await mgr1.initialize()
        await mgr1.process_input("Grow!")
        assert mgr1.creature_state.stage == CreatureStage.GILLMAN
        await mgr1.shutdown()

        # Session 2: verify stage persisted
        llm2 = MockLLM("Still evolved.")
        mgr2 = ConversationManager(config=cfg, llm=llm2)
        await mgr2.initialize()
        assert mgr2.creature_state.stage == CreatureStage.GILLMAN

    def test_death_record_round_trip(self, tmp_path):
        """Death records serialize/deserialize correctly."""
        from seaman_brain.needs.death import DeathRecord

        original = DeathRecord(
            cause=DeathCause.STARVATION,
            message="Starved to death.",
            creature_stage=CreatureStage.PODFISH,
            creature_age=12000.0,
            interaction_count=55,
        )

        data = original.to_dict()
        json_str = json.dumps(data)
        loaded_data = json.loads(json_str)
        loaded = DeathRecord.from_dict(loaded_data)

        assert loaded.cause == DeathCause.STARVATION
        assert loaded.creature_stage == CreatureStage.PODFISH
        assert loaded.creature_age == 12000.0
        assert loaded.interaction_count == 55

    def test_event_system_state_persistence(self):
        """EventSystem tracking state (fired one-shots) survives serialization."""
        events = EventSystem(rng_seed=42)
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=5,
        )
        tank = TankEnvironment()
        clock = GameClock()

        # Fire some events
        events.check_events(state, tank, clock.get_time_context())
        fired_before = events.get_fired_one_shots()

        # Serialize and restore
        saved = events.to_dict()
        events2 = EventSystem(rng_seed=42)
        events2.load_state(saved)

        assert events2.get_fired_one_shots() == fired_before
