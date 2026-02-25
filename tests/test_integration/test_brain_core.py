"""Integration tests for the brain core conversation loop.

These tests exercise the full conversation pipeline end-to-end with
real subsystem interactions — only external services (LLM, Ollama
embeddings) are mocked at the boundary.

Run with:  pytest -m integration
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from seaman_brain.config import (
    CreatureConfig,
    EvolutionThreshold,
    MemoryConfig,
    SeamanConfig,
)
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.state import CreatureState
from seaman_brain.types import ChatMessage, CreatureStage, MessageRole

# All tests in this file use the integration marker.
pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """A mock LLM provider returning canned responses.

    Supports a queue of responses; after exhaustion, repeats the last one.
    """

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
    extraction_interval: int = 5,
    evolution_thresholds: dict[str, EvolutionThreshold] | None = None,
    db_path: str | None = None,
) -> SeamanConfig:
    """Build a SeamanConfig suitable for integration tests."""
    creature_cfg = CreatureConfig(
        save_path=save_path,
        evolution_thresholds=evolution_thresholds or {},
    )
    memory_kwargs: dict = {"extraction_interval": extraction_interval}
    if db_path is not None:
        memory_kwargs["db_path"] = db_path
    return SeamanConfig(
        creature=creature_cfg,
        memory=MemoryConfig(**memory_kwargs),
    )


# ---------------------------------------------------------------------------
# 1. Full end-to-end: user sends message, gets personality-appropriate response
# ---------------------------------------------------------------------------

class TestEndToEndConversation:
    """User sends a message and receives a personality-appropriate response."""

    async def test_basic_round_trip(self, tmp_path):
        """Full pipeline: user input -> system prompt -> LLM -> constraints -> response."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("The water is cold and I don't care about you yet.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        response = await mgr.process_input("Hello, little creature!")
        assert isinstance(response, str)
        assert len(response) > 0
        # Verify LLM was called with a system prompt containing SEAMAN identity
        call_args = llm.chat.call_args[0][0]
        assert call_args[0].role == MessageRole.SYSTEM
        assert "SEAMAN" in call_args[0].content

    async def test_response_has_constraints_applied(self, tmp_path):
        """Personality constraints strip forbidden AI-assistant phrases."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM(
            "As an AI language model, I would be happy to help you! "
            "The tank is murky today."
        )
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        response = await mgr.process_input("Tell me about the tank")
        assert "as an ai" not in response.lower()
        assert "happy to help" not in response.lower()
        # The real content should remain
        assert "tank" in response.lower() or "murky" in response.lower()

    async def test_episodic_memory_records_both_sides(self, tmp_path):
        """Both user and assistant messages are in episodic memory after a turn."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Go away.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        await mgr.process_input("Hi there")
        messages = mgr._episodic.get_all()
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hi there"
        assert messages[1].role == MessageRole.ASSISTANT

    async def test_multi_turn_conversation(self, tmp_path):
        """Multiple turns accumulate in episodic memory and LLM sees full context."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("First response.", "Second response.", "Third response.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        r1 = await mgr.process_input("Message one")
        r2 = await mgr.process_input("Message two")
        r3 = await mgr.process_input("Message three")

        assert r1 and r2 and r3
        messages = mgr._episodic.get_all()
        # 3 user + 3 assistant = 6 messages
        assert len(messages) == 6
        assert mgr.creature_state.interaction_count == 3

    async def test_trust_increases_over_interactions(self, tmp_path):
        """Trust gradually increases with repeated interactions."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Fine.")
        state = CreatureState(trust_level=0.0)
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        for _ in range(10):
            await mgr.process_input("Hello")

        assert mgr.creature_state.trust_level > 0.0
        assert mgr.creature_state.interaction_count == 10


# ---------------------------------------------------------------------------
# 2. Memory extraction triggers after N messages
# ---------------------------------------------------------------------------

class TestMemoryExtraction:
    """Memory extraction should trigger at the configured interval."""

    async def test_extraction_triggers_at_interval(self, tmp_path):
        """After extraction_interval messages, the extractor runs."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            extraction_interval=3,
        )
        llm = MockLLM(
            "Response 1.",
            "Response 2.",
            # Third round: LLM returns for conversation, then extraction prompt returns facts
            "Response 3.",
            "User likes fish\nUser's name is Dave",
        )
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Mock embedding to avoid hitting real Ollama
        if mgr._embeddings is not None:
            mgr._embeddings.embed = AsyncMock(side_effect=_fake_embed)

        await mgr.process_input("Message 1")
        await mgr.process_input("Message 2")
        # Third message should trigger extraction (interval=3)
        await mgr.process_input("Message 3")

        # The extractor counter should have been reset after extraction
        if mgr._extractor is not None:
            assert mgr._extractor.message_count == 0

    async def test_extraction_stores_facts_in_semantic_memory(self, tmp_path):
        """Extracted facts are persisted as embeddings in semantic memory."""
        db_path = str(tmp_path / "lancedb")
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            extraction_interval=1,
            db_path=db_path,
        )
        llm = MockLLM(
            "Whatever.",
            "User likes cats\nUser lives in Tokyo",
        )
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Mock embedding at the provider level
        if mgr._embeddings is not None:
            mgr._embeddings.embed = AsyncMock(side_effect=_fake_embed)

        await mgr.process_input("I like cats and I live in Tokyo")

        # Verify facts were stored in semantic memory
        if mgr._semantic is not None:
            count = await mgr._semantic.count()
            assert count >= 1

    async def test_no_extraction_before_interval(self, tmp_path):
        """Extraction should NOT trigger before the interval is reached."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            extraction_interval=10,
        )
        llm = MockLLM("Response.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        if mgr._extractor is not None:
            mgr._extractor.extract_and_store = AsyncMock(return_value=[])

        # Send fewer messages than extraction interval
        for _ in range(3):
            await mgr.process_input("Hello")

        if mgr._extractor is not None:
            mgr._extractor.extract_and_store.assert_not_awaited()


# ---------------------------------------------------------------------------
# 3. Creature evolves after reaching interaction threshold
# ---------------------------------------------------------------------------

class TestEvolution:
    """Creature evolves when interaction and trust thresholds are met."""

    async def test_evolves_at_threshold(self, tmp_path):
        """Creature transitions from MUSHROOMER to GILLMAN at threshold."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=3, trust=0.0),
            },
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=2,
            trust_level=0.5,
        )
        llm = MockLLM("Evolving response.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        old_traits = mgr.traits
        await mgr.process_input("Grow!")

        assert mgr.creature_state.stage == CreatureStage.GILLMAN
        assert mgr.traits is not old_traits  # Traits updated to GILLMAN defaults

    async def test_multi_stage_evolution_over_sessions(self, tmp_path):
        """Creature can evolve through multiple stages across interactions."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(
            save_path=save_dir,
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=2, trust=0.0),
                "podfish": EvolutionThreshold(interactions=4, trust=0.0),
            },
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=1,
            trust_level=0.5,
        )
        llm = MockLLM("Response.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        # Interaction 2: should evolve to GILLMAN
        await mgr.process_input("Step 1")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN

        # Interactions 3-4: need one more to hit podfish
        await mgr.process_input("Step 2")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN  # Not yet

        await mgr.process_input("Step 3")
        assert mgr.creature_state.stage == CreatureStage.PODFISH

    async def test_no_evolution_below_threshold(self, tmp_path):
        """Creature stays at current stage when thresholds not met."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=100, trust=0.9),
            },
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.0,
        )
        llm = MockLLM("Still small.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        for _ in range(5):
            await mgr.process_input("Hello")

        assert mgr.creature_state.stage == CreatureStage.MUSHROOMER

    async def test_evolution_changes_system_prompt(self, tmp_path):
        """After evolution, the LLM receives a system prompt reflecting the new stage."""
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

        # First call triggers evolution before LLM call
        await mgr.process_input("Evolve!")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN

        # Check that the system prompt reflects GILLMAN identity (not MUSHROOMER)
        call_args = llm.chat.call_args[0][0]
        system_content = call_args[0].content.lower()
        # GILLMAN identity: "fish-like creature" vs MUSHROOMER: "mushroom-like larva"
        assert "fish-like creature" in system_content
        assert "mushroom-like larva" not in system_content


# ---------------------------------------------------------------------------
# 4. State persists across simulated sessions (save/load)
# ---------------------------------------------------------------------------

class TestSessionPersistence:
    """State survives a full shutdown + restart cycle."""

    async def test_state_persists_across_sessions(self, tmp_path):
        """Creature state saved in session 1 is loaded in session 2."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(save_path=save_dir)

        # === Session 1 ===
        llm1 = MockLLM("Session 1 response.")
        state1 = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.0,
        )
        mgr1 = ConversationManager(config=cfg, llm=llm1, creature_state=state1)
        await mgr1.initialize()

        await mgr1.process_input("Hello session 1")
        await mgr1.process_input("Second message")
        await mgr1.shutdown()

        # Verify save file exists
        save_file = tmp_path / "saves" / "creature.json"
        assert save_file.exists()

        # === Session 2 ===
        # Load state from disk (no creature_state argument)
        llm2 = MockLLM("Session 2 response.")
        mgr2 = ConversationManager(config=cfg, llm=llm2)
        await mgr2.initialize()

        # State should reflect session 1's changes
        assert mgr2.creature_state.interaction_count == 2
        assert mgr2.creature_state.trust_level > 0.0

    async def test_evolution_persists_across_sessions(self, tmp_path):
        """Creature stage survives shutdown and reload."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(
            save_path=save_dir,
            evolution_thresholds={
                "gillman": EvolutionThreshold(interactions=1, trust=0.0),
            },
        )

        # === Session 1: evolve to GILLMAN ===
        llm1 = MockLLM("Evolved!")
        state1 = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.5,
        )
        mgr1 = ConversationManager(config=cfg, llm=llm1, creature_state=state1)
        await mgr1.initialize()
        await mgr1.process_input("Grow!")
        assert mgr1.creature_state.stage == CreatureStage.GILLMAN
        await mgr1.shutdown()

        # === Session 2: verify stage persisted ===
        llm2 = MockLLM("Still evolved.")
        mgr2 = ConversationManager(config=cfg, llm=llm2)
        await mgr2.initialize()
        assert mgr2.creature_state.stage == CreatureStage.GILLMAN

    async def test_save_file_is_valid_json(self, tmp_path):
        """The save file is well-formed JSON with expected fields."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(save_path=save_dir)
        llm = MockLLM("OK.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        await mgr.process_input("Test")
        await mgr.shutdown()

        save_file = tmp_path / "saves" / "creature.json"
        data = json.loads(save_file.read_text())
        assert "stage" in data
        assert "interaction_count" in data
        assert "trust_level" in data
        assert data["interaction_count"] == 1

    async def test_backup_created_on_second_save(self, tmp_path):
        """A .bak backup file is created when overwriting an existing save."""
        save_dir = str(tmp_path / "saves")
        cfg = _make_config(save_path=save_dir)
        llm = MockLLM("First.", "Second.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        await mgr.process_input("First message")
        # First save creates creature.json
        assert (tmp_path / "saves" / "creature.json").exists()

        await mgr.process_input("Second message")
        # Second save should create a .bak backup
        assert (tmp_path / "saves" / "creature.json.bak").exists()


# ---------------------------------------------------------------------------
# 5. Graceful degradation when embedding service unavailable
# ---------------------------------------------------------------------------

class TestGracefulDegradation:
    """System works even when embedding/memory subsystems fail."""

    async def test_conversation_works_without_embeddings(self, tmp_path):
        """Conversation continues when embedding provider fails to initialize."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Still talking without embeddings.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())

        # Patch EmbeddingProvider to fail on init
        with patch(
            "seaman_brain.conversation.manager.EmbeddingProvider",
            side_effect=ConnectionError("Ollama not running"),
        ):
            await mgr.initialize()

        # Memory subsystem should be degraded
        assert mgr._embeddings is None
        assert mgr._semantic is None
        assert mgr._retriever is None

        # But conversation still works
        response = await mgr.process_input("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    async def test_memory_retrieval_failure_does_not_crash(self, tmp_path):
        """When memory retrieval raises, conversation still returns a response."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Still here.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Force retriever to fail mid-conversation
        if mgr._retriever is not None:
            mgr._retriever.retrieve = AsyncMock(
                side_effect=ConnectionError("embedding service down")
            )

        response = await mgr.process_input("Can you remember anything?")
        assert isinstance(response, str)
        assert len(response) > 0
        # State should still be updated
        assert mgr.creature_state.interaction_count == 1

    async def test_extraction_failure_does_not_block_response(self, tmp_path):
        """When memory extraction fails, the user still gets a response."""
        cfg = _make_config(
            save_path=str(tmp_path / "saves"),
            extraction_interval=1,
        )
        llm = MockLLM("Response despite extraction failure.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Force extraction to fail
        if mgr._extractor is not None:
            mgr._extractor.extract_and_store = AsyncMock(
                side_effect=RuntimeError("extraction exploded")
            )

        response = await mgr.process_input("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    async def test_llm_failure_returns_fallback(self, tmp_path):
        """When LLM is unavailable mid-conversation, a fallback response is returned."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=ConnectionError("LLM went away"))
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        response = await mgr.process_input("Anyone there?")
        assert response == "..."

    async def test_save_failure_does_not_crash(self, tmp_path):
        """When state persistence fails, the response is still returned."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Saved or not, here I am.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Make persistence.save raise (sync method — use MagicMock, not AsyncMock)
        mgr._persistence.save = MagicMock(side_effect=OSError("disk full"))

        response = await mgr.process_input("Hello")
        assert isinstance(response, str)
        assert len(response) > 0
