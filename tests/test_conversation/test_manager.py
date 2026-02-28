"""Tests for ConversationManager — the main conversation orchestrator."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.config import CreatureConfig, EvolutionThreshold, SeamanConfig
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.state import CreatureState
from seaman_brain.personality.traits import TraitProfile
from seaman_brain.types import ChatMessage, CreatureStage, MessageRole

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockLLM:
    """A mock LLM provider that returns a canned response."""

    def __init__(self, response: str = "Whatever.") -> None:
        self._response = response
        self.chat = AsyncMock(return_value=response)

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        yield self._response


def _make_config(save_path: str = "data/saves") -> SeamanConfig:
    """Create a minimal SeamanConfig suitable for testing."""
    return SeamanConfig(
        creature=CreatureConfig(save_path=save_path),
    )


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------

class TestInitialize:
    """Tests for ConversationManager.initialize()."""

    async def test_initialize_sets_flag(self, tmp_path):
        """Initialize sets is_initialized to True."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(
            config=cfg,
            llm=MockLLM(),
            creature_state=CreatureState(),
        )
        assert not mgr.is_initialized
        await mgr.initialize()
        assert mgr.is_initialized

    async def test_initialize_idempotent(self, tmp_path):
        """Calling initialize() twice is safe and doesn't reset state."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(interaction_count=5)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()
        assert mgr.creature_state.interaction_count == 5
        await mgr.initialize()
        assert mgr.creature_state.interaction_count == 5

    async def test_initialize_creates_episodic_memory(self, tmp_path):
        """Initialize creates episodic memory buffer from config."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()
        assert mgr._episodic is not None
        assert mgr._episodic.max_size == cfg.memory.buffer_size

    async def test_initialize_loads_default_traits(self, tmp_path):
        """Initialize loads default trait profile for the creature's stage."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()
        assert mgr.traits is not None
        assert isinstance(mgr.traits, TraitProfile)

    async def test_initialize_without_llm_logs_error(self, tmp_path):
        """Initialize without LLM (and failing factory) logs error but doesn't crash."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        with patch(
            "seaman_brain.conversation.manager.create_provider",
            side_effect=ImportError("no module"),
        ):
            mgr = ConversationManager(config=cfg, creature_state=CreatureState())
            await mgr.initialize()
            assert mgr.is_initialized
            assert mgr._llm is None

    async def test_initialize_creates_persistence(self, tmp_path):
        """Initialize creates StatePersistence with configured save path."""
        save_path = str(tmp_path / "saves")
        cfg = _make_config(save_path=save_path)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()
        assert mgr._persistence is not None

    async def test_initialize_resolves_active_bloodline(self, tmp_path):
        """Initialize resolves the active bloodline from _active.txt."""
        import json

        save_base = tmp_path / "saves"
        # Set up a bloodline named "alpha"
        alpha_dir = save_base / "alpha"
        alpha_dir.mkdir(parents=True)
        state = CreatureState(trust_level=0.77)
        (alpha_dir / "creature.json").write_text(
            json.dumps(state.to_dict()), encoding="utf-8"
        )
        # Mark "alpha" as active
        (save_base / "_active.txt").write_text("alpha", encoding="utf-8")

        cfg = _make_config(save_path=str(save_base))
        mgr = ConversationManager(config=cfg, llm=MockLLM())
        await mgr.initialize()

        # Should have loaded from alpha/, not from base dir
        assert mgr.creature_state is not None
        assert mgr.creature_state.trust_level == pytest.approx(0.77)

    async def test_initialize_migrates_flat_saves(self, tmp_path):
        """Initialize calls migrate_flat_saves before resolving active bloodline."""
        import json

        save_base = tmp_path / "saves"
        save_base.mkdir(parents=True)
        # Old flat layout: creature.json at root
        state = CreatureState(trust_level=0.55)
        (save_base / "creature.json").write_text(
            json.dumps(state.to_dict()), encoding="utf-8"
        )

        cfg = _make_config(save_path=str(save_base))
        mgr = ConversationManager(config=cfg, llm=MockLLM())
        await mgr.initialize()

        # Migration should have moved to default/ and active is "default"
        assert (save_base / "default" / "creature.json").exists()
        assert not (save_base / "creature.json").exists()
        assert mgr.creature_state is not None
        assert mgr.creature_state.trust_level == pytest.approx(0.55)

    async def test_initialize_warmup_calls_llm(self, tmp_path):
        """Initialize sends a warmup chat call to preload the LLM."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()
        # Warmup should have called chat once with a single "." message
        llm.chat.assert_awaited_once()
        warmup_msgs = llm.chat.call_args[0][0]
        assert len(warmup_msgs) == 1
        assert warmup_msgs[0].content == "."

    async def test_initialize_warmup_failure_nonfatal(self, tmp_path):
        """LLM warmup failure doesn't prevent initialization."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=ConnectionError("Ollama not running"))
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()
        assert mgr.is_initialized


# ---------------------------------------------------------------------------
# Happy-path process_input tests
# ---------------------------------------------------------------------------

class TestProcessInput:
    """Tests for the main process_input conversation loop."""

    async def test_basic_conversation(self, tmp_path):
        """User sends a message and gets a response."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("The water is fine.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        llm.chat.reset_mock()  # Clear warmup call from initialize()
        response = await mgr.process_input("Hello there!")
        assert isinstance(response, str)
        assert len(response) > 0
        llm.chat.assert_awaited_once()

    async def test_increments_interaction_count(self, tmp_path):
        """process_input increments the creature's interaction count."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(interaction_count=0)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        await mgr.process_input("Hi")
        assert mgr.creature_state.interaction_count == 1
        await mgr.process_input("Hi again")
        assert mgr.creature_state.interaction_count == 2

    async def test_updates_last_interaction(self, tmp_path):
        """process_input updates last_interaction timestamp."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        old_time = datetime(2020, 1, 1, tzinfo=UTC)
        state = CreatureState(last_interaction=old_time)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        await mgr.process_input("Hello")
        assert mgr.creature_state.last_interaction > old_time

    async def test_bumps_trust(self, tmp_path):
        """process_input gradually increases trust level."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(trust_level=0.0)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        await mgr.process_input("Hey")
        assert mgr.creature_state.trust_level > 0.0

    async def test_episodic_stores_user_and_assistant(self, tmp_path):
        """Both user and assistant messages are stored in episodic memory."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(
            config=cfg, llm=MockLLM("Fine."), creature_state=CreatureState()
        )
        await mgr.initialize()

        await mgr.process_input("Hello")
        messages = mgr._episodic.get_all()
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Hello"
        assert messages[1].role == MessageRole.ASSISTANT

    async def test_context_includes_system_prompt(self, tmp_path):
        """The LLM receives a context with a system prompt as the first message."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Sure.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        await mgr.process_input("Tell me something")
        # Check the messages sent to LLM
        call_args = llm.chat.call_args[0][0]
        assert call_args[0].role == MessageRole.SYSTEM
        assert "SEAMAN" in call_args[0].content

    async def test_constraints_applied_to_response(self, tmp_path):
        """Output filtering strips forbidden phrases from the LLM response."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("As an AI, I would be happy to help you today!")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        response = await mgr.process_input("Help me")
        assert "as an ai" not in response.lower()
        assert "happy to help" not in response.lower()

    async def test_saves_state_after_input(self, tmp_path):
        """State is persisted to disk after each process_input call."""
        save_dir = tmp_path / "saves"
        cfg = _make_config(save_path=str(save_dir))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        await mgr.process_input("Hello")
        # Saves to active bloodline subdirectory (default)
        assert (save_dir / "default" / "creature.json").exists()


# ---------------------------------------------------------------------------
# Evolution tests
# ---------------------------------------------------------------------------

class TestEvolution:
    """Tests for evolution triggers during conversation."""

    async def test_evolution_triggers_on_threshold(self, tmp_path):
        """Creature evolves when interaction and trust thresholds are met."""
        cfg = SeamanConfig(
            creature=CreatureConfig(
                save_path=str(tmp_path / "saves"),
                evolution_thresholds={
                    "gillman": EvolutionThreshold(interactions=2, trust=0.1),
                },
            ),
        )
        # Start at 1 interaction, just below threshold
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=1,
            trust_level=0.5,
        )
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        # This interaction bumps count to 2, meeting the threshold
        await mgr.process_input("Evolve!")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN

    async def test_evolution_updates_traits(self, tmp_path):
        """Evolution replaces the trait profile with the new stage's defaults."""
        cfg = SeamanConfig(
            creature=CreatureConfig(
                save_path=str(tmp_path / "saves"),
                evolution_thresholds={
                    "gillman": EvolutionThreshold(interactions=1, trust=0.0),
                },
            ),
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.5,
        )
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        old_traits = mgr.traits
        await mgr.process_input("Grow!")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN
        # Traits should differ because stage changed
        assert mgr.traits is not old_traits

    async def test_no_evolution_below_threshold(self, tmp_path):
        """Creature does NOT evolve when thresholds are not met."""
        cfg = SeamanConfig(
            creature=CreatureConfig(
                save_path=str(tmp_path / "saves"),
                evolution_thresholds={
                    "gillman": EvolutionThreshold(interactions=100, trust=0.9),
                },
            ),
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.0,
        )
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        await mgr.process_input("Stay small")
        assert mgr.creature_state.stage == CreatureStage.MUSHROOMER

    async def test_max_stage_no_evolution(self, tmp_path):
        """Creature at max stage does not evolve further."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(
            stage=CreatureStage.FROGMAN,
            interaction_count=999,
            trust_level=1.0,
        )
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        await mgr.process_input("More evolution?")
        assert mgr.creature_state.stage == CreatureStage.FROGMAN


# ---------------------------------------------------------------------------
# Graceful degradation tests
# ---------------------------------------------------------------------------

class TestDegradation:
    """Tests for graceful degradation when subsystems are unavailable."""

    async def test_process_without_init_raises(self, tmp_path):
        """process_input raises RuntimeError if not initialized."""
        mgr = ConversationManager(config=_make_config(), llm=MockLLM())
        with pytest.raises(RuntimeError, match="not initialized"):
            await mgr.process_input("Hello")

    async def test_process_without_llm_raises(self, tmp_path):
        """process_input raises RuntimeError when no LLM is available."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        with patch(
            "seaman_brain.conversation.manager.create_provider",
            side_effect=ImportError("nope"),
        ):
            mgr = ConversationManager(config=cfg, creature_state=CreatureState())
            await mgr.initialize()
            with pytest.raises(RuntimeError, match="No LLM provider"):
                await mgr.process_input("Hello")

    async def test_llm_failure_returns_fallback(self, tmp_path):
        """When the LLM raises, a fallback response is returned."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=ConnectionError("LLM down"))
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        response = await mgr.process_input("Hello")
        assert response == "..."

    async def test_memory_retrieval_failure_continues(self, tmp_path):
        """When memory retrieval fails, conversation continues without memories."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Still here.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Force retriever to fail
        if mgr._retriever is not None:
            mgr._retriever.retrieve = AsyncMock(side_effect=ConnectionError("embed down"))

        response = await mgr.process_input("Hello")
        assert isinstance(response, str)
        assert len(response) > 0

    async def test_memory_extraction_failure_continues(self, tmp_path):
        """When extraction fails, the response is still returned."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Hmph.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Force extractor to fail but still trigger extraction
        if mgr._extractor is not None:
            mgr._extractor._extraction_interval = 1  # Trigger every message
            mgr._extractor.extract_and_store = AsyncMock(
                side_effect=RuntimeError("extraction broken")
            )

        response = await mgr.process_input("Hello")
        assert isinstance(response, str)

    async def test_save_failure_does_not_crash(self, tmp_path):
        """When save fails, process_input still returns the response."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Fine.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()
        mgr._persistence.save = MagicMock(side_effect=OSError("disk full"))

        response = await mgr.process_input("Hello")
        assert response  # Still got a response


# ---------------------------------------------------------------------------
# Shutdown tests
# ---------------------------------------------------------------------------

class TestShutdown:
    """Tests for shutdown behavior."""

    async def test_shutdown_saves_state(self, tmp_path):
        """shutdown() saves creature state to disk."""
        save_dir = tmp_path / "saves"
        cfg = _make_config(save_path=str(save_dir))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        await mgr.shutdown()
        # Saves to active bloodline subdirectory (default)
        assert (save_dir / "default" / "creature.json").exists()

    async def test_shutdown_clears_initialized(self, tmp_path):
        """shutdown() resets the initialized flag."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()
        assert mgr.is_initialized

        await mgr.shutdown()
        assert not mgr.is_initialized

    async def test_shutdown_before_init_is_safe(self):
        """shutdown() before initialize() does not crash."""
        mgr = ConversationManager()
        await mgr.shutdown()  # Should not raise


# ---------------------------------------------------------------------------
# LLM settings hot-swap tests
# ---------------------------------------------------------------------------

class TestUpdateLLMSettings:
    """Tests for update_llm_settings() — live model/temperature changes."""

    async def test_updates_model_on_provider(self, tmp_path):
        """update_llm_settings updates the model attribute on the LLM provider."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.model = "old-model"
        llm.temperature = 0.5
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        mgr.update_llm_settings("new-model", 0.9)
        assert llm.model == "new-model"
        assert llm.temperature == 0.9

    async def test_updates_temperature_on_provider(self, tmp_path):
        """update_llm_settings updates temperature independently."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.model = "test-model"
        llm.temperature = 0.3
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        mgr.update_llm_settings("test-model", 0.7)
        assert llm.temperature == 0.7

    async def test_no_crash_without_llm(self, tmp_path):
        """update_llm_settings is safe when no LLM provider is available."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        with patch(
            "seaman_brain.conversation.manager.create_provider",
            side_effect=ImportError("nope"),
        ):
            mgr = ConversationManager(config=cfg, creature_state=CreatureState())
            await mgr.initialize()
            mgr.update_llm_settings("any-model", 0.5)  # Should not raise

    async def test_no_crash_without_model_attr(self, tmp_path):
        """update_llm_settings is safe when provider lacks model/temperature attrs."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        # MockLLM doesn't have model/temperature by default
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()
        mgr.update_llm_settings("new-model", 0.5)  # Should not raise


# ---------------------------------------------------------------------------
# Utility method tests
# ---------------------------------------------------------------------------

class TestStateSummary:
    """Tests for get_state_summary()."""

    async def test_summary_returns_expected_keys(self, tmp_path):
        """get_state_summary returns expected keys after initialization."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        summary = mgr.get_state_summary()
        assert "stage" in summary
        assert "mood" in summary
        assert "trust_level" in summary
        assert "hunger" in summary
        assert "health" in summary
        assert "interaction_count" in summary

    def test_summary_before_init_returns_empty(self):
        """get_state_summary returns empty dict if not initialized."""
        mgr = ConversationManager()
        assert mgr.get_state_summary() == {}

    async def test_summary_reflects_state_changes(self, tmp_path):
        """Summary reflects updated interaction count after process_input."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(interaction_count=0)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        await mgr.process_input("Hello")
        summary = mgr.get_state_summary()
        assert summary["interaction_count"] == 1


# ---------------------------------------------------------------------------
# Fallback response tests
# ---------------------------------------------------------------------------

class TestFallbackResponse:
    """Tests for the static fallback response."""

    def test_fallback_default(self):
        assert ConversationManager._fallback_response(evolved=False) == "..."

    def test_fallback_evolved(self):
        resp = ConversationManager._fallback_response(evolved=True)
        assert "different" in resp.lower()

    async def test_llm_error_after_evolution_uses_evolved_fallback(self, tmp_path):
        """If LLM fails right after evolution, the evolved fallback is used."""
        cfg = SeamanConfig(
            creature=CreatureConfig(
                save_path=str(tmp_path / "saves"),
                evolution_thresholds={
                    "gillman": EvolutionThreshold(interactions=1, trust=0.0),
                },
            ),
        )
        state = CreatureState(
            stage=CreatureStage.MUSHROOMER,
            interaction_count=0,
            trust_level=0.5,
        )
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=ConnectionError("LLM crashed"))
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=state)
        await mgr.initialize()

        response = await mgr.process_input("Evolve now!")
        assert mgr.creature_state.stage == CreatureStage.GILLMAN
        assert "different" in response.lower()


# ---------------------------------------------------------------------------
# Memory extraction integration
# ---------------------------------------------------------------------------

class TestMemoryExtraction:
    """Tests for memory extraction triggering."""

    async def test_extraction_triggered_at_interval(self, tmp_path):
        """Memory extraction is triggered when message count reaches interval."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        # Set extraction interval to 2 messages
        cfg = cfg.model_copy(
            update={"memory": cfg.memory.model_copy(update={"extraction_interval": 2})}
        )
        llm = MockLLM("Response.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Mock the extractor to track calls
        if mgr._extractor is not None:
            mgr._extractor.extract_and_store = AsyncMock(return_value=[])

            await mgr.process_input("Message 1")
            mgr._extractor.extract_and_store.assert_not_awaited()

            await mgr.process_input("Message 2")
            mgr._extractor.extract_and_store.assert_awaited_once()

    async def test_no_extraction_without_extractor(self, tmp_path):
        """Conversation works even if extractor is None."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM("OK."), creature_state=CreatureState())
        await mgr.initialize()
        mgr._extractor = None

        response = await mgr.process_input("Hello")
        assert isinstance(response, str)


# ---------------------------------------------------------------------------
# Vision observation wiring tests
# ---------------------------------------------------------------------------

class TestVisionObservations:
    """Tests for vision observation injection into prompt builder."""

    async def test_set_vision_observations(self, tmp_path):
        """set_vision_observations stores observations for prompt injection."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        mgr.set_vision_observations(["Human looks bored", "Room is dark"])
        assert mgr._vision_observations == ["Human looks bored", "Room is dark"]

    async def test_observations_injected_into_prompt(self, tmp_path):
        """Vision observations appear in the system prompt sent to LLM."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("I see you.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        mgr.set_vision_observations(["Human is smiling"])
        await mgr.process_input("What do you see?")

        # Check system prompt in the LLM call
        call_args = llm.chat.call_args[0][0]
        system_content = call_args[0].content
        assert "WHAT YOU CAN SEE RIGHT NOW:" in system_content
        assert "Human is smiling" in system_content

    async def test_empty_observations_not_injected(self, tmp_path):
        """Empty observations don't add a vision section."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Whatever.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        mgr.set_vision_observations([])
        await mgr.process_input("Hello")

        call_args = llm.chat.call_args[0][0]
        system_content = call_args[0].content
        assert "WHAT YOU CAN SEE" not in system_content

    async def test_default_observations_empty(self, tmp_path):
        """Default vision observations are empty."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        assert mgr._vision_observations == []


# ---------------------------------------------------------------------------
# Autonomous remark tests
# ---------------------------------------------------------------------------

class TestGenerateAutonomousRemark:
    """Tests for generate_autonomous_remark() — unprompted LLM speech."""

    async def test_returns_remark(self, tmp_path):
        """Generates a remark from a situation description."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Ugh, feed me already.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        llm.chat.reset_mock()  # Clear warmup call from initialize()
        result = await mgr.generate_autonomous_remark("You are very hungry.")
        assert result is not None
        assert len(result) > 0
        llm.chat.assert_awaited_once()

    async def test_no_user_message_in_episodic(self, tmp_path):
        """Autonomous remark does NOT add a USER message to episodic memory."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Fine, whatever.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        await mgr.generate_autonomous_remark("You are bored.")
        messages = mgr._episodic.get_all()
        # Should only have the ASSISTANT message, no USER
        assert len(messages) == 1
        assert messages[0].role == MessageRole.ASSISTANT

    async def test_assistant_stored_in_episodic(self, tmp_path):
        """Autonomous remark stores the ASSISTANT response in episodic memory."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Interesting perspective.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        result = await mgr.generate_autonomous_remark("You notice something.")
        messages = mgr._episodic.get_all()
        assert len(messages) == 1
        assert messages[0].content == result

    async def test_no_interaction_count_bump(self, tmp_path):
        """Autonomous remark does NOT increment interaction_count."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(interaction_count=5)
        mgr = ConversationManager(config=cfg, llm=MockLLM("Hmm."), creature_state=state)
        await mgr.initialize()

        await mgr.generate_autonomous_remark("You are observing.")
        assert mgr.creature_state.interaction_count == 5

    async def test_no_trust_bump(self, tmp_path):
        """Autonomous remark does NOT change trust level."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(trust_level=0.3)
        mgr = ConversationManager(config=cfg, llm=MockLLM("Hmm."), creature_state=state)
        await mgr.initialize()

        await mgr.generate_autonomous_remark("You are reflecting.")
        assert mgr.creature_state.trust_level == pytest.approx(0.3)

    async def test_constraints_applied(self, tmp_path):
        """Personality constraints are applied to autonomous remarks."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("As an AI, I would be happy to help!")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        result = await mgr.generate_autonomous_remark("You are observing.")
        assert result is not None
        assert "as an ai" not in result.lower()

    async def test_returns_none_when_not_initialized(self, tmp_path):
        """Returns None if manager is not initialized."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        # NOT calling initialize()
        result = await mgr.generate_autonomous_remark("Hello")
        assert result is None

    async def test_returns_none_when_no_llm(self, tmp_path):
        """Returns None if no LLM provider is available."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        with patch(
            "seaman_brain.conversation.manager.create_provider",
            side_effect=ImportError("nope"),
        ):
            mgr = ConversationManager(config=cfg, creature_state=CreatureState())
            await mgr.initialize()
            result = await mgr.generate_autonomous_remark("Hello")
            assert result is None

    async def test_returns_none_on_llm_failure(self, tmp_path):
        """Returns None when the LLM call raises an exception."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=ConnectionError("LLM down"))
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        result = await mgr.generate_autonomous_remark("You are hungry.")
        assert result is None

    async def test_returns_none_on_empty_response(self, tmp_path):
        """Returns None when the LLM returns an empty string."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        result = await mgr.generate_autonomous_remark("You are bored.")
        assert result is None

    async def test_situation_in_llm_context(self, tmp_path):
        """The situation text appears as a trailing USER message sent to LLM."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Noted.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        await mgr.generate_autonomous_remark("You are extremely bored.")
        call_args = llm.chat.call_args[0][0]
        # Situation directive is now a USER message at the end of context
        last_msg = call_args[-1]
        assert last_msg.role.value == "user"
        assert "CURRENT SITUATION:" in last_msg.content
        assert "extremely bored" in last_msg.content


# ---------------------------------------------------------------------------
# Streaming process_input tests
# ---------------------------------------------------------------------------

class TestProcessInputStream:
    """Tests for process_input_stream() — the async generator variant."""

    async def test_stream_yields_tokens(self, tmp_path):
        """Tokens from the LLM stream are yielded to the caller."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Streamed.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        tokens = []
        async for token in mgr.process_input_stream("Hello"):
            tokens.append(token)

        assert len(tokens) >= 1
        assert "".join(tokens) == "Streamed."

    async def test_stream_stores_episodic(self, tmp_path):
        """User and assistant messages are stored in episodic memory after streaming."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("Fine.")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        tokens = []
        async for token in mgr.process_input_stream("Test"):
            tokens.append(token)

        messages = mgr._episodic.get_all()
        assert len(messages) == 2
        assert messages[0].role == MessageRole.USER
        assert messages[0].content == "Test"
        assert messages[1].role == MessageRole.ASSISTANT

    async def test_stream_increments_interaction_count(self, tmp_path):
        """Streaming increments the creature's interaction count."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(interaction_count=0)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        async for _ in mgr.process_input_stream("Hi"):
            pass

        assert mgr.creature_state.interaction_count == 1

    async def test_stream_updates_trust(self, tmp_path):
        """Streaming bumps trust level."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        state = CreatureState(trust_level=0.0)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
        await mgr.initialize()

        async for _ in mgr.process_input_stream("Hi"):
            pass

        assert mgr.creature_state.trust_level > 0.0

    async def test_stream_applies_constraints(self, tmp_path):
        """Personality constraints are applied to the accumulated text."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM("As an AI, I would be happy to help!")
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        await mgr.initialize()

        # Consume the generator to trigger post-stream processing
        async for _ in mgr.process_input_stream("Help"):
            pass

        # Check the stored assistant message — constraints should have filtered
        messages = mgr._episodic.get_all()
        assistant_msg = messages[-1]
        assert "as an ai" not in assistant_msg.content.lower()

    async def test_stream_saves_state(self, tmp_path):
        """State is persisted after stream completes."""
        save_dir = tmp_path / "saves"
        cfg = _make_config(save_path=str(save_dir))
        mgr = ConversationManager(
            config=cfg, llm=MockLLM(), creature_state=CreatureState()
        )
        await mgr.initialize()

        async for _ in mgr.process_input_stream("Save test"):
            pass

        # Saves to active bloodline subdirectory (default)
        assert (save_dir / "default" / "creature.json").exists()

    async def test_stream_not_initialized_raises(self, tmp_path):
        """Raises RuntimeError if not initialized."""
        mgr = ConversationManager(config=_make_config(), llm=MockLLM())
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in mgr.process_input_stream("Hello"):
                pass

    async def test_stream_no_llm_raises(self, tmp_path):
        """Raises RuntimeError when no LLM is available."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        with patch(
            "seaman_brain.conversation.manager.create_provider",
            side_effect=ImportError("nope"),
        ):
            mgr = ConversationManager(config=cfg, creature_state=CreatureState())
            await mgr.initialize()
            with pytest.raises(RuntimeError, match="No LLM provider"):
                async for _ in mgr.process_input_stream("Hello"):
                    pass


# ---------------------------------------------------------------------------
# LLM Timeout tests (Fix #2)
# ---------------------------------------------------------------------------
import asyncio  # noqa: E402


class TestLLMTimeouts:
    """Tests for LLM call timeout handling."""

    async def test_warmup_timeout_caught(self, tmp_path):
        """Warmup timeout is caught gracefully."""
        import seaman_brain.conversation.manager as mgr_mod

        # Temporarily reduce timeout for test speed
        orig = mgr_mod._LLM_WARMUP_TIMEOUT
        mgr_mod._LLM_WARMUP_TIMEOUT = 0.1

        try:
            slow_llm = MockLLM()
            slow_llm.chat = AsyncMock(side_effect=asyncio.TimeoutError)

            cfg = _make_config(save_path=str(tmp_path / "saves"))
            mgr = ConversationManager(
                config=cfg, llm=slow_llm, creature_state=CreatureState()
            )
            # Should not raise — warmup timeout is caught
            await mgr.initialize()
            assert mgr.is_initialized
        finally:
            mgr_mod._LLM_WARMUP_TIMEOUT = orig

    async def test_chat_timeout_returns_fallback(self, tmp_path):
        """Chat call timeout returns fallback response."""
        import seaman_brain.conversation.manager as mgr_mod

        orig = mgr_mod._LLM_CHAT_TIMEOUT
        mgr_mod._LLM_CHAT_TIMEOUT = 0.1

        try:
            slow_llm = MockLLM()
            # First call (warmup) succeeds, subsequent calls timeout
            call_count = 0

            async def _slow_chat(msgs):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    await asyncio.sleep(999)
                return "ok"

            slow_llm.chat = _slow_chat

            cfg = _make_config(save_path=str(tmp_path / "saves"))
            mgr = ConversationManager(
                config=cfg, llm=slow_llm, creature_state=CreatureState()
            )
            await mgr.initialize()

            result = await mgr.process_input("Hello")
            # Should get fallback response (not hang)
            assert result == "..."
        finally:
            mgr_mod._LLM_CHAT_TIMEOUT = orig

    async def test_autonomous_timeout_returns_none(self, tmp_path):
        """Autonomous LLM call timeout returns None."""
        import seaman_brain.conversation.manager as mgr_mod

        orig = mgr_mod._LLM_CHAT_TIMEOUT
        mgr_mod._LLM_CHAT_TIMEOUT = 0.1

        try:
            slow_llm = MockLLM()
            call_count = 0

            async def _slow_chat(msgs):
                nonlocal call_count
                call_count += 1
                if call_count > 1:
                    await asyncio.sleep(999)
                return "ok"

            slow_llm.chat = _slow_chat

            cfg = _make_config(save_path=str(tmp_path / "saves"))
            mgr = ConversationManager(
                config=cfg, llm=slow_llm, creature_state=CreatureState()
            )
            await mgr.initialize()

            result = await mgr.generate_autonomous_remark("test situation")
            assert result is None
        finally:
            mgr_mod._LLM_CHAT_TIMEOUT = orig

    async def test_stream_timeout_aborts_gracefully(self, tmp_path):
        """Stream aborts when LLM stalls mid-stream, returning partial tokens."""
        import seaman_brain.conversation.manager as mgr_mod

        orig = mgr_mod._LLM_STREAM_TOKEN_TIMEOUT
        mgr_mod._LLM_STREAM_TOKEN_TIMEOUT = 0.1

        try:
            class HangingLLM:
                """LLM that yields a few tokens then hangs forever."""

                chat = AsyncMock(return_value="ok")

                async def stream(self, messages):
                    yield "Hello"
                    yield " world"
                    # Hang forever — simulates Ollama stall
                    await asyncio.sleep(999)
                    yield " never"

            cfg = _make_config(save_path=str(tmp_path / "saves"))
            mgr = ConversationManager(
                config=cfg, llm=HangingLLM(), creature_state=CreatureState()
            )
            await mgr.initialize()

            tokens: list[str] = []
            async for token in mgr.process_input_stream("Hi"):
                tokens.append(token)

            # Should have received partial tokens before the stall
            assert tokens == ["Hello", " world"]

            # Episodic memory should store the partial response
            messages = mgr._episodic.get_all()
            assert len(messages) == 2
            assert messages[1].role == MessageRole.ASSISTANT
            assert messages[1].content == "Hello world"
        finally:
            mgr_mod._LLM_STREAM_TOKEN_TIMEOUT = orig


# ---------------------------------------------------------------------------
# Bloodline switch tests
# ---------------------------------------------------------------------------


class TestSwitchBloodline:
    """Tests for switch_bloodline() — live bloodline switching."""

    async def test_switch_updates_creature_state(self, tmp_path):
        """switch_bloodline updates the creature state to the new one."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        new_state = CreatureState(trust_level=0.99, stage=CreatureStage.GILLMAN)
        mgr.switch_bloodline("alpha", new_state)

        assert mgr.creature_state is new_state
        assert mgr.creature_state.trust_level == pytest.approx(0.99)

    async def test_switch_updates_traits_for_new_stage(self, tmp_path):
        """switch_bloodline updates traits to match the new stage."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()
        old_traits = mgr.traits

        new_state = CreatureState(stage=CreatureStage.GILLMAN)
        mgr.switch_bloodline("beta", new_state)

        assert mgr.traits is not old_traits

    async def test_switch_updates_persistence_path(self, tmp_path):
        """switch_bloodline changes the persistence save directory."""
        save_base = tmp_path / "saves"
        cfg = _make_config(save_path=str(save_base))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        new_state = CreatureState(trust_level=0.5)
        mgr.switch_bloodline("gamma", new_state)

        # Save should go to gamma/ subdirectory
        mgr._save_state()
        assert (save_base / "gamma" / "creature.json").exists()

    async def test_switch_saves_old_state_first(self, tmp_path):
        """switch_bloodline saves current state before switching."""
        save_base = tmp_path / "saves"
        cfg = _make_config(save_path=str(save_base))
        old_state = CreatureState(trust_level=0.33)
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=old_state)
        await mgr.initialize()

        # The old state should be saved to the default bloodline dir
        new_state = CreatureState(trust_level=0.88)
        mgr.switch_bloodline("new_bl", new_state)

        # Verify old state was saved
        assert (save_base / "default" / "creature.json").exists()

    async def test_switch_clears_episodic_memory(self, tmp_path):
        """switch_bloodline clears episodic memory so new creature starts fresh."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        # Add some messages to episodic memory
        mgr._episodic.add(ChatMessage(role=MessageRole.USER, content="Hello old creature"))
        mgr._episodic.add(ChatMessage(role=MessageRole.ASSISTANT, content="Go away."))
        assert len(mgr._episodic) == 2

        new_state = CreatureState(stage=CreatureStage.GILLMAN)
        mgr.switch_bloodline("fresh", new_state)

        assert len(mgr._episodic) == 0


class TestUpdatePersonalityTraits:
    """Tests for update_personality_traits() — live trait hot-swap."""

    async def test_updates_traits(self, tmp_path):
        """update_personality_traits sets a new TraitProfile from dict."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        mgr.update_personality_traits({"cynicism": 0.9, "wit": 0.1, "patience": 0.3})

        assert mgr.traits.cynicism == pytest.approx(0.9)
        assert mgr.traits.wit == pytest.approx(0.1)
        assert mgr.traits.patience == pytest.approx(0.3)

    async def test_ignores_unknown_keys(self, tmp_path):
        """update_personality_traits silently ignores keys not in TraitProfile."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        mgr.update_personality_traits({"cynicism": 0.8, "bogus_key": 999.0})

        assert mgr.traits.cynicism == pytest.approx(0.8)

    async def test_defaults_unset_fields(self, tmp_path):
        """Fields not in the dict get TraitProfile defaults."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=CreatureState())
        await mgr.initialize()

        mgr.update_personality_traits({"cynicism": 0.1})

        assert mgr.traits.cynicism == pytest.approx(0.1)
        # wit should be the default (0.5)
        assert mgr.traits.wit == pytest.approx(0.5)
