"""Tests for CLI slash commands."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

from seaman_brain.cli.commands import (
    CommandResult,
    execute_command,
    parse_command,
)
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.state import CreatureState
from seaman_brain.memory.episodic import EpisodicMemory
from seaman_brain.personality.traits import TraitProfile, get_default_profile
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


def _make_manager(
    tmp_path,
    creature_state: CreatureState | None = None,
    traits: TraitProfile | None = None,
) -> ConversationManager:
    """Create a ConversationManager suitable for testing commands."""
    from seaman_brain.config import CreatureConfig, SeamanConfig

    cfg = SeamanConfig(creature=CreatureConfig(save_path=str(tmp_path / "saves")))
    state = creature_state or CreatureState()
    mgr = ConversationManager(config=cfg, llm=MockLLM(), creature_state=state)
    # Manually set internals that initialize() would set
    mgr._initialized = True
    mgr._episodic = EpisodicMemory(max_size=20)
    mgr._traits = traits or get_default_profile(state.stage)
    return mgr


# ---------------------------------------------------------------------------
# parse_command tests
# ---------------------------------------------------------------------------

class TestParseCommand:
    """Tests for parse_command()."""

    def test_parses_simple_command(self):
        """Slash commands are parsed correctly."""
        result = parse_command("/state")
        assert result is not None
        assert result[0] == "/state"
        assert result[1] == []

    def test_parses_command_with_args(self):
        """Commands with arguments are parsed correctly."""
        result = parse_command("/reset --force")
        assert result is not None
        assert result[0] == "/reset"
        assert result[1] == ["--force"]

    def test_returns_none_for_regular_text(self):
        """Regular text is not parsed as a command."""
        assert parse_command("hello") is None
        assert parse_command("") is None
        assert parse_command("  not a command  ") is None

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped."""
        result = parse_command("  /quit  ")
        assert result is not None
        assert result[0] == "/quit"

    def test_command_lowercased(self):
        """Commands are lowercased."""
        result = parse_command("/STATE")
        assert result is not None
        assert result[0] == "/state"

    def test_empty_string(self):
        """Empty input returns None."""
        assert parse_command("") is None

    def test_only_slash(self):
        """Just a slash is parsed as a command."""
        result = parse_command("/")
        assert result is not None
        assert result[0] == "/"


# ---------------------------------------------------------------------------
# execute_command tests - /help
# ---------------------------------------------------------------------------

class TestHelpCommand:
    """Tests for the /help command."""

    def test_help_returns_commands(self, tmp_path):
        """Help shows available commands."""
        mgr = _make_manager(tmp_path)
        result = execute_command("/help", [], mgr)
        assert isinstance(result, CommandResult)
        assert "/state" in result.output
        assert "/quit" in result.output
        assert "/help" in result.output
        assert not result.should_quit

    def test_help_lists_all_commands(self, tmp_path):
        """Help lists every registered command."""
        mgr = _make_manager(tmp_path)
        result = execute_command("/help", [], mgr)
        for cmd in ("/state", "/memory", "/stage", "/traits", "/reset", "/quit", "/help"):
            assert cmd in result.output


# ---------------------------------------------------------------------------
# execute_command tests - /state
# ---------------------------------------------------------------------------

class TestStateCommand:
    """Tests for the /state command."""

    def test_state_shows_summary(self, tmp_path):
        """State command shows creature state summary."""
        state = CreatureState(
            stage=CreatureStage.GILLMAN,
            mood="sardonic",
            trust_level=0.5,
            hunger=0.3,
            health=0.9,
        )
        mgr = _make_manager(tmp_path, creature_state=state)
        result = execute_command("/state", [], mgr)
        assert "Creature State:" in result.output
        assert "Gillman" in result.output or "gillman" in result.output
        assert "0.50" in result.output  # trust
        assert not result.should_quit

    def test_state_no_creature(self, tmp_path):
        """State command handles missing creature state."""
        mgr = _make_manager(tmp_path)
        mgr._creature_state = None
        result = execute_command("/state", [], mgr)
        assert "No creature state" in result.output


# ---------------------------------------------------------------------------
# execute_command tests - /memory
# ---------------------------------------------------------------------------

class TestMemoryCommand:
    """Tests for the /memory command."""

    def test_memory_shows_messages(self, tmp_path):
        """Memory command shows recent messages."""
        mgr = _make_manager(tmp_path)
        mgr._episodic.add(ChatMessage(role=MessageRole.USER, content="hello"))
        mgr._episodic.add(ChatMessage(role=MessageRole.ASSISTANT, content="go away"))
        result = execute_command("/memory", [], mgr)
        assert "[USER]" in result.output
        assert "hello" in result.output
        assert "[ASSISTANT]" in result.output
        assert "go away" in result.output

    def test_memory_empty(self, tmp_path):
        """Memory command handles empty memory."""
        mgr = _make_manager(tmp_path)
        result = execute_command("/memory", [], mgr)
        assert "No messages" in result.output

    def test_memory_not_initialized(self, tmp_path):
        """Memory command handles uninitialized memory."""
        mgr = _make_manager(tmp_path)
        mgr._episodic = None
        result = execute_command("/memory", [], mgr)
        assert "not initialized" in result.output

    def test_memory_truncates_long_messages(self, tmp_path):
        """Long messages are truncated in memory display."""
        mgr = _make_manager(tmp_path)
        long_msg = "x" * 200
        mgr._episodic.add(ChatMessage(role=MessageRole.USER, content=long_msg))
        result = execute_command("/memory", [], mgr)
        assert "..." in result.output


# ---------------------------------------------------------------------------
# execute_command tests - /stage
# ---------------------------------------------------------------------------

class TestStageCommand:
    """Tests for the /stage command."""

    def test_stage_shows_info(self, tmp_path):
        """Stage command shows evolution info."""
        state = CreatureState(
            stage=CreatureStage.PODFISH,
            interaction_count=50,
            trust_level=0.6,
        )
        mgr = _make_manager(tmp_path, creature_state=state)
        result = execute_command("/stage", [], mgr)
        assert "PODFISH" in result.output
        assert "50" in result.output
        assert "0.60" in result.output

    def test_stage_no_creature(self, tmp_path):
        """Stage command handles missing creature state."""
        mgr = _make_manager(tmp_path)
        mgr._creature_state = None
        result = execute_command("/stage", [], mgr)
        assert "No creature state" in result.output


# ---------------------------------------------------------------------------
# execute_command tests - /traits
# ---------------------------------------------------------------------------

class TestTraitsCommand:
    """Tests for the /traits command."""

    def test_traits_shows_all_dimensions(self, tmp_path):
        """Traits command shows all 8 trait dimensions."""
        traits = TraitProfile(cynicism=0.8, wit=0.7, patience=0.2)
        mgr = _make_manager(tmp_path, traits=traits)
        result = execute_command("/traits", [], mgr)
        assert "cynicism" in result.output
        assert "wit" in result.output
        assert "patience" in result.output
        assert "0.80" in result.output

    def test_traits_no_profile(self, tmp_path):
        """Traits command handles missing trait profile."""
        mgr = _make_manager(tmp_path)
        mgr._traits = None
        result = execute_command("/traits", [], mgr)
        assert "No trait profile" in result.output

    def test_traits_visual_bar(self, tmp_path):
        """Traits command shows visual bar indicators."""
        traits = TraitProfile(cynicism=0.5)
        mgr = _make_manager(tmp_path, traits=traits)
        result = execute_command("/traits", [], mgr)
        assert "[#####" in result.output  # 0.5 * 10 = 5 hashes


# ---------------------------------------------------------------------------
# execute_command tests - /reset
# ---------------------------------------------------------------------------

class TestResetCommand:
    """Tests for the /reset command."""

    def test_reset_resets_state(self, tmp_path):
        """Reset command restores creature to defaults."""
        state = CreatureState(
            stage=CreatureStage.FROGMAN,
            interaction_count=999,
            trust_level=0.9,
        )
        mgr = _make_manager(tmp_path, creature_state=state)
        mgr._episodic.add(ChatMessage(role=MessageRole.USER, content="hello"))
        result = execute_command("/reset", [], mgr)
        assert "reset" in result.output.lower()
        assert mgr._creature_state.stage == CreatureStage.MUSHROOMER
        assert mgr._creature_state.interaction_count == 0
        assert len(mgr._episodic) == 0

    def test_reset_updates_traits(self, tmp_path):
        """Reset command updates traits to match new stage."""
        state = CreatureState(stage=CreatureStage.FROGMAN)
        mgr = _make_manager(tmp_path, creature_state=state)
        execute_command("/reset", [], mgr)
        expected = get_default_profile(CreatureStage.MUSHROOMER)
        assert mgr._traits.cynicism == expected.cynicism


# ---------------------------------------------------------------------------
# execute_command tests - /quit
# ---------------------------------------------------------------------------

class TestQuitCommand:
    """Tests for the /quit command."""

    def test_quit_sets_should_quit(self, tmp_path):
        """Quit command signals terminal to exit."""
        mgr = _make_manager(tmp_path)
        result = execute_command("/quit", [], mgr)
        assert result.should_quit is True

    def test_quit_has_message(self, tmp_path):
        """Quit command provides an exit message."""
        mgr = _make_manager(tmp_path)
        result = execute_command("/quit", [], mgr)
        assert "exit" in result.output.lower() or "saving" in result.output.lower()


# ---------------------------------------------------------------------------
# execute_command tests - unknown command
# ---------------------------------------------------------------------------

class TestUnknownCommand:
    """Tests for unknown commands."""

    def test_unknown_command(self, tmp_path):
        """Unknown commands return helpful error."""
        mgr = _make_manager(tmp_path)
        result = execute_command("/foobar", [], mgr)
        assert "Unknown command" in result.output
        assert "/help" in result.output
        assert not result.should_quit
