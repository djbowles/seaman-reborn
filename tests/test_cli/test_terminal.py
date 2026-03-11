"""Tests for the SeamanTerminal interactive UI."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

from seaman_brain.cli.terminal import SeamanTerminal, format_header, format_response
from seaman_brain.config import CreatureConfig, SeamanConfig
from seaman_brain.conversation.manager import ConversationManager
from seaman_brain.creature.state import CreatureState
from seaman_brain.types import ChatMessage, CreatureStage

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


def _make_config(save_path: str) -> SeamanConfig:
    return SeamanConfig(creature=CreatureConfig(save_path=save_path))


def _make_terminal(tmp_path, state: CreatureState | None = None) -> SeamanTerminal:
    """Create a SeamanTerminal with a pre-built manager for testing."""
    cfg = _make_config(save_path=str(tmp_path / "saves"))
    mgr = ConversationManager(
        config=cfg,
        llm=MockLLM(),
        creature_state=state or CreatureState(),
    )
    return SeamanTerminal(config=cfg, manager=mgr)


# ---------------------------------------------------------------------------
# format_header tests
# ---------------------------------------------------------------------------

class TestFormatHeader:
    """Tests for format_header()."""

    def test_basic_header(self):
        """Header includes stage, mood, and trust indicators."""
        header = format_header("mushroomer", "neutral", 0.5)
        assert "[MUSHROOMER]" in header
        assert "(-)" in header
        assert "Trust:" in header

    def test_different_stage(self):
        """Header shows correct stage indicator."""
        header = format_header("frogman", "content", 1.0)
        assert "[FROGMAN]" in header
        assert "(+)" in header

    def test_unknown_stage(self):
        """Unknown stages fall back to uppercase name."""
        header = format_header("unknown_stage", "neutral", 0.0)
        assert "[UNKNOWN_STAGE]" in header

    def test_unknown_mood(self):
        """Unknown moods fall back to raw name."""
        header = format_header("mushroomer", "confused", 0.5)
        assert "(confused)" in header

    def test_trust_bar_empty(self):
        """Zero trust shows empty bar."""
        header = format_header("mushroomer", "neutral", 0.0)
        assert "[          ]" in header

    def test_trust_bar_full(self):
        """Full trust shows full bar."""
        header = format_header("mushroomer", "neutral", 1.0)
        assert "[##########]" in header

    def test_trust_bar_partial(self):
        """Partial trust shows proportional bar."""
        header = format_header("mushroomer", "neutral", 0.3)
        assert "[###       ]" in header

    def test_all_stages_have_indicators(self):
        """Every CreatureStage has a defined indicator."""
        for stage in CreatureStage:
            header = format_header(stage.value, "neutral", 0.5)
            assert f"[{stage.value.upper()}]" in header


# ---------------------------------------------------------------------------
# format_response tests
# ---------------------------------------------------------------------------

class TestFormatResponse:
    """Tests for format_response()."""

    def test_strips_whitespace(self):
        """Response text is stripped."""
        assert format_response("  hello  ") == "hello"

    def test_empty_string_returns_dots(self):
        """Empty response returns '...'."""
        assert format_response("") == "..."

    def test_none_returns_dots(self):
        """None response returns '...'."""
        assert format_response(None) == "..."

    def test_preserves_content(self):
        """Normal content is preserved."""
        text = "Go away, I'm sleeping."
        assert format_response(text) == text


# ---------------------------------------------------------------------------
# SeamanTerminal initialization tests
# ---------------------------------------------------------------------------

class TestTerminalInit:
    """Tests for SeamanTerminal initialization."""

    def test_creates_with_manager(self, tmp_path):
        """Terminal can be created with a pre-built manager."""
        terminal = _make_terminal(tmp_path)
        assert terminal.manager is not None

    def test_creates_without_manager(self):
        """Terminal can be created without a manager (created on initialize)."""
        terminal = SeamanTerminal()
        assert terminal.manager is None

    async def test_initialize_creates_manager(self, tmp_path):
        """Initialize creates a ConversationManager if none provided."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        terminal = SeamanTerminal(config=cfg)
        # Patch ConversationManager at the source to avoid real Ollama
        with patch(
            "seaman_brain.conversation.manager.create_provider",
            side_effect=ImportError("mocked"),
        ):
            terminal.console = MagicMock()
            await terminal.initialize()
            assert terminal.manager is not None

    async def test_initialize_calls_manager_init(self, tmp_path):
        """Initialize calls manager.initialize()."""
        terminal = _make_terminal(tmp_path)
        # Mock the console to suppress output
        terminal.console = MagicMock()
        await terminal.initialize()
        assert terminal.manager.is_initialized


# ---------------------------------------------------------------------------
# SeamanTerminal display tests
# ---------------------------------------------------------------------------

class TestTerminalDisplay:
    """Tests for terminal display methods."""

    async def test_display_welcome(self, tmp_path):
        """Welcome message is displayed on initialize."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()
        # Console.print should have been called with the welcome panel
        terminal.console.print.assert_called()
        call_args = str(terminal.console.print.call_args_list)
        assert "MUSHROOMER" in call_args or "Panel" in call_args

    def test_display_header(self, tmp_path):
        """Header display shows creature status."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        terminal._manager._initialized = True
        terminal._display_header()
        terminal.console.print.assert_called_once()

    def test_display_response(self, tmp_path):
        """Response display shows creature prefix."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        terminal._display_response("Go away.")
        call_args = str(terminal.console.print.call_args_list)
        assert "Seaman" in call_args
        assert "Go away" in call_args


# ---------------------------------------------------------------------------
# SeamanTerminal run loop tests
# ---------------------------------------------------------------------------

class TestTerminalRun:
    """Tests for the main run loop."""

    async def test_quit_command_exits(self, tmp_path):
        """Typing /quit exits the run loop."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        # Simulate user typing /quit
        terminal._get_input = AsyncMock(side_effect=["/quit"])
        await terminal.run()
        # Should have saved state
        assert not terminal._running

    async def test_eof_exits(self, tmp_path):
        """EOF (Ctrl+D) exits the run loop."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        # Simulate EOF
        terminal._get_input = AsyncMock(return_value=None)
        await terminal.run()
        assert not terminal._running

    async def test_empty_input_skipped(self, tmp_path):
        """Empty input is ignored and loop continues."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        # Simulate empty then quit
        terminal._get_input = AsyncMock(side_effect=["", "  ", "/quit"])
        await terminal.run()
        # Should reach quit after skipping empties

    async def test_regular_input_processed(self, tmp_path):
        """Regular text is sent to the conversation manager."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        # Simulate user message then quit
        terminal._get_input = AsyncMock(side_effect=["hello", "/quit"])
        await terminal.run()

        # LLM should have been called (router wraps the mock, access via _local)
        llm = terminal.manager._llm
        if hasattr(llm, '_local'):
            assert llm._local.chat.called
        else:
            assert llm.chat.called

    async def test_command_output_displayed(self, tmp_path):
        """Command results are displayed."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        terminal._get_input = AsyncMock(side_effect=["/help", "/quit"])
        await terminal.run()

        # Should have printed command output
        calls = [str(c) for c in terminal.console.print.call_args_list]
        combined = " ".join(calls)
        assert "/state" in combined or "Available" in combined

    async def test_llm_error_displays_message(self, tmp_path):
        """LLM errors show error message to user."""
        cfg = _make_config(save_path=str(tmp_path / "saves"))
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        mgr = ConversationManager(config=cfg, llm=llm, creature_state=CreatureState())
        terminal = SeamanTerminal(config=cfg, manager=mgr)
        terminal.console = MagicMock()
        await terminal.initialize()

        terminal._get_input = AsyncMock(side_effect=["hello", "/quit"])
        await terminal.run()

        # The fallback response "..." should be displayed (from manager's graceful degradation)
        # or an error message
        calls = str(terminal.console.print.call_args_list)
        # Manager catches LLM errors and returns fallback, so "Seaman" should appear
        assert "Seaman" in calls or "Error" in calls


# ---------------------------------------------------------------------------
# SeamanTerminal shutdown tests
# ---------------------------------------------------------------------------

class TestTerminalShutdown:
    """Tests for graceful shutdown."""

    async def test_shutdown_saves_state(self, tmp_path):
        """Shutdown calls manager.shutdown()."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        # Mock shutdown to track calls
        terminal.manager.shutdown = AsyncMock()

        await terminal._shutdown()
        terminal.manager.shutdown.assert_called_once()

    async def test_shutdown_displays_message(self, tmp_path):
        """Shutdown displays goodbye message."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        await terminal._shutdown()
        calls = str(terminal.console.print.call_args_list)
        assert "Goodbye" in calls or "saved" in calls.lower()

    async def test_shutdown_error_handled(self, tmp_path):
        """Shutdown errors are caught and displayed."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        terminal.manager.shutdown = AsyncMock(side_effect=RuntimeError("boom"))
        await terminal._shutdown()
        calls = str(terminal.console.print.call_args_list)
        assert "error" in calls.lower() or "boom" in calls.lower()

    async def test_run_saves_on_exit(self, tmp_path):
        """Run loop saves state when exiting via /quit."""
        terminal = _make_terminal(tmp_path)
        terminal.console = MagicMock()
        await terminal.initialize()

        terminal._get_input = AsyncMock(side_effect=["/quit"])
        terminal._shutdown = AsyncMock()
        await terminal.run()
        terminal._shutdown.assert_called_once()
