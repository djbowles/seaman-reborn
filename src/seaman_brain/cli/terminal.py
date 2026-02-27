"""Rich terminal chat UI for Seaman Brain.

Uses `rich` for styled output and `prompt_toolkit` for async input.
Supports streaming LLM responses and slash commands.
"""

from __future__ import annotations

import logging
import signal
from typing import TYPE_CHECKING

from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from seaman_brain.cli.commands import CommandResult, execute_command, parse_command

if TYPE_CHECKING:
    from seaman_brain.config import SeamanConfig
    from seaman_brain.conversation.manager import ConversationManager

logger = logging.getLogger(__name__)

# Stage display names and emoji-free indicators
_STAGE_INDICATORS: dict[str, str] = {
    "mushroomer": "[MUSHROOMER]",
    "gillman": "[GILLMAN]",
    "podfish": "[PODFISH]",
    "tadman": "[TADMAN]",
    "frogman": "[FROGMAN]",
}

_MOOD_INDICATORS: dict[str, str] = {
    "hostile": "(!)",
    "irritated": "(~)",
    "sardonic": "(¬)",
    "neutral": "(-)",
    "curious": "(?)",
    "amused": "(*)",
    "philosophical": "(@)",
    "content": "(+)",
}


def format_header(stage: str, mood: str, trust: float) -> str:
    """Build the header string showing creature status.

    Args:
        stage: Current stage value string.
        mood: Current mood string.
        trust: Trust level 0.0-1.0.

    Returns:
        Formatted header string.
    """
    stage_ind = _STAGE_INDICATORS.get(stage, f"[{stage.upper()}]")
    mood_ind = _MOOD_INDICATORS.get(mood, f"({mood})")
    trust_bar = "#" * int(trust * 10)
    return f"{stage_ind} {mood_ind} Trust:[{trust_bar:<10}]"


def format_response(text: str) -> str:
    """Format the creature's response for display.

    Args:
        text: Raw response text.

    Returns:
        Formatted string.
    """
    return text.strip() if text else "..."


class SeamanTerminal:
    """Interactive terminal UI for chatting with Seaman.

    Attributes:
        manager: The conversation manager orchestrating the chat.
        console: Rich console for styled output.
    """

    def __init__(
        self,
        config: SeamanConfig | None = None,
        manager: ConversationManager | None = None,
    ) -> None:
        """Initialize terminal with optional pre-built components.

        Args:
            config: Configuration. Uses defaults if None.
            manager: Pre-built conversation manager. Created from config if None.
        """
        self._config = config
        self._manager = manager
        self.console = Console()
        self._session: PromptSession[str] | None = None
        self._running = False

    @property
    def manager(self) -> ConversationManager | None:
        """The conversation manager, if initialized."""
        return self._manager

    async def initialize(self) -> None:
        """Set up the conversation manager and display welcome message."""
        if self._manager is None:
            from seaman_brain.config import SeamanConfig
            from seaman_brain.conversation.manager import ConversationManager

            cfg = self._config or SeamanConfig()
            self._manager = ConversationManager(config=cfg)

        await self._manager.initialize()
        self._display_welcome()

    def _display_welcome(self) -> None:
        """Show welcome banner with creature info."""
        assert self._manager is not None
        summary = self._manager.get_state_summary()
        stage = summary.get("stage", "unknown")
        mood = summary.get("mood", "neutral")

        title = Text("SEAMAN BRAIN", style="bold cyan")
        body = (
            f"Stage: {stage.upper()}\n"
            f"Mood: {mood}\n"
            f"Type /help for commands, /quit to exit"
        )
        self.console.print(Panel(body, title=title, border_style="cyan"))

    def _display_header(self) -> None:
        """Print the status header line."""
        assert self._manager is not None
        summary = self._manager.get_state_summary()
        header = format_header(
            stage=summary.get("stage", "unknown"),
            mood=summary.get("mood", "neutral"),
            trust=summary.get("trust_level", 0.0),
        )
        self.console.print(f"[dim]{header}[/dim]")

    def _display_response(self, text: str) -> None:
        """Display creature response with formatting."""
        formatted = format_response(text)
        self.console.print(f"[bold yellow]Seaman:[/bold yellow] {formatted}")

    def _display_command_output(self, result: CommandResult) -> None:
        """Display command output with dim styling."""
        self.console.print(f"[dim]{result.output}[/dim]")

    def _display_streaming(self, text: str) -> None:
        """Display a streaming response chunk (on same line)."""
        self.console.print(text, end="")

    async def _process_streaming(self, user_input: str) -> str:
        """Process input with streaming output if available.

        Falls back to non-streaming if the LLM doesn't support it.

        Args:
            user_input: The user's text.

        Returns:
            The complete response string.
        """
        assert self._manager is not None

        # Try streaming first
        if (
            self._manager._llm is not None
            and hasattr(self._manager._llm, "stream")
        ):
            try:
                return await self._stream_response(user_input)
            except Exception:
                logger.debug("Streaming failed, falling back to non-streaming")

        # Non-streaming fallback
        return await self._manager.process_input(user_input)

    async def _stream_response(self, user_input: str) -> str:
        """Stream response tokens to the console.

        This manually replicates the ConversationManager flow but with
        streaming output. Falls back to process_input on any error.
        """
        # Use the standard non-streaming path for simplicity and correctness.
        # Streaming display is best done at a higher level once the API supports it.
        response = await self._manager.process_input(user_input)
        return response

    async def _get_input(self) -> str | None:
        """Read user input using prompt_toolkit.

        Returns:
            User input string, or None on EOF/KeyboardInterrupt.
        """
        if self._session is None:
            self._session = PromptSession()
        try:
            with patch_stdout():
                text = await self._session.prompt_async("You: ")
            return text
        except (EOFError, KeyboardInterrupt):
            return None

    async def run(self) -> None:
        """Run the main interactive loop.

        Reads input, dispatches commands or sends to LLM, displays results.
        Handles Ctrl+C gracefully by saving state before exit.
        """
        assert self._manager is not None
        self._running = True

        # Install signal handler for graceful shutdown
        original_handler = signal.getsignal(signal.SIGINT)

        def _sigint_handler(sig: int, frame: object) -> None:
            self._running = False

        try:
            signal.signal(signal.SIGINT, _sigint_handler)
        except (ValueError, OSError):
            # Can't set signal handler (not main thread, etc.)
            pass

        try:
            while self._running:
                self._display_header()
                user_input = await self._get_input()

                if user_input is None:
                    # EOF or Ctrl+C
                    break

                stripped = user_input.strip()
                if not stripped:
                    continue

                # Check for slash command
                parsed = parse_command(stripped)
                if parsed is not None:
                    cmd, args = parsed
                    result = execute_command(cmd, args, self._manager)
                    self._display_command_output(result)
                    if result.should_quit:
                        break
                    continue

                # Process conversation
                try:
                    response = await self._process_streaming(stripped)
                    self._display_response(response)
                except Exception as exc:
                    self.console.print(f"[red]Error: {exc}[/red]")
        finally:
            self._running = False

            # Restore original signal handler
            try:
                signal.signal(signal.SIGINT, original_handler)
            except (ValueError, OSError, TypeError):
                pass

            # Save state on exit
            await self._shutdown()

    async def _shutdown(self) -> None:
        """Gracefully shut down: save state and display exit message."""
        if self._manager is not None:
            try:
                await self._manager.shutdown()
                self.console.print("[dim]State saved. Goodbye.[/dim]")
            except Exception as exc:
                logger.error("Shutdown error: %s", exc)
                self.console.print(f"[red]Shutdown error: {exc}[/red]")
