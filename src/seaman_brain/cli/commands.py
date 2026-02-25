"""Debug slash commands for the terminal interface.

Commands:
    /state  - Show creature state summary
    /memory - Show recent episodic memory
    /stage  - Show current evolution stage info
    /traits - Show current trait profile
    /reset  - Reset creature to default state
    /quit   - Save and exit
    /help   - Show available commands
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from seaman_brain.conversation.manager import ConversationManager


@dataclass
class CommandResult:
    """Result of executing a slash command.

    Attributes:
        output: Text to display to the user.
        should_quit: Whether the terminal should exit after this command.
    """

    output: str
    should_quit: bool = False


def parse_command(text: str) -> tuple[str, list[str]] | None:
    """Parse a slash command from user input.

    Args:
        text: Raw user input string.

    Returns:
        Tuple of (command_name, args) if input is a command, None otherwise.
    """
    stripped = text.strip()
    if not stripped.startswith("/"):
        return None
    parts = stripped.split()
    command = parts[0].lower()
    args = parts[1:]
    return (command, args)


def _cmd_help() -> CommandResult:
    """Show available commands."""
    lines = [
        "Available commands:",
        "  /state  - Show creature state (stage, mood, trust, needs)",
        "  /memory - Show recent conversation history",
        "  /stage  - Show current evolution stage info",
        "  /traits - Show current personality traits",
        "  /reset  - Reset creature to default state",
        "  /quit   - Save state and exit",
        "  /help   - Show this help message",
    ]
    return CommandResult(output="\n".join(lines))


def _cmd_state(manager: ConversationManager) -> CommandResult:
    """Show creature state summary."""
    summary = manager.get_state_summary()
    if not summary:
        return CommandResult(output="No creature state available.")
    lines = ["Creature State:"]
    for key, value in summary.items():
        label = key.replace("_", " ").title()
        if isinstance(value, float):
            lines.append(f"  {label}: {value:.2f}")
        else:
            lines.append(f"  {label}: {value}")
    return CommandResult(output="\n".join(lines))


def _cmd_memory(manager: ConversationManager) -> CommandResult:
    """Show recent episodic memory."""
    if manager._episodic is None:
        return CommandResult(output="Memory not initialized.")
    messages = manager._episodic.get_recent(10)
    if not messages:
        return CommandResult(output="No messages in memory.")
    lines = ["Recent Memory (last 10):"]
    for msg in messages:
        role = msg.role.value.upper()
        content = msg.content[:80]
        if len(msg.content) > 80:
            content += "..."
        lines.append(f"  [{role}] {content}")
    return CommandResult(output="\n".join(lines))


def _cmd_stage(manager: ConversationManager) -> CommandResult:
    """Show current evolution stage info."""
    state = manager.creature_state
    if state is None:
        return CommandResult(output="No creature state available.")
    stage = state.stage
    lines = [
        f"Evolution Stage: {stage.value.upper()}",
        f"  Interaction Count: {state.interaction_count}",
        f"  Trust Level: {state.trust_level:.2f}",
    ]
    if manager._evolution is not None:
        remaining = manager._evolution.stages_remaining(state)
        lines.append(f"  Stages Remaining: {remaining}")
    return CommandResult(output="\n".join(lines))


def _cmd_traits(manager: ConversationManager) -> CommandResult:
    """Show current personality traits."""
    traits = manager.traits
    if traits is None:
        return CommandResult(output="No trait profile available.")
    lines = ["Personality Traits:"]
    for name in (
        "cynicism", "wit", "patience", "curiosity",
        "warmth", "verbosity", "formality", "aggression",
    ):
        value = getattr(traits, name, 0.0)
        bar = "#" * int(value * 10)
        lines.append(f"  {name:>12}: {value:.2f} [{bar:<10}]")
    return CommandResult(output="\n".join(lines))


def _cmd_reset(manager: ConversationManager) -> CommandResult:
    """Reset creature to default state."""
    from seaman_brain.creature.state import CreatureState
    from seaman_brain.personality.traits import get_default_profile

    new_state = CreatureState()
    manager._creature_state = new_state
    manager._traits = get_default_profile(new_state.stage)
    if manager._episodic is not None:
        manager._episodic.clear()
    return CommandResult(output="Creature state reset to defaults.")


def _cmd_quit() -> CommandResult:
    """Signal the terminal to save and exit."""
    return CommandResult(output="Saving state and exiting...", should_quit=True)


# Command dispatch table
_COMMANDS: dict[str, str] = {
    "/help": "help",
    "/state": "state",
    "/memory": "memory",
    "/stage": "stage",
    "/traits": "traits",
    "/reset": "reset",
    "/quit": "quit",
}


def execute_command(
    command: str,
    _args: list[str],
    manager: ConversationManager,
) -> CommandResult:
    """Execute a parsed slash command.

    Args:
        command: The command string (e.g. "/state").
        _args: Additional arguments (currently unused).
        manager: The conversation manager instance.

    Returns:
        CommandResult with output text and optional quit flag.
    """
    handler = _COMMANDS.get(command)
    if handler is None:
        msg = f"Unknown command: {command}. Type /help for available commands."
        return CommandResult(output=msg)

    if handler == "help":
        return _cmd_help()
    if handler == "state":
        return _cmd_state(manager)
    if handler == "memory":
        return _cmd_memory(manager)
    if handler == "stage":
        return _cmd_stage(manager)
    if handler == "traits":
        return _cmd_traits(manager)
    if handler == "reset":
        return _cmd_reset(manager)
    if handler == "quit":
        return _cmd_quit()

    return CommandResult(output=f"Unknown command: {command}. Type /help for available commands.")
