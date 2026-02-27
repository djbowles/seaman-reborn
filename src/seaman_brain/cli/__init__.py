"""CLI subsystem - terminal interface and debug commands."""

from seaman_brain.cli.commands import CommandResult, execute_command, parse_command
from seaman_brain.cli.terminal import SeamanTerminal

__all__ = [
    "CommandResult",
    "SeamanTerminal",
    "execute_command",
    "parse_command",
]
