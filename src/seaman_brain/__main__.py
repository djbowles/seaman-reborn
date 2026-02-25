"""CLI entry point for Seaman Brain.

Usage:
    python -m seaman_brain            # Launch interactive terminal
    python -m seaman_brain --gui      # Launch Pygame graphical interface
    python -m seaman_brain --api      # Launch WebSocket API server
    python -m seaman_brain --version  # Print version
"""

import argparse
import sys

from seaman_brain import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seaman-brain",
        description="Seaman Reborn - sardonic creature cognition engine",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"seaman-brain {__version__}",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--gui",
        action="store_true",
        help="Launch Pygame graphical interface",
    )
    mode.add_argument(
        "--api",
        action="store_true",
        help="Launch WebSocket API server for UE5 bridge",
    )
    args = parser.parse_args()

    if args.gui:
        from seaman_brain.gui.window import GameWindow

        window = GameWindow()
        window.run()
    elif args.api:
        # TODO: Launch API server (US-043)
        print(f"Seaman Brain v{__version__} - API server mode not yet implemented")
        sys.exit(0)
    else:
        # Launch interactive terminal
        import asyncio

        from seaman_brain.cli.terminal import SeamanTerminal

        async def _run_terminal() -> None:
            terminal = SeamanTerminal()
            await terminal.initialize()
            await terminal.run()

        asyncio.run(_run_terminal())


if __name__ == "__main__":
    main()
