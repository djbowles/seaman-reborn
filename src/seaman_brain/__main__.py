"""CLI entry point for Seaman Brain.

Usage:
    python -m seaman_brain            # Launch interactive terminal
    python -m seaman_brain --gui      # Launch Pygame graphical interface
    python -m seaman_brain --api      # Launch WebSocket API server
    python -m seaman_brain --version  # Print version
"""

import argparse
import logging
from pathlib import Path

from seaman_brain import __version__

_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_LOG_DATE_FORMAT = "%H:%M:%S"


def _setup_logging(*, debug: bool = False) -> None:
    """Configure logging to console and optionally to data/seaman.log."""
    level = logging.DEBUG if debug else logging.INFO

    handlers: list[logging.Handler] = [logging.StreamHandler()]

    # File handler — write to data/seaman.log if data/ exists
    log_dir = Path("data")
    if log_dir.is_dir():
        file_handler = logging.FileHandler(
            log_dir / "seaman.log", encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        datefmt=_LOG_DATE_FORMAT,
        handlers=handlers,
    )


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
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging",
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

    _setup_logging(debug=args.debug)

    if args.gui:
        from seaman_brain.gui.game_loop import GameEngine

        engine = GameEngine()
        engine.run()
    elif args.api:
        from seaman_brain.api.server import BrainServer

        server = BrainServer()
        server.run()
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
