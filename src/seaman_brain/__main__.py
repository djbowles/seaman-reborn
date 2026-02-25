"""CLI entry point for Seaman Brain.

Usage:
    python -m seaman_brain          # Launch interactive terminal
    python -m seaman_brain --version  # Print version
"""

import argparse
import sys

from seaman_brain import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="seaman-brain",
        description="Seaman Reborn - AI Brain Core",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"seaman-brain {__version__}",
    )
    parser.parse_args()

    # TODO: Launch terminal UI (US-021)
    print(f"Seaman Brain v{__version__} - interactive mode not yet implemented")
    sys.exit(0)


if __name__ == "__main__":
    main()
