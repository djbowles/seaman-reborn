"""Creature state save/load persistence via JSON files."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from seaman_brain.creature.state import CreatureState

logger = logging.getLogger(__name__)


class StatePersistence:
    """Handles saving and loading CreatureState to/from JSON files.

    Creates backups of existing save files before overwriting.
    Returns a default CreatureState when no save file exists.
    """

    def __init__(self, save_dir: str | Path = "data/saves") -> None:
        """Initialize persistence with a save directory.

        Args:
            save_dir: Directory where save files are stored.
        """
        self._save_dir = Path(save_dir)

    def save(self, state: CreatureState, filename: str = "creature.json") -> Path:
        """Save creature state to a JSON file.

        Creates the save directory if it doesn't exist.
        Creates a .bak backup of any existing save file before overwriting.

        Args:
            state: The creature state to persist.
            filename: Name of the save file.

        Returns:
            Path to the saved file.

        Raises:
            OSError: If the file cannot be written.
        """
        self._save_dir.mkdir(parents=True, exist_ok=True)
        save_path = self._save_dir / filename

        # Backup existing save before overwriting
        if save_path.exists():
            backup_path = save_path.with_suffix(".json.bak")
            shutil.copy2(save_path, backup_path)
            logger.debug("Backed up %s -> %s", save_path, backup_path)

        data = state.to_dict()
        save_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.debug("Saved creature state to %s", save_path)
        return save_path

    def load(self, filename: str = "creature.json") -> CreatureState:
        """Load creature state from a JSON file.

        Returns a default CreatureState if the file doesn't exist.

        Args:
            filename: Name of the save file to load.

        Returns:
            Loaded CreatureState, or default if file not found.

        Raises:
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If the data contains invalid field values.
        """
        save_path = self._save_dir / filename

        if not save_path.exists():
            logger.info("No save file at %s, returning default state", save_path)
            return CreatureState()

        raw = save_path.read_text(encoding="utf-8")
        data = json.loads(raw)
        logger.debug("Loaded creature state from %s", save_path)
        return CreatureState.from_dict(data)

    def exists(self, filename: str = "creature.json") -> bool:
        """Check if a save file exists.

        Args:
            filename: Name of the save file to check.

        Returns:
            True if the save file exists.
        """
        return (self._save_dir / filename).exists()

    def delete(self, filename: str = "creature.json") -> bool:
        """Delete a save file.

        Args:
            filename: Name of the save file to delete.

        Returns:
            True if the file was deleted, False if it didn't exist.
        """
        save_path = self._save_dir / filename
        if save_path.exists():
            save_path.unlink()
            logger.debug("Deleted save file %s", save_path)
            return True
        return False
