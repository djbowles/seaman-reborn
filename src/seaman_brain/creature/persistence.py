"""Creature state save/load persistence via JSON files."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from seaman_brain.creature.state import CreatureState

logger = logging.getLogger(__name__)


@dataclass
class BloodlineInfo:
    """Summary information about a saved bloodline.

    Fields:
        name: Directory name of the bloodline.
        save_dir: Full path to the bloodline's save directory.
        generation_count: Number of generation files in lineage/.
        stage: Current creature stage name.
        last_modified: Last modification time of creature.json.
    """

    name: str
    save_dir: Path
    generation_count: int
    stage: str
    last_modified: datetime


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

    @classmethod
    def migrate_flat_saves(cls, base_dir: str | Path = "data/saves") -> None:
        """Migrate old flat save layout into default/ subdirectory.

        If ``base_dir/creature.json`` exists at the root level (old layout),
        moves it and related files into ``base_dir/default/``.

        Args:
            base_dir: The base saves directory.
        """
        base = Path(base_dir)
        old_save = base / "creature.json"
        if not old_save.exists():
            return

        default_dir = base / "default"
        default_dir.mkdir(parents=True, exist_ok=True)

        # Move creature files into default/ (overwrite orphans even if dir exists)
        for pattern in ("creature.json", "creature.json.bak"):
            src = base / pattern
            if src.exists():
                shutil.move(str(src), str(default_dir / pattern))

        # Move lineage directory (only if not already present in default/)
        old_lineage = base / "lineage"
        if old_lineage.exists() and old_lineage.is_dir():
            dest_lineage = default_dir / "lineage"
            if not dest_lineage.exists():
                shutil.move(str(old_lineage), str(dest_lineage))
            else:
                # Merge individual generation files
                for gen_file in old_lineage.iterdir():
                    shutil.move(str(gen_file), str(dest_lineage / gen_file.name))
                old_lineage.rmdir()

        # Write active marker
        (base / "_active.txt").write_text("default", encoding="utf-8")

        logger.info("Migrated flat saves into %s", default_dir)

    @classmethod
    def list_bloodlines(cls, base_dir: str | Path = "data/saves") -> list[BloodlineInfo]:
        """Scan for bloodline subdirectories containing creature.json.

        Args:
            base_dir: The base saves directory.

        Returns:
            List of BloodlineInfo for each discovered bloodline.
        """
        base = Path(base_dir)
        if not base.exists():
            return []

        bloodlines: list[BloodlineInfo] = []
        for entry in sorted(base.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("_"):
                continue
            creature_file = entry / "creature.json"
            if not creature_file.exists():
                continue

            # Count generation files
            lineage_dir = entry / "lineage"
            gen_count = 0
            if lineage_dir.exists():
                gen_count = sum(
                    1 for f in lineage_dir.iterdir()
                    if f.name.startswith("gen_") and f.suffix == ".json"
                )

            # Read stage from creature.json
            stage = "unknown"
            try:
                data = json.loads(creature_file.read_text(encoding="utf-8"))
                stage = data.get("stage", "unknown")
            except Exception:
                pass

            last_mod = datetime.fromtimestamp(creature_file.stat().st_mtime)

            bloodlines.append(BloodlineInfo(
                name=entry.name,
                save_dir=entry,
                generation_count=gen_count,
                stage=stage,
                last_modified=last_mod,
            ))

        return bloodlines

    @classmethod
    def get_active_bloodline(cls, base_dir: str | Path = "data/saves") -> str:
        """Read the active bloodline name from _active.txt.

        Args:
            base_dir: The base saves directory.

        Returns:
            Name of the active bloodline, or "default".
        """
        active_file = Path(base_dir) / "_active.txt"
        if active_file.exists():
            return active_file.read_text(encoding="utf-8").strip()
        return "default"

    @classmethod
    def set_active_bloodline(
        cls, name: str, base_dir: str | Path = "data/saves"
    ) -> None:
        """Write the active bloodline name to _active.txt.

        Args:
            name: Bloodline name to set as active.
            base_dir: The base saves directory.
        """
        base = Path(base_dir)
        base.mkdir(parents=True, exist_ok=True)
        (base / "_active.txt").write_text(name, encoding="utf-8")
