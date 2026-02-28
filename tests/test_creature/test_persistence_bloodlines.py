"""Tests for bloodline management in creature/persistence.py.

Covers migration from flat saves to subdirectory layout,
list_bloodlines scanning, active bloodline tracking, and
multi-directory save management.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from seaman_brain.creature.persistence import BloodlineInfo, StatePersistence
from seaman_brain.creature.state import CreatureState


@pytest.fixture()
def base_dir(tmp_path: Path) -> Path:
    """Create a temporary base saves directory."""
    saves = tmp_path / "saves"
    saves.mkdir()
    return saves


def _write_creature(directory: Path, stage: str = "mushroomer") -> None:
    """Helper to write a minimal creature.json in a directory."""
    directory.mkdir(parents=True, exist_ok=True)
    state = CreatureState()
    data = state.to_dict()
    data["stage"] = stage
    (directory / "creature.json").write_text(json.dumps(data), encoding="utf-8")


class TestMigrateFlatSaves:
    """Tests for StatePersistence.migrate_flat_saves."""

    def test_migrates_old_flat_layout(self, base_dir: Path):
        """Migrates creature.json from root into default/ subdirectory."""
        # Write old-style flat save
        state = CreatureState()
        (base_dir / "creature.json").write_text(
            json.dumps(state.to_dict()), encoding="utf-8"
        )
        (base_dir / "creature.json.bak").write_text("{}", encoding="utf-8")

        # Create old lineage dir
        lineage = base_dir / "lineage"
        lineage.mkdir()
        (lineage / "gen_1.json").write_text("{}", encoding="utf-8")

        StatePersistence.migrate_flat_saves(base_dir)

        # Old files should be gone
        assert not (base_dir / "creature.json").exists()
        assert not (base_dir / "creature.json.bak").exists()
        assert not (base_dir / "lineage").exists()

        # New structure should exist
        default = base_dir / "default"
        assert (default / "creature.json").exists()
        assert (default / "creature.json.bak").exists()
        assert (default / "lineage" / "gen_1.json").exists()

        # Active marker
        assert (base_dir / "_active.txt").read_text(encoding="utf-8") == "default"

    def test_no_migration_when_no_flat_save(self, base_dir: Path):
        """No migration when creature.json doesn't exist at root."""
        StatePersistence.migrate_flat_saves(base_dir)
        assert not (base_dir / "default").exists()

    def test_merges_when_default_exists(self, base_dir: Path):
        """Merges orphaned root files into existing default/."""
        # Root has a newer creature.json
        (base_dir / "creature.json").write_text(
            '{"stage":"podfish"}', encoding="utf-8"
        )
        (base_dir / "creature.json.bak").write_text("{}", encoding="utf-8")

        # default/ already has an older save
        (base_dir / "default").mkdir()
        (base_dir / "default" / "creature.json").write_text(
            '{"stage":"gillman"}', encoding="utf-8"
        )

        StatePersistence.migrate_flat_saves(base_dir)

        # Root files should be gone (moved into default/)
        assert not (base_dir / "creature.json").exists()
        assert not (base_dir / "creature.json.bak").exists()

        # default/ should have the root's files (overwritten)
        data = json.loads(
            (base_dir / "default" / "creature.json").read_text(encoding="utf-8")
        )
        assert data["stage"] == "podfish"
        assert (base_dir / "default" / "creature.json.bak").exists()

    def test_merges_lineage_when_both_exist(self, base_dir: Path):
        """Merges lineage files when both root and default have lineage/."""
        (base_dir / "creature.json").write_text("{}", encoding="utf-8")

        # Root lineage with gen_3
        root_lineage = base_dir / "lineage"
        root_lineage.mkdir()
        (root_lineage / "gen_3.json").write_text("{}", encoding="utf-8")

        # default/ already has lineage with gen_1 and gen_2
        default_dir = base_dir / "default"
        default_dir.mkdir()
        (default_dir / "creature.json").write_text("{}", encoding="utf-8")
        default_lineage = default_dir / "lineage"
        default_lineage.mkdir()
        (default_lineage / "gen_1.json").write_text("{}", encoding="utf-8")
        (default_lineage / "gen_2.json").write_text("{}", encoding="utf-8")

        StatePersistence.migrate_flat_saves(base_dir)

        # Root lineage should be gone
        assert not root_lineage.exists()
        # All generation files should be in default/lineage/
        assert (default_lineage / "gen_1.json").exists()
        assert (default_lineage / "gen_2.json").exists()
        assert (default_lineage / "gen_3.json").exists()


class TestListBloodlines:
    """Tests for StatePersistence.list_bloodlines."""

    def test_returns_empty_for_nonexistent_dir(self, tmp_path: Path):
        """Returns empty list when dir doesn't exist."""
        result = StatePersistence.list_bloodlines(tmp_path / "nonexistent")
        assert result == []

    def test_returns_empty_for_no_bloodlines(self, base_dir: Path):
        """Returns empty list when no subdirectories have creature.json."""
        result = StatePersistence.list_bloodlines(base_dir)
        assert result == []

    def test_finds_single_bloodline(self, base_dir: Path):
        """Finds a single bloodline subdirectory."""
        _write_creature(base_dir / "default")
        result = StatePersistence.list_bloodlines(base_dir)
        assert len(result) == 1
        assert result[0].name == "default"
        assert result[0].stage == "mushroomer"

    def test_finds_multiple_bloodlines(self, base_dir: Path):
        """Finds multiple bloodline subdirectories."""
        _write_creature(base_dir / "alpha", stage="mushroomer")
        _write_creature(base_dir / "beta", stage="gillman")
        _write_creature(base_dir / "gamma", stage="podfish")

        result = StatePersistence.list_bloodlines(base_dir)
        assert len(result) == 3
        names = {bl.name for bl in result}
        assert names == {"alpha", "beta", "gamma"}

    def test_ignores_underscore_dirs(self, base_dir: Path):
        """Ignores directories starting with underscore."""
        _write_creature(base_dir / "default")
        (base_dir / "_cache").mkdir()
        (base_dir / "_cache" / "creature.json").write_text("{}", encoding="utf-8")

        result = StatePersistence.list_bloodlines(base_dir)
        names = [bl.name for bl in result]
        assert "_cache" not in names

    def test_ignores_dirs_without_creature_json(self, base_dir: Path):
        """Ignores directories without creature.json."""
        _write_creature(base_dir / "valid")
        (base_dir / "empty_dir").mkdir()

        result = StatePersistence.list_bloodlines(base_dir)
        assert len(result) == 1
        assert result[0].name == "valid"

    def test_counts_generation_files(self, base_dir: Path):
        """Counts gen_N.json files in lineage/ subdirectory."""
        _write_creature(base_dir / "default")
        lineage = base_dir / "default" / "lineage"
        lineage.mkdir()
        (lineage / "gen_1.json").write_text("{}", encoding="utf-8")
        (lineage / "gen_2.json").write_text("{}", encoding="utf-8")
        (lineage / "family_tree.json").write_text("{}", encoding="utf-8")

        result = StatePersistence.list_bloodlines(base_dir)
        assert result[0].generation_count == 2

    def test_handles_corrupt_creature_json(self, base_dir: Path):
        """Handles corrupted creature.json gracefully."""
        bl_dir = base_dir / "corrupt"
        bl_dir.mkdir()
        (bl_dir / "creature.json").write_text("not valid json!!!", encoding="utf-8")

        result = StatePersistence.list_bloodlines(base_dir)
        assert len(result) == 1
        assert result[0].stage == "unknown"

    def test_bloodline_info_has_last_modified(self, base_dir: Path):
        """BloodlineInfo includes a last_modified timestamp."""
        _write_creature(base_dir / "default")
        result = StatePersistence.list_bloodlines(base_dir)
        assert result[0].last_modified is not None


class TestActiveBloodline:
    """Tests for active bloodline tracking."""

    def test_get_active_default_when_no_file(self, base_dir: Path):
        """Returns 'default' when _active.txt doesn't exist."""
        result = StatePersistence.get_active_bloodline(base_dir)
        assert result == "default"

    def test_get_active_reads_file(self, base_dir: Path):
        """Reads active name from _active.txt."""
        (base_dir / "_active.txt").write_text("alpha", encoding="utf-8")
        result = StatePersistence.get_active_bloodline(base_dir)
        assert result == "alpha"

    def test_set_active_writes_file(self, base_dir: Path):
        """set_active_bloodline writes to _active.txt."""
        StatePersistence.set_active_bloodline("beta", base_dir)
        content = (base_dir / "_active.txt").read_text(encoding="utf-8")
        assert content == "beta"

    def test_set_active_creates_dir(self, tmp_path: Path):
        """set_active_bloodline creates the base directory if needed."""
        new_dir = tmp_path / "new_saves"
        StatePersistence.set_active_bloodline("gamma", new_dir)
        assert (new_dir / "_active.txt").exists()


class TestRenameBloodline:
    """Tests for StatePersistence.rename_bloodline."""

    def test_rename_happy_path(self, base_dir: Path):
        """Renaming moves the directory."""
        _write_creature(base_dir / "alpha")
        StatePersistence.rename_bloodline("alpha", "beta", base_dir)
        assert not (base_dir / "alpha").exists()
        assert (base_dir / "beta" / "creature.json").exists()

    def test_rename_updates_active(self, base_dir: Path):
        """Renaming the active bloodline updates _active.txt."""
        _write_creature(base_dir / "alpha")
        StatePersistence.set_active_bloodline("alpha", base_dir)
        StatePersistence.rename_bloodline("alpha", "beta", base_dir)
        assert StatePersistence.get_active_bloodline(base_dir) == "beta"

    def test_rename_non_active_preserves_active(self, base_dir: Path):
        """Renaming a non-active bloodline doesn't change _active.txt."""
        _write_creature(base_dir / "alpha")
        _write_creature(base_dir / "gamma")
        StatePersistence.set_active_bloodline("gamma", base_dir)
        StatePersistence.rename_bloodline("alpha", "beta", base_dir)
        assert StatePersistence.get_active_bloodline(base_dir) == "gamma"

    def test_rename_empty_name_raises(self, base_dir: Path):
        """Empty new name raises ValueError."""
        _write_creature(base_dir / "alpha")
        with pytest.raises(ValueError, match="empty"):
            StatePersistence.rename_bloodline("alpha", "  ", base_dir)

    def test_rename_path_separator_raises(self, base_dir: Path):
        """Name with path separators raises ValueError."""
        _write_creature(base_dir / "alpha")
        with pytest.raises(ValueError, match="path separator"):
            StatePersistence.rename_bloodline("alpha", "foo/bar", base_dir)
        with pytest.raises(ValueError, match="path separator"):
            StatePersistence.rename_bloodline("alpha", "foo\\bar", base_dir)

    def test_rename_underscore_prefix_raises(self, base_dir: Path):
        """Name starting with underscore raises ValueError."""
        _write_creature(base_dir / "alpha")
        with pytest.raises(ValueError, match="underscore"):
            StatePersistence.rename_bloodline("alpha", "_hidden", base_dir)

    def test_rename_collision_raises(self, base_dir: Path):
        """Renaming to an existing name raises ValueError."""
        _write_creature(base_dir / "alpha")
        _write_creature(base_dir / "beta")
        with pytest.raises(ValueError, match="already exists"):
            StatePersistence.rename_bloodline("alpha", "beta", base_dir)

    def test_rename_source_missing_raises(self, base_dir: Path):
        """Renaming a nonexistent bloodline raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            StatePersistence.rename_bloodline("nope", "beta", base_dir)


class TestBloodlineInfoDataclass:
    """Tests for the BloodlineInfo dataclass."""

    def test_dataclass_fields(self):
        """BloodlineInfo has all required fields."""
        from datetime import datetime

        info = BloodlineInfo(
            name="test",
            save_dir=Path("data/saves/test"),
            generation_count=3,
            stage="gillman",
            last_modified=datetime.now(),
        )
        assert info.name == "test"
        assert info.generation_count == 3
        assert info.stage == "gillman"


class TestMultiDirectorySaves:
    """Tests for save/load in bloodline subdirectories."""

    def test_save_load_in_subdirectory(self, base_dir: Path):
        """Can save and load creature state in a bloodline subdirectory."""
        bl_dir = base_dir / "my_bloodline"
        persistence = StatePersistence(save_dir=bl_dir)

        state = CreatureState()
        state.trust_level = 0.75
        persistence.save(state)

        loaded = persistence.load()
        assert loaded.trust_level == pytest.approx(0.75)

    def test_independent_bloodline_saves(self, base_dir: Path):
        """Different bloodlines have independent saves."""
        p1 = StatePersistence(save_dir=base_dir / "alpha")
        p2 = StatePersistence(save_dir=base_dir / "beta")

        s1 = CreatureState()
        s1.trust_level = 0.3
        p1.save(s1)

        s2 = CreatureState()
        s2.trust_level = 0.9
        p2.save(s2)

        assert p1.load().trust_level == pytest.approx(0.3)
        assert p2.load().trust_level == pytest.approx(0.9)
