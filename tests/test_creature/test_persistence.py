"""Tests for StatePersistence save/load."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from seaman_brain.creature.persistence import StatePersistence
from seaman_brain.creature.state import CreatureState
from seaman_brain.types import CreatureStage

# --- Happy path ---

class TestSaveLoad:
    def test_save_creates_file(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        state = CreatureState()
        path = persistence.save(state)
        assert path.exists()
        assert path.name == "creature.json"

    def test_roundtrip(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        now = datetime.now(UTC)
        original = CreatureState(
            stage=CreatureStage.GILLMAN,
            age=7200.0,
            interaction_count=35,
            mood="content",
            trust_level=0.5,
            hunger=0.2,
            health=0.9,
            comfort=0.8,
            last_fed=now,
            last_interaction=now,
            birth_time=now - timedelta(hours=2),
        )
        persistence.save(original)
        loaded = persistence.load()
        assert loaded.stage == original.stage
        assert loaded.age == original.age
        assert loaded.interaction_count == original.interaction_count
        assert loaded.mood == original.mood
        assert loaded.trust_level == original.trust_level
        assert loaded.hunger == original.hunger
        assert loaded.health == original.health
        assert loaded.comfort == original.comfort
        assert loaded.birth_time == original.birth_time

    def test_save_creates_directory(self, tmp_path: str) -> None:
        nested = tmp_path / "deep" / "nested" / "saves"
        persistence = StatePersistence(save_dir=nested)
        persistence.save(CreatureState())
        assert (nested / "creature.json").exists()

    def test_custom_filename(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        state = CreatureState(mood="special")
        persistence.save(state, filename="slot1.json")
        loaded = persistence.load(filename="slot1.json")
        assert loaded.mood == "special"

    def test_save_valid_json(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        persistence.save(CreatureState())
        raw = (tmp_path / "creature.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        assert "stage" in data
        assert "mood" in data
        assert isinstance(data["age"], float)


# --- Backup creation ---

class TestBackup:
    def test_backup_created_on_overwrite(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        # First save
        first_state = CreatureState(mood="first")
        persistence.save(first_state)
        # Second save should create backup
        second_state = CreatureState(mood="second")
        persistence.save(second_state)

        backup_path = tmp_path / "creature.json.bak"
        assert backup_path.exists()
        backup_data = json.loads(backup_path.read_text(encoding="utf-8"))
        assert backup_data["mood"] == "first"

    def test_backup_updated_each_save(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        persistence.save(CreatureState(mood="v1"))
        persistence.save(CreatureState(mood="v2"))
        persistence.save(CreatureState(mood="v3"))

        # Backup should contain v2 (the previous save)
        backup_data = json.loads(
            (tmp_path / "creature.json.bak").read_text(encoding="utf-8")
        )
        assert backup_data["mood"] == "v2"
        # Main file should contain v3
        main_data = json.loads(
            (tmp_path / "creature.json").read_text(encoding="utf-8")
        )
        assert main_data["mood"] == "v3"

    def test_no_backup_on_first_save(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        persistence.save(CreatureState())
        assert not (tmp_path / "creature.json.bak").exists()


# --- Missing file handling ---

class TestMissingFile:
    def test_load_missing_returns_default(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        state = persistence.load()
        assert state.stage == CreatureStage.MUSHROOMER
        assert state.mood == "neutral"
        assert state.trust_level == 0.0

    def test_load_missing_custom_filename(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        state = persistence.load(filename="nonexistent.json")
        assert state.stage == CreatureStage.MUSHROOMER

    def test_exists_returns_false_for_missing(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        assert not persistence.exists()

    def test_exists_returns_true_after_save(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        persistence.save(CreatureState())
        assert persistence.exists()


# --- Error handling ---

class TestErrorHandling:
    def test_invalid_json_raises(self, tmp_path: str) -> None:
        save_file = tmp_path / "creature.json"
        save_file.write_text("not valid json {{{", encoding="utf-8")
        persistence = StatePersistence(save_dir=tmp_path)
        with pytest.raises(json.JSONDecodeError):
            persistence.load()

    def test_invalid_stage_in_file_raises(self, tmp_path: str) -> None:
        save_file = tmp_path / "creature.json"
        save_file.write_text(
            json.dumps({"stage": "INVALID_STAGE"}),
            encoding="utf-8",
        )
        persistence = StatePersistence(save_dir=tmp_path)
        with pytest.raises(ValueError):
            persistence.load()

    def test_empty_json_object_returns_default(self, tmp_path: str) -> None:
        save_file = tmp_path / "creature.json"
        save_file.write_text("{}", encoding="utf-8")
        persistence = StatePersistence(save_dir=tmp_path)
        state = persistence.load()
        assert state.stage == CreatureStage.MUSHROOMER


# --- Delete ---

class TestDelete:
    def test_delete_existing_file(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        persistence.save(CreatureState())
        assert persistence.exists()
        result = persistence.delete()
        assert result is True
        assert not persistence.exists()

    def test_delete_missing_file(self, tmp_path: str) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        result = persistence.delete()
        assert result is False


# --- All stages roundtrip ---

class TestAllStagesRoundtrip:
    @pytest.mark.parametrize("stage", list(CreatureStage))
    def test_stage_roundtrip(self, tmp_path: str, stage: CreatureStage) -> None:
        persistence = StatePersistence(save_dir=tmp_path)
        original = CreatureState(stage=stage)
        persistence.save(original, filename=f"{stage.value}.json")
        loaded = persistence.load(filename=f"{stage.value}.json")
        assert loaded.stage == stage
