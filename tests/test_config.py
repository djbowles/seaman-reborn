"""Tests for TOML config loader with Pydantic models (US-003)."""

from __future__ import annotations

import pytest

import seaman_brain.config as _config_mod
from seaman_brain.config import (
    PresetConfig,
    SeamanConfig,
    StageConfig,
    VisionConfig,
    _flush_save,
    load_config,
    load_config_with_stage,
    load_presets,
    load_stage_config,
    save_user_settings,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path, monkeypatch):
    """Prevent real data/user_settings.toml from leaking into config tests."""
    monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", tmp_path / "no_user.toml")

MINIMAL_TOML = """\
[llm]
provider = "ollama"
model = "test-model"
temperature = 0.5
max_tokens = 256
context_window = 4096
max_response_tokens = 2048
base_url = "http://localhost:11434"

[memory]
buffer_size = 10
vector_dims = 384
top_k = 3
extraction_interval = 5

[personality]
stages_path = "config/stages"

[personality.base_traits]
cynicism = 0.7
wit = 0.8

[gui]
window_width = 800
window_height = 600
fps = 60
theme = "dark"

[api]
host = "0.0.0.0"
port = 9000
"""

STAGE_TOML = """\
[traits]
cynicism = 0.3
warmth = 0.9

[behavior]
max_response_words = 50
speech_style = "friendly"
"""


@pytest.fixture
def config_with_toml(config_dir):
    """Create a config dir with a minimal default.toml."""
    (config_dir / "default.toml").write_text(MINIMAL_TOML)
    return config_dir


@pytest.fixture
def config_with_stage(config_with_toml):
    """Create config dir with default.toml and a stage override."""
    stages = config_with_toml / "stages"
    (stages / "mushroomer.toml").write_text(STAGE_TOML)
    return config_with_toml


# --- Happy path tests ---


class TestLoadConfigHappyPath:
    """Tests for successful config loading."""

    def test_load_default_config(self, config_with_toml):
        """Load a minimal default.toml and verify key fields."""
        cfg = load_config(config_with_toml)

        assert isinstance(cfg, SeamanConfig)
        assert cfg.llm.provider == "ollama"
        assert cfg.llm.model == "test-model"
        assert cfg.llm.temperature == 0.5
        assert cfg.llm.max_tokens == 256
        assert cfg.llm.context_window == 4096
        assert cfg.llm.max_response_tokens == 2048

    def test_memory_config_loaded(self, config_with_toml):
        """Memory section parsed correctly."""
        cfg = load_config(config_with_toml)

        assert cfg.memory.buffer_size == 10
        assert cfg.memory.vector_dims == 384
        assert cfg.memory.top_k == 3

    def test_gui_config_loaded(self, config_with_toml):
        """GUI section parsed correctly."""
        cfg = load_config(config_with_toml)

        assert cfg.gui.window_width == 800
        assert cfg.gui.window_height == 600
        assert cfg.gui.fps == 60
        assert cfg.gui.theme == "dark"

    def test_api_config_loaded(self, config_with_toml):
        """API section parsed correctly."""
        cfg = load_config(config_with_toml)

        assert cfg.api.host == "0.0.0.0"
        assert cfg.api.port == 9000

    def test_personality_traits_loaded(self, config_with_toml):
        """Personality base traits parsed from TOML."""
        cfg = load_config(config_with_toml)

        assert cfg.personality.base_traits["cynicism"] == 0.7
        assert cfg.personality.base_traits["wit"] == 0.8

    def test_defaults_fill_missing_sections(self, config_with_toml):
        """Sections not in TOML get Pydantic defaults."""
        cfg = load_config(config_with_toml)

        # Audio not in our minimal TOML, should get defaults
        assert cfg.audio.tts_provider == "pyttsx3"
        assert cfg.audio.tts_enabled is True

        # CLI not in our minimal TOML
        assert cfg.cli.show_debug is False
        assert cfg.cli.stream_responses is True


# --- Stage config tests ---


class TestStageConfig:
    """Tests for stage-specific config loading and merging."""

    def test_load_stage_config(self, config_with_stage):
        """Stage TOML provides trait overrides."""
        stage = load_stage_config("mushroomer", config_with_stage)

        assert isinstance(stage, StageConfig)
        assert stage.traits["cynicism"] == 0.3
        assert stage.traits["warmth"] == 0.9

    def test_stage_behavior_loaded(self, config_with_stage):
        """Stage TOML provides behavior section."""
        stage = load_stage_config("mushroomer", config_with_stage)

        assert stage.behavior["max_response_words"] == 50
        assert stage.behavior["speech_style"] == "friendly"

    def test_missing_stage_returns_empty(self, config_with_stage):
        """A non-existent stage file returns empty StageConfig."""
        stage = load_stage_config("nonexistent", config_with_stage)

        assert stage.traits == {}
        assert stage.behavior == {}

    def test_stage_override_merges_traits(self, config_with_stage):
        """load_config_with_stage merges stage traits over base traits."""
        cfg = load_config_with_stage("mushroomer", config_with_stage)

        # Overridden by stage
        assert cfg.personality.base_traits["cynicism"] == 0.3
        assert cfg.personality.base_traits["warmth"] == 0.9
        # Kept from base
        assert cfg.personality.base_traits["wit"] == 0.8

    def test_stage_override_preserves_other_config(self, config_with_stage):
        """Stage override only affects personality, other sections untouched."""
        cfg = load_config_with_stage("mushroomer", config_with_stage)

        assert cfg.llm.model == "test-model"
        assert cfg.gui.theme == "dark"


# --- Error handling tests ---


class TestConfigErrors:
    """Tests for error conditions."""

    def test_missing_default_toml_raises(self, config_dir):
        """FileNotFoundError when default.toml doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_config(config_dir)

    def test_missing_config_dir_raises(self, tmp_path):
        """FileNotFoundError when config dir doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent")

    def test_empty_toml_uses_all_defaults(self, config_dir):
        """An empty TOML file should produce a valid config with all defaults."""
        (config_dir / "default.toml").write_text("")
        cfg = load_config(config_dir)

        assert isinstance(cfg, SeamanConfig)
        assert cfg.llm.provider == "ollama"
        assert cfg.memory.buffer_size == 20
        assert cfg.gui.fps == 30


# --- Edge case tests ---


class TestConfigEdgeCases:
    """Edge cases and special scenarios."""

    def test_real_default_toml_loads(self):
        """The actual project config/default.toml loads successfully."""
        cfg = load_config("config")

        assert cfg.llm.provider == "ollama"
        assert cfg.llm.model == "qwen3-coder:30b"
        assert cfg.memory.embeddings.provider == "ollama"
        assert cfg.creature.auto_save is True
        assert len(cfg.creature.evolution_thresholds) == 4

    def test_real_stage_configs_load(self):
        """All real stage TOML files load successfully."""
        for stage in ["mushroomer", "gillman", "podfish", "tadman", "frogman"]:
            sc = load_stage_config(stage, "config")
            assert isinstance(sc, StageConfig)
            assert len(sc.traits) > 0
            assert "speech_style" in sc.behavior

    def test_seaman_config_defaults_without_toml(self):
        """SeamanConfig can be constructed with pure defaults (no TOML)."""
        cfg = SeamanConfig()

        assert cfg.llm.provider == "ollama"
        assert cfg.memory.buffer_size == 20
        assert cfg.personality.stages_path == "config/stages"
        assert cfg.environment.initial_temperature == 24.0
        assert cfg.needs.hunger_rate == 0.02

    def test_evolution_thresholds_parsed(self):
        """Evolution thresholds from real config are typed correctly."""
        cfg = load_config("config")

        assert cfg.creature.evolution_thresholds["gillman"].interactions == 20
        assert cfg.creature.evolution_thresholds["gillman"].trust == 0.3
        assert cfg.creature.evolution_thresholds["frogman"].interactions == 200
        assert cfg.creature.evolution_thresholds["frogman"].trust == 0.8

    def test_path_accepts_string_and_path(self, config_with_toml):
        """load_config accepts both str and Path."""
        cfg_str = load_config(str(config_with_toml))
        cfg_path = load_config(config_with_toml)

        assert cfg_str.llm.model == cfg_path.llm.model


# --- Preset config tests ---

PRESET_TOML = """\
[traditional]
name = "Traditional (1999)"
description = "Classic Seaman"

[traditional.traits]
cynicism = 0.95
wit = 0.6

[modern]
name = "Modern"
description = "Sardonic philosopher"

[modern.traits]
cynicism = 0.7
wit = 0.9
"""


class TestPresetConfig:
    """Tests for personality preset loading."""

    def test_load_presets_from_file(self, tmp_path):
        """Presets TOML loads and parses into PresetConfig dict."""
        preset_file = tmp_path / "presets.toml"
        preset_file.write_text(PRESET_TOML)
        presets = load_presets(preset_file)

        assert "traditional" in presets
        assert "modern" in presets
        assert isinstance(presets["traditional"], PresetConfig)

    def test_preset_values_correct(self, tmp_path):
        """Preset name, description, and traits are correct."""
        preset_file = tmp_path / "presets.toml"
        preset_file.write_text(PRESET_TOML)
        presets = load_presets(preset_file)

        trad = presets["traditional"]
        assert trad.name == "Traditional (1999)"
        assert trad.description == "Classic Seaman"
        assert trad.traits["cynicism"] == 0.95
        assert trad.traits["wit"] == 0.6

    def test_preset_missing_file_raises(self, tmp_path):
        """FileNotFoundError when presets file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Presets file not found"):
            load_presets(tmp_path / "nonexistent.toml")

    def test_real_presets_toml_loads(self):
        """The actual config/presets.toml loads successfully."""
        presets = load_presets("config/presets.toml")

        assert "traditional" in presets
        assert "modern" in presets
        assert "custom" in presets
        assert presets["traditional"].traits["cynicism"] == 0.95
        assert presets["modern"].traits["wit"] == 0.9
        assert presets["custom"].traits["cynicism"] == 0.5

    def test_personality_config_has_presets_path(self):
        """PersonalityConfig includes presets_path field."""
        cfg = SeamanConfig()
        assert cfg.personality.presets_path == "config/presets.toml"

    def test_preset_config_model(self):
        """PresetConfig can be constructed directly."""
        p = PresetConfig(name="Test", description="Desc", traits={"wit": 0.5})
        assert p.name == "Test"
        assert p.traits["wit"] == 0.5

    def test_preset_config_default_traits(self):
        """PresetConfig defaults to empty traits dict."""
        p = PresetConfig(name="Bare", description="Bare preset")
        assert p.traits == {}


# --- Vision config tests ---


class TestVisionConfig:
    """Tests for VisionConfig and integration with SeamanConfig."""

    def test_vision_config_defaults(self):
        """VisionConfig has sensible defaults."""
        vc = VisionConfig()
        assert vc.enabled is False
        assert vc.vision_model == "qwen3-vl:8b"
        assert vc.source == "webcam"
        assert vc.capture_interval == 30.0
        assert vc.max_observations == 3
        assert vc.webcam_index == 0

    def test_vision_config_on_seaman_config(self):
        """SeamanConfig includes vision field with defaults."""
        cfg = SeamanConfig()
        assert hasattr(cfg, "vision")
        assert isinstance(cfg.vision, VisionConfig)
        assert cfg.vision.enabled is False

    def test_vision_config_from_toml(self):
        """Real default.toml loads vision section correctly."""
        cfg = load_config("config")
        assert cfg.vision.vision_model == "qwen3-vl:8b"
        assert cfg.vision.source == "webcam"
        assert cfg.vision.capture_interval == 30.0

    def test_vision_config_custom_values(self):
        """VisionConfig accepts custom values."""
        vc = VisionConfig(
            enabled=True,
            vision_model="custom-vl:7b",
            source="tank",
            capture_interval=10.0,
            max_observations=5,
            webcam_index=1,
        )
        assert vc.enabled is True
        assert vc.vision_model == "custom-vl:7b"
        assert vc.source == "tank"
        assert vc.capture_interval == 10.0
        assert vc.max_observations == 5
        assert vc.webcam_index == 1

    def test_vision_config_from_minimal_toml(self, config_with_toml):
        """Vision defaults fill in when not specified in TOML."""
        cfg = load_config(config_with_toml)
        assert cfg.vision.enabled is False
        assert cfg.vision.vision_model == "qwen3-vl:8b"


class TestUserSettingsPersistence:
    """Tests for save/load of user settings across launches."""

    def test_save_creates_file(self, tmp_path, monkeypatch):
        """save_user_settings creates the TOML file."""
        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)

        cfg = SeamanConfig()
        save_user_settings(cfg)
        _flush_save()
        assert settings_file.exists()

    def test_round_trip_audio(self, tmp_path, monkeypatch):
        """Audio settings survive save/load cycle."""
        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)

        cfg = SeamanConfig()
        cfg.audio.tts_volume = 0.42
        cfg.audio.sfx_enabled = False
        save_user_settings(cfg)
        _flush_save()

        loaded = load_config("config", user_settings_path=settings_file)
        assert loaded.audio.tts_volume == pytest.approx(0.42)
        assert loaded.audio.sfx_enabled is False

    def test_round_trip_vision(self, tmp_path, monkeypatch):
        """Vision settings survive save/load cycle."""
        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)

        cfg = SeamanConfig()
        cfg.vision.enabled = True
        cfg.vision.source = "tank"
        cfg.vision.webcam_index = 2
        save_user_settings(cfg)
        _flush_save()

        loaded = load_config("config", user_settings_path=settings_file)
        assert loaded.vision.enabled is True
        assert loaded.vision.source == "tank"
        assert loaded.vision.webcam_index == 2

    def test_round_trip_llm(self, tmp_path, monkeypatch):
        """LLM settings survive save/load cycle."""
        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)

        cfg = SeamanConfig()
        cfg.llm.model = "custom-model:7b"
        cfg.llm.temperature = 0.3
        save_user_settings(cfg)
        _flush_save()

        loaded = load_config("config", user_settings_path=settings_file)
        assert loaded.llm.model == "custom-model:7b"
        assert loaded.llm.temperature == pytest.approx(0.3)

    def test_missing_user_settings_uses_defaults(self):
        """When no user settings file exists, defaults are used."""
        cfg = load_config("config", user_settings_path="/nonexistent/path.toml")
        assert cfg.llm.model == "qwen3-coder:30b"

    def test_user_settings_override_defaults(self, tmp_path, monkeypatch):
        """User settings override default.toml values."""
        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings_file.write_text('[vision]\nenabled = true\n')

        loaded = load_config("config", user_settings_path=settings_file)
        assert loaded.vision.enabled is True
        # Other vision defaults still intact
        assert loaded.vision.vision_model == "qwen3-vl:8b"


# ── Debounced Save Tests ──────────────────────────────────────────────


class TestDebouncedSave:
    """Tests for debounced save_user_settings (Fix #21)."""

    def test_rapid_saves_coalesced(self, tmp_path, monkeypatch):
        """Multiple rapid calls produce only one file write."""
        import time

        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        # Reset debounce state
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)

        cfg1 = SeamanConfig()
        cfg1.llm.model = "model-first"
        cfg2 = SeamanConfig()
        cfg2.llm.model = "model-second"
        cfg3 = SeamanConfig()
        cfg3.llm.model = "model-third"

        # Fire three rapid saves
        save_user_settings(cfg1)
        save_user_settings(cfg2)
        save_user_settings(cfg3)

        # Wait for debounce timer to flush
        time.sleep(1.0)

        assert settings_file.exists()
        content = settings_file.read_text()
        # Only the last value should be written
        assert "model-third" in content
        assert "model-first" not in content

    def test_concurrent_save_no_corruption(self, tmp_path, monkeypatch):
        """Concurrent saves from different threads don't corrupt the file."""
        import threading
        import time

        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)

        errors: list[Exception] = []

        def _save_from_thread(model_name: str):
            try:
                cfg = SeamanConfig()
                cfg.llm.model = model_name
                save_user_settings(cfg)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=_save_from_thread, args=(f"thread-model-{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Wait for debounce timer to flush
        time.sleep(1.0)

        assert not errors
        assert settings_file.exists()
        content = settings_file.read_text()
        # File should contain valid TOML (has header and model key)
        assert "[llm]" in content
        assert "model = " in content

    def test_flush_save_writes_immediately(self, tmp_path, monkeypatch):
        """_flush_save writes the pending config to disk."""
        settings_file = tmp_path / "data" / "user_settings.toml"
        monkeypatch.setattr(_config_mod, "_USER_SETTINGS_PATH", settings_file)

        cfg = SeamanConfig()
        cfg.llm.model = "flush-test-model"
        monkeypatch.setattr(_config_mod, "_pending_save_config", cfg)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)

        _config_mod._flush_save()

        assert settings_file.exists()
        content = settings_file.read_text()
        assert "flush-test-model" in content

    def test_flush_save_none_config_noop(self, monkeypatch):
        """_flush_save with no pending config does nothing."""
        monkeypatch.setattr(_config_mod, "_pending_save_config", None)
        monkeypatch.setattr(_config_mod, "_pending_save_timer", None)
        _config_mod._flush_save()  # Should not raise
