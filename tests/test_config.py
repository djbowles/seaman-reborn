"""Tests for TOML config loader with Pydantic models (US-003)."""

from __future__ import annotations

import pytest

from seaman_brain.config import (
    SeamanConfig,
    StageConfig,
    load_config,
    load_config_with_stage,
    load_stage_config,
)

# --- Fixtures ---

MINIMAL_TOML = """\
[llm]
provider = "ollama"
model = "test-model"
temperature = 0.5
max_tokens = 256
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
