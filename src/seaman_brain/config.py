"""TOML configuration loader with Pydantic models.

Loads config/default.toml and provides typed access to all subsystem settings.
User-changed settings are saved to data/user_settings.toml and merged on load.
Stage-specific TOML files can override personality traits.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_USER_SETTINGS_PATH = Path("data/user_settings.toml")


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: str = "ollama"
    model: str = "qwen3-coder:30b"
    temperature: float = 0.8
    max_tokens: int = 512
    base_url: str = "http://localhost:11434"


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = "ollama"
    model: str = "all-minilm:l6-v2"


class MemoryConfig(BaseModel):
    """Memory subsystem configuration."""

    buffer_size: int = 20
    vector_dims: int = 384
    top_k: int = 5
    extraction_interval: int = 5
    db_path: str = "data/lancedb"
    similarity_weight: float = 0.7
    recency_weight: float = 0.3
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)


class PresetConfig(BaseModel):
    """A personality preset definition."""

    name: str
    description: str
    traits: dict[str, float] = Field(default_factory=dict)


class PersonalityConfig(BaseModel):
    """Personality subsystem configuration."""

    base_traits: dict[str, float] = Field(default_factory=lambda: {
        "cynicism": 0.7,
        "wit": 0.8,
        "patience": 0.3,
        "curiosity": 0.6,
        "warmth": 0.2,
        "verbosity": 0.5,
        "formality": 0.2,
        "aggression": 0.4,
    })
    stages_path: str = "config/stages"
    presets_path: str = "config/presets.toml"


class EvolutionThreshold(BaseModel):
    """Threshold requirements for a creature stage transition."""

    interactions: int
    trust: float


class CreatureConfig(BaseModel):
    """Creature subsystem configuration."""

    save_path: str = "data/saves"
    initial_stage: str = "mushroomer"
    auto_save: bool = True
    evolution_thresholds: dict[str, EvolutionThreshold] = Field(default_factory=dict)


class AudioConfig(BaseModel):
    """Audio subsystem configuration."""

    tts_enabled: bool = True
    stt_enabled: bool = False
    sfx_enabled: bool = True
    tts_provider: str = "pyttsx3"
    stt_provider: str = "speech_recognition"
    tts_voice: str = ""
    tts_rate: int = 150
    tts_volume: float = 0.8
    sfx_volume: float = 0.5
    ambient_volume: float = 0.3
    audio_output_device: str = ""  # empty = system default
    audio_input_device: str = ""  # empty = system default


class EnvironmentConfig(BaseModel):
    """Tank environment configuration."""

    initial_temperature: float = 24.0
    optimal_temp_min: float = 20.0
    optimal_temp_max: float = 28.0
    lethal_temp_min: float = 10.0
    lethal_temp_max: float = 38.0
    cleanliness_decay_rate: float = 0.01
    oxygen_decay_rate: float = 0.005
    initial_environment: str = "aquarium"


class NeedsConfig(BaseModel):
    """Creature needs configuration."""

    hunger_rate: float = 0.02
    comfort_decay_rate: float = 0.01
    health_regen_rate: float = 0.005
    health_damage_rate: float = 0.01
    stimulation_decay_rate: float = 0.015
    critical_hunger_threshold: float = 0.8
    critical_health_threshold: float = 0.2
    starvation_time_hours: float = 1.0
    feeding_cooldown_seconds: int = 30


class VisionConfig(BaseModel):
    """Vision pipeline configuration."""

    enabled: bool = False
    vision_model: str = "qwen3-vl:8b"
    source: str = "webcam"  # "webcam" | "tank" | "off"
    capture_interval: float = 30.0
    max_observations: int = 3
    webcam_index: int = 0


class GUIConfig(BaseModel):
    """Pygame GUI configuration."""

    window_width: int = 1024
    window_height: int = 768
    fps: int = 30
    theme: str = "classic"
    show_debug_hud: bool = False
    chat_panel_height_ratio: float = 0.33


class APIConfig(BaseModel):
    """FastAPI WebSocket bridge configuration."""

    host: str = "127.0.0.1"
    port: int = 8420
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    broadcast_interval_ms: int = 500


class CLIConfig(BaseModel):
    """CLI interface configuration."""

    show_debug: bool = False
    stream_responses: bool = True
    prompt_style: str = "seaman"


class StageConfig(BaseModel):
    """Stage-specific personality overrides loaded from stage TOML files."""

    traits: dict[str, float] = Field(default_factory=dict)
    behavior: dict[str, Any] = Field(default_factory=dict)


class SeamanConfig(BaseModel):
    """Top-level configuration aggregating all subsystem configs."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig)
    creature: CreatureConfig = Field(default_factory=CreatureConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    environment: EnvironmentConfig = Field(default_factory=EnvironmentConfig)
    needs: NeedsConfig = Field(default_factory=NeedsConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    gui: GUIConfig = Field(default_factory=GUIConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    cli: CLIConfig = Field(default_factory=CLIConfig)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(
    config_dir: str | Path = "config",
    user_settings_path: str | Path | None = None,
) -> SeamanConfig:
    """Load configuration from TOML files.

    Reads config_dir/default.toml, then merges user settings on top
    if they exist, so user-changed settings persist across launches.
    Raises FileNotFoundError if default.toml does not exist.

    Args:
        config_dir: Path to the configuration directory.
        user_settings_path: Override for user settings file location.
            Defaults to data/user_settings.toml.

    Returns:
        Fully populated SeamanConfig instance.
    """
    config_path = Path(config_dir)
    default_toml = config_path / "default.toml"

    if not default_toml.exists():
        raise FileNotFoundError(f"Config file not found: {default_toml}")

    with open(default_toml, "rb") as f:
        raw = tomllib.load(f)

    # Merge user settings on top of defaults
    settings_path = Path(user_settings_path) if user_settings_path else _USER_SETTINGS_PATH
    user = _load_user_settings(settings_path)
    if user:
        raw = _deep_merge(raw, user)
        logger.info("Merged user settings from %s", settings_path)

    return SeamanConfig.model_validate(raw)


def _load_user_settings(path: Path | None = None) -> dict[str, Any]:
    """Load user settings overlay from a TOML file.

    Args:
        path: Settings file path. Defaults to _USER_SETTINGS_PATH.

    Returns:
        Dict of user overrides, or empty dict if file doesn't exist.
    """
    p = path or _USER_SETTINGS_PATH
    if not p.exists():
        return {}
    try:
        with open(p, "rb") as f:
            return tomllib.load(f)
    except Exception as exc:
        logger.warning("Failed to load user settings: %s", exc)
        return {}


def save_user_settings(config: SeamanConfig) -> None:
    """Save user-configurable settings to data/user_settings.toml.

    Only saves sections that the user can change via the settings panel:
    audio, vision, personality (base_traits), llm (model, temperature).

    Args:
        config: Current SeamanConfig to extract settings from.
    """
    _USER_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = ["# Auto-saved user settings — overrides defaults", ""]

    # LLM
    lines.append("[llm]")
    lines.append(f'model = "{config.llm.model}"')
    lines.append(f"temperature = {config.llm.temperature}")
    lines.append("")

    # Personality traits
    lines.append("[personality.base_traits]")
    for trait, val in config.personality.base_traits.items():
        lines.append(f"{trait} = {val}")
    lines.append("")

    # Audio
    lines.append("[audio]")
    lines.append(f"tts_enabled = {str(config.audio.tts_enabled).lower()}")
    lines.append(f"stt_enabled = {str(config.audio.stt_enabled).lower()}")
    lines.append(f"sfx_enabled = {str(config.audio.sfx_enabled).lower()}")
    lines.append(f'tts_voice = "{config.audio.tts_voice}"')
    lines.append(f"tts_rate = {config.audio.tts_rate}")
    lines.append(f"tts_volume = {config.audio.tts_volume}")
    lines.append(f"sfx_volume = {config.audio.sfx_volume}")
    lines.append(f"ambient_volume = {config.audio.ambient_volume}")
    lines.append(f'audio_output_device = "{config.audio.audio_output_device}"')
    lines.append(f'audio_input_device = "{config.audio.audio_input_device}"')
    lines.append("")

    # Vision
    lines.append("[vision]")
    lines.append(f"enabled = {str(config.vision.enabled).lower()}")
    lines.append(f'source = "{config.vision.source}"')
    lines.append(f"capture_interval = {config.vision.capture_interval}")
    lines.append(f"webcam_index = {config.vision.webcam_index}")
    lines.append("")

    try:
        _USER_SETTINGS_PATH.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("User settings saved to %s", _USER_SETTINGS_PATH)
    except Exception as exc:
        logger.warning("Failed to save user settings: %s", exc)


def load_stage_config(
    stage: str,
    config_dir: str | Path = "config",
) -> StageConfig:
    """Load stage-specific personality overrides.

    Reads config_dir/stages/{stage}.toml and returns trait/behavior overrides.
    Returns an empty StageConfig if the file doesn't exist.

    Args:
        stage: Stage name (e.g. "mushroomer", "gillman").
        config_dir: Path to the configuration directory.

    Returns:
        StageConfig with trait and behavior overrides.
    """
    stage_file = Path(config_dir) / "stages" / f"{stage}.toml"

    if not stage_file.exists():
        return StageConfig()

    with open(stage_file, "rb") as f:
        raw = tomllib.load(f)

    return StageConfig.model_validate(raw)


def load_config_with_stage(
    stage: str,
    config_dir: str | Path = "config",
) -> SeamanConfig:
    """Load config with stage-specific personality trait overrides merged in.

    Args:
        stage: Stage name (e.g. "mushroomer").
        config_dir: Path to the configuration directory.

    Returns:
        SeamanConfig with personality traits overridden by stage config.
    """
    config = load_config(config_dir)
    stage_config = load_stage_config(stage, config_dir)

    if stage_config.traits:
        merged_traits = _deep_merge(config.personality.base_traits, stage_config.traits)
        config = config.model_copy(
            update={"personality": config.personality.model_copy(
                update={"base_traits": merged_traits}
            )}
        )

    return config


def load_presets(path: str | Path = "config/presets.toml") -> dict[str, PresetConfig]:
    """Load personality presets from a TOML file.

    Each top-level key in the TOML becomes a preset name. Keys must have
    ``name``, ``description``, and ``traits`` sub-keys.

    Args:
        path: Path to the presets TOML file.

    Returns:
        Dict mapping preset key to PresetConfig.

    Raises:
        FileNotFoundError: If the presets file does not exist.
    """
    preset_path = Path(path)

    if not preset_path.exists():
        raise FileNotFoundError(f"Presets file not found: {preset_path}")

    with open(preset_path, "rb") as f:
        raw = tomllib.load(f)

    presets: dict[str, PresetConfig] = {}
    for key, data in raw.items():
        if isinstance(data, dict):
            presets[key] = PresetConfig.model_validate(data)

    return presets
