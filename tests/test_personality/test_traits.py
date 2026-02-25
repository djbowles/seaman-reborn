"""Tests for personality trait system (US-014)."""

from __future__ import annotations

from seaman_brain.config import PersonalityConfig
from seaman_brain.personality.traits import (
    STAGE_DEFAULTS,
    TRAIT_NAMES,
    TraitProfile,
    get_default_profile,
    load_trait_profile,
    profile_from_config,
)
from seaman_brain.types import CreatureStage

# --- Happy path tests ---


class TestTraitProfileCreation:
    """Tests for creating TraitProfile instances."""

    def test_default_values(self) -> None:
        """Default profile has all traits at 0.5."""
        profile = TraitProfile()
        for name in TRAIT_NAMES:
            assert getattr(profile, name) == 0.5

    def test_custom_values(self) -> None:
        """TraitProfile accepts custom float values."""
        profile = TraitProfile(
            cynicism=0.9, wit=0.1, patience=0.3, curiosity=0.7,
            warmth=0.4, verbosity=0.6, formality=0.8, aggression=0.2,
        )
        assert profile.cynicism == 0.9
        assert profile.wit == 0.1
        assert profile.patience == 0.3
        assert profile.curiosity == 0.7
        assert profile.warmth == 0.4
        assert profile.verbosity == 0.6
        assert profile.formality == 0.8
        assert profile.aggression == 0.2

    def test_to_dict(self) -> None:
        """to_dict returns all 8 traits as a dictionary."""
        profile = TraitProfile(cynicism=0.7, wit=0.3)
        d = profile.to_dict()
        assert len(d) == 8
        assert d["cynicism"] == 0.7
        assert d["wit"] == 0.3
        assert d["patience"] == 0.5  # default

    def test_from_dict(self) -> None:
        """from_dict creates a profile from a dict."""
        traits = {"cynicism": 0.8, "wit": 0.2, "warmth": 0.9}
        profile = TraitProfile.from_dict(traits)
        assert profile.cynicism == 0.8
        assert profile.wit == 0.2
        assert profile.warmth == 0.9
        assert profile.patience == 0.5  # default for unspecified

    def test_roundtrip_dict(self) -> None:
        """to_dict -> from_dict roundtrip preserves values."""
        original = TraitProfile(
            cynicism=0.1, wit=0.2, patience=0.3, curiosity=0.4,
            warmth=0.5, verbosity=0.6, formality=0.7, aggression=0.8,
        )
        rebuilt = TraitProfile.from_dict(original.to_dict())
        assert original.to_dict() == rebuilt.to_dict()

    def test_eight_trait_names(self) -> None:
        """TRAIT_NAMES contains exactly 8 dimensions."""
        assert len(TRAIT_NAMES) == 8
        expected = {
            "cynicism", "wit", "patience", "curiosity",
            "warmth", "verbosity", "formality", "aggression",
        }
        assert set(TRAIT_NAMES) == expected


# --- Default profiles per stage ---


class TestStageDefaults:
    """Tests for hardcoded stage default profiles."""

    def test_all_stages_have_defaults(self) -> None:
        """Every CreatureStage has a default TraitProfile."""
        for stage in CreatureStage:
            assert stage in STAGE_DEFAULTS

    def test_mushroomer_is_primitive(self) -> None:
        """Mushroomer should be low verbosity, high aggression."""
        p = STAGE_DEFAULTS[CreatureStage.MUSHROOMER]
        assert p.verbosity == 0.2
        assert p.aggression == 0.6
        assert p.warmth == 0.1

    def test_podfish_is_peak_sardonic(self) -> None:
        """Podfish is 'classic Seaman' — high cynicism and wit."""
        p = STAGE_DEFAULTS[CreatureStage.PODFISH]
        assert p.cynicism == 0.8
        assert p.wit == 0.9

    def test_frogman_is_wise(self) -> None:
        """Frogman should be high patience, curiosity, and warmth."""
        p = STAGE_DEFAULTS[CreatureStage.FROGMAN]
        assert p.patience == 0.7
        assert p.curiosity == 0.9
        assert p.warmth == 0.5

    def test_get_default_profile_returns_copy(self) -> None:
        """get_default_profile returns a fresh copy, not the original."""
        p1 = get_default_profile(CreatureStage.PODFISH)
        p2 = get_default_profile(CreatureStage.PODFISH)
        assert p1 is not p2
        assert p1.to_dict() == p2.to_dict()

    def test_stages_evolve_progressively(self) -> None:
        """Later stages should generally have higher warmth and patience."""
        stages = list(CreatureStage)
        first = STAGE_DEFAULTS[stages[0]]
        last = STAGE_DEFAULTS[stages[-1]]
        assert last.warmth > first.warmth
        assert last.patience > first.patience


# --- Clamping ---


class TestClamping:
    """Tests for trait value clamping to [0.0, 1.0]."""

    def test_clamp_above_one(self) -> None:
        """Values above 1.0 are clamped to 1.0."""
        profile = TraitProfile(cynicism=1.5, wit=2.0)
        assert profile.cynicism == 1.0
        assert profile.wit == 1.0

    def test_clamp_below_zero(self) -> None:
        """Values below 0.0 are clamped to 0.0."""
        profile = TraitProfile(patience=-0.5, warmth=-1.0)
        assert profile.patience == 0.0
        assert profile.warmth == 0.0

    def test_clamp_boundary_values(self) -> None:
        """Exact boundary values 0.0 and 1.0 are preserved."""
        profile = TraitProfile(cynicism=0.0, wit=1.0)
        assert profile.cynicism == 0.0
        assert profile.wit == 1.0

    def test_clamp_from_dict(self) -> None:
        """from_dict also clamps values."""
        profile = TraitProfile.from_dict({"cynicism": 5.0, "wit": -3.0})
        assert profile.cynicism == 1.0
        assert profile.wit == 0.0


# --- TOML loading ---


class TestLoadTraitProfile:
    """Tests for loading traits from TOML config files."""

    def test_load_from_toml(self, tmp_path: object) -> None:
        """load_trait_profile reads stage TOML and creates profile."""
        from pathlib import Path

        config_dir = Path(str(tmp_path)) / "config"
        stages_dir = config_dir / "stages"
        stages_dir.mkdir(parents=True)

        toml_content = b'[traits]\ncynicism = 0.99\nwit = 0.11\n'
        (stages_dir / "mushroomer.toml").write_bytes(toml_content)

        profile = load_trait_profile(CreatureStage.MUSHROOMER, str(config_dir))
        assert profile.cynicism == 0.99
        assert profile.wit == 0.11
        # Unspecified traits fall back to STAGE_DEFAULTS for mushroomer
        assert profile.patience == STAGE_DEFAULTS[CreatureStage.MUSHROOMER].patience

    def test_load_missing_toml_uses_defaults(self, tmp_path: object) -> None:
        """Missing TOML file falls back to hardcoded defaults."""
        from pathlib import Path

        config_dir = Path(str(tmp_path)) / "config"
        stages_dir = config_dir / "stages"
        stages_dir.mkdir(parents=True)
        # No TOML file created

        profile = load_trait_profile(CreatureStage.GILLMAN, str(config_dir))
        expected = STAGE_DEFAULTS[CreatureStage.GILLMAN]
        assert profile.to_dict() == expected.to_dict()

    def test_load_empty_traits_uses_defaults(self, tmp_path: object) -> None:
        """TOML file with empty [traits] section uses defaults."""
        from pathlib import Path

        config_dir = Path(str(tmp_path)) / "config"
        stages_dir = config_dir / "stages"
        stages_dir.mkdir(parents=True)

        toml_content = b'[behavior]\nmax_response_words = 50\n'
        (stages_dir / "podfish.toml").write_bytes(toml_content)

        profile = load_trait_profile(CreatureStage.PODFISH, str(config_dir))
        expected = STAGE_DEFAULTS[CreatureStage.PODFISH]
        assert profile.to_dict() == expected.to_dict()

    def test_load_clamps_toml_values(self, tmp_path: object) -> None:
        """Out-of-range values in TOML are clamped."""
        from pathlib import Path

        config_dir = Path(str(tmp_path)) / "config"
        stages_dir = config_dir / "stages"
        stages_dir.mkdir(parents=True)

        toml_content = b'[traits]\ncynicism = 9.9\nwit = -5.0\n'
        (stages_dir / "frogman.toml").write_bytes(toml_content)

        profile = load_trait_profile(CreatureStage.FROGMAN, str(config_dir))
        assert profile.cynicism == 1.0
        assert profile.wit == 0.0


# --- Config integration ---


class TestProfileFromConfig:
    """Tests for creating profiles from PersonalityConfig."""

    def test_from_default_config(self) -> None:
        """profile_from_config works with default PersonalityConfig."""
        config = PersonalityConfig()
        profile = profile_from_config(config)
        assert profile.cynicism == config.base_traits["cynicism"]
        assert profile.wit == config.base_traits["wit"]

    def test_from_custom_config(self) -> None:
        """profile_from_config uses custom base_traits."""
        config = PersonalityConfig(base_traits={
            "cynicism": 0.1, "wit": 0.2, "patience": 0.3, "curiosity": 0.4,
            "warmth": 0.5, "verbosity": 0.6, "formality": 0.7, "aggression": 0.8,
        })
        profile = profile_from_config(config)
        assert profile.cynicism == 0.1
        assert profile.aggression == 0.8


# --- Edge cases ---


class TestEdgeCases:
    """Edge case tests for the trait system."""

    def test_from_dict_ignores_unknown_keys(self) -> None:
        """Unknown trait names in dict are silently ignored."""
        traits = {"cynicism": 0.8, "unknown_trait": 0.5, "foo": 0.1}
        profile = TraitProfile.from_dict(traits)
        assert profile.cynicism == 0.8
        assert not hasattr(profile, "unknown_trait")

    def test_from_dict_empty(self) -> None:
        """Empty dict creates a profile with all defaults."""
        profile = TraitProfile.from_dict({})
        assert profile.to_dict() == TraitProfile().to_dict()

    def test_integer_values_accepted(self) -> None:
        """Integer values (0, 1) are accepted and converted."""
        profile = TraitProfile(cynicism=0, wit=1)
        assert profile.cynicism == 0.0
        assert profile.wit == 1.0

    def test_trait_names_match_dataclass_fields(self) -> None:
        """TRAIT_NAMES matches the actual dataclass fields."""
        profile = TraitProfile()
        for name in TRAIT_NAMES:
            assert hasattr(profile, name)
