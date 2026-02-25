"""Tests for creature self-model and body awareness (US-048)."""

from __future__ import annotations

from seaman_brain.creature.genome import CreatureGenome
from seaman_brain.creature.self_model import (
    SelfModel,
    build_self_description,
    get_prompt_injection,
)
from seaman_brain.types import CreatureStage

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------


def _genome_with(**overrides: float) -> CreatureGenome:
    """Create a genome with specific trait overrides (defaults to 0.5)."""
    traits = {
        "body_size": 0.5,
        "head_ratio": 0.5,
        "eye_size": 0.5,
        "fin_length": 0.5,
        "limb_proportion": 0.5,
        "hue": 0.5,
        "saturation": 0.5,
        "pattern_complexity": 0.5,
        "metabolism_rate": 0.5,
        "voice_pitch": 0.5,
        "face_expressiveness": 0.5,
        "aggression_baseline": 0.5,
    }
    traits.update(overrides)
    return CreatureGenome(traits=traits)


# -----------------------------------------------------------------------
# Description generation per stage
# -----------------------------------------------------------------------


class TestDescriptionPerStage:
    """Verify that description detail varies by evolutionary stage."""

    def test_mushroomer_minimal_description(self) -> None:
        """Mushroomer should have minimal self-awareness — short description."""
        genome = _genome_with(body_size=0.8)
        model = build_self_description(genome, CreatureStage.MUSHROOMER)
        assert "dimly aware" in model.description
        # Should NOT contain chromatic or behavioral details
        assert "skin" not in model.description
        assert "voice" not in model.description

    def test_gillman_basic_description(self) -> None:
        """Gillman should describe morphological traits but not chromatic."""
        genome = _genome_with(body_size=0.8, eye_size=0.9)
        model = build_self_description(genome, CreatureStage.GILLMAN)
        assert "beginning to notice" in model.description
        assert "large and imposing" in model.description
        assert "bulging eyes" in model.description
        # Should NOT contain chromatic traits
        assert "skin is" not in model.description

    def test_podfish_moderate_description(self) -> None:
        """Podfish should include morphological and chromatic traits."""
        genome = _genome_with(saturation=0.9, pattern_complexity=0.8)
        model = build_self_description(genome, CreatureStage.PODFISH)
        assert "keenly aware" in model.description
        assert "vividly colored" in model.description
        assert "intricate" in model.description
        # Should NOT contain behavioral traits
        assert "voice" not in model.description

    def test_tadman_detailed_description(self) -> None:
        """Tadman should include morphological, chromatic, and behavioral traits."""
        genome = _genome_with(voice_pitch=0.1, face_expressiveness=0.9)
        model = build_self_description(genome, CreatureStage.TADMAN)
        assert "sophisticated understanding" in model.description
        assert "deep and rumbling" in model.description
        assert "wildly expressive" in model.description

    def test_frogman_rich_description(self) -> None:
        """Frogman should have the most detailed description with synthesis."""
        genome = _genome_with(body_size=0.9, aggression_baseline=0.1)
        model = build_self_description(genome, CreatureStage.FROGMAN)
        assert "complete awareness" in model.description
        assert "story" in model.description
        assert "large and imposing" in model.description

    def test_all_stages_produce_nonempty_description(self) -> None:
        """Every stage should produce a non-empty description."""
        genome = _genome_with()
        for stage in CreatureStage:
            model = build_self_description(genome, stage)
            assert model.description, f"Empty description for {stage}"


# -----------------------------------------------------------------------
# Change detection between updates
# -----------------------------------------------------------------------


class TestChangeDetection:
    """Verify that changes are detected between self-model updates."""

    def test_no_changes_on_first_build(self) -> None:
        """First build should have no changes (no previous state)."""
        genome = _genome_with(body_size=0.8)
        model = build_self_description(genome, CreatureStage.PODFISH)
        assert model.new_changes == []

    def test_detects_trait_increase(self) -> None:
        """Should detect when a trait level increases."""
        genome_before = _genome_with(body_size=0.2)  # low
        genome_after = _genome_with(body_size=0.8)   # high

        model1 = build_self_description(genome_before, CreatureStage.PODFISH)
        model2 = build_self_description(
            genome_after, CreatureStage.PODFISH, self_model=model1,
        )

        assert len(model2.new_changes) > 0
        assert any("grown larger" in c for c in model2.new_changes)

    def test_detects_trait_decrease(self) -> None:
        """Should detect when a trait level decreases."""
        genome_before = _genome_with(eye_size=0.9)   # high
        genome_after = _genome_with(eye_size=0.1)     # low

        model1 = build_self_description(genome_before, CreatureStage.PODFISH)
        model2 = build_self_description(
            genome_after, CreatureStage.PODFISH, self_model=model1,
        )

        assert len(model2.new_changes) > 0
        assert any("narrowed" in c for c in model2.new_changes)

    def test_no_changes_when_same_level(self) -> None:
        """Should detect no changes when trait stays in same level band."""
        genome = _genome_with(body_size=0.5)  # mid

        model1 = build_self_description(genome, CreatureStage.PODFISH)
        # Same genome, same stage — no changes
        model2 = build_self_description(
            genome, CreatureStage.PODFISH, self_model=model1,
        )

        assert model2.new_changes == []

    def test_detects_multiple_changes(self) -> None:
        """Should detect multiple trait changes simultaneously."""
        genome_before = _genome_with(body_size=0.2, eye_size=0.2, saturation=0.2)
        genome_after = _genome_with(body_size=0.8, eye_size=0.8, saturation=0.8)

        model1 = build_self_description(genome_before, CreatureStage.FROGMAN)
        model2 = build_self_description(
            genome_after, CreatureStage.FROGMAN, self_model=model1,
        )

        assert len(model2.new_changes) >= 3

    def test_previous_levels_stored(self) -> None:
        """SelfModel should store current trait levels for next comparison."""
        genome = _genome_with(body_size=0.8)
        model = build_self_description(genome, CreatureStage.PODFISH)
        assert "body_size" in model.previous_levels
        assert model.previous_levels["body_size"] == "high"


# -----------------------------------------------------------------------
# Prompt injection format
# -----------------------------------------------------------------------


class TestPromptInjection:
    """Verify the prompt injection formatting."""

    def test_basic_format(self) -> None:
        """Should include [YOUR BODY] header and description."""
        genome = _genome_with(body_size=0.8)
        model = build_self_description(genome, CreatureStage.PODFISH)
        injection = get_prompt_injection(model)

        assert injection.startswith("[YOUR BODY]")
        assert model.description in injection

    def test_includes_changes_section(self) -> None:
        """Should include changes section when there are new changes."""
        model = SelfModel(
            description="You are large.",
            new_changes=["body has grown larger", "eyes have grown larger"],
        )
        injection = get_prompt_injection(model)

        assert "[RECENT PHYSICAL CHANGES]" in injection
        assert "- Your body has grown larger" in injection
        assert "- Your eyes have grown larger" in injection

    def test_no_changes_section_when_empty(self) -> None:
        """Should NOT include changes section when no changes."""
        model = SelfModel(description="You are large.", new_changes=[])
        injection = get_prompt_injection(model)

        assert "[RECENT PHYSICAL CHANGES]" not in injection

    def test_empty_description_returns_empty(self) -> None:
        """Empty self-model should return empty string for injection."""
        model = SelfModel()
        injection = get_prompt_injection(model)
        assert injection == ""


# -----------------------------------------------------------------------
# Edge cases and error handling
# -----------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_default_genome_produces_valid_description(self) -> None:
        """A default genome (all 0.5) should still produce a description."""
        genome = CreatureGenome()  # all traits default to 0.5
        model = build_self_description(genome, CreatureStage.PODFISH)
        assert len(model.description) > 0

    def test_extreme_low_genome(self) -> None:
        """Genome with all traits at 0.0 should work."""
        traits = {name: 0.0 for name in [
            "body_size", "head_ratio", "eye_size", "fin_length",
            "limb_proportion", "hue", "saturation", "pattern_complexity",
            "metabolism_rate", "voice_pitch", "face_expressiveness",
            "aggression_baseline",
        ]}
        genome = CreatureGenome(traits=traits)
        model = build_self_description(genome, CreatureStage.FROGMAN)
        assert "small and compact" in model.description

    def test_extreme_high_genome(self) -> None:
        """Genome with all traits at 1.0 should work."""
        traits = {name: 1.0 for name in [
            "body_size", "head_ratio", "eye_size", "fin_length",
            "limb_proportion", "hue", "saturation", "pattern_complexity",
            "metabolism_rate", "voice_pitch", "face_expressiveness",
            "aggression_baseline",
        ]}
        genome = CreatureGenome(traits=traits)
        model = build_self_description(genome, CreatureStage.FROGMAN)
        assert "large and imposing" in model.description

    def test_self_model_dataclass_defaults(self) -> None:
        """SelfModel should have sensible defaults."""
        model = SelfModel()
        assert model.description == ""
        assert model.new_changes == []
        assert model.previous_levels == {}

    def test_previous_description_param_accepted(self) -> None:
        """build_self_description should accept previous_description param."""
        genome = _genome_with()
        # Should not raise
        model = build_self_description(
            genome, CreatureStage.PODFISH, previous_description="old desc",
        )
        assert len(model.description) > 0
