"""Tests for personality prompt builder."""

from __future__ import annotations

import pytest

from seaman_brain.personality.prompt_builder import (
    PromptBuilder,
    _get_stage_description,
    _memories_section,
    _mood_section,
    _trait_tone_instructions,
    _vision_section,
)
from seaman_brain.personality.traits import TraitProfile
from seaman_brain.types import CreatureStage

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def builder(tmp_path):
    """PromptBuilder pointing at an empty config dir (uses fallbacks)."""
    config_dir = tmp_path / "config"
    (config_dir / "stages").mkdir(parents=True)
    return PromptBuilder(config_dir=str(config_dir))


@pytest.fixture
def default_traits():
    """Default TraitProfile (all 0.5)."""
    return TraitProfile()


@pytest.fixture
def sardonic_traits():
    """High-cynicism, high-wit Podfish-like traits."""
    return TraitProfile(
        cynicism=0.8, wit=0.9, patience=0.3, curiosity=0.7,
        warmth=0.2, verbosity=0.5, formality=0.2, aggression=0.4,
    )


@pytest.fixture
def warm_traits():
    """Warm Frogman-like traits."""
    return TraitProfile(
        cynicism=0.6, wit=0.9, patience=0.7, curiosity=0.9,
        warmth=0.5, verbosity=0.7, formality=0.4, aggression=0.2,
    )


# ---------------------------------------------------------------------------
# Happy path tests
# ---------------------------------------------------------------------------

class TestBuildBasic:
    """Test basic prompt building."""

    def test_build_returns_string(self, builder, default_traits):
        """build() returns a non-empty string."""
        result = builder.build(CreatureStage.PODFISH, default_traits)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_build_contains_identity(self, builder, default_traits):
        """Prompt contains 'YOU ARE SEAMAN' identity header."""
        result = builder.build(CreatureStage.MUSHROOMER, default_traits)
        assert "YOU ARE SEAMAN" in result

    def test_build_contains_negative_constraints(self, builder, default_traits):
        """Prompt contains the negative constraints section."""
        result = builder.build(CreatureStage.PODFISH, default_traits)
        assert 'NEVER say "As an AI"' in result
        assert "NEVER be solicitous" in result
        assert "NEVER break character" in result

    def test_build_contains_stage_description(self, builder, sardonic_traits):
        """Prompt contains stage-appropriate description."""
        result = builder.build(CreatureStage.PODFISH, sardonic_traits)
        assert "sardonic" in result.lower()

    def test_build_contains_speech_style(self, builder, default_traits):
        """Prompt contains speech style guidance for the stage."""
        result = builder.build(CreatureStage.MUSHROOMER, default_traits)
        assert "short fragments" in result.lower()


class TestBuildWithMemories:
    """Test prompt building with remembered facts."""

    def test_build_with_memories(self, builder, default_traits):
        """Memories are included in the prompt."""
        memories = ["The human's name is Dave", "Dave likes cats"]
        result = builder.build(
            CreatureStage.PODFISH, default_traits, memories=memories,
        )
        assert "Dave" in result
        assert "cats" in result

    def test_build_with_empty_memories(self, builder, default_traits):
        """Empty memory list produces no memory section."""
        result_no_mem = builder.build(
            CreatureStage.PODFISH, default_traits, memories=[],
        )
        assert "THINGS YOU REMEMBER" not in result_no_mem

    def test_build_with_none_memories(self, builder, default_traits):
        """None memories is handled gracefully."""
        result = builder.build(
            CreatureStage.PODFISH, default_traits, memories=None,
        )
        assert "THINGS YOU REMEMBER" not in result

    def test_memories_section_format(self):
        """Memory section has header, bullets, and usage instruction."""
        memories = ["Fact one", "Fact two"]
        section = _memories_section(memories)
        assert "THINGS YOU REMEMBER" in section
        assert "- Fact one" in section
        assert "- Fact two" in section
        assert "naturally" in section.lower()


class TestBuildDifferentStages:
    """Test prompt building across different evolutionary stages."""

    def test_mushroomer_primitive(self, builder):
        """Mushroomer prompt enforces primitive speech."""
        traits = TraitProfile(
            cynicism=0.5, wit=0.3, patience=0.2, curiosity=0.4,
            warmth=0.1, verbosity=0.2, formality=0.1, aggression=0.6,
        )
        result = builder.build(CreatureStage.MUSHROOMER, traits)
        assert "primitive" in result.lower() or "grunt" in result.lower()
        assert "short" in result.lower()

    def test_podfish_sardonic(self, builder, sardonic_traits):
        """Podfish prompt emphasizes sardonic wit."""
        result = builder.build(CreatureStage.PODFISH, sardonic_traits)
        assert "sardonic" in result.lower() or "wit" in result.lower()

    def test_frogman_wise(self, builder, warm_traits):
        """Frogman prompt emphasizes wisdom."""
        result = builder.build(CreatureStage.FROGMAN, warm_traits)
        assert "wise" in result.lower() or "wisdom" in result.lower()

    def test_all_stages_produce_different_prompts(self, builder, default_traits):
        """Each stage produces a distinct prompt."""
        prompts = {
            stage: builder.build(stage, default_traits)
            for stage in CreatureStage
        }
        # All prompts should be unique
        prompt_texts = list(prompts.values())
        assert len(set(prompt_texts)) == len(prompt_texts)


class TestBuildWithCreatureState:
    """Test prompt building with creature state dict."""

    def test_build_with_mood(self, builder, default_traits):
        """Mood is reflected in the prompt."""
        state = {"mood": "hostile", "trust_level": 0.1}
        result = builder.build(
            CreatureStage.PODFISH, default_traits, creature_state=state,
        )
        assert "HOSTILE" in result

    def test_build_with_high_trust(self, builder, default_traits):
        """High trust adds bonding language."""
        state = {"trust_level": 0.9}
        result = builder.build(
            CreatureStage.FROGMAN, default_traits, creature_state=state,
        )
        assert "bond" in result.lower()

    def test_build_with_low_trust(self, builder, default_traits):
        """Low trust adds distrust language."""
        state = {"trust_level": 0.1}
        result = builder.build(
            CreatureStage.MUSHROOMER, default_traits, creature_state=state,
        )
        assert "distrust" in result.lower()

    def test_build_with_hunger(self, builder, default_traits):
        """High hunger triggers starvation complaints."""
        state = {"hunger": 0.9}
        result = builder.build(
            CreatureStage.PODFISH, default_traits, creature_state=state,
        )
        assert "starving" in result.lower() or "hunger" in result.lower()

    def test_build_with_low_health(self, builder, default_traits):
        """Low health triggers illness language."""
        state = {"health": 0.2}
        result = builder.build(
            CreatureStage.PODFISH, default_traits, creature_state=state,
        )
        assert "unwell" in result.lower() or "terrible" in result.lower()

    def test_build_with_new_interaction(self, builder, default_traits):
        """Low interaction count triggers suspicion."""
        state = {"interaction_count": 2}
        result = builder.build(
            CreatureStage.GILLMAN, default_traits, creature_state=state,
        )
        assert "suspicious" in result.lower() or "new" in result.lower()

    def test_build_with_empty_state(self, builder, default_traits):
        """Empty state dict is handled gracefully."""
        result = builder.build(
            CreatureStage.PODFISH, default_traits, creature_state={},
        )
        assert "YOU ARE SEAMAN" in result

    def test_build_with_none_state(self, builder, default_traits):
        """None state is handled gracefully."""
        result = builder.build(
            CreatureStage.PODFISH, default_traits, creature_state=None,
        )
        assert "YOU ARE SEAMAN" in result


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestTraitToneInstructions:
    """Test trait-to-tone generation."""

    def test_high_cynicism(self):
        """High cynicism produces distrust language."""
        traits = TraitProfile(cynicism=0.9)
        tone = _trait_tone_instructions(traits)
        assert "cynical" in tone.lower() or "distrust" in tone.lower()

    def test_low_patience(self):
        """Low patience produces irritation language."""
        traits = TraitProfile(patience=0.1)
        tone = _trait_tone_instructions(traits)
        assert "patience" in tone.lower() or "irritat" in tone.lower()

    def test_high_wit(self):
        """High wit produces humor language."""
        traits = TraitProfile(wit=0.8)
        tone = _trait_tone_instructions(traits)
        assert "humor" in tone.lower() or "wordplay" in tone.lower()

    def test_high_curiosity(self):
        """High curiosity produces probing language."""
        traits = TraitProfile(curiosity=0.8)
        tone = _trait_tone_instructions(traits)
        assert "curious" in tone.lower() or "probing" in tone.lower()

    def test_high_warmth(self):
        """Moderate warmth produces affection language."""
        traits = TraitProfile(warmth=0.5)
        tone = _trait_tone_instructions(traits)
        assert "affection" in tone.lower() or "grudging" in tone.lower()

    def test_very_low_warmth(self):
        """Very low warmth produces cold language."""
        traits = TraitProfile(warmth=0.1)
        tone = _trait_tone_instructions(traits)
        assert "nothing" in tone.lower() or "merely" in tone.lower()

    def test_high_aggression(self):
        """High aggression produces combative language."""
        traits = TraitProfile(aggression=0.7)
        tone = _trait_tone_instructions(traits)
        assert "combative" in tone.lower() or "provoke" in tone.lower()


class TestMoodSection:
    """Test mood section generation."""

    def test_hostile_mood(self):
        """Hostile mood produces aggressive text."""
        text = _mood_section("hostile", None)
        assert "HOSTILE" in text

    def test_philosophical_mood(self):
        """Philosophical mood produces pondering text."""
        text = _mood_section("philosophical", None)
        assert "PHILOSOPHICAL" in text

    def test_unknown_mood(self):
        """Unknown mood falls back to raw mood name."""
        text = _mood_section("confused", None)
        assert "confused" in text.lower()

    def test_none_mood(self):
        """None mood produces no mood text."""
        text = _mood_section(None, None)
        assert text == ""

    def test_trust_levels(self):
        """Different trust levels produce different text."""
        low = _mood_section(None, 0.1)
        mid = _mood_section(None, 0.5)
        high = _mood_section(None, 0.9)
        assert low != mid != high

    def test_mood_and_trust_combined(self):
        """Mood and trust together both appear."""
        text = _mood_section("sardonic", 0.5)
        assert "SARDONIC" in text
        assert "accustomed" in text.lower()


class TestStageDescription:
    """Test stage description loading."""

    def test_fallback_descriptions(self, tmp_path):
        """All stages have fallback descriptions when no TOML exists."""
        config_dir = tmp_path / "config"
        (config_dir / "stages").mkdir(parents=True)
        for stage in CreatureStage:
            desc = _get_stage_description(stage, str(config_dir))
            assert len(desc) > 0

    def test_toml_description_override(self, tmp_path):
        """TOML behavior.description overrides the fallback."""
        config_dir = tmp_path / "config"
        stages_dir = config_dir / "stages"
        stages_dir.mkdir(parents=True)
        toml_content = (
            '[behavior]\n'
            'description = "A custom test creature"\n'
        )
        (stages_dir / "podfish.toml").write_text(toml_content)
        desc = _get_stage_description(CreatureStage.PODFISH, str(config_dir))
        assert desc == "A custom test creature"


class TestNeedsHints:
    """Test needs-driven behavior hints."""

    def test_starving_hint(self, builder):
        """Very high hunger triggers starvation hint."""
        hint = builder._needs_hints({"hunger": 0.9})
        assert "STARVING" in hint

    def test_moderate_hunger_hint(self, builder):
        """Moderate hunger triggers food mention."""
        hint = builder._needs_hints({"hunger": 0.6})
        assert "hungry" in hint.lower()

    def test_low_hunger_no_hint(self, builder):
        """Low hunger produces no hint."""
        hint = builder._needs_hints({"hunger": 0.3})
        assert hint == ""

    def test_low_health_hint(self, builder):
        """Low health triggers illness hint."""
        hint = builder._needs_hints({"health": 0.2})
        assert "unwell" in hint.lower() or "terrible" in hint.lower()

    def test_new_human_hint(self, builder):
        """Low interaction count triggers suspicion."""
        hint = builder._needs_hints({"interaction_count": 3})
        assert "suspicious" in hint.lower() or "new" in hint.lower()

    def test_empty_state(self, builder):
        """Empty state produces no hints."""
        hint = builder._needs_hints({})
        assert hint == ""


# ---------------------------------------------------------------------------
# Vision section tests
# ---------------------------------------------------------------------------

class TestVisionSection:
    """Test vision observation section generation."""

    def test_empty_observations(self):
        """Empty observations produce no section."""
        assert _vision_section([]) == ""

    def test_single_observation(self):
        """Single observation produces a formatted section."""
        section = _vision_section(["The human looks bored"])
        assert "WHAT YOU CAN SEE RIGHT NOW:" in section
        assert "- The human looks bored" in section
        assert "React naturally" in section

    def test_multiple_observations(self):
        """Multiple observations are all listed."""
        obs = ["Human is typing", "The room is dark", "A cat sits nearby"]
        section = _vision_section(obs)
        for o in obs:
            assert f"- {o}" in section

    def test_vision_in_build_output(self, builder, default_traits):
        """Observations appear in the full build() output."""
        result = builder.build(
            CreatureStage.PODFISH, default_traits,
            observations=["The human is smiling"],
        )
        assert "WHAT YOU CAN SEE RIGHT NOW:" in result
        assert "The human is smiling" in result

    def test_no_vision_when_none(self, builder, default_traits):
        """None observations produce no vision section."""
        result = builder.build(
            CreatureStage.PODFISH, default_traits,
            observations=None,
        )
        assert "WHAT YOU CAN SEE" not in result

    def test_no_vision_when_empty(self, builder, default_traits):
        """Empty observations list produces no vision section."""
        result = builder.build(
            CreatureStage.PODFISH, default_traits,
            observations=[],
        )
        assert "WHAT YOU CAN SEE" not in result
