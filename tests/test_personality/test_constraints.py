"""Tests for personality constraints and output filtering."""

from __future__ import annotations

import pytest

from seaman_brain.personality.constraints import (
    FORBIDDEN_PHRASES,
    _clean_whitespace,
    _max_length_for_verbosity,
    _strip_forbidden,
    _strip_think_blocks,
    _truncate,
    apply_constraints,
)
from seaman_brain.personality.traits import TraitProfile

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_profile() -> TraitProfile:
    """A default TraitProfile (verbosity=0.5 -> max ~155 chars)."""
    return TraitProfile()


@pytest.fixture
def terse_profile() -> TraitProfile:
    """A very terse creature (verbosity=0.0 -> max ~10 chars)."""
    return TraitProfile(verbosity=0.0)


@pytest.fixture
def verbose_profile() -> TraitProfile:
    """A very verbose creature (verbosity=1.0 -> max 300 chars)."""
    return TraitProfile(verbosity=1.0)


# ---------------------------------------------------------------------------
# Happy path: phrase stripping
# ---------------------------------------------------------------------------

class TestPhraseStripping:
    """Tests for forbidden phrase removal."""

    def test_strips_as_an_ai(self, default_profile: TraitProfile) -> None:
        response = "As an AI, I think fish are cool."
        result = apply_constraints(response, default_profile)
        assert "as an ai" not in result.lower()
        assert "fish are cool" in result

    def test_strips_happy_to_help(self, default_profile: TraitProfile) -> None:
        response = "I would be happy to discuss that. The tank is dirty."
        result = apply_constraints(response, default_profile)
        assert "happy to" not in result.lower()
        assert "tank is dirty" in result

    def test_strips_multiple_phrases(self, default_profile: TraitProfile) -> None:
        response = (
            "Great question! As an AI, I'd be happy to help. "
            "The water is fine."
        )
        result = apply_constraints(response, default_profile)
        assert "great question" not in result.lower()
        assert "as an ai" not in result.lower()
        assert "happy to" not in result.lower()
        assert "water is fine" in result

    def test_case_insensitive_stripping(self, default_profile: TraitProfile) -> None:
        response = "AS AN AI, I THINK you're boring. I Hope This Helps!"
        result = apply_constraints(response, default_profile)
        assert "as an ai" not in result.lower()
        assert "i hope this helps" not in result.lower()
        assert "boring" in result.lower()

    def test_strips_certainly_pattern(self, default_profile: TraitProfile) -> None:
        response = "Certainly! I can tell you about fish."
        result = apply_constraints(response, default_profile)
        assert "certainly! i" not in result.lower()
        assert "fish" in result

    def test_clean_response_unchanged(self, default_profile: TraitProfile) -> None:
        response = "You look terrible today. Feed me."
        result = apply_constraints(response, default_profile)
        assert result == response


# ---------------------------------------------------------------------------
# Happy path: length enforcement
# ---------------------------------------------------------------------------

class TestLengthEnforcement:
    """Tests for verbosity-based length limits."""

    def test_terse_enforces_short_length(self, terse_profile: TraitProfile) -> None:
        response = "A" * 100
        result = apply_constraints(response, terse_profile)
        assert len(result) <= 10

    def test_verbose_allows_long_responses(self, verbose_profile: TraitProfile) -> None:
        response = "Word " * 50  # ~250 chars
        result = apply_constraints(response, verbose_profile)
        assert len(result) <= 300

    def test_default_verbosity_medium_length(self, default_profile: TraitProfile) -> None:
        max_len = _max_length_for_verbosity(0.5)
        assert max_len == 155
        response = "X" * 200
        result = apply_constraints(response, default_profile)
        assert len(result) <= 155

    def test_short_response_not_truncated(self, terse_profile: TraitProfile) -> None:
        response = "No."
        result = apply_constraints(response, terse_profile)
        assert result == "No."

    def test_truncates_at_sentence_boundary(self, terse_profile: TraitProfile) -> None:
        response = "Hi. Bye."
        result = apply_constraints(response, terse_profile)
        # Max 10 chars, should break at "Hi." (sentence boundary)
        assert result.endswith(".")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_returns_empty(self, default_profile: TraitProfile) -> None:
        assert apply_constraints("", default_profile) == ""

    def test_whitespace_only_returns_empty(self, default_profile: TraitProfile) -> None:
        assert apply_constraints("   \n\t  ", default_profile) == ""

    def test_none_like_empty(self, default_profile: TraitProfile) -> None:
        # Empty string is falsy
        assert apply_constraints("", default_profile) == ""

    def test_all_forbidden_stripped_returns_empty(
        self, default_profile: TraitProfile,
    ) -> None:
        response = "As an AI, I would be happy to help."
        result = apply_constraints(response, default_profile)
        # After stripping, only "help." remains
        assert "as an ai" not in result.lower()
        assert "happy to" not in result.lower()

    def test_leading_punctuation_cleaned_after_strip(
        self, default_profile: TraitProfile,
    ) -> None:
        response = "As an AI, the water is cold."
        result = apply_constraints(response, default_profile)
        # Should not start with ", " after stripping "As an AI"
        assert not result.startswith(",")
        assert not result.startswith(" ")

    def test_multiple_spaces_collapsed(self, default_profile: TraitProfile) -> None:
        response = "The   water    is     fine."
        result = apply_constraints(response, default_profile)
        assert "  " not in result

    def test_forbidden_phrases_list_nonempty(self) -> None:
        assert len(FORBIDDEN_PHRASES) > 20

    def test_verbosity_clamped_below_zero(self) -> None:
        assert _max_length_for_verbosity(-1.0) == 10

    def test_verbosity_clamped_above_one(self) -> None:
        assert _max_length_for_verbosity(2.0) == 300


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestHelpers:
    """Tests for internal helper functions."""

    def test_strip_forbidden_no_match(self) -> None:
        text = "The fish glares at you."
        assert _strip_forbidden(text) == text

    def test_clean_whitespace_collapses_newlines(self) -> None:
        text = "Line 1\n\n\n\n\nLine 2"
        result = _clean_whitespace(text)
        assert result == "Line 1\n\nLine 2"

    def test_truncate_short_text_unchanged(self) -> None:
        assert _truncate("Hello", 100) == "Hello"

    def test_truncate_breaks_at_sentence(self) -> None:
        text = "First sentence. Second sentence. Third sentence."
        result = _truncate(text, 35)
        assert result.endswith(".")
        assert "Third" not in result

    def test_truncate_breaks_at_word(self) -> None:
        text = "word " * 20  # 100 chars, no sentence boundaries
        result = _truncate(text, 30)
        assert not result.endswith(" ")
        assert len(result) <= 30

    def test_max_length_linear_interpolation(self) -> None:
        assert _max_length_for_verbosity(0.0) == 10
        assert _max_length_for_verbosity(0.5) == 155
        assert _max_length_for_verbosity(1.0) == 300

    def test_max_length_destressed(self) -> None:
        assert _max_length_for_verbosity(0.0, destressed=True) == 10
        assert _max_length_for_verbosity(0.5, destressed=True) == 405
        assert _max_length_for_verbosity(1.0, destressed=True) == 800


# ---------------------------------------------------------------------------
# Think block stripping (Qwen3-style <think>...</think>)
# ---------------------------------------------------------------------------


class TestThinkBlockStripping:
    """Tests for stripping <think> reasoning blocks from LLM output."""

    def test_strip_simple_think_block(self) -> None:
        text = "<think>\nI should be sarcastic.\n</think>\nUgh, whatever."
        result = _strip_think_blocks(text)
        assert "<think>" not in result
        assert "sarcastic" not in result
        assert "Ugh, whatever." in result

    def test_strip_think_block_inline(self) -> None:
        text = "<think>reasoning</think>Hello there."
        result = _strip_think_blocks(text)
        assert result == "Hello there."

    def test_strip_multiline_think_block(self) -> None:
        text = (
            "<think>\nLine 1\nLine 2\nLine 3\n</think>\n"
            "The actual response."
        )
        result = _strip_think_blocks(text)
        assert "Line 1" not in result
        assert "The actual response." in result

    def test_no_think_block_unchanged(self) -> None:
        text = "Just a normal response."
        assert _strip_think_blocks(text) == text

    def test_case_insensitive(self) -> None:
        text = "<THINK>reasoning</THINK>Result."
        result = _strip_think_blocks(text)
        assert result == "Result."

    def test_apply_constraints_strips_think(
        self, default_profile: TraitProfile,
    ) -> None:
        text = "<think>\nI am a podfish.\n</think>\nLeave me alone."
        result = apply_constraints(text, default_profile)
        assert "podfish" not in result
        assert "Leave me alone." in result


class TestAsteriskStripping:
    """Tests for stripping * characters from LLM output."""

    def test_strips_action_markers(self, default_profile: TraitProfile) -> None:
        text = "*sighs deeply* Whatever."
        result = apply_constraints(text, default_profile)
        assert "*" not in result
        assert "sighs deeply" in result
        assert "Whatever." in result

    def test_strips_emphasis(self, default_profile: TraitProfile) -> None:
        text = "I am *very* annoyed."
        result = apply_constraints(text, default_profile)
        assert "*" not in result
        assert "very" in result

    def test_no_asterisks_unchanged(self, default_profile: TraitProfile) -> None:
        text = "Just a normal sentence."
        result = apply_constraints(text, default_profile)
        assert result == text


# ---------------------------------------------------------------------------
# Destress mode tests
# ---------------------------------------------------------------------------

class TestDestressMode:
    """Tests for destress soliloquy length override."""

    def test_destress_allows_longer_response(self) -> None:
        """destressed=True uses higher ceiling, doesn't truncate at 300."""
        profile = TraitProfile(verbosity=1.0)
        response = "A" * 600
        result = apply_constraints(response, profile, destressed=True)
        assert len(result) > 300
        assert len(result) <= 800

    def test_normal_truncates_at_lower_limit(self) -> None:
        """Without destress, text IS truncated at 300."""
        profile = TraitProfile(verbosity=1.0)
        response = "A" * 600
        result = apply_constraints(response, profile, destressed=False)
        assert len(result) <= 300

    def test_destress_default_false(self) -> None:
        """destressed defaults to False (backward compat)."""
        profile = TraitProfile(verbosity=1.0)
        response = "A" * 600
        result = apply_constraints(response, profile)
        assert len(result) <= 300
