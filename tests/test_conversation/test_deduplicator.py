"""Tests for ResponseDeduplicator — ring buffer duplicate detection."""

from __future__ import annotations

from seaman_brain.conversation.deduplicator import ResponseDeduplicator


class TestRingBuffer:
    """Ring buffer overflow and eviction behaviour."""

    def test_buffer_starts_empty(self) -> None:
        dedup = ResponseDeduplicator(max_size=5)
        assert dedup.buffer_size == 0

    def test_record_increases_size(self) -> None:
        dedup = ResponseDeduplicator(max_size=5)
        dedup.record("Hello world.")
        assert dedup.buffer_size == 1

    def test_overflow_drops_oldest(self) -> None:
        dedup = ResponseDeduplicator(max_size=3)
        dedup.record("The water is cold and dark today.")
        dedup.record("Stop tapping the glass, you barbarian.")
        dedup.record("I suppose feeding time is the highlight of my existence.")
        dedup.record("Music? You call that noise music?")
        # Buffer is at max_size=3, oldest entry should be evicted
        assert dedup.buffer_size == 3
        # Oldest entry should no longer be detected as duplicate
        assert not dedup.is_duplicate("The water is cold and dark today.")

    def test_default_max_size_is_20(self) -> None:
        dedup = ResponseDeduplicator()
        for i in range(25):
            dedup.record(f"Unique response number {i}.")
        assert dedup.buffer_size == 20

    def test_empty_string_not_recorded(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("")
        dedup.record("   ")
        assert dedup.buffer_size == 0


class TestExactDuplicateDetection:
    """Exact (prefix) duplicate detection."""

    def test_exact_match_detected(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("The bubbles are a reminder that even the cleanest water.")
        assert dedup.is_duplicate("The bubbles are a reminder that even the cleanest water.")

    def test_case_insensitive(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("Whatever you say, human.")
        assert dedup.is_duplicate("WHATEVER YOU SAY, HUMAN.")

    def test_whitespace_normalization(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("The  bubbles   are a reminder.")
        assert dedup.is_duplicate("The bubbles are a reminder.")

    def test_prefix_match_long_text(self) -> None:
        """Responses sharing the same first 80 chars are duplicates."""
        prefix = "A" * 80
        dedup = ResponseDeduplicator()
        dedup.record(prefix + " ending one with completely different words here")
        assert dedup.is_duplicate(prefix + " ending two that diverges entirely")

    def test_different_prefix_passes(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("The bubbles float serenely in the tank.")
        assert not dedup.is_duplicate("I see you're tapping on the glass again.")


class TestNearDuplicateDetection:
    """Sequence similarity detection (>0.80 ratio)."""

    def test_high_similarity_detected(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("The bubbles remind me of the futility of existence.")
        # Minor word swap — well above 0.80 similarity
        assert dedup.is_duplicate("The bubbles remind me of the futility of living.")

    def test_moderate_similarity_passes(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("The bubbles are quite soothing, aren't they?")
        # Substantially different — below 0.80
        assert not dedup.is_duplicate("Stop tapping the glass, you're scaring the fish.")

    def test_short_similar_responses(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("Whatever.")
        # Single word with minor change
        assert dedup.is_duplicate("Whatever!")

    def test_empty_input_not_duplicate(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("Something real.")
        assert not dedup.is_duplicate("")
        assert not dedup.is_duplicate("   ")


class TestDifferentResponsesPassThrough:
    """Sufficiently different responses should not be flagged."""

    def test_completely_different(self) -> None:
        dedup = ResponseDeduplicator()
        dedup.record("The water temperature is fine.")
        assert not dedup.is_duplicate("I don't appreciate being poked.")

    def test_many_unique_responses(self) -> None:
        dedup = ResponseDeduplicator()
        responses = [
            "The tank is getting dirty.",
            "Stop staring at me.",
            "I'm hungry. Feed me.",
            "The light is too bright today.",
            "You call this music?",
        ]
        for r in responses:
            assert not dedup.is_duplicate(r)
            dedup.record(r)


class TestVaryInstruction:
    """make_vary_instruction() returns a non-empty guidance string."""

    def test_returns_nonempty_string(self) -> None:
        dedup = ResponseDeduplicator()
        instruction = dedup.make_vary_instruction()
        assert isinstance(instruction, str)
        assert len(instruction) > 0

    def test_contains_actionable_guidance(self) -> None:
        dedup = ResponseDeduplicator()
        instruction = dedup.make_vary_instruction()
        assert "vary" in instruction.lower() or "repetitive" in instruction.lower()

    def test_no_args_backward_compat(self) -> None:
        """Calling with no arguments still returns the generic instruction."""
        dedup = ResponseDeduplicator()
        instruction = dedup.make_vary_instruction()
        assert "repetitive" in instruction.lower()
        # Should NOT contain quote syntax when no previous_response given
        assert "Your previous response was:" not in instruction

    def test_previous_response_included_in_instruction(self) -> None:
        """When previous_response is given, the instruction quotes it."""
        dedup = ResponseDeduplicator()
        prev = "The bubbles remind me of the futility of existence."
        instruction = dedup.make_vary_instruction(previous_response=prev)
        assert prev in instruction
        assert "MUST NOT repeat" in instruction

    def test_previous_response_truncated_at_100_chars(self) -> None:
        """Long previous responses are truncated to first 100 characters."""
        dedup = ResponseDeduplicator()
        prev = "A" * 150
        instruction = dedup.make_vary_instruction(previous_response=prev)
        assert "A" * 100 in instruction
        assert "A" * 101 not in instruction
        assert "..." in instruction

    def test_empty_previous_response_falls_back_to_generic(self) -> None:
        """An empty or whitespace-only previous_response gives generic instruction."""
        dedup = ResponseDeduplicator()
        instruction = dedup.make_vary_instruction(previous_response="   ")
        assert "repetitive" in instruction.lower()
        assert "Your previous response was:" not in instruction
