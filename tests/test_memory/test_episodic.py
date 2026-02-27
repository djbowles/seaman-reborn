"""Tests for the episodic memory buffer."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from seaman_brain.memory.episodic import EpisodicMemory
from seaman_brain.types import ChatMessage, MessageRole

# --- Helpers ---

def _msg(content: str, role: MessageRole = MessageRole.USER) -> ChatMessage:
    """Create a ChatMessage with fixed timestamp for deterministic tests."""
    return ChatMessage(role=role, content=content, timestamp=datetime(2026, 1, 1, tzinfo=UTC))


# --- Happy path ---

class TestHappyPath:
    """Tests for normal operation of EpisodicMemory."""

    def test_add_and_get_all(self) -> None:
        mem = EpisodicMemory(max_size=5)
        msgs = [_msg(f"msg-{i}") for i in range(3)]
        for m in msgs:
            mem.add(m)
        assert mem.get_all() == msgs

    def test_get_recent_returns_last_n(self) -> None:
        mem = EpisodicMemory(max_size=10)
        msgs = [_msg(f"msg-{i}") for i in range(5)]
        for m in msgs:
            mem.add(m)
        recent = mem.get_recent(3)
        assert recent == msgs[2:]
        assert len(recent) == 3

    def test_len_tracks_buffer_size(self) -> None:
        mem = EpisodicMemory(max_size=10)
        assert len(mem) == 0
        mem.add(_msg("a"))
        assert len(mem) == 1
        mem.add(_msg("b"))
        assert len(mem) == 2

    def test_clear_empties_buffer(self) -> None:
        mem = EpisodicMemory(max_size=5)
        for i in range(3):
            mem.add(_msg(f"msg-{i}"))
        assert len(mem) == 3
        mem.clear()
        assert len(mem) == 0
        assert mem.get_all() == []

    def test_max_size_property(self) -> None:
        mem = EpisodicMemory(max_size=42)
        assert mem.max_size == 42

    def test_default_max_size_is_20(self) -> None:
        mem = EpisodicMemory()
        assert mem.max_size == 20

    def test_preserves_message_roles(self) -> None:
        mem = EpisodicMemory(max_size=5)
        mem.add(_msg("system prompt", role=MessageRole.SYSTEM))
        mem.add(_msg("user input", role=MessageRole.USER))
        mem.add(_msg("assistant reply", role=MessageRole.ASSISTANT))
        all_msgs = mem.get_all()
        assert all_msgs[0].role == MessageRole.SYSTEM
        assert all_msgs[1].role == MessageRole.USER
        assert all_msgs[2].role == MessageRole.ASSISTANT


# --- Overflow / eviction ---

class TestOverflow:
    """Tests for automatic eviction when buffer is full."""

    def test_evicts_oldest_when_full(self) -> None:
        mem = EpisodicMemory(max_size=3)
        msgs = [_msg(f"msg-{i}") for i in range(5)]
        for m in msgs:
            mem.add(m)
        # Only the last 3 should remain
        assert len(mem) == 3
        result = mem.get_all()
        assert result == msgs[2:]

    def test_eviction_preserves_order(self) -> None:
        mem = EpisodicMemory(max_size=2)
        mem.add(_msg("first"))
        mem.add(_msg("second"))
        mem.add(_msg("third"))
        result = mem.get_all()
        assert result[0].content == "second"
        assert result[1].content == "third"

    def test_single_slot_buffer(self) -> None:
        mem = EpisodicMemory(max_size=1)
        mem.add(_msg("a"))
        mem.add(_msg("b"))
        assert len(mem) == 1
        assert mem.get_all()[0].content == "b"

    def test_exactly_at_capacity(self) -> None:
        mem = EpisodicMemory(max_size=3)
        msgs = [_msg(f"msg-{i}") for i in range(3)]
        for m in msgs:
            mem.add(m)
        assert len(mem) == 3
        assert mem.get_all() == msgs


# --- Edge cases ---

class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_get_recent_zero_returns_empty(self) -> None:
        mem = EpisodicMemory(max_size=5)
        mem.add(_msg("a"))
        assert mem.get_recent(0) == []

    def test_get_recent_negative_returns_empty(self) -> None:
        mem = EpisodicMemory(max_size=5)
        mem.add(_msg("a"))
        assert mem.get_recent(-1) == []

    def test_get_recent_exceeding_buffer_returns_all(self) -> None:
        mem = EpisodicMemory(max_size=10)
        msgs = [_msg(f"msg-{i}") for i in range(3)]
        for m in msgs:
            mem.add(m)
        result = mem.get_recent(100)
        assert result == msgs

    def test_get_all_on_empty_buffer(self) -> None:
        mem = EpisodicMemory(max_size=5)
        assert mem.get_all() == []

    def test_get_recent_on_empty_buffer(self) -> None:
        mem = EpisodicMemory(max_size=5)
        assert mem.get_recent(5) == []

    def test_clear_on_empty_is_noop(self) -> None:
        mem = EpisodicMemory(max_size=5)
        mem.clear()  # should not raise
        assert len(mem) == 0

    def test_add_after_clear(self) -> None:
        mem = EpisodicMemory(max_size=3)
        mem.add(_msg("before"))
        mem.clear()
        mem.add(_msg("after"))
        assert len(mem) == 1
        assert mem.get_all()[0].content == "after"


# --- Error handling ---

class TestErrorHandling:
    """Tests for invalid inputs and error conditions."""

    def test_max_size_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            EpisodicMemory(max_size=0)

    def test_max_size_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size must be >= 1"):
            EpisodicMemory(max_size=-5)
