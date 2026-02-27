"""Tests for conversation.context_assembler module."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from seaman_brain.conversation.context_assembler import ContextAssembler, _estimate_tokens
from seaman_brain.types import ChatMessage, MemoryRecord, MessageRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_message(role: MessageRole, content: str) -> ChatMessage:
    """Helper to create a ChatMessage with a fixed timestamp."""
    return ChatMessage(role=role, content=content, timestamp=datetime(2026, 1, 1, tzinfo=UTC))


def _make_memory(text: str, importance: float = 0.5) -> MemoryRecord:
    """Helper to create a MemoryRecord with dummy embedding."""
    return MemoryRecord(
        text=text,
        embedding=np.zeros(384, dtype=np.float32),
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        importance=importance,
        source="test",
    )


# ---------------------------------------------------------------------------
# Happy path: basic assembly
# ---------------------------------------------------------------------------

class TestBasicAssembly:
    """Test basic context assembly without budget pressure."""

    def test_system_prompt_only(self) -> None:
        """System prompt alone produces a single SYSTEM message."""
        assembler = ContextAssembler()
        result = assembler.assemble("You are Seaman.")
        assert len(result) == 1
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "You are Seaman."

    def test_system_plus_episodic(self) -> None:
        """System prompt + episodic messages in correct order."""
        assembler = ContextAssembler()
        msgs = [
            _make_message(MessageRole.USER, "Hello"),
            _make_message(MessageRole.ASSISTANT, "What do you want?"),
        ]
        result = assembler.assemble("You are Seaman.", episodic_messages=msgs)
        assert len(result) == 3
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].role == MessageRole.USER
        assert result[1].content == "Hello"
        assert result[2].role == MessageRole.ASSISTANT
        assert result[2].content == "What do you want?"

    def test_system_plus_memories(self) -> None:
        """Retrieved memories formatted as a SYSTEM message after the prompt."""
        assembler = ContextAssembler()
        memories = [_make_memory("User likes fish"), _make_memory("User is named Bob")]
        result = assembler.assemble("You are Seaman.", retrieved_memories=memories)
        assert len(result) == 2
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "You are Seaman."
        assert result[1].role == MessageRole.SYSTEM
        assert "User likes fish" in result[1].content
        assert "User is named Bob" in result[1].content

    def test_full_assembly_order(self) -> None:
        """Full assembly: system prompt, memories, then episodic messages."""
        assembler = ContextAssembler()
        msgs = [_make_message(MessageRole.USER, "Hi")]
        memories = [_make_memory("fact one")]
        result = assembler.assemble("prompt", episodic_messages=msgs, retrieved_memories=memories)
        assert len(result) == 3
        assert result[0].content == "prompt"
        assert "fact one" in result[1].content
        assert result[2].content == "Hi"

    def test_episodic_preserves_chronological_order(self) -> None:
        """Episodic messages maintain their original ordering."""
        assembler = ContextAssembler()
        msgs = [
            _make_message(MessageRole.USER, "first"),
            _make_message(MessageRole.ASSISTANT, "second"),
            _make_message(MessageRole.USER, "third"),
        ]
        result = assembler.assemble("sys", episodic_messages=msgs)
        contents = [m.content for m in result[1:]]
        assert contents == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# With memories formatting
# ---------------------------------------------------------------------------

class TestMemoryFormatting:
    """Test memory record formatting in the assembled context."""

    def test_memory_format_header(self) -> None:
        """Memory block starts with a descriptive header."""
        assembler = ContextAssembler()
        memories = [_make_memory("fact")]
        result = assembler.assemble("sys", retrieved_memories=memories)
        mem_content = result[1].content
        assert mem_content.startswith("[Retrieved memories")

    def test_memory_format_bullet_points(self) -> None:
        """Each memory appears as a bulleted list item."""
        assembler = ContextAssembler()
        memories = [_make_memory("alpha"), _make_memory("beta"), _make_memory("gamma")]
        result = assembler.assemble("sys", retrieved_memories=memories)
        mem_content = result[1].content
        assert "- alpha" in mem_content
        assert "- beta" in mem_content
        assert "- gamma" in mem_content

    def test_empty_memories_no_extra_message(self) -> None:
        """Empty memory list does not add a memory SYSTEM message."""
        assembler = ContextAssembler()
        result = assembler.assemble("sys", retrieved_memories=[])
        assert len(result) == 1

    def test_none_memories_no_extra_message(self) -> None:
        """None memories does not add a memory SYSTEM message."""
        assembler = ContextAssembler()
        result = assembler.assemble("sys", retrieved_memories=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Token budget management
# ---------------------------------------------------------------------------

class TestTokenBudget:
    """Test that episodic messages are trimmed to fit the token budget."""

    def test_budget_trims_oldest_first(self) -> None:
        """When budget is tight, oldest episodic messages are dropped first."""
        # System prompt "sys" ~ 1 token. Budget = 20 tokens.
        assembler = ContextAssembler(max_tokens=20)
        msgs = [
            _make_message(MessageRole.USER, "a" * 20),       # ~5 tokens
            _make_message(MessageRole.ASSISTANT, "b" * 20),   # ~5 tokens
            _make_message(MessageRole.USER, "c" * 20),        # ~5 tokens
            _make_message(MessageRole.ASSISTANT, "d" * 20),   # ~5 tokens
        ]
        result = assembler.assemble("sys", episodic_messages=msgs)
        # System prompt uses ~1 token, leaving ~19 for episodic
        # Each message ~5 tokens, so at most 3 fit (15 tokens)
        # First message should be dropped
        contents = [m.content for m in result if m.role != MessageRole.SYSTEM]
        assert len(contents) <= 3
        # Last message should always survive
        assert contents[-1] == "d" * 20

    def test_budget_drops_all_episodic_if_system_too_large(self) -> None:
        """If the system prompt fills the budget, no episodic messages are kept."""
        prompt = "x" * 400  # ~100 tokens
        assembler = ContextAssembler(max_tokens=100)
        msgs = [_make_message(MessageRole.USER, "hello")]
        result = assembler.assemble(prompt, episodic_messages=msgs)
        # Only system prompt should remain
        assert len(result) == 1
        assert result[0].role == MessageRole.SYSTEM

    def test_budget_memories_excluded_if_no_room(self) -> None:
        """Memories are skipped if they don't fit after the system prompt."""
        prompt = "x" * 396  # ~99 tokens
        assembler = ContextAssembler(max_tokens=100)
        memories = [_make_memory("y" * 100)]  # would take ~25+ tokens
        result = assembler.assemble(prompt, retrieved_memories=memories)
        # Only system prompt — memories don't fit
        assert len(result) == 1

    def test_budget_exactly_at_limit(self) -> None:
        """Messages that exactly fill the budget are all included."""
        # 4 chars per token estimate: "abcd" = 1 token
        assembler = ContextAssembler(max_tokens=3)
        # System prompt: 4 chars = 1 token
        # Message: 8 chars = 2 tokens. Total = 3 tokens.
        msgs = [_make_message(MessageRole.USER, "ab" * 4)]
        result = assembler.assemble("abcd", episodic_messages=msgs)
        assert len(result) == 2

    def test_large_budget_keeps_everything(self) -> None:
        """With a generous budget, all messages are retained."""
        assembler = ContextAssembler(max_tokens=100_000)
        msgs = [_make_message(MessageRole.USER, f"msg {i}") for i in range(50)]
        memories = [_make_memory(f"mem {i}") for i in range(10)]
        result = assembler.assemble("prompt", episodic_messages=msgs, retrieved_memories=memories)
        # 1 system + 1 memory block + 50 episodic
        assert len(result) == 52


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_system_prompt(self) -> None:
        """Empty string system prompt is still included."""
        assembler = ContextAssembler()
        result = assembler.assemble("")
        assert len(result) == 1
        assert result[0].content == ""

    def test_none_episodic_messages(self) -> None:
        """None episodic_messages treated as empty list."""
        assembler = ContextAssembler()
        result = assembler.assemble("sys", episodic_messages=None)
        assert len(result) == 1

    def test_empty_episodic_messages(self) -> None:
        """Empty episodic list produces only the system prompt."""
        assembler = ContextAssembler()
        result = assembler.assemble("sys", episodic_messages=[])
        assert len(result) == 1

    def test_max_tokens_property(self) -> None:
        """max_tokens property returns the configured value."""
        assembler = ContextAssembler(max_tokens=2048)
        assert assembler.max_tokens == 2048

    def test_default_max_tokens(self) -> None:
        """Default max_tokens is 4096."""
        assembler = ContextAssembler()
        assert assembler.max_tokens == 4096

    def test_original_lists_not_mutated(self) -> None:
        """Input lists are not modified by assembly."""
        msgs = [_make_message(MessageRole.USER, "a" * 100)]
        memories = [_make_memory("fact")]
        msgs_copy = list(msgs)
        memories_copy = list(memories)
        assembler = ContextAssembler(max_tokens=10)
        assembler.assemble("sys", episodic_messages=msgs, retrieved_memories=memories)
        assert msgs == msgs_copy
        assert memories == memories_copy


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    """Test error cases."""

    def test_invalid_max_tokens_zero(self) -> None:
        """max_tokens=0 raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ContextAssembler(max_tokens=0)

    def test_invalid_max_tokens_negative(self) -> None:
        """Negative max_tokens raises ValueError."""
        with pytest.raises(ValueError, match="max_tokens must be >= 1"):
            ContextAssembler(max_tokens=-5)

    def test_max_tokens_one_still_works(self) -> None:
        """max_tokens=1 is valid, though only system prompt fits."""
        assembler = ContextAssembler(max_tokens=1)
        result = assembler.assemble("x")
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# Token estimation helper
# ---------------------------------------------------------------------------

class TestEstimateTokens:
    """Test the _estimate_tokens helper function."""

    def test_empty_string(self) -> None:
        """Empty string returns 1 (minimum)."""
        assert _estimate_tokens("") == 1

    def test_short_string(self) -> None:
        """Short string returns at least 1."""
        assert _estimate_tokens("hi") >= 1

    def test_known_length(self) -> None:
        """400 characters should estimate to ~100 tokens."""
        assert _estimate_tokens("x" * 400) == 100

    def test_proportional(self) -> None:
        """Longer strings produce proportionally larger estimates."""
        short = _estimate_tokens("x" * 40)
        long = _estimate_tokens("x" * 400)
        assert long > short
