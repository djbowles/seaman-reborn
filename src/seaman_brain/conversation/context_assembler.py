"""Context assembly for LLM calls.

Combines the system prompt, retrieved long-term memories, and recent episodic
messages into a well-structured list[ChatMessage] ready for the LLM, while
respecting a configurable token budget.
"""

from __future__ import annotations

from seaman_brain.types import ChatMessage, MemoryRecord, MessageRole


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token."""
    return max(1, len(text) // 4)


class ContextAssembler:
    """Assembles the full context array sent to the LLM.

    Ordering:
        1. System prompt (SYSTEM message)
        2. Memory context (SYSTEM message with formatted retrieved memories)
        3. Episodic messages in chronological order

    Token budget management trims episodic messages (oldest first) when
    the total estimated token count exceeds the configured limit.
    """

    def __init__(self, max_tokens: int = 4096) -> None:
        """Initialize the assembler.

        Args:
            max_tokens: Total token budget for the assembled context.
                        Must be >= 1.
        """
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {max_tokens}")
        self._max_tokens = max_tokens

    @property
    def max_tokens(self) -> int:
        """The token budget for context assembly."""
        return self._max_tokens

    def assemble(
        self,
        system_prompt: str,
        episodic_messages: list[ChatMessage] | None = None,
        retrieved_memories: list[MemoryRecord] | None = None,
    ) -> list[ChatMessage]:
        """Assemble the full context for an LLM call.

        Args:
            system_prompt: The system prompt string (from PromptBuilder).
            episodic_messages: Recent conversation messages in chronological order.
            retrieved_memories: Long-term memories ranked by relevance.

        Returns:
            Ordered list of ChatMessage ready for the LLM provider.
        """
        messages: list[ChatMessage] = []
        used_tokens = 0

        # 1. System prompt — always first, always included
        system_msg = ChatMessage(role=MessageRole.SYSTEM, content=system_prompt)
        used_tokens += _estimate_tokens(system_prompt)
        messages.append(system_msg)

        # 2. Retrieved memories as a secondary SYSTEM message
        memories = retrieved_memories or []
        if memories:
            memory_text = self._format_memories(memories)
            mem_tokens = _estimate_tokens(memory_text)
            if used_tokens + mem_tokens <= self._max_tokens:
                messages.append(
                    ChatMessage(role=MessageRole.SYSTEM, content=memory_text)
                )
                used_tokens += mem_tokens

        # 3. Episodic messages — trim oldest first to fit budget
        episodes = list(episodic_messages or [])
        episode_tokens = [_estimate_tokens(m.content) for m in episodes]

        total_episode_tokens = sum(episode_tokens)
        budget_remaining = self._max_tokens - used_tokens

        if total_episode_tokens > budget_remaining:
            # Drop oldest messages until we fit
            while episodes and sum(episode_tokens) > budget_remaining:
                episodes.pop(0)
                episode_tokens.pop(0)

        messages.extend(episodes)
        return messages

    @staticmethod
    def _format_memories(memories: list[MemoryRecord]) -> str:
        """Format retrieved MemoryRecords into a context block."""
        lines = ["[Retrieved memories — use naturally, do not recite]"]
        for mem in memories:
            lines.append(f"- {mem.text}")
        return "\n".join(lines)
