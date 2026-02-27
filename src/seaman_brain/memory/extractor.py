"""Background fact extraction from conversations.

Uses the LLM to identify and extract memorable facts from conversation
history, then stores them as embedded vectors in semantic memory.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np

from seaman_brain.config import MemoryConfig
from seaman_brain.llm.base import LLMProvider
from seaman_brain.memory.embeddings import EmbeddingProvider
from seaman_brain.memory.semantic import SemanticMemory
from seaman_brain.types import ChatMessage, MemoryRecord, MessageRole

EXTRACTION_PROMPT = """\
You are a memory extraction engine. Analyze the conversation below and extract \
key facts worth remembering long-term. Focus on:

- User preferences (likes, dislikes, habits)
- Personal information (name, occupation, relationships)
- Topics discussed and opinions expressed
- Observations about behavior or personality

Return ONLY a newline-separated list of short factual statements. \
No numbering, no bullets, no commentary. If there are no facts worth \
remembering, return exactly: NONE

Conversation:
{conversation}"""


class MemoryExtractor:
    """Extracts memorable facts from conversations using an LLM.

    Periodically analyzes recent conversation messages to identify facts
    worth storing in long-term semantic memory.
    """

    def __init__(
        self,
        llm: LLMProvider,
        embeddings: EmbeddingProvider,
        semantic: SemanticMemory,
        config: MemoryConfig | None = None,
    ) -> None:
        cfg = config or MemoryConfig()
        self._llm = llm
        self._embeddings = embeddings
        self._semantic = semantic
        self._extraction_interval = cfg.extraction_interval
        self._message_count = 0

    @property
    def extraction_interval(self) -> int:
        """Number of messages between extraction runs."""
        return self._extraction_interval

    @property
    def message_count(self) -> int:
        """Messages received since last extraction."""
        return self._message_count

    def should_extract(self) -> bool:
        """Check if enough messages have accumulated for extraction."""
        return self._message_count >= self._extraction_interval

    def increment_counter(self) -> None:
        """Record that a message was processed."""
        self._message_count += 1

    def reset_counter(self) -> None:
        """Reset the message counter after an extraction."""
        self._message_count = 0

    async def extract(self, messages: list[ChatMessage]) -> list[str]:
        """Extract key facts from conversation messages using the LLM.

        Args:
            messages: Recent conversation messages to analyze.

        Returns:
            List of extracted fact strings. Empty list if no facts found
            or messages are empty.
        """
        if not messages:
            return []

        # Filter to only user/assistant messages (skip system)
        conversation_messages = [
            m for m in messages
            if m.role in (MessageRole.USER, MessageRole.ASSISTANT)
        ]
        if not conversation_messages:
            return []

        # Format conversation for the prompt
        conversation_text = "\n".join(
            f"{m.role.value}: {m.content}" for m in conversation_messages
        )
        prompt = EXTRACTION_PROMPT.format(conversation=conversation_text)

        extraction_request = ChatMessage(
            role=MessageRole.USER,
            content=prompt,
        )

        try:
            response = await self._llm.chat([extraction_request])
        except Exception:
            return []

        return self._parse_facts(response)

    async def extract_and_store(
        self,
        messages: list[ChatMessage],
    ) -> list[str]:
        """Extract facts from conversation and store them in semantic memory.

        Args:
            messages: Recent conversation messages to analyze.

        Returns:
            List of fact strings that were successfully stored.
        """
        facts = await self.extract(messages)
        if not facts:
            return []

        stored: list[str] = []
        for fact in facts:
            try:
                embedding = await self._embeddings.embed(fact)
                if not embedding:
                    continue

                record = MemoryRecord(
                    text=fact,
                    embedding=np.array(embedding, dtype=np.float32),
                    timestamp=datetime.now(UTC),
                    importance=0.5,
                    source="extraction",
                )
                await self._semantic.add(record)
                stored.append(fact)
            except Exception:
                continue

        self.reset_counter()
        return stored

    @staticmethod
    def _parse_facts(response: str) -> list[str]:
        """Parse LLM response into individual fact strings.

        Args:
            response: Raw LLM response text.

        Returns:
            List of non-empty fact strings, or empty list if NONE.
        """
        if not response or not response.strip():
            return []

        text = response.strip()
        if text.upper() == "NONE":
            return []

        facts = [line.strip() for line in text.splitlines()]
        # Filter empty lines and the NONE sentinel
        return [f for f in facts if f and f.upper() != "NONE"]
