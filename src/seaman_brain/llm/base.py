"""LLM provider abstract protocol.

Defines the runtime-checkable Protocol that all LLM providers must implement.
Providers handle async chat completion and streaming for creature cognition.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from seaman_brain.types import ChatMessage


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM inference providers.

    All providers (Ollama, cloud APIs, etc.) must implement this interface.
    Methods are async since LLM calls are I/O-bound.
    """

    async def chat(self, messages: list[ChatMessage]) -> str:
        """Send messages and return the complete response.

        Args:
            messages: Ordered conversation history.

        Returns:
            The assistant's response text.
        """
        ...

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Send messages and stream response tokens.

        Args:
            messages: Ordered conversation history.

        Yields:
            Individual response tokens/chunks as they arrive.
        """
        ...
