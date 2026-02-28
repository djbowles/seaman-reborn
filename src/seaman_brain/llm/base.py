"""LLM provider abstract protocol.

Defines the runtime-checkable Protocol that all LLM providers must implement.
Providers handle async chat completion and streaming for creature cognition.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

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


@runtime_checkable
class ToolCapableLLM(Protocol):
    """Protocol for LLM providers that support tool/function calling.

    Extends the base LLM capability with tool-use support.
    Kept separate to avoid breaking providers that don't support tools.
    """

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send messages with tool definitions and return structured response.

        Args:
            messages: Ordered conversation history.
            tools: List of tool definitions in provider-specific format.

        Returns:
            Dict with "content" (str | None) and "tool_calls" (list | None).
        """
        ...
