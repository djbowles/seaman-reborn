"""Anthropic cloud LLM provider.

Implements the LLMProvider protocol using the anthropic Python library
for cloud inference via the Anthropic Messages API.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from seaman_brain.config import LLMConfig
from seaman_brain.types import ChatMessage, MessageRole


class AnthropicProvider:
    """LLM provider using the Anthropic Messages API.

    Wraps the anthropic AsyncAnthropic client to provide async chat and streaming
    that conforms to the LLMProvider protocol.

    The Anthropic API separates system messages from conversation messages,
    so _format_messages extracts system content and returns the rest as
    user/assistant pairs.
    """

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None) -> None:
        cfg = config or LLMConfig(provider="anthropic", model="claude-sonnet-4-20250514")
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

        self._client = AsyncAnthropic(api_key=resolved_key)

    def _format_messages(
        self, messages: list[ChatMessage]
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert ChatMessage list to Anthropic's system + messages format.

        Anthropic requires system messages to be passed separately via the
        `system` parameter, not in the messages array. All system messages
        are concatenated into a single system string.

        Returns:
            A tuple of (system_prompt, messages) where system_prompt may be
            None if no system messages are present.
        """
        system_parts: list[str] = []
        conversation: list[dict[str, str]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content)
            else:
                conversation.append({"role": msg.role.value, "content": msg.content})

        system_prompt = "\n".join(system_parts) if system_parts else None
        return system_prompt, conversation

    async def chat(self, messages: list[ChatMessage]) -> str:
        """Send messages to Anthropic and return the complete response.

        Args:
            messages: Ordered conversation history.

        Returns:
            The assistant's response text.

        Raises:
            ConnectionError: If the Anthropic API is unreachable or returns an error.
        """
        system_prompt, conversation = self._format_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call Anthropic API: {exc}"
            ) from exc

        if response.content and response.content[0].text:
            return response.content[0].text
        return ""

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Send messages to Anthropic and stream response chunks.

        Args:
            messages: Ordered conversation history.

        Yields:
            Individual response tokens/chunks as they arrive.

        Raises:
            ConnectionError: If the Anthropic API is unreachable or returns an error.
        """
        system_prompt, conversation = self._format_messages(messages)

        kwargs: dict = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        try:
            response_stream = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call Anthropic API: {exc}"
            ) from exc

        async for event in response_stream:
            if event.type == "content_block_delta" and hasattr(event.delta, "text"):
                text = event.delta.text
                if text:
                    yield text
