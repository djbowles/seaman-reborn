"""OpenAI cloud LLM provider.

Implements the LLMProvider protocol using the openai Python library
for cloud inference via the OpenAI API.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from seaman_brain.config import LLMConfig
from seaman_brain.types import ChatMessage


class OpenAIProvider:
    """LLM provider using the OpenAI API.

    Wraps the openai AsyncOpenAI client to provide async chat and streaming
    that conforms to the LLMProvider protocol.
    """

    def __init__(self, config: LLMConfig | None = None, api_key: str | None = None) -> None:
        cfg = config or LLMConfig(provider="openai", model="gpt-4o")
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens

        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key to the constructor."
            )

        self._client = AsyncOpenAI(api_key=resolved_key)

    def _format_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Convert ChatMessage list to OpenAI message format."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

    async def chat(self, messages: list[ChatMessage]) -> str:
        """Send messages to OpenAI and return the complete response.

        Args:
            messages: Ordered conversation history.

        Returns:
            The assistant's response text.

        Raises:
            ConnectionError: If the OpenAI API is unreachable or returns an error.
        """
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=self._format_messages(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call OpenAI API: {exc}"
            ) from exc
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            return choice.message.content
        return ""

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Send messages to OpenAI and stream response chunks.

        Args:
            messages: Ordered conversation history.

        Yields:
            Individual response tokens/chunks as they arrive.

        Raises:
            ConnectionError: If the OpenAI API is unreachable or returns an error.
        """
        try:
            response_stream = await self._client.chat.completions.create(
                model=self.model,
                messages=self._format_messages(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call OpenAI API: {exc}"
            ) from exc
        async for chunk in response_stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
