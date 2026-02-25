"""Ollama local LLM provider.

Implements the LLMProvider protocol using the ollama Python library
for local inference via the Ollama API server.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ollama import AsyncClient

from seaman_brain.config import LLMConfig
from seaman_brain.types import ChatMessage


class OllamaProvider:
    """LLM provider using a local Ollama server.

    Wraps the ollama AsyncClient to provide async chat and streaming
    that conforms to the LLMProvider protocol.
    """

    def __init__(self, config: LLMConfig | None = None) -> None:
        cfg = config or LLMConfig()
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.num_ctx = cfg.max_tokens
        self.base_url = cfg.base_url
        self._client = AsyncClient(host=self.base_url)

    def _format_messages(
        self, messages: list[ChatMessage]
    ) -> list[dict[str, str]]:
        """Convert ChatMessage list to ollama message format."""
        return [
            {"role": msg.role.value, "content": msg.content}
            for msg in messages
        ]

    async def chat(self, messages: list[ChatMessage]) -> str:
        """Send messages to Ollama and return the complete response.

        Args:
            messages: Ordered conversation history.

        Returns:
            The assistant's response text.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
        """
        try:
            response = await self._client.chat(
                model=self.model,
                messages=self._format_messages(messages),
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        return response.message.content or ""

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """Send messages to Ollama and stream response chunks.

        Args:
            messages: Ordered conversation history.

        Yields:
            Individual response tokens/chunks as they arrive.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
        """
        try:
            response_stream = await self._client.chat(
                model=self.model,
                messages=self._format_messages(messages),
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
                stream=True,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to connect to Ollama at {self.base_url}: {exc}"
            ) from exc
        async for chunk in response_stream:
            content = chunk.message.content
            if content:
                yield content
