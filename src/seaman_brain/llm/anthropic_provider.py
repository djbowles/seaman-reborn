"""Anthropic cloud LLM provider.

Implements the LLMProvider protocol using the anthropic Python library
for cloud inference via the Anthropic Messages API.

Supports:
- Tool/function calling (ToolCapableLLM protocol)
- Prompt caching via cache_control on system prompt blocks
- Extended thinking for deeper reasoning on complex queries
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from seaman_brain.config import LLMConfig
from seaman_brain.types import ChatMessage, MessageRole

# Marker string that separates cacheable prefix from dynamic suffix
_CACHE_MARKER = "\n\n---CACHE_BREAK---\n\n"


class AnthropicProvider:
    """LLM provider using the Anthropic Messages API.

    Wraps the anthropic AsyncAnthropic client to provide async chat and streaming
    that conforms to the LLMProvider protocol.

    The Anthropic API separates system messages from conversation messages,
    so _format_messages extracts system content and returns the rest as
    user/assistant pairs.
    """

    def __init__(
        self, config: LLMConfig | None = None, api_key: str | None = None,
    ) -> None:
        cfg = config or LLMConfig(
            provider="anthropic", model="claude-sonnet-4-20250514",
        )
        self.model = cfg.model
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self._enable_caching = getattr(cfg, "enable_prompt_caching", True)

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY "
                "environment variable or pass api_key to the constructor."
            )

        self._client = AsyncAnthropic(api_key=resolved_key)

    def _format_messages(
        self, messages: list[ChatMessage],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Convert ChatMessage list to Anthropic's system + messages format.

        Anthropic requires system messages to be passed separately via the
        ``system`` parameter, not in the messages array. All system messages
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
                conversation.append(
                    {"role": msg.role.value, "content": msg.content},
                )

        system_prompt = "\n".join(system_parts) if system_parts else None
        return system_prompt, conversation

    def _format_system(
        self, system_prompt: str | None,
    ) -> str | list[dict[str, Any]] | None:
        """Format system prompt, with optional cache control.

        When prompt caching is enabled and the system prompt contains the
        ``---CACHE_BREAK---`` marker, the prompt is split into two blocks:
        a cacheable prefix (with ``cache_control``) and a dynamic suffix.

        Returns:
            A plain string, a list of Anthropic content blocks, or None.
        """
        if system_prompt is None:
            return None

        if _CACHE_MARKER in system_prompt and self._enable_caching:
            prefix, suffix = system_prompt.split(_CACHE_MARKER, 1)
            blocks: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": prefix,
                    "cache_control": {"type": "ephemeral"},
                },
            ]
            if suffix:
                blocks.append({"type": "text", "text": suffix})
            return blocks

        return system_prompt

    # ------------------------------------------------------------------
    # chat() — with optional extended thinking
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[ChatMessage],
        *,
        thinking: bool = False,
        thinking_budget: int = 4096,
    ) -> str:
        """Send messages to Anthropic and return the complete response.

        Args:
            messages: Ordered conversation history.
            thinking: Enable extended thinking (deep reasoning mode).
            thinking_budget: Max tokens for the internal thinking step.

        Returns:
            The assistant's response text (thinking blocks are filtered).

        Raises:
            ConnectionError: If the Anthropic API is unreachable.
        """
        system_prompt, conversation = self._format_messages(messages)
        formatted_system = self._format_system(system_prompt)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "max_tokens": self.max_tokens,
        }

        if thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget,
            }
            kwargs["temperature"] = 1.0  # Anthropic requires temp=1 for thinking
        else:
            kwargs["temperature"] = self.temperature

        if formatted_system is not None:
            kwargs["system"] = formatted_system

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call Anthropic API: {exc}"
            ) from exc

        # Filter out thinking blocks, return only text content
        for block in response.content:
            if getattr(block, "type", None) == "text" and block.text:
                return block.text
        return ""

    # ------------------------------------------------------------------
    # stream()
    # ------------------------------------------------------------------

    async def stream(
        self, messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        """Send messages to Anthropic and stream response chunks.

        Args:
            messages: Ordered conversation history.

        Yields:
            Individual response tokens/chunks as they arrive.

        Raises:
            ConnectionError: If the Anthropic API is unreachable.
        """
        system_prompt, conversation = self._format_messages(messages)
        formatted_system = self._format_system(system_prompt)

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if formatted_system is not None:
            kwargs["system"] = formatted_system

        try:
            response_stream = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call Anthropic API: {exc}"
            ) from exc

        async for event in response_stream:
            if (
                event.type == "content_block_delta"
                and hasattr(event.delta, "text")
            ):
                text = event.delta.text
                if text:
                    yield text

    # ------------------------------------------------------------------
    # chat_with_tools() — ToolCapableLLM protocol
    # ------------------------------------------------------------------

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send messages with tool definitions and return structured response.

        Translates the internal tool format (Ollama-style) to Anthropic's
        tool format, then parses tool_use blocks from the response.

        Args:
            messages: Ordered conversation history.
            tools: Tool definitions (Ollama/OpenAI function-calling format).

        Returns:
            Dict with ``content`` (str | None) and ``tool_calls``
            (list | None).

        Raises:
            ConnectionError: If the Anthropic API is unreachable.
        """
        system_prompt, conversation = self._format_messages(messages)
        formatted_system = self._format_system(system_prompt)

        # Translate tool definitions: Ollama format -> Anthropic format
        anthropic_tools: list[dict[str, Any]] = []
        for tool in tools:
            func = tool.get("function", tool)
            anthropic_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "input_schema": func.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            })

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": conversation,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "tools": anthropic_tools,
        }
        if formatted_system is not None:
            kwargs["system"] = formatted_system

        try:
            response = await self._client.messages.create(**kwargs)
        except Exception as exc:
            raise ConnectionError(
                f"Failed to call Anthropic API: {exc}"
            ) from exc

        # Parse response for text content and tool_use blocks
        content_text: str | None = None
        tool_calls: list[dict[str, Any]] | None = None

        for block in response.content:
            if block.type == "text":
                content_text = block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "function": {
                        "name": block.name,
                        "arguments": block.input,
                    },
                    "id": block.id,
                })

        return {"content": content_text, "tool_calls": tool_calls}
