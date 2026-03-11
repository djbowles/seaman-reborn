"""LLM call router for hybrid local/cloud inference.

Routes LLM calls between a local provider (fast, for reactions and autonomous
remarks) and a cloud provider (high quality, for user conversation) based on
a configurable routing mode.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from seaman_brain.llm.base import LLMProvider
from seaman_brain.types import ChatMessage

logger = logging.getLogger(__name__)

# Valid routing modes
ROUTING_MODES = ("local", "hybrid", "cloud")

# Valid call types for routing decisions
CALL_TYPES = ("conversation", "reaction", "autonomous")


class LLMRouter:
    """Routes LLM calls between local and cloud providers based on call type.

    Modes:
        "local": All calls go to local provider (default, preserves current behavior)
        "hybrid": User conversation -> cloud, reactions/autonomous -> local
        "cloud": All calls go to cloud provider

    The router implements the same chat()/stream() interface as LLMProvider,
    so it can be used as a drop-in replacement anywhere an LLMProvider is
    expected.  It also implements chat_with_tools() for tool-capable providers.
    """

    def __init__(
        self,
        local: LLMProvider,
        cloud: LLMProvider | None = None,
        mode: str = "local",
    ) -> None:
        """Initialize the router with local and optional cloud providers.

        Args:
            local: The local LLM provider (e.g. Ollama).
            cloud: Optional cloud LLM provider (e.g. Anthropic, OpenAI).
            mode: Routing mode — one of "local", "hybrid", "cloud".
        """
        self._local = local
        self._cloud = cloud
        self._mode = self._validate_mode(mode)
        self._current_call_type: str = "conversation"

    @staticmethod
    def _validate_mode(mode: str) -> str:
        """Validate and normalize a routing mode string.

        Args:
            mode: The mode string to validate.

        Returns:
            The validated mode string.

        Raises:
            ValueError: If the mode is not recognized.
        """
        normalized = mode.lower().strip()
        if normalized not in ROUTING_MODES:
            raise ValueError(
                f"Unknown routing mode: '{mode}'. "
                f"Supported modes: {', '.join(ROUTING_MODES)}"
            )
        return normalized

    @property
    def mode(self) -> str:
        """Current routing mode."""
        return self._mode

    @mode.setter
    def mode(self, value: str) -> None:
        """Set the routing mode.

        Args:
            value: One of "local", "hybrid", "cloud".
        """
        self._mode = self._validate_mode(value)
        logger.info("LLM routing mode changed to: %s", self._mode)

    @property
    def current_call_type(self) -> str:
        """The call type that will be used for the next routing decision."""
        return self._current_call_type

    @property
    def local_provider(self) -> LLMProvider:
        """The local LLM provider."""
        return self._local

    @property
    def cloud_provider(self) -> LLMProvider | None:
        """The cloud LLM provider, if configured."""
        return self._cloud

    @cloud_provider.setter
    def cloud_provider(self, value: LLMProvider | None) -> None:
        """Set or replace the cloud provider at runtime."""
        self._cloud = value
        if value is not None:
            logger.info("Cloud LLM provider updated")
        else:
            logger.info("Cloud LLM provider cleared")

    def set_call_type(self, call_type: str) -> None:
        """Set the type of the next LLM call for routing decisions.

        Args:
            call_type: One of "conversation", "reaction", "autonomous".
        """
        if call_type not in CALL_TYPES:
            logger.warning(
                "Unknown call type '%s', defaulting to 'conversation'",
                call_type,
            )
            call_type = "conversation"
        self._current_call_type = call_type

    def _pick_provider(self) -> LLMProvider:
        """Select provider based on mode and call type.

        Falls back to local if cloud is None regardless of mode.

        Returns:
            The selected LLMProvider.
        """
        if self._mode == "cloud" and self._cloud is not None:
            return self._cloud

        if self._mode == "hybrid" and self._cloud is not None:
            if self._current_call_type == "conversation":
                return self._cloud
            return self._local

        return self._local

    async def chat(
        self, messages: list[ChatMessage], **kwargs: Any,
    ) -> str:
        """Send messages to the selected provider and return the response.

        Extra keyword arguments (e.g. ``thinking=True``) are forwarded
        to the provider if it accepts them.

        Args:
            messages: Ordered conversation history.
            **kwargs: Provider-specific options forwarded to chat().

        Returns:
            The assistant's response text.
        """
        provider = self._pick_provider()
        if kwargs:
            try:
                return await provider.chat(messages, **kwargs)
            except TypeError:
                # Provider doesn't accept extra kwargs — basic call
                return await provider.chat(messages)
        return await provider.chat(messages)

    async def stream(
        self, messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        """Send messages to the selected provider and stream response tokens.

        Args:
            messages: Ordered conversation history.

        Yields:
            Individual response tokens/chunks as they arrive.
        """
        provider = self._pick_provider()
        async for token in provider.stream(messages):
            yield token

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Send messages with tools to the selected provider.

        If the provider implements ``ToolCapableLLM``, forwards the call.
        Otherwise falls back to plain ``chat()`` returning
        ``{"content": text, "tool_calls": None}``.

        Args:
            messages: Ordered conversation history.
            tools: Tool definitions in Ollama/OpenAI function-calling format.

        Returns:
            Dict with ``content`` (str | None) and ``tool_calls``
            (list | None).
        """
        from seaman_brain.llm.base import ToolCapableLLM

        provider = self._pick_provider()
        if isinstance(provider, ToolCapableLLM):
            return await provider.chat_with_tools(messages, tools)

        # Fallback: plain chat, no tools
        text = await provider.chat(messages)
        return {"content": text, "tool_calls": None}
