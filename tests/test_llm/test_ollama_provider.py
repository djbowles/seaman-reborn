"""Tests for Ollama LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from seaman_brain.config import LLMConfig
from seaman_brain.llm.base import LLMProvider, ToolCapableLLM
from seaman_brain.llm.ollama_provider import OllamaProvider
from seaman_brain.types import ChatMessage, MessageRole

# --- Fixtures ---


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM config."""
    return LLMConfig(
        provider="ollama",
        model="test-model:7b",
        temperature=0.5,
        max_tokens=512,
        context_window=8192,
        max_response_tokens=4096,
        base_url="http://localhost:11434",
    )


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """Create sample chat messages."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are sardonic."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]


def _make_chat_response(content: str) -> MagicMock:
    """Create a mock ChatResponse with given content."""
    response = MagicMock()
    response.message.content = content
    return response


async def _make_stream_chunks(chunks: list[str]):
    """Create an async iterator of mock stream chunks."""
    for text in chunks:
        chunk = MagicMock()
        chunk.message.content = text
        yield chunk


# --- Happy path tests ---


class TestOllamaProviderHappyPath:
    """Test OllamaProvider normal operation."""

    def test_satisfies_llm_protocol(self, llm_config: LLMConfig) -> None:
        """OllamaProvider satisfies the LLMProvider protocol."""
        provider = OllamaProvider(llm_config)
        assert isinstance(provider, LLMProvider)

    def test_config_applied(self, llm_config: LLMConfig) -> None:
        """Constructor applies config values to provider attributes."""
        provider = OllamaProvider(llm_config)
        assert provider.model == "test-model:7b"
        assert provider.temperature == 0.5
        assert provider.num_ctx == 8192
        assert provider.num_predict == 4096
        assert provider.base_url == "http://localhost:11434"

    def test_default_config(self) -> None:
        """Constructor uses default LLMConfig when none provided."""
        provider = OllamaProvider()
        assert provider.model == "qwen3:8b"
        assert provider.temperature == 0.8

    async def test_chat_returns_response(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns the assistant's response text."""
        provider = OllamaProvider(llm_config)
        mock_response = _make_chat_response("I am deeply unimpressed.")
        provider._client.chat = AsyncMock(return_value=mock_response)

        result = await provider.chat(sample_messages)

        assert result == "I am deeply unimpressed."
        provider._client.chat.assert_called_once_with(
            model="test-model:7b",
            messages=[
                {"role": "system", "content": "You are sardonic."},
                {"role": "user", "content": "Hello"},
            ],
            options={"temperature": 0.5, "num_ctx": 8192, "num_predict": 4096},
            think=False,
        )

    async def test_stream_yields_chunks(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() yields response chunks as they arrive."""
        provider = OllamaProvider(llm_config)
        provider._client.chat = AsyncMock(
            return_value=_make_stream_chunks(["Hello", " there", " human"])
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " there", " human"]
        call_kwargs = provider._client.chat.call_args.kwargs
        assert call_kwargs["think"] is False

    async def test_format_messages(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """_format_messages converts ChatMessage to ollama dict format."""
        provider = OllamaProvider(llm_config)
        formatted = provider._format_messages(sample_messages)
        assert formatted == [
            {"role": "system", "content": "You are sardonic."},
            {"role": "user", "content": "Hello"},
        ]


# --- Edge case tests ---


class TestOllamaProviderEdgeCases:
    """Edge cases for OllamaProvider."""

    async def test_chat_empty_messages(self, llm_config: LLMConfig) -> None:
        """chat() works with an empty message list."""
        provider = OllamaProvider(llm_config)
        mock_response = _make_chat_response("No context given.")
        provider._client.chat = AsyncMock(return_value=mock_response)

        result = await provider.chat([])
        assert result == "No context given."

    async def test_chat_none_content_returns_empty_string(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns empty string when response content is None."""
        provider = OllamaProvider(llm_config)
        mock_response = _make_chat_response(None)
        mock_response.message.content = None
        provider._client.chat = AsyncMock(return_value=mock_response)

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_stream_skips_empty_chunks(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() skips chunks with empty/None content."""
        provider = OllamaProvider(llm_config)

        async def mixed_chunks():
            for text in ["Hello", "", None, " world"]:
                chunk = MagicMock()
                chunk.message.content = text
                yield chunk

        provider._client.chat = AsyncMock(return_value=mixed_chunks())

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    async def test_format_messages_empty_list(self, llm_config: LLMConfig) -> None:
        """_format_messages handles empty list."""
        provider = OllamaProvider(llm_config)
        assert provider._format_messages([]) == []


# --- Error handling tests ---


class TestOllamaProviderErrors:
    """Error handling for OllamaProvider."""

    async def test_chat_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() wraps connection errors with clear message."""
        provider = OllamaProvider(llm_config)
        provider._client.chat = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
            await provider.chat(sample_messages)

    async def test_stream_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() wraps connection errors with clear message."""
        provider = OllamaProvider(llm_config)
        provider._client.chat = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to connect to Ollama"):
            async for _ in provider.stream(sample_messages):
                pass  # pragma: no cover

    async def test_chat_error_includes_base_url(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """Connection error message includes the base URL for debugging."""
        provider = OllamaProvider(llm_config)
        provider._client.chat = AsyncMock(
            side_effect=TimeoutError("timed out")
        )

        with pytest.raises(ConnectionError, match="http://localhost:11434"):
            await provider.chat(sample_messages)

    async def test_chat_preserves_original_exception(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """ConnectionError chains the original exception."""
        provider = OllamaProvider(llm_config)
        original = OSError("network down")
        provider._client.chat = AsyncMock(side_effect=original)

        with pytest.raises(ConnectionError) as exc_info:
            await provider.chat(sample_messages)

        assert exc_info.value.__cause__ is original


# --- Tool calling tests ---


class TestOllamaProviderToolCalling:
    """Tests for OllamaProvider.chat_with_tools."""

    def test_satisfies_tool_capable_protocol(self, llm_config: LLMConfig) -> None:
        """OllamaProvider satisfies the ToolCapableLLM protocol."""
        provider = OllamaProvider(llm_config)
        assert isinstance(provider, ToolCapableLLM)

    async def test_chat_with_tools_returns_content(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat_with_tools returns content when no tool calls."""
        provider = OllamaProvider(llm_config)
        mock_response = _make_chat_response("I see you clearly.")
        mock_response.message.tool_calls = None
        provider._client.chat = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "look_at_user"}}]
        result = await provider.chat_with_tools(sample_messages, tools)

        assert result["content"] == "I see you clearly."
        assert result["tool_calls"] is None
        # Verify tools and think=False were passed to the client
        call_kwargs = provider._client.chat.call_args.kwargs
        assert call_kwargs["tools"] == tools
        assert call_kwargs["think"] is False

    async def test_chat_with_tools_returns_tool_calls(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat_with_tools extracts tool_calls from response."""
        provider = OllamaProvider(llm_config)
        mock_response = _make_chat_response("")
        mock_response.message.content = None

        # Mock a tool call
        tc = MagicMock()
        tc.function.name = "look_at_user"
        tc.function.arguments = {}
        mock_response.message.tool_calls = [tc]

        provider._client.chat = AsyncMock(return_value=mock_response)

        tools = [{"type": "function", "function": {"name": "look_at_user"}}]
        result = await provider.chat_with_tools(sample_messages, tools)

        assert result["content"] is None
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["function"]["name"] == "look_at_user"

    async def test_chat_with_tools_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat_with_tools wraps connection errors."""
        provider = OllamaProvider(llm_config)
        provider._client.chat = AsyncMock(side_effect=Exception("Connection refused"))

        with pytest.raises(ConnectionError, match="Failed to connect"):
            await provider.chat_with_tools(sample_messages, [])
