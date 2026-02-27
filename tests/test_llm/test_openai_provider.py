"""Tests for OpenAI LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.config import LLMConfig
from seaman_brain.llm.base import LLMProvider
from seaman_brain.llm.openai_provider import OpenAIProvider
from seaman_brain.types import ChatMessage, MessageRole

# --- Fixtures ---


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM config for OpenAI."""
    return LLMConfig(
        provider="openai",
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """Create sample chat messages."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are sardonic."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]


def _make_provider(config: LLMConfig) -> OpenAIProvider:
    """Create an OpenAIProvider with a fake API key."""
    return OpenAIProvider(config, api_key="sk-test-key-12345")


def _make_chat_response(content: str | None) -> MagicMock:
    """Create a mock ChatCompletion response."""
    message = MagicMock()
    message.content = content

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    return response


def _make_empty_response() -> MagicMock:
    """Create a mock ChatCompletion response with no choices."""
    response = MagicMock()
    response.choices = []
    return response


async def _make_stream_chunks(chunks: list[str | None]):
    """Create an async iterator of mock stream chunks."""
    for text in chunks:
        delta = MagicMock()
        delta.content = text

        choice = MagicMock()
        choice.delta = delta

        chunk = MagicMock()
        chunk.choices = [choice]
        yield chunk


# --- Happy path tests ---


class TestOpenAIProviderHappyPath:
    """Test OpenAIProvider normal operation."""

    def test_satisfies_llm_protocol(self, llm_config: LLMConfig) -> None:
        """OpenAIProvider satisfies the LLMProvider protocol."""
        provider = _make_provider(llm_config)
        assert isinstance(provider, LLMProvider)

    def test_config_applied(self, llm_config: LLMConfig) -> None:
        """Constructor applies config values to provider attributes."""
        provider = _make_provider(llm_config)
        assert provider.model == "gpt-4o"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1024

    def test_api_key_from_constructor(self, llm_config: LLMConfig) -> None:
        """Constructor accepts api_key directly."""
        provider = OpenAIProvider(llm_config, api_key="sk-direct-key")
        assert provider._client.api_key == "sk-direct-key"

    @patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"})
    def test_api_key_from_env(self, llm_config: LLMConfig) -> None:
        """Constructor reads API key from environment variable."""
        provider = OpenAIProvider(llm_config)
        assert provider._client.api_key == "sk-env-key"

    async def test_chat_returns_response(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns the assistant's response text."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("I am deeply unimpressed.")
        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await provider.chat(sample_messages)

        assert result == "I am deeply unimpressed."
        provider._client.chat.completions.create.assert_called_once_with(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are sardonic."},
                {"role": "user", "content": "Hello"},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

    async def test_stream_yields_chunks(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() yields response chunks as they arrive."""
        provider = _make_provider(llm_config)
        provider._client.chat.completions.create = AsyncMock(
            return_value=_make_stream_chunks(["Hello", " there", " human"])
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " there", " human"]


# --- Edge case tests ---


class TestOpenAIProviderEdgeCases:
    """Edge cases for OpenAIProvider."""

    async def test_chat_empty_messages(self, llm_config: LLMConfig) -> None:
        """chat() works with an empty message list."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("No context given.")
        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await provider.chat([])
        assert result == "No context given."

    async def test_chat_none_content_returns_empty_string(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns empty string when response content is None."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response(None)
        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_chat_empty_choices_returns_empty_string(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns empty string when response has no choices."""
        provider = _make_provider(llm_config)
        mock_response = _make_empty_response()
        provider._client.chat.completions.create = AsyncMock(
            return_value=mock_response
        )

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_stream_skips_empty_chunks(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() skips chunks with empty/None content."""
        provider = _make_provider(llm_config)
        provider._client.chat.completions.create = AsyncMock(
            return_value=_make_stream_chunks(["Hello", None, "", " world"])
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    async def test_stream_skips_empty_choices(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() skips chunks with no choices."""
        provider = _make_provider(llm_config)

        async def mixed_chunks():
            # Normal chunk
            delta = MagicMock()
            delta.content = "Hello"
            choice = MagicMock()
            choice.delta = delta
            chunk = MagicMock()
            chunk.choices = [choice]
            yield chunk
            # Empty choices chunk
            chunk2 = MagicMock()
            chunk2.choices = []
            yield chunk2
            # Normal chunk
            delta3 = MagicMock()
            delta3.content = " world"
            choice3 = MagicMock()
            choice3.delta = delta3
            chunk3 = MagicMock()
            chunk3.choices = [choice3]
            yield chunk3

        provider._client.chat.completions.create = AsyncMock(
            return_value=mixed_chunks()
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    def test_format_messages(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """_format_messages converts ChatMessage to OpenAI dict format."""
        provider = _make_provider(llm_config)
        formatted = provider._format_messages(sample_messages)
        assert formatted == [
            {"role": "system", "content": "You are sardonic."},
            {"role": "user", "content": "Hello"},
        ]

    def test_format_messages_empty_list(self, llm_config: LLMConfig) -> None:
        """_format_messages handles empty list."""
        provider = _make_provider(llm_config)
        assert provider._format_messages([]) == []


# --- Error handling tests ---


class TestOpenAIProviderErrors:
    """Error handling for OpenAIProvider."""

    def test_missing_api_key_raises_value_error(self, llm_config: LLMConfig) -> None:
        """Constructor raises ValueError when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            # Ensure OPENAI_API_KEY is not in env
            import os
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OpenAI API key required"):
                OpenAIProvider(llm_config)

    async def test_chat_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() wraps API errors with clear message."""
        provider = _make_provider(llm_config)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to call OpenAI API"):
            await provider.chat(sample_messages)

    async def test_stream_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() wraps API errors with clear message."""
        provider = _make_provider(llm_config)
        provider._client.chat.completions.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to call OpenAI API"):
            async for _ in provider.stream(sample_messages):
                pass  # pragma: no cover

    async def test_chat_preserves_original_exception(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """ConnectionError chains the original exception."""
        provider = _make_provider(llm_config)
        original = OSError("network down")
        provider._client.chat.completions.create = AsyncMock(side_effect=original)

        with pytest.raises(ConnectionError) as exc_info:
            await provider.chat(sample_messages)

        assert exc_info.value.__cause__ is original

    def test_constructor_key_precedence(self, llm_config: LLMConfig) -> None:
        """Constructor api_key takes precedence over environment variable."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env-key"}):
            provider = OpenAIProvider(llm_config, api_key="sk-direct-key")
            assert provider._client.api_key == "sk-direct-key"
