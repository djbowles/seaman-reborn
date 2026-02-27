"""Tests for Anthropic LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.config import LLMConfig
from seaman_brain.llm.anthropic_provider import AnthropicProvider
from seaman_brain.llm.base import LLMProvider
from seaman_brain.types import ChatMessage, MessageRole

# --- Fixtures ---


@pytest.fixture
def llm_config() -> LLMConfig:
    """Create a test LLM config for Anthropic."""
    return LLMConfig(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        temperature=0.7,
        max_tokens=1024,
    )


@pytest.fixture
def sample_messages() -> list[ChatMessage]:
    """Create sample chat messages with system, user, and assistant roles."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are sardonic."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]


@pytest.fixture
def user_only_messages() -> list[ChatMessage]:
    """Create sample messages with no system message."""
    return [
        ChatMessage(role=MessageRole.USER, content="Hello"),
        ChatMessage(role=MessageRole.ASSISTANT, content="What do you want?"),
        ChatMessage(role=MessageRole.USER, content="Tell me a joke."),
    ]


def _make_provider(config: LLMConfig) -> AnthropicProvider:
    """Create an AnthropicProvider with a fake API key."""
    return AnthropicProvider(config, api_key="sk-ant-test-key-12345")


def _make_chat_response(text: str | None) -> MagicMock:
    """Create a mock Anthropic Message response."""
    content_block = MagicMock()
    content_block.text = text

    response = MagicMock()
    response.content = [content_block] if text is not None else []
    return response


def _make_empty_response() -> MagicMock:
    """Create a mock Anthropic Message response with no content."""
    response = MagicMock()
    response.content = []
    return response


async def _make_stream_events(chunks: list[str | None]):
    """Create an async iterator of mock stream events."""
    for text in chunks:
        event = MagicMock()
        if text is not None and text != "":
            event.type = "content_block_delta"
            delta = MagicMock()
            delta.text = text
            event.delta = delta
        elif text == "":
            event.type = "content_block_delta"
            delta = MagicMock()
            delta.text = ""
            event.delta = delta
        else:
            # None -> simulate a non-delta event (e.g., message_start)
            event.type = "message_start"
        yield event


# --- Happy path tests ---


class TestAnthropicProviderHappyPath:
    """Test AnthropicProvider normal operation."""

    def test_satisfies_llm_protocol(self, llm_config: LLMConfig) -> None:
        """AnthropicProvider satisfies the LLMProvider protocol."""
        provider = _make_provider(llm_config)
        assert isinstance(provider, LLMProvider)

    def test_config_applied(self, llm_config: LLMConfig) -> None:
        """Constructor applies config values to provider attributes."""
        provider = _make_provider(llm_config)
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1024

    def test_api_key_from_constructor(self, llm_config: LLMConfig) -> None:
        """Constructor accepts api_key directly."""
        provider = AnthropicProvider(llm_config, api_key="sk-ant-direct-key")
        assert provider._client.api_key == "sk-ant-direct-key"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"})
    def test_api_key_from_env(self, llm_config: LLMConfig) -> None:
        """Constructor reads API key from environment variable."""
        provider = AnthropicProvider(llm_config)
        assert provider._client.api_key == "sk-ant-env-key"

    async def test_chat_returns_response(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns the assistant's response text."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("I am deeply unimpressed.")
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(sample_messages)

        assert result == "I am deeply unimpressed."
        provider._client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.7,
            max_tokens=1024,
            system="You are sardonic.",
        )

    async def test_chat_without_system_message(
        self, llm_config: LLMConfig, user_only_messages: list[ChatMessage]
    ) -> None:
        """chat() works without system messages (no system kwarg sent)."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("Fine, whatever.")
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(user_only_messages)

        assert result == "Fine, whatever."
        provider._client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-20250514",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "What do you want?"},
                {"role": "user", "content": "Tell me a joke."},
            ],
            temperature=0.7,
            max_tokens=1024,
        )

    async def test_stream_yields_chunks(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() yields response chunks as they arrive."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            return_value=_make_stream_events(["Hello", " there", " human"])
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " there", " human"]


# --- Edge case tests ---


class TestAnthropicProviderEdgeCases:
    """Edge cases for AnthropicProvider."""

    async def test_chat_empty_messages(self, llm_config: LLMConfig) -> None:
        """chat() works with an empty message list."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("No context given.")
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat([])
        assert result == "No context given."

    async def test_chat_empty_content_returns_empty_string(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns empty string when response has no content blocks."""
        provider = _make_provider(llm_config)
        mock_response = _make_empty_response()
        provider._client.messages.create = AsyncMock(return_value=mock_response)

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_chat_none_text_returns_empty_string(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() returns empty string when content block text is empty."""
        provider = _make_provider(llm_config)
        content_block = MagicMock()
        content_block.text = ""
        response = MagicMock()
        response.content = [content_block]
        provider._client.messages.create = AsyncMock(return_value=response)

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_stream_skips_non_delta_events(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() skips events that are not content_block_delta."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            return_value=_make_stream_events(["Hello", None, " world"])
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    async def test_stream_skips_empty_text(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() skips delta events with empty text."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            return_value=_make_stream_events(["Hello", "", " world"])
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    def test_format_messages_separates_system(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """_format_messages extracts system messages from conversation."""
        provider = _make_provider(llm_config)
        system_prompt, conversation = provider._format_messages(sample_messages)
        assert system_prompt == "You are sardonic."
        assert conversation == [{"role": "user", "content": "Hello"}]

    def test_format_messages_multiple_system(self, llm_config: LLMConfig) -> None:
        """_format_messages concatenates multiple system messages."""
        provider = _make_provider(llm_config)
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are sardonic."),
            ChatMessage(role=MessageRole.SYSTEM, content="You hate small talk."),
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ]
        system_prompt, conversation = provider._format_messages(messages)
        assert system_prompt == "You are sardonic.\nYou hate small talk."
        assert conversation == [{"role": "user", "content": "Hello"}]

    def test_format_messages_no_system(
        self, llm_config: LLMConfig, user_only_messages: list[ChatMessage]
    ) -> None:
        """_format_messages returns None system when no system messages present."""
        provider = _make_provider(llm_config)
        system_prompt, conversation = provider._format_messages(user_only_messages)
        assert system_prompt is None
        assert len(conversation) == 3

    def test_format_messages_empty_list(self, llm_config: LLMConfig) -> None:
        """_format_messages handles empty list."""
        provider = _make_provider(llm_config)
        system_prompt, conversation = provider._format_messages([])
        assert system_prompt is None
        assert conversation == []


# --- Error handling tests ---


class TestAnthropicProviderErrors:
    """Error handling for AnthropicProvider."""

    def test_missing_api_key_raises_value_error(self, llm_config: LLMConfig) -> None:
        """Constructor raises ValueError when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(ValueError, match="Anthropic API key required"):
                AnthropicProvider(llm_config)

    async def test_chat_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """chat() wraps API errors with clear message."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to call Anthropic API"):
            await provider.chat(sample_messages)

    async def test_stream_connection_error(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """stream() wraps API errors with clear message."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to call Anthropic API"):
            async for _ in provider.stream(sample_messages):
                pass  # pragma: no cover

    async def test_chat_preserves_original_exception(
        self, llm_config: LLMConfig, sample_messages: list[ChatMessage]
    ) -> None:
        """ConnectionError chains the original exception."""
        provider = _make_provider(llm_config)
        original = OSError("network down")
        provider._client.messages.create = AsyncMock(side_effect=original)

        with pytest.raises(ConnectionError) as exc_info:
            await provider.chat(sample_messages)

        assert exc_info.value.__cause__ is original

    def test_constructor_key_precedence(self, llm_config: LLMConfig) -> None:
        """Constructor api_key takes precedence over environment variable."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            provider = AnthropicProvider(llm_config, api_key="sk-ant-direct-key")
            assert provider._client.api_key == "sk-ant-direct-key"
