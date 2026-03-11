"""Tests for Anthropic LLM provider."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.config import LLMConfig
from seaman_brain.llm.anthropic_provider import AnthropicProvider
from seaman_brain.llm.base import LLMProvider, ToolCapableLLM
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


def _make_provider(
    config: LLMConfig, enable_caching: bool = True,
) -> AnthropicProvider:
    """Create an AnthropicProvider with a fake API key."""
    config = config.model_copy(
        update={"enable_prompt_caching": enable_caching},
    )
    return AnthropicProvider(config, api_key="sk-ant-test-key-12345")


def _make_text_block(text: str) -> MagicMock:
    """Create a mock text content block."""
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _make_thinking_block(text: str) -> MagicMock:
    """Create a mock thinking content block."""
    block = MagicMock()
    block.type = "thinking"
    block.thinking = text
    return block


def _make_tool_use_block(
    name: str, input_data: dict, block_id: str = "toolu_123",
) -> MagicMock:
    """Create a mock tool_use content block."""
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = input_data
    block.id = block_id
    return block


def _make_chat_response(text: str | None) -> MagicMock:
    """Create a mock Anthropic Message response with a text block."""
    response = MagicMock()
    if text is not None:
        response.content = [_make_text_block(text)]
    else:
        response.content = []
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

    def test_satisfies_tool_capable_protocol(
        self, llm_config: LLMConfig,
    ) -> None:
        """AnthropicProvider satisfies the ToolCapableLLM protocol."""
        provider = _make_provider(llm_config)
        assert isinstance(provider, ToolCapableLLM)

    def test_config_applied(self, llm_config: LLMConfig) -> None:
        """Constructor applies config values to provider attributes."""
        provider = _make_provider(llm_config)
        assert provider.model == "claude-sonnet-4-20250514"
        assert provider.temperature == 0.7
        assert provider.max_tokens == 1024

    def test_enable_caching_from_config(self, llm_config: LLMConfig) -> None:
        """Constructor reads enable_prompt_caching from config."""
        provider = _make_provider(llm_config, enable_caching=True)
        assert provider._enable_caching is True
        provider2 = _make_provider(llm_config, enable_caching=False)
        assert provider2._enable_caching is False

    def test_api_key_from_constructor(self, llm_config: LLMConfig) -> None:
        """Constructor accepts api_key directly."""
        provider = AnthropicProvider(
            llm_config, api_key="sk-ant-direct-key",
        )
        assert provider._client.api_key == "sk-ant-direct-key"

    @patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"})
    def test_api_key_from_env(self, llm_config: LLMConfig) -> None:
        """Constructor reads API key from environment variable."""
        provider = AnthropicProvider(llm_config)
        assert provider._client.api_key == "sk-ant-env-key"

    async def test_chat_returns_response(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """chat() returns the assistant's response text."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("I am deeply unimpressed.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

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
        self,
        llm_config: LLMConfig,
        user_only_messages: list[ChatMessage],
    ) -> None:
        """chat() works without system messages (no system kwarg sent)."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("Fine, whatever.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

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
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """stream() yields response chunks as they arrive."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            return_value=_make_stream_events(
                ["Hello", " there", " human"],
            ),
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " there", " human"]


# --- Edge case tests ---


class TestAnthropicProviderEdgeCases:
    """Edge cases for AnthropicProvider."""

    async def test_chat_empty_messages(
        self, llm_config: LLMConfig,
    ) -> None:
        """chat() works with an empty message list."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("No context given.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        result = await provider.chat([])
        assert result == "No context given."

    async def test_chat_empty_content_returns_empty_string(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """chat() returns empty string when response has no content blocks."""
        provider = _make_provider(llm_config)
        mock_response = _make_empty_response()
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_chat_none_text_returns_empty_string(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """chat() returns empty string when content block text is empty."""
        provider = _make_provider(llm_config)
        block = MagicMock()
        block.type = "text"
        block.text = ""
        response = MagicMock()
        response.content = [block]
        provider._client.messages.create = AsyncMock(
            return_value=response,
        )

        result = await provider.chat(sample_messages)
        assert result == ""

    async def test_stream_skips_non_delta_events(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """stream() skips events that are not content_block_delta."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            return_value=_make_stream_events(["Hello", None, " world"]),
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    async def test_stream_skips_empty_text(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """stream() skips delta events with empty text."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            return_value=_make_stream_events(["Hello", "", " world"]),
        )

        tokens: list[str] = []
        async for token in provider.stream(sample_messages):
            tokens.append(token)

        assert tokens == ["Hello", " world"]

    def test_format_messages_separates_system(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """_format_messages extracts system messages from conversation."""
        provider = _make_provider(llm_config)
        system_prompt, conversation = provider._format_messages(
            sample_messages,
        )
        assert system_prompt == "You are sardonic."
        assert conversation == [{"role": "user", "content": "Hello"}]

    def test_format_messages_multiple_system(
        self, llm_config: LLMConfig,
    ) -> None:
        """_format_messages concatenates multiple system messages."""
        provider = _make_provider(llm_config)
        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM, content="You are sardonic.",
            ),
            ChatMessage(
                role=MessageRole.SYSTEM, content="You hate small talk.",
            ),
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ]
        system_prompt, conversation = provider._format_messages(messages)
        assert system_prompt == "You are sardonic.\nYou hate small talk."
        assert conversation == [{"role": "user", "content": "Hello"}]

    def test_format_messages_no_system(
        self,
        llm_config: LLMConfig,
        user_only_messages: list[ChatMessage],
    ) -> None:
        """_format_messages returns None when no system messages present."""
        provider = _make_provider(llm_config)
        system_prompt, conversation = provider._format_messages(
            user_only_messages,
        )
        assert system_prompt is None
        assert len(conversation) == 3

    def test_format_messages_empty_list(
        self, llm_config: LLMConfig,
    ) -> None:
        """_format_messages handles empty list."""
        provider = _make_provider(llm_config)
        system_prompt, conversation = provider._format_messages([])
        assert system_prompt is None
        assert conversation == []


# --- Tool use tests ---


class TestAnthropicToolUse:
    """Tests for chat_with_tools() — ToolCapableLLM protocol."""

    async def test_chat_with_tools_basic(
        self, llm_config: LLMConfig,
    ) -> None:
        """Tool definitions are translated and tool_use blocks are parsed."""
        provider = _make_provider(llm_config)

        # Mock response with both text and tool_use blocks
        response = MagicMock()
        response.content = [
            _make_tool_use_block(
                "look_at_user", {}, block_id="toolu_abc",
            ),
        ]
        provider._client.messages.create = AsyncMock(
            return_value=response,
        )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "look_at_user",
                    "description": "Look at the user via webcam.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]
        messages = [
            ChatMessage(role=MessageRole.USER, content="Look at me"),
        ]

        result = await provider.chat_with_tools(messages, tools)

        assert result["content"] is None
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1
        tc = result["tool_calls"][0]
        assert tc["function"]["name"] == "look_at_user"
        assert tc["id"] == "toolu_abc"

        # Verify tool format was translated to Anthropic format
        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        tool_def = call_kwargs["tools"][0]
        assert tool_def["name"] == "look_at_user"
        assert "input_schema" in tool_def

    async def test_chat_with_tools_text_only(
        self, llm_config: LLMConfig,
    ) -> None:
        """Response with no tool_calls returns text and None tool_calls."""
        provider = _make_provider(llm_config)

        response = MagicMock()
        response.content = [_make_text_block("Just talking.")]
        provider._client.messages.create = AsyncMock(
            return_value=response,
        )

        result = await provider.chat_with_tools(
            [ChatMessage(role=MessageRole.USER, content="hi")],
            [],
        )

        assert result["content"] == "Just talking."
        assert result["tool_calls"] is None

    async def test_chat_with_tools_connection_error(
        self, llm_config: LLMConfig,
    ) -> None:
        """chat_with_tools() wraps API errors in ConnectionError."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            side_effect=Exception("Network error"),
        )

        with pytest.raises(
            ConnectionError, match="Failed to call Anthropic API",
        ):
            await provider.chat_with_tools(
                [ChatMessage(role=MessageRole.USER, content="hi")],
                [],
            )

    async def test_chat_with_tools_text_and_tool(
        self, llm_config: LLMConfig,
    ) -> None:
        """Response can contain both text and tool_use blocks."""
        provider = _make_provider(llm_config)

        response = MagicMock()
        response.content = [
            _make_text_block("Let me look..."),
            _make_tool_use_block("look_at_user", {}),
        ]
        provider._client.messages.create = AsyncMock(
            return_value=response,
        )

        result = await provider.chat_with_tools(
            [ChatMessage(role=MessageRole.USER, content="look at me")],
            [{"function": {"name": "look_at_user", "description": "Look"}}],
        )

        assert result["content"] == "Let me look..."
        assert result["tool_calls"] is not None
        assert len(result["tool_calls"]) == 1


# --- Prompt caching tests ---


class TestPromptCaching:
    """Tests for prompt caching via _format_system()."""

    def test_caching_with_marker_splits_blocks(
        self, llm_config: LLMConfig,
    ) -> None:
        """System prompt with cache marker is split into cached blocks."""
        provider = _make_provider(llm_config, enable_caching=True)

        prompt = "Stable prefix\n\n---CACHE_BREAK---\n\nDynamic suffix"
        result = provider._format_system(prompt)

        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "Stable prefix"
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert result[1]["type"] == "text"
        assert result[1]["text"] == "Dynamic suffix"

    def test_caching_without_marker_returns_string(
        self, llm_config: LLMConfig,
    ) -> None:
        """System prompt without cache marker is returned as plain string."""
        provider = _make_provider(llm_config, enable_caching=True)

        prompt = "Just a normal system prompt"
        result = provider._format_system(prompt)

        assert isinstance(result, str)
        assert result == prompt

    def test_caching_disabled_returns_string(
        self, llm_config: LLMConfig,
    ) -> None:
        """With caching disabled, marker is not split."""
        provider = _make_provider(llm_config, enable_caching=False)

        prompt = "Prefix\n\n---CACHE_BREAK---\n\nSuffix"
        result = provider._format_system(prompt)

        assert isinstance(result, str)
        assert result == prompt

    def test_caching_none_prompt(self, llm_config: LLMConfig) -> None:
        """None prompt returns None."""
        provider = _make_provider(llm_config, enable_caching=True)
        assert provider._format_system(None) is None

    async def test_caching_used_in_chat(
        self, llm_config: LLMConfig,
    ) -> None:
        """chat() passes cached system blocks to the API."""
        provider = _make_provider(llm_config, enable_caching=True)
        mock_response = _make_chat_response("Cached response.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        messages = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content="Prefix\n\n---CACHE_BREAK---\n\nSuffix",
            ),
            ChatMessage(role=MessageRole.USER, content="Hello"),
        ]
        await provider.chat(messages)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        system_val = call_kwargs["system"]
        assert isinstance(system_val, list)
        assert system_val[0]["cache_control"] == {"type": "ephemeral"}

    def test_caching_empty_suffix(self, llm_config: LLMConfig) -> None:
        """Cache marker with empty suffix produces only prefix block."""
        provider = _make_provider(llm_config, enable_caching=True)

        prompt = "Only prefix\n\n---CACHE_BREAK---\n\n"
        result = provider._format_system(prompt)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["text"] == "Only prefix"


# --- Extended thinking tests ---


class TestExtendedThinking:
    """Tests for extended thinking in chat()."""

    async def test_thinking_enabled(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """thinking=True passes thinking params and forces temperature=1.0."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("Deep thought.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        await provider.chat(
            sample_messages, thinking=True, thinking_budget=8192,
        )

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"] == {
            "type": "enabled",
            "budget_tokens": 8192,
        }
        assert call_kwargs["temperature"] == 1.0

    async def test_thinking_filters_thinking_blocks(
        self, llm_config: LLMConfig,
    ) -> None:
        """Thinking blocks are filtered from response, only text returned."""
        provider = _make_provider(llm_config)

        response = MagicMock()
        response.content = [
            _make_thinking_block("Let me ponder this deeply..."),
            _make_text_block("The answer is 42."),
        ]
        provider._client.messages.create = AsyncMock(
            return_value=response,
        )

        messages = [
            ChatMessage(role=MessageRole.USER, content="meaning of life?"),
        ]
        result = await provider.chat(messages, thinking=True)

        assert result == "The answer is 42."
        assert "ponder" not in result

    async def test_thinking_disabled_uses_normal_temperature(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """Without thinking, normal temperature is used."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("Normal response.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        await provider.chat(sample_messages)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert "thinking" not in call_kwargs
        assert call_kwargs["temperature"] == 0.7

    async def test_thinking_default_budget(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """Default thinking budget is 4096 when not specified."""
        provider = _make_provider(llm_config)
        mock_response = _make_chat_response("Thought about it.")
        provider._client.messages.create = AsyncMock(
            return_value=mock_response,
        )

        await provider.chat(sample_messages, thinking=True)

        call_kwargs = provider._client.messages.create.call_args.kwargs
        assert call_kwargs["thinking"]["budget_tokens"] == 4096


# --- Error handling tests ---


class TestAnthropicProviderErrors:
    """Error handling for AnthropicProvider."""

    def test_missing_api_key_raises_value_error(
        self, llm_config: LLMConfig,
    ) -> None:
        """Constructor raises ValueError when no API key is available."""
        with patch.dict("os.environ", {}, clear=True):
            import os

            os.environ.pop("ANTHROPIC_API_KEY", None)
            with pytest.raises(
                ValueError, match="Anthropic API key required",
            ):
                AnthropicProvider(llm_config)

    async def test_chat_connection_error(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """chat() wraps API errors with clear message."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            side_effect=Exception("Connection refused"),
        )

        with pytest.raises(
            ConnectionError, match="Failed to call Anthropic API",
        ):
            await provider.chat(sample_messages)

    async def test_stream_connection_error(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """stream() wraps API errors with clear message."""
        provider = _make_provider(llm_config)
        provider._client.messages.create = AsyncMock(
            side_effect=Exception("Connection refused"),
        )

        with pytest.raises(
            ConnectionError, match="Failed to call Anthropic API",
        ):
            async for _ in provider.stream(sample_messages):
                pass  # pragma: no cover

    async def test_chat_preserves_original_exception(
        self,
        llm_config: LLMConfig,
        sample_messages: list[ChatMessage],
    ) -> None:
        """ConnectionError chains the original exception."""
        provider = _make_provider(llm_config)
        original = OSError("network down")
        provider._client.messages.create = AsyncMock(side_effect=original)

        with pytest.raises(ConnectionError) as exc_info:
            await provider.chat(sample_messages)

        assert exc_info.value.__cause__ is original

    def test_constructor_key_precedence(
        self, llm_config: LLMConfig,
    ) -> None:
        """Constructor api_key takes precedence over environment variable."""
        with patch.dict(
            "os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-key"},
        ):
            provider = AnthropicProvider(
                llm_config, api_key="sk-ant-direct-key",
            )
            assert provider._client.api_key == "sk-ant-direct-key"
