"""Tests for LLM provider abstract protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from seaman_brain.llm.base import LLMProvider
from seaman_brain.types import ChatMessage, MessageRole

# --- Mock implementation for protocol compliance testing ---


class MockLLMProvider:
    """A conforming mock implementation of LLMProvider."""

    def __init__(self, response: str = "Hello, human.") -> None:
        self.response = response
        self.last_messages: list[ChatMessage] | None = None

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.last_messages = messages
        return self.response

    async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        self.last_messages = messages
        for token in self.response.split():
            yield token


class IncompleteProvider:
    """A provider missing the stream() method — should NOT satisfy the protocol."""

    async def chat(self, messages: list[ChatMessage]) -> str:
        return "partial"


class EmptyClass:
    """Totally unrelated class — should NOT satisfy the protocol."""

    pass


# --- Happy path tests ---


class TestLLMProviderProtocol:
    """Test that the LLMProvider protocol works correctly."""

    def test_mock_satisfies_protocol(self) -> None:
        """A class implementing both chat() and stream() satisfies LLMProvider."""
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_protocol_is_runtime_checkable(self) -> None:
        """LLMProvider can be used with isinstance() at runtime."""
        provider = MockLLMProvider("test")
        assert isinstance(provider, LLMProvider)
        # Also verify the class itself
        assert issubclass(MockLLMProvider, LLMProvider)

    async def test_chat_returns_string(self) -> None:
        """chat() returns the expected string response."""
        provider = MockLLMProvider("Sardonic reply.")
        messages = [
            ChatMessage(role=MessageRole.USER, content="Hi"),
        ]
        result = await provider.chat(messages)
        assert result == "Sardonic reply."
        assert provider.last_messages == messages

    async def test_stream_yields_tokens(self) -> None:
        """stream() yields individual tokens from the response."""
        provider = MockLLMProvider("I am Seaman")
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are sardonic."),
            ChatMessage(role=MessageRole.USER, content="Who are you?"),
        ]
        tokens: list[str] = []
        async for token in provider.stream(messages):
            tokens.append(token)
        assert tokens == ["I", "am", "Seaman"]
        assert provider.last_messages == messages

    async def test_chat_with_empty_messages(self) -> None:
        """chat() works with an empty message list (edge case)."""
        provider = MockLLMProvider("Default response")
        result = await provider.chat([])
        assert result == "Default response"
        assert provider.last_messages == []


# --- Edge case tests ---


class TestProtocolEdgeCases:
    """Edge cases for protocol compliance checking."""

    def test_incomplete_provider_does_not_satisfy(self) -> None:
        """A class missing stream() does NOT satisfy LLMProvider."""
        provider = IncompleteProvider()
        assert not isinstance(provider, LLMProvider)

    def test_empty_class_does_not_satisfy(self) -> None:
        """A completely unrelated class does NOT satisfy LLMProvider."""
        obj = EmptyClass()
        assert not isinstance(obj, LLMProvider)

    def test_none_does_not_satisfy(self) -> None:
        """None does not satisfy the protocol."""
        assert not isinstance(None, LLMProvider)

    def test_string_does_not_satisfy(self) -> None:
        """Built-in types do not satisfy the protocol."""
        assert not isinstance("not a provider", LLMProvider)


# --- Error handling / robustness tests ---


class TestProviderErrorHandling:
    """Test that providers can properly raise and propagate errors."""

    async def test_chat_can_raise_exception(self) -> None:
        """Providers can raise exceptions from chat() and they propagate."""

        class ErrorProvider:
            async def chat(self, messages: list[ChatMessage]) -> str:
                raise ConnectionError("LLM server unavailable")

            async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
                yield ""  # pragma: no cover

        provider = ErrorProvider()
        assert isinstance(provider, LLMProvider)
        with pytest.raises(ConnectionError, match="LLM server unavailable"):
            await provider.chat([])

    async def test_stream_can_raise_mid_iteration(self) -> None:
        """Providers can raise exceptions during streaming."""

        class StreamErrorProvider:
            async def chat(self, messages: list[ChatMessage]) -> str:
                return ""  # pragma: no cover

            async def stream(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
                yield "token1"
                raise TimeoutError("Stream timed out")

        provider = StreamErrorProvider()
        assert isinstance(provider, LLMProvider)
        tokens: list[str] = []
        with pytest.raises(TimeoutError, match="Stream timed out"):
            async for token in provider.stream([]):
                tokens.append(token)
        assert tokens == ["token1"]

    async def test_chat_with_large_message_list(self) -> None:
        """chat() handles a large number of messages without issue."""
        provider = MockLLMProvider("ok")
        messages = [
            ChatMessage(role=MessageRole.USER, content=f"Message {i}")
            for i in range(100)
        ]
        result = await provider.chat(messages)
        assert result == "ok"
        assert len(provider.last_messages) == 100
