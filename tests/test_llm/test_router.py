"""Tests for LLMRouter — hybrid local/cloud LLM routing."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from seaman_brain.llm.router import CALL_TYPES, ROUTING_MODES, LLMRouter
from seaman_brain.types import ChatMessage, MessageRole

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeProvider:
    """Minimal LLMProvider for testing. Tracks calls and returns a response."""

    def __init__(self, name: str = "fake", response: str = "ok") -> None:
        self.name = name
        self.response = response
        self.chat_calls: list[list[ChatMessage]] = []
        self.chat_kwargs: list[dict[str, Any]] = []
        self.stream_calls: list[list[ChatMessage]] = []

    async def chat(
        self, messages: list[ChatMessage], **kwargs: Any,
    ) -> str:
        self.chat_calls.append(messages)
        self.chat_kwargs.append(kwargs)
        return self.response

    async def stream(
        self, messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        self.stream_calls.append(messages)
        for token in self.response.split():
            yield token


class FakeToolProvider(FakeProvider):
    """FakeProvider that also supports chat_with_tools."""

    def __init__(
        self,
        name: str = "tool",
        response: str = "ok",
        tool_result: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(name, response)
        self._tool_result = tool_result or {
            "content": response,
            "tool_calls": None,
        }
        self.tool_calls: list[tuple[list[ChatMessage], list]] = []

    async def chat_with_tools(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.tool_calls.append((messages, tools))
        return self._tool_result


class StrictProvider:
    """Provider that does NOT accept **kwargs — only positional messages."""

    def __init__(self, response: str = "strict") -> None:
        self.response = response
        self.chat_calls: list[list[ChatMessage]] = []

    async def chat(self, messages: list[ChatMessage]) -> str:
        self.chat_calls.append(messages)
        return self.response

    async def stream(
        self, messages: list[ChatMessage],
    ) -> AsyncIterator[str]:
        for token in self.response.split():
            yield token


def _msgs(text: str = "hi") -> list[ChatMessage]:
    """Create a simple message list for testing."""
    return [ChatMessage(role=MessageRole.USER, content=text)]


# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


class TestRouterConstruction:
    """Tests for LLMRouter initialization and mode validation."""

    def test_default_mode_is_local(self) -> None:
        """Default routing mode is 'local'."""
        router = LLMRouter(local=FakeProvider())
        assert router.mode == "local"

    def test_custom_mode_accepted(self) -> None:
        """Router accepts valid mode strings."""
        for mode in ROUTING_MODES:
            router = LLMRouter(local=FakeProvider(), mode=mode)
            assert router.mode == mode

    def test_invalid_mode_raises(self) -> None:
        """Router rejects unknown mode strings."""
        with pytest.raises(ValueError, match="Unknown routing mode"):
            LLMRouter(local=FakeProvider(), mode="invalid")

    def test_mode_setter_validates(self) -> None:
        """Setting mode validates the value."""
        router = LLMRouter(local=FakeProvider())
        router.mode = "hybrid"
        assert router.mode == "hybrid"
        with pytest.raises(ValueError, match="Unknown routing mode"):
            router.mode = "bad"

    def test_mode_case_insensitive(self) -> None:
        """Mode comparison is case-insensitive."""
        router = LLMRouter(local=FakeProvider(), mode="HYBRID")
        assert router.mode == "hybrid"

    def test_cloud_provider_property(self) -> None:
        """Cloud provider is accessible and settable."""
        local = FakeProvider("local")
        cloud = FakeProvider("cloud")
        router = LLMRouter(local=local, cloud=cloud)
        assert router.cloud_provider is cloud
        router.cloud_provider = None
        assert router.cloud_provider is None

    def test_local_provider_property(self) -> None:
        """Local provider is accessible."""
        local = FakeProvider("local")
        router = LLMRouter(local=local)
        assert router.local_provider is local

    def test_default_call_type(self) -> None:
        """Default call type is 'conversation'."""
        router = LLMRouter(local=FakeProvider())
        assert router.current_call_type == "conversation"


# ---------------------------------------------------------------------------
# Local mode routing
# ---------------------------------------------------------------------------


class TestLocalMode:
    """Tests for 'local' routing mode — all calls go to local provider."""

    async def test_chat_uses_local(self) -> None:
        """In local mode, chat() goes to local provider."""
        local = FakeProvider("local", "local response")
        cloud = FakeProvider("cloud", "cloud response")
        router = LLMRouter(local=local, cloud=cloud, mode="local")

        result = await router.chat(_msgs())
        assert result == "local response"
        assert len(local.chat_calls) == 1
        assert len(cloud.chat_calls) == 0

    async def test_stream_uses_local(self) -> None:
        """In local mode, stream() goes to local provider."""
        local = FakeProvider("local", "hello world")
        cloud = FakeProvider("cloud", "cloud says")
        router = LLMRouter(local=local, cloud=cloud, mode="local")

        tokens = []
        async for token in router.stream(_msgs()):
            tokens.append(token)
        assert tokens == ["hello", "world"]
        assert len(local.stream_calls) == 1
        assert len(cloud.stream_calls) == 0

    async def test_all_call_types_use_local(self) -> None:
        """In local mode, every call type goes to local."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="local")

        for ct in CALL_TYPES:
            router.set_call_type(ct)
            await router.chat(_msgs())

        assert len(local.chat_calls) == len(CALL_TYPES)
        assert len(cloud.chat_calls) == 0


# ---------------------------------------------------------------------------
# Cloud mode routing
# ---------------------------------------------------------------------------


class TestCloudMode:
    """Tests for 'cloud' routing mode — all calls go to cloud provider."""

    async def test_chat_uses_cloud(self) -> None:
        """In cloud mode, chat() goes to cloud provider."""
        local = FakeProvider("local", "local response")
        cloud = FakeProvider("cloud", "cloud response")
        router = LLMRouter(local=local, cloud=cloud, mode="cloud")

        result = await router.chat(_msgs())
        assert result == "cloud response"
        assert len(cloud.chat_calls) == 1
        assert len(local.chat_calls) == 0

    async def test_stream_uses_cloud(self) -> None:
        """In cloud mode, stream() goes to cloud provider."""
        local = FakeProvider("local", "local says")
        cloud = FakeProvider("cloud", "cloud says")
        router = LLMRouter(local=local, cloud=cloud, mode="cloud")

        tokens = []
        async for token in router.stream(_msgs()):
            tokens.append(token)
        assert tokens == ["cloud", "says"]
        assert len(cloud.stream_calls) == 1
        assert len(local.stream_calls) == 0

    async def test_all_call_types_use_cloud(self) -> None:
        """In cloud mode, every call type goes to cloud."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="cloud")

        for ct in CALL_TYPES:
            router.set_call_type(ct)
            await router.chat(_msgs())

        assert len(cloud.chat_calls) == len(CALL_TYPES)
        assert len(local.chat_calls) == 0


# ---------------------------------------------------------------------------
# Hybrid mode routing
# ---------------------------------------------------------------------------


class TestHybridMode:
    """Tests for 'hybrid' — conversation->cloud, others->local."""

    async def test_conversation_goes_to_cloud(self) -> None:
        """In hybrid mode, 'conversation' call type goes to cloud."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        router.set_call_type("conversation")
        result = await router.chat(_msgs())
        assert result == "cloud"
        assert len(cloud.chat_calls) == 1
        assert len(local.chat_calls) == 0

    async def test_reaction_goes_to_local(self) -> None:
        """In hybrid mode, 'reaction' call type goes to local."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        router.set_call_type("reaction")
        result = await router.chat(_msgs())
        assert result == "local"
        assert len(local.chat_calls) == 1
        assert len(cloud.chat_calls) == 0

    async def test_autonomous_goes_to_local(self) -> None:
        """In hybrid mode, 'autonomous' call type goes to local."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        router.set_call_type("autonomous")
        result = await router.chat(_msgs())
        assert result == "local"
        assert len(local.chat_calls) == 1
        assert len(cloud.chat_calls) == 0

    async def test_stream_conversation_goes_to_cloud(self) -> None:
        """In hybrid mode, streaming 'conversation' goes to cloud."""
        local = FakeProvider("local", "local says")
        cloud = FakeProvider("cloud", "cloud says")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        router.set_call_type("conversation")
        tokens = []
        async for token in router.stream(_msgs()):
            tokens.append(token)
        assert tokens == ["cloud", "says"]
        assert len(cloud.stream_calls) == 1

    async def test_stream_reaction_goes_to_local(self) -> None:
        """In hybrid mode, streaming 'reaction' goes to local."""
        local = FakeProvider("local", "local says")
        cloud = FakeProvider("cloud", "cloud says")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        router.set_call_type("reaction")
        tokens = []
        async for token in router.stream(_msgs()):
            tokens.append(token)
        assert tokens == ["local", "says"]
        assert len(local.stream_calls) == 1

    async def test_mixed_call_types(self) -> None:
        """Hybrid mode correctly routes different call types in sequence."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        # conversation -> cloud
        router.set_call_type("conversation")
        await router.chat(_msgs("conv"))

        # reaction -> local
        router.set_call_type("reaction")
        await router.chat(_msgs("react"))

        # autonomous -> local
        router.set_call_type("autonomous")
        await router.chat(_msgs("auto"))

        # conversation again -> cloud
        router.set_call_type("conversation")
        await router.chat(_msgs("conv2"))

        assert len(cloud.chat_calls) == 2
        assert len(local.chat_calls) == 2


# ---------------------------------------------------------------------------
# Fallback behavior
# ---------------------------------------------------------------------------


class TestFallback:
    """Tests for fallback behavior when cloud is None."""

    async def test_cloud_mode_falls_back_to_local(self) -> None:
        """Cloud mode falls back to local if cloud provider is None."""
        local = FakeProvider("local", "local fallback")
        router = LLMRouter(local=local, cloud=None, mode="cloud")

        result = await router.chat(_msgs())
        assert result == "local fallback"
        assert len(local.chat_calls) == 1

    async def test_hybrid_mode_falls_back_to_local(self) -> None:
        """Hybrid mode falls back to local for conversation if cloud is None."""
        local = FakeProvider("local", "local fallback")
        router = LLMRouter(local=local, cloud=None, mode="hybrid")

        router.set_call_type("conversation")
        result = await router.chat(_msgs())
        assert result == "local fallback"
        assert len(local.chat_calls) == 1

    async def test_stream_falls_back_to_local(self) -> None:
        """Stream also falls back to local when cloud is None."""
        local = FakeProvider("local", "local stream")
        router = LLMRouter(local=local, cloud=None, mode="cloud")

        tokens = []
        async for token in router.stream(_msgs()):
            tokens.append(token)
        assert tokens == ["local", "stream"]
        assert len(local.stream_calls) == 1

    async def test_cloud_cleared_at_runtime(self) -> None:
        """Clearing cloud provider at runtime causes fallback to local."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="cloud")

        # First call goes to cloud
        result = await router.chat(_msgs())
        assert result == "cloud"

        # Clear cloud provider
        router.cloud_provider = None

        # Second call falls back to local
        result = await router.chat(_msgs())
        assert result == "local"


# ---------------------------------------------------------------------------
# set_call_type behavior
# ---------------------------------------------------------------------------


class TestSetCallType:
    """Tests for set_call_type persistence and validation."""

    def test_set_call_type_persists(self) -> None:
        """Call type persists across multiple provider selections."""
        router = LLMRouter(local=FakeProvider())
        router.set_call_type("reaction")
        assert router.current_call_type == "reaction"

    def test_set_call_type_all_valid_types(self) -> None:
        """All valid call types are accepted."""
        router = LLMRouter(local=FakeProvider())
        for ct in CALL_TYPES:
            router.set_call_type(ct)
            assert router.current_call_type == ct

    def test_unknown_call_type_defaults_to_conversation(self) -> None:
        """Unknown call type falls back to 'conversation' with warning."""
        router = LLMRouter(local=FakeProvider())
        router.set_call_type("unknown_type")
        assert router.current_call_type == "conversation"

    async def test_call_type_affects_routing(self) -> None:
        """Changing call type between calls changes which provider is used."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        router.set_call_type("conversation")
        await router.chat(_msgs())
        assert len(cloud.chat_calls) == 1

        router.set_call_type("reaction")
        await router.chat(_msgs())
        assert len(local.chat_calls) == 1


# ---------------------------------------------------------------------------
# Mode switching at runtime
# ---------------------------------------------------------------------------


class TestModeSwitch:
    """Tests for changing routing mode at runtime."""

    async def test_switch_local_to_cloud(self) -> None:
        """Switching from local to cloud changes routing."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="local")

        # Local mode
        result = await router.chat(_msgs())
        assert result == "local"

        # Switch to cloud
        router.mode = "cloud"
        result = await router.chat(_msgs())
        assert result == "cloud"

    async def test_switch_to_hybrid(self) -> None:
        """Switching to hybrid mode enables split routing."""
        local = FakeProvider("local", "local")
        cloud = FakeProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="local")

        router.mode = "hybrid"

        router.set_call_type("conversation")
        result = await router.chat(_msgs())
        assert result == "cloud"

        router.set_call_type("reaction")
        result = await router.chat(_msgs())
        assert result == "local"

    async def test_swap_cloud_provider(self) -> None:
        """Replacing the cloud provider at runtime works."""
        local = FakeProvider("local", "local")
        cloud1 = FakeProvider("cloud1", "cloud1")
        cloud2 = FakeProvider("cloud2", "cloud2")
        router = LLMRouter(local=local, cloud=cloud1, mode="cloud")

        result = await router.chat(_msgs())
        assert result == "cloud1"

        router.cloud_provider = cloud2
        result = await router.chat(_msgs())
        assert result == "cloud2"


# ---------------------------------------------------------------------------
# chat_with_tools routing
# ---------------------------------------------------------------------------


class TestChatWithTools:
    """Tests for chat_with_tools() routing through the router."""

    async def test_forwards_to_tool_capable_provider(self) -> None:
        """Tool-capable cloud provider receives chat_with_tools calls."""
        local = FakeProvider("local", "local")
        cloud = FakeToolProvider(
            "cloud",
            "cloud",
            tool_result={"content": "I see you!", "tool_calls": None},
        )
        router = LLMRouter(local=local, cloud=cloud, mode="cloud")

        tools = [{"function": {"name": "look", "description": "Look"}}]
        result = await router.chat_with_tools(_msgs(), tools)

        assert result["content"] == "I see you!"
        assert result["tool_calls"] is None
        assert len(cloud.tool_calls) == 1

    async def test_fallback_plain_chat_no_tools(self) -> None:
        """Provider without chat_with_tools falls back to plain chat."""
        local = FakeProvider("local", "plain response")
        router = LLMRouter(local=local, mode="local")

        tools = [{"function": {"name": "look", "description": "Look"}}]
        result = await router.chat_with_tools(_msgs(), tools)

        assert result["content"] == "plain response"
        assert result["tool_calls"] is None
        assert len(local.chat_calls) == 1

    async def test_tool_routing_respects_mode(self) -> None:
        """chat_with_tools routes based on mode and call type."""
        local = FakeToolProvider("local", "local")
        cloud = FakeToolProvider("cloud", "cloud")
        router = LLMRouter(local=local, cloud=cloud, mode="hybrid")

        # conversation -> cloud
        router.set_call_type("conversation")
        await router.chat_with_tools(_msgs(), [])
        assert len(cloud.tool_calls) == 1
        assert len(local.tool_calls) == 0

        # reaction -> local
        router.set_call_type("reaction")
        await router.chat_with_tools(_msgs(), [])
        assert len(local.tool_calls) == 1


# ---------------------------------------------------------------------------
# **kwargs forwarding in chat()
# ---------------------------------------------------------------------------


class TestChatKwargsForwarding:
    """Tests for **kwargs forwarding through chat()."""

    async def test_kwargs_forwarded_to_provider(self) -> None:
        """Extra kwargs (thinking=True) are forwarded to the provider."""
        cloud = FakeProvider("cloud", "deep thought")
        router = LLMRouter(local=FakeProvider(), cloud=cloud, mode="cloud")

        result = await router.chat(
            _msgs(), thinking=True, thinking_budget=8192,
        )

        assert result == "deep thought"
        assert len(cloud.chat_kwargs) == 1
        assert cloud.chat_kwargs[0]["thinking"] is True
        assert cloud.chat_kwargs[0]["thinking_budget"] == 8192

    async def test_kwargs_fallback_on_strict_provider(self) -> None:
        """Provider that rejects kwargs falls back to basic chat()."""
        strict = StrictProvider("strict response")
        router = LLMRouter(local=strict, mode="local")

        result = await router.chat(_msgs(), thinking=True)

        assert result == "strict response"
        assert len(strict.chat_calls) == 1

    async def test_no_kwargs_plain_call(self) -> None:
        """Without kwargs, chat() makes a plain call."""
        local = FakeProvider("local", "plain")
        router = LLMRouter(local=local, mode="local")

        result = await router.chat(_msgs())

        assert result == "plain"
        assert len(local.chat_calls) == 1
        assert local.chat_kwargs[0] == {}
