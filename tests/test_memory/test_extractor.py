"""Tests for the MemoryExtractor class."""

from __future__ import annotations

from datetime import UTC, datetime

from seaman_brain.config import MemoryConfig
from seaman_brain.memory.extractor import MemoryExtractor
from seaman_brain.types import ChatMessage, MessageRole


def _make_message(role: MessageRole, content: str) -> ChatMessage:
    """Create a ChatMessage with a fixed timestamp."""
    return ChatMessage(role=role, content=content, timestamp=datetime.now(UTC))


# --- Happy Path Tests ---


async def test_extract_returns_facts(mocker):
    """Extract parses LLM response into fact list."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = (
        "User likes cats\n"
        "User works as a programmer\n"
        "User mentioned a friend named Bob"
    )
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [
        _make_message(MessageRole.USER, "I love cats"),
        _make_message(MessageRole.ASSISTANT, "Cats are decent creatures."),
        _make_message(MessageRole.USER, "My friend Bob and I are programmers"),
    ]

    facts = await extractor.extract(messages)

    assert len(facts) == 3
    assert "User likes cats" in facts
    assert "User works as a programmer" in facts
    mock_llm.chat.assert_awaited_once()


async def test_extract_and_store_persists_facts(mocker):
    """extract_and_store embeds and saves facts to semantic memory."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "User likes pizza\nUser has a dog"

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * 384

    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [
        _make_message(MessageRole.USER, "I like pizza and I have a dog"),
        _make_message(MessageRole.ASSISTANT, "Pizza-loving dog owners are okay."),
    ]

    stored = await extractor.extract_and_store(messages)

    assert len(stored) == 2
    assert "User likes pizza" in stored
    assert "User has a dog" in stored
    assert mock_embeddings.embed.await_count == 2
    assert mock_semantic.add.await_count == 2


async def test_extract_and_store_resets_counter(mocker):
    """extract_and_store resets the message counter after completion."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "Some fact"

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * 384

    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    extractor.increment_counter()
    extractor.increment_counter()
    assert extractor.message_count == 2

    messages = [_make_message(MessageRole.USER, "hello")]
    await extractor.extract_and_store(messages)

    assert extractor.message_count == 0


async def test_should_extract_at_interval(mocker):
    """should_extract returns True when message count reaches interval."""
    mock_llm = mocker.AsyncMock()
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    config = MemoryConfig(extraction_interval=3)
    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic, config)

    assert extractor.extraction_interval == 3
    assert not extractor.should_extract()

    extractor.increment_counter()
    extractor.increment_counter()
    assert not extractor.should_extract()

    extractor.increment_counter()
    assert extractor.should_extract()


async def test_extract_filters_system_messages(mocker):
    """System messages are excluded from the conversation sent to LLM."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "User said hello"
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [
        _make_message(MessageRole.SYSTEM, "You are a creature"),
        _make_message(MessageRole.USER, "Hello"),
        _make_message(MessageRole.ASSISTANT, "Hi there."),
    ]

    facts = await extractor.extract(messages)

    assert len(facts) == 1
    # Verify system message was not included in the prompt
    call_args = mock_llm.chat.call_args[0][0]
    prompt_content = call_args[0].content
    assert "system:" not in prompt_content.lower()
    assert "user: Hello" in prompt_content


# --- Edge Case Tests ---


async def test_extract_empty_messages(mocker):
    """Returns empty list for empty message list."""
    mock_llm = mocker.AsyncMock()
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    facts = await extractor.extract([])

    assert facts == []
    mock_llm.chat.assert_not_awaited()


async def test_extract_only_system_messages(mocker):
    """Returns empty when only system messages are present."""
    mock_llm = mocker.AsyncMock()
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [
        _make_message(MessageRole.SYSTEM, "You are a creature"),
        _make_message(MessageRole.SYSTEM, "Stay in character"),
    ]

    facts = await extractor.extract(messages)

    assert facts == []
    mock_llm.chat.assert_not_awaited()


async def test_extract_none_response(mocker):
    """LLM returning 'NONE' yields empty fact list."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "NONE"
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "Just testing")]

    facts = await extractor.extract(messages)

    assert facts == []


async def test_extract_empty_response(mocker):
    """LLM returning empty string yields empty fact list."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = ""
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "hi")]

    facts = await extractor.extract(messages)

    assert facts == []


async def test_extract_response_with_blank_lines(mocker):
    """Blank lines in LLM response are filtered out."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "Fact one\n\n\nFact two\n\n"
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "hello")]

    facts = await extractor.extract(messages)

    assert facts == ["Fact one", "Fact two"]


async def test_extract_and_store_skips_empty_embeddings(mocker):
    """Facts with empty embeddings are skipped during storage."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "Fact A\nFact B\nFact C"

    # Return empty embedding for second fact
    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.side_effect = [
        [0.1] * 384,
        [],
        [0.3] * 384,
    ]

    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "hello")]

    stored = await extractor.extract_and_store(messages)

    assert len(stored) == 2
    assert "Fact A" in stored
    assert "Fact C" in stored
    assert mock_semantic.add.await_count == 2


async def test_extract_and_store_no_facts(mocker):
    """extract_and_store with no facts returns empty list."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "NONE"
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "hi")]

    stored = await extractor.extract_and_store(messages)

    assert stored == []
    mock_embeddings.embed.assert_not_awaited()
    mock_semantic.add.assert_not_awaited()


# --- Error Handling Tests ---


async def test_extract_llm_error_returns_empty(mocker):
    """LLM errors during extraction return empty list (graceful degradation)."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.side_effect = ConnectionError("LLM unavailable")
    mock_embeddings = mocker.AsyncMock()
    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "hello")]

    facts = await extractor.extract(messages)

    assert facts == []


async def test_extract_and_store_embedding_error_skips_fact(mocker):
    """Embedding errors for individual facts skip that fact, store others."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "Fact A\nFact B"

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.side_effect = [
        ConnectionError("Ollama down"),
        [0.2] * 384,
    ]

    mock_semantic = mocker.AsyncMock()

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "test")]

    stored = await extractor.extract_and_store(messages)

    assert len(stored) == 1
    assert "Fact B" in stored
    assert mock_semantic.add.await_count == 1


async def test_extract_and_store_semantic_error_skips_fact(mocker):
    """Semantic store errors for individual facts skip that fact."""
    mock_llm = mocker.AsyncMock()
    mock_llm.chat.return_value = "Fact A\nFact B"

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * 384

    mock_semantic = mocker.AsyncMock()
    mock_semantic.add.side_effect = [
        ValueError("Dimension mismatch"),
        None,
    ]

    extractor = MemoryExtractor(mock_llm, mock_embeddings, mock_semantic)
    messages = [_make_message(MessageRole.USER, "test")]

    stored = await extractor.extract_and_store(messages)

    assert len(stored) == 1
    assert "Fact B" in stored


async def test_parse_facts_whitespace_only():
    """_parse_facts handles whitespace-only input."""
    assert MemoryExtractor._parse_facts("   \n  \n  ") == []


async def test_parse_facts_none_mixed_with_facts():
    """_parse_facts handles NONE appearing as a line among real facts."""
    result = MemoryExtractor._parse_facts("Real fact\nNONE\nAnother fact")
    assert result == ["Real fact", "Another fact"]
