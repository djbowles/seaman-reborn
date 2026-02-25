"""Tests for the HybridRetriever class."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from seaman_brain.config import MemoryConfig
from seaman_brain.memory.retriever import HybridRetriever
from seaman_brain.types import MemoryRecord

VECTOR_DIMS = 384


def _make_record(
    text: str = "test memory",
    importance: float = 0.5,
    source: str = "test",
    timestamp: datetime | None = None,
) -> MemoryRecord:
    """Create a MemoryRecord with a random embedding."""
    return MemoryRecord(
        text=text,
        embedding=np.random.default_rng(42).random(VECTOR_DIMS).astype(np.float32),
        timestamp=timestamp or datetime.now(UTC),
        importance=importance,
        source=source,
    )


# --- Happy Path Tests ---


async def test_retrieve_returns_records(mocker):
    """Basic retrieval returns ranked records."""
    now = datetime.now(UTC)
    records = [
        _make_record("old important", importance=0.9, timestamp=now - timedelta(hours=2)),
        _make_record("new unimportant", importance=0.2, timestamp=now - timedelta(seconds=10)),
        _make_record("mid mid", importance=0.5, timestamp=now - timedelta(hours=1)),
    ]

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = records

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("test query", top_k=3)

    assert len(results) == 3
    mock_embeddings.embed.assert_awaited_once_with("test query")
    mock_semantic.search.assert_awaited_once()


async def test_retrieve_respects_top_k(mocker):
    """Retrieve returns at most top_k results."""
    now = datetime.now(UTC)
    records = [_make_record(f"mem {i}", timestamp=now - timedelta(minutes=i)) for i in range(10)]

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = records

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("query", top_k=3)

    assert len(results) <= 3


async def test_retrieve_reranks_by_combined_score(mocker):
    """Higher importance + recency gets ranked first."""
    now = datetime.now(UTC)
    # High importance but old
    old_important = _make_record("old", importance=0.9, timestamp=now - timedelta(days=10))
    # Low importance but very recent
    new_trivial = _make_record("new", importance=0.1, timestamp=now - timedelta(seconds=1))
    # Medium both
    balanced = _make_record("balanced", importance=0.6, timestamp=now - timedelta(hours=1))

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = [old_important, new_trivial, balanced]

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    config = MemoryConfig(similarity_weight=0.7, recency_weight=0.3)
    retriever = HybridRetriever(mock_semantic, mock_embeddings, config=config)
    results = await retriever.retrieve("query", top_k=3)

    # All three returned, and order depends on combined scores
    assert len(results) == 3
    texts = [r.text for r in results]
    # balanced (0.6 sim + high recency) beats old (0.9 sim + zero recency)
    assert texts[0] == "balanced"


async def test_retrieve_configurable_weights(mocker):
    """When recency_weight is very high, newest memory wins."""
    now = datetime.now(UTC)
    old_high = _make_record("old_high", importance=0.9, timestamp=now - timedelta(days=30))
    new_low = _make_record("new_low", importance=0.1, timestamp=now - timedelta(seconds=1))

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = [old_high, new_low]

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    config = MemoryConfig(similarity_weight=0.1, recency_weight=0.9)
    retriever = HybridRetriever(mock_semantic, mock_embeddings, config=config)
    results = await retriever.retrieve("query", top_k=2)

    assert results[0].text == "new_low"


# --- Edge Case Tests ---


async def test_retrieve_empty_memory(mocker):
    """Returns empty list when semantic store has no memories."""
    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = []

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("query", top_k=5)

    assert results == []


async def test_retrieve_top_k_zero(mocker):
    """top_k <= 0 returns empty list without calling embeddings."""
    mock_semantic = mocker.AsyncMock()
    mock_embeddings = mocker.AsyncMock()

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("query", top_k=0)

    assert results == []
    mock_embeddings.embed.assert_not_awaited()


async def test_retrieve_empty_query_embedding(mocker):
    """Returns empty list when embedding of query is empty."""
    mock_semantic = mocker.AsyncMock()
    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = []

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("", top_k=5)

    assert results == []
    mock_semantic.search.assert_not_awaited()


async def test_retrieve_same_timestamp_candidates(mocker):
    """Candidates with identical timestamps don't cause division by zero."""
    now = datetime.now(UTC)
    records = [
        _make_record("a", importance=0.8, timestamp=now),
        _make_record("b", importance=0.3, timestamp=now),
    ]

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = records

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("query", top_k=2)

    assert len(results) == 2
    # When timestamps are equal, only importance matters
    assert results[0].text == "a"


async def test_retrieve_single_candidate(mocker):
    """Works correctly with only one candidate."""
    record = _make_record("only one", importance=0.7)

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = [record]

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("query", top_k=5)

    assert len(results) == 1
    assert results[0].text == "only one"


# --- Error Handling Tests ---


async def test_retrieve_embedding_error_propagates(mocker):
    """ConnectionError from embedding provider propagates."""
    mock_semantic = mocker.AsyncMock()
    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.side_effect = ConnectionError("Ollama down")

    retriever = HybridRetriever(mock_semantic, mock_embeddings)

    with pytest.raises(ConnectionError, match="Ollama down"):
        await retriever.retrieve("query", top_k=5)


async def test_retrieve_semantic_search_error_propagates(mocker):
    """Errors from semantic search propagate."""
    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.side_effect = RuntimeError("DB error")

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)

    with pytest.raises(RuntimeError, match="DB error"):
        await retriever.retrieve("query", top_k=5)


async def test_retrieve_importance_clamped(mocker):
    """Importance values outside [0,1] are clamped."""
    now = datetime.now(UTC)
    records = [
        _make_record("negative", importance=-0.5, timestamp=now),
        _make_record("over_one", importance=1.5, timestamp=now),
    ]

    mock_semantic = mocker.AsyncMock()
    mock_semantic.search.return_value = records

    mock_embeddings = mocker.AsyncMock()
    mock_embeddings.embed.return_value = [0.1] * VECTOR_DIMS

    retriever = HybridRetriever(mock_semantic, mock_embeddings)
    results = await retriever.retrieve("query", top_k=2)

    # Should not crash; over_one clamped to 1.0 ranks higher
    assert len(results) == 2
    assert results[0].text == "over_one"
