"""Tests for the SemanticMemory class (LanceDB vector store)."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from seaman_brain.config import MemoryConfig
from seaman_brain.memory.semantic import SemanticMemory
from seaman_brain.types import MemoryRecord

# --- Fixtures ---

VECTOR_DIMS = 384


def _make_record(
    text: str = "test memory",
    importance: float = 0.5,
    source: str = "test",
    embedding: np.ndarray | None = None,
) -> MemoryRecord:
    """Create a MemoryRecord with a random or specified embedding."""
    if embedding is None:
        embedding = np.random.default_rng(42).random(VECTOR_DIMS).astype(np.float32)
    return MemoryRecord(
        text=text,
        embedding=embedding,
        timestamp=datetime.now(UTC),
        importance=importance,
        source=source,
    )


@pytest.fixture
def db_config(tmp_path) -> MemoryConfig:
    """MemoryConfig pointing at a temporary LanceDB directory."""
    db_dir = tmp_path / "lancedb"
    db_dir.mkdir()
    return MemoryConfig(
        db_path=str(db_dir),
        vector_dims=VECTOR_DIMS,
        top_k=5,
    )


@pytest.fixture
def store(db_config: MemoryConfig) -> SemanticMemory:
    """A fresh SemanticMemory instance backed by a tmp directory."""
    return SemanticMemory(db_config)


# ============================================================
# Happy-path tests
# ============================================================


class TestSemanticMemoryHappyPath:
    """Core operations: add, search, count."""

    async def test_add_and_count(self, store: SemanticMemory) -> None:
        """Adding a record increments the row count."""
        assert await store.count() == 0
        await store.add(_make_record("first memory"))
        assert await store.count() == 1
        await store.add(_make_record("second memory"))
        assert await store.count() == 2

    async def test_search_returns_similar(self, store: SemanticMemory) -> None:
        """Search returns the closest record to the query vector."""
        rng = np.random.default_rng(123)
        vec_a = rng.random(VECTOR_DIMS).astype(np.float32)
        vec_b = rng.random(VECTOR_DIMS).astype(np.float32)

        await store.add(_make_record("memory A", embedding=vec_a))
        await store.add(_make_record("memory B", embedding=vec_b))

        results = await store.search(vec_a.tolist(), top_k=1)
        assert len(results) == 1
        assert results[0].text == "memory A"

    async def test_search_returns_memory_records(self, store: SemanticMemory) -> None:
        """Search results are proper MemoryRecord objects with all fields."""
        rec = _make_record("detailed record", importance=0.9, source="conversation")
        await store.add(rec)

        results = await store.search(rec.embedding.tolist(), top_k=1)
        assert len(results) == 1
        result = results[0]
        assert isinstance(result, MemoryRecord)
        assert result.text == "detailed record"
        assert result.importance == pytest.approx(0.9, abs=1e-5)
        assert result.source == "conversation"
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.embedding, np.ndarray)

    async def test_search_respects_top_k(self, store: SemanticMemory) -> None:
        """Search returns at most top_k results."""
        rng = np.random.default_rng(0)
        for i in range(10):
            vec = rng.random(VECTOR_DIMS).astype(np.float32)
            await store.add(_make_record(f"mem-{i}", embedding=vec))

        query = rng.random(VECTOR_DIMS).astype(np.float32)
        results = await store.search(query.tolist(), top_k=3)
        assert len(results) == 3

    async def test_search_uses_config_top_k(self, store: SemanticMemory) -> None:
        """When top_k is not passed, uses the config default (5)."""
        rng = np.random.default_rng(7)
        for i in range(10):
            vec = rng.random(VECTOR_DIMS).astype(np.float32)
            await store.add(_make_record(f"mem-{i}", embedding=vec))

        query = rng.random(VECTOR_DIMS).astype(np.float32)
        results = await store.search(query.tolist())
        assert len(results) == 5  # config top_k

    async def test_delete_all(self, store: SemanticMemory) -> None:
        """delete_all removes all records from the store."""
        for i in range(3):
            await store.add(_make_record(f"mem-{i}"))
        assert await store.count() == 3
        await store.delete_all()
        assert await store.count() == 0


# ============================================================
# Edge-case tests
# ============================================================


class TestSemanticMemoryEdgeCases:
    """Boundary conditions and unusual inputs."""

    async def test_search_empty_store(self, store: SemanticMemory) -> None:
        """Searching an empty store returns an empty list."""
        query = np.random.default_rng(0).random(VECTOR_DIMS).astype(np.float32)
        results = await store.search(query.tolist())
        assert results == []

    async def test_search_top_k_zero(self, store: SemanticMemory) -> None:
        """Requesting top_k=0 returns an empty list."""
        await store.add(_make_record("some memory"))
        results = await store.search([0.1] * VECTOR_DIMS, top_k=0)
        assert results == []

    async def test_search_top_k_exceeds_rows(self, store: SemanticMemory) -> None:
        """Requesting more results than stored returns all stored."""
        await store.add(_make_record("only one"))
        query = np.random.default_rng(0).random(VECTOR_DIMS).astype(np.float32)
        results = await store.search(query.tolist(), top_k=100)
        assert len(results) == 1

    async def test_default_config(self, tmp_path) -> None:
        """SemanticMemory works with default MemoryConfig."""
        cfg = MemoryConfig(db_path=str(tmp_path / "default_db"))
        sm = SemanticMemory(cfg)
        assert await sm.count() == 0

    async def test_delete_all_on_empty(self, store: SemanticMemory) -> None:
        """delete_all on empty store does not raise."""
        await store.delete_all()
        assert await store.count() == 0

    async def test_table_reuse_across_calls(self, db_config: MemoryConfig) -> None:
        """Table is lazily created once, then reused on subsequent operations."""
        store = SemanticMemory(db_config)
        await store.add(_make_record("first"))
        # Second instance against same db_path should find existing table
        store2 = SemanticMemory(db_config)
        assert await store2.count() == 1

    async def test_add_preserves_timestamp(self, store: SemanticMemory) -> None:
        """Stored timestamp round-trips correctly through ISO format."""
        ts = datetime(2025, 6, 15, 12, 30, 45, tzinfo=UTC)
        rec = _make_record("timestamped")
        rec.timestamp = ts
        await store.add(rec)

        results = await store.search(rec.embedding.tolist(), top_k=1)
        assert results[0].timestamp == ts


# ============================================================
# Error-handling tests
# ============================================================


class TestSemanticMemoryErrors:
    """Invalid inputs and failure conditions."""

    async def test_add_empty_embedding(self, store: SemanticMemory) -> None:
        """Adding a record with an empty embedding raises ValueError."""
        rec = _make_record("bad record", embedding=np.array([], dtype=np.float32))
        with pytest.raises(ValueError, match="empty embedding"):
            await store.add(rec)

    async def test_add_wrong_dimensions(self, store: SemanticMemory) -> None:
        """Adding a record with wrong embedding dimensions raises ValueError."""
        wrong_vec = np.ones(128, dtype=np.float32)  # expected 384
        rec = _make_record("wrong dims", embedding=wrong_vec)
        with pytest.raises(ValueError, match="dimension mismatch"):
            await store.add(rec)

    async def test_search_empty_vector(self, store: SemanticMemory) -> None:
        """Searching with an empty vector raises ValueError."""
        with pytest.raises(ValueError, match="empty vector"):
            await store.search([])
