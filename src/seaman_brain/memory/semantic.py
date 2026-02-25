"""LanceDB semantic vector memory store.

Persistent long-term memory using LanceDB for vector similarity search.
Stores MemoryRecord entries with embeddings and retrieves similar memories
based on cosine distance.
"""

from __future__ import annotations

from datetime import datetime

import lancedb
import numpy as np
import pyarrow as pa

from seaman_brain.config import MemoryConfig
from seaman_brain.types import MemoryRecord

TABLE_NAME = "memories"


def _build_schema(vector_dims: int) -> pa.Schema:
    """Build the PyArrow schema for the memories table."""
    return pa.schema([
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), vector_dims)),
        pa.field("timestamp", pa.string()),
        pa.field("importance", pa.float32()),
        pa.field("source", pa.string()),
    ])


class SemanticMemory:
    """Persistent vector memory store backed by LanceDB.

    Stores MemoryRecord entries with their embedding vectors and supports
    similarity search for retrieving relevant memories.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        cfg = config or MemoryConfig()
        self._db_path = cfg.db_path
        self._vector_dims = cfg.vector_dims
        self._top_k = cfg.top_k
        self._db: lancedb.AsyncConnection | None = None
        self._table: lancedb.AsyncTable | None = None

    async def _ensure_table(self) -> lancedb.AsyncTable:
        """Lazily connect to DB and create/open the memories table."""
        if self._table is not None:
            return self._table

        self._db = await lancedb.connect_async(self._db_path)
        tables_response = await self._db.list_tables()
        table_names = tables_response.tables

        if TABLE_NAME in table_names:
            self._table = await self._db.open_table(TABLE_NAME)
        else:
            schema = _build_schema(self._vector_dims)
            self._table = await self._db.create_table(TABLE_NAME, schema=schema)

        return self._table

    async def add(self, record: MemoryRecord) -> None:
        """Store a memory record with its embedding vector.

        Args:
            record: The MemoryRecord to store. Must have a non-empty embedding
                    matching the configured vector_dims.

        Raises:
            ValueError: If the embedding is empty or has wrong dimensions.
        """
        embedding = record.embedding.tolist()
        if not embedding:
            raise ValueError("Cannot store a memory with an empty embedding.")
        if len(embedding) != self._vector_dims:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._vector_dims}, "
                f"got {len(embedding)}."
            )

        table = await self._ensure_table()
        await table.add([{
            "text": record.text,
            "vector": embedding,
            "timestamp": record.timestamp.isoformat(),
            "importance": float(record.importance),
            "source": record.source,
        }])

    async def search(
        self, vector: list[float], top_k: int | None = None
    ) -> list[MemoryRecord]:
        """Search for similar memories by vector distance.

        Args:
            vector: The query embedding vector.
            top_k: Maximum number of results to return. Defaults to config top_k.

        Returns:
            List of MemoryRecord sorted by similarity (closest first).

        Raises:
            ValueError: If the query vector is empty.
        """
        if not vector:
            raise ValueError("Cannot search with an empty vector.")

        k = top_k if top_k is not None else self._top_k
        if k <= 0:
            return []

        table = await self._ensure_table()
        row_count = await table.count_rows()
        if row_count == 0:
            return []

        query = await table.search(vector)
        results = await query.limit(k).to_list()

        records: list[MemoryRecord] = []
        for row in results:
            records.append(MemoryRecord(
                text=row["text"],
                embedding=np.array(row["vector"], dtype=np.float32),
                timestamp=datetime.fromisoformat(row["timestamp"]),
                importance=float(row["importance"]),
                source=row["source"],
            ))
        return records

    async def count(self) -> int:
        """Return the number of stored memories."""
        table = await self._ensure_table()
        return await table.count_rows()

    async def delete_all(self) -> None:
        """Delete all stored memories."""
        table = await self._ensure_table()
        count = await table.count_rows()
        if count > 0:
            await table.delete("true")
