"""Hybrid memory retriever combining similarity and recency.

Retrieves memories from the semantic store and re-ranks them using a
weighted combination of vector similarity and temporal recency scores.
"""

from __future__ import annotations

from datetime import UTC, datetime

from seaman_brain.config import MemoryConfig
from seaman_brain.memory.embeddings import EmbeddingProvider
from seaman_brain.memory.semantic import SemanticMemory
from seaman_brain.types import MemoryRecord


class HybridRetriever:
    """Combines semantic similarity with temporal recency for memory retrieval.

    Embeds the query, searches the vector store, then re-ranks results
    using configurable weights for similarity vs recency.
    """

    def __init__(
        self,
        semantic: SemanticMemory,
        embeddings: EmbeddingProvider,
        config: MemoryConfig | None = None,
    ) -> None:
        cfg = config or MemoryConfig()
        self._semantic = semantic
        self._embeddings = embeddings
        self._similarity_weight = cfg.similarity_weight
        self._recency_weight = cfg.recency_weight

    async def retrieve(
        self, query: str, top_k: int = 5
    ) -> list[MemoryRecord]:
        """Retrieve memories ranked by weighted similarity + recency.

        Args:
            query: The text to search for.
            top_k: Maximum number of results to return.

        Returns:
            List of MemoryRecords ranked by combined score (highest first).
        """
        if top_k <= 0:
            return []

        embedding = await self._embeddings.embed(query)
        if not embedding:
            return []

        # Fetch more candidates than needed so re-ranking can reorder
        fetch_k = min(top_k * 3, 50)
        candidates = await self._semantic.search(embedding, top_k=fetch_k)
        if not candidates:
            return []

        scored = self._score_candidates(candidates)
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [record for record, _ in scored[:top_k]]

    def _score_candidates(
        self, candidates: list[MemoryRecord]
    ) -> list[tuple[MemoryRecord, float]]:
        """Compute weighted combination scores for each candidate.

        Similarity score is derived from the candidate's importance field
        (which proxies for retrieval rank from vector search).
        Recency score is computed from the timestamp relative to the
        newest and oldest candidate.
        """
        if not candidates:
            return []

        now = datetime.now(UTC)
        ages = [(now - r.timestamp).total_seconds() for r in candidates]
        max_age = max(ages) if ages else 1.0
        # Avoid division by zero when all candidates have the same timestamp
        if max_age == 0:
            max_age = 1.0

        scored: list[tuple[MemoryRecord, float]] = []
        for record, age in zip(candidates, ages):
            # Similarity: use importance as a normalized [0,1] proxy
            sim_score = max(0.0, min(1.0, record.importance))
            # Recency: newer = higher score
            recency_score = 1.0 - (age / max_age)
            combined = (
                self._similarity_weight * sim_score
                + self._recency_weight * recency_score
            )
            scored.append((record, combined))

        return scored
