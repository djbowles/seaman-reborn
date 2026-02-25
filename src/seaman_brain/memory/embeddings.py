"""Embedding provider for vector operations.

Wraps the Ollama embeddings API to generate text embeddings
for semantic memory storage and retrieval.
"""

from __future__ import annotations

from ollama import AsyncClient

from seaman_brain.config import MemoryConfig


class EmbeddingProvider:
    """Generates text embeddings via Ollama's embedding API.

    Uses a configurable embedding model (default: all-minilm:l6-v2)
    to convert text into dense vector representations for similarity search.
    """

    def __init__(self, config: MemoryConfig | None = None) -> None:
        cfg = config or MemoryConfig()
        self.model = cfg.embeddings.model
        self.base_url = "http://localhost:11434"
        self._client = AsyncClient(host=self.base_url)

    async def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text.

        Args:
            text: The text to embed. Empty strings produce a zero vector.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
        """
        if not text or not text.strip():
            return []

        try:
            response = await self._client.embed(
                model=self.model,
                input=text,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to get embeddings from Ollama at {self.base_url}: {exc}"
            ) from exc

        embeddings = response.embeddings
        if not embeddings or not embeddings[0]:
            return []
        return list(embeddings[0])

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts.

        Args:
            texts: List of texts to embed. Empty strings produce empty vectors.

        Returns:
            A list of embedding vectors, one per input text.

        Raises:
            ConnectionError: If the Ollama server is unreachable.
        """
        if not texts:
            return []

        # Separate empty/whitespace texts from real texts
        non_empty_indices: list[int] = []
        non_empty_texts: list[str] = []
        for i, t in enumerate(texts):
            if t and t.strip():
                non_empty_indices.append(i)
                non_empty_texts.append(t)

        # If all texts are empty, return empty vectors for each
        if not non_empty_texts:
            return [[] for _ in texts]

        try:
            response = await self._client.embed(
                model=self.model,
                input=non_empty_texts,
            )
        except Exception as exc:
            raise ConnectionError(
                f"Failed to get embeddings from Ollama at {self.base_url}: {exc}"
            ) from exc

        # Build result with empty vectors for empty inputs
        result: list[list[float]] = [[] for _ in texts]
        embeddings = response.embeddings
        for idx, emb_idx in enumerate(non_empty_indices):
            if idx < len(embeddings):
                result[emb_idx] = list(embeddings[idx])

        return result
