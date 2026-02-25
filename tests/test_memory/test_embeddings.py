"""Tests for the EmbeddingProvider class."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.config import EmbeddingsConfig, MemoryConfig
from seaman_brain.memory.embeddings import EmbeddingProvider

# --- Fixtures ---


@pytest.fixture
def mock_embed_response():
    """Create a mock EmbedResponse with configurable embeddings."""
    def _make(embeddings: list[list[float]]):
        resp = MagicMock()
        resp.embeddings = embeddings
        return resp
    return _make


@pytest.fixture
def provider():
    """Create an EmbeddingProvider with default config."""
    with patch("seaman_brain.memory.embeddings.AsyncClient"):
        return EmbeddingProvider()


@pytest.fixture
def custom_provider():
    """Create an EmbeddingProvider with custom config."""
    config = MemoryConfig(
        embeddings=EmbeddingsConfig(model="custom-embed-model")
    )
    with patch("seaman_brain.memory.embeddings.AsyncClient"):
        return EmbeddingProvider(config)


# --- Happy Path Tests ---


class TestEmbedHappyPath:
    """Tests for successful single-text embedding."""

    async def test_embed_returns_vector(self, provider, mock_embed_response):
        """embed() returns a list of floats for valid text."""
        expected = [0.1, 0.2, 0.3, 0.4, 0.5]
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([expected])
        )

        result = await provider.embed("hello world")

        assert result == expected
        provider._client.embed.assert_awaited_once_with(
            model="all-minilm:l6-v2",
            input="hello world",
        )

    async def test_embed_uses_configured_model(self, custom_provider, mock_embed_response):
        """embed() uses the model from config."""
        custom_provider._client.embed = AsyncMock(
            return_value=mock_embed_response([[1.0, 2.0]])
        )

        await custom_provider.embed("test")

        custom_provider._client.embed.assert_awaited_once_with(
            model="custom-embed-model",
            input="test",
        )

    async def test_embed_384_dims(self, provider, mock_embed_response):
        """embed() handles full-size 384-dim vectors."""
        expected = [float(i) / 384 for i in range(384)]
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([expected])
        )

        result = await provider.embed("a longer text input")

        assert len(result) == 384
        assert result == expected


class TestEmbedBatchHappyPath:
    """Tests for successful batch embedding."""

    async def test_embed_batch_multiple_texts(self, provider, mock_embed_response):
        """embed_batch() returns one vector per input text."""
        vectors = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response(vectors)
        )

        result = await provider.embed_batch(["one", "two", "three"])

        assert len(result) == 3
        assert result == vectors
        provider._client.embed.assert_awaited_once_with(
            model="all-minilm:l6-v2",
            input=["one", "two", "three"],
        )

    async def test_embed_batch_single_text(self, provider, mock_embed_response):
        """embed_batch() works with a single text."""
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([[0.1, 0.2]])
        )

        result = await provider.embed_batch(["solo"])

        assert len(result) == 1
        assert result[0] == [0.1, 0.2]


# --- Edge Case Tests ---


class TestEmbedEdgeCases:
    """Tests for edge cases in single-text embedding."""

    async def test_embed_empty_string(self, provider):
        """embed() returns empty list for empty string."""
        result = await provider.embed("")
        assert result == []

    async def test_embed_whitespace_only(self, provider):
        """embed() returns empty list for whitespace-only string."""
        result = await provider.embed("   \n\t  ")
        assert result == []

    async def test_embed_empty_response(self, provider, mock_embed_response):
        """embed() returns empty list when server returns empty embeddings."""
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([])
        )

        result = await provider.embed("something")

        assert result == []

    async def test_embed_empty_inner_vector(self, provider, mock_embed_response):
        """embed() returns empty list when server returns empty inner vector."""
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([[]])
        )

        result = await provider.embed("something")

        assert result == []

    async def test_embed_long_text(self, provider, mock_embed_response):
        """embed() handles very long text without issues."""
        long_text = "word " * 10000
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([[0.1, 0.2]])
        )

        result = await provider.embed(long_text)

        assert result == [0.1, 0.2]


class TestEmbedBatchEdgeCases:
    """Tests for edge cases in batch embedding."""

    async def test_embed_batch_empty_list(self, provider):
        """embed_batch() returns empty list for empty input."""
        result = await provider.embed_batch([])
        assert result == []

    async def test_embed_batch_all_empty_strings(self, provider):
        """embed_batch() returns empty vectors for all-empty inputs."""
        result = await provider.embed_batch(["", "  ", "\n"])
        assert result == [[], [], []]

    async def test_embed_batch_mixed_empty(self, provider, mock_embed_response):
        """embed_batch() handles mix of real and empty texts."""
        provider._client.embed = AsyncMock(
            return_value=mock_embed_response([[0.1, 0.2], [0.3, 0.4]])
        )

        result = await provider.embed_batch(["hello", "", "world", "  "])

        assert len(result) == 4
        assert result[0] == [0.1, 0.2]
        assert result[1] == []
        assert result[2] == [0.3, 0.4]
        assert result[3] == []
        # Only non-empty texts sent to server
        provider._client.embed.assert_awaited_once_with(
            model="all-minilm:l6-v2",
            input=["hello", "world"],
        )


# --- Error Handling Tests ---


class TestEmbedErrorHandling:
    """Tests for error scenarios."""

    async def test_embed_connection_error(self, provider):
        """embed() wraps server errors as ConnectionError."""
        provider._client.embed = AsyncMock(
            side_effect=OSError("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to get embeddings"):
            await provider.embed("test")

    async def test_embed_connection_error_chained(self, provider):
        """embed() preserves the original exception as __cause__."""
        original = OSError("Connection refused")
        provider._client.embed = AsyncMock(side_effect=original)

        with pytest.raises(ConnectionError) as exc_info:
            await provider.embed("test")

        assert exc_info.value.__cause__ is original

    async def test_embed_connection_error_includes_url(self, provider):
        """embed() includes the base URL in the error message."""
        provider._client.embed = AsyncMock(
            side_effect=RuntimeError("timeout")
        )

        with pytest.raises(ConnectionError, match="localhost:11434"):
            await provider.embed("test")

    async def test_embed_batch_connection_error(self, provider):
        """embed_batch() wraps server errors as ConnectionError."""
        provider._client.embed = AsyncMock(
            side_effect=OSError("Connection refused")
        )

        with pytest.raises(ConnectionError, match="Failed to get embeddings"):
            await provider.embed_batch(["one", "two"])

    async def test_embed_batch_connection_error_chained(self, provider):
        """embed_batch() preserves the original exception as __cause__."""
        original = TimeoutError("timed out")
        provider._client.embed = AsyncMock(side_effect=original)

        with pytest.raises(ConnectionError) as exc_info:
            await provider.embed_batch(["test"])

        assert exc_info.value.__cause__ is original


# --- Configuration Tests ---


class TestProviderConfig:
    """Tests for provider configuration."""

    def test_default_model(self, provider):
        """Default model is all-minilm:l6-v2."""
        assert provider.model == "all-minilm:l6-v2"

    def test_custom_model(self, custom_provider):
        """Custom model from config is used."""
        assert custom_provider.model == "custom-embed-model"

    def test_default_config_when_none(self):
        """Provider creates default config when None passed."""
        with patch("seaman_brain.memory.embeddings.AsyncClient"):
            p = EmbeddingProvider(None)
        assert p.model == "all-minilm:l6-v2"
