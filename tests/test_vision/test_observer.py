"""Tests for VisionObserver — vision LLM caller."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from seaman_brain.config import VisionConfig
from seaman_brain.vision.observer import _TANK_PROMPT, _WEBCAM_PROMPT, VisionObserver


@pytest.fixture
def config():
    """Default VisionConfig."""
    return VisionConfig(vision_model="test-vl:8b")


@pytest.fixture
def observer(config):
    """VisionObserver with test config."""
    return VisionObserver(config)


def _make_ollama_mock(response_text: str) -> MagicMock:
    """Create a mock ollama module with a configured AsyncClient."""
    mock_response = MagicMock()
    mock_response.message.content = response_text

    mock_client = AsyncMock()
    mock_client.chat = AsyncMock(return_value=mock_response)

    mock_ollama = MagicMock()
    mock_ollama.AsyncClient.return_value = mock_client
    return mock_ollama, mock_client


class TestObserveHappyPath:
    """Tests for successful vision observations."""

    async def test_observe_webcam_returns_text(self, observer):
        """Webcam observation returns the model's text response."""
        mock_ollama, mock_client = _make_ollama_mock(
            "The human looks tired and bored."
        )

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await observer.observe(b"fake-jpeg-bytes", source="webcam")

        assert result == "The human looks tired and bored."
        mock_client.chat.assert_awaited_once()

        # Verify webcam prompt was used
        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert _WEBCAM_PROMPT in messages[0]["content"]

    async def test_observe_tank_uses_tank_prompt(self, observer):
        """Tank observation uses the tank-specific prompt."""
        mock_ollama, mock_client = _make_ollama_mock(
            "The tank water is slightly murky."
        )

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await observer.observe(b"fake-jpeg-bytes", source="tank")

        assert result == "The tank water is slightly murky."

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert _TANK_PROMPT in messages[0]["content"]

    async def test_observe_strips_whitespace(self, observer):
        """Observation text is stripped of leading/trailing whitespace."""
        mock_ollama, _ = _make_ollama_mock("  The human is typing.  \n")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await observer.observe(b"fake-jpeg", source="webcam")

        assert result == "The human is typing."

    async def test_observe_sends_correct_model(self, config, observer):
        """The configured vision model is passed to ollama."""
        mock_ollama, mock_client = _make_ollama_mock("Something.")

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            await observer.observe(b"bytes", source="webcam")

        call_args = mock_client.chat.call_args
        model = call_args.kwargs.get("model") or call_args[1].get("model")
        assert model == "test-vl:8b"

    async def test_observe_sends_images(self, observer):
        """The image bytes are included in the message."""
        mock_ollama, mock_client = _make_ollama_mock("Observed.")

        frame = b"jpeg-data-here"
        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            await observer.observe(frame, source="webcam")

        call_args = mock_client.chat.call_args
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert frame in messages[0]["images"]


class TestObserveErrorHandling:
    """Tests for error conditions."""

    async def test_observe_empty_bytes_returns_empty(self, observer):
        """Empty frame bytes returns empty string without calling LLM."""
        result = await observer.observe(b"", source="webcam")
        assert result == ""

    async def test_observe_ollama_error_returns_empty(self, observer):
        """Ollama connection error returns empty string."""
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(side_effect=ConnectionError("Ollama offline"))

        mock_ollama = MagicMock()
        mock_ollama.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await observer.observe(b"bytes", source="webcam")

        assert result == ""

    async def test_observe_import_error_returns_empty(self, observer):
        """Missing ollama package returns empty string."""
        with patch.dict("sys.modules", {"ollama": None}):
            result = await observer.observe(b"bytes", source="webcam")
        assert result == ""

    async def test_observe_none_content_returns_empty(self, observer):
        """None content from model returns empty string."""
        mock_ollama, _ = _make_ollama_mock(None)
        # Override to return None content
        mock_response = MagicMock()
        mock_response.message.content = None
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_ollama.AsyncClient.return_value = mock_client

        with patch.dict("sys.modules", {"ollama": mock_ollama}):
            result = await observer.observe(b"bytes", source="webcam")

        assert result == ""
