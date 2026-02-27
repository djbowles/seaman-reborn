"""Vision LLM caller — sends frames to a vision-language model.

VisionObserver takes raw image bytes, sends them to the configured
vision model via ``ollama.AsyncClient``, and returns a short text
description of what the creature "sees."

This module has no dependency on any game subsystem — pure async I/O.
"""

from __future__ import annotations

import logging

from seaman_brain.config import VisionConfig

logger = logging.getLogger(__name__)

_WEBCAM_PROMPT = (
    "You are looking through the eyes of a creature in a tank at the human "
    "who owns you. Describe what you see in 1-2 sentences — their expression, "
    "what they're doing, their environment."
)

_TANK_PROMPT = (
    "You are a creature looking at your own tank. Describe the current state "
    "in 1-2 sentences — water clarity, decorations, overall condition."
)


class VisionObserver:
    """Calls a vision-language model to describe image frames.

    Produces short text observations from either webcam or tank captures.
    Observations are plain strings ready for injection into the system prompt.

    Reuses a single ``AsyncClient`` instance across calls to avoid
    creating a new connection per observation.
    """

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._client: object | None = None

    def _get_client(self):
        """Lazily create and return a persistent AsyncClient."""
        if self._client is None:
            import ollama

            self._client = ollama.AsyncClient()
        return self._client

    async def observe(self, frame_bytes: bytes, source: str = "webcam") -> str:
        """Send an image frame to the vision model and get a text description.

        Args:
            frame_bytes: Raw image bytes (JPEG or BMP).
            source: ``"webcam"`` or ``"tank"`` — determines the prompt.

        Returns:
            A short text description, or ``""`` on error.
        """
        if not frame_bytes:
            return ""

        prompt = _WEBCAM_PROMPT if source == "webcam" else _TANK_PROMPT

        try:
            client = self._get_client()
            response = await client.chat(
                model=self._config.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [frame_bytes],
                    }
                ],
            )
            text = response.message.content or ""
            logger.debug("Vision observation (%s): %s", source, text[:80])
            return text.strip()
        except Exception as exc:
            logger.warning("Vision observation failed (%s): %s", source, exc)
            return ""
