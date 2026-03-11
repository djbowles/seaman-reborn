"""Response deduplication to prevent repetitive LLM outputs.

Small 8B models (e.g. qwen3:8b) frequently produce near-identical responses
for similar stimuli.  ``ResponseDeduplicator`` keeps a ring buffer of recent
responses and detects duplicates via prefix matching and sequence similarity,
enabling the conversation manager to request a varied retry.
"""

from __future__ import annotations

import re
from collections import deque
from difflib import SequenceMatcher

_PREFIX_LENGTH = 80
_SIMILARITY_THRESHOLD = 0.80
_DEFAULT_BUFFER_SIZE = 20

_WS_RE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for comparison."""
    return _WS_RE.sub(" ", text.lower()).strip()


class ResponseDeduplicator:
    """Detects near-duplicate LLM responses using a ring buffer.

    Two detection strategies are applied in order:

    1. **Prefix match** — the first 80 characters (after normalisation) are
       compared.  This catches the very common case where the model picks
       the exact same sentence opener.
    2. **Sequence similarity** — ``difflib.SequenceMatcher.ratio()`` above
       0.80 flags responses that are paraphrases with minor wording changes.

    Parameters
    ----------
    max_size:
        Maximum number of recent responses to retain.  Oldest entries are
        evicted when the buffer is full.
    """

    def __init__(self, max_size: int = _DEFAULT_BUFFER_SIZE) -> None:
        self._max_size = max_size
        self._buffer: deque[str] = deque(maxlen=max_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_duplicate(self, text: str) -> bool:
        """Return ``True`` if *text* is too similar to a recent response.

        Args:
            text: The candidate response to check.

        Returns:
            Whether the response is a duplicate of a buffered entry.
        """
        if not text or not text.strip():
            return False

        norm = _normalize(text)
        prefix = norm[:_PREFIX_LENGTH]

        for stored in self._buffer:
            stored_norm = _normalize(stored)
            # Strategy 1 — prefix match
            if prefix and stored_norm[:_PREFIX_LENGTH] == prefix:
                return True
            # Strategy 2 — sequence similarity
            if SequenceMatcher(None, norm, stored_norm).ratio() > _SIMILARITY_THRESHOLD:
                return True

        return False

    def record(self, text: str) -> None:
        """Add *text* to the ring buffer.

        Args:
            text: The response to remember.
        """
        if text and text.strip():
            self._buffer.append(text)

    def make_vary_instruction(self, previous_response: str = "") -> str:
        """Return a system-level instruction asking the LLM to vary its output.

        Args:
            previous_response: When provided, the instruction explicitly quotes
                the first 100 characters so the model has concrete text to avoid.

        Returns:
            A non-empty instruction string.
        """
        if previous_response.strip():
            snippet = previous_response[:100]
            ellipsis = "..." if len(previous_response) > 100 else ""
            return (
                f"Your previous response was: '{snippet}{ellipsis}'. "
                "You MUST NOT repeat this. Say something completely different — "
                "use different words, a different perspective, and a different "
                "sentence structure."
            )
        return (
            "Your previous responses have been repetitive. "
            "Vary your phrasing, perspective, and vocabulary significantly. "
            "Do NOT reuse sentence structures or metaphors from earlier replies."
        )

    @property
    def buffer_size(self) -> int:
        """Current number of responses in the buffer."""
        return len(self._buffer)
