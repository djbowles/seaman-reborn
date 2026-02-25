"""Rolling episodic memory buffer.

Short-term conversation memory that maintains the most recent messages
within a configurable window, automatically evicting oldest entries when full.
"""

from __future__ import annotations

from collections import deque

from seaman_brain.types import ChatMessage


class EpisodicMemory:
    """Fixed-size rolling buffer for short-term conversation context.

    Stores the most recent ChatMessages up to max_size, automatically
    discarding the oldest when the buffer is full.
    """

    def __init__(self, max_size: int = 20) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._buffer: deque[ChatMessage] = deque(maxlen=max_size)
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Maximum number of messages the buffer can hold."""
        return self._max_size

    def add(self, message: ChatMessage) -> None:
        """Append a message to the buffer, evicting the oldest if full."""
        self._buffer.append(message)

    def get_recent(self, n: int) -> list[ChatMessage]:
        """Return the last n messages in chronological order.

        If n exceeds the current buffer length, returns all messages.
        If n <= 0, returns an empty list.
        """
        if n <= 0:
            return []
        return list(self._buffer)[-n:]

    def get_all(self) -> list[ChatMessage]:
        """Return all messages in chronological order."""
        return list(self._buffer)

    def clear(self) -> None:
        """Remove all messages from the buffer."""
        self._buffer.clear()

    def __len__(self) -> int:
        """Return the current number of messages in the buffer."""
        return len(self._buffer)
