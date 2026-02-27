"""VRAM-aware model scheduler for GPU resource coordination.

Prevents concurrent heavy model usage (e.g. chat + vision) that would
exceed 32GB VRAM or cause Ollama model-swap thrashing. Uses simple
slot-based mutual exclusion — no GPU memory queries needed.
"""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class ModelScheduler:
    """Gates concurrent GPU-heavy model usage via named slots.

    Slots ``"chat"`` and ``"vision"`` are mutually exclusive — only one
    can be active at a time.  Lightweight models (embeddings, TTS, STT)
    don't need scheduling and should not acquire slots.

    Thread-safe: called from both the sync Pygame loop and async bridge.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: set[str] = set()

    # Heavy slots that block each other
    _HEAVY_SLOTS = frozenset({"chat", "vision"})

    def acquire(self, slot: str) -> bool:
        """Try to acquire a slot. Returns True if granted.

        A heavy slot (chat, vision) is denied if any other heavy slot
        is already active.

        Args:
            slot: Slot name (``"chat"`` or ``"vision"``).

        Returns:
            True if the slot was acquired, False if blocked.
        """
        with self._lock:
            if slot in self._active:
                logger.debug("Slot %r already held, denying double-acquire", slot)
                return False

            if slot in self._HEAVY_SLOTS:
                # Check if any *other* heavy slot is active
                active_heavy = self._active & self._HEAVY_SLOTS
                if active_heavy:
                    logger.debug(
                        "Slot %r blocked by active heavy slot(s): %s",
                        slot, active_heavy,
                    )
                    return False

            self._active.add(slot)
            logger.debug("Slot %r acquired (active: %s)", slot, self._active)
            return True

    def release(self, slot: str) -> None:
        """Release a previously acquired slot.

        Safe to call even if the slot was never acquired (no-op).

        Args:
            slot: Slot name to release.
        """
        with self._lock:
            self._active.discard(slot)
            logger.debug("Slot %r released (active: %s)", slot, self._active)

    def is_active(self, slot: str) -> bool:
        """Check if a slot is currently held.

        Args:
            slot: Slot name to check.

        Returns:
            True if the slot is active.
        """
        with self._lock:
            return slot in self._active
