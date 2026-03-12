"""Streaming chat response handler — TTS sentence splitting + pending management.

Extracts the async response pipeline from GameEngine: queue-based token
streaming from the ConversationManager async thread, incremental TTS at
sentence/clause boundaries, and timeout-guarded pending future management.
"""

from __future__ import annotations

import logging
import queue
import time
from typing import Any

from seaman_brain.gui.game_systems import _PENDING_TIMEOUT, find_tts_split

logger = logging.getLogger(__name__)


class ResponseHandler:
    """Manages streaming LLM responses, TTS splitting, and pending futures.

    Args:
        chat_panel: ChatPanel instance for displaying messages.
        audio_bridge: PygameAudioBridge for TTS playback (may be None).
        scheduler: ModelScheduler for GPU slot management (may be None).
    """

    def __init__(
        self,
        *,
        chat_panel: Any,
        audio_bridge: Any | None = None,
        scheduler: Any | None = None,
    ) -> None:
        self._chat = chat_panel
        self._audio = audio_bridge
        self._scheduler = scheduler

        # Stream state
        self._stream_queue: queue.Queue[str | None] = queue.Queue()
        self._stream_accumulated: str = ""
        self._tts_buffer: str = ""
        self._stream_complete: bool = False

        # Pending user-chat future
        self._pending: Any | None = None
        self._pending_time: float = 0.0

        # Pending autonomous-remark future
        self._pending_auto: Any | None = None
        self._pending_auto_time: float = 0.0
        self._pending_auto_behavior: Any | None = None

    @property
    def is_busy(self) -> bool:
        """Whether a user-chat response is in flight."""
        return self._pending is not None

    @property
    def is_auto_busy(self) -> bool:
        """Whether an autonomous remark is in flight."""
        return self._pending_auto is not None

    # ── Stream management ─────────────────────────────────────────────

    def start_stream(self) -> None:
        """Reset stream state for a new response."""
        # Drain any leftover tokens
        while not self._stream_queue.empty():
            try:
                self._stream_queue.get_nowait()
            except queue.Empty:
                break
        self._stream_accumulated = ""
        self._tts_buffer = ""
        self._stream_complete = False

    def put_token(self, token: str | None) -> None:
        """Feed a token from the async thread (None = sentinel)."""
        self._stream_queue.put(token)

    def drain_stream(self) -> None:
        """Drain queued tokens into the chat panel and TTS buffer."""
        while not self._stream_queue.empty():
            try:
                token = self._stream_queue.get_nowait()
                if token is None:
                    self._stream_complete = True
                else:
                    self._stream_accumulated += token
                    self._chat.update_streaming(self._stream_accumulated)
                    self._tts_buffer += token
            except queue.Empty:
                break

        # Speak complete sentences as they arrive
        if self._audio is not None:
            while True:
                pos = find_tts_split(self._tts_buffer)
                if pos is None:
                    break
                sentence = self._tts_buffer[:pos].strip()
                self._tts_buffer = self._tts_buffer[pos:]
                if sentence:
                    self._audio.play_voice(sentence)

    # ── Pending user-chat ─────────────────────────────────────────────

    def start_response(self, future: Any) -> None:
        """Track a new pending user-chat future."""
        self._pending = future
        self._pending_time = time.monotonic()

    def cancel_response(self) -> None:
        """Cancel the in-flight user-chat response."""
        if self._pending is not None:
            self._pending.cancel()
        self._cleanup_response()

    def check_pending(self) -> None:
        """Poll the pending user-chat future — timeout, complete, or wait."""
        if self._pending is None:
            return

        self.drain_stream()

        if not self._pending.done():
            if time.monotonic() - self._pending_time > _PENDING_TIMEOUT:
                logger.warning("Pending chat response timed out")
                self._pending.cancel()
                self._chat.finish_streaming()
                self._chat.add_message(
                    "creature", "*yawns* ...lost my train of thought.",
                )
                self._cleanup_response()
            return

        if self._pending.cancelled():
            self._cleanup_response()
            return

        try:
            result = self._pending.result(timeout=0)
            had_stream = bool(self._stream_accumulated)
            self._chat.finish_streaming()
            if not had_stream and result:
                self._chat.add_message("creature", result)
            remaining = self._tts_buffer.strip()
            if remaining and self._audio is not None:
                self._audio.play_voice(remaining)
            elif not remaining and result and self._audio is not None:
                if not had_stream:
                    self._audio.play_voice(result)
        except Exception as exc:
            self._chat.finish_streaming()
            self._chat.add_message("creature", f"*glitches* {exc}")
            logger.error("Conversation error: %s", exc, exc_info=True)
        finally:
            self._cleanup_response()

    def _cleanup_response(self) -> None:
        """Reset all user-chat response state."""
        self._tts_buffer = ""
        self._stream_accumulated = ""
        self._pending = None
        if self._scheduler is not None:
            self._scheduler.release("chat")

    # ── Pending autonomous remark ─────────────────────────────────────

    def start_autonomous(self, future: Any, behavior: Any = None) -> None:
        """Track a pending autonomous LLM remark."""
        self._pending_auto = future
        self._pending_auto_time = time.monotonic()
        self._pending_auto_behavior = behavior

    def cancel_autonomous(self) -> None:
        """Cancel in-flight autonomous remark."""
        if self._pending_auto is not None:
            self._pending_auto.cancel()
        self._cleanup_autonomous()

    def check_pending_autonomous(self) -> tuple[str | None, Any | None]:
        """Poll the autonomous remark future.

        Returns:
            (result_text, fallback_behavior) — text if ready, behavior
            if it should be applied as fallback, or (None, None) if still pending.
        """
        if self._pending_auto is None:
            return None, None

        if not self._pending_auto.done():
            if time.monotonic() - self._pending_auto_time > _PENDING_TIMEOUT:
                logger.warning("Pending autonomous remark timed out")
                self._pending_auto.cancel()
                behavior = self._pending_auto_behavior
                self._cleanup_autonomous()
                return None, behavior
            return None, None

        if self._pending_auto.cancelled():
            self._cleanup_autonomous()
            return None, None

        behavior = self._pending_auto_behavior
        try:
            result = self._pending_auto.result(timeout=0)
            self._cleanup_autonomous()
            if result:
                return result, None
            return None, behavior
        except Exception as exc:
            logger.warning("Autonomous remark failed: %s", exc)
            self._cleanup_autonomous()
            return None, behavior

    def _cleanup_autonomous(self) -> None:
        """Reset autonomous remark state."""
        self._pending_auto = None
        self._pending_auto_behavior = None
        if self._scheduler is not None:
            self._scheduler.release("chat")
