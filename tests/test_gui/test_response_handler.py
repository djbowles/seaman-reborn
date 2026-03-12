"""Tests for ResponseHandler — streaming chat + TTS splitting."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.response_handler import ResponseHandler  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield


def _make_handler(**kw):
    return ResponseHandler(
        chat_panel=kw.get("chat", MagicMock()),
        audio_bridge=kw.get("audio", None),
        scheduler=kw.get("scheduler", None),
    )


class TestStreamDrain:
    def test_drains_tokens_from_queue(self):
        chat = MagicMock()
        handler = _make_handler(chat=chat)
        handler.start_stream()
        handler.put_token("Hello ")
        handler.put_token("world")
        handler.drain_stream()
        chat.update_streaming.assert_called()
        assert "Hello world" in handler._stream_accumulated

    def test_sentinel_marks_complete(self):
        handler = _make_handler()
        handler.start_stream()
        handler.put_token("Hi")
        handler.put_token(None)
        handler.drain_stream()
        assert handler._stream_complete

    def test_start_stream_clears_state(self):
        handler = _make_handler()
        handler._stream_accumulated = "old"
        handler._tts_buffer = "old"
        handler.start_stream()
        assert handler._stream_accumulated == ""
        assert handler._tts_buffer == ""


class TestTTSSplitting:
    def test_sentence_triggers_tts(self):
        audio = MagicMock()
        handler = _make_handler(audio=audio)
        handler.start_stream()
        handler.put_token("Hello world. More text")
        handler.drain_stream()
        audio.play_voice.assert_called_once()
        # "Hello world" should have been spoken
        args = audio.play_voice.call_args[0][0]
        assert "Hello world" in args

    def test_no_tts_without_boundary(self):
        audio = MagicMock()
        handler = _make_handler(audio=audio)
        handler.start_stream()
        handler.put_token("Hello world")
        handler.drain_stream()
        audio.play_voice.assert_not_called()


class TestPendingResponse:
    def test_start_and_check_done(self):
        chat = MagicMock()
        handler = _make_handler(chat=chat)
        future = MagicMock()
        future.done.return_value = True
        future.cancelled.return_value = False
        future.result.return_value = "Reply text"

        handler.start_stream()
        handler.start_response(future)
        handler.check_pending()
        chat.finish_streaming.assert_called_once()

    def test_timeout_cancels_pending(self):
        chat = MagicMock()
        handler = _make_handler(chat=chat)
        future = MagicMock()
        future.done.return_value = False
        handler.start_response(future)
        handler._pending_time = 0.0  # force timeout
        handler.check_pending()
        future.cancel.assert_called_once()

    def test_cancelled_cleans_up(self):
        handler = _make_handler()
        future = MagicMock()
        future.done.return_value = True
        future.cancelled.return_value = True
        handler.start_response(future)
        handler.check_pending()
        assert handler._pending is None

    def test_cancel_response(self):
        handler = _make_handler()
        future = MagicMock()
        handler.start_response(future)
        handler.cancel_response()
        future.cancel.assert_called_once()
        assert handler._pending is None

    def test_scheduler_released_on_cleanup(self):
        scheduler = MagicMock()
        handler = _make_handler(scheduler=scheduler)
        future = MagicMock()
        future.done.return_value = True
        future.cancelled.return_value = True
        handler.start_response(future)
        handler.check_pending()
        scheduler.release.assert_called_with("chat")

    def test_is_busy(self):
        handler = _make_handler()
        assert not handler.is_busy
        handler.start_response(MagicMock())
        assert handler.is_busy


class TestAutonomousRemark:
    def test_start_and_check_done(self):
        handler = _make_handler()
        future = MagicMock()
        future.done.return_value = True
        future.cancelled.return_value = False
        future.result.return_value = "Remark"
        handler.start_autonomous(future)
        text, behavior = handler.check_pending_autonomous()
        assert text == "Remark"
        assert behavior is None

    def test_fallback_when_empty_result(self):
        handler = _make_handler()
        future = MagicMock()
        future.done.return_value = True
        future.cancelled.return_value = False
        future.result.return_value = ""
        mock_behavior = MagicMock()
        handler.start_autonomous(future, behavior=mock_behavior)
        text, behavior = handler.check_pending_autonomous()
        assert text is None
        assert behavior is mock_behavior

    def test_timeout_returns_fallback_behavior(self):
        scheduler = MagicMock()
        handler = _make_handler(scheduler=scheduler)
        future = MagicMock()
        future.done.return_value = False
        mock_behavior = MagicMock()
        handler.start_autonomous(future, behavior=mock_behavior)
        handler._pending_auto_time = 0.0  # force timeout
        text, behavior = handler.check_pending_autonomous()
        assert text is None
        assert behavior is mock_behavior
        scheduler.release.assert_called_with("chat")

    def test_cancel_autonomous(self):
        handler = _make_handler()
        future = MagicMock()
        handler.start_autonomous(future)
        handler.cancel_autonomous()
        future.cancel.assert_called_once()
        assert not handler.is_auto_busy
