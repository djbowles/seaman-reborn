"""Tests for input routing and keyboard shortcuts."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.KEYDOWN = 768
_pygame_mock.MOUSEBUTTONDOWN = 1025
_pygame_mock.MOUSEMOTION = 1024
_pygame_mock.MOUSEWHEEL = 1027
_pygame_mock.MOUSEBUTTONUP = 1026
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_F2 = 283
_pygame_mock.K_TAB = 9
_pygame_mock.K_m = 109
_pygame_mock.K_RETURN = 13
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.input_handler import InputHandler  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.input_handler as mod
    mod.pygame = _pygame_mock
    yield


class TestKeyboardShortcuts:
    def test_escape_calls_handler(self):
        handler = InputHandler()
        cb = MagicMock()
        handler.on_escape = cb
        event = MagicMock(type=768, key=27)
        handler.handle_event(event)
        cb.assert_called_once()

    def test_f2_calls_toggle_settings(self):
        handler = InputHandler()
        cb = MagicMock()
        handler.on_toggle_settings = cb
        event = MagicMock(type=768, key=283)
        handler.handle_event(event)
        cb.assert_called_once()

    def test_m_calls_toggle_mic(self):
        handler = InputHandler()
        cb = MagicMock()
        handler.on_toggle_mic = cb
        event = MagicMock(type=768, key=109)
        handler.handle_event(event)
        cb.assert_called_once()

    def test_keys_suppressed_when_chat_focused(self):
        handler = InputHandler()
        handler.chat_focused = True
        cb = MagicMock()
        handler.on_toggle_mic = cb
        event = MagicMock(type=768, key=109)
        handler.handle_event(event)
        cb.assert_not_called()
