"""Tests for the glass overlay chat panel with message bubbles."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.K_RETURN = 13
_pygame_mock.K_BACKSPACE = 8
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 14
_font_mock.render.return_value = _font_surface
_font_mock.size.side_effect = lambda text: (len(text) * 7, 14)
_font_mock.get_linesize.return_value = 16
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
_surface_mock = MagicMock()
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.chat_panel import ChatPanel  # noqa: E402
from seaman_brain.gui.layout import ScreenLayout  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.chat_panel as mod
    mod.pygame = _pygame_mock
    yield


@pytest.fixture()
def layout():
    return ScreenLayout(1024, 768)


@pytest.fixture()
def panel(layout):
    return ChatPanel(layout)


class TestConstruction:
    def test_creates(self, panel):
        assert panel is not None

    def test_empty_messages(self, panel):
        assert len(panel._messages) == 0

    def test_empty_input(self, panel):
        assert panel._input_text == ""


class TestMessages:
    def test_add_system_message(self, panel):
        panel.add_message("system", "Test message")
        assert len(panel._messages) == 1
        assert panel._messages[0]["role"] == "system"
        assert panel._messages[0]["text"] == "Test message"

    def test_add_creature_message(self, panel):
        panel.add_message("creature", "*yawns*")
        assert panel._messages[0]["role"] == "creature"

    def test_add_user_message(self, panel):
        panel.add_message("user", "Hello")
        assert panel._messages[0]["role"] == "user"

    def test_max_history_limit(self, panel):
        for i in range(300):
            panel.add_message("user", f"msg {i}")
        assert len(panel._messages) <= 200

    def test_streaming_update(self, panel):
        panel.add_message("creature", "Hello", streaming=True)
        assert panel._messages[0]["streaming"] is True
        panel.update_streaming("Hello world")
        assert panel._messages[-1]["text"] == "Hello world"

    def test_finish_streaming(self, panel):
        panel.add_message("creature", "partial", streaming=True)
        panel.finish_streaming()
        assert panel._messages[-1]["streaming"] is False


class TestInput:
    def test_set_input_text(self, panel):
        panel._input_text = "Hello"
        assert panel._input_text == "Hello"

    def test_submit_clears_input(self, panel):
        panel._input_text = "Hello"
        cb = MagicMock()
        panel.on_submit = cb
        panel._submit()
        assert panel._input_text == ""
        cb.assert_called_once_with("Hello")

    def test_submit_empty_does_nothing(self, panel):
        cb = MagicMock()
        panel.on_submit = cb
        panel._submit()
        cb.assert_not_called()

    def test_handle_key_return_submits(self, panel):
        panel._input_text = "test"
        cb = MagicMock()
        panel.on_submit = cb
        panel._input_focused = True
        panel.handle_key(13, "")  # K_RETURN
        cb.assert_called_once_with("test")

    def test_handle_key_backspace(self, panel):
        panel._input_text = "abc"
        panel._input_focused = True
        panel.handle_key(8, "")  # K_BACKSPACE
        assert panel._input_text == "ab"

    def test_handle_key_char(self, panel):
        panel._input_text = ""
        panel._input_focused = True
        panel.handle_key(0, "x")
        assert panel._input_text == "x"


class TestRendering:
    def test_render_does_not_crash(self, panel):
        panel.add_message("system", "Test")
        panel.add_message("creature", "Hi")
        panel.add_message("user", "Hey")
        surface = MagicMock()
        panel.render(surface)

    def test_render_empty_does_not_crash(self, panel):
        surface = MagicMock()
        panel.render(surface)


class TestScrolling:
    def test_auto_scroll_on_new_message(self, panel):
        for i in range(20):
            panel.add_message("user", f"msg {i}")
        assert panel._auto_scroll is True

    def test_scroll_lock_on_manual_scroll(self, panel):
        for i in range(20):
            panel.add_message("user", f"msg {i}")
        panel.handle_scroll(1)  # scroll up
        assert panel._auto_scroll is False

    def test_scroll_to_bottom_resets_auto(self, panel):
        for i in range(20):
            panel.add_message("user", f"msg {i}")
        panel.handle_scroll(1)
        panel.scroll_to_bottom()
        assert panel._auto_scroll is True


class TestResize:
    def test_resize(self, panel):
        new_layout = ScreenLayout(1920, 1080)
        panel.resize(new_layout)
        # No crash, layout updated
