"""Tests for the chat panel overlay (US-038).

Pygame is mocked to avoid requiring a display server in CI.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.K_TAB = 9
_pygame_mock.K_RETURN = 13
_pygame_mock.K_KP_ENTER = 271
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_BACKSPACE = 8
_pygame_mock.K_DELETE = 127
_pygame_mock.K_HOME = 278
_pygame_mock.K_END = 279
_pygame_mock.K_LEFT = 276
_pygame_mock.K_RIGHT = 275
_pygame_mock.K_PAGEUP = 280
_pygame_mock.K_PAGEDOWN = 281
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None

# Surface mock
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768

# Font mock
_font_mock = MagicMock()
_font_mock.get_linesize.return_value = 18
_font_mock.size.return_value = (100, 18)
_text_surf_mock = MagicMock()
_text_surf_mock.get_width.return_value = 100
_text_surf_mock.get_height.return_value = 18
_font_mock.render.return_value = _text_surf_mock
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock

# Draw mock
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.line.return_value = None

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)

# Surface constructor mock — returns a fresh MagicMock each time
def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 100
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 100
    return s

_pygame_mock.Surface = _make_surface

# Install pygame mock before importing chat_panel
sys.modules["pygame"] = _pygame_mock

from seaman_brain.config import GUIConfig  # noqa: E402
from seaman_brain.gui.chat_panel import ChatPanel  # noqa: E402
from seaman_brain.types import MessageRole  # noqa: E402


def _make_event(event_type: int, **kwargs) -> MagicMock:
    """Create a mock Pygame event."""
    ev = MagicMock()
    ev.type = event_type
    for k, v in kwargs.items():
        setattr(ev, k, v)
    return ev


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset draw mocks and re-install pygame mock between tests."""
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.chat_panel as chat_mod
    chat_mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    _pygame_mock.Surface = _make_surface
    _pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.get_linesize.return_value = 18
    _font_mock.size.return_value = (100, 18)
    _text_surf_mock.get_width.return_value = 100
    _text_surf_mock.get_height.return_value = 18
    _font_mock.render.return_value = _text_surf_mock
    yield


# ── Construction Tests ────────────────────────────────────────────────


class TestChatPanelConstruction:
    """Tests for ChatPanel initialization."""

    def test_default_visible(self):
        """ChatPanel is visible by default."""
        panel = ChatPanel()
        assert panel.visible is True

    def test_default_no_messages(self):
        """ChatPanel starts with no messages."""
        panel = ChatPanel()
        assert panel.message_count == 0

    def test_default_empty_input(self):
        """ChatPanel starts with empty input text."""
        panel = ChatPanel()
        assert panel.input_text == ""

    def test_default_not_streaming(self):
        """ChatPanel starts not streaming."""
        panel = ChatPanel()
        assert panel.is_streaming is False

    def test_custom_config(self):
        """ChatPanel accepts custom GUIConfig."""
        config = GUIConfig(window_width=800, window_height=600)
        panel = ChatPanel(gui_config=config)
        assert panel._config.window_width == 800

    def test_on_submit_callback(self):
        """ChatPanel stores the submit callback."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        assert panel._on_submit is cb


# ── Message Management Tests ─────────────────────────────────────────


class TestMessageManagement:
    """Tests for adding, clearing, and managing messages."""

    def test_add_user_message(self):
        """Add a user message to history."""
        panel = ChatPanel()
        panel.add_message(MessageRole.USER, "Hello creature")
        assert panel.message_count == 1

    def test_add_assistant_message(self):
        """Add an assistant message to history."""
        panel = ChatPanel()
        panel.add_message(MessageRole.ASSISTANT, "What do you want?")
        assert panel.message_count == 1

    def test_add_multiple_messages(self):
        """Add multiple messages in sequence."""
        panel = ChatPanel()
        panel.add_message(MessageRole.USER, "Hi")
        panel.add_message(MessageRole.ASSISTANT, "Go away")
        panel.add_message(MessageRole.USER, "No")
        assert panel.message_count == 3

    def test_message_role_preserved(self):
        """Message roles are preserved in history."""
        panel = ChatPanel()
        panel.add_message(MessageRole.USER, "Hello")
        panel.add_message(MessageRole.ASSISTANT, "Leave")
        msgs = list(panel._messages)
        assert msgs[0].role == MessageRole.USER
        assert msgs[1].role == MessageRole.ASSISTANT

    def test_message_text_preserved(self):
        """Message text is preserved in history."""
        panel = ChatPanel()
        panel.add_message(MessageRole.USER, "Tell me about yourself")
        msgs = list(panel._messages)
        assert msgs[0].text == "Tell me about yourself"

    def test_clear_messages(self):
        """Clear removes all messages."""
        panel = ChatPanel()
        panel.add_message(MessageRole.USER, "Hi")
        panel.add_message(MessageRole.ASSISTANT, "Bye")
        panel.clear_messages()
        assert panel.message_count == 0

    def test_message_history_limit(self):
        """Messages beyond _MAX_HISTORY are evicted."""
        panel = ChatPanel()
        for i in range(250):
            panel.add_message(MessageRole.USER, f"msg {i}")
        assert panel.message_count <= 200

    def test_add_message_resets_scroll(self):
        """Adding a message resets scroll to bottom."""
        panel = ChatPanel()
        panel._scroll_offset = 100
        panel.add_message(MessageRole.USER, "New message")
        assert panel._scroll_offset == 0


# ── Text Input Tests ─────────────────────────────────────────────────


class TestTextInput:
    """Tests for text input handling."""

    def test_type_character(self):
        """Typing a character adds it to input."""
        panel = ChatPanel()
        event = _make_event(_pygame_mock.KEYDOWN, key=0, unicode="a")
        panel.handle_event(event)
        assert panel.input_text == "a"
        assert panel._cursor_pos == 1

    def test_type_multiple_characters(self):
        """Typing multiple characters builds input string."""
        panel = ChatPanel()
        for ch in "hello":
            event = _make_event(_pygame_mock.KEYDOWN, key=0, unicode=ch)
            panel.handle_event(event)
        assert panel.input_text == "hello"
        assert panel._cursor_pos == 5

    def test_backspace_deletes_char(self):
        """Backspace removes character before cursor."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 5
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_BACKSPACE, unicode="")
        panel.handle_event(event)
        assert panel.input_text == "hell"
        assert panel._cursor_pos == 4

    def test_backspace_at_start_does_nothing(self):
        """Backspace at position 0 does nothing."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 0
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_BACKSPACE, unicode="")
        panel.handle_event(event)
        assert panel.input_text == "hello"
        assert panel._cursor_pos == 0

    def test_delete_removes_char_after_cursor(self):
        """Delete key removes character after cursor."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 2
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_DELETE, unicode="")
        panel.handle_event(event)
        assert panel.input_text == "helo"
        assert panel._cursor_pos == 2

    def test_delete_at_end_does_nothing(self):
        """Delete at end of text does nothing."""
        panel = ChatPanel()
        panel._input_text = "hi"
        panel._cursor_pos = 2
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_DELETE, unicode="")
        panel.handle_event(event)
        assert panel.input_text == "hi"

    def test_home_moves_cursor_to_start(self):
        """Home key moves cursor to position 0."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 3
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_HOME, unicode="")
        panel.handle_event(event)
        assert panel._cursor_pos == 0

    def test_end_moves_cursor_to_end(self):
        """End key moves cursor to end of text."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 0
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_END, unicode="")
        panel.handle_event(event)
        assert panel._cursor_pos == 5

    def test_left_arrow_moves_cursor_left(self):
        """Left arrow moves cursor one position left."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 3
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_LEFT, unicode="")
        panel.handle_event(event)
        assert panel._cursor_pos == 2

    def test_left_arrow_at_start_does_nothing(self):
        """Left arrow at position 0 stays at 0."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 0
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_LEFT, unicode="")
        panel.handle_event(event)
        assert panel._cursor_pos == 0

    def test_right_arrow_moves_cursor_right(self):
        """Right arrow moves cursor one position right."""
        panel = ChatPanel()
        panel._input_text = "hello"
        panel._cursor_pos = 2
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RIGHT, unicode="")
        panel.handle_event(event)
        assert panel._cursor_pos == 3

    def test_right_arrow_at_end_does_nothing(self):
        """Right arrow at end of text stays at end."""
        panel = ChatPanel()
        panel._input_text = "hi"
        panel._cursor_pos = 2
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RIGHT, unicode="")
        panel.handle_event(event)
        assert panel._cursor_pos == 2

    def test_insert_in_middle(self):
        """Typing with cursor in middle inserts at cursor position."""
        panel = ChatPanel()
        panel._input_text = "hllo"
        panel._cursor_pos = 1
        event = _make_event(_pygame_mock.KEYDOWN, key=0, unicode="e")
        panel.handle_event(event)
        assert panel.input_text == "hello"
        assert panel._cursor_pos == 2

    def test_non_printable_ignored(self):
        """Non-printable characters are not inserted."""
        panel = ChatPanel()
        event = _make_event(_pygame_mock.KEYDOWN, key=999, unicode="")
        consumed = panel.handle_event(event)
        assert panel.input_text == ""
        assert consumed is False


# ── Submit Tests ─────────────────────────────────────────────────────


class TestSubmit:
    """Tests for message submission."""

    def test_enter_submits_message(self):
        """Enter key submits input and adds to history."""
        panel = ChatPanel()
        panel._input_text = "Hello creature"
        panel._cursor_pos = 14
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        panel.handle_event(event)
        assert panel.input_text == ""
        assert panel.message_count == 1

    def test_enter_calls_callback(self):
        """Enter key invokes the on_submit callback."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        panel._input_text = "feed it"
        panel._cursor_pos = 7
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        panel.handle_event(event)
        cb.assert_called_once_with("feed it")

    def test_enter_empty_input_does_nothing(self):
        """Enter with empty input does not submit."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        panel.handle_event(event)
        cb.assert_not_called()
        assert panel.message_count == 0

    def test_enter_whitespace_only_does_nothing(self):
        """Enter with whitespace-only input does not submit."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        panel._input_text = "   "
        panel._cursor_pos = 3
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        panel.handle_event(event)
        cb.assert_not_called()
        assert panel.message_count == 0

    def test_kp_enter_also_submits(self):
        """Numpad Enter also submits input."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        panel._input_text = "test"
        panel._cursor_pos = 4
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_KP_ENTER, unicode="")
        panel.handle_event(event)
        cb.assert_called_once_with("test")

    def test_submit_strips_whitespace(self):
        """Submit strips leading/trailing whitespace from input."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        panel._input_text = "  hello  "
        panel._cursor_pos = 9
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        panel.handle_event(event)
        cb.assert_called_once_with("hello")

    def test_submit_clears_input(self):
        """Submitting clears the input field and resets cursor."""
        panel = ChatPanel()
        panel._input_text = "test"
        panel._cursor_pos = 4
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        panel.handle_event(event)
        assert panel.input_text == ""
        assert panel._cursor_pos == 0

    def test_submit_callback_error_handled(self):
        """Errors in submit callback are caught, not raised."""
        cb = MagicMock(side_effect=RuntimeError("boom"))
        panel = ChatPanel(on_submit=cb)
        panel._input_text = "test"
        panel._cursor_pos = 4
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_RETURN, unicode="")
        # Should not raise
        panel.handle_event(event)
        assert panel.message_count == 1  # Message still added


# ── Visibility Toggle Tests ──────────────────────────────────────────


class TestVisibilityToggle:
    """Tests for Tab key toggle."""

    def test_tab_hides_panel(self):
        """Tab key hides the visible panel."""
        panel = ChatPanel()
        assert panel.visible is True
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_TAB, unicode="")
        consumed = panel.handle_event(event)
        assert panel.visible is False
        assert consumed is True

    def test_tab_shows_panel(self):
        """Tab key shows the hidden panel."""
        panel = ChatPanel()
        panel.visible = False
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_TAB, unicode="")
        consumed = panel.handle_event(event)
        assert panel.visible is True
        assert consumed is True

    def test_keys_ignored_when_hidden(self):
        """Key events (except Tab) are not consumed when panel is hidden."""
        panel = ChatPanel()
        panel.visible = False
        event = _make_event(_pygame_mock.KEYDOWN, key=0, unicode="a")
        consumed = panel.handle_event(event)
        assert consumed is False
        assert panel.input_text == ""

    def test_non_keydown_events_not_consumed(self):
        """Non-KEYDOWN events are not consumed."""
        panel = ChatPanel()
        event = _make_event(999, key=0)
        consumed = panel.handle_event(event)
        assert consumed is False


# ── Streaming Tests ──────────────────────────────────────────────────


class TestStreaming:
    """Tests for streaming creature responses."""

    def test_start_streaming(self):
        """start_streaming enables streaming mode."""
        panel = ChatPanel()
        panel.start_streaming()
        assert panel.is_streaming is True

    def test_append_stream_chunk(self):
        """append_stream accumulates text."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("Hello ")
        panel.append_stream("world")
        assert panel._stream_text == "Hello world"

    def test_finish_streaming_adds_message(self):
        """finish_streaming adds the accumulated text as a message."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("Sardonic response")
        panel.finish_streaming()
        assert panel.is_streaming is False
        assert panel.message_count == 1
        msgs = list(panel._messages)
        assert msgs[0].role == MessageRole.ASSISTANT
        assert msgs[0].text == "Sardonic response"

    def test_finish_empty_stream_no_message(self):
        """Finishing an empty stream does not add a message."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.finish_streaming()
        assert panel.message_count == 0

    def test_append_without_streaming_ignored(self):
        """append_stream does nothing when not streaming."""
        panel = ChatPanel()
        panel.append_stream("ignored")
        assert panel._stream_text == ""

    def test_stream_resets_on_finish(self):
        """Stream text is cleared after finish."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("data")
        panel.finish_streaming()
        assert panel._stream_text == ""


# ── Rendering Tests ──────────────────────────────────────────────────


class TestRendering:
    """Tests for render output."""

    def test_render_when_visible(self):
        """Render draws to surface when visible."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        # Should blit at least the panel background
        assert _surface_mock.blit.called

    def test_render_when_hidden_does_nothing(self):
        """Render does nothing when panel is hidden."""
        panel = ChatPanel()
        panel.visible = False
        panel.render(_surface_mock)
        assert not _surface_mock.blit.called

    def test_render_with_messages(self):
        """Render handles messages in history."""
        panel = ChatPanel()
        panel.add_message(MessageRole.USER, "Hello")
        panel.add_message(MessageRole.ASSISTANT, "Go away")
        # Should not raise
        panel.render(_surface_mock)
        assert _surface_mock.blit.called

    def test_render_with_streaming(self):
        """Render shows streaming text."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("In progress...")
        panel.render(_surface_mock)
        assert _surface_mock.blit.called

    def test_render_with_input_text(self):
        """Render shows input text."""
        panel = ChatPanel()
        panel._input_text = "typing something"
        panel._cursor_pos = 16
        panel.render(_surface_mock)
        assert _surface_mock.blit.called

    def test_render_empty_input_shows_placeholder(self):
        """Render shows placeholder when input is empty."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        # Font renders placeholder text
        assert _font_mock.render.called
        rendered_calls = [
            c for c in _font_mock.render.call_args_list
            if "Type a message" in str(c)
        ]
        assert len(rendered_calls) > 0

    def test_render_cursor_blink(self):
        """Cursor visibility toggles with update."""
        panel = ChatPanel()
        panel._cursor_visible = True
        panel.update(0.6)  # Exceeds blink rate
        # Cursor should have toggled
        assert panel._cursor_visible is False


# ── Scroll Tests ─────────────────────────────────────────────────────


class TestScrolling:
    """Tests for scroll behavior."""

    def test_pageup_scrolls_up(self):
        """PageUp scrolls message area up."""
        panel = ChatPanel()
        panel._total_content_height = 1000
        panel._msg_area_rect = (0, 0, 1024, 200)
        panel._font_height = 18
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_PAGEUP, unicode="")
        panel.handle_event(event)
        assert panel._scroll_offset > 0

    def test_pagedown_scrolls_down(self):
        """PageDown scrolls message area down."""
        panel = ChatPanel()
        panel._scroll_offset = 100
        panel._font_height = 18
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_PAGEDOWN, unicode="")
        panel.handle_event(event)
        assert panel._scroll_offset < 100

    def test_scroll_down_clamped_to_zero(self):
        """Scroll offset cannot go below 0."""
        panel = ChatPanel()
        panel._scroll_offset = 10
        panel._font_height = 18
        event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_PAGEDOWN, unicode="")
        panel.handle_event(event)
        assert panel._scroll_offset >= 0

    def test_scroll_up_clamped_to_max(self):
        """Scroll offset cannot exceed total content height - visible area."""
        panel = ChatPanel()
        panel._total_content_height = 300
        panel._msg_area_rect = (0, 0, 1024, 200)
        panel._font_height = 18
        # Scroll way up
        for _ in range(20):
            event = _make_event(_pygame_mock.KEYDOWN, key=_pygame_mock.K_PAGEUP, unicode="")
            panel.handle_event(event)
        max_scroll = 300 - 200
        assert panel._scroll_offset <= max_scroll


# ── Update Tests ─────────────────────────────────────────────────────


class TestUpdate:
    """Tests for update loop."""

    def test_update_advances_cursor_timer(self):
        """Update advances the cursor blink timer."""
        panel = ChatPanel()
        panel._cursor_timer = 0.0
        panel.update(0.1)
        assert panel._cursor_timer == pytest.approx(0.1)

    def test_cursor_blinks_after_interval(self):
        """Cursor toggles visibility after blink interval."""
        panel = ChatPanel()
        panel._cursor_visible = True
        panel._cursor_timer = 0.0
        panel.update(0.54)  # > 0.53 blink rate
        assert panel._cursor_visible is False

    def test_zero_dt_no_blink(self):
        """Zero dt does not toggle cursor."""
        panel = ChatPanel()
        panel._cursor_visible = True
        panel._cursor_timer = 0.0
        panel.update(0.0)
        assert panel._cursor_visible is True


# ── Word Wrap Tests ──────────────────────────────────────────────────


class TestWordWrap:
    """Tests for text word-wrapping."""

    def test_short_text_single_line(self):
        """Short text fits on one line."""
        panel = ChatPanel()
        panel._ensure_font()
        # Mock font.size to return small width
        _font_mock.size.return_value = (50, 18)
        lines = panel._wrap_text("hello", 200)
        assert len(lines) == 1
        assert lines[0] == "hello"

    def test_empty_text_returns_empty_line(self):
        """Empty text returns list with one empty string."""
        panel = ChatPanel()
        panel._ensure_font()
        lines = panel._wrap_text("", 200)
        assert lines == [""]

    def test_newline_creates_multiple_lines(self):
        """Text with newlines creates multiple paragraphs."""
        panel = ChatPanel()
        panel._ensure_font()
        _font_mock.size.return_value = (50, 18)
        lines = panel._wrap_text("line1\nline2", 200)
        assert len(lines) == 2

    def test_wrap_long_text(self):
        """Long text wraps across multiple lines."""
        panel = ChatPanel()
        panel._ensure_font()
        # Make each word measure 80px, so 2 words per line at width 200
        def mock_size(text):
            words = text.split()
            return (len(words) * 80, 18)
        _font_mock.size.side_effect = mock_size
        lines = panel._wrap_text("one two three four", 200)
        assert len(lines) >= 2
        _font_mock.size.side_effect = None
        _font_mock.size.return_value = (100, 18)


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_render_zero_size_surface(self):
        """Render handles a zero-size surface gracefully."""
        panel = ChatPanel()
        tiny_surface = MagicMock()
        tiny_surface.get_width.return_value = 0
        tiny_surface.get_height.return_value = 0
        # Should not raise
        panel.render(tiny_surface)

    def test_many_rapid_messages(self):
        """Adding many messages rapidly doesn't break anything."""
        panel = ChatPanel()
        for i in range(100):
            panel.add_message(MessageRole.USER, f"Message {i}")
            panel.add_message(MessageRole.ASSISTANT, f"Response {i}")
        assert panel.message_count == 200

    def test_stream_then_regular_message(self):
        """Streaming followed by regular message works correctly."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("streaming...")
        panel.finish_streaming()
        panel.add_message(MessageRole.USER, "another message")
        assert panel.message_count == 2

    def test_multiple_streams_sequential(self):
        """Multiple sequential streams work correctly."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("first")
        panel.finish_streaming()
        panel.start_streaming()
        panel.append_stream("second")
        panel.finish_streaming()
        assert panel.message_count == 2
        msgs = list(panel._messages)
        assert msgs[0].text == "first"
        assert msgs[1].text == "second"

    def test_system_message_handled(self):
        """System messages are accepted and stored."""
        panel = ChatPanel()
        panel.add_message(MessageRole.SYSTEM, "System: creature evolved")
        assert panel.message_count == 1
        assert list(panel._messages)[0].role == MessageRole.SYSTEM


# ── Header Tests ─────────────────────────────────────────────────────


class TestChatHeader:
    """Tests for the Chat header bar."""

    def test_render_draws_header(self):
        """Render draws 'Chat' header text."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        rendered_texts = [str(call.args[0]) for call in _font_mock.render.call_args_list]
        assert any("Chat" in t for t in rendered_texts)

    def test_header_blitted(self):
        """Render blits the header surface."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        # Multiple blits: panel bg, header, input bg, placeholder, send button
        assert _surface_mock.blit.call_count >= 3


# ── Send Button Tests ────────────────────────────────────────────────


class TestSendButton:
    """Tests for the Send button."""

    def test_render_draws_send_button(self):
        """Render draws 'Send' button text."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        rendered_texts = [str(call.args[0]) for call in _font_mock.render.call_args_list]
        assert any("Send" in t for t in rendered_texts)

    def test_send_button_click_submits(self):
        """Clicking Send button submits input text."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        panel._input_text = "Hello creature"
        panel._cursor_pos = 14
        # Render first to compute layout
        panel.render(_surface_mock)
        # Click on the send button rect
        sx, sy, sw, sh = panel._send_rect
        result = panel.handle_click(sx + 5, sy + 5)
        assert result is True
        cb.assert_called_once_with("Hello creature")
        assert panel.input_text == ""

    def test_send_button_click_empty_does_nothing(self):
        """Clicking Send with empty input does nothing."""
        cb = MagicMock()
        panel = ChatPanel(on_submit=cb)
        panel.render(_surface_mock)
        sx, sy, sw, sh = panel._send_rect
        result = panel.handle_click(sx + 5, sy + 5)
        assert result is True  # Click consumed, but nothing submitted
        cb.assert_not_called()

    def test_click_outside_send_returns_false(self):
        """Clicking outside Send button returns False."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        result = panel.handle_click(5, 5)
        assert result is False

    def test_mouse_move_hover_send(self):
        """Moving mouse over Send button sets hover state."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        sx, sy, sw, sh = panel._send_rect
        panel.handle_mouse_move(sx + 5, sy + 5)
        assert panel._send_hover is True

    def test_mouse_move_outside_send(self):
        """Moving mouse away from Send button clears hover."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        panel._send_hover = True
        panel.handle_mouse_move(5, 5)
        assert panel._send_hover is False


# ── Message Truncation Tests (Fix #18) ────────────────────────────────


class TestMessageTruncation:
    """Tests for _MAX_MESSAGE_LENGTH truncation in add_message."""

    def test_long_message_truncated(self):
        """Messages longer than _MAX_MESSAGE_LENGTH are truncated with '...'."""
        from seaman_brain.gui.chat_panel import _MAX_MESSAGE_LENGTH

        panel = ChatPanel()
        long_text = "A" * (_MAX_MESSAGE_LENGTH + 500)
        panel.add_message(MessageRole.USER, long_text)
        msg = panel._messages[-1]
        assert len(msg.text) == _MAX_MESSAGE_LENGTH + 3  # +3 for "..."
        assert msg.text.endswith("...")

    def test_short_message_not_truncated(self):
        """Messages under _MAX_MESSAGE_LENGTH pass through unchanged."""
        panel = ChatPanel()
        short = "Hello, Seaman!"
        panel.add_message(MessageRole.USER, short)
        assert panel._messages[-1].text == short

    def test_exact_limit_not_truncated(self):
        """Message at exactly _MAX_MESSAGE_LENGTH is not truncated."""
        from seaman_brain.gui.chat_panel import _MAX_MESSAGE_LENGTH

        panel = ChatPanel()
        text = "B" * _MAX_MESSAGE_LENGTH
        panel.add_message(MessageRole.ASSISTANT, text)
        assert panel._messages[-1].text == text


# ── Character-Level Word Breaking Tests (Fix #18) ─────────────────────


class TestCharacterLevelBreaking:
    """Tests for character-level breaking of words wider than max_width."""

    def test_long_spaceless_word_breaks(self):
        """A word wider than max_width is broken character-by-character."""
        panel = ChatPanel()
        # Mock font.size: each char is 20px, so max_width=100 fits 5 chars
        panel._font = MagicMock()
        panel._font.size = lambda t: (len(t) * 20, 16)

        result = panel._wrap_text("AAAAABBBBBCCCCC", max_width=100)
        # 15 chars at 20px each = 300px total; should break into 5-char chunks
        assert len(result) == 3
        assert result[0] == "AAAAA"
        assert result[1] == "BBBBB"
        assert result[2] == "CCCCC"

    def test_mixed_short_and_long_words(self):
        """Normal words fit, but a long word in the middle gets broken."""
        panel = ChatPanel()
        panel._font = MagicMock()
        panel._font.size = lambda t: (len(t) * 10, 16)

        result = panel._wrap_text("hi AAAAAAAAAAAA ok", max_width=50)
        # "hi" fits (20px), "AAAAAAAAAAAA" (120px) breaks into 5-char chunks,
        # "ok" (20px) fits
        assert any(len(line) <= 5 for line in result)
        # Reconstruct text to make sure nothing was lost
        joined = "".join(result)
        assert "hi" in joined
        assert "AAAAAAAAAAAA" in joined
        assert "ok" in joined

    def test_empty_string_returns_empty(self):
        """Empty string returns single empty-string list."""
        panel = ChatPanel()
        panel._font = MagicMock()
        result = panel._wrap_text("", max_width=100)
        assert result == [""]

    def test_no_font_returns_text_unchanged(self):
        """With no font, text is returned as-is in a single-element list."""
        panel = ChatPanel()
        panel._font = None
        result = panel._wrap_text("hello world", max_width=100)
        assert result == ["hello world"]


# ── Thinking Indicator Tests (Phase 1) ────────────────────────────────


class TestThinkingIndicator:
    """Tests for the animated thinking dots when LLM is processing."""

    def test_thinking_timer_advances_while_streaming(self):
        """_thinking_timer advances when streaming is active."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.update(0.5)
        assert panel._thinking_timer == pytest.approx(0.5)

    def test_thinking_timer_resets_when_not_streaming(self):
        """_thinking_timer resets to 0 when streaming stops."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.update(1.0)
        assert panel._thinking_timer > 0
        panel.finish_streaming()
        panel.update(0.1)
        assert panel._thinking_timer == pytest.approx(0.0)

    def test_thinking_dots_rendered_during_empty_stream(self):
        """Thinking dots are rendered when streaming but no text yet."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.update(0.5)  # Advance timer so dots show
        panel.render(_surface_mock)
        # Font should render dot text (., .., or ...)
        rendered_texts = [str(call.args[0]) for call in _font_mock.render.call_args_list]
        dot_renders = [t for t in rendered_texts if t in (".", "..", "...")]
        assert len(dot_renders) > 0, f"Expected dot renders, got: {rendered_texts}"

    def test_thinking_dots_cycle_animation(self):
        """Dots cycle through ., .., ... based on timer."""
        from seaman_brain.gui.chat_panel import _THINKING_DOT_INTERVAL

        panel = ChatPanel()
        panel.start_streaming()

        # At time 0 -> 1 dot
        panel._thinking_timer = 0.0
        dot_count_0 = int(panel._thinking_timer / _THINKING_DOT_INTERVAL) % 3 + 1
        assert dot_count_0 == 1

        # At time 0.4 -> 2 dots
        panel._thinking_timer = _THINKING_DOT_INTERVAL
        dot_count_1 = int(panel._thinking_timer / _THINKING_DOT_INTERVAL) % 3 + 1
        assert dot_count_1 == 2

        # At time 0.8 -> 3 dots
        panel._thinking_timer = _THINKING_DOT_INTERVAL * 2
        dot_count_2 = int(panel._thinking_timer / _THINKING_DOT_INTERVAL) % 3 + 1
        assert dot_count_2 == 3

        # At time 1.2 -> wraps back to 1 dot
        panel._thinking_timer = _THINKING_DOT_INTERVAL * 3
        dot_count_3 = int(panel._thinking_timer / _THINKING_DOT_INTERVAL) % 3 + 1
        assert dot_count_3 == 1

    def test_thinking_not_shown_when_stream_has_text(self):
        """Thinking dots are NOT shown when streaming already has text."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("Hello")
        panel.update(0.5)
        panel.render(_surface_mock)
        # The rendered text should include "Hello", not dots
        rendered_texts = [str(call.args[0]) for call in _font_mock.render.call_args_list]
        # "Hello" should be rendered, not dots
        hello_renders = [t for t in rendered_texts if "Hello" in t]
        assert len(hello_renders) > 0

    def test_thinking_not_shown_when_not_streaming(self):
        """No thinking dots when not streaming."""
        panel = ChatPanel()
        panel.update(0.5)
        panel.render(_surface_mock)
        rendered_texts = [str(call.args[0]) for call in _font_mock.render.call_args_list]
        dot_only_renders = [
            t for t in rendered_texts if t in (".", "..", "...")
        ]
        assert len(dot_only_renders) == 0


class TestStreamingLengthLimit:
    """Tests for streaming text length limit (#18)."""

    def test_stream_text_capped_at_max_length(self):
        """Streaming text stops accumulating past _MAX_STREAM_LENGTH."""
        from seaman_brain.gui.chat_panel import _MAX_STREAM_LENGTH

        panel = ChatPanel()
        panel.start_streaming()
        # Send more text than the limit
        chunk = "x" * 500
        for _ in range(20):
            panel.append_stream(chunk)
        assert len(panel._stream_text) <= _MAX_STREAM_LENGTH + len(chunk)

    def test_stream_text_accepts_under_limit(self):
        """Streaming text accumulates normally under the limit."""
        panel = ChatPanel()
        panel.start_streaming()
        panel.append_stream("hello ")
        panel.append_stream("world")
        assert panel._stream_text == "hello world"

    def test_finish_streaming_truncates_via_add_message(self):
        """finish_streaming() goes through add_message() which truncates."""
        from seaman_brain.gui.chat_panel import _MAX_MESSAGE_LENGTH

        panel = ChatPanel()
        panel.start_streaming()
        panel._stream_text = "x" * 5000
        panel.finish_streaming()
        assert len(list(panel._messages)[-1].text) <= _MAX_MESSAGE_LENGTH + 3


class TestSurfaceDimensionValidation:
    """Tests for surface dimension clamping (#19)."""

    def test_render_with_tiny_window(self):
        """Rendering with very small window doesn't crash."""
        tiny_surface = MagicMock()
        tiny_surface.get_width.return_value = 10
        tiny_surface.get_height.return_value = 10

        panel = ChatPanel()
        panel.render(tiny_surface)  # No exception

    def test_render_with_huge_window(self):
        """Rendering with very large window doesn't crash."""
        huge_surface = MagicMock()
        huge_surface.get_width.return_value = 16384
        huge_surface.get_height.return_value = 16384

        panel = ChatPanel()
        panel.render(huge_surface)  # No exception


class TestFontFallbackChain:
    """Tests for font fallback chain (#20)."""

    def test_font_initialized_after_render(self):
        """Font is initialized after first render."""
        panel = ChatPanel()
        panel.render(_surface_mock)
        assert panel._font is not None

    def test_font_fallback_when_sysfont_fails(self):
        """Falls back to Font(None) when all SysFont calls fail."""
        _pygame_mock.font.SysFont.side_effect = RuntimeError("no font")
        try:
            panel = ChatPanel()
            panel._font = None  # Reset
            panel._ensure_font()
            assert panel._font is not None
        finally:
            _pygame_mock.font.SysFont.side_effect = None
            _pygame_mock.font.SysFont.return_value = _font_mock
