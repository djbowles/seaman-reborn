"""Chat panel overlay for Pygame - message history and text input.

Renders a semi-transparent chat panel at the bottom of the screen showing
conversation history (user messages right-aligned, creature left-aligned)
and a text input field with cursor and basic editing support.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import pygame

from seaman_brain.config import GUIConfig
from seaman_brain.types import MessageRole

logger = logging.getLogger(__name__)

# ── Colors ───────────────────────────────────────────────────────────────

_PANEL_BG = (10, 15, 30, 220)  # Semi-transparent dark blue
_INPUT_BG = (20, 30, 50, 220)  # Darker input area
_INPUT_BORDER = (60, 100, 160)
_INPUT_BORDER_FOCUS = (80, 160, 240)
_CURSOR_COLOR = (200, 220, 255)
_USER_MSG_COLOR = (140, 200, 255)  # Light blue for user
_CREATURE_MSG_COLOR = (180, 255, 180)  # Light green for creature
_SYSTEM_MSG_COLOR = (180, 180, 180)  # Gray for system
_USER_BG = (30, 50, 80, 160)  # User message bubble bg
_CREATURE_BG = (20, 50, 30, 160)  # Creature message bubble bg
_PLACEHOLDER_COLOR = (100, 120, 140)
_SCROLLBAR_BG = (30, 40, 60, 120)
_SCROLLBAR_THUMB = (80, 120, 180, 180)

# ── Constants ────────────────────────────────────────────────────────────

_INPUT_HEIGHT = 36
_INPUT_PADDING = 8
_MSG_PADDING = 6
_MSG_MARGIN = 4
_BUBBLE_RADIUS = 6
_SCROLLBAR_WIDTH = 8
_MAX_HISTORY = 200
_MAX_MESSAGE_LENGTH = 2000
_FONT_SIZE = 15
_CURSOR_BLINK_RATE = 0.53  # seconds
_HEADER_HEIGHT = 24
_HEADER_BG = (15, 25, 45, 240)
_HEADER_TEXT_COLOR = (180, 200, 220)
_SEND_BUTTON_WIDTH = 60
_SEND_BUTTON_BG = (30, 60, 100)
_SEND_BUTTON_BG_HOVER = (50, 85, 140)
_SEND_BUTTON_BORDER = (80, 130, 200)
_SEND_BUTTON_TEXT = (200, 220, 255)


@dataclass
class ChatMessage:
    """A rendered chat message with layout info."""

    role: MessageRole
    text: str
    wrapped_lines: list[str] = field(default_factory=list)
    height: int = 0


class ChatPanel:
    """Semi-transparent chat overlay at bottom of the Pygame window.

    Features:
        - Scrollable message history (user right-aligned, creature left-aligned)
        - Text input with cursor, basic editing (backspace, delete, home, end)
        - Enter sends message via callback
        - Streaming display for creature responses
        - Toggle visibility with Tab key

    Attributes:
        visible: Whether the panel is currently shown.
    """

    def __init__(
        self,
        gui_config: GUIConfig | None = None,
        on_submit: callable | None = None,
    ) -> None:
        """Initialize the chat panel.

        Args:
            gui_config: GUI configuration for sizing. Uses defaults if None.
            on_submit: Callback invoked with the message text when user presses Enter.
        """
        self._config = gui_config or GUIConfig()
        self._on_submit = on_submit

        self.visible = True
        self._messages: deque[ChatMessage] = deque(maxlen=_MAX_HISTORY)

        # Text input state
        self._input_text = ""
        self._cursor_pos = 0
        self._cursor_visible = True
        self._cursor_timer = 0.0

        # Streaming state
        self._streaming = False
        self._stream_text = ""

        # Scroll state
        self._scroll_offset = 0  # 0 = bottom (newest), positive = scrolled up
        self._total_content_height = 0

        # Font (lazy-initialized on first render)
        self._font: pygame.font.Font | None = None
        self._font_height = 0

        # Send button state
        self._send_hover = False
        self._send_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

        # Panel dimensions (calculated on render)
        self._panel_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._input_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        self._msg_area_rect: tuple[int, int, int, int] = (0, 0, 0, 0)

    @property
    def input_text(self) -> str:
        """Current text in the input field."""
        return self._input_text

    @property
    def is_streaming(self) -> bool:
        """Whether a streaming response is in progress."""
        return self._streaming

    @property
    def message_count(self) -> int:
        """Number of messages in history."""
        return len(self._messages)

    def _ensure_font(self) -> None:
        """Initialize the font if not yet done."""
        if self._font is None:
            try:
                self._font = pygame.font.SysFont("consolas", _FONT_SIZE)
                self._font_height = self._font.get_linesize()
            except Exception:
                self._font = pygame.font.Font(None, _FONT_SIZE)
                self._font_height = self._font.get_linesize()

    def _calculate_layout(self, surface_width: int, surface_height: int) -> None:
        """Calculate panel dimensions based on window size."""
        panel_h = int(surface_height * self._config.chat_panel_height_ratio)
        panel_y = surface_height - panel_h
        self._panel_rect = (0, panel_y, surface_width, panel_h)

        # Input area at bottom of panel (shrunk to make room for Send button)
        input_y = panel_y + panel_h - _INPUT_HEIGHT - _INPUT_PADDING
        input_x = _INPUT_PADDING
        input_w = surface_width - 2 * _INPUT_PADDING - _SEND_BUTTON_WIDTH - _INPUT_PADDING
        self._input_rect = (input_x, input_y, input_w, _INPUT_HEIGHT)

        # Send button (right of input)
        send_x = input_x + input_w + _INPUT_PADDING
        self._send_rect = (send_x, input_y, _SEND_BUTTON_WIDTH, _INPUT_HEIGHT)

        # Message area above input, below header
        msg_y = panel_y + _HEADER_HEIGHT + _MSG_PADDING
        msg_h = (
            panel_h - _HEADER_HEIGHT - _INPUT_HEIGHT
            - _INPUT_PADDING * 2 - _MSG_PADDING
        )
        self._msg_area_rect = (_MSG_PADDING, msg_y, surface_width - 2 * _MSG_PADDING, msg_h)

    def _wrap_text(self, text: str, max_width: int) -> list[str]:
        """Word-wrap text to fit within max_width pixels.

        Args:
            text: The text to wrap.
            max_width: Maximum pixel width per line.

        Returns:
            A list of wrapped lines.
        """
        if self._font is None or not text:
            return [text] if text else [""]

        lines: list[str] = []
        for paragraph in text.split("\n"):
            if not paragraph:
                lines.append("")
                continue
            words = paragraph.split(" ")
            current_line = ""
            for word in words:
                test = f"{current_line} {word}".strip() if current_line else word
                test_w = self._font.size(test)[0]
                if test_w <= max_width:
                    current_line = test
                else:
                    if current_line:
                        lines.append(current_line)
                    # Break word character-by-character if it exceeds max_width
                    if self._font.size(word)[0] > max_width:
                        chunk = ""
                        for ch in word:
                            if self._font.size(chunk + ch)[0] > max_width and chunk:
                                lines.append(chunk)
                                chunk = ch
                            else:
                                chunk += ch
                        current_line = chunk
                    else:
                        current_line = word
            if current_line:
                lines.append(current_line)
        return lines if lines else [""]

    def add_message(self, role: MessageRole, text: str) -> None:
        """Add a message to the chat history.

        Args:
            role: Who sent the message (USER, ASSISTANT, SYSTEM).
            text: The message content.
        """
        if len(text) > _MAX_MESSAGE_LENGTH:
            text = text[:_MAX_MESSAGE_LENGTH] + "..."
        msg = ChatMessage(role=role, text=text)
        self._messages.append(msg)
        # Reset scroll to bottom when new message arrives
        self._scroll_offset = 0

    def start_streaming(self) -> None:
        """Begin a streaming creature response."""
        self._streaming = True
        self._stream_text = ""

    def append_stream(self, chunk: str) -> None:
        """Append a chunk to the current streaming response.

        Args:
            chunk: Text chunk to append.
        """
        if self._streaming:
            self._stream_text += chunk

    def finish_streaming(self) -> None:
        """Complete the streaming response and add it to history."""
        if self._streaming:
            if self._stream_text:
                self.add_message(MessageRole.ASSISTANT, self._stream_text)
            self._streaming = False
            self._stream_text = ""

    def clear_messages(self) -> None:
        """Clear all message history."""
        self._messages.clear()
        self._scroll_offset = 0

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle a Pygame event for the chat panel.

        Args:
            event: The Pygame event to process.

        Returns:
            True if the event was consumed by the panel.
        """
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_TAB:
                self.visible = not self.visible
                return True

            if not self.visible:
                return False

            return self._handle_key(event)

        return False

    def _handle_key(self, event: pygame.event.Event) -> bool:
        """Handle a KEYDOWN event for text input.

        Args:
            event: The KEYDOWN event.

        Returns:
            True if the event was consumed.
        """
        key = event.key

        if key == pygame.K_RETURN or key == pygame.K_KP_ENTER:
            self._submit_input()
            return True
        elif key == pygame.K_BACKSPACE:
            if self._cursor_pos > 0:
                self._input_text = (
                    self._input_text[: self._cursor_pos - 1]
                    + self._input_text[self._cursor_pos :]
                )
                self._cursor_pos -= 1
            return True
        elif key == pygame.K_DELETE:
            if self._cursor_pos < len(self._input_text):
                self._input_text = (
                    self._input_text[: self._cursor_pos]
                    + self._input_text[self._cursor_pos + 1 :]
                )
            return True
        elif key == pygame.K_HOME:
            self._cursor_pos = 0
            return True
        elif key == pygame.K_END:
            self._cursor_pos = len(self._input_text)
            return True
        elif key == pygame.K_LEFT:
            if self._cursor_pos > 0:
                self._cursor_pos -= 1
            return True
        elif key == pygame.K_RIGHT:
            if self._cursor_pos < len(self._input_text):
                self._cursor_pos += 1
            return True
        elif key == pygame.K_PAGEUP:
            self._scroll_up(5)
            return True
        elif key == pygame.K_PAGEDOWN:
            self._scroll_down(5)
            return True
        elif hasattr(event, "unicode") and event.unicode and event.unicode.isprintable():
            self._input_text = (
                self._input_text[: self._cursor_pos]
                + event.unicode
                + self._input_text[self._cursor_pos :]
            )
            self._cursor_pos += len(event.unicode)
            return True

        return False

    def _submit_input(self) -> None:
        """Submit the current input text."""
        text = self._input_text.strip()
        if not text:
            return

        self.add_message(MessageRole.USER, text)
        self._input_text = ""
        self._cursor_pos = 0

        if self._on_submit is not None:
            try:
                self._on_submit(text)
            except Exception as exc:
                logger.error("Chat submit callback error: %s", exc)

    def _scroll_up(self, lines: int = 1) -> None:
        """Scroll the message area up (toward older messages)."""
        max_scroll = max(0, self._total_content_height - self._msg_area_rect[3])
        self._scroll_offset = min(self._scroll_offset + lines * self._font_height, max_scroll)

    def _scroll_down(self, lines: int = 1) -> None:
        """Scroll the message area down (toward newer messages)."""
        self._scroll_offset = max(0, self._scroll_offset - lines * self._font_height)

    def update(self, dt: float) -> None:
        """Update the chat panel animation state.

        Args:
            dt: Delta time in seconds.
        """
        # Cursor blink
        self._cursor_timer += dt
        if self._cursor_timer >= _CURSOR_BLINK_RATE:
            self._cursor_timer -= _CURSOR_BLINK_RATE
            self._cursor_visible = not self._cursor_visible

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle a mouse click on the chat panel.

        Args:
            mx: Mouse x position.
            my: Mouse y position.

        Returns:
            True if the click was on the Send button and consumed.
        """
        sx, sy, sw, sh = self._send_rect
        if sx <= mx <= sx + sw and sy <= my <= sy + sh:
            self._submit_input()
            return True
        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Update hover state for Send button.

        Args:
            mx: Mouse x position.
            my: Mouse y position.
        """
        sx, sy, sw, sh = self._send_rect
        self._send_hover = sx <= mx <= sx + sw and sy <= my <= sy + sh

    def render(self, surface: pygame.Surface) -> None:
        """Render the chat panel onto the given surface.

        Args:
            surface: The Pygame surface to draw on.
        """
        if not self.visible:
            return

        self._ensure_font()
        if self._font is None:
            return

        sw = surface.get_width()
        sh = surface.get_height()
        self._calculate_layout(sw, sh)

        # Draw panel background
        self._render_panel_bg(surface)

        # Draw header
        self._render_header(surface)

        # Draw messages
        self._render_messages(surface)

        # Draw input area
        self._render_input(surface)

        # Draw Send button
        self._render_send_button(surface)

    def _render_panel_bg(self, surface: pygame.Surface) -> None:
        """Draw the semi-transparent panel background."""
        px, py, pw, ph = self._panel_rect
        bg_surface = pygame.Surface((pw, ph), pygame.SRCALPHA)
        bg_surface.fill(_PANEL_BG)
        surface.blit(bg_surface, (px, py))

    def _render_header(self, surface: pygame.Surface) -> None:
        """Draw the 'Chat' header bar at top of panel."""
        if self._font is None:
            return
        px, py, pw, _ = self._panel_rect
        header_surf = pygame.Surface((pw, _HEADER_HEIGHT), pygame.SRCALPHA)
        header_surf.fill(_HEADER_BG)
        surface.blit(header_surf, (px, py))
        text_surf = self._font.render("Chat", True, _HEADER_TEXT_COLOR)
        tx = px + _INPUT_PADDING
        ty = py + (_HEADER_HEIGHT - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

    def _render_send_button(self, surface: pygame.Surface) -> None:
        """Draw the Send button next to the input field."""
        if self._font is None:
            return
        sx, sy, sw, sh = self._send_rect
        bg = _SEND_BUTTON_BG_HOVER if self._send_hover else _SEND_BUTTON_BG
        btn_rect = pygame.Rect(sx, sy, sw, sh)
        pygame.draw.rect(surface, bg, btn_rect)
        pygame.draw.rect(surface, _SEND_BUTTON_BORDER, btn_rect, 1)
        text_surf = self._font.render("Send", True, _SEND_BUTTON_TEXT)
        tx = sx + (sw - text_surf.get_width()) // 2
        ty = sy + (sh - text_surf.get_height()) // 2
        surface.blit(text_surf, (tx, ty))

    def _render_messages(self, surface: pygame.Surface) -> None:
        """Render the scrollable message history."""
        if self._font is None:
            return

        mx, my, mw, mh = self._msg_area_rect
        max_bubble_w = int(mw * 0.75)

        # Build layout for all messages + streaming
        rendered: list[tuple[ChatMessage, list[str], int]] = []
        total_h = 0

        all_msgs = list(self._messages)
        for msg in all_msgs:
            lines = self._wrap_text(msg.text, max_bubble_w - 2 * _MSG_PADDING)
            msg.wrapped_lines = lines
            h = len(lines) * self._font_height + 2 * _MSG_PADDING + _MSG_MARGIN
            msg.height = h
            rendered.append((msg, lines, h))
            total_h += h

        # Streaming message
        if self._streaming and self._stream_text:
            stream_msg = ChatMessage(role=MessageRole.ASSISTANT, text=self._stream_text)
            lines = self._wrap_text(self._stream_text, max_bubble_w - 2 * _MSG_PADDING)
            h = len(lines) * self._font_height + 2 * _MSG_PADDING + _MSG_MARGIN
            rendered.append((stream_msg, lines, h))
            total_h += h

        self._total_content_height = total_h

        # Clamp surface dimensions to valid range
        mw = max(1, min(mw, 8192))
        mh = max(1, min(mh, 8192))

        # Create a clipping surface for the message area
        clip_surface = pygame.Surface((mw, mh), pygame.SRCALPHA)
        clip_surface.fill((0, 0, 0, 0))

        # Draw messages bottom-up, applying scroll offset
        draw_y = mh - self._scroll_offset
        for msg, lines, h in reversed(rendered):
            draw_y -= h
            if draw_y > mh:
                continue
            if draw_y + h < 0:
                break

            is_user = msg.role == MessageRole.USER
            bubble_w = 0
            for line in lines:
                lw = self._font.size(line)[0]
                bubble_w = max(bubble_w, lw)
            bubble_w += 2 * _MSG_PADDING
            bubble_h = len(lines) * self._font_height + 2 * _MSG_PADDING

            if is_user:
                bx = mw - bubble_w - _SCROLLBAR_WIDTH - 2
            else:
                bx = 2

            # Bubble background
            bg_color = _USER_BG if is_user else _CREATURE_BG
            bubble_surf = pygame.Surface((bubble_w, bubble_h), pygame.SRCALPHA)
            bubble_surf.fill(bg_color)
            clip_surface.blit(bubble_surf, (bx, draw_y))

            # Text
            text_color = _USER_MSG_COLOR if is_user else _CREATURE_MSG_COLOR
            if msg.role == MessageRole.SYSTEM:
                text_color = _SYSTEM_MSG_COLOR
            for i, line in enumerate(lines):
                line_surf = self._font.render(line, True, text_color)
                clip_surface.blit(
                    line_surf,
                    (bx + _MSG_PADDING, draw_y + _MSG_PADDING + i * self._font_height),
                )

        # Scrollbar
        if total_h > mh and mh > 0:
            visible_ratio = mh / total_h
            thumb_h = max(20, int(mh * visible_ratio))
            max_scroll = total_h - mh
            if max_scroll > 0:
                scroll_ratio = self._scroll_offset / max_scroll
            else:
                scroll_ratio = 0.0
            thumb_y = int((mh - thumb_h) * (1.0 - scroll_ratio))
            sb_x = mw - _SCROLLBAR_WIDTH

            sb_bg = pygame.Surface((_SCROLLBAR_WIDTH, mh), pygame.SRCALPHA)
            sb_bg.fill(_SCROLLBAR_BG)
            clip_surface.blit(sb_bg, (sb_x, 0))

            sb_thumb = pygame.Surface((_SCROLLBAR_WIDTH, thumb_h), pygame.SRCALPHA)
            sb_thumb.fill(_SCROLLBAR_THUMB)
            clip_surface.blit(sb_thumb, (sb_x, thumb_y))

        surface.blit(clip_surface, (mx, my))

    def _render_input(self, surface: pygame.Surface) -> None:
        """Render the text input area."""
        if self._font is None:
            return

        ix, iy, iw, ih = self._input_rect

        # Input background
        input_bg = pygame.Surface((iw, ih), pygame.SRCALPHA)
        input_bg.fill(_INPUT_BG)
        surface.blit(input_bg, (ix, iy))

        # Border
        border_color = _INPUT_BORDER_FOCUS if self.visible else _INPUT_BORDER
        pygame.draw.rect(surface, border_color, (ix, iy, iw, ih), 1)

        # Text or placeholder
        text_x = ix + _INPUT_PADDING
        text_y = iy + (ih - self._font_height) // 2

        if self._input_text:
            text_surf = self._font.render(self._input_text, True, _USER_MSG_COLOR)
            # Scroll text if it extends beyond input width
            text_w = text_surf.get_width()
            visible_w = iw - 2 * _INPUT_PADDING
            if text_w > visible_w:
                # Ensure cursor is visible by scrolling
                cursor_x = self._font.size(self._input_text[: self._cursor_pos])[0]
                offset = max(0, cursor_x - visible_w + 20)
                # Create clipped subsurface view
                clipped = pygame.Surface((visible_w, ih), pygame.SRCALPHA)
                clipped.blit(text_surf, (-offset, 0))
                surface.blit(clipped, (text_x, text_y))
            else:
                surface.blit(text_surf, (text_x, text_y))
        else:
            placeholder = self._font.render("Type a message...", True, _PLACEHOLDER_COLOR)
            surface.blit(placeholder, (text_x, text_y))

        # Cursor
        if self._cursor_visible and self.visible:
            cursor_text = self._input_text[: self._cursor_pos]
            cursor_x_offset = self._font.size(cursor_text)[0]
            # Adjust for text scrolling
            text_w = self._font.size(self._input_text)[0] if self._input_text else 0
            visible_w = iw - 2 * _INPUT_PADDING
            if text_w > visible_w:
                full_cursor_x = cursor_x_offset
                offset = max(0, full_cursor_x - visible_w + 20)
                cursor_x_offset -= offset

            cx = text_x + cursor_x_offset
            cy_top = text_y + 1
            cy_bot = text_y + self._font_height - 1
            pygame.draw.line(surface, _CURSOR_COLOR, (cx, cy_top), (cx, cy_bot), 1)
