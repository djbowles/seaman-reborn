"""Glass overlay chat panel with message bubbles.

Renders over the tank area: semi-transparent dark glass background,
system messages (no bubble, dim text), creature messages (left-aligned
warm bubble), user messages (right-aligned lighter bubble), pill-shaped
input bar at bottom, auto-scroll with scroll-lock on manual scroll-up.
"""
from __future__ import annotations

from collections.abc import Callable

import pygame

from seaman_brain.gui.layout import ScreenLayout
from seaman_brain.gui.theme import Colors, Fonts

# ── Constants ────────────────────────────────────────────────────────

_MAX_HISTORY = 200
_INPUT_HEIGHT = 32
_MSG_PADDING = 8
_BUBBLE_RADIUS = 6
_SCROLLBAR_W = 6
_GLASS_ALPHA = 216  # ~85% of 255

# Bubble colors (RGBA)
_CREATURE_BUBBLE = (40, 35, 25)
_USER_BUBBLE = (30, 30, 40)
_SYSTEM_COLOR = Colors.TEXT_30


class ChatPanel:
    """Glass overlay chat panel with message bubbles and text input."""

    def __init__(self, layout: ScreenLayout) -> None:
        self._layout = layout
        self._font: pygame.font.Font | None = None

        self._messages: list[dict] = []
        self._input_text: str = ""
        self._input_focused: bool = True
        self._auto_scroll: bool = True
        self._scroll_offset: int = 0
        self._cursor_blink: float = 0.0

        self.on_submit: Callable[[str], None] | None = None

    def resize(self, layout: ScreenLayout) -> None:
        """Update layout reference on window resize."""
        self._layout = layout

    def _ensure_font(self) -> None:
        if self._font is None:
            if Fonts.body is not None:
                self._font = Fonts.body
            else:
                for name in ("consolas", "couriernew", "courier"):
                    try:
                        self._font = pygame.font.SysFont(name, 11)
                        return
                    except Exception:
                        continue
                self._font = pygame.font.Font(None, 11)

    # ── Messages ─────────────────────────────────────────────────────

    def add_message(
        self, role: str, text: str, streaming: bool = False
    ) -> None:
        """Add a message to the chat history."""
        self._messages.append({
            "role": role,
            "text": text,
            "streaming": streaming,
        })
        if len(self._messages) > _MAX_HISTORY:
            self._messages = self._messages[-_MAX_HISTORY:]
        if self._auto_scroll:
            self._scroll_to_end()

    def update_streaming(self, text: str) -> None:
        """Update the last streaming message's text."""
        if self._messages and self._messages[-1].get("streaming"):
            self._messages[-1]["text"] = text
            if self._auto_scroll:
                self._scroll_to_end()

    def finish_streaming(self) -> None:
        """Mark the last streaming message as complete."""
        if self._messages and self._messages[-1].get("streaming"):
            self._messages[-1]["streaming"] = False

    # ── Input ────────────────────────────────────────────────────────

    def handle_key(self, key: int, char: str) -> None:
        """Handle a keyboard event in the input bar."""
        if not self._input_focused:
            return
        if key == 13:  # K_RETURN
            self._submit()
        elif key == 8:  # K_BACKSPACE
            self._input_text = self._input_text[:-1]
        elif char and ord(char) >= 32:
            self._input_text += char

    def _submit(self) -> None:
        """Submit the current input text."""
        text = self._input_text.strip()
        if not text:
            return
        self._input_text = ""
        if self.on_submit is not None:
            self.on_submit(text)

    # ── Scrolling ────────────────────────────────────────────────────

    def handle_scroll(self, direction: int) -> None:
        """Scroll the message list. Positive = up, negative = down."""
        if direction > 0:
            self._scroll_offset = max(0, self._scroll_offset - 1)
            self._auto_scroll = False
        elif direction < 0:
            self._scroll_offset += 1
            total = len(self._messages)
            if self._scroll_offset >= total:
                self._scroll_offset = max(0, total - 1)

    def scroll_to_bottom(self) -> None:
        """Scroll to the most recent message and re-enable auto-scroll."""
        self._auto_scroll = True
        self._scroll_to_end()

    def _scroll_to_end(self) -> None:
        self._scroll_offset = max(0, len(self._messages) - 1)

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle click — focus input if in chat area."""
        chat = self._layout.chat
        if (chat.x <= mx < chat.x + chat.w
                and chat.y <= my < chat.y + chat.h):
            self._input_focused = True
            return True
        return False

    # ── Rendering ────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface) -> None:
        """Render the glass overlay, message bubbles, and input bar."""
        self._ensure_font()
        if self._font is None:
            return

        chat = self._layout.chat
        font = self._font
        line_h = font.get_linesize()

        # Glass background
        glass = pygame.Surface((chat.w, chat.h), pygame.SRCALPHA)
        glass.fill((8, 8, 8, _GLASS_ALPHA))
        surface.blit(glass, (chat.x, chat.y))

        # Top border
        pygame.draw.line(
            surface, Colors.BORDER[:3],
            (chat.x, chat.y), (chat.x + chat.w, chat.y), 1,
        )

        # Message area (above input)
        msg_area_h = chat.h - _INPUT_HEIGHT - 4
        msg_y = chat.y + msg_area_h  # bottom of message area

        # Render messages bottom-up
        if self._messages:
            visible_bottom = msg_y - 4
            for msg in reversed(self._messages):
                if visible_bottom <= chat.y + 4:
                    break
                visible_bottom = self._render_message(
                    surface, font, msg, chat.x + _MSG_PADDING,
                    chat.x + chat.w - _MSG_PADDING - _SCROLLBAR_W,
                    visible_bottom, line_h,
                )

        # Input bar
        self._render_input(surface, font, chat, line_h)

    def _render_message(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        msg: dict,
        left: int,
        right: int,
        bottom_y: int,
        line_h: int,
    ) -> int:
        """Render a single message bubble. Returns the new bottom_y."""
        role = msg["role"]
        text = msg["text"]
        max_w = right - left - 20

        # Word wrap
        lines = self._wrap_text(font, text, max_w)
        bubble_h = len(lines) * line_h + 8
        top_y = bottom_y - bubble_h - 4

        if role == "system":
            # No bubble, dim centered text
            for i, line in enumerate(lines):
                line_surf = font.render(line, True, _SYSTEM_COLOR)
                x = left + (max_w - line_surf.get_width()) // 2
                surface.blit(line_surf, (x, top_y + 4 + i * line_h))
        elif role == "creature":
            # Left-aligned warm bubble
            bubble_w = min(max_w, max(
                (font.size(ln)[0] for ln in lines), default=50
            ) + 16)
            pygame.draw.rect(
                surface, _CREATURE_BUBBLE,
                (left, top_y, bubble_w, bubble_h),
                border_radius=_BUBBLE_RADIUS,
            )
            for i, line in enumerate(lines):
                line_surf = font.render(line, True, Colors.TEXT_90)
                surface.blit(
                    line_surf, (left + 8, top_y + 4 + i * line_h)
                )
        else:
            # Right-aligned user bubble
            bubble_w = min(max_w, max(
                (font.size(ln)[0] for ln in lines), default=50
            ) + 16)
            bx = right - bubble_w
            pygame.draw.rect(
                surface, _USER_BUBBLE,
                (bx, top_y, bubble_w, bubble_h),
                border_radius=_BUBBLE_RADIUS,
            )
            for i, line in enumerate(lines):
                line_surf = font.render(line, True, Colors.TEXT_90)
                surface.blit(
                    line_surf, (bx + 8, top_y + 4 + i * line_h)
                )

        return top_y

    def _render_input(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        chat: object,
        line_h: int,
    ) -> None:
        """Render the pill-shaped input bar."""
        ix = chat.x + 8
        iy = chat.y + chat.h - _INPUT_HEIGHT - 2
        iw = chat.w - 16
        ih = _INPUT_HEIGHT

        # Pill background
        pygame.draw.rect(
            surface, Colors.SURFACE_5[:3],
            (ix, iy, iw, ih), border_radius=ih // 2,
        )
        pygame.draw.rect(
            surface, Colors.BORDER[:3],
            (ix, iy, iw, ih), 1, border_radius=ih // 2,
        )

        # Text
        display = self._input_text or ""
        text_surf = font.render(display, True, Colors.TEXT_90)
        ty = iy + (ih - text_surf.get_height()) // 2
        surface.blit(text_surf, (ix + 12, ty))

    def _wrap_text(
        self, font: pygame.font.Font, text: str, max_w: int
    ) -> list[str]:
        """Word-wrap text to fit within max_w pixels."""
        words = text.split()
        if not words:
            return [text]
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            w, _ = font.size(test)
            if w <= max_w:
                current = test
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines
