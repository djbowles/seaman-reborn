"""Input routing — keyboard shortcuts and mouse event dispatch.

All keyboard shortcuts and mouse routing live here. GameEngine sets
callback attributes for each action; InputHandler calls them.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pygame


class InputHandler:
    """Routes pygame events to callbacks.

    Attributes set by GameEngine after construction:
        on_escape: Called when Escape pressed
        on_toggle_settings: Called on F2
        on_toggle_mic: Called on M
        on_tab: Called on Tab
        chat_focused: When True, suppress shortcut keys (let chat handle them)
    """

    def __init__(self) -> None:
        self.chat_focused = False

        # Callbacks (set by GameEngine)
        self.on_escape: Callable[[], Any] | None = None
        self.on_toggle_settings: Callable[[], Any] | None = None
        self.on_toggle_mic: Callable[[], Any] | None = None
        self.on_tab: Callable[[], Any] | None = None
        self.on_mouse_click: Callable[[Any], Any] | None = None
        self.on_mouse_move: Callable[[Any], Any] | None = None
        self.on_mouse_up: Callable[[Any], Any] | None = None
        self.on_mouse_scroll: Callable[[Any], Any] | None = None
        self.on_key_down: Callable[[Any], Any] | None = None

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route a single pygame event to the appropriate callback."""
        if event.type == pygame.KEYDOWN:
            self._handle_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.on_mouse_click:
                self.on_mouse_click(event)
        elif event.type == pygame.MOUSEMOTION:
            if self.on_mouse_move:
                self.on_mouse_move(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.on_mouse_up:
                self.on_mouse_up(event)
        elif event.type == pygame.MOUSEWHEEL:
            if self.on_mouse_scroll:
                self.on_mouse_scroll(event)

    def _handle_key(self, event: pygame.event.Event) -> None:
        """Route keyboard events, respecting chat focus."""
        key = event.key

        # Escape and F2 always work regardless of chat focus
        if key == pygame.K_ESCAPE:
            if self.on_escape:
                self.on_escape()
            return
        if key == pygame.K_F2:
            if self.on_toggle_settings:
                self.on_toggle_settings()
            return

        # When chat is focused, pass keys to chat handler instead
        if self.chat_focused:
            if self.on_key_down:
                self.on_key_down(event)
            return

        # Global shortcuts (only when chat not focused)
        if key == pygame.K_m:
            if self.on_toggle_mic:
                self.on_toggle_mic()
        elif key == pygame.K_TAB:
            if self.on_tab:
                self.on_tab()
        elif self.on_key_down:
            self.on_key_down(event)
