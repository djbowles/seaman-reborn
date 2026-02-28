"""Lineage manager overlay panel for managing creature bloodlines.

Provides a centered overlay panel (similar to SettingsPanel) that lets
the user browse, create, load, rename, and delete bloodline save
directories. Accessible via HUD button or F2 shortcut.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pygame

from seaman_brain.creature.persistence import BloodlineInfo, StatePersistence
from seaman_brain.creature.state import CreatureState
from seaman_brain.gui.widgets import Button

logger = logging.getLogger(__name__)

# ── Colors ───────────────────────────────────────────────────────────

_OVERLAY_BG = (0, 0, 0, 160)
_PANEL_BG = (18, 28, 48)
_PANEL_BORDER = (50, 70, 100)
_HEADER_BG = (12, 22, 42)
_TITLE_COLOR = (200, 220, 240)
_TEXT_COLOR = (180, 200, 220)
_TEXT_DIM = (120, 140, 160)
_SELECTED_BG = (40, 65, 100)
_HOVER_BG = (30, 50, 80)
_STATUS_GREEN = (60, 200, 100)
_ACTIVE_BADGE = (60, 200, 100)
_CLOSE_COLOR = (220, 80, 80)

_FONT_SIZE = 14
_TITLE_FONT_SIZE = 20
_PANEL_WIDTH = 700
_PANEL_HEIGHT = 520
_HEADER_HEIGHT = 36
_LIST_WIDTH = 260
_ITEM_HEIGHT = 48
_CONTENT_PADDING = 16


class LineagePanel:
    """Overlay panel for managing creature bloodlines.

    Shows a list of bloodlines on the left and details on the right.
    Supports new, load, delete, and rename operations.

    Attributes:
        visible: Whether the panel is currently shown.
    """

    def __init__(
        self,
        screen_width: int = 1024,
        screen_height: int = 768,
        save_base_dir: str = "data/saves",
        on_switch: Callable[[str], Any] | None = None,
        on_new: Callable[[str], Any] | None = None,
        on_delete: Callable[[str], Any] | None = None,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        self._screen_w = screen_width
        self._screen_h = screen_height
        self._save_base_dir = save_base_dir
        self.on_switch = on_switch
        self.on_new = on_new
        self.on_delete = on_delete
        self.on_close = on_close

        self.visible = False

        # Panel positioning
        self._panel_x = (screen_width - _PANEL_WIDTH) // 2
        self._panel_y = (screen_height - _PANEL_HEIGHT) // 2

        # Fonts (lazy)
        self._font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None

        # State
        self._bloodlines: list[BloodlineInfo] = []
        self._selected_index = 0
        self._hovered_index = -1
        self._active_name = "default"
        self._scroll_offset = 0
        self._status_text = ""

        # Confirm delete state
        self._confirm_delete = False

        # Rename state
        self._rename_active = False
        self._rename_text = ""
        self._rename_cursor = 0

            # Widgets
        self._close_button: Button | None = None
        self._new_button: Button | None = None
        self._load_button: Button | None = None
        self._delete_button: Button | None = None
        self._rename_button: Button | None = None
        self._confirm_yes: Button | None = None
        self._confirm_no: Button | None = None
        self._widgets_built = False

    def _ensure_fonts(self) -> None:
        """Initialize fonts if not yet done."""
        if self._font is None:
            for name in ("consolas", "couriernew", "courier"):
                try:
                    self._font = pygame.font.SysFont(name, _FONT_SIZE)
                    self._title_font = pygame.font.SysFont(
                        name, _TITLE_FONT_SIZE, bold=True
                    )
                    break
                except Exception:
                    continue
            if self._font is None:
                self._font = pygame.font.Font(None, _FONT_SIZE)
                self._title_font = pygame.font.Font(None, _TITLE_FONT_SIZE)

    def _build_widgets(self) -> None:
        """Build action buttons. Called once after fonts are ready."""
        if self._widgets_built:
            return
        self._widgets_built = True

        px = self._panel_x
        py = self._panel_y
        right_x = px + _LIST_WIDTH + 2 * _CONTENT_PADDING
        btn_y = py + _PANEL_HEIGHT - 60
        btn_w = 80
        btn_h = 30
        btn_gap = 8

        self._close_button = Button(
            px + _PANEL_WIDTH - 36, py + 4, 28, 28, "X",
            on_click=self._close,
        )

        self._new_button = Button(
            right_x, btn_y, btn_w, btn_h, "New",
            on_click=self._on_new_click,
        )
        self._load_button = Button(
            right_x + btn_w + btn_gap, btn_y, btn_w, btn_h, "Load",
            on_click=self._on_load_click,
        )
        self._rename_button = Button(
            right_x + 2 * (btn_w + btn_gap), btn_y, btn_w, btn_h, "Rename",
            on_click=self._on_rename_click,
        )
        self._delete_button = Button(
            right_x + 3 * (btn_w + btn_gap), btn_y, btn_w, btn_h, "Delete",
            on_click=self._on_delete_click,
        )

        # Confirmation buttons (hidden by default)
        self._confirm_yes = Button(
            right_x + 50, btn_y - 40, 80, 28, "Yes",
            on_click=self._confirm_delete_yes,
        )
        self._confirm_no = Button(
            right_x + 140, btn_y - 40, 80, 28, "No",
            on_click=self._confirm_delete_no,
        )

    # ── Public interface ──────────────────────────────────────────────

    def open(self) -> None:
        """Show the lineage panel and refresh the bloodline list."""
        self.visible = True
        self._confirm_delete = False
        self._status_text = ""
        self.refresh_list()

    def close(self) -> None:
        """Hide the lineage panel."""
        self.visible = False
        self._confirm_delete = False

    def refresh_list(self) -> None:
        """Reload bloodlines from disk."""
        try:
            # Ensure migration has happened
            StatePersistence.migrate_flat_saves(self._save_base_dir)
            self._bloodlines = StatePersistence.list_bloodlines(self._save_base_dir)
            self._active_name = StatePersistence.get_active_bloodline(
                self._save_base_dir
            )
        except Exception as exc:
            logger.error("Failed to refresh bloodline list: %s", exc, exc_info=True)
            self._bloodlines = []
            self._status_text = f"Error loading bloodlines: {exc}"

        # Clamp selection
        if self._bloodlines:
            self._selected_index = min(
                self._selected_index, len(self._bloodlines) - 1
            )
        else:
            self._selected_index = 0

    # ── Rendering ─────────────────────────────────────────────────────

    def render(self, surface: pygame.Surface) -> None:
        """Render the lineage overlay onto the surface."""
        if not self.visible:
            return

        self._ensure_fonts()
        self._build_widgets()

        if self._font is None or self._title_font is None:
            return

        px = self._panel_x
        py = self._panel_y

        # Semi-transparent overlay
        overlay = pygame.Surface(
            (self._screen_w, self._screen_h), pygame.SRCALPHA
        )
        overlay.fill(_OVERLAY_BG)
        surface.blit(overlay, (0, 0))

        # Panel background
        pygame.draw.rect(
            surface, _PANEL_BG, (px, py, _PANEL_WIDTH, _PANEL_HEIGHT)
        )
        pygame.draw.rect(
            surface, _PANEL_BORDER, (px, py, _PANEL_WIDTH, _PANEL_HEIGHT), 1
        )

        # Header
        pygame.draw.rect(
            surface, _HEADER_BG, (px, py, _PANEL_WIDTH, _HEADER_HEIGHT)
        )
        title = self._title_font.render("Lineage Manager", True, _TITLE_COLOR)
        surface.blit(title, (px + 12, py + 6))

        # Close button
        if self._close_button is not None:
            self._close_button.render(surface)

        # Left: bloodline list
        self._render_list(surface)

        # Right: details
        self._render_details(surface)

        # Action buttons (hidden during rename)
        if not self._rename_active:
            if self._new_button is not None:
                self._new_button.render(surface)
            if self._load_button is not None:
                self._load_button.render(surface)
            if self._rename_button is not None:
                self._rename_button.render(surface)
            if self._delete_button is not None:
                self._delete_button.render(surface)
        else:
            self._render_rename_input(surface)

        # Confirmation overlay
        if self._confirm_delete:
            self._render_confirm(surface)

        # Status bar
        if self._status_text and self._font is not None:
            status = self._font.render(self._status_text, True, _TEXT_DIM)
            surface.blit(status, (px + 12, py + _PANEL_HEIGHT - 24))

    def _render_list(self, surface: pygame.Surface) -> None:
        """Render the bloodline list on the left side."""
        if self._font is None:
            return

        px = self._panel_x + _CONTENT_PADDING
        py = self._panel_y + _HEADER_HEIGHT + _CONTENT_PADDING
        max_visible = (_PANEL_HEIGHT - _HEADER_HEIGHT - 80) // _ITEM_HEIGHT

        for i in range(min(max_visible, len(self._bloodlines))):
            idx = i + self._scroll_offset
            if idx >= len(self._bloodlines):
                break

            bl = self._bloodlines[idx]
            item_y = py + i * _ITEM_HEIGHT

            # Background (selected / hovered)
            if idx == self._selected_index:
                pygame.draw.rect(
                    surface, _SELECTED_BG,
                    (px, item_y, _LIST_WIDTH, _ITEM_HEIGHT),
                )
            elif idx == self._hovered_index:
                pygame.draw.rect(
                    surface, _HOVER_BG,
                    (px, item_y, _LIST_WIDTH, _ITEM_HEIGHT),
                )

            # Name
            name_surf = self._font.render(bl.name, True, _TEXT_COLOR)
            surface.blit(name_surf, (px + 8, item_y + 4))

            # Stage and gen count
            info = f"{bl.stage} · Gen {bl.generation_count}"
            info_surf = self._font.render(info, True, _TEXT_DIM)
            surface.blit(info_surf, (px + 8, item_y + 22))

            # Active badge
            if bl.name == self._active_name:
                badge = self._font.render("[active]", True, _ACTIVE_BADGE)
                bx = px + _LIST_WIDTH - badge.get_width() - 8
                surface.blit(badge, (bx, item_y + 4))

    def _render_details(self, surface: pygame.Surface) -> None:
        """Render bloodline details on the right side."""
        if self._font is None:
            return

        if not self._bloodlines:
            no_data = self._font.render(
                "No bloodlines found. Create one!", True, _TEXT_DIM
            )
            rx = self._panel_x + _LIST_WIDTH + 2 * _CONTENT_PADDING
            ry = self._panel_y + _HEADER_HEIGHT + _CONTENT_PADDING + 20
            surface.blit(no_data, (rx, ry))
            return

        if self._selected_index >= len(self._bloodlines):
            return

        bl = self._bloodlines[self._selected_index]
        rx = self._panel_x + _LIST_WIDTH + 2 * _CONTENT_PADDING
        ry = self._panel_y + _HEADER_HEIGHT + _CONTENT_PADDING

        lines = [
            f"Name: {bl.name}",
            f"Stage: {bl.stage}",
            f"Generations: {bl.generation_count}",
            f"Last played: {bl.last_modified:%Y-%m-%d %H:%M}",
            f"Path: {bl.save_dir}",
        ]

        if bl.name == self._active_name:
            lines.append("Status: ACTIVE")

        for i, line in enumerate(lines):
            color = _STATUS_GREEN if "ACTIVE" in line else _TEXT_COLOR
            surf = self._font.render(line, True, color)
            surface.blit(surf, (rx, ry + i * 24))

    def _render_confirm(self, surface: pygame.Surface) -> None:
        """Render delete confirmation UI."""
        if self._font is None:
            return

        rx = self._panel_x + _LIST_WIDTH + 2 * _CONTENT_PADDING
        ry = self._panel_y + _PANEL_HEIGHT - 100
        msg = self._font.render("Delete this bloodline?", True, _CLOSE_COLOR)
        surface.blit(msg, (rx + 50, ry))

        if self._confirm_yes is not None:
            self._confirm_yes.render(surface)
        if self._confirm_no is not None:
            self._confirm_no.render(surface)

    # ── Event handling ────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle mouse click. Returns True if consumed."""
        if not self.visible:
            return False

        # Close button
        if self._close_button is not None and self._close_button.handle_click(mx, my):
            return True

        # Confirm buttons
        if self._confirm_delete:
            if self._confirm_yes is not None and self._confirm_yes.handle_click(mx, my):
                return True
            if self._confirm_no is not None and self._confirm_no.handle_click(mx, my):
                return True
            return True  # Consume click during confirmation

        # Action buttons
        if not self._rename_active:
            if self._new_button is not None and self._new_button.handle_click(mx, my):
                return True
            if self._load_button is not None and self._load_button.handle_click(mx, my):
                return True
            if (
                self._rename_button is not None
                and self._rename_button.handle_click(mx, my)
            ):
                return True
            if (
                self._delete_button is not None
                and self._delete_button.handle_click(mx, my)
            ):
                return True

        # List item click
        list_x = self._panel_x + _CONTENT_PADDING
        list_y = self._panel_y + _HEADER_HEIGHT + _CONTENT_PADDING
        if list_x <= mx <= list_x + _LIST_WIDTH and my >= list_y:
            clicked_i = (my - list_y) // _ITEM_HEIGHT + self._scroll_offset
            if 0 <= clicked_i < len(self._bloodlines):
                self._selected_index = clicked_i
                return True

        # Consume click inside panel
        panel_rect = pygame.Rect(
            self._panel_x, self._panel_y, _PANEL_WIDTH, _PANEL_HEIGHT
        )
        if panel_rect.collidepoint(mx, my):
            return True

        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Handle mouse motion for hover states."""
        if not self.visible:
            return

        if self._close_button is not None:
            self._close_button.handle_mouse_move(mx, my)

        for btn in (
            self._new_button, self._load_button, self._rename_button,
            self._delete_button, self._confirm_yes, self._confirm_no,
        ):
            if btn is not None:
                btn.handle_mouse_move(mx, my)

        # List hover
        list_x = self._panel_x + _CONTENT_PADDING
        list_y = self._panel_y + _HEADER_HEIGHT + _CONTENT_PADDING
        if list_x <= mx <= list_x + _LIST_WIDTH and my >= list_y:
            idx = (my - list_y) // _ITEM_HEIGHT + self._scroll_offset
            if 0 <= idx < len(self._bloodlines):
                self._hovered_index = idx
            else:
                self._hovered_index = -1
        else:
            self._hovered_index = -1

    # ── Key event handling (rename) ──────────────────────────────────

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle a Pygame event (key presses during rename mode).

        Args:
            event: A Pygame event.

        Returns:
            True if the event was consumed.
        """
        if not self.visible or not self._rename_active:
            return False

        if event.type != pygame.KEYDOWN:
            return False

        if event.key == pygame.K_RETURN or event.key == getattr(pygame, "K_KP_ENTER", 271):
            self._commit_rename()
            return True
        if event.key == pygame.K_ESCAPE:
            self._cancel_rename()
            return True
        if event.key == pygame.K_BACKSPACE:
            if self._rename_cursor > 0:
                self._rename_text = (
                    self._rename_text[: self._rename_cursor - 1]
                    + self._rename_text[self._rename_cursor:]
                )
                self._rename_cursor -= 1
            return True
        if event.key == pygame.K_DELETE:
            if self._rename_cursor < len(self._rename_text):
                self._rename_text = (
                    self._rename_text[: self._rename_cursor]
                    + self._rename_text[self._rename_cursor + 1:]
                )
            return True
        if event.key == pygame.K_LEFT:
            self._rename_cursor = max(0, self._rename_cursor - 1)
            return True
        if event.key == pygame.K_RIGHT:
            self._rename_cursor = min(len(self._rename_text), self._rename_cursor + 1)
            return True
        if event.key == pygame.K_HOME:
            self._rename_cursor = 0
            return True
        if event.key == pygame.K_END:
            self._rename_cursor = len(self._rename_text)
            return True

        # Printable character
        if hasattr(event, "unicode") and event.unicode and event.unicode.isprintable():
            ch = event.unicode
            self._rename_text = (
                self._rename_text[: self._rename_cursor]
                + ch
                + self._rename_text[self._rename_cursor:]
            )
            self._rename_cursor += 1
            return True

        return True  # Consume all keys during rename

    def _on_rename_click(self) -> None:
        """Enter rename mode with the selected bloodline's name."""
        if not self._bloodlines:
            return
        bl = self._bloodlines[self._selected_index]
        self._rename_active = True
        self._rename_text = bl.name
        self._rename_cursor = len(bl.name)
        self._confirm_delete = False

    def _commit_rename(self) -> None:
        """Apply the rename and exit rename mode."""
        if not self._bloodlines:
            self._cancel_rename()
            return

        bl = self._bloodlines[self._selected_index]
        new_name = self._rename_text.strip()

        if not new_name or new_name == bl.name:
            self._cancel_rename()
            return

        try:
            StatePersistence.rename_bloodline(
                bl.name, new_name, self._save_base_dir
            )
            self._status_text = f"Renamed: {bl.name} -> {new_name}"
            self.refresh_list()
            # Re-select the renamed bloodline
            for i, b in enumerate(self._bloodlines):
                if b.name == new_name:
                    self._selected_index = i
                    break
        except (ValueError, FileNotFoundError, OSError) as exc:
            self._status_text = f"Rename failed: {exc}"
            logger.error("Rename failed: %s", exc)

        self._rename_active = False

    def _cancel_rename(self) -> None:
        """Exit rename mode without applying."""
        self._rename_active = False
        self._rename_text = ""
        self._rename_cursor = 0

    def _render_rename_input(self, surface: pygame.Surface) -> None:
        """Render the text input box for rename mode."""
        if self._font is None:
            return

        px = self._panel_x
        py = self._panel_y
        right_x = px + _LIST_WIDTH + 2 * _CONTENT_PADDING
        input_y = py + _PANEL_HEIGHT - 60
        input_w = _PANEL_WIDTH - _LIST_WIDTH - 3 * _CONTENT_PADDING
        input_h = 30

        # Background
        pygame.draw.rect(
            surface, (25, 40, 65), (right_x, input_y, input_w, input_h)
        )
        pygame.draw.rect(
            surface, (100, 140, 200), (right_x, input_y, input_w, input_h), 1
        )

        # Text
        display_text = self._rename_text
        text_surf = self._font.render(display_text, True, _TEXT_COLOR)
        surface.blit(text_surf, (right_x + 6, input_y + 7))

        # Cursor
        cursor_x = right_x + 6 + self._font.size(
            self._rename_text[: self._rename_cursor]
        )[0]
        pygame.draw.line(
            surface, _TEXT_COLOR,
            (cursor_x, input_y + 4), (cursor_x, input_y + input_h - 4),
        )

        # Hint text below
        hint = self._font.render("Enter=save  Esc=cancel", True, _TEXT_DIM)
        surface.blit(hint, (right_x, input_y + input_h + 4))

    # ── Callbacks ─────────────────────────────────────────────────────

    def _close(self) -> None:
        """Close the panel and notify the owner to restore game state."""
        self.close()
        if self.on_close is not None:
            self.on_close()

    def _on_new_click(self) -> None:
        """Handle New button click — create a bloodline."""
        # Generate a unique name
        existing = {bl.name for bl in self._bloodlines}
        name = "bloodline_1"
        counter = 1
        while name in existing:
            counter += 1
            name = f"bloodline_{counter}"

        base = Path(self._save_base_dir) / name
        base.mkdir(parents=True, exist_ok=True)

        # Create a fresh creature save
        fresh = CreatureState()
        persistence = StatePersistence(save_dir=base)
        persistence.save(fresh)

        self._status_text = f"Created: {name}"
        self.refresh_list()

        if self.on_new is not None:
            self.on_new(name)

    def _on_load_click(self) -> None:
        """Handle Load button — switch to selected bloodline."""
        if not self._bloodlines:
            return

        bl = self._bloodlines[self._selected_index]
        if bl.name == self._active_name:
            self._status_text = f"{bl.name} is already active"
            return

        StatePersistence.set_active_bloodline(bl.name, self._save_base_dir)
        self._active_name = bl.name
        self._status_text = f"Switched to: {bl.name}"

        if self.on_switch is not None:
            self.on_switch(bl.name)

    def _on_delete_click(self) -> None:
        """Handle Delete button — show confirmation."""
        if not self._bloodlines:
            return

        bl = self._bloodlines[self._selected_index]
        if bl.name == self._active_name:
            self._status_text = "Cannot delete the active bloodline"
            return

        self._confirm_delete = True

    def _confirm_delete_yes(self) -> None:
        """Confirmed delete — remove the bloodline directory."""
        self._confirm_delete = False
        if not self._bloodlines:
            return

        bl = self._bloodlines[self._selected_index]
        if bl.name == self._active_name:
            self._status_text = "Cannot delete the active bloodline"
            return

        try:
            shutil.rmtree(bl.save_dir)
            self._status_text = f"Deleted: {bl.name}"
            self._selected_index = max(0, self._selected_index - 1)
            self.refresh_list()
            if self.on_delete is not None:
                self.on_delete(bl.name)
        except Exception as exc:
            self._status_text = f"Delete failed: {exc}"
            logger.error("Failed to delete bloodline %s: %s", bl.name, exc)

    def _confirm_delete_no(self) -> None:
        """Cancel delete confirmation."""
        self._confirm_delete = False
