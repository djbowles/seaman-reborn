"""Tests for the lineage manager panel (gui/lineage_panel.py).

Pygame is mocked at module level to avoid requiring a display server.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Pygame Mock Setup ─────────────────────────────────────────────────

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None

_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768

_font_mock = MagicMock()
_text_surf_mock = MagicMock()
_text_surf_mock.get_width.return_value = 80
_text_surf_mock.get_height.return_value = 16
_font_mock.render.return_value = _text_surf_mock
_font_mock.get_linesize.return_value = 16
_font_mock.size.return_value = (80, 16)
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock

_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.line.return_value = None


def _make_rect(x, y, w, h):
    r = MagicMock()
    r.x = x
    r.y = y
    r.width = w
    r.height = h
    r.collidepoint = lambda mx, my: x <= mx <= x + w and y <= my <= y + h
    return r


_pygame_mock.Rect = _make_rect


def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 1024
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 768
    return s


_pygame_mock.Surface = _make_surface

sys.modules["pygame"] = _pygame_mock

from seaman_brain.gui.lineage_panel import LineagePanel  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Re-install pygame mock between tests."""
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.lineage_panel as lp_mod
    import seaman_brain.gui.widgets as widgets_mod
    widgets_mod.pygame = _pygame_mock
    widgets_mod._FONT = None
    lp_mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    _pygame_mock.Rect = _make_rect
    _pygame_mock.Surface = _make_surface
    _pygame_mock.font.SysFont.return_value = _font_mock
    _pygame_mock.font.Font.return_value = _font_mock
    _font_mock.render.return_value = _text_surf_mock
    _font_mock.get_linesize.return_value = 16
    _font_mock.size.return_value = (80, 16)
    _text_surf_mock.get_width.return_value = 80
    _text_surf_mock.get_height.return_value = 16


@pytest.fixture()
def save_dir(tmp_path: Path) -> Path:
    """Create a temporary save directory with a bloodline."""
    base = tmp_path / "saves"
    base.mkdir()
    default = base / "default"
    default.mkdir()
    state = {
        "stage": "mushroomer",
        "age": 100.0,
        "interaction_count": 5,
        "mood": "neutral",
        "trust_level": 0.2,
        "hunger": 0.1,
        "health": 0.9,
        "comfort": 0.8,
    }
    (default / "creature.json").write_text(json.dumps(state), encoding="utf-8")
    (base / "_active.txt").write_text("default", encoding="utf-8")
    return base


@pytest.fixture()
def panel(save_dir: Path) -> LineagePanel:
    """Create a lineage panel with the temp save dir."""
    return LineagePanel(save_base_dir=str(save_dir))


class TestLineagePanelConstruction:
    """Tests for LineagePanel initialization."""

    def test_starts_hidden(self, panel: LineagePanel):
        """Panel starts hidden."""
        assert panel.visible is False

    def test_open_shows_panel(self, panel: LineagePanel):
        """open() makes the panel visible."""
        panel.open()
        assert panel.visible is True

    def test_close_hides_panel(self, panel: LineagePanel):
        """close() hides the panel."""
        panel.open()
        panel.close()
        assert panel.visible is False


class TestBloodlineList:
    """Tests for bloodline discovery and listing."""

    def test_refresh_list_finds_bloodlines(self, panel: LineagePanel):
        """refresh_list populates the bloodline list."""
        panel.open()
        assert len(panel._bloodlines) >= 1
        assert panel._bloodlines[0].name == "default"

    def test_active_name_read(self, panel: LineagePanel):
        """Active name is read from _active.txt."""
        panel.open()
        assert panel._active_name == "default"

    def test_empty_save_dir(self, tmp_path: Path):
        """Panel handles empty save directory gracefully."""
        empty = tmp_path / "empty_saves"
        empty.mkdir()
        p = LineagePanel(save_base_dir=str(empty))
        p.open()
        assert len(p._bloodlines) == 0

    def test_refresh_list_handles_os_error(self, tmp_path: Path):
        """refresh_list catches errors and opens with empty list."""
        from unittest.mock import patch as _patch

        base = tmp_path / "saves"
        base.mkdir()
        p = LineagePanel(save_base_dir=str(base))

        with _patch(
            "seaman_brain.gui.lineage_panel.StatePersistence.list_bloodlines",
            side_effect=OSError("Permission denied"),
        ):
            p.open()  # Should NOT raise

        assert len(p._bloodlines) == 0
        assert "Error" in p._status_text
        assert p.visible is True

    def test_refresh_list_handles_generic_exception(self, tmp_path: Path):
        """refresh_list catches generic exceptions gracefully."""
        from unittest.mock import patch as _patch

        base = tmp_path / "saves"
        base.mkdir()
        p = LineagePanel(save_base_dir=str(base))

        with _patch(
            "seaman_brain.gui.lineage_panel.StatePersistence.migrate_flat_saves",
            side_effect=RuntimeError("unexpected"),
        ):
            p.open()  # Should NOT raise

        assert len(p._bloodlines) == 0
        assert "Error" in p._status_text


class TestNewBloodline:
    """Tests for creating new bloodlines."""

    def test_new_creates_directory(self, panel: LineagePanel, save_dir: Path):
        """New bloodline creates a directory with creature.json."""
        panel.open()
        initial_count = len(panel._bloodlines)
        panel._build_widgets()
        panel._on_new_click()
        assert len(panel._bloodlines) == initial_count + 1
        # New directory should exist
        new_bl = panel._bloodlines[-1]
        assert (Path(new_bl.save_dir) / "creature.json").exists()

    def test_new_generates_unique_name(self, panel: LineagePanel, save_dir: Path):
        """New bloodline generates a non-colliding name."""
        panel.open()
        panel._build_widgets()
        panel._on_new_click()
        panel._on_new_click()
        names = [bl.name for bl in panel._bloodlines]
        assert len(names) == len(set(names))  # All unique

    def test_new_fires_callback(self, save_dir: Path):
        """New bloodline fires the on_new callback."""
        cb = MagicMock()
        p = LineagePanel(save_base_dir=str(save_dir), on_new=cb)
        p.open()
        p._build_widgets()
        p._on_new_click()
        cb.assert_called_once()


class TestLoadBloodline:
    """Tests for switching active bloodline."""

    def test_load_switches_active(self, panel: LineagePanel, save_dir: Path):
        """Load changes the active bloodline."""
        panel.open()
        panel._build_widgets()

        # Create a second bloodline
        panel._on_new_click()
        # Find the non-active bloodline
        for i, bl in enumerate(panel._bloodlines):
            if bl.name != panel._active_name:
                panel._selected_index = i
                new_name = bl.name
                break

        panel._on_load_click()
        assert panel._active_name == new_name

    def test_load_same_is_noop(self, panel: LineagePanel):
        """Loading the already-active bloodline is a noop."""
        panel.open()
        panel._build_widgets()
        panel._on_load_click()
        assert "already active" in panel._status_text

    def test_load_fires_callback(self, save_dir: Path):
        """Load fires the on_switch callback."""
        cb = MagicMock()
        p = LineagePanel(save_base_dir=str(save_dir), on_switch=cb)
        p.open()
        p._build_widgets()
        p._on_new_click()  # Create second
        # Find the non-active bloodline
        for i, bl in enumerate(p._bloodlines):
            if bl.name != p._active_name:
                p._selected_index = i
                break
        p._on_load_click()
        cb.assert_called_once()


class TestDeleteBloodline:
    """Tests for deleting bloodlines."""

    def test_cannot_delete_active(self, panel: LineagePanel):
        """Cannot delete the active bloodline."""
        panel.open()
        panel._build_widgets()
        panel._selected_index = 0  # default (active)
        panel._on_delete_click()
        assert "Cannot delete" in panel._status_text
        assert panel._confirm_delete is False

    def _select_non_active(self, panel: LineagePanel) -> int:
        """Find and select a non-active bloodline, returning its index."""
        for i, bl in enumerate(panel._bloodlines):
            if bl.name != panel._active_name:
                panel._selected_index = i
                return i
        return -1

    def test_delete_shows_confirmation(self, panel: LineagePanel, save_dir: Path):
        """Delete shows confirmation dialog for non-active bloodline."""
        panel.open()
        panel._build_widgets()
        panel._on_new_click()  # Create non-active bloodline
        self._select_non_active(panel)
        panel._on_delete_click()
        assert panel._confirm_delete is True

    def test_confirm_delete_removes_directory(self, panel: LineagePanel, save_dir: Path):
        """Confirming delete removes the bloodline directory."""
        panel.open()
        panel._build_widgets()
        panel._on_new_click()
        idx = self._select_non_active(panel)
        bl = panel._bloodlines[idx]
        bl_dir = Path(bl.save_dir)
        assert bl_dir.exists()

        panel._confirm_delete = True
        panel._confirm_delete_yes()

        assert not bl_dir.exists()
        assert panel._confirm_delete is False

    def test_cancel_delete(self, panel: LineagePanel, save_dir: Path):
        """Cancelling delete keeps the directory."""
        panel.open()
        panel._build_widgets()
        panel._on_new_click()
        idx = self._select_non_active(panel)
        bl_dir = Path(panel._bloodlines[idx].save_dir)
        panel._confirm_delete = True
        panel._confirm_delete_no()
        assert bl_dir.exists()
        assert panel._confirm_delete is False

    def test_delete_fires_callback(self, save_dir: Path):
        """Delete fires the on_delete callback."""
        cb = MagicMock()
        p = LineagePanel(save_base_dir=str(save_dir), on_delete=cb)
        p.open()
        p._build_widgets()
        p._on_new_click()
        self._select_non_active(p)
        p._confirm_delete = True
        p._confirm_delete_yes()
        cb.assert_called_once()


class TestRendering:
    """Tests for lineage panel rendering."""

    def test_render_when_hidden_noop(self, panel: LineagePanel):
        """Rendering when hidden draws nothing."""
        panel.visible = False
        panel.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count == 0

    def test_render_when_visible(self, panel: LineagePanel):
        """Rendering when visible draws panel."""
        panel.open()
        panel.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count > 0

    def test_render_empty_list(self, tmp_path: Path):
        """Rendering with no bloodlines shows help text."""
        empty = tmp_path / "empty"
        empty.mkdir()
        p = LineagePanel(save_base_dir=str(empty))
        p.open()
        p.render(_surface_mock)

    def test_render_with_confirm_dialog(self, panel: LineagePanel, save_dir: Path):
        """Rendering with confirmation dialog doesn't crash."""
        panel.open()
        panel._build_widgets()
        panel._on_new_click()
        panel._selected_index = len(panel._bloodlines) - 1
        panel._confirm_delete = True
        panel.render(_surface_mock)


class TestClickHandling:
    """Tests for lineage panel click handling."""

    def test_click_when_hidden_not_consumed(self, panel: LineagePanel):
        """Clicks when hidden pass through."""
        panel.visible = False
        assert panel.handle_click(500, 400) is False

    def test_click_inside_panel_consumed(self, panel: LineagePanel):
        """Clicks inside panel are consumed."""
        panel.open()
        panel.render(_surface_mock)
        cx = panel._panel_x + 350
        cy = panel._panel_y + 260
        result = panel.handle_click(cx, cy)
        assert result is True

    def test_mouse_move_when_hidden_noop(self, panel: LineagePanel):
        """Mouse move when hidden doesn't crash."""
        panel.visible = False
        panel.handle_mouse_move(500, 400)

    def test_mouse_move_when_visible(self, panel: LineagePanel):
        """Mouse move when visible doesn't crash."""
        panel.open()
        panel.render(_surface_mock)
        panel.handle_mouse_move(500, 400)


class TestCloseCallback:
    """Tests for the on_close callback when X button is clicked."""

    def test_close_button_fires_on_close(self, save_dir: Path):
        """Clicking the X button fires the on_close callback."""
        cb = MagicMock()
        p = LineagePanel(save_base_dir=str(save_dir), on_close=cb)
        p.open()
        p.render(_surface_mock)  # Build widgets

        # Invoke the close button's callback directly
        p._close()
        assert p.visible is False
        cb.assert_called_once()

    def test_close_without_callback_no_crash(self, panel: LineagePanel):
        """_close works even when on_close is None."""
        panel.open()
        panel.render(_surface_mock)
        panel._close()  # Should not raise
        assert panel.visible is False

    def test_programmatic_close_does_not_fire_callback(self, save_dir: Path):
        """close() (not _close) does not fire on_close."""
        cb = MagicMock()
        p = LineagePanel(save_base_dir=str(save_dir), on_close=cb)
        p.open()
        p.close()
        cb.assert_not_called()
