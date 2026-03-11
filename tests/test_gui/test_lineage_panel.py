"""Tests for the lineage slide-out drawer."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 80
_font_surface.get_height.return_value = 14
_font_mock.render.return_value = _font_surface
_font_mock.size.return_value = (80, 14)
_font_mock.get_linesize.return_value = 16
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
_surface_mock = MagicMock()
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.lineage_panel import LineagePanel  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.lineage_panel as mod
    mod.pygame = _pygame_mock
    yield


class TestLineagePanel:
    def test_construction(self):
        lp = LineagePanel(width=400)
        assert lp is not None

    def test_set_bloodlines(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["Alpha", "Beta"], active="Alpha")
        assert lp._active == "Alpha"

    def test_select_callback(self):
        cb = MagicMock()
        lp = LineagePanel(width=400, on_select=cb)
        lp.set_bloodlines(["Alpha", "Beta"], active="Alpha")
        lp._select("Beta")
        cb.assert_called_once_with("Beta")

    def test_render_does_not_crash(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["Alpha"], active="Alpha")
        surface = MagicMock()
        lp.render(surface, progress=1.0)


class TestBloodlineList:
    def test_empty_bloodlines(self):
        lp = LineagePanel(width=400)
        assert lp._bloodlines == []
        assert lp._active is None

    def test_set_bloodlines_updates_list(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["A", "B", "C"], active="B")
        assert lp._bloodlines == ["A", "B", "C"]
        assert lp._active == "B"
        assert lp._selected_index == 1

    def test_set_bloodlines_active_not_in_list(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["A", "B"], active="Z")
        assert lp._active == "Z"
        assert lp._selected_index == 0


class TestSelection:
    def test_select_fires_callback(self):
        cb = MagicMock()
        lp = LineagePanel(width=400, on_select=cb)
        lp.set_bloodlines(["A", "B"], active="A")
        lp._select("B")
        cb.assert_called_once_with("B")

    def test_select_without_callback(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["A", "B"], active="A")
        lp._select("B")  # No crash

    def test_select_updates_active(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["A", "B"], active="A")
        lp._select("B")
        assert lp._active == "B"


class TestRendering:
    def test_render_zero_progress_noop(self):
        lp = LineagePanel(width=400)
        surface = MagicMock()
        lp.render(surface, progress=0.0)

    def test_render_half_progress(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["A"], active="A")
        surface = MagicMock()
        lp.render(surface, progress=0.5)

    def test_render_empty_list(self):
        lp = LineagePanel(width=400)
        surface = MagicMock()
        lp.render(surface, progress=1.0)


class TestMouseHandling:
    def test_handle_click_returns_bool(self):
        lp = LineagePanel(width=400)
        result = lp.handle_click(100, 100)
        assert isinstance(result, bool)

    def test_handle_mouse_move_no_crash(self):
        lp = LineagePanel(width=400)
        lp.handle_mouse_move(100, 100)

    def test_handle_mouse_up_no_crash(self):
        lp = LineagePanel(width=400)
        lp.handle_mouse_up()
