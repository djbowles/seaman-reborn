"""Tests for the thin GameEngine orchestrator."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_gui_mock = MagicMock()
sys.modules["pygame_gui"] = _pygame_gui_mock

_pygame_mock = MagicMock()
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.MOUSEBUTTONDOWN = 1025
_pygame_mock.SRCALPHA = 65536
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_F2 = 283
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768
_pygame_mock.display.set_mode.return_value = _surface_mock
_clock_mock = MagicMock()
_clock_mock.tick.return_value = 33
_pygame_mock.time.Clock.return_value = _clock_mock
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 16
_font_mock.render.return_value = _font_surface
_font_mock.size.return_value = (100, 16)
_font_mock.get_linesize.return_value = 18
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
sys.modules["pygame"] = _pygame_mock

import pytest  # noqa: E402

from seaman_brain.gui.game_loop import GameEngine  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = _pygame_gui_mock
    import seaman_brain.gui.game_loop as mod
    mod.pygame = _pygame_mock
    yield


class TestGameEngineInit:
    def test_construction_does_not_crash(self):
        engine = GameEngine()
        assert engine is not None

    def test_has_scene_manager(self):
        engine = GameEngine()
        assert engine._scene_manager is not None

    def test_has_input_handler(self):
        engine = GameEngine()
        assert engine._input_handler is not None

    def test_has_layout(self):
        engine = GameEngine()
        assert engine._layout is not None


class TestComponents:
    def test_has_hud(self):
        engine = GameEngine()
        assert engine._hud is not None

    def test_has_chat_panel(self):
        engine = GameEngine()
        assert engine._chat_panel is not None

    def test_has_tank_renderer(self):
        engine = GameEngine()
        assert engine._tank_renderer is not None

    def test_has_settings_panel(self):
        engine = GameEngine()
        assert engine._settings_panel is not None

    def test_has_lineage_panel(self):
        engine = GameEngine()
        assert engine._lineage_panel is not None
