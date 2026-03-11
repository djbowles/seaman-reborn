"""Tests for the settings slide-out drawer."""
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

from seaman_brain.gui.settings_panel import SettingsPanel, SettingsTab  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.settings_panel as mod
    mod.pygame = _pygame_mock
    yield


class TestSettingsPanel:
    def test_construction(self):
        sp = SettingsPanel(width=400)
        assert sp.active_tab == SettingsTab.PERSONALITY

    def test_tab_switch(self):
        sp = SettingsPanel(width=400)
        sp.set_tab(SettingsTab.AUDIO)
        assert sp.active_tab == SettingsTab.AUDIO

    def test_personality_change_callback(self):
        cb = MagicMock()
        sp = SettingsPanel(width=400, on_personality_change=cb)
        sp._fire_personality_change({"cynicism": 0.8})
        cb.assert_called_once()

    def test_render_does_not_crash(self):
        sp = SettingsPanel(width=400)
        surface = MagicMock()
        sp.render(surface, progress=1.0)

    def test_settings_persistence_callback(self):
        cb = MagicMock()
        sp = SettingsPanel(width=400, on_audio_change=cb)
        sp._fire_audio_change("tts_provider", "riva")
        cb.assert_called_once_with("tts_provider", "riva")


class TestSettingsTabEnum:
    def test_four_tabs(self):
        assert len(SettingsTab) == 4

    def test_tab_values(self):
        assert SettingsTab.PERSONALITY.value == "Personality"
        assert SettingsTab.LLM.value == "LLM Model"
        assert SettingsTab.AUDIO.value == "Audio"
        assert SettingsTab.VISION.value == "Vision"


class TestTabStrip:
    def test_set_tab_personality(self):
        sp = SettingsPanel(width=400)
        sp.set_tab(SettingsTab.PERSONALITY)
        assert sp.active_tab == SettingsTab.PERSONALITY

    def test_set_tab_llm(self):
        sp = SettingsPanel(width=400)
        sp.set_tab(SettingsTab.LLM)
        assert sp.active_tab == SettingsTab.LLM

    def test_set_tab_vision(self):
        sp = SettingsPanel(width=400)
        sp.set_tab(SettingsTab.VISION)
        assert sp.active_tab == SettingsTab.VISION


class TestCallbacks:
    def test_fire_llm_apply(self):
        cb = MagicMock()
        sp = SettingsPanel(width=400, on_llm_apply=cb)
        sp._fire_llm_apply("qwen3:8b", 0.7)
        cb.assert_called_once_with("qwen3:8b", 0.7)

    def test_fire_vision_change(self):
        cb = MagicMock()
        sp = SettingsPanel(width=400, on_vision_change=cb)
        sp._fire_vision_change("source", "webcam")
        cb.assert_called_once_with("source", "webcam")

    def test_no_callback_no_crash(self):
        sp = SettingsPanel(width=400)
        sp._fire_personality_change({"cynicism": 0.5})
        sp._fire_audio_change("volume", 0.8)
        sp._fire_llm_apply("model", 0.5)
        sp._fire_vision_change("source", "off")


class TestRendering:
    def test_render_zero_progress_noop(self):
        sp = SettingsPanel(width=400)
        surface = MagicMock()
        sp.render(surface, progress=0.0)

    def test_render_half_progress(self):
        sp = SettingsPanel(width=400)
        surface = MagicMock()
        sp.render(surface, progress=0.5)

    def test_render_each_tab(self):
        sp = SettingsPanel(width=400)
        surface = MagicMock()
        for tab in SettingsTab:
            sp.set_tab(tab)
            sp.render(surface, progress=1.0)


class TestMouseHandling:
    def test_handle_click_returns_bool(self):
        sp = SettingsPanel(width=400)
        result = sp.handle_click(100, 100)
        assert isinstance(result, bool)

    def test_handle_mouse_move_no_crash(self):
        sp = SettingsPanel(width=400)
        sp.handle_mouse_move(100, 100)

    def test_handle_mouse_up_no_crash(self):
        sp = SettingsPanel(width=400)
        sp.handle_mouse_up()
