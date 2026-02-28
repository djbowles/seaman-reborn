"""Tests for the settings panel overlay (gui/settings_panel.py).

Pygame is mocked at module level to avoid requiring a display server in CI.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None

# Surface mock
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768

# Font mock
_font_mock = MagicMock()
_font_mock.get_linesize.return_value = 16
_font_mock.size.return_value = (80, 16)
_text_surf_mock = MagicMock()
_text_surf_mock.get_width.return_value = 80
_text_surf_mock.get_height.return_value = 16
_font_mock.render.return_value = _text_surf_mock
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock

# Draw mock
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.line.return_value = None

# Rect mock with collidepoint
def _make_rect(x, y, w, h):
    r = MagicMock()
    r.x = x
    r.y = y
    r.width = w
    r.height = h
    r.collidepoint = lambda mx, my: x <= mx <= x + w and y <= my <= y + h
    return r


_pygame_mock.Rect = _make_rect

# Surface constructor mock
def _make_surface(*args, **kwargs):
    s = MagicMock()
    s.get_width.return_value = args[0][0] if args and isinstance(args[0], tuple) else 1024
    s.get_height.return_value = args[0][1] if args and isinstance(args[0], tuple) else 768
    return s


_pygame_mock.Surface = _make_surface

# Install pygame mock
sys.modules["pygame"] = _pygame_mock

from seaman_brain.config import SeamanConfig  # noqa: E402
from seaman_brain.gui.settings_panel import SettingsPanel, SettingsTab  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset mocks and re-install pygame mock between tests."""
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.settings_panel as sp_mod
    import seaman_brain.gui.widgets as widgets_mod
    widgets_mod.pygame = _pygame_mock
    widgets_mod._FONT = None
    sp_mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _font_mock.reset_mock()
    _text_surf_mock.reset_mock()
    # Restore return values
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
def config() -> SeamanConfig:
    """Default config for tests."""
    return SeamanConfig()


@pytest.fixture()
def panel(config: SeamanConfig) -> SettingsPanel:
    """Create a settings panel for testing."""
    return SettingsPanel(config=config)


# ── Construction Tests ────────────────────────────────────────────────


class TestSettingsPanelConstruction:
    """Tests for SettingsPanel initialization."""

    def test_default_state(self, panel: SettingsPanel):
        """Panel starts hidden with personality tab active."""
        assert panel.visible is False
        assert panel.active_tab == SettingsTab.PERSONALITY

    def test_custom_screen_size(self, config: SeamanConfig):
        """Panel respects custom screen dimensions."""
        p = SettingsPanel(config=config, screen_width=800, screen_height=600)
        assert p._screen_w == 800
        assert p._screen_h == 600

    def test_presets_loaded(self, panel: SettingsPanel):
        """Presets are loaded from config path."""
        # May or may not have real presets depending on test environment
        assert isinstance(panel._presets, dict)


# ── Open/Close Tests ─────────────────────────────────────────────────


class TestOpenClose:
    """Tests for showing and hiding the settings panel."""

    def test_open_makes_visible(self, panel: SettingsPanel):
        """open() sets visible to True."""
        panel.open()
        assert panel.visible is True

    def test_close_hides_panel(self, panel: SettingsPanel):
        """close() sets visible to False."""
        panel.open()
        panel.close()
        assert panel.visible is False

    def test_close_collapses_dropdown(self, panel: SettingsPanel):
        """close() collapses any open dropdown."""
        panel.open()
        # Force build widgets
        panel.render(_surface_mock)
        if panel._model_dropdown is not None:
            panel._model_dropdown.expanded = True
        panel.close()
        if panel._model_dropdown is not None:
            assert panel._model_dropdown.expanded is False


# ── Rendering Tests ───────────────────────────────────────────────────


class TestRendering:
    """Tests for settings panel rendering."""

    def test_render_when_hidden_is_noop(self, panel: SettingsPanel):
        """Rendering a hidden panel draws nothing."""
        panel.visible = False
        panel.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count == 0

    def test_render_when_visible_draws_panel(self, panel: SettingsPanel):
        """Rendering a visible panel draws background and widgets."""
        panel.open()
        panel.render(_surface_mock)
        assert _pygame_mock.draw.rect.call_count > 0
        assert _font_mock.render.call_count > 0

    def test_render_personality_tab(self, panel: SettingsPanel):
        """Personality tab renders without crash."""
        panel.open()
        panel.active_tab = SettingsTab.PERSONALITY
        panel.render(_surface_mock)

    def test_render_llm_tab(self, panel: SettingsPanel):
        """LLM tab renders without crash."""
        panel.open()
        panel.active_tab = SettingsTab.LLM
        panel.render(_surface_mock)

    def test_render_audio_tab(self, panel: SettingsPanel):
        """Audio tab renders without crash."""
        panel.open()
        panel.active_tab = SettingsTab.AUDIO
        panel.render(_surface_mock)


# ── Tab Switching Tests ───────────────────────────────────────────────


class TestTabSwitching:
    """Tests for switching between settings tabs."""

    def test_switch_to_llm_tab(self, panel: SettingsPanel):
        """_switch_tab changes active tab."""
        panel._switch_tab(SettingsTab.LLM)
        assert panel.active_tab == SettingsTab.LLM

    def test_switch_to_audio_tab(self, panel: SettingsPanel):
        """Switch to audio tab works."""
        panel._switch_tab(SettingsTab.AUDIO)
        assert panel.active_tab == SettingsTab.AUDIO

    def test_switch_away_from_llm_closes_dropdown(self, panel: SettingsPanel):
        """Switching away from LLM tab closes the model dropdown."""
        panel.open()
        panel.render(_surface_mock)  # Build widgets
        panel.active_tab = SettingsTab.LLM
        if panel._model_dropdown is not None:
            panel._model_dropdown.expanded = True
        panel._switch_tab(SettingsTab.PERSONALITY)
        if panel._model_dropdown is not None:
            assert panel._model_dropdown.expanded is False


# ── Personality Preset Tests ──────────────────────────────────────────


class TestPersonalityPresets:
    """Tests for personality preset selection."""

    def test_select_preset_updates_active(self, panel: SettingsPanel):
        """Selecting a preset updates _active_preset."""
        panel.open()
        panel.render(_surface_mock)  # Build widgets
        panel._select_preset("traditional")
        assert panel._active_preset == "traditional"

    def test_select_preset_updates_config_traits(self, panel: SettingsPanel):
        """Selecting a preset updates config personality traits."""
        panel.open()
        panel.render(_surface_mock)
        panel._select_preset("traditional")
        if "traditional" in panel._presets:
            expected_cynicism = panel._presets["traditional"].traits.get("cynicism", 0.95)
            assert panel._config.personality.base_traits["cynicism"] == expected_cynicism

    def test_select_preset_fires_callback(self, config: SeamanConfig):
        """Selecting a preset calls on_personality_change."""
        cb = MagicMock()
        p = SettingsPanel(config=config, on_personality_change=cb)
        p.open()
        p.render(_surface_mock)
        p._select_preset("modern")
        if "modern" in p._presets:
            cb.assert_called_once()

    def test_select_nonexistent_preset_noop(self, panel: SettingsPanel):
        """Selecting a nonexistent preset does nothing."""
        panel.open()
        panel.render(_surface_mock)
        old_preset = panel._active_preset
        panel._select_preset("nonexistent_preset_xyz")
        assert panel._active_preset == old_preset

    def test_trait_slider_change(self, panel: SettingsPanel):
        """Trait slider change updates config."""
        panel.open()
        panel.render(_surface_mock)
        panel._on_trait_change("cynicism", 0.42)
        assert panel._config.personality.base_traits["cynicism"] == pytest.approx(0.42)

    def test_trait_slider_fires_callback(self, config: SeamanConfig):
        """Trait slider change calls on_personality_change."""
        cb = MagicMock()
        p = SettingsPanel(config=config, on_personality_change=cb)
        p._on_trait_change("wit", 0.8)
        cb.assert_called_once()


# ── LLM Settings Tests ───────────────────────────────────────────────


class TestLLMSettings:
    """Tests for LLM model selection and apply."""

    def test_set_model_list(self, panel: SettingsPanel):
        """set_model_list populates the model dropdown."""
        panel.open()
        panel.render(_surface_mock)  # Build widgets
        panel.set_model_list(["model-a", "model-b", "qwen3-coder:30b"])
        assert panel._model_dropdown is not None
        assert panel._model_dropdown.items == ["model-a", "model-b", "qwen3-coder:30b"]
        assert panel._models_loaded is True

    def test_set_model_list_selects_current(self, panel: SettingsPanel):
        """set_model_list selects the current config model."""
        panel.open()
        panel.render(_surface_mock)
        panel.set_model_list(["model-a", "qwen3-coder:30b", "model-c"])
        assert panel._model_dropdown is not None
        assert panel._model_dropdown.selected_index == 1

    def test_apply_llm_settings(self, config: SeamanConfig):
        """Apply updates config model and temperature."""
        cb = MagicMock()
        p = SettingsPanel(config=config, on_llm_apply=cb)
        p.open()
        p.render(_surface_mock)
        p.set_model_list(["test-model", "other-model"])
        p._model_dropdown.selected_index = 0
        p._temp_slider.value = 1.5
        p._apply_llm_settings()
        assert config.llm.model == "test-model"
        assert config.llm.temperature == pytest.approx(1.5)
        cb.assert_called_once_with("test-model", 1.5)

    def test_apply_with_loading_placeholder(self, panel: SettingsPanel):
        """Apply with 'Loading...' placeholder doesn't update config."""
        panel.open()
        panel.render(_surface_mock)
        old_model = panel._config.llm.model
        panel._apply_llm_settings()
        assert panel._config.llm.model == old_model
        assert panel._llm_status == "No model selected"


# ── Audio Settings Tests ──────────────────────────────────────────────


class TestAudioSettings:
    """Tests for audio toggle and volume changes."""

    def test_audio_toggle_updates_config(self, panel: SettingsPanel):
        """Toggling TTS updates audio config."""
        panel.open()
        panel.render(_surface_mock)
        panel._on_audio_setting("tts_enabled", False)
        assert panel._config.audio.tts_enabled is False

    def test_audio_volume_updates_config(self, panel: SettingsPanel):
        """Volume slider change updates audio config."""
        panel._on_audio_setting("tts_volume", 0.6)
        assert panel._config.audio.tts_volume == pytest.approx(0.6)

    def test_audio_change_fires_callback(self, config: SeamanConfig):
        """Audio change calls on_audio_change."""
        cb = MagicMock()
        p = SettingsPanel(config=config, on_audio_change=cb)
        p._on_audio_setting("sfx_enabled", True)
        cb.assert_called_once_with("sfx_enabled", True)


# ── Click Handling Tests ──────────────────────────────────────────────


class TestClickHandling:
    """Tests for mouse click routing."""

    def test_click_when_hidden_returns_false(self, panel: SettingsPanel):
        """Clicks when panel is hidden are not consumed."""
        panel.visible = False
        assert panel.handle_click(500, 400) is False

    def test_click_inside_panel_consumed(self, panel: SettingsPanel):
        """Clicks inside the panel area are consumed."""
        panel.open()
        panel.render(_surface_mock)
        # Click at center of panel
        cx = panel._panel_x + 350
        cy = panel._panel_y + 260
        result = panel.handle_click(cx, cy)
        assert result is True

    def test_click_outside_panel_not_consumed(self, panel: SettingsPanel):
        """Clicks outside the panel area pass through."""
        panel.open()
        panel.render(_surface_mock)
        result = panel.handle_click(5, 5)
        assert result is False


# ── Mouse Move Handling Tests ─────────────────────────────────────────


class TestMouseHandling:
    """Tests for mouse move and mouse up handling."""

    def test_mouse_move_when_hidden_noop(self, panel: SettingsPanel):
        """Mouse move when hidden does nothing."""
        panel.visible = False
        panel.handle_mouse_move(500, 400)  # Should not raise

    def test_mouse_up_when_hidden_noop(self, panel: SettingsPanel):
        """Mouse up when hidden does nothing."""
        panel.visible = False
        panel.handle_mouse_up()  # Should not raise

    def test_mouse_move_visible(self, panel: SettingsPanel):
        """Mouse move when visible updates hover states."""
        panel.open()
        panel.render(_surface_mock)
        panel.handle_mouse_move(500, 400)  # Should not raise

    def test_mouse_up_releases_sliders(self, panel: SettingsPanel):
        """Mouse up when visible releases any dragging sliders."""
        panel.open()
        panel.render(_surface_mock)
        panel.active_tab = SettingsTab.PERSONALITY
        panel._active_preset = "custom"
        # Simulate a drag
        if panel._trait_sliders:
            panel._trait_sliders[0]._dragging = True
        panel.handle_mouse_up()
        if panel._trait_sliders:
            assert panel._trait_sliders[0]._dragging is False


# ── SettingsTab Enum Tests ────────────────────────────────────────────


class TestSettingsTabEnum:
    """Tests for the SettingsTab enum."""

    def test_four_tabs(self):
        """There are exactly four settings tabs."""
        assert len(SettingsTab) == 4

    def test_tab_values(self):
        """Tab values are the expected display strings."""
        assert SettingsTab.PERSONALITY.value == "Personality"
        assert SettingsTab.LLM.value == "LLM Model"
        assert SettingsTab.AUDIO.value == "Audio"
        assert SettingsTab.VISION.value == "Vision"


# ── Vision Tab Tests ─────────────────────────────────────────────────


class TestVisionTab:
    """Tests for vision tab in settings panel."""

    def test_render_vision_tab(self, panel: SettingsPanel):
        """Vision tab renders without crash."""
        panel.open()
        panel.active_tab = SettingsTab.VISION
        panel.render(_surface_mock)

    def test_vision_source_dropdown_built(self, panel: SettingsPanel):
        """Vision source dropdown is created during build."""
        panel.open()
        panel.render(_surface_mock)
        assert panel._vision_source_dropdown is not None

    def test_vision_interval_slider_built(self, panel: SettingsPanel):
        """Vision interval slider is created during build."""
        panel.open()
        panel.render(_surface_mock)
        assert panel._vision_interval_slider is not None

    def test_vision_look_button_built(self, panel: SettingsPanel):
        """Vision look button is created during build."""
        panel.open()
        panel.render(_surface_mock)
        assert panel._vision_look_button is not None

    def test_vision_setting_updates_config(self, panel: SettingsPanel):
        """Vision source change updates config."""
        panel._on_vision_setting("source", "tank")
        assert panel._config.vision.source == "tank"
        assert panel._config.vision.enabled is True

    def test_vision_source_off_disables(self, panel: SettingsPanel):
        """Setting source to 'off' disables vision."""
        panel._on_vision_setting("source", "off")
        assert panel._config.vision.source == "off"
        assert panel._config.vision.enabled is False

    def test_vision_interval_updates_config(self, panel: SettingsPanel):
        """Vision interval change updates config."""
        panel._on_vision_setting("capture_interval", 15.0)
        assert panel._config.vision.capture_interval == 15.0

    def test_vision_change_fires_callback(self, config: SeamanConfig):
        """Vision change calls on_vision_change."""
        cb = MagicMock()
        p = SettingsPanel(config=config, on_vision_change=cb)
        p._on_vision_setting("source", "webcam")
        cb.assert_called_once_with("source", "webcam")

    def test_vision_look_fires_callback(self, config: SeamanConfig):
        """Look Now button calls on_vision_change."""
        cb = MagicMock()
        p = SettingsPanel(config=config, on_vision_change=cb)
        p._on_vision_look()
        cb.assert_called_once_with("look_now", True)

    def test_set_last_observation(self, panel: SettingsPanel):
        """set_last_observation updates the display text."""
        panel.set_last_observation("The human is smiling")
        assert panel._last_observation_text == "The human is smiling"

    def test_switch_away_closes_vision_dropdown(self, panel: SettingsPanel):
        """Switching away from Vision tab closes source dropdown."""
        panel.open()
        panel.render(_surface_mock)
        panel.active_tab = SettingsTab.VISION
        if panel._vision_source_dropdown is not None:
            panel._vision_source_dropdown.expanded = True
        panel._switch_tab(SettingsTab.PERSONALITY)
        if panel._vision_source_dropdown is not None:
            assert panel._vision_source_dropdown.expanded is False

    def test_close_panel_closes_vision_dropdown(self, panel: SettingsPanel):
        """Closing panel closes vision source dropdown."""
        panel.open()
        panel.render(_surface_mock)
        if panel._vision_source_dropdown is not None:
            panel._vision_source_dropdown.expanded = True
        panel.close()
        if panel._vision_source_dropdown is not None:
            assert panel._vision_source_dropdown.expanded is False

    def test_mouse_move_vision_tab(self, panel: SettingsPanel):
        """Mouse move on vision tab doesn't crash."""
        panel.open()
        panel.render(_surface_mock)
        panel.active_tab = SettingsTab.VISION
        panel.handle_mouse_move(500, 400)

    def test_mouse_up_vision_tab(self, panel: SettingsPanel):
        """Mouse up on vision tab releases slider."""
        panel.open()
        panel.render(_surface_mock)
        panel.active_tab = SettingsTab.VISION
        panel.handle_mouse_up()


# ── Device List Refresh Tests (Fix #26) ──────────────────────────────


class TestRefreshDeviceLists:
    """Tests for refresh_device_lists updating dropdown options."""

    def test_refresh_updates_output_dropdown(self, panel: SettingsPanel):
        """refresh_device_lists updates output device dropdown."""
        panel.open()
        panel.render(_surface_mock)  # Build widgets
        panel.active_tab = SettingsTab.AUDIO
        panel.render(_surface_mock)  # Build audio widgets

        with patch(
            "seaman_brain.gui.settings_panel.list_audio_output_devices",
            return_value=[(0, "Speaker A"), (1, "Speaker B")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_audio_input_devices",
            return_value=[(0, "Mic A")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_tts_voices",
            return_value=[("v1", "Voice One")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_webcams",
            return_value=[(0, "Camera 0")],
        ):
            panel.refresh_device_lists()
            panel.apply_pending_refresh()

        if panel._output_device_dropdown is not None:
            assert panel._output_device_dropdown.items == ["Speaker A", "Speaker B"]

    def test_refresh_updates_input_dropdown(self, panel: SettingsPanel):
        """refresh_device_lists updates input device dropdown."""
        panel.open()
        panel.render(_surface_mock)
        panel.active_tab = SettingsTab.AUDIO
        panel.render(_surface_mock)

        with patch(
            "seaman_brain.gui.settings_panel.list_audio_output_devices",
            return_value=[(0, "Speaker A")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_audio_input_devices",
            return_value=[(0, "Mic X"), (1, "Mic Y"), (2, "Mic Z")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_tts_voices",
            return_value=[("v1", "Voice One")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_webcams",
            return_value=[(0, "Camera 0")],
        ):
            panel.refresh_device_lists()
            panel.apply_pending_refresh()

        if panel._input_device_dropdown is not None:
            assert panel._input_device_dropdown.items == ["Mic X", "Mic Y", "Mic Z"]

    def test_refresh_before_build_is_noop(self, config: SeamanConfig):
        """refresh_device_lists before widgets are built does nothing."""
        p = SettingsPanel(config=config)
        # Should not crash even though no widgets have been built
        p.refresh_device_lists()

    def test_refresh_updates_camera_dropdown(self, panel: SettingsPanel):
        """refresh_device_lists updates camera device dropdown."""
        panel.open()
        panel.render(_surface_mock)
        panel.active_tab = SettingsTab.VISION
        panel.render(_surface_mock)

        with patch(
            "seaman_brain.gui.settings_panel.list_audio_output_devices",
            return_value=[(0, "Speaker A")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_audio_input_devices",
            return_value=[(0, "Mic A")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_tts_voices",
            return_value=[("v1", "Voice One")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_webcams",
            return_value=[(0, "Cam Front"), (2, "Cam Rear")],
        ):
            panel.refresh_device_lists()
            panel.apply_pending_refresh()

        if panel._vision_cam_dropdown is not None:
            assert panel._vision_cam_dropdown.items == ["Cam Front", "Cam Rear"]
            assert panel._cam_device_indices == [0, 2]

    def test_refresh_updates_output_after_apply(self, panel: SettingsPanel):
        """refresh_device_lists queues results, apply_pending_refresh applies them."""
        panel.open()
        panel.render(_surface_mock)

        with patch(
            "seaman_brain.gui.settings_panel.list_audio_output_devices",
            return_value=[(0, "Speaker X"), (1, "Speaker Y")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_audio_input_devices",
            return_value=[(0, "Mic A")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_tts_voices",
            return_value=[("v1", "Voice One")],
        ), patch(
            "seaman_brain.gui.settings_panel.list_webcams",
            return_value=[(0, "Camera 0")],
        ):
            panel.refresh_device_lists()

        # Before apply, pending_refresh should be set
        assert panel._pending_refresh is not None

        panel.apply_pending_refresh()
        assert panel._pending_refresh is None
        if panel._output_device_dropdown is not None:
            assert panel._output_device_dropdown.items == ["Speaker X", "Speaker Y"]


# ── Pending Refresh Thread Safety ─────────────────────────────────────


class TestPendingRefreshPattern:
    """Tests for the thread-safe pending refresh pattern."""

    def test_apply_pending_refresh_noop_when_none(self, panel: SettingsPanel):
        """apply_pending_refresh does nothing when no pending data."""
        panel._pending_refresh = None
        panel.apply_pending_refresh()  # Should not raise

    def test_apply_clears_pending(self, panel: SettingsPanel):
        """apply_pending_refresh clears _pending_refresh after applying."""
        panel.open()
        panel.render(_surface_mock)
        panel._pending_refresh = {"output": (["A", "B"], 0)}
        panel.apply_pending_refresh()
        assert panel._pending_refresh is None

    def test_refreshing_flag_prevents_concurrent(self, panel: SettingsPanel):
        """Second refresh call while first is running is skipped."""
        panel.open()
        panel.render(_surface_mock)

        # Simulate _refreshing being True (as if another thread is running)
        with panel._refresh_lock:
            panel._refreshing = True

        with patch(
            "seaman_brain.gui.settings_panel.list_audio_output_devices",
        ) as mock_list:
            panel.refresh_device_lists()
        # Should not have been called since _refreshing was True
        mock_list.assert_not_called()

    def test_refresh_error_clears_refreshing_flag(self, panel: SettingsPanel):
        """Refresh errors still clear the _refreshing flag."""
        panel.open()
        panel.render(_surface_mock)
        with patch(
            "seaman_brain.gui.settings_panel.list_audio_output_devices",
            side_effect=RuntimeError("device error"),
        ), patch(
            "seaman_brain.gui.settings_panel.list_audio_input_devices",
            return_value=[],
        ), patch(
            "seaman_brain.gui.settings_panel.list_tts_voices",
            return_value=[],
        ), patch(
            "seaman_brain.gui.settings_panel.list_webcams",
            return_value=[],
        ):
            panel.refresh_device_lists()
        assert panel._refreshing is False
