"""Settings screen overlay with Personality, LLM, and Audio tabs.

Renders as a semi-transparent panel over frozen gameplay. Uses the
widget library (Button, Toggle, Slider, Dropdown) for all controls.

Session-scoped: changes live in memory only, not persisted to TOML.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from enum import Enum
from typing import Any

import pygame

from seaman_brain.config import PresetConfig, SeamanConfig, load_presets
from seaman_brain.gui.device_utils import (
    list_audio_input_devices,
    list_audio_output_devices,
    list_tts_voices,
    list_webcams,
)
from seaman_brain.gui.widgets import Button, Dropdown, Slider, Toggle

logger = logging.getLogger(__name__)

# ── Colors ───────────────────────────────────────────────────────────

_OVERLAY_BG = (0, 0, 0, 160)
_PANEL_BG = (18, 28, 48)
_PANEL_BORDER = (50, 70, 100)
_HEADER_BG = (12, 22, 42)
_TITLE_COLOR = (200, 220, 240)
_TEXT_COLOR = (180, 200, 220)
_TEXT_DIM = (120, 140, 160)
_TAB_ACTIVE = (40, 65, 100)
_TAB_INACTIVE = (25, 38, 60)
_TAB_TEXT = (200, 220, 240)
_STATUS_GREEN = (60, 200, 100)
_STATUS_YELLOW = (220, 200, 60)
_CLOSE_COLOR = (220, 80, 80)

_FONT_SIZE = 14
_TITLE_FONT_SIZE = 20
_PANEL_WIDTH = 700
_PANEL_HEIGHT = 520
_HEADER_HEIGHT = 36
_TAB_HEIGHT = 30
_CONTENT_PADDING = 16

_TRAIT_NAMES = [
    "cynicism", "wit", "patience", "curiosity",
    "warmth", "verbosity", "formality", "aggression",
]


def _find_saved_index(items: list[str], saved_value: str) -> int:
    """Find the dropdown index matching a saved config value.

    Returns 0 (System Default) if not found or saved_value is empty.
    """
    if not saved_value:
        return 0
    for i, name in enumerate(items):
        if name == saved_value:
            return i
    return 0


class SettingsTab(Enum):
    """Settings panel tabs."""

    PERSONALITY = "Personality"
    LLM = "LLM Model"
    AUDIO = "Audio"
    VISION = "Vision"


class SettingsPanel:
    """Settings overlay panel with three tabs.

    Manages personality presets, LLM model selection, and audio toggles.
    Rendered as a centered overlay on top of the game.

    Attributes:
        visible: Whether the settings panel is currently shown.
        active_tab: The currently active tab.
    """

    def __init__(
        self,
        config: SeamanConfig,
        screen_width: int = 1024,
        screen_height: int = 768,
        on_personality_change: Callable[[dict[str, float]], Any] | None = None,
        on_llm_apply: Callable[[str, float], Any] | None = None,
        on_audio_change: Callable[[str, Any], Any] | None = None,
        on_vision_change: Callable[[str, Any], Any] | None = None,
        on_close: Callable[[], Any] | None = None,
    ) -> None:
        self._config = config
        self._screen_w = screen_width
        self._screen_h = screen_height
        self.on_personality_change = on_personality_change
        self.on_llm_apply = on_llm_apply
        self.on_audio_change = on_audio_change
        self.on_vision_change = on_vision_change
        self.on_close = on_close

        self.visible = False
        self.active_tab = SettingsTab.PERSONALITY

        # Panel positioning
        self._panel_x = (screen_width - _PANEL_WIDTH) // 2
        self._panel_y = (screen_height - _PANEL_HEIGHT) // 2

        # Fonts (lazy)
        self._font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None

        # Status text
        self._status_text = ""

        # Presets
        self._presets: dict[str, PresetConfig] = {}
        self._active_preset = "modern"
        self._load_presets()

        # Build widgets
        self._tab_buttons: list[Button] = []
        self._close_button: Button | None = None
        self._personality_widgets: list[Any] = []
        self._preset_buttons: list[Button] = []
        self._trait_sliders: list[Slider] = []
        self._llm_widgets: list[Any] = []
        self._audio_widgets: list[Any] = []

        # LLM state
        self._model_dropdown: Dropdown | None = None
        self._temp_slider: Slider | None = None
        self._llm_apply_button: Button | None = None
        self._llm_status = ""
        self._models_loaded = False

        # Audio state
        self._tts_toggle: Toggle | None = None
        self._stt_toggle: Toggle | None = None
        self._sfx_toggle: Toggle | None = None
        self._tts_vol_slider: Slider | None = None
        self._sfx_vol_slider: Slider | None = None
        self._ambient_vol_slider: Slider | None = None

        # Vision state
        self._vision_widgets: list[Any] = []
        self._vision_source_dropdown: Dropdown | None = None
        self._vision_interval_slider: Slider | None = None
        self._vision_look_button: Button | None = None
        self._vision_cam_dropdown: Dropdown | None = None
        self._last_observation_text = ""
        self._last_vision_source: str = config.vision.source

        # Device dropdowns
        self._output_device_dropdown: Dropdown | None = None
        self._input_device_dropdown: Dropdown | None = None
        self._tts_voice_dropdown: Dropdown | None = None
        self._voice_ids: list[str] = []

        self._widgets_built = False

        # Thread-safe device refresh (populated by background thread)
        self._pending_refresh: dict[str, Any] | None = None
        self._refreshing = False
        self._refresh_lock = threading.Lock()

    def _ensure_fonts(self) -> None:
        """Initialize fonts if not yet done."""
        if self._font is None:
            try:
                self._font = pygame.font.SysFont("consolas", _FONT_SIZE)
                self._title_font = pygame.font.SysFont("consolas", _TITLE_FONT_SIZE, bold=True)
            except Exception:
                self._font = pygame.font.Font(None, _FONT_SIZE)
                self._title_font = pygame.font.Font(None, _TITLE_FONT_SIZE)

    def _load_presets(self) -> None:
        """Load personality presets from config."""
        try:
            self._presets = load_presets(self._config.personality.presets_path)
        except FileNotFoundError:
            logger.warning("Presets file not found, using defaults")
            self._presets = {}

    def _build_widgets(self) -> None:
        """Build all tab widgets. Called once after fonts are ready."""
        if self._widgets_built:
            return
        self._widgets_built = True

        px = self._panel_x
        py = self._panel_y
        content_x = px + _CONTENT_PADDING
        content_y = py + _HEADER_HEIGHT + _TAB_HEIGHT + _CONTENT_PADDING

        # Close button
        self._close_button = Button(
            px + _PANEL_WIDTH - 36, py + 4, 28, 28, "X",
            on_click=self._close,
        )

        # Tab buttons
        tab_y = py + _HEADER_HEIGHT
        tab_w = _PANEL_WIDTH // len(SettingsTab)
        for i, tab in enumerate(SettingsTab):
            btn = Button(
                px + i * tab_w, tab_y, tab_w, _TAB_HEIGHT, tab.value,
                on_click=lambda t=tab: self._switch_tab(t),
            )
            self._tab_buttons.append(btn)

        # ── Personality tab widgets ──
        self._build_personality_widgets(content_x, content_y)

        # ── LLM tab widgets ──
        self._build_llm_widgets(content_x, content_y)

        # ── Audio tab widgets ──
        self._build_audio_widgets(content_x, content_y)

        # ── Vision tab widgets ──
        self._build_vision_widgets(content_x, content_y)

    def _build_personality_widgets(self, x: int, y: int) -> None:
        """Build preset buttons and trait sliders."""
        # Preset radio buttons
        preset_keys = ["traditional", "modern", "custom"]
        btn_w = 180
        btn_h = 28
        for i, key in enumerate(preset_keys):
            preset = self._presets.get(key)
            label = preset.name if preset else key.capitalize()
            btn = Button(
                x + i * (btn_w + 10), y, btn_w, btn_h, label,
                on_click=lambda k=key: self._select_preset(k),
            )
            if key == self._active_preset:
                btn.selected = True
            self._preset_buttons.append(btn)

        # Trait sliders (below presets)
        slider_y = y + 50
        slider_w = _PANEL_WIDTH - 2 * _CONTENT_PADDING
        for i, trait in enumerate(_TRAIT_NAMES):
            val = self._config.personality.base_traits.get(trait, 0.5)
            slider = Slider(
                x, slider_y + i * 28, slider_w, 24,
                trait.capitalize(), value=val, min_val=0.0, max_val=1.0,
                on_change=lambda v, t=trait: self._on_trait_change(t, v),
            )
            self._trait_sliders.append(slider)

    def _build_llm_widgets(self, x: int, y: int) -> None:
        """Build LLM model dropdown, temperature slider, and apply button."""
        w = _PANEL_WIDTH - 2 * _CONTENT_PADDING

        self._model_dropdown = Dropdown(
            x, y, w, 26, "Model",
            items=["Loading..."], selected_index=0,
        )

        self._temp_slider = Slider(
            x, y + 40, w, 24, "Temperature",
            value=self._config.llm.temperature,
            min_val=0.0, max_val=2.0,
        )

        self._llm_apply_button = Button(
            x + w - 120, y + 80, 120, 30, "Apply",
            on_click=self._apply_llm_settings,
        )

        self._llm_widgets = [self._model_dropdown, self._temp_slider, self._llm_apply_button]

    def _build_audio_widgets(self, x: int, y: int) -> None:
        """Build audio toggles, volume sliders, and device dropdowns."""
        w = _PANEL_WIDTH - 2 * _CONTENT_PADDING

        self._tts_toggle = Toggle(
            x, y, w, 24, "Text-to-Speech",
            value=self._config.audio.tts_enabled,
            on_change=lambda v: self._on_audio_setting("tts_enabled", v),
        )
        self._stt_toggle = Toggle(
            x, y + 32, w, 24, "Speech-to-Text",
            value=self._config.audio.stt_enabled,
            on_change=lambda v: self._on_audio_setting("stt_enabled", v),
        )
        self._sfx_toggle = Toggle(
            x, y + 64, w, 24, "Sound Effects",
            value=self._config.audio.sfx_enabled,
            on_change=lambda v: self._on_audio_setting("sfx_enabled", v),
        )

        slider_y = y + 108
        self._tts_vol_slider = Slider(
            x, slider_y, w, 24, "TTS Volume",
            value=self._config.audio.tts_volume,
            on_change=lambda v: self._on_audio_setting("tts_volume", v),
        )
        self._sfx_vol_slider = Slider(
            x, slider_y + 32, w, 24, "SFX Volume",
            value=self._config.audio.sfx_volume,
            on_change=lambda v: self._on_audio_setting("sfx_volume", v),
        )
        self._ambient_vol_slider = Slider(
            x, slider_y + 64, w, 24, "Ambient Vol",
            value=self._config.audio.ambient_volume,
            on_change=lambda v: self._on_audio_setting("ambient_volume", v),
        )

        # Device dropdowns — match saved config values to find correct index
        device_y = slider_y + 104
        out_devs = list_audio_output_devices()
        out_names = [name for _, name in out_devs]
        out_idx = _find_saved_index(out_names, self._config.audio.audio_output_device)
        self._output_device_dropdown = Dropdown(
            x, device_y, w, 26, "Output",
            items=out_names, selected_index=out_idx,
            on_change=lambda _i, v: self._on_audio_setting("audio_output_device", v),
        )

        in_devs = list_audio_input_devices()
        in_names = [name for _, name in in_devs]
        in_idx = _find_saved_index(in_names, self._config.audio.audio_input_device)
        self._input_device_dropdown = Dropdown(
            x, device_y + 34, w, 26, "Input",
            items=in_names, selected_index=in_idx,
            on_change=lambda _i, v: self._on_audio_setting("audio_input_device", v),
        )

        voices = list_tts_voices(self._config.audio.tts_provider)
        voice_names = [name for _, name in voices]
        self._voice_ids = [vid for vid, _ in voices]
        voice_idx = _find_saved_index(self._voice_ids, self._config.audio.tts_voice)
        self._tts_voice_dropdown = Dropdown(
            x, device_y + 68, w, 26, "TTS Voice",
            items=voice_names, selected_index=voice_idx,
            on_change=lambda i, _v: self._on_audio_setting(
                "tts_voice", self._voice_ids[i] if i < len(self._voice_ids) else ""
            ),
        )

        self._audio_widgets = [
            self._tts_toggle, self._stt_toggle, self._sfx_toggle,
            self._tts_vol_slider, self._sfx_vol_slider, self._ambient_vol_slider,
            self._output_device_dropdown, self._input_device_dropdown,
            self._tts_voice_dropdown,
        ]

    def _build_vision_widgets(self, x: int, y: int) -> None:
        """Build vision source dropdown, camera dropdown, interval slider, and look button."""
        w = _PANEL_WIDTH - 2 * _CONTENT_PADDING

        source_items = ["Webcam", "Tank", "Off"]
        current = self._config.vision.source
        source_map = {"webcam": 0, "tank": 1, "off": 2}
        idx = source_map.get(current, 0)

        self._vision_source_dropdown = Dropdown(
            x, y, w, 26, "Source",
            items=source_items, selected_index=idx,
        )

        # Camera selection dropdown — store device indices for mapping
        cams = list_webcams()
        cam_names = [name for _, name in cams]
        self._cam_device_indices = [dev_idx for dev_idx, _ in cams]
        cam_idx = 0
        saved_cam = self._config.vision.webcam_index
        for i, dev_idx in enumerate(self._cam_device_indices):
            if dev_idx == saved_cam:
                cam_idx = i
                break
        self._vision_cam_dropdown = Dropdown(
            x, y + 34, w, 26, "Camera",
            items=cam_names, selected_index=cam_idx,
            on_change=lambda _i, _v: self._on_vision_setting(
                "webcam_index",
                self._cam_device_indices[_i] if _i < len(self._cam_device_indices) else _i,
            ),
        )

        self._vision_interval_slider = Slider(
            x, y + 74, w, 24, "Capture Interval",
            value=self._config.vision.capture_interval,
            min_val=5.0, max_val=120.0,
            on_change=lambda v: self._on_vision_setting("capture_interval", v),
        )

        self._vision_look_button = Button(
            x, y + 114, 120, 30, "Look Now",
            on_click=self._on_vision_look,
        )

        self._vision_widgets = [
            self._vision_source_dropdown,
            self._vision_cam_dropdown,
            self._vision_interval_slider,
            self._vision_look_button,
        ]

    # ── Public interface ──────────────────────────────────────────────

    def open(self) -> None:
        """Show the settings panel."""
        self.visible = True
        self._status_text = ""

    def close(self) -> None:
        """Hide the settings panel."""
        self.visible = False
        # Close any open dropdowns
        for dd in (
            self._model_dropdown,
            self._vision_source_dropdown,
            self._vision_cam_dropdown,
            self._output_device_dropdown,
            self._input_device_dropdown,
            self._tts_voice_dropdown,
        ):
            if dd is not None:
                dd.expanded = False

    def set_model_list(self, models: list[str]) -> None:
        """Update the LLM model dropdown with available models.

        Called after async Ollama query completes.
        """
        if self._model_dropdown is None:
            return
        # Find current model in the list
        current = self._config.llm.model
        idx = 0
        for i, m in enumerate(models):
            if m == current:
                idx = i
                break
        self._model_dropdown.set_items(models, selected_index=idx)
        self._models_loaded = True
        self._llm_status = f"Current: {current}"

    def update(self, dt: float) -> None:
        """Per-frame update (currently unused but available for animations)."""

    def render(self, surface: pygame.Surface) -> None:
        """Render the settings overlay onto the surface."""
        if not self.visible:
            return

        self._ensure_fonts()
        self._build_widgets()

        if self._font is None or self._title_font is None:
            return

        px = self._panel_x
        py = self._panel_y

        # Semi-transparent overlay
        overlay = pygame.Surface((self._screen_w, self._screen_h), pygame.SRCALPHA)
        overlay.fill(_OVERLAY_BG)
        surface.blit(overlay, (0, 0))

        # Panel background
        pygame.draw.rect(surface, _PANEL_BG, (px, py, _PANEL_WIDTH, _PANEL_HEIGHT))
        pygame.draw.rect(surface, _PANEL_BORDER, (px, py, _PANEL_WIDTH, _PANEL_HEIGHT), 1)

        # Header
        pygame.draw.rect(surface, _HEADER_BG, (px, py, _PANEL_WIDTH, _HEADER_HEIGHT))
        title_surf = self._title_font.render("Settings", True, _TITLE_COLOR)
        surface.blit(title_surf, (px + 12, py + 6))

        # Close button
        if self._close_button is not None:
            self._close_button.render(surface)

        # Tabs
        for i, btn in enumerate(self._tab_buttons):
            tab = list(SettingsTab)[i]
            btn.selected = (tab == self.active_tab)
            btn.render(surface)

        # Content area
        if self.active_tab == SettingsTab.PERSONALITY:
            self._render_personality_tab(surface)
        elif self.active_tab == SettingsTab.LLM:
            self._render_llm_tab(surface)
        elif self.active_tab == SettingsTab.AUDIO:
            self._render_audio_tab(surface)
        elif self.active_tab == SettingsTab.VISION:
            self._render_vision_tab(surface)

        # Status bar at bottom
        if self._status_text and self._font is not None:
            status_surf = self._font.render(self._status_text, True, _TEXT_DIM)
            sx = px + 12
            sy = py + _PANEL_HEIGHT - 24
            surface.blit(status_surf, (sx, sy))

    def _render_personality_tab(self, surface: pygame.Surface) -> None:
        """Render the personality tab content."""
        # Preset buttons
        for btn in self._preset_buttons:
            btn.selected = (btn.label == self._presets.get(
                self._active_preset, PresetConfig(name="", description="")
            ).name)
            btn.render(surface)

        # Description text
        if self._font is not None:
            preset = self._presets.get(self._active_preset)
            if preset:
                desc_surf = self._font.render(preset.description, True, _TEXT_DIM)
                surface.blit(desc_surf, (
                    self._panel_x + _CONTENT_PADDING,
                    self._panel_y + _HEADER_HEIGHT + _TAB_HEIGHT + _CONTENT_PADDING + 32,
                ))

        # Trait sliders (only interactive for "custom" preset)
        for slider in self._trait_sliders:
            slider.render(surface)

    def _render_llm_tab(self, surface: pygame.Surface) -> None:
        """Render the LLM model tab content."""
        for widget in self._llm_widgets:
            widget.render(surface)

        # Status text
        if self._llm_status and self._font is not None:
            status_surf = self._font.render(self._llm_status, True, _STATUS_GREEN)
            surface.blit(status_surf, (
                self._panel_x + _CONTENT_PADDING,
                self._panel_y + _HEADER_HEIGHT + _TAB_HEIGHT + _CONTENT_PADDING + 120,
            ))

    def _render_audio_tab(self, surface: pygame.Surface) -> None:
        """Render the audio tab content."""
        for widget in self._audio_widgets:
            widget.render(surface)

    def _render_vision_tab(self, surface: pygame.Surface) -> None:
        """Render the vision tab content."""
        for widget in self._vision_widgets:
            widget.render(surface)

        if self._font is not None:
            # Vision status
            source = self._config.vision.source
            enabled = self._config.vision.enabled
            if enabled and source != "off":
                status = f"Vision: active ({source})"
                color = _STATUS_GREEN
            else:
                status = "Vision: off"
                color = _TEXT_DIM
            status_surf = self._font.render(status, True, color)
            surface.blit(status_surf, (
                self._panel_x + _CONTENT_PADDING,
                self._panel_y + _HEADER_HEIGHT + _TAB_HEIGHT + _CONTENT_PADDING + 154,
            ))

            # Last observation text
            if self._last_observation_text:
                obs_surf = self._font.render(
                    f"Last: {self._last_observation_text[:80]}", True, _TEXT_DIM
                )
                surface.blit(obs_surf, (
                    self._panel_x + _CONTENT_PADDING,
                    self._panel_y + _HEADER_HEIGHT + _TAB_HEIGHT + _CONTENT_PADDING + 179,
                ))

    # ── Event handling ────────────────────────────────────────────────

    def handle_click(self, mx: int, my: int) -> bool:
        """Handle mouse click. Returns True if consumed by settings panel."""
        if not self.visible:
            return False

        # Close button
        if self._close_button is not None and self._close_button.handle_click(mx, my):
            return True

        # Tab buttons
        for btn in self._tab_buttons:
            if btn.handle_click(mx, my):
                return True

        # Active tab widgets
        if self.active_tab == SettingsTab.PERSONALITY:
            for btn in self._preset_buttons:
                if btn.handle_click(mx, my):
                    return True
            if self._active_preset == "custom":
                for slider in self._trait_sliders:
                    if slider.handle_click(mx, my):
                        return True
        elif self.active_tab == SettingsTab.LLM:
            # Dropdown should be checked first (may be expanded over other widgets)
            if self._model_dropdown is not None and self._model_dropdown.handle_click(mx, my):
                return True
            if self._temp_slider is not None and self._temp_slider.handle_click(mx, my):
                return True
            if self._llm_apply_button is not None and self._llm_apply_button.handle_click(mx, my):
                return True
        elif self.active_tab == SettingsTab.AUDIO:
            for widget in self._audio_widgets:
                if widget.handle_click(mx, my):
                    return True
        elif self.active_tab == SettingsTab.VISION:
            if (
                self._vision_source_dropdown is not None
                and self._vision_source_dropdown.handle_click(mx, my)
            ):
                # Apply source change only if value actually changed
                val = self._vision_source_dropdown.selected_value
                if val:
                    new_source = val.lower()
                    if new_source != self._last_vision_source:
                        self._last_vision_source = new_source
                        self._on_vision_setting("source", new_source)
                return True
            for widget in self._vision_widgets:
                if widget is not self._vision_source_dropdown:
                    if widget.handle_click(mx, my):
                        return True

        # Click inside panel but not on any widget — consume to prevent passthrough
        panel_rect = pygame.Rect(self._panel_x, self._panel_y, _PANEL_WIDTH, _PANEL_HEIGHT)
        if panel_rect.collidepoint(mx, my):
            return True

        return False

    def handle_mouse_move(self, mx: int, my: int) -> None:
        """Handle mouse motion for hover states and slider dragging."""
        if not self.visible:
            return

        if self._close_button is not None:
            self._close_button.handle_mouse_move(mx, my)

        for btn in self._tab_buttons:
            btn.handle_mouse_move(mx, my)

        if self.active_tab == SettingsTab.PERSONALITY:
            for btn in self._preset_buttons:
                btn.handle_mouse_move(mx, my)
            if self._active_preset == "custom":
                for slider in self._trait_sliders:
                    slider.handle_mouse_move(mx, my)
        elif self.active_tab == SettingsTab.LLM:
            if self._model_dropdown is not None:
                self._model_dropdown.handle_mouse_move(mx, my)
            if self._temp_slider is not None:
                self._temp_slider.handle_mouse_move(mx, my)
            if self._llm_apply_button is not None:
                self._llm_apply_button.handle_mouse_move(mx, my)
        elif self.active_tab == SettingsTab.AUDIO:
            for widget in self._audio_widgets:
                widget.handle_mouse_move(mx, my)
        elif self.active_tab == SettingsTab.VISION:
            for widget in self._vision_widgets:
                widget.handle_mouse_move(mx, my)

    def handle_mouse_up(self) -> None:
        """Handle mouse button release (stop slider dragging)."""
        if not self.visible:
            return

        if self.active_tab == SettingsTab.PERSONALITY:
            for slider in self._trait_sliders:
                slider.handle_mouse_up()
        elif self.active_tab == SettingsTab.LLM:
            if self._temp_slider is not None:
                self._temp_slider.handle_mouse_up()
        elif self.active_tab == SettingsTab.AUDIO:
            for widget in self._audio_widgets:
                widget.handle_mouse_up()
        elif self.active_tab == SettingsTab.VISION:
            if self._vision_interval_slider is not None:
                self._vision_interval_slider.handle_mouse_up()

    def handle_scroll(self, direction: int) -> bool:
        """Forward mouse wheel to any expanded dropdown. Returns True if consumed."""
        if not self.visible:
            return False
        if self.active_tab == SettingsTab.LLM:
            if self._model_dropdown is not None and self._model_dropdown.handle_scroll(direction):
                return True
        elif self.active_tab == SettingsTab.AUDIO:
            for widget in self._audio_widgets:
                if hasattr(widget, "handle_scroll") and widget.handle_scroll(direction):
                    return True
        elif self.active_tab == SettingsTab.VISION:
            for widget in self._vision_widgets:
                if hasattr(widget, "handle_scroll") and widget.handle_scroll(direction):
                    return True
        return False

    # ── Callbacks ─────────────────────────────────────────────────────

    def _close(self) -> None:
        """Close the settings panel and notify the owner."""
        self.close()
        if self.on_close is not None:
            self.on_close()

    def _switch_tab(self, tab: SettingsTab) -> None:
        """Switch to a different tab."""
        self.active_tab = tab
        # Close dropdowns when switching away
        if tab != SettingsTab.LLM and self._model_dropdown is not None:
            self._model_dropdown.expanded = False
        if tab != SettingsTab.VISION and self._vision_source_dropdown is not None:
            self._vision_source_dropdown.expanded = False

    def _select_preset(self, key: str) -> None:
        """Apply a personality preset."""
        preset = self._presets.get(key)
        if preset is None:
            return

        self._active_preset = key

        # Update button selection
        for btn in self._preset_buttons:
            btn.selected = (btn.label == preset.name)

        # Apply traits to config
        new_traits = dict(preset.traits)
        self._config.personality.base_traits.update(new_traits)

        # Update slider values
        for slider in self._trait_sliders:
            trait_key = slider.label.lower()
            if trait_key in new_traits:
                slider.value = new_traits[trait_key]

        self._status_text = f"Preset: {preset.name}"

        if self.on_personality_change is not None:
            self.on_personality_change(dict(self._config.personality.base_traits))

    def _on_trait_change(self, trait: str, value: float) -> None:
        """Handle individual trait slider change."""
        self._config.personality.base_traits[trait] = value
        self._status_text = f"{trait}: {value:.2f}"

        if self.on_personality_change is not None:
            self.on_personality_change(dict(self._config.personality.base_traits))

    def _apply_llm_settings(self) -> None:
        """Apply LLM model and temperature changes."""
        model = ""
        if self._model_dropdown is not None and self._model_dropdown.selected_value:
            model = self._model_dropdown.selected_value

        temp = self._config.llm.temperature
        if self._temp_slider is not None:
            temp = self._temp_slider.value

        if model and model != "Loading...":
            self._config.llm.model = model
            self._config.llm.temperature = temp
            self._llm_status = f"Applied: {model} (temp={temp:.1f})"
            self._status_text = self._llm_status

            if self.on_llm_apply is not None:
                self.on_llm_apply(model, temp)
        else:
            self._llm_status = "No model selected"

    def _on_audio_setting(self, key: str, value: Any) -> None:
        """Handle an audio setting change."""
        setattr(self._config.audio, key, value)
        self._status_text = f"Audio: {key} = {value}"

        if self.on_audio_change is not None:
            self.on_audio_change(key, value)

    def _on_vision_setting(self, key: str, value: Any) -> None:
        """Handle a vision setting change."""
        if key == "source":
            self._config.vision.source = str(value)
            # Update enabled based on source
            self._config.vision.enabled = str(value) != "off"
        elif key == "capture_interval":
            self._config.vision.capture_interval = float(value)
        self._status_text = f"Vision: {key} = {value}"

        if self.on_vision_change is not None:
            self.on_vision_change(key, value)

    def _on_vision_look(self) -> None:
        """Handle 'Look Now' button click."""
        self._status_text = "Vision: looking..."
        if self.on_vision_change is not None:
            self.on_vision_change("look_now", True)

    def refresh_device_lists(self) -> None:
        """Re-enumerate audio/video devices and queue results for main thread.

        This method is safe to call from a background thread. It collects
        device lists into ``_pending_refresh`` which is applied on the main
        thread via ``apply_pending_refresh()``.
        """
        if not self._widgets_built:
            return

        with self._refresh_lock:
            if self._refreshing:
                return
            self._refreshing = True

        try:
            result: dict[str, Any] = {}

            if self._output_device_dropdown is not None:
                out_devs = list_audio_output_devices()
                out_names = [name for _, name in out_devs]
                out_idx = _find_saved_index(
                    out_names, self._config.audio.audio_output_device
                )
                result["output"] = (out_names, out_idx)

            if self._input_device_dropdown is not None:
                in_devs = list_audio_input_devices()
                in_names = [name for _, name in in_devs]
                in_idx = _find_saved_index(
                    in_names, self._config.audio.audio_input_device
                )
                result["input"] = (in_names, in_idx)

            if self._tts_voice_dropdown is not None:
                voices = list_tts_voices(self._config.audio.tts_provider)
                voice_names = [name for _, name in voices]
                voice_ids = [vid for vid, _ in voices]
                voice_idx = _find_saved_index(voice_ids, self._config.audio.tts_voice)
                result["tts_voice"] = (voice_names, voice_ids, voice_idx)

            if self._vision_cam_dropdown is not None:
                cams = list_webcams()
                cam_names = [name for _, name in cams]
                cam_indices = [dev_idx for dev_idx, _ in cams]
                cam_idx = 0
                saved_cam = self._config.vision.webcam_index
                for i, dev_idx in enumerate(cam_indices):
                    if dev_idx == saved_cam:
                        cam_idx = i
                        break
                result["camera"] = (cam_names, cam_indices, cam_idx)

            self._pending_refresh = result
        except Exception as exc:
            logger.error("Device refresh failed: %s", exc, exc_info=True)
        finally:
            with self._refresh_lock:
                self._refreshing = False

    def apply_pending_refresh(self) -> None:
        """Apply queued device list results to dropdowns (main thread only).

        Called from the game loop's ``_update()`` to ensure pygame widget
        updates happen on the main thread.
        """
        pending = self._pending_refresh
        if pending is None:
            return
        self._pending_refresh = None

        if "output" in pending and self._output_device_dropdown is not None:
            names, idx = pending["output"]
            self._output_device_dropdown.set_items(names, selected_index=idx)

        if "input" in pending and self._input_device_dropdown is not None:
            names, idx = pending["input"]
            self._input_device_dropdown.set_items(names, selected_index=idx)

        if "tts_voice" in pending and self._tts_voice_dropdown is not None:
            names, voice_ids, idx = pending["tts_voice"]
            self._voice_ids = voice_ids
            self._tts_voice_dropdown.set_items(names, selected_index=idx)

        if "camera" in pending and self._vision_cam_dropdown is not None:
            names, indices, idx = pending["camera"]
            self._cam_device_indices = indices
            self._vision_cam_dropdown.set_items(names, selected_index=idx)

    def set_last_observation(self, text: str) -> None:
        """Update the displayed last observation text.

        Args:
            text: The most recent vision observation.
        """
        self._last_observation_text = text
