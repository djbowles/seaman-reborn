"""Thin GameEngine orchestrator.

Wires together layout, scene management, input handling, rendering,
and all game subsystems. Delegates to focused modules:
- game_systems.py: needs, mood, behavior, events, evolution, death
- scene_manager.py: PLAYING/SETTINGS/LINEAGE state machine + drawer animation
- input_handler.py: keyboard/mouse event routing
- render_engine.py: gradient cache + particles

Loop order: events → input_handler → game_systems.tick(dt) →
scene_manager.update(dt) → render.
"""
from __future__ import annotations

import logging
import queue
import time
from typing import Any

import pygame

from seaman_brain.audio.manager import AudioManager
from seaman_brain.behavior.autonomous import BehaviorEngine
from seaman_brain.behavior.events import EventSystem
from seaman_brain.behavior.mood import MoodEngine
from seaman_brain.config import (
    SeamanConfig,
    flush_pending_save,
    load_config,
    save_user_settings,
)
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.clock import GameClock
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.gui.audio_integration import PygameAudioBridge
from seaman_brain.gui.chat_panel import ChatPanel
from seaman_brain.gui.game_systems import GameState
from seaman_brain.gui.hud import HUD
from seaman_brain.gui.input_handler import InputHandler
from seaman_brain.gui.layout import ScreenLayout
from seaman_brain.gui.lineage_panel import LineagePanel
from seaman_brain.gui.scene_manager import SceneManager
from seaman_brain.gui.settings_panel import SettingsPanel
from seaman_brain.gui.sprites import CreatureRenderer
from seaman_brain.gui.tank_renderer import TankRenderer
from seaman_brain.gui.window import GameWindow
from seaman_brain.llm.scheduler import ModelScheduler
from seaman_brain.needs.death import DeathEngine
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine

logger = logging.getLogger(__name__)


class GameEngine:
    """Thin orchestrator wiring all subsystems into the main loop."""

    def __init__(self) -> None:
        # Config
        try:
            self._config = load_config()
        except FileNotFoundError:
            self._config = SeamanConfig()

        cfg = self._config
        w = cfg.gui.window_width
        h = cfg.gui.window_height

        # Layout + scene
        self._layout = ScreenLayout(w, h)
        self._scene_manager = SceneManager()
        self._input_handler = InputHandler()

        # GUI components
        tank = self._layout.tank
        self._tank_renderer = TankRenderer(tank.w, tank.h)
        self._creature_renderer = CreatureRenderer()
        self._hud = HUD(self._layout)
        self._chat_panel = ChatPanel(self._layout)
        self._chat_panel.on_submit = self._on_chat_submit
        self._settings_panel = SettingsPanel(
            width=self._layout.drawer_width,
            on_personality_change=self._on_personality_change,
            on_llm_apply=self._on_llm_apply,
            on_audio_change=self._on_audio_change,
            on_vision_change=self._on_vision_change,
        )
        self._lineage_panel = LineagePanel(
            width=self._layout.drawer_width,
            on_select=self._on_lineage_select,
        )

        # Window
        self.window = GameWindow(config=cfg)

        # Game state
        self._creature_state = CreatureState()
        self._tank_env = TankEnvironment.from_config(cfg.environment)
        self._clock = GameClock()
        self._needs = CreatureNeeds()

        # Game logic engines
        self._needs_engine = NeedsEngine(
            config=cfg.needs, env_config=cfg.environment,
        )
        self._mood_engine = MoodEngine()
        self._behavior_engine = BehaviorEngine()
        self._event_system = EventSystem()
        self._evolution_engine = EvolutionEngine(config=cfg.creature)
        self._death_engine = DeathEngine(
            needs_config=cfg.needs, env_config=cfg.environment,
        )
        self._scheduler = ModelScheduler(enabled=False)

        # Audio / vision (lazy)
        self._audio_manager: AudioManager | None = None
        self._audio_bridge: PygameAudioBridge | None = None

        # Conversation state
        self._pending_response: Any = None
        self._pending_response_time: float = 0.0
        self._stream_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_sentence_buffer: str = ""
        self._stt_queued_text: str | None = None
        self._stt_queued_time: float = 0.0

        # Misc state
        self.game_over = False
        self._needs_timer = 0.0
        self._behavior_timer = 0.0
        self._event_timer = 0.0

        # Wire input handler
        self._input_handler.on_escape = self._on_escape
        self._input_handler.on_toggle_settings = self._toggle_settings
        self._input_handler.on_key_down = self._on_key_down
        self._input_handler.on_mouse_click = self._on_mouse_click
        self._input_handler.on_mouse_move = self._on_mouse_move
        self._input_handler.on_mouse_up = self._on_mouse_up
        self._input_handler.on_mouse_scroll = self._on_mouse_scroll

    # ── Lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialize window, audio, and register callbacks."""
        self.window.initialize()

        # Audio
        try:
            self._audio_manager = AudioManager(config=self._config.audio)
        except Exception as exc:
            logger.warning("AudioManager init failed: %s", exc)

        self._audio_bridge = PygameAudioBridge(
            audio_manager=self._audio_manager,
            audio_config=self._config.audio,
            async_loop=self.window._loop,
            on_stt_result=self._on_stt_result,
        )

        # Register update + render callbacks
        self.window.register_update(self._update)
        self.window.register_renderer(self._render)

    def run(self) -> None:
        """Start the game."""
        self.initialize()
        try:
            self.window.run()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown."""
        flush_pending_save()
        if self._audio_bridge is not None:
            self._audio_bridge.cleanup()

    # ── Main loop ─────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        """Per-frame update."""
        self._scene_manager.update(dt)
        self._hud.update(dt)

        if self._scene_manager.state != GameState.PLAYING:
            return
        if self.game_over:
            return

        # Tick tank
        self._tank_env.update(dt)

        # Update creature renderer
        self._creature_renderer.update(dt)

    def _render(self, surface: pygame.Surface) -> None:
        """Per-frame render."""
        layout = self._layout
        tank = layout.tank

        # Tank background
        self._tank_renderer.render(surface, tank.x, tank.y)

        # Creature
        self._creature_renderer.render(surface, tank.x, tank.y)

        # HUD (top bar + sidebar)
        self._hud.render(surface)

        # Chat overlay
        self._chat_panel.render(surface)

        # Drawer overlays
        state = self._scene_manager.state
        progress = self._scene_manager.drawer_progress
        if state == GameState.SETTINGS:
            self._settings_panel.render(surface, progress)
        elif state == GameState.LINEAGE:
            self._lineage_panel.render(surface, progress)

    # ── Input handlers ────────────────────────────────────────────────

    def _on_escape(self) -> None:
        if self._scene_manager.state != GameState.PLAYING:
            self._scene_manager.close_drawer()

    def _toggle_settings(self) -> None:
        sm = self._scene_manager
        if sm.state == GameState.SETTINGS:
            sm.close_drawer()
        else:
            sm.open_settings()

    def _on_key_down(self, event: Any) -> None:
        key = getattr(event, "key", 0)
        char = getattr(event, "unicode", "")
        self._chat_panel.handle_key(key, char)

    def _on_mouse_click(self, event: Any) -> None:
        mx = getattr(event, "pos", (0, 0))[0]
        my = getattr(event, "pos", (0, 0))[1]

        # HUD clicks
        action = self._hud.handle_click(mx, my)
        if action is not None:
            return

        # Check settings/lineage rect hits
        if self._hud.settings_rect:
            rx, ry, rw, rh = self._hud.settings_rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._toggle_settings()
                return

        # Drawer clicks
        state = self._scene_manager.state
        if state == GameState.SETTINGS:
            self._settings_panel.handle_click(mx, my)
            return
        if state == GameState.LINEAGE:
            self._lineage_panel.handle_click(mx, my)
            return

        # Chat clicks
        self._chat_panel.handle_click(mx, my)

    def _on_mouse_move(self, event: Any) -> None:
        mx = getattr(event, "pos", (0, 0))[0]
        my = getattr(event, "pos", (0, 0))[1]
        self._hud.handle_hover(mx, my)

    def _on_mouse_up(self, event: Any) -> None:
        state = self._scene_manager.state
        if state == GameState.SETTINGS:
            self._settings_panel.handle_mouse_up()
        elif state == GameState.LINEAGE:
            self._lineage_panel.handle_mouse_up()

    def _on_mouse_scroll(self, event: Any) -> None:
        y = getattr(event, "y", 0)
        if y != 0:
            self._chat_panel.handle_scroll(y)

    # ── Chat ──────────────────────────────────────────────────────────

    def _on_chat_submit(self, text: str) -> None:
        """Handle chat input submission."""
        self._chat_panel.add_message("user", text)

    def _on_stt_result(self, text: str) -> None:
        """Handle speech-to-text result."""
        self._stt_queued_text = text
        self._stt_queued_time = time.monotonic()

    # ── Settings callbacks ────────────────────────────────────────────

    def _on_personality_change(self, traits: dict[str, float]) -> None:
        save_user_settings(self._config)

    def _on_llm_apply(self, model: str, temperature: float) -> None:
        self._config.llm.model = model
        self._config.llm.temperature = temperature
        save_user_settings(self._config)

    def _on_audio_change(self, key: str, value: Any) -> None:
        setattr(self._config.audio, key, value)
        save_user_settings(self._config)

    def _on_vision_change(self, key: str, value: Any) -> None:
        setattr(self._config.vision, key, value)
        save_user_settings(self._config)

    # ── Lineage callbacks ─────────────────────────────────────────────

    def _on_lineage_select(self, name: str) -> None:
        logger.info("Switching to bloodline: %s", name)
