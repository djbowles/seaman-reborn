"""Full game engine integrating all Pygame subsystems.

Orchestrates the tank renderer, creature sprites, chat panel, HUD,
interactions, audio bridge, needs system, mood engine, autonomous
behaviors, and event system into the main game loop.

Loop order: process input -> update needs/mood/behaviors/events ->
update animations -> render tank -> render creature -> render HUD ->
render chat -> flip display.
"""

from __future__ import annotations

import logging
import queue
import re
import threading
import time
from enum import Enum
from typing import Any

import pygame

from seaman_brain.audio.manager import AudioManager
from seaman_brain.behavior.autonomous import (
    BehaviorEngine,
    BehaviorType,
    IdleBehavior,
    get_behavior_situation,
)
from seaman_brain.behavior.events import EventSystem
from seaman_brain.behavior.mood import CreatureMood, MoodEngine
from seaman_brain.config import SeamanConfig, load_config, save_user_settings
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.clock import GameClock
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.gui.action_bar import ActionBar
from seaman_brain.gui.audio_integration import AudioChannel, PygameAudioBridge
from seaman_brain.gui.chat_panel import ChatPanel
from seaman_brain.gui.hud import HUD
from seaman_brain.gui.interactions import InteractionManager, InteractionType
from seaman_brain.gui.lineage_panel import LineagePanel
from seaman_brain.gui.settings_panel import SettingsPanel
from seaman_brain.gui.sprites import AnimationState, CreatureRenderer
from seaman_brain.gui.tank_renderer import TankRenderer
from seaman_brain.gui.window import GameWindow
from seaman_brain.llm.scheduler import ModelScheduler
from seaman_brain.needs.care import AERATOR_COOLDOWN_SECONDS, CLEANING_DURATION_SECONDS
from seaman_brain.needs.death import DeathCause, DeathEngine
from seaman_brain.needs.feeding import FoodType
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine
from seaman_brain.personality.traits import TraitProfile
from seaman_brain.types import CreatureStage, MessageRole
from seaman_brain.vision.bridge import VisionBridge

logger = logging.getLogger(__name__)

# ── Colors for overlays ──────────────────────────────────────────────

_GAMEOVER_BG = (10, 5, 5, 200)
_GAMEOVER_TEXT = (220, 60, 60)
_GAMEOVER_HINT = (180, 180, 180)
_EVOLUTION_GLOW = (255, 220, 100)
_NOTIFICATION_BG = (20, 40, 60, 200)
_NOTIFICATION_TEXT = (220, 220, 180)
_FOOD_MENU_BG = (20, 35, 58, 230)
_FOOD_MENU_BORDER = (60, 90, 130)
_FOOD_MENU_HOVER = (40, 65, 100)
_FOOD_MENU_TEXT = (200, 220, 240)
_FOOD_MENU_ITEM_H = 32
_FOOD_MENU_WIDTH = 120

# ── Game state enum ──────────────────────────────────────────────────


class GameState(Enum):
    """Top-level game state for input/update gating."""

    PLAYING = "playing"
    SETTINGS = "settings"
    LINEAGE = "lineage"


_NEEDS_UPDATE_INTERVAL = 1.0  # seconds between needs ticks
_BEHAVIOR_CHECK_INTERVAL = 15.0  # seconds between behavior checks
_EVENT_CHECK_INTERVAL = 3.0  # seconds between event checks
_VISION_LOOK_TIMEOUT = 30.0  # seconds before Look Now gives up
_STT_DEBOUNCE_SECONDS = 0.5  # wait for speech to settle before submitting
_PENDING_TIMEOUT = 60.0  # seconds before a stuck pending future is force-cancelled

# Sentence boundary for incremental TTS: .!? followed by whitespace or end-of-string
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:\s|$)")

# Situation prompts for interaction reactions via LLM
_INTERACTION_SITUATIONS: dict[str, str] = {
    "feed": "Your owner just fed you. React to receiving food.",
    "tap_glass": "Your owner just tapped the glass of your tank. React to the disturbance.",
    "clean": "Your owner just cleaned your tank. React to the improved cleanliness.",
    "aerate": "Your owner just aerated your tank water. React to the fresh bubbles.",
    "temp_up": "Your owner just raised your tank temperature. React to the warmth change.",
    "temp_down": "Your owner just lowered your tank temperature. React to the temperature drop.",
    "drain": "Your owner just changed the water level in your tank. React to the change.",
}


class GameEngine:
    """Orchestrates all Pygame subsystems into a cohesive game.

    Wires up the GameWindow (display + async bridge), TankRenderer,
    CreatureRenderer, ChatPanel, HUD, InteractionManager, AudioBridge,
    NeedsEngine, MoodEngine, BehaviorEngine, EventSystem, EvolutionEngine,
    and DeathEngine into the main loop.

    Attributes:
        window: The underlying GameWindow.
        game_over: Whether the creature has died.
    """

    def __init__(self, config: SeamanConfig | None = None) -> None:
        if config is None:
            try:
                cfg = load_config()
            except FileNotFoundError:
                cfg = SeamanConfig()
        else:
            cfg = config
        self._config = cfg

        # Core window
        self.window = GameWindow(config=cfg)

        # Game state
        self._creature_state = CreatureState()
        self._tank = TankEnvironment.from_config(cfg.environment)
        self._clock = GameClock()
        self._needs = CreatureNeeds()

        # Subsystems
        self._tank_renderer = TankRenderer(
            gui_config=cfg.gui, env_config=cfg.environment
        )
        self._creature_renderer = CreatureRenderer(
            stage=self._creature_state.stage,
            gui_config=cfg.gui,
        )
        self._chat_panel = ChatPanel(
            gui_config=cfg.gui,
            on_submit=self._on_chat_submit,
        )
        self._hud = HUD(gui_config=cfg.gui)
        self._interaction_manager = InteractionManager(
            gui_config=cfg.gui,
            env_config=cfg.environment,
            needs_config=cfg.needs,
        )
        self._action_bar = ActionBar(on_action=self._on_action_bar)
        self._audio_manager: AudioManager | None = None
        self._audio_bridge: PygameAudioBridge | None = None
        self._vision_bridge: VisionBridge | None = None
        self._scheduler = ModelScheduler(enabled=False)

        # Game logic engines
        self._needs_engine = NeedsEngine(config=cfg.needs, env_config=cfg.environment)
        self._mood_engine = MoodEngine()
        self._behavior_engine = BehaviorEngine()
        self._event_system = EventSystem()
        self._evolution_engine = EvolutionEngine(config=cfg.creature)
        self._death_engine = DeathEngine(
            needs_config=cfg.needs, env_config=cfg.environment
        )

        # Timers
        self._needs_timer = 0.0
        self._behavior_timer = 0.0
        self._event_timer = 0.0
        self._interaction_count_delta = 0

        # Game over state
        self.game_over = False
        self._death_cause: DeathCause | None = None
        self._death_message = ""

        # Evolution celebration state
        self._evolution_active = False
        self._evolution_timer = 0.0
        self._evolution_duration = 3.0  # seconds

        # Notifications
        self._notifications: list[tuple[str, float]] = []  # (text, remaining_seconds)
        self._notification_duration = 4.0

        # Game state (playing vs settings overlay)
        self._game_state = GameState.PLAYING
        self._settings_panel: SettingsPanel | None = None
        self._lineage_panel: LineagePanel | None = None

        # Pending conversation future
        self._pending_response: Any = None
        self._pending_response_time: float = 0.0
        self._stream_queue: queue.Queue[str | None] = queue.Queue()
        self._tts_sentence_buffer: str = ""

        # Pending autonomous LLM remark (lower priority than user chat)
        self._pending_autonomous: Any = None
        self._pending_autonomous_time: float = 0.0
        self._pending_autonomous_behavior: IdleBehavior | None = None

        # STT queue — debounced and non-cancelling
        self._stt_queued_text: str | None = None
        self._stt_queued_time: float = 0.0

        # "Look Now" observation tracking
        self._vision_look_prev_count: int | None = None
        self._vision_look_start_time: float = 0.0

        # Thread-safe queue for model list updates from async callback
        self._pending_model_list: list[str] | None = None

        # Food selection submenu
        self._food_menu_visible = False
        self._food_menu_items: list[tuple[FoodType, pygame.Rect]] = []
        self._food_menu_hovered: int = -1

        # Diagnostic heartbeat
        self._heartbeat_timer: float = 0.0
        self._frame_count: int = 0

        # Font for overlays (lazy)
        self._overlay_font: pygame.font.Font | None = None
        self._overlay_title_font: pygame.font.Font | None = None

    @property
    def creature_state(self) -> CreatureState:
        """Current creature state."""
        return self._creature_state

    @property
    def tank(self) -> TankEnvironment:
        """Current tank environment."""
        return self._tank

    def _ensure_overlay_fonts(self) -> None:
        """Initialize overlay fonts if not yet done."""
        if self._overlay_font is None:
            for name in ("consolas", "couriernew", "courier"):
                try:
                    self._overlay_font = pygame.font.SysFont(name, 16)
                    self._overlay_title_font = pygame.font.SysFont(name, 28, bold=True)
                    break
                except Exception:
                    continue
            if self._overlay_font is None:
                self._overlay_font = pygame.font.Font(None, 16)
                self._overlay_title_font = pygame.font.Font(None, 28)

    def initialize(self) -> None:
        """Initialize all subsystems and register with the window."""
        self.window.initialize()

        # Create AudioManager for TTS/STT and pass to audio bridge
        try:
            self._audio_manager = AudioManager(config=self._config.audio)
            logger.info("AudioManager initialized (TTS: %s)", self._config.audio.tts_provider)
        except Exception as exc:
            logger.warning("AudioManager creation failed: %s", exc)
            self._audio_manager = None

        self._audio_bridge = PygameAudioBridge(
            audio_manager=self._audio_manager,
            audio_config=self._config.audio,
            async_loop=self.window._loop,
            on_stt_result=self._on_stt_result,
        )

        # Start full-duplex audio pipeline if AEC is enabled
        if self._audio_manager is not None and self._config.audio.aec_enabled:
            self._audio_manager.start_pipeline(loop=self.window._loop)

        # Set up vision bridge if enabled
        if self._config.vision.enabled:
            self._vision_bridge = VisionBridge(
                vision_config=self._config.vision,
                async_loop=self.window._loop,
                scheduler=self._scheduler,
            )

        # Compute layout: action bar on right, tank shrunk
        action_bar_w = 160
        window_w = self._config.gui.window_width
        window_h = self._config.gui.window_height
        tank_w = window_w - action_bar_w
        top_margin = 45

        self._tank_renderer.set_render_area(0, top_margin, tank_w, window_h - top_margin)
        render_area = self._tank_renderer.render_area
        self._creature_renderer.set_bounds(*render_area)
        self._interaction_manager.set_tank_area(*render_area)
        self._interaction_manager.disable_buttons()
        self._action_bar.set_panel_area(tank_w, top_margin, action_bar_w, window_h - top_margin)

        # Settings panel
        self._settings_panel = SettingsPanel(
            config=self._config,
            screen_width=self._config.gui.window_width,
            screen_height=self._config.gui.window_height,
            on_personality_change=self._on_personality_change,
            on_llm_apply=self._on_llm_apply,
            on_audio_change=self._on_audio_change,
            on_vision_change=self._on_vision_change,
            on_close=self._on_settings_close,
        )

        # Lineage panel
        self._lineage_panel = LineagePanel(
            screen_width=self._config.gui.window_width,
            screen_height=self._config.gui.window_height,
            save_base_dir=self._config.creature.save_path,
            on_switch=self._switch_bloodline,
            on_new=self._new_bloodline,
            on_delete=self._delete_bloodline,
            on_close=self._on_lineage_close,
        )

        # Register event handlers
        self.window.register_event_handler(pygame.MOUSEBUTTONDOWN, self._on_mouse_click)
        self.window.register_event_handler(pygame.MOUSEMOTION, self._on_mouse_move)
        self.window.register_event_handler(pygame.MOUSEBUTTONUP, self._on_mouse_up)
        self.window.register_event_handler(pygame.MOUSEWHEEL, self._on_mouse_scroll)
        self.window.register_event_handler(pygame.KEYDOWN, self._on_key_down)

        # Register update and render callbacks
        self.window.register_update(self._update)
        self.window.register_renderer(self._render)

        logger.info("GameEngine initialized")

    def run(self) -> None:
        """Start the game loop."""
        self.initialize()
        self.window.run()

    def _update(self, dt: float) -> None:
        """Per-frame update of all game subsystems.

        Args:
            dt: Delta time in seconds since last frame.
        """
        # Diagnostic heartbeat — log game loop health every 30s
        self._frame_count += 1
        self._heartbeat_timer += dt
        if self._heartbeat_timer >= 30.0:
            logger.info(
                "Heartbeat: state=%s game_over=%s evolution=%s "
                "pending_response=%s pending_autonomous=%s frames=%d",
                self._game_state.value,
                self.game_over,
                self._evolution_active,
                self._pending_response is not None,
                self._pending_autonomous is not None,
                self._frame_count,
            )
            self._heartbeat_timer = 0.0

        # Apply pending model list from async thread (thread-safe)
        if self._pending_model_list is not None and self._settings_panel is not None:
            try:
                self._settings_panel.set_model_list(self._pending_model_list)
            except Exception as exc:
                logger.error("Failed to apply model list: %s", exc, exc_info=True)
            self._pending_model_list = None

        # Settings overlay: only update the panel, skip gameplay
        if self._game_state == GameState.SETTINGS:
            if self._settings_panel is not None:
                try:
                    self._settings_panel.update(dt)
                    self._settings_panel.apply_pending_refresh()
                except Exception as exc:
                    logger.error("Settings panel update error: %s", exc, exc_info=True)
            # Still process pending vision results so "Look Now" works
            if self._vision_bridge is not None:
                self._vision_bridge._check_pending()
                self._check_vision_look_result()
            return

        # Lineage overlay: skip gameplay
        if self._game_state == GameState.LINEAGE:
            return

        if self.game_over:
            return

        if self._evolution_active:
            self._update_evolution_celebration(dt)
            return

        # ── Tank + needs + death ──────────────────────────────────────
        try:
            self._tank.update(dt, self._config.environment)
        except Exception as exc:
            logger.error("Tank update error: %s", exc, exc_info=True)

        # Accumulate timers
        self._needs_timer += dt
        self._behavior_timer += dt
        self._event_timer += dt

        # Periodic needs update
        if self._needs_timer >= _NEEDS_UPDATE_INTERVAL:
            elapsed = self._needs_timer
            self._needs_timer = 0.0
            self._creature_state.age += elapsed
            try:
                self._update_needs(elapsed)
            except Exception as exc:
                logger.error("Needs update error (continuing with stale state): %s", exc)

        try:
            cause = self._death_engine.check_death(
                self._creature_state, self._needs, self._tank
            )
            if cause is not None:
                self._handle_death(cause)
                return
        except Exception as exc:
            logger.error("Death check error: %s", exc, exc_info=True)

        # ── Mood + behavior + events + evolution ─────────────────────
        try:
            time_context = self._clock.get_time_context()
            traits = self._get_traits()
            mood = self._mood_engine.calculate_mood(
                needs=self._needs,
                trust=self._creature_state.trust_level,
                time_context=time_context,
                recent_interactions=self._creature_state.interaction_count,
                traits=traits,
            )
            self._creature_state.mood = mood.value
        except Exception as exc:
            logger.error("Mood update error: %s", exc, exc_info=True)
            time_context = {}

        try:
            if self._behavior_timer >= _BEHAVIOR_CHECK_INTERVAL:
                self._behavior_timer = 0.0
                self._check_behaviors(time_context)

            if self._event_timer >= _EVENT_CHECK_INTERVAL:
                self._event_timer = 0.0
                self._check_events(time_context)

            self._check_evolution()
        except Exception as exc:
            logger.error("Behavior/event/evolution error: %s", exc, exc_info=True)

        # ── Animations + action bar ──────────────────────────────────
        try:
            self._tank_renderer.update(dt, self._tank)
            self._creature_renderer.update(dt)
            self._chat_panel.update(dt)
            self._hud.update(dt)
            self._interaction_manager.update(dt)

            im = self._interaction_manager
            self._action_bar.update_cooldowns(
                feed_remaining=im.feeding_engine.cooldown_remaining(self._creature_state),
                feed_max=self._config.needs.feeding_cooldown_seconds,
                clean_remaining=im.care_engine.cleaning_cooldown_remaining(),
                clean_max=CLEANING_DURATION_SECONDS,
                aerate_remaining=im.care_engine.aerating_cooldown_remaining(),
                aerate_max=AERATOR_COOLDOWN_SECONDS,
            )
        except Exception as exc:
            logger.error("Animation/action bar error: %s", exc, exc_info=True)

        # ── Audio + vision bridges ───────────────────────────────────
        try:
            if self._audio_bridge is not None:
                self._audio_bridge.update(dt)
                self._hud.mic_active = self._audio_bridge.mic_active
            self._hud.tts_active = (
                self._audio_manager is not None and self._audio_manager.tts_enabled
            )

            if self._vision_bridge is not None:
                self._vision_bridge.update(dt, self.window.screen)
                self._check_vision_look_result()
                manager = self.window.manager
                if manager is not None:
                    manager.set_vision_observations(
                        self._vision_bridge.get_recent_observations()
                    )
                    # Wire vision bridge for LLM-initiated vision (once)
                    if (
                        hasattr(manager, "set_vision_bridge")
                        and getattr(manager, "_vision_bridge", None) is None
                    ):
                        manager.set_vision_bridge(self._vision_bridge)
        except Exception as exc:
            logger.error("Audio/vision bridge error: %s", exc, exc_info=True)

        # ── Notifications + STT + pending responses + stage sync ─────
        try:
            alive: list[tuple[str, float]] = []
            for text, remaining in self._notifications:
                remaining -= dt
                if remaining > 0:
                    alive.append((text, remaining))
            self._notifications = alive

            self._check_stt_queue()
            self._check_pending_response()
            self._check_pending_autonomous()

            if self._creature_renderer.stage != self._creature_state.stage:
                self._creature_renderer.set_stage(self._creature_state.stage)
        except Exception as exc:
            logger.error("Notification/STT/response error: %s", exc, exc_info=True)

    def _update_needs(self, elapsed: float) -> None:
        """Update creature needs based on elapsed time."""
        self._needs = self._needs_engine.update(
            elapsed_seconds=elapsed,
            creature_state=self._creature_state,
            tank=self._tank,
            interaction_count_delta=self._interaction_count_delta,
        )
        self._needs_engine.apply_to_state(self._creature_state, self._needs)
        self._interaction_count_delta = 0

    def _get_traits(self) -> TraitProfile:
        """Get the current TraitProfile from ConversationManager, or default."""
        manager = self.window.manager
        if manager is not None and hasattr(manager, "traits") and manager.traits is not None:
            return manager.traits
        return TraitProfile()

    def _check_behaviors(self, time_context: dict) -> None:
        """Check for autonomous creature behaviors.

        Verbal (LLM-generated) remarks are suppressed unless the creature
        is in critical condition — starving, low health, or very
        uncomfortable.  This keeps idle chatter from overwhelming the
        conversation and from triggering TTS/STT feedback.
        """
        creature_dict = {
            "stage": self._creature_state.stage.value,
            "mood": self._creature_state.mood,
            "trust": self._creature_state.trust_level,
            "hunger": self._creature_state.hunger,
        }
        traits = self._get_traits()
        behavior = self._behavior_engine.get_idle_behavior(
            creature_state=creature_dict,
            needs=self._needs,
            mood=self._mood_engine.current_mood,
            time_context=time_context,
            traits=traits,
        )
        if behavior is not None:
            if behavior.needs_llm:
                # Only let the creature speak unprompted when it's suffering
                needs_critical = (
                    self._needs.hunger >= 0.7
                    or self._needs.health <= 0.3
                    or self._needs.comfort <= 0.2
                )
                if needs_critical:
                    self._request_autonomous_remark(behavior)
                # else: silently skip the remark
            else:
                self._apply_behavior(behavior)

    def _apply_behavior(self, behavior: IdleBehavior) -> None:
        """Apply an autonomous behavior — set animation and optionally speak."""
        # Map behavior type to animation state
        anim_map = {
            BehaviorType.IDLE_SWIM: AnimationState.SWIMMING,
            BehaviorType.SLEEP: AnimationState.SLEEPING,
            BehaviorType.EAT: AnimationState.EATING,
            BehaviorType.OBSERVE: AnimationState.IDLE,
            BehaviorType.COMPLAIN: AnimationState.TALKING,
            BehaviorType.TAP_GLASS: AnimationState.IDLE,
        }
        anim = anim_map.get(behavior.action_type, AnimationState.IDLE)
        self._creature_renderer.set_animation(anim)

        if behavior.message:
            self._chat_panel.add_message(MessageRole.ASSISTANT, behavior.message)
            if self._audio_bridge is not None:
                self._audio_bridge.play_voice(behavior.message)

    def _request_autonomous_remark(self, behavior: IdleBehavior) -> None:
        """Submit an autonomous LLM remark for a verbal behavior.

        Gates on both ``_pending_response`` and ``_pending_autonomous`` being
        empty.  If the LLM is busy, falls back to the canned message.
        """
        # LLM busy — fall back to canned message
        if self._pending_response is not None or self._pending_autonomous is not None:
            logger.debug(
                "Autonomous LLM busy (chat=%s, auto=%s), using canned for %s",
                self._pending_response is not None,
                self._pending_autonomous is not None,
                behavior.action_type.value,
            )
            self._apply_behavior(behavior)
            return

        manager = self.window.manager
        if manager is None or not manager.is_initialized:
            logger.debug("Autonomous LLM: manager not ready, using canned")
            self._apply_behavior(behavior)
            return

        situation = get_behavior_situation(
            behavior.action_type,
            self._mood_engine.current_mood,
            self._needs,
        )
        if situation is None:
            self._apply_behavior(behavior)
            return

        logger.info(
            "Requesting autonomous LLM remark: %s (mood=%s)",
            behavior.action_type.value,
            self._mood_engine.current_mood.value,
        )
        self._pending_autonomous_behavior = behavior
        self._scheduler.acquire("chat")

        async def _generate() -> str | None:
            return await manager.generate_autonomous_remark(situation)

        try:
            self._pending_autonomous = self.window.submit_async(_generate())
            self._pending_autonomous_time = time.monotonic()
        except RuntimeError:
            logger.warning("Async bridge dead — falling back to canned behavior")
            self._scheduler.release("chat")
            self._pending_autonomous_behavior = None
            self._apply_behavior(behavior)

    def _check_pending_autonomous(self) -> None:
        """Check if a pending autonomous LLM remark is ready."""
        if self._pending_autonomous is None:
            return

        if not self._pending_autonomous.done():
            # Force-cancel if stuck beyond timeout
            if time.monotonic() - self._pending_autonomous_time > _PENDING_TIMEOUT:
                logger.warning(
                    "Pending autonomous remark timed out after %.0fs", _PENDING_TIMEOUT
                )
                self._pending_autonomous.cancel()
                self._pending_autonomous = None
                self._pending_autonomous_behavior = None
                self._scheduler.release("chat")
            return

        # Cancelled by user chat — silently discard
        if self._pending_autonomous.cancelled():
            logger.debug("Autonomous remark cancelled (user chat took priority)")
            self._pending_autonomous = None
            self._pending_autonomous_behavior = None
            self._scheduler.release("chat")
            return

        behavior = self._pending_autonomous_behavior
        try:
            result = self._pending_autonomous.result(timeout=0)
            if result:
                logger.info("Autonomous LLM remark delivered: %s", result[:80])
                self._chat_panel.add_message(MessageRole.ASSISTANT, result)
                if self._audio_bridge is not None:
                    self._audio_bridge.play_voice(result)
            elif behavior is not None:
                logger.warning("Autonomous LLM returned None, using canned fallback")
                self._apply_behavior(behavior)
        except Exception as exc:
            logger.warning("Autonomous remark failed: %s", exc)
            if behavior is not None:
                self._apply_behavior(behavior)
        finally:
            self._pending_autonomous = None
            self._pending_autonomous_behavior = None
            self._scheduler.release("chat")

    _INTERACTION_FALLBACKS: dict[str, str] = {
        "feed": "*munches*",
        "tap_glass": "*startles*",
        "clean": "*looks around at the clean tank*",
        "aerate": "*watches the bubbles*",
        "temp_up": "*stretches in the warmth*",
        "temp_down": "*shivers*",
        "drain": "*blinks at the water level*",
    }

    def _request_interaction_reaction(self, action_key: str) -> None:
        """Submit an LLM reaction to a player interaction.

        Same gating as ``_request_autonomous_remark``. When the LLM is busy,
        shows a canned fallback emote instead of silent skip.
        """
        if self._pending_response is not None or self._pending_autonomous is not None:
            fallback = self._INTERACTION_FALLBACKS.get(action_key)
            if fallback:
                self._chat_panel.add_message(MessageRole.ASSISTANT, fallback)
            return

        manager = self.window.manager
        if manager is None or not manager.is_initialized:
            return

        situation = _INTERACTION_SITUATIONS.get(action_key)
        if situation is None:
            return

        logger.info("Requesting interaction reaction: %s", action_key)
        self._scheduler.acquire("chat")

        async def _generate() -> str | None:
            return await manager.generate_autonomous_remark(situation)

        try:
            self._pending_autonomous = self.window.submit_async(_generate())
            self._pending_autonomous_behavior = None  # No fallback for interactions
        except RuntimeError:
            logger.warning("Async bridge dead — skipping interaction reaction")
            self._scheduler.release("chat")

    def _check_events(self, time_context: dict) -> None:
        """Check for game events and apply them."""
        fired = self._event_system.check_events(
            creature_state=self._creature_state,
            tank=self._tank,
            time_context=time_context,
        )
        for event in fired:
            self._event_system.apply_effects(
                event, self._creature_state, self._tank
            )
            self._add_notification(event.message)
            if event.effects.trigger_dialogue:
                self._chat_panel.add_message(MessageRole.SYSTEM, event.message)

    def _check_evolution(self) -> None:
        """Check if creature is ready to evolve."""
        new_stage = self._evolution_engine.check_evolution(self._creature_state)
        if new_stage is not None:
            self._start_evolution(new_stage)

    def _start_evolution(self, new_stage: CreatureStage) -> None:
        """Begin evolution celebration sequence."""
        # Cancel any pending autonomous remark — evolution takes priority
        if self._pending_autonomous is not None:
            self._pending_autonomous.cancel()
            self._pending_autonomous = None
            self._pending_autonomous_behavior = None
            self._scheduler.release("chat")

        old_stage = self._creature_state.stage
        self._evolution_engine.evolve(self._creature_state, new_stage)
        self._evolution_active = True
        self._evolution_timer = 0.0

        msg = f"Evolution! {old_stage.value} -> {new_stage.value}!"
        self._chat_panel.add_message(MessageRole.SYSTEM, msg)
        self._add_notification(msg)

        if self._audio_bridge is not None:
            self._audio_bridge.play_sfx("evolution_chime")

        logger.info("Evolution: %s -> %s", old_stage.value, new_stage.value)

    def _update_evolution_celebration(self, dt: float) -> None:
        """Update the evolution celebration animation."""
        self._evolution_timer += dt
        if self._evolution_timer >= self._evolution_duration:
            self._evolution_active = False
            self._creature_renderer.set_stage(self._creature_state.stage)
            self._creature_renderer.set_animation(AnimationState.IDLE)

    def _handle_death(self, cause: DeathCause) -> None:
        """Handle creature death — game over screen."""
        self.game_over = True
        self._death_cause = cause
        new_state, record = self._death_engine.on_death(cause, self._creature_state)
        self._death_message = record.message

        self._chat_panel.add_message(MessageRole.SYSTEM, record.message)

        if self._audio_bridge is not None:
            self._audio_bridge.stop_ambient()

        logger.info("Creature died: %s — %s", cause.value, record.message)

    def _on_action_bar(self, action_key: str) -> None:
        """Handle action bar button clicks.

        Args:
            action_key: The action key string (feed, temp_up, temp_down, etc.).
        """
        if self.game_over:
            return

        im = self._interaction_manager
        creature = self._creature_state
        tank = self._tank
        action_succeeded = False

        if action_key == "feed":
            available = im.feeding_engine.get_available_foods(creature.stage)
            if not available:
                self._add_notification("No food available for this stage.")
            elif len(available) == 1:
                self._feed_creature(available[0])
                action_succeeded = True
            else:
                self._show_food_menu(available)
                return  # Menu shown — no LLM reaction yet

        elif action_key == "aerate":
            if tank.environment_type == EnvironmentType.TERRARIUM:
                care_result = im.care_engine.sprinkle(tank, creature)
            else:
                care_result = im.care_engine.aerate_tank(tank)
            self._add_notification(care_result.message)
            if care_result.success:
                action_succeeded = True
                if self._audio_bridge is not None:
                    self._audio_bridge.play_sfx("bubbles")

        elif action_key == "temp_up":
            care_result = im.care_engine.adjust_temperature(tank, 1.0, creature)
            self._add_notification(care_result.message)
            action_succeeded = True

        elif action_key == "temp_down":
            care_result = im.care_engine.adjust_temperature(tank, -1.0, creature)
            self._add_notification(care_result.message)
            action_succeeded = True

        elif action_key == "clean":
            care_result = im.care_engine.clean_tank(tank)
            self._add_notification(care_result.message)
            action_succeeded = True

        elif action_key == "drain":
            if tank.water_level > 0.0:
                care_result = im.care_engine.drain_tank(tank, creature)
            else:
                care_result = im.care_engine.fill_tank(tank, creature)
            self._add_notification(care_result.message)
            action_succeeded = True

        elif action_key == "tap_glass":
            creature.interaction_count += 1
            self._interaction_count_delta += 1
            self._add_notification("You tap the glass...")
            if self._audio_bridge is not None:
                self._audio_bridge.play_sfx("glass_tap")
            action_succeeded = True

        # Request LLM reaction to the interaction
        if action_succeeded:
            self._request_interaction_reaction(action_key)

    def _show_food_menu(self, foods: list[FoodType]) -> None:
        """Show a popup menu of food choices next to the Feed button."""
        feed_btn = next((b for b in self._action_bar.buttons if b.key == "feed"), None)
        if feed_btn is None:
            return

        menu_x = feed_btn.x - _FOOD_MENU_WIDTH - 4
        menu_y = feed_btn.y

        items: list[tuple[FoodType, pygame.Rect]] = []
        for i, food in enumerate(foods):
            rect = pygame.Rect(
                menu_x, menu_y + i * _FOOD_MENU_ITEM_H,
                _FOOD_MENU_WIDTH, _FOOD_MENU_ITEM_H,
            )
            items.append((food, rect))

        self._food_menu_items = items
        self._food_menu_visible = True
        self._food_menu_hovered = -1

    def _close_food_menu(self) -> None:
        """Close the food selection submenu."""
        self._food_menu_visible = False
        self._food_menu_items = []
        self._food_menu_hovered = -1

    def _feed_creature(self, food_type: FoodType) -> None:
        """Feed the creature with the given food type and trigger reactions."""
        im = self._interaction_manager
        creature = self._creature_state
        result = im.feeding_engine.feed(creature, food_type)
        self._interaction_count_delta += 1
        if result.success:
            self._add_notification(result.message)
            self._creature_renderer.set_animation(AnimationState.EATING)
            if self._audio_bridge is not None:
                self._audio_bridge.play_sfx("feeding_splash")
            self._request_interaction_reaction("feed")
        else:
            self._add_notification(result.message)

    def _on_stt_result(self, text: str) -> None:
        """Handle transcribed speech — queue for debounced, non-cancelling submission.

        Unlike typed input, STT results do NOT cancel in-flight LLM calls.
        This prevents rapid speech recognition from killing every response
        before it can complete (~8-33s for the 30B model).
        """
        if not text or not text.strip():
            return
        logger.info("STT result: %s", text)
        self._stt_queued_text = text
        self._stt_queued_time = time.monotonic()

    def _on_chat_submit(self, text: str) -> None:
        """Handle typed chat input — cancels in-flight LLM calls.

        Typed input is intentional and should always win.  STT input uses
        a separate path (``_on_stt_result`` + ``_check_stt_queue``) that
        does NOT cancel in-flight calls.
        """
        if self.game_over or not text.strip():
            return

        # Typed input takes priority — clear any queued STT and TTS
        self._stt_queued_text = None
        if self._audio_bridge is not None:
            self._audio_bridge.cancel_pending_voice()

        # User chat always wins — cancel any in-flight autonomous remark
        if self._pending_autonomous is not None:
            self._pending_autonomous.cancel()
            self._pending_autonomous = None
            self._pending_autonomous_behavior = None

        # Cancel previous user chat if still in flight (prevents orphan buildup)
        if self._pending_response is not None:
            self._pending_response.cancel()
            self._chat_panel.finish_streaming()
            logger.debug("Cancelled previous pending chat for new input")

        self._submit_chat(text)

    def _submit_chat(self, text: str) -> None:
        """Submit text to ConversationManager via streaming."""
        self._creature_renderer.set_animation(AnimationState.TALKING)
        self._interaction_count_delta += 1

        manager = self.window.manager
        if manager is not None and manager.is_initialized:
            self._chat_panel.start_streaming()
            self._scheduler.acquire("chat")

            # Clear any leftover state from a previous cancelled stream
            self._tts_sentence_buffer = ""
            while not self._stream_queue.empty():
                try:
                    self._stream_queue.get_nowait()
                except queue.Empty:
                    break

            async def _process() -> str:
                tokens: list[str] = []
                async for token in manager.process_input_stream(text):
                    tokens.append(token)
                    self._stream_queue.put(token)
                self._stream_queue.put(None)  # sentinel: stream finished
                return "".join(tokens)

            try:
                self._pending_response = self.window.submit_async(_process())
                self._pending_response_time = time.monotonic()
            except RuntimeError:
                logger.warning("Async bridge dead — cannot submit chat")
                self._chat_panel.finish_streaming()
                self._chat_panel.add_message(
                    MessageRole.ASSISTANT,
                    "*yawns* Brain not ready yet... try again.",
                )
                self._creature_renderer.set_animation(AnimationState.IDLE)
                self._scheduler.release("chat")
        else:
            self._chat_panel.add_message(
                MessageRole.ASSISTANT,
                "*yawns* Brain not ready yet... try again.",
            )
            self._creature_renderer.set_animation(AnimationState.IDLE)

    def _check_stt_queue(self) -> None:
        """Submit queued STT text after debounce, without cancelling in-flight calls.

        Called each frame.  Waits for:
        1. Debounce period to elapse (more speech may be coming).
        2. No pending LLM response in flight (avoids killing slow responses).
        """
        if self._stt_queued_text is None:
            return

        # Wait for debounce — more speech fragments may arrive
        if time.monotonic() - self._stt_queued_time < _STT_DEBOUNCE_SECONDS:
            return

        # Wait for LLM to be idle — don't cancel in-flight responses
        if self._pending_response is not None:
            return

        text = self._stt_queued_text
        self._stt_queued_text = None

        # Cancel autonomous remark if one is pending (user speech > autonomous)
        if self._pending_autonomous is not None:
            self._pending_autonomous.cancel()
            self._pending_autonomous = None
            self._pending_autonomous_behavior = None

        logger.info("STT submitted (debounced): %s", text)
        self._chat_panel.add_message(MessageRole.USER, text)
        self._submit_chat(text)

    def _check_pending_response(self) -> None:
        """Check if a pending async conversation response is ready.

        Drains the stream queue each frame so tokens appear in real-time,
        fires TTS per sentence boundary, then finalizes once the async
        future completes.
        """
        if self._pending_response is None:
            return

        # Drain streaming tokens into the chat panel (real-time display)
        while not self._stream_queue.empty():
            try:
                token = self._stream_queue.get_nowait()
                if token is not None:
                    self._chat_panel.append_stream(token)
                    self._tts_sentence_buffer += token
            except queue.Empty:
                break

        # Incremental TTS: speak complete sentences as they arrive
        if self._audio_bridge is not None:
            while True:
                match = _SENTENCE_BOUNDARY.search(self._tts_sentence_buffer)
                if not match:
                    break
                split_pos = match.start() + 1  # include the punctuation
                sentence = self._tts_sentence_buffer[:split_pos].strip()
                self._tts_sentence_buffer = self._tts_sentence_buffer[split_pos:]
                if sentence:
                    self._audio_bridge.play_voice(sentence)

        if not self._pending_response.done():
            # Force-cancel if stuck beyond timeout
            if time.monotonic() - self._pending_response_time > _PENDING_TIMEOUT:
                logger.warning("Pending chat response timed out after %.0fs", _PENDING_TIMEOUT)
                self._pending_response.cancel()
                self._chat_panel.finish_streaming()
                self._chat_panel.add_message(
                    MessageRole.ASSISTANT, "*yawns* ...lost my train of thought."
                )
                self._tts_sentence_buffer = ""
                self._pending_response = None
                self._creature_renderer.set_animation(AnimationState.IDLE)
                self._scheduler.release("chat")
            return

        # Cancelled by a newer chat submission — silently discard
        if self._pending_response.cancelled():
            self._tts_sentence_buffer = ""
            self._pending_response = None
            self._creature_renderer.set_animation(AnimationState.IDLE)
            self._scheduler.release("chat")
            return

        try:
            result = self._pending_response.result(timeout=0)
            # finish_streaming promotes streamed text to a permanent message;
            # if no tokens were streamed (e.g. non-streaming fallback), add
            # the result directly.
            had_stream = bool(self._chat_panel._stream_text)
            self._chat_panel.finish_streaming()
            if not had_stream and result:
                self._chat_panel.add_message(MessageRole.ASSISTANT, result)
            # Speak any remaining text that didn't end with sentence punctuation
            remaining = self._tts_sentence_buffer.strip()
            if remaining and self._audio_bridge is not None:
                self._audio_bridge.play_voice(remaining)
            elif not remaining and result and self._audio_bridge is not None:
                # Non-streaming fallback: no incremental TTS happened
                if not had_stream:
                    self._audio_bridge.play_voice(result)
        except Exception as exc:
            self._chat_panel.finish_streaming()
            self._chat_panel.add_message(
                MessageRole.ASSISTANT, f"*glitches* {exc}"
            )
            logger.error("Conversation error: %s", exc, exc_info=True)
        finally:
            self._tts_sentence_buffer = ""
            self._pending_response = None
            self._creature_renderer.set_animation(AnimationState.IDLE)
            self._scheduler.release("chat")

    def _on_mouse_click(self, event: pygame.event.Event) -> None:
        """Handle mouse click events."""
        mx, my = event.pos

        # Settings overlay gets first priority
        if self._game_state == GameState.SETTINGS and self._settings_panel is not None:
            try:
                self._settings_panel.handle_click(mx, my)
            except Exception as exc:
                logger.error("Settings click error: %s", exc, exc_info=True)
            return

        # Lineage overlay gets priority
        if self._game_state == GameState.LINEAGE and self._lineage_panel is not None:
            try:
                self._lineage_panel.handle_click(mx, my)
            except Exception as exc:
                logger.error("Lineage click error: %s", exc)
            return

        # Check HUD settings button
        if self._hud.settings_rect is not None:
            if self._hud.settings_rect.collidepoint(mx, my):
                self._toggle_settings()
                return

        # Check HUD lineage button
        if self._hud.lineage_rect is not None:
            if self._hud.lineage_rect.collidepoint(mx, my):
                self._toggle_lineage()
                return

        # Check HUD mic button
        if self._hud.mic_rect is not None:
            if self._hud.mic_rect.collidepoint(mx, my):
                self._toggle_mic()
                return

        if self.game_over:
            # Any click during game over restarts
            self._restart_game()
            return

        # Food submenu clicks — check before action bar
        if self._food_menu_visible:
            for i, (food, rect) in enumerate(self._food_menu_items):
                if rect.collidepoint(mx, my):
                    self._close_food_menu()
                    self._feed_creature(food)
                    return
            # Click outside menu closes it
            self._close_food_menu()
            return

        # Action bar clicks
        if self._action_bar.handle_click(mx, my):
            return

        # Chat panel Send button clicks
        if self._chat_panel.visible and self._chat_panel.handle_click(mx, my):
            return

        result = self._interaction_manager.handle_click(
            mx, my, self._creature_state, self._tank
        )
        if result is not None:
            self._interaction_count_delta += 1
            if result.message:
                self._add_notification(result.message)
            if result.interaction_type == InteractionType.TAP_GLASS:
                if self._audio_bridge is not None:
                    self._audio_bridge.play_sfx("glass_tap")
            elif result.interaction_type == InteractionType.FEED:
                if result.feeding_result and result.feeding_result.success:
                    if self._audio_bridge is not None:
                        self._audio_bridge.play_sfx("feeding_splash")
                    self._creature_renderer.set_animation(AnimationState.EATING)

    def _on_mouse_move(self, event: pygame.event.Event) -> None:
        """Handle mouse motion events."""
        mx, my = event.pos

        if self._game_state == GameState.SETTINGS and self._settings_panel is not None:
            try:
                self._settings_panel.handle_mouse_move(mx, my)
            except Exception as exc:
                logger.error("Settings mouse move error: %s", exc, exc_info=True)
            return

        if self._game_state == GameState.LINEAGE and self._lineage_panel is not None:
            try:
                self._lineage_panel.handle_mouse_move(mx, my)
            except Exception as exc:
                logger.error("Lineage mouse move error: %s", exc)
            return

        # Food menu hover
        if self._food_menu_visible:
            self._food_menu_hovered = -1
            for i, (_, rect) in enumerate(self._food_menu_items):
                if rect.collidepoint(mx, my):
                    self._food_menu_hovered = i
                    break

        self._creature_renderer.set_mouse_position(float(mx), float(my))
        self._interaction_manager.handle_mouse_move(mx, my)
        self._action_bar.handle_mouse_move(mx, my)
        self._chat_panel.handle_mouse_move(mx, my)

    def _on_mouse_up(self, event: pygame.event.Event) -> None:
        """Handle mouse button release (slider drag stop)."""
        if self._game_state == GameState.SETTINGS and self._settings_panel is not None:
            try:
                self._settings_panel.handle_mouse_up()
            except Exception as exc:
                logger.error("Settings mouse up error: %s", exc, exc_info=True)

    def _on_mouse_scroll(self, event: pygame.event.Event) -> None:
        """Handle mouse wheel for dropdown scrolling."""
        if self._game_state == GameState.SETTINGS and self._settings_panel is not None:
            try:
                self._settings_panel.handle_scroll(event.y)
            except Exception as exc:
                logger.error("Settings scroll error: %s", exc, exc_info=True)

    def _on_key_down(self, event: pygame.event.Event) -> None:
        """Handle key press events."""
        # ESC: close overlays if open, quit if in gameplay
        if event.key == pygame.K_ESCAPE:
            if self._food_menu_visible:
                self._close_food_menu()
            elif self._game_state == GameState.SETTINGS:
                self._toggle_settings()
            elif self._game_state == GameState.LINEAGE:
                self._toggle_lineage()
            else:
                self.window.running = False
            return

        # F1: toggle settings
        if event.key == pygame.K_F1:
            self._toggle_settings()
            return

        # F2: toggle lineage
        if event.key == pygame.K_F2:
            self._toggle_lineage()
            return

        # Death screen: Enter/Space restarts
        if self.game_over:
            if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE):
                self._restart_game()
            return

        # When overlays are open, forward keys to the overlay then consume
        if self._game_state == GameState.LINEAGE and self._lineage_panel is not None:
            self._lineage_panel.handle_event(event)
            return
        if self._game_state == GameState.SETTINGS:
            return

        # Chat panel gets first chance
        if self._chat_panel.handle_event(event):
            return

        # Audio controls
        if self._audio_bridge is not None:
            if self._audio_bridge.handle_key_event(event.key):
                return

        # Vision trigger
        if event.key == pygame.K_v:
            if self._vision_bridge is not None:
                self._vision_bridge.trigger_observation(self.window.screen)
                self._add_notification("Looking...")
            return

        # HUD toggle
        if event.key == pygame.K_h:
            self._hud.toggle_mode()

    def _toggle_settings(self) -> None:
        """Toggle the settings overlay open/closed."""
        if self._settings_panel is None:
            return

        if self._game_state == GameState.SETTINGS:
            logger.info("Game state: SETTINGS -> PLAYING")
            self._game_state = GameState.PLAYING
            self._settings_panel.close()
        else:
            # Close lineage overlay first if open
            if self._game_state == GameState.LINEAGE and self._lineage_panel is not None:
                self._lineage_panel.close()
            logger.info("Game state: %s -> SETTINGS", self._game_state.value)
            self._game_state = GameState.SETTINGS
            self._settings_panel.open()
            # Refresh device lists on a background thread to avoid blocking
            # the main Pygame loop (pyttsx3.init + cv2.VideoCapture can block)
            threading.Thread(
                target=self._settings_panel.refresh_device_lists,
                daemon=True,
            ).start()
            # Async-load Ollama model list when opening settings
            self._load_model_list_async()

    def _on_settings_close(self) -> None:
        """Callback when settings panel X button is clicked."""
        logger.info("Game state: SETTINGS -> PLAYING (panel closed)")
        self._game_state = GameState.PLAYING

    def _on_lineage_close(self) -> None:
        """Callback when lineage panel X button is clicked."""
        logger.info("Game state: LINEAGE -> PLAYING (panel closed)")
        self._game_state = GameState.PLAYING

    def _load_model_list_async(self) -> None:
        """Fetch available Ollama models in the background."""
        if self._settings_panel is None:
            return

        async def _fetch_models() -> list[str]:
            try:
                import ollama
                client = ollama.AsyncClient()
                response = await client.list()
                return [m.model for m in response.models]
            except Exception as exc:
                logger.warning("Failed to fetch Ollama models: %s", exc)
                return [self._config.llm.model]

        def _on_done(future: Any) -> None:
            try:
                models = future.result(timeout=0)
                # Queue for main thread — pygame isn't thread-safe
                self._pending_model_list = models
            except Exception as exc:
                logger.error("Failed to load Ollama model list: %s", exc)
                self._pending_model_list = [self._config.llm.model + " (offline)"]

        try:
            future = self.window.submit_async(_fetch_models())
            future.add_done_callback(_on_done)
        except RuntimeError:
            logger.warning("Async bridge dead — cannot fetch model list")
            self._pending_model_list = [self._config.llm.model + " (offline)"]

    def _on_personality_change(self, traits: dict[str, float]) -> None:
        """Callback when personality settings change."""
        try:
            logger.info("Personality traits updated: %s", traits)
            manager = self.window.manager
            if manager is not None:
                manager.update_personality_traits(traits)
            save_user_settings(self._config)
        except Exception as exc:
            logger.error("Personality change error: %s", exc)

    def _on_llm_apply(self, model: str, temperature: float) -> None:
        """Callback when LLM settings are applied."""
        try:
            logger.info("LLM settings applied: model=%s, temp=%.1f", model, temperature)
            save_user_settings(self._config)
            # Propagate to the live LLM provider
            manager = self.window.manager
            if manager is not None:
                manager.update_llm_settings(model, temperature)
        except Exception as exc:
            logger.error("LLM apply error: %s", exc)

    def _ensure_audio_manager(self) -> None:
        """Attempt to create AudioManager if it's None (lazy retry)."""
        if self._audio_manager is not None:
            return
        try:
            self._audio_manager = AudioManager(config=self._config.audio)
            logger.info("AudioManager created (lazy retry)")
            if self._audio_bridge is not None:
                self._audio_bridge._audio_manager = self._audio_manager
        except Exception as exc:
            logger.warning("AudioManager lazy retry failed: %s", exc)

    def _on_audio_change(self, key: str, value: Any) -> None:
        """Callback when audio settings change."""
        try:
            self._ensure_audio_manager()
            logger.info("Audio setting changed: %s = %s", key, value)
            if self._audio_bridge is not None:
                # Apply audio changes at runtime where possible
                if key == "tts_enabled":
                    self._audio_bridge._config.tts_enabled = value
                    if self._audio_manager is not None:
                        self._audio_manager.tts_enabled = value
                elif key == "sfx_enabled":
                    self._audio_bridge._config.sfx_enabled = value
                    if self._audio_manager is not None:
                        self._audio_manager.sfx_enabled = value
                elif key == "tts_volume":
                    self._audio_bridge._config.tts_volume = value
                    self._audio_bridge.set_volume(AudioChannel.VOICE, value)
                elif key == "sfx_volume":
                    self._audio_bridge._config.sfx_volume = value
                    self._audio_bridge.set_volume(AudioChannel.SFX, value)
                elif key == "ambient_volume":
                    self._audio_bridge._config.ambient_volume = value
                    self._audio_bridge.set_volume(AudioChannel.AMBIENT, value)
                elif key == "stt_enabled":
                    self._audio_bridge._config.stt_enabled = value
                    if self._audio_manager is not None:
                        self._audio_manager.stt_enabled = value
                    if value:
                        # Check if upgrade actually succeeded
                        if (
                            self._audio_manager is not None
                            and not self._audio_manager.stt_enabled
                        ):
                            self._add_notification("STT unavailable — check audio deps")
                        elif self._audio_bridge._audio_manager is not None:
                            self._audio_bridge.toggle_microphone()
                            self._add_notification("Speech-to-text enabled")
                        else:
                            self._add_notification("STT not yet available")
                    else:
                        if self._audio_bridge.mic_active:
                            self._audio_bridge.toggle_microphone()
                        self._add_notification("Speech-to-text disabled")
                elif key == "tts_voice":
                    voice_name = str(value)
                    if self._audio_manager is not None:
                        self._audio_manager.update_tts_voice(voice_name)
                    self._add_notification(f"Voice: {voice_name}")
                elif key == "audio_output_device":
                    if self._audio_bridge is not None:
                        self._audio_bridge._reinit_mixer()
                    self._add_notification(f"Output: {value}")
                elif key == "audio_input_device":
                    device_name = str(value)
                    if self._audio_manager is not None:
                        self._audio_manager.set_input_device(device_name)
                    self._add_notification(f"Mic: {device_name}")
                elif key == "tts_provider":
                    if self._audio_manager is not None:
                        name = self._audio_manager.swap_tts_provider(self._config.audio)
                        self._add_notification(f"TTS engine: {name}")
                    else:
                        self._add_notification(f"TTS engine: {value}")
                elif key == "stt_provider":
                    if self._audio_manager is not None:
                        name = self._audio_manager.swap_stt_provider(self._config.audio)
                        self._add_notification(f"STT engine: {name}")
                    else:
                        self._add_notification(f"STT engine: {value}")
                elif key == "aec_enabled":
                    if self._audio_manager is not None:
                        self._audio_manager.toggle_aec(
                            bool(value), loop=self.window._loop
                        )
                    mode = "full-duplex" if value else "half-duplex"
                    self._add_notification(f"Audio: {mode}")
                elif key == "barge_in_enabled":
                    self._add_notification(
                        "Barge-in " + ("enabled" if value else "disabled")
                    )
            save_user_settings(self._config)
        except Exception as exc:
            logger.error("Audio change error: %s", exc)

    def _restart_game(self) -> None:
        """Restart with a fresh creature (new egg)."""
        self.game_over = False
        self._death_cause = None
        self._death_message = ""
        self._creature_state = CreatureState()
        self._tank = TankEnvironment.from_config(self._config.environment)
        self._needs = CreatureNeeds()
        self._creature_renderer.set_stage(CreatureStage.MUSHROOMER)
        self._creature_renderer.set_animation(AnimationState.IDLE)
        self._mood_engine.set_mood(CreatureMood.NEUTRAL)
        self._behavior_engine.reset_cooldowns()
        self._event_system.reset()
        self._chat_panel.clear_messages()
        self._chat_panel.add_message(
            MessageRole.SYSTEM, "A new egg has appeared in the tank..."
        )
        self._notifications.clear()
        # Cancel any pending autonomous remark
        if self._pending_autonomous is not None:
            self._pending_autonomous.cancel()
            self._pending_autonomous = None
            self._pending_autonomous_behavior = None
        logger.info("Game restarted with new creature")

    def _add_notification(self, text: str) -> None:
        """Add a temporary notification to display."""
        self._notifications.append((text, self._notification_duration))

    # ── Rendering ─────────────────────────────────────────────────────

    def _render(self, surface: pygame.Surface) -> None:
        """Render all game subsystems in order.

        Args:
            surface: The display surface.
        """
        self._ensure_overlay_fonts()

        if self.game_over:
            self._render_game_over(surface)
            return

        # Each sub-renderer is wrapped individually so one crash
        # doesn't kill the entire frame.
        try:
            self._tank_renderer.render(surface, self._tank)
        except Exception as exc:
            logger.error("Tank render error: %s", exc, exc_info=True)

        try:
            self._creature_renderer.render(surface)
        except Exception as exc:
            logger.error("Creature render error: %s", exc, exc_info=True)

        try:
            self._interaction_manager.render(surface)
        except Exception as exc:
            logger.error("Interaction render error: %s", exc, exc_info=True)

        try:
            self._action_bar.render(surface)
        except Exception as exc:
            logger.error("Action bar render error: %s", exc, exc_info=True)

        if self._food_menu_visible:
            try:
                self._render_food_menu(surface)
            except Exception as exc:
                logger.error("Food menu render error: %s", exc, exc_info=True)

        try:
            self._hud.render(surface, self._creature_state, self._tank)
        except Exception as exc:
            logger.error("HUD render error: %s", exc, exc_info=True)

        try:
            self._chat_panel.render(surface)
        except Exception as exc:
            logger.error("Chat panel render error: %s", exc, exc_info=True)

        try:
            self._render_notifications(surface)
        except Exception as exc:
            logger.error("Notification render error: %s", exc, exc_info=True)

        if self._evolution_active:
            try:
                self._render_evolution_overlay(surface)
            except Exception as exc:
                logger.error("Evolution overlay render error: %s", exc, exc_info=True)

        if self._game_state == GameState.SETTINGS and self._settings_panel is not None:
            try:
                self._settings_panel.render(surface)
            except Exception as exc:
                logger.error("Settings render error: %s", exc, exc_info=True)

        if self._game_state == GameState.LINEAGE and self._lineage_panel is not None:
            try:
                self._lineage_panel.render(surface)
            except Exception as exc:
                logger.error("Lineage render error: %s", exc, exc_info=True)

    def _render_game_over(self, surface: pygame.Surface) -> None:
        """Render the game-over screen."""
        if self._overlay_font is None or self._overlay_title_font is None:
            return

        w = surface.get_width()
        h = surface.get_height()

        # Dark overlay
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill(_GAMEOVER_BG)
        surface.blit(overlay, (0, 0))

        # Title
        title = self._overlay_title_font.render("GAME OVER", True, _GAMEOVER_TEXT)
        tx = (w - title.get_width()) // 2
        surface.blit(title, (tx, h // 3))

        # Death cause
        if self._death_cause is not None:
            cause_text = f"Cause: {self._death_cause.value}"
            cause_surf = self._overlay_font.render(cause_text, True, _GAMEOVER_TEXT)
            surface.blit(cause_surf, ((w - cause_surf.get_width()) // 2, h // 3 + 50))

        # Death message
        if self._death_message:
            msg_surf = self._overlay_font.render(
                self._death_message, True, _GAMEOVER_HINT
            )
            surface.blit(msg_surf, ((w - msg_surf.get_width()) // 2, h // 3 + 80))

        # Restart hint
        hint = self._overlay_font.render(
            "Click anywhere to hatch a new egg...", True, _GAMEOVER_HINT
        )
        surface.blit(hint, ((w - hint.get_width()) // 2, h * 2 // 3))

    def _render_evolution_overlay(self, surface: pygame.Surface) -> None:
        """Render the evolution celebration effect."""
        if self._overlay_title_font is None:
            return

        w = surface.get_width()
        h = surface.get_height()

        # Pulsing golden glow
        progress = self._evolution_timer / self._evolution_duration
        alpha = int(80 * (1.0 - progress))
        if alpha > 0:
            overlay = pygame.Surface((w, h), pygame.SRCALPHA)
            overlay.fill((*_EVOLUTION_GLOW, alpha))
            surface.blit(overlay, (0, 0))

        # Stage name
        stage_name = self._creature_state.stage.value.upper()
        text = f"EVOLVED: {stage_name}"
        text_surf = self._overlay_title_font.render(text, True, _EVOLUTION_GLOW)
        tx = (w - text_surf.get_width()) // 2
        ty = h // 4
        surface.blit(text_surf, (tx, ty))

    def _render_food_menu(self, surface: pygame.Surface) -> None:
        """Render the food selection popup next to the Feed button."""
        if not self._food_menu_items or self._overlay_font is None:
            return

        for i, (food, rect) in enumerate(self._food_menu_items):
            bg = _FOOD_MENU_HOVER if i == self._food_menu_hovered else _FOOD_MENU_BG
            bg_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            bg_surf.fill(bg)
            surface.blit(bg_surf, (rect.x, rect.y))
            pygame.draw.rect(surface, _FOOD_MENU_BORDER, rect, 1)
            label = food.value.capitalize()
            text_surf = self._overlay_font.render(label, True, _FOOD_MENU_TEXT)
            tx = rect.x + 8
            ty = rect.y + (rect.height - text_surf.get_height()) // 2
            surface.blit(text_surf, (tx, ty))

    def _render_notifications(self, surface: pygame.Surface) -> None:
        """Render temporary notification toasts."""
        if self._overlay_font is None or not self._notifications:
            return

        w = surface.get_width()
        y = 50  # Below top bar
        for text, remaining in self._notifications:
            alpha = min(1.0, remaining / 0.5)  # Fade out in last 0.5s
            bg_alpha = int(200 * alpha)

            text_surf = self._overlay_font.render(text, True, _NOTIFICATION_TEXT)
            tw = text_surf.get_width()
            bg_w = tw + 20
            bg_x = (w - bg_w) // 2

            bg = pygame.Surface((bg_w, 28), pygame.SRCALPHA)
            bg.fill((*_NOTIFICATION_BG[:3], bg_alpha))
            surface.blit(bg, (bg_x, y))
            surface.blit(text_surf, (bg_x + 10, y + 4))
            y += 32

    def _on_vision_change(self, key: str, value: Any) -> None:
        """Callback when vision settings change."""
        try:
            logger.info("Vision setting changed: %s = %s", key, value)
            if key == "source":
                source = str(value)
                if source == "off":
                    self._config.vision.enabled = False
                    if self._vision_bridge is not None:
                        self._vision_bridge.set_source("off")
                else:
                    self._config.vision.enabled = True
                    # Create bridge on-demand if it doesn't exist
                    if self._vision_bridge is None:
                        self._vision_bridge = VisionBridge(
                            vision_config=self._config.vision,
                            async_loop=self.window._loop,
                            scheduler=self._scheduler,
                        )
                    self._vision_bridge.set_source(source)
                save_user_settings(self._config)
            elif key == "capture_interval":
                if self._vision_bridge is not None:
                    self._vision_bridge._config.capture_interval = float(value)
                save_user_settings(self._config)
            elif key == "webcam_index":
                idx = int(value)
                self._config.vision.webcam_index = idx
                if self._vision_bridge is not None:
                    self._vision_bridge.set_webcam_index(idx)
                save_user_settings(self._config)
            elif key == "look_now":
                self._trigger_vision_look()
        except Exception as exc:
            logger.error("Vision change error: %s", exc, exc_info=True)

    def _trigger_vision_look(self) -> None:
        """Trigger an on-demand vision capture and show the result."""
        # Create bridge on-demand if needed
        if self._vision_bridge is None:
            source = self._config.vision.source
            if source == "off":
                self._add_notification("Vision source is off — set a source first.")
                return
            self._vision_bridge = VisionBridge(
                vision_config=self._config.vision,
                async_loop=self.window._loop,
                scheduler=self._scheduler,
            )

        if self._vision_bridge.source == "off":
            self._add_notification("Vision source is off — set a source first.")
            return

        # Snapshot observation count before trigger so we can detect new ones
        prev_count = len(self._vision_bridge.get_recent_observations())
        self._vision_bridge.trigger_observation(self.window.screen)
        self._add_notification("Looking...")
        self._vision_look_prev_count = prev_count
        self._vision_look_start_time = time.monotonic()

    def _check_vision_look_result(self) -> None:
        """Check if a 'Look Now' observation has completed and update the panel."""
        if self._vision_look_prev_count is None or self._vision_bridge is None:
            return

        # Immediate capture failure (webcam unavailable, no frame, no loop)
        if self._vision_bridge._last_capture_failed:
            self._add_notification("Webcam capture failed")
            self._vision_look_prev_count = None
            self._vision_bridge._last_capture_failed = False
            return

        # Timeout — VLM call took too long
        elapsed = time.monotonic() - self._vision_look_start_time
        if elapsed > _VISION_LOOK_TIMEOUT:
            self._add_notification("Vision timed out")
            self._vision_look_prev_count = None
            return

        # Success — new observation arrived
        observations = self._vision_bridge.get_recent_observations()
        if len(observations) > self._vision_look_prev_count:
            newest = observations[0]
            if self._settings_panel is not None:
                self._settings_panel.set_last_observation(newest)
            self._add_notification(f"Saw: {newest[:60]}")
            self._vision_look_prev_count = None

    def _toggle_lineage(self) -> None:
        """Toggle the lineage overlay open/closed."""
        if self._lineage_panel is None:
            return

        if self._game_state == GameState.LINEAGE:
            try:
                self._lineage_panel.close()
            except Exception as exc:
                logger.error("Lineage panel close failed: %s", exc, exc_info=True)
            logger.info("Game state: LINEAGE -> PLAYING")
            self._game_state = GameState.PLAYING
        else:
            # Close settings overlay first if open
            if self._game_state == GameState.SETTINGS and self._settings_panel is not None:
                self._settings_panel.close()
            try:
                self._lineage_panel.open()
                logger.info("Game state: %s -> LINEAGE", self._game_state.value)
                self._game_state = GameState.LINEAGE
            except Exception as exc:
                logger.error("Lineage panel open failed: %s", exc, exc_info=True)
                self._game_state = GameState.PLAYING
                self._add_notification("Failed to open lineage panel")

    def _toggle_mic(self) -> None:
        """Toggle microphone input on/off via the audio bridge."""
        if self._audio_bridge is not None:
            self._audio_bridge.toggle_microphone()
            self._hud.mic_active = self._audio_bridge.mic_active
            state = "on" if self._hud.mic_active else "off"
            self._add_notification(f"Mic {state}")
        else:
            self._add_notification("Audio not available")

    def _switch_bloodline(self, name: str) -> None:
        """Switch to a different bloodline save directory."""
        from seaman_brain.creature.persistence import StatePersistence

        try:
            logger.info("Switching to bloodline: %s", name)
            save_path = f"{self._config.creature.save_path}/{name}"
            persistence = StatePersistence(save_path)
            new_state = persistence.load()

            # Cancel pending operations
            if self._pending_response is not None:
                self._pending_response.cancel()
                self._pending_response = None
                self._scheduler.release("chat")
            if self._pending_autonomous is not None:
                self._pending_autonomous.cancel()
                self._pending_autonomous = None
                self._pending_autonomous_behavior = None
                self._scheduler.release("chat")

            # Update game state
            self._creature_state = new_state
            self._creature_renderer.set_stage(new_state.stage)
            self._creature_renderer.set_animation(AnimationState.IDLE)
            self._needs = CreatureNeeds()
            self._tank = TankEnvironment.from_config(self._config.environment)
            self._chat_panel.clear_messages()
            self._chat_panel.add_message(
                MessageRole.SYSTEM, f"Loaded bloodline: {name}"
            )

            # Sync ConversationManager persistence path and state
            if self._manager is not None:
                self._manager.switch_bloodline(name, new_state)

            self._add_notification(f"Loaded bloodline: {name}")
            self._toggle_lineage()
        except Exception as exc:
            logger.error("Failed to switch bloodline: %s", exc)
            self._add_notification(f"Failed to load: {exc}")

    def _new_bloodline(self, name: str) -> None:
        """Create a new bloodline with a fresh creature."""
        try:
            logger.info("Creating new bloodline: %s", name)
            self._add_notification(f"Created bloodline: {name}")
            if self._lineage_panel is not None:
                self._lineage_panel.refresh_list()
        except Exception as exc:
            logger.error("Failed to create bloodline: %s", exc)

    def _delete_bloodline(self, name: str) -> None:
        """Delete a bloodline save directory."""
        try:
            logger.info("Deleted bloodline: %s", name)
            self._add_notification(f"Deleted bloodline: {name}")
            if self._lineage_panel is not None:
                self._lineage_panel.refresh_list()
        except Exception as exc:
            logger.error("Failed to delete bloodline: %s", exc)

    def shutdown(self) -> None:
        """Clean shutdown of all subsystems.

        Order matters: subsystem bridges are shut down first (they set
        shutdown guards to prevent new async submissions), then the
        window shuts down the async loop and Pygame.
        """
        # 0. Stop full-duplex audio pipeline if running
        if self._audio_manager is not None:
            try:
                self._audio_manager.stop_pipeline()
            except Exception:
                pass

        # 1. Signal bridges to stop accepting new work
        if self._audio_bridge is not None:
            self._audio_bridge.shutdown()
        if self._vision_bridge is not None:
            self._vision_bridge.shutdown()

        # 2. Now safe to cancel pending tasks and stop the async loop
        self.window.shutdown()
        logger.info("GameEngine shutdown complete")
