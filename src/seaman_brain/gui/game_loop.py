"""Thin GameEngine orchestrator.

Wires together layout, scene management, input handling, rendering,
and all game subsystems. Delegates to focused modules:
- game_systems.py: needs, mood, behavior, events, evolution, death
- response_handler.py: streaming chat, TTS splitting, pending management
- scene_manager.py: PLAYING/SETTINGS/LINEAGE state machine + drawer animation
- input_handler.py: keyboard/mouse event routing
- render_engine.py: gradient cache + particles

Loop order: events → input_handler → game_systems.tick(dt) →
scene_manager.update(dt) → render.
"""
from __future__ import annotations

import logging
import time
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
from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
from seaman_brain.gui.audio_integration import PygameAudioBridge
from seaman_brain.gui.chat_panel import ChatPanel
from seaman_brain.gui.game_systems import (
    _INTERACTION_FALLBACKS,
    _STT_DEBOUNCE_SECONDS,
    GameState,
    GameSystems,
    TickResult,
    _build_interaction_situation,
)
from seaman_brain.gui.hud import HUD
from seaman_brain.gui.input_handler import InputHandler
from seaman_brain.gui.interactions import InteractionManager
from seaman_brain.gui.layout import ScreenLayout
from seaman_brain.gui.lineage_panel import LineagePanel
from seaman_brain.gui.response_handler import ResponseHandler
from seaman_brain.gui.scene_manager import SceneManager
from seaman_brain.gui.settings_panel import SettingsPanel
from seaman_brain.gui.sprites import AnimationState, CreatureRenderer
from seaman_brain.gui.tank_renderer import TankRenderer
from seaman_brain.gui.window import GameWindow
from seaman_brain.llm.scheduler import ModelScheduler
from seaman_brain.needs.care import AERATOR_COOLDOWN_SECONDS, CLEANING_DURATION_SECONDS
from seaman_brain.needs.death import DeathCause, DeathEngine
from seaman_brain.needs.feeding import FoodType
from seaman_brain.needs.system import NeedsEngine
from seaman_brain.personality.traits import TraitProfile
from seaman_brain.types import CreatureStage

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

        # Interaction manager (feeding, care, tap glass)
        self._interaction_manager = InteractionManager(
            gui_config=cfg.gui,
            env_config=cfg.environment,
            needs_config=cfg.needs,
        )

        # Window
        self.window = GameWindow(config=cfg)

        # Game state
        self._creature_state = CreatureState()
        self._tank_env = TankEnvironment.from_config(cfg.environment)
        self._clock = GameClock()

        # Game logic engines (kept for direct use in evolution/death handling)
        self._mood_engine = MoodEngine()
        self._evolution_engine = EvolutionEngine(config=cfg.creature)
        self._death_engine = DeathEngine(
            needs_config=cfg.needs, env_config=cfg.environment,
        )
        self._scheduler = ModelScheduler(enabled=False)

        # Game systems — timer-based subsystem tick
        self._game_systems = GameSystems(
            needs_engine=NeedsEngine(
                config=cfg.needs, env_config=cfg.environment,
            ),
            mood_engine=self._mood_engine,
            behavior_engine=BehaviorEngine(),
            event_system=EventSystem(),
            evolution_engine=self._evolution_engine,
            death_engine=self._death_engine,
            creature_state=self._creature_state,
            clock=self._clock,
            tank=self._tank_env,
        )
        self._game_systems._traits_fn = self._get_traits

        # Audio / vision (lazy)
        self._audio_manager: AudioManager | None = None
        self._audio_bridge: PygameAudioBridge | None = None

        # Response handler — streaming chat, TTS, pending management
        self._response_handler = ResponseHandler(
            chat_panel=self._chat_panel,
            audio_bridge=None,  # wired in initialize()
            scheduler=self._scheduler,
        )

        # STT debounce
        self._stt_queued_text: str | None = None
        self._stt_queued_time: float = 0.0

        # Misc state
        self.game_over = False
        self._death_cause: DeathCause | None = None
        self._death_message = ""

        # Evolution celebration
        self._evolution_active = False
        self._evolution_timer = 0.0
        self._evolution_duration = 3.0

        # Wire input handler
        self._input_handler.on_escape = self._on_escape
        self._input_handler.on_toggle_settings = self._toggle_settings
        self._input_handler.on_key_down = self._on_key_down
        self._input_handler.on_mouse_click = self._on_mouse_click
        self._input_handler.on_mouse_move = self._on_mouse_move
        self._input_handler.on_mouse_up = self._on_mouse_up
        self._input_handler.on_mouse_scroll = self._on_mouse_scroll

        # Wire HUD food dropdown callback
        self._hud.on_feed = self._on_food_selected

        # Set available food types on HUD
        available = self._interaction_manager.feeding_engine.get_available_foods(
            self._creature_state.stage,
        )
        self._hud.set_food_types([f.value for f in available])

    # ── Lifecycle ─────────────────────────────────────────────────────

    def initialize(self) -> None:
        """Initialize window, audio, and register callbacks."""
        self.window.initialize()

        # Ensure Riva containers are running and URIs are resolved
        try:
            from seaman_brain.audio.riva_launcher import ensure_riva_running

            ensure_riva_running(self._config.audio)
        except Exception as exc:
            logger.warning("Riva launcher: %s", exc)

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

        # Wire audio to response handler
        self._response_handler._audio = self._audio_bridge

        # Register input handler for all event types
        for etype in (
            pygame.KEYDOWN,
            pygame.MOUSEBUTTONDOWN,
            pygame.MOUSEMOTION,
            pygame.MOUSEBUTTONUP,
            pygame.MOUSEWHEEL,
        ):
            self.window.register_event_handler(
                etype, self._input_handler.handle_event,
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
            self._audio_bridge.shutdown()

    # ── Main loop ─────────────────────────────────────────────────────

    def _update(self, dt: float) -> None:
        """Per-frame update of all game subsystems."""
        self._scene_manager.update(dt)
        self._hud.update(dt)

        if self._scene_manager.state != GameState.PLAYING:
            return
        if self.game_over:
            return
        if self._evolution_active:
            self._update_evolution_celebration(dt)
            return

        # Tank environment
        try:
            self._tank_env.update(dt)
        except Exception as exc:
            logger.error("Tank update error: %s", exc, exc_info=True)

        # Game systems tick — needs, mood, behavior, events, evolution, death
        result = self._game_systems.tick(dt)
        if result is not None:
            self._handle_tick_result(result)

        self._update_renderers(dt)
        self._update_audio(dt)
        self._sync_hud()
        self._update_responses()

    def _handle_tick_result(self, result: TickResult) -> None:
        """React to game systems tick — trigger UI effects."""
        if result.death_cause is not None:
            self._handle_death(result.death_cause)
            return

        # Sync mood to creature renderer
        self._creature_renderer.set_mood(self._creature_state.mood)

        if result.behavior is not None:
            self._handle_behavior(result.behavior)

        for event in result.fired_events:
            if hasattr(event, "effects") and event.effects.trigger_dialogue:
                self._chat_panel.add_message("system", event.message)

        if result.new_stage is not None:
            self._start_evolution(result.new_stage)

    def _handle_behavior(self, behavior: IdleBehavior) -> None:
        """Decide how to handle an autonomous behavior."""
        if behavior.needs_llm:
            needs = self._game_systems.needs
            needs_critical = (
                needs.hunger >= 0.7
                or needs.health <= 0.3
                or needs.comfort <= 0.2
            )
            if needs_critical:
                self._request_autonomous_remark(behavior)
        else:
            self._apply_behavior(behavior)

    def _update_renderers(self, dt: float) -> None:
        """Update animations and sync mood/stage to renderers."""
        try:
            self._tank_renderer.update(dt)
            self._creature_renderer.update(dt)
        except Exception as exc:
            logger.error("Animation error: %s", exc, exc_info=True)

        try:
            self._tank_renderer.set_mood(self._creature_state.mood)
            if self._creature_renderer.stage != self._creature_state.stage:
                self._creature_renderer.set_stage(self._creature_state.stage)
        except Exception as exc:
            logger.error("Renderer sync error: %s", exc, exc_info=True)

    def _update_audio(self, dt: float) -> None:
        """Update audio bridge and HUD indicators."""
        try:
            if self._audio_bridge is not None:
                self._audio_bridge.update(dt)
                self._hud.mic_active = self._audio_bridge.mic_active
            self._hud.tts_active = (
                self._audio_manager is not None
                and self._audio_manager.tts_enabled
            )
        except Exception as exc:
            logger.error("Audio bridge error: %s", exc, exc_info=True)

    def _sync_hud(self) -> None:
        """Push current state to HUD."""
        try:
            needs = self._game_systems.needs
            self._hud.update_needs(
                hunger=needs.hunger,
                health=needs.health,
                comfort=needs.comfort,
                trust=self._creature_state.trust_level,
            )
            self._hud.update_creature_info(
                stage=self._creature_state.stage.value.capitalize(),
                mood=self._creature_state.mood,
                name=self._creature_state.name
                if hasattr(self._creature_state, "name")
                else "Seaman",
            )
        except Exception as exc:
            logger.error("HUD sync error: %s", exc, exc_info=True)

    def _update_responses(self) -> None:
        """Check STT queue and pending responses."""
        try:
            self._check_stt_queue()

            was_busy = self._response_handler.is_busy
            self._response_handler.check_pending()
            if was_busy and not self._response_handler.is_busy:
                self._creature_renderer.set_animation(AnimationState.IDLE)

            text, behavior = self._response_handler.check_pending_autonomous()
            if text:
                self._chat_panel.add_message("creature", text)
                if self._audio_bridge is not None:
                    self._audio_bridge.play_voice(text)
            elif behavior is not None:
                self._apply_behavior(behavior)
        except Exception as exc:
            logger.error("Response check error: %s", exc, exc_info=True)

    # ── Behavior / evolution / death ─────────────────────────────────

    def _apply_behavior(self, behavior: IdleBehavior) -> None:
        """Apply an autonomous behavior — set animation and optionally speak."""
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
            self._chat_panel.add_message("creature", behavior.message)
            if self._audio_bridge is not None:
                self._audio_bridge.play_voice(behavior.message)

    def _request_autonomous_remark(self, behavior: IdleBehavior) -> None:
        """Submit an autonomous LLM remark for a verbal behavior."""
        if self._response_handler.is_busy or self._response_handler.is_auto_busy:
            self._apply_behavior(behavior)
            return

        manager = self.window.manager
        if manager is None or not manager.is_initialized:
            self._apply_behavior(behavior)
            return

        situation = get_behavior_situation(
            behavior.action_type,
            self._mood_engine.current_mood,
            self._game_systems.needs,
        )
        if situation is None:
            self._apply_behavior(behavior)
            return

        logger.info(
            "Requesting autonomous LLM remark: %s (mood=%s)",
            behavior.action_type.value,
            self._mood_engine.current_mood.value,
        )
        self._scheduler.acquire("chat")

        async def _generate() -> str | None:
            return await manager.generate_autonomous_remark(situation)

        try:
            future = self.window.submit_async(_generate())
            self._response_handler.start_autonomous(future, behavior)
        except RuntimeError:
            logger.warning("Async bridge dead — falling back to canned behavior")
            self._scheduler.release("chat")
            self._apply_behavior(behavior)

    def _request_interaction_reaction(self, action_key: str) -> None:
        """Submit an LLM reaction to a player interaction."""
        if self._response_handler.is_busy or self._response_handler.is_auto_busy:
            fallback = _INTERACTION_FALLBACKS.get(action_key)
            if fallback:
                self._chat_panel.add_message("creature", fallback)
            return

        manager = self.window.manager
        if manager is None or not manager.is_initialized:
            return

        situation = _build_interaction_situation(
            action_key, self._creature_state, self._tank_env,
            self._game_systems.needs,
        )
        if situation is None:
            return

        logger.info("Requesting interaction reaction: %s", action_key)
        self._scheduler.acquire("chat")

        async def _generate() -> str | None:
            return await manager.generate_autonomous_remark(situation)

        try:
            future = self.window.submit_async(_generate())
            self._response_handler.start_autonomous(future)
        except RuntimeError:
            logger.warning("Async bridge dead — skipping interaction reaction")
            self._scheduler.release("chat")

    def _start_evolution(self, new_stage: CreatureStage) -> None:
        """Begin evolution celebration sequence."""
        self._response_handler.cancel_autonomous()

        old_stage = self._creature_state.stage
        self._evolution_engine.evolve(self._creature_state, new_stage)
        self._evolution_active = True
        self._evolution_timer = 0.0

        msg = f"Evolution! {old_stage.value} -> {new_stage.value}!"
        self._chat_panel.add_message("system", msg)

        if self._audio_bridge is not None:
            self._audio_bridge.play_sfx("evolution_chime")

        # Update food types for new stage
        available = self._interaction_manager.feeding_engine.get_available_foods(
            new_stage,
        )
        self._hud.set_food_types([f.value for f in available])
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
        new_state, record = self._death_engine.on_death(
            cause, self._creature_state,
        )
        self._death_message = record.message
        self._chat_panel.add_message("system", record.message)

        if self._audio_bridge is not None:
            self._audio_bridge.stop_ambient()

        logger.info("Creature died: %s — %s", cause.value, record.message)

    # ── Traits ─────────────────────────────────────────────────────────

    def _get_traits(self) -> TraitProfile:
        """Get current TraitProfile from ConversationManager, or default."""
        manager = self.window.manager
        if (
            manager is not None
            and hasattr(manager, "traits")
            and manager.traits is not None
        ):
            return manager.traits
        return TraitProfile()

    # ── Render ────────────────────────────────────────────────────────

    def _render(self, surface: pygame.Surface) -> None:
        """Per-frame render."""
        self._tank_renderer.render(surface)
        self._creature_renderer.render(surface)
        self._hud.render(surface)
        self._chat_panel.render(surface)

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

    def _toggle_lineage(self) -> None:
        sm = self._scene_manager
        if sm.state == GameState.LINEAGE:
            sm.close_drawer()
        else:
            sm.open_lineage()

    def _on_key_down(self, event: Any) -> None:
        key = getattr(event, "key", 0)
        char = getattr(event, "unicode", "")
        self._chat_panel.handle_key(key, char)

    def _on_mouse_click(self, event: Any) -> None:
        mx = getattr(event, "pos", (0, 0))[0]
        my = getattr(event, "pos", (0, 0))[1]

        # HUD sidebar action clicks
        action = self._hud.handle_click(mx, my)
        if action is not None:
            self._on_action(action)
            return

        # Top bar button checks
        if self._hud.settings_rect:
            rx, ry, rw, rh = self._hud.settings_rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._toggle_settings()
                return

        if self._hud.lineage_rect:
            rx, ry, rw, rh = self._hud.lineage_rect
            if rx <= mx < rx + rw and ry <= my < ry + rh:
                self._toggle_lineage()
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

    # ── HUD actions ───────────────────────────────────────────────────

    def _on_action(self, action_key: str) -> None:
        """Handle HUD sidebar action button clicks."""
        if self.game_over:
            return

        im = self._interaction_manager
        creature = self._creature_state
        tank = self._tank_env
        action_succeeded = False

        if action_key == "feed":
            # Feed is handled via the food dropdown → _on_food_selected
            available = im.feeding_engine.get_available_foods(creature.stage)
            if not available:
                self._chat_panel.add_message("system", "No food available.")
            elif len(available) == 1:
                self._feed_creature(available[0])
                action_succeeded = True
            # else: dropdown is already open from HUD.handle_action_click

        elif action_key == "aerator":
            if tank.environment_type == EnvironmentType.TERRARIUM:
                care_result = im.care_engine.sprinkle(tank, creature)
            else:
                care_result = im.care_engine.aerate_tank(tank)
            self._chat_panel.add_message("system", care_result.message)
            if care_result.success:
                action_succeeded = True
                self._hud.set_cooldown("aerator", AERATOR_COOLDOWN_SECONDS)
                if self._audio_bridge is not None:
                    self._audio_bridge.play_sfx("bubbles")

        elif action_key == "temp_up":
            care_result = im.care_engine.adjust_temperature(tank, 1.0, creature)
            self._chat_panel.add_message("system", care_result.message)
            action_succeeded = True

        elif action_key == "temp_down":
            care_result = im.care_engine.adjust_temperature(
                tank, -1.0, creature,
            )
            self._chat_panel.add_message("system", care_result.message)
            action_succeeded = True

        elif action_key == "clean":
            care_result = im.care_engine.clean_tank(tank)
            self._chat_panel.add_message("system", care_result.message)
            if care_result.success:
                action_succeeded = True
                self._hud.set_cooldown("clean", CLEANING_DURATION_SECONDS)

        elif action_key == "drain":
            if tank.water_level > 0.0:
                care_result = im.care_engine.drain_tank(tank, creature)
            else:
                care_result = im.care_engine.fill_tank(tank, creature)
            self._chat_panel.add_message("system", care_result.message)
            action_succeeded = True

        elif action_key == "fill":
            care_result = im.care_engine.fill_tank(tank, creature)
            self._chat_panel.add_message("system", care_result.message)
            action_succeeded = True

        elif action_key == "tap_glass":
            creature.interaction_count += 1
            self._game_systems._interaction_count_delta += 1
            self._chat_panel.add_message("system", "You tap the glass...")
            if self._audio_bridge is not None:
                self._audio_bridge.play_sfx("glass_tap")
            action_succeeded = True

        if action_succeeded:
            self._request_interaction_reaction(action_key)

    def _on_food_selected(self, food_name: str) -> None:
        """Handle food dropdown selection from HUD."""
        try:
            food_type = FoodType(food_name)
        except ValueError:
            logger.warning("Unknown food type: %s", food_name)
            return
        self._feed_creature(food_type)

    def _feed_creature(self, food_type: FoodType) -> None:
        """Feed the creature with the given food type."""
        im = self._interaction_manager
        creature = self._creature_state
        result = im.feeding_engine.feed(creature, food_type)
        self._game_systems._interaction_count_delta += 1
        if result.success:
            self._chat_panel.add_message("system", result.message)
            self._creature_renderer.set_animation(AnimationState.EATING)
            self._hud.set_cooldown(
                "feed",
                im.feeding_engine.cooldown_remaining(creature),
            )
            if self._audio_bridge is not None:
                self._audio_bridge.play_sfx("feeding_splash")
            self._request_interaction_reaction("feed")
        else:
            self._chat_panel.add_message("system", result.message)

    # ── Chat ──────────────────────────────────────────────────────────

    def _on_chat_submit(self, text: str) -> None:
        """Handle typed chat input — cancels in-flight LLM calls."""
        if self.game_over or not text.strip():
            return

        self._chat_panel.add_message("user", text)

        # Typed input takes priority — clear STT and TTS
        self._stt_queued_text = None
        if self._audio_bridge is not None:
            self._audio_bridge.cancel_pending_voice()

        # Cancel any in-flight responses
        self._response_handler.cancel_autonomous()
        if self._response_handler.is_busy:
            self._response_handler.cancel_response()
            self._chat_panel.finish_streaming()
            logger.debug("Cancelled previous pending chat for new input")

        self._submit_chat(text)

    def _submit_chat(self, text: str) -> None:
        """Submit text to ConversationManager via streaming."""
        self._creature_renderer.set_animation(AnimationState.TALKING)
        self._game_systems._interaction_count_delta += 1

        manager = self.window.manager
        if manager is not None and manager.is_initialized:
            self._chat_panel.add_message("creature", "", streaming=True)
            self._response_handler.start_stream()
            self._scheduler.acquire("chat")

            handler = self._response_handler

            async def _process() -> str:
                tokens: list[str] = []
                async for token in manager.process_input_stream(text):
                    tokens.append(token)
                    handler.put_token(token)
                handler.put_token(None)  # sentinel
                return "".join(tokens)

            try:
                future = self.window.submit_async(_process())
                self._response_handler.start_response(future)
            except RuntimeError:
                logger.warning("Async bridge dead — cannot submit chat")
                self._chat_panel.finish_streaming()
                self._chat_panel.add_message(
                    "creature", "*yawns* Brain not ready yet... try again.",
                )
                self._creature_renderer.set_animation(AnimationState.IDLE)
                self._scheduler.release("chat")
        else:
            self._chat_panel.add_message(
                "creature", "*yawns* Brain not ready yet... try again.",
            )
            self._creature_renderer.set_animation(AnimationState.IDLE)

    def _check_stt_queue(self) -> None:
        """Submit queued STT text after debounce."""
        if self._stt_queued_text is None:
            return
        if time.monotonic() - self._stt_queued_time < _STT_DEBOUNCE_SECONDS:
            return
        if self._response_handler.is_busy:
            return

        text = self._stt_queued_text
        self._stt_queued_text = None

        self._response_handler.cancel_autonomous()

        logger.info("STT submitted (debounced): %s", text)
        self._chat_panel.add_message("user", text)
        self._submit_chat(text)

    def _on_stt_result(self, text: str) -> None:
        """Handle speech-to-text result — queue for debounced submission."""
        if not text or not text.strip():
            return
        logger.info("STT result: %s", text)
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
