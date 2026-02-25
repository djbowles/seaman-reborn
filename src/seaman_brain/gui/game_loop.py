"""Full game engine integrating all Pygame subsystems.

Orchestrates the tank renderer, creature sprites, chat panel, HUD,
interactions, audio bridge, needs system, mood engine, autonomous
behaviors, and event system into the main game loop.

Loop order: process input -> update needs/mood/behaviors/events ->
update animations -> render tank -> render creature -> render HUD ->
render chat -> flip display.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pygame

from seaman_brain.behavior.autonomous import BehaviorEngine, BehaviorType, IdleBehavior
from seaman_brain.behavior.events import EventSystem
from seaman_brain.behavior.mood import CreatureMood, MoodEngine
from seaman_brain.config import SeamanConfig
from seaman_brain.creature.evolution import EvolutionEngine
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.clock import GameClock
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.gui.audio_integration import PygameAudioBridge
from seaman_brain.gui.chat_panel import ChatPanel
from seaman_brain.gui.hud import HUD
from seaman_brain.gui.interactions import InteractionManager, InteractionType
from seaman_brain.gui.sprites import AnimationState, CreatureRenderer
from seaman_brain.gui.tank_renderer import TankRenderer
from seaman_brain.gui.window import GameWindow
from seaman_brain.needs.death import DeathCause, DeathEngine
from seaman_brain.needs.system import CreatureNeeds, NeedsEngine
from seaman_brain.personality.traits import TraitProfile
from seaman_brain.types import CreatureStage, MessageRole

logger = logging.getLogger(__name__)

# ── Colors for overlays ──────────────────────────────────────────────

_GAMEOVER_BG = (10, 5, 5, 200)
_GAMEOVER_TEXT = (220, 60, 60)
_GAMEOVER_HINT = (180, 180, 180)
_EVOLUTION_GLOW = (255, 220, 100)
_NOTIFICATION_BG = (20, 40, 60, 200)
_NOTIFICATION_TEXT = (220, 220, 180)

# ── Game state enum ──────────────────────────────────────────────────

_NEEDS_UPDATE_INTERVAL = 1.0  # seconds between needs ticks
_BEHAVIOR_CHECK_INTERVAL = 5.0  # seconds between behavior checks
_EVENT_CHECK_INTERVAL = 3.0  # seconds between event checks


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
        cfg = config or SeamanConfig()
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
        self._audio_bridge: PygameAudioBridge | None = None

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

        # Pending conversation future
        self._pending_response: Any = None

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
            try:
                self._overlay_font = pygame.font.SysFont("consolas", 16)
                self._overlay_title_font = pygame.font.SysFont("consolas", 28, bold=True)
            except Exception:
                self._overlay_font = pygame.font.Font(None, 16)
                self._overlay_title_font = pygame.font.Font(None, 28)

    def initialize(self) -> None:
        """Initialize all subsystems and register with the window."""
        self.window.initialize()

        # Set up audio bridge with the window's async loop
        self._audio_bridge = PygameAudioBridge(
            audio_config=self._config.audio,
            async_loop=self.window._loop,
        )

        # Sync tank area to renderers
        render_area = self._tank_renderer.render_area
        self._creature_renderer.set_bounds(*render_area)
        self._interaction_manager.set_tank_area(*render_area)

        # Register event handlers
        self.window.register_event_handler(pygame.MOUSEBUTTONDOWN, self._on_mouse_click)
        self.window.register_event_handler(pygame.MOUSEMOTION, self._on_mouse_move)
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
        if self.game_over:
            return

        if self._evolution_active:
            self._update_evolution_celebration(dt)
            return

        # Update tank environment degradation
        self._tank.update(dt, self._config.environment)

        # Accumulate timers
        self._needs_timer += dt
        self._behavior_timer += dt
        self._event_timer += dt

        # Periodic needs update
        if self._needs_timer >= _NEEDS_UPDATE_INTERVAL:
            elapsed = self._needs_timer
            self._needs_timer = 0.0
            self._update_needs(elapsed)

        # Check death
        cause = self._death_engine.check_death(
            self._creature_state, self._needs, self._tank
        )
        if cause is not None:
            self._handle_death(cause)
            return

        # Update mood — use traits from ConversationManager if available
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

        # Periodic behavior check
        if self._behavior_timer >= _BEHAVIOR_CHECK_INTERVAL:
            self._behavior_timer = 0.0
            self._check_behaviors(time_context)

        # Periodic event check
        if self._event_timer >= _EVENT_CHECK_INTERVAL:
            self._event_timer = 0.0
            self._check_events(time_context)

        # Check evolution
        self._check_evolution()

        # Update animations
        self._tank_renderer.update(dt, self._tank)
        self._creature_renderer.update(dt)
        self._chat_panel.update(dt)
        self._hud.update(dt)
        self._interaction_manager.update(dt)
        if self._audio_bridge is not None:
            self._audio_bridge.update(dt)

        # Update notifications
        alive: list[tuple[str, float]] = []
        for text, remaining in self._notifications:
            remaining -= dt
            if remaining > 0:
                alive.append((text, remaining))
        self._notifications = alive

        # Check for pending conversation response
        self._check_pending_response()

        # Sync creature stage to renderer
        if self._creature_renderer.stage != self._creature_state.stage:
            self._creature_renderer.set_stage(self._creature_state.stage)

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
        """Check for autonomous creature behaviors."""
        creature_dict = {
            "stage": self._creature_state.stage.value,
            "mood": self._creature_state.mood,
            "trust": self._creature_state.trust_level,
            "hunger": self._creature_state.hunger,
        }
        behavior = self._behavior_engine.get_idle_behavior(
            creature_state=creature_dict,
            needs=self._needs,
            mood=self._mood_engine.current_mood,
            time_context=time_context,
        )
        if behavior is not None:
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

    def _on_chat_submit(self, text: str) -> None:
        """Handle chat input submission — send to ConversationManager."""
        if self.game_over or not text.strip():
            return

        self._creature_renderer.set_animation(AnimationState.TALKING)
        self._interaction_count_delta += 1

        manager = self.window.manager
        if manager is not None and self.window._loop is not None:
            self._chat_panel.start_streaming()

            async def _process() -> str:
                return await manager.process_input(text)

            self._pending_response = asyncio.run_coroutine_threadsafe(
                _process(), self.window._loop
            )
        else:
            self._chat_panel.add_message(
                MessageRole.ASSISTANT,
                "*yawns* Brain not ready yet... try again.",
            )
            self._creature_renderer.set_animation(AnimationState.IDLE)

    def _check_pending_response(self) -> None:
        """Check if a pending async conversation response is ready."""
        if self._pending_response is None:
            return

        if not self._pending_response.done():
            return

        try:
            result = self._pending_response.result(timeout=0)
            self._chat_panel.finish_streaming()
            self._chat_panel.add_message(MessageRole.ASSISTANT, result)
            if self._audio_bridge is not None:
                self._audio_bridge.play_voice(result)
        except Exception as exc:
            self._chat_panel.finish_streaming()
            self._chat_panel.add_message(
                MessageRole.ASSISTANT, f"*glitches* {exc}"
            )
            logger.error("Conversation error: %s", exc)
        finally:
            self._pending_response = None
            self._creature_renderer.set_animation(AnimationState.IDLE)

    def _on_mouse_click(self, event: pygame.event.Event) -> None:
        """Handle mouse click events."""
        if self.game_over:
            # Any click during game over restarts
            self._restart_game()
            return

        mx, my = event.pos
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
        self._creature_renderer.set_mouse_position(float(mx), float(my))
        self._interaction_manager.handle_mouse_move(mx, my)

    def _on_key_down(self, event: pygame.event.Event) -> None:
        """Handle key press events."""
        # Chat panel gets first chance
        if self._chat_panel.handle_event(event):
            return

        # Audio controls
        if self._audio_bridge is not None:
            if self._audio_bridge.handle_key_event(event.key):
                return

        # HUD toggle
        if event.key == pygame.K_h:
            self._hud.toggle_mode()

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

        # 1. Tank background
        self._tank_renderer.render(surface, self._tank)

        # 2. Creature
        self._creature_renderer.render(surface)

        # 3. Interaction effects (ripples, food drops, buttons)
        self._interaction_manager.render(surface)

        # 4. HUD
        self._hud.render(surface, self._creature_state, self._tank)

        # 5. Chat panel
        self._chat_panel.render(surface)

        # 6. Notifications
        self._render_notifications(surface)

        # 7. Evolution celebration overlay
        if self._evolution_active:
            self._render_evolution_overlay(surface)

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

    def shutdown(self) -> None:
        """Clean shutdown of all subsystems."""
        if self._audio_bridge is not None:
            self._audio_bridge.shutdown()
        self.window.shutdown()
        logger.info("GameEngine shutdown complete")
