"""Pygame window setup and main game loop.

Manages Pygame initialization, the main loop (input -> update -> render),
event routing to game subsystems, async bridge to ConversationManager,
and clean shutdown with state saving.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from typing import Any

import pygame

from seaman_brain.config import GUIConfig, SeamanConfig

logger = logging.getLogger(__name__)

# Default colors
_BG_COLOR = (10, 20, 40)
_TEXT_COLOR = (200, 220, 240)
_HEADER_BG = (15, 30, 55)
_STATUS_GREEN = (60, 200, 100)
_STATUS_YELLOW = (220, 200, 60)
_STATUS_RED = (220, 60, 60)


class GameWindow:
    """Manages the Pygame window, main loop, and subsystem coordination.

    The game loop runs: handle_events() -> update(dt) -> render() -> clock.tick().
    An async bridge runs ConversationManager in a background thread.

    Attributes:
        config: GUI configuration (window size, FPS, etc.).
        running: Whether the main loop is active.
    """

    def __init__(
        self,
        config: SeamanConfig | None = None,
        gui_config: GUIConfig | None = None,
    ) -> None:
        """Initialize the game window.

        Args:
            config: Full application config. Uses defaults if None.
            gui_config: GUI-specific config override. Derived from config if None.
        """
        self._full_config = config or SeamanConfig()
        self.config = gui_config or self._full_config.gui

        self.running = False
        self._screen: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None

        # Async bridge
        self._loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self._manager: Any | None = None
        self._manager_initialized = False

        # Event handlers: pygame event type -> list of callbacks
        self._event_handlers: dict[int, list[Callable[[pygame.event.Event], None]]] = {}

        # Subsystem update callbacks: called each frame with dt
        self._update_callbacks: list[Callable[[float], None]] = []

        # Subsystem render callbacks: called each frame with screen surface
        self._render_callbacks: list[Callable[[pygame.Surface], None]] = []

        # Status message for display
        self._status_message = "Initializing..."
        self._frame_count = 0

    @property
    def screen(self) -> pygame.Surface | None:
        """The main display surface, if initialized."""
        return self._screen

    @property
    def width(self) -> int:
        """Window width in pixels."""
        return self.config.window_width

    @property
    def height(self) -> int:
        """Window height in pixels."""
        return self.config.window_height

    @property
    def fps(self) -> int:
        """Target frames per second."""
        return self.config.fps

    @property
    def manager(self) -> Any | None:
        """The ConversationManager, if initialized."""
        return self._manager

    def register_event_handler(
        self, event_type: int, handler: Callable[[pygame.event.Event], None]
    ) -> None:
        """Register a callback for a specific Pygame event type.

        Args:
            event_type: Pygame event constant (e.g. pygame.KEYDOWN).
            handler: Callback receiving the pygame event.
        """
        self._event_handlers.setdefault(event_type, []).append(handler)

    def unregister_event_handler(
        self, event_type: int, handler: Callable[[pygame.event.Event], None]
    ) -> None:
        """Remove a previously registered event handler.

        Args:
            event_type: Pygame event constant.
            handler: The callback to remove.
        """
        handlers = self._event_handlers.get(event_type, [])
        if handler in handlers:
            handlers.remove(handler)

    def register_update(self, callback: Callable[[float], None]) -> None:
        """Register a per-frame update callback.

        Args:
            callback: Called each frame with delta time in seconds.
        """
        self._update_callbacks.append(callback)

    def register_renderer(self, callback: Callable[[pygame.Surface], None]) -> None:
        """Register a per-frame render callback.

        Args:
            callback: Called each frame with the display surface.
        """
        self._render_callbacks.append(callback)

    def initialize(self) -> None:
        """Initialize Pygame, create the window, and set up fonts.

        Call this before run(). Idempotent.
        """
        if self._screen is not None:
            return

        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.config.window_width, self.config.window_height)
        )
        pygame.display.set_caption("Seaman Reborn")
        self._clock = pygame.time.Clock()

        pygame.font.init()
        self._font = pygame.font.SysFont("consolas", 16)
        self._title_font = pygame.font.SysFont("consolas", 24, bold=True)

        self._status_message = "Ready"
        logger.info(
            "Pygame initialized: %dx%d @ %d FPS",
            self.config.window_width,
            self.config.window_height,
            self.config.fps,
        )

    def _start_async_bridge(self) -> None:
        """Start the background asyncio loop for ConversationManager."""
        self._loop = asyncio.new_event_loop()

        def _run_loop() -> None:
            assert self._loop is not None
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._async_thread = threading.Thread(
            target=_run_loop, daemon=True, name="seaman-async"
        )
        self._async_thread.start()

    def _init_manager_async(self) -> None:
        """Schedule ConversationManager initialization on the async thread."""
        if self._loop is None:
            return

        async def _init() -> None:
            try:
                from seaman_brain.conversation.manager import ConversationManager

                self._manager = ConversationManager(config=self._full_config)
                await self._manager.initialize()
                self._manager_initialized = True
                self._status_message = "Brain online"
                logger.info("ConversationManager initialized in async bridge")
            except Exception as exc:
                logger.error("Failed to initialize ConversationManager: %s", exc)
                self._status_message = f"Brain error: {exc}"

        asyncio.run_coroutine_threadsafe(_init(), self._loop)

    def submit_async(self, coro: Any) -> Any:
        """Submit a coroutine to the background async loop.

        Args:
            coro: An awaitable coroutine.

        Returns:
            A concurrent.futures.Future for the result.

        Raises:
            RuntimeError: If the async bridge is not running.
        """
        if self._loop is None:
            raise RuntimeError("Async bridge not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def handle_events(self) -> None:
        """Process all pending Pygame events.

        Routes events to registered handlers. QUIT event stops the loop.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return

            # Dispatch to registered handlers (ESC handled by GameEngine)
            handlers = self._event_handlers.get(event.type, [])
            for handler in handlers:
                try:
                    handler(event)
                except Exception as exc:
                    logger.error("Event handler error: %s", exc)

    def update(self, dt: float) -> None:
        """Update all game subsystems.

        Args:
            dt: Delta time in seconds since last frame.
        """
        self._frame_count += 1

        for callback in self._update_callbacks:
            try:
                callback(dt)
            except Exception as exc:
                logger.error("Update callback error: %s", exc)

    def render(self) -> None:
        """Render the current frame.

        Clears the screen, calls registered renderers, draws the
        default status overlay, and flips the display.
        """
        if self._screen is None:
            return

        # Clear
        self._screen.fill(_BG_COLOR)

        # Registered renderers
        for callback in self._render_callbacks:
            try:
                callback(self._screen)
            except Exception as exc:
                logger.error("Render callback error: %s", exc)

        # Default status overlay (only when no subsystem renderers registered)
        if not self._render_callbacks:
            self._render_status_overlay()

        # Flip
        pygame.display.flip()

    def _render_status_overlay(self) -> None:
        """Draw a minimal status bar at top of screen."""
        if self._screen is None or self._font is None or self._title_font is None:
            return

        screen = self._screen
        w = self.config.window_width

        # Header bar
        pygame.draw.rect(screen, _HEADER_BG, (0, 0, w, 40))

        # Title
        title_surf = self._title_font.render("Seaman Reborn", True, _TEXT_COLOR)
        screen.blit(title_surf, (10, 8))

        # Status
        color = _STATUS_GREEN if self._manager_initialized else _STATUS_YELLOW
        status_surf = self._font.render(self._status_message, True, color)
        screen.blit(status_surf, (w - status_surf.get_width() - 10, 12))

        # FPS counter (debug)
        if self.config.show_debug_hud and self._clock is not None:
            fps_text = f"FPS: {self._clock.get_fps():.0f}"
            fps_surf = self._font.render(fps_text, True, (120, 140, 160))
            screen.blit(fps_surf, (w - fps_surf.get_width() - 10, 30))

    def run(self) -> None:
        """Run the main game loop.

        Initializes Pygame if needed, starts the async bridge,
        and runs until the window is closed or Escape is pressed.
        """
        self.initialize()
        assert self._clock is not None

        self._start_async_bridge()
        self._init_manager_async()
        self.running = True

        try:
            while self.running:
                # Delta time in seconds
                dt = self._clock.tick(self.config.fps) / 1000.0

                self.handle_events()
                if not self.running:
                    break

                self.update(dt)
                self.render()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Clean shutdown: save state, stop async loop, quit Pygame."""
        logger.info("Shutting down GameWindow...")
        self.running = False

        # Save creature state via async bridge
        if self._manager is not None and self._loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._manager.shutdown(), self._loop
                )
                future.result(timeout=5.0)
                logger.info("Creature state saved.")
            except Exception as exc:
                logger.error("Error saving state during shutdown: %s", exc)

        # Stop async loop
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._async_thread is not None:
                self._async_thread.join(timeout=3.0)
            self._loop = None
            self._async_thread = None

        # Quit Pygame
        try:
            pygame.quit()
        except Exception:
            pass

        self._screen = None
        self._clock = None
        logger.info("GameWindow shutdown complete.")
