"""Game state machine with drawer open/close transitions.

Manages PLAYING/SETTINGS/LINEAGE states and smooth drawer slide animation.
"""
from __future__ import annotations

from seaman_brain.gui.game_systems import GameState

_DRAWER_ANIM_DURATION = 0.3  # 300ms ease-out


class SceneManager:
    """Tracks which scene is active and animates drawer transitions."""

    def __init__(self) -> None:
        self.state = GameState.PLAYING
        self.drawer_open = False
        self.drawer_progress = 0.0  # 0.0=closed, 1.0=fully open
        self._drawer_target = 0.0

    def open_settings(self) -> None:
        self.state = GameState.SETTINGS
        self.drawer_open = True
        self._drawer_target = 1.0

    def open_lineage(self) -> None:
        self.state = GameState.LINEAGE
        self.drawer_open = True
        self._drawer_target = 1.0

    def close_drawer(self) -> None:
        self.state = GameState.PLAYING
        self.drawer_open = False
        self._drawer_target = 0.0

    def update(self, dt: float) -> None:
        """Animate drawer progress toward target with ease-out."""
        if self.drawer_progress < self._drawer_target:
            remaining = 1.0 - self.drawer_progress
            step = remaining * dt / _DRAWER_ANIM_DURATION
            self.drawer_progress = min(
                1.0, self.drawer_progress + max(step, dt / _DRAWER_ANIM_DURATION)
            )
        elif self.drawer_progress > self._drawer_target:
            remaining = self.drawer_progress
            step = remaining * dt / _DRAWER_ANIM_DURATION
            self.drawer_progress = max(
                0.0, self.drawer_progress - max(step, dt / _DRAWER_ANIM_DURATION)
            )
