"""HUD and status display - need bars, mood indicator, tank gauges.

Renders creature vital statistics as a Pygame overlay:
- Top bar: creature name/stage, mood indicator (text + color), session timer
- Need bars: hunger, health, comfort, stimulation as colored progress bars
- Tank indicators: temperature gauge, cleanliness meter, oxygen level
- Trust meter: visual representation (0-1)
- Color coding: green=good, yellow=warning, red=critical
- Compact mode (icon-only) and expanded mode (icon + label + value)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pygame

from seaman_brain.config import GUIConfig
from seaman_brain.creature.state import CreatureState
from seaman_brain.environment.tank import TankEnvironment
from seaman_brain.types import CreatureStage

logger = logging.getLogger(__name__)

# ── Colors ───────────────────────────────────────────────────────────────

_BG_COLOR = (15, 25, 45, 200)  # Semi-transparent dark blue
_TOP_BAR_BG = (10, 18, 35, 220)
_TEXT_COLOR = (200, 220, 240)
_TEXT_DIM = (130, 150, 170)
_LABEL_COLOR = (160, 180, 200)

# Status colors (bar fill)
_COLOR_GREEN = (60, 200, 100)
_COLOR_YELLOW = (220, 200, 60)
_COLOR_RED = (220, 60, 60)
_COLOR_BLUE = (60, 140, 220)

# Bar background
_BAR_BG = (30, 40, 60)
_BAR_BORDER = (60, 80, 110)

# Mood colors
_MOOD_COLORS: dict[str, tuple[int, int, int]] = {
    "hostile": (220, 50, 50),
    "irritated": (220, 120, 50),
    "sardonic": (180, 140, 80),
    "neutral": (160, 180, 200),
    "curious": (80, 180, 220),
    "amused": (120, 200, 160),
    "philosophical": (160, 120, 220),
    "content": (80, 220, 120),
}

# Stage display names
_STAGE_NAMES: dict[CreatureStage, str] = {
    CreatureStage.MUSHROOMER: "Mushroomer",
    CreatureStage.GILLMAN: "Gillman",
    CreatureStage.PODFISH: "Podfish",
    CreatureStage.TADMAN: "Tadman",
    CreatureStage.FROGMAN: "Frogman",
}

# Icons for compact mode (Unicode characters)
_ICONS: dict[str, str] = {
    "hunger": "H",
    "health": "+",
    "comfort": "C",
    "stimulation": "S",
    "temperature": "T",
    "cleanliness": "~",
    "oxygen": "O",
    "trust": "*",
}

# ── Constants ────────────────────────────────────────────────────────────

_TOP_BAR_HEIGHT = 36
_BAR_HEIGHT = 14
_BAR_HEIGHT_COMPACT = 10
_BAR_WIDTH = 120
_BAR_WIDTH_COMPACT = 60
_SECTION_PADDING = 8
_BAR_SPACING = 20
_BAR_SPACING_COMPACT = 14
_FONT_SIZE = 14
_TITLE_FONT_SIZE = 16
_LABEL_WIDTH = 90
_VALUE_WIDTH = 40


def _status_color(value: float, *, inverted: bool = False) -> tuple[int, int, int]:
    """Return green/yellow/red based on value (0-1).

    Args:
        value: The metric value in range [0, 1].
        inverted: If True, high values are bad (e.g. hunger: 1.0 = starving).

    Returns:
        An RGB color tuple.
    """
    effective = 1.0 - value if inverted else value
    if effective >= 0.6:
        return _COLOR_GREEN
    if effective >= 0.3:
        return _COLOR_YELLOW
    return _COLOR_RED


def _format_percent(value: float) -> str:
    """Format a 0-1 float as a percentage string."""
    return f"{int(value * 100)}%"


def _format_temperature(temp: float) -> str:
    """Format temperature in Celsius."""
    return f"{temp:.1f}C"


@dataclass
class HUDMetric:
    """A single metric to display on the HUD.

    Fields:
        icon: Short icon/symbol for compact mode.
        label: Full label for expanded mode.
        value: Current value (0.0-1.0 for bars, or raw for temperature).
        display_text: Formatted text for the value.
        color: RGB color for the bar fill.
        bar_value: Normalized 0-1 value for the progress bar.
    """

    icon: str
    label: str
    value: float
    display_text: str
    color: tuple[int, int, int]
    bar_value: float


class HUD:
    """Heads-up display overlay for creature and tank status.

    Renders status bars, mood indicator, and tank gauges on a Pygame surface.
    Supports compact mode (icon + bar only) and expanded mode (icon + label + bar + value).

    Attributes:
        compact: Whether to use compact mode.
    """

    def __init__(self, gui_config: GUIConfig | None = None) -> None:
        """Initialize the HUD.

        Args:
            gui_config: GUI configuration for sizing. Uses defaults if None.
        """
        self._config = gui_config or GUIConfig()
        self.compact = False

        # Font (lazy-initialized)
        self._font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None
        self._font_height = 0

        # Session timer
        self._session_time = 0.0

    @property
    def session_time(self) -> float:
        """Elapsed session time in seconds."""
        return self._session_time

    def _ensure_fonts(self) -> None:
        """Initialize fonts if not yet done."""
        if self._font is None:
            try:
                self._font = pygame.font.SysFont("consolas", _FONT_SIZE)
                self._title_font = pygame.font.SysFont("consolas", _TITLE_FONT_SIZE, bold=True)
            except Exception:
                self._font = pygame.font.Font(None, _FONT_SIZE)
                self._title_font = pygame.font.Font(None, _TITLE_FONT_SIZE)
            self._font_height = self._font.get_linesize()

    def toggle_mode(self) -> None:
        """Toggle between compact and expanded display modes."""
        self.compact = not self.compact

    def update(self, dt: float) -> None:
        """Update the HUD state.

        Args:
            dt: Delta time in seconds since last frame.
        """
        self._session_time += dt

    def _build_need_metrics(self, creature: CreatureState) -> list[HUDMetric]:
        """Build the need bar metrics from creature state.

        Args:
            creature: Current creature state.

        Returns:
            List of HUDMetric for hunger, health, comfort, stimulation.
        """
        metrics: list[HUDMetric] = []

        # Hunger: 0=full, 1=starving → inverted color (high = bad)
        hunger_bar = 1.0 - creature.hunger  # fullness for display
        metrics.append(HUDMetric(
            icon=_ICONS["hunger"],
            label="Hunger",
            value=creature.hunger,
            display_text=_format_percent(hunger_bar),
            color=_status_color(creature.hunger, inverted=True),
            bar_value=hunger_bar,
        ))

        # Health: 0=dead, 1=healthy → normal color
        metrics.append(HUDMetric(
            icon=_ICONS["health"],
            label="Health",
            value=creature.health,
            display_text=_format_percent(creature.health),
            color=_status_color(creature.health),
            bar_value=creature.health,
        ))

        # Comfort: 0=miserable, 1=happy → normal color
        metrics.append(HUDMetric(
            icon=_ICONS["comfort"],
            label="Comfort",
            value=creature.comfort,
            display_text=_format_percent(creature.comfort),
            color=_status_color(creature.comfort),
            bar_value=creature.comfort,
        ))

        return metrics

    def _build_tank_metrics(self, tank: TankEnvironment) -> list[HUDMetric]:
        """Build the tank indicator metrics.

        Args:
            tank: Current tank environment state.

        Returns:
            List of HUDMetric for temperature, cleanliness, oxygen.
        """
        metrics: list[HUDMetric] = []

        # Temperature: show as color (blue=cold, green=good, red=hot)
        # Normalize to 0-1 for bar display (10-38 range by default)
        temp_norm = max(0.0, min(1.0, (tank.temperature - 10.0) / 28.0))
        # Color based on distance from optimal (20-28 default)
        if 20.0 <= tank.temperature <= 28.0:
            temp_color = _COLOR_GREEN
        elif tank.temperature < 20.0:
            temp_color = _COLOR_BLUE
        else:
            temp_color = _COLOR_RED

        metrics.append(HUDMetric(
            icon=_ICONS["temperature"],
            label="Temp",
            value=tank.temperature,
            display_text=_format_temperature(tank.temperature),
            color=temp_color,
            bar_value=temp_norm,
        ))

        # Cleanliness: 0=filthy, 1=spotless → normal color
        metrics.append(HUDMetric(
            icon=_ICONS["cleanliness"],
            label="Clean",
            value=tank.cleanliness,
            display_text=_format_percent(tank.cleanliness),
            color=_status_color(tank.cleanliness),
            bar_value=tank.cleanliness,
        ))

        # Oxygen: 0=none, 1=saturated → normal color
        metrics.append(HUDMetric(
            icon=_ICONS["oxygen"],
            label="Oxygen",
            value=tank.oxygen_level,
            display_text=_format_percent(tank.oxygen_level),
            color=_status_color(tank.oxygen_level),
            bar_value=tank.oxygen_level,
        ))

        return metrics

    def _build_trust_metric(self, creature: CreatureState) -> HUDMetric:
        """Build the trust meter metric.

        Args:
            creature: Current creature state.

        Returns:
            HUDMetric for trust level.
        """
        return HUDMetric(
            icon=_ICONS["trust"],
            label="Trust",
            value=creature.trust_level,
            display_text=_format_percent(creature.trust_level),
            color=_status_color(creature.trust_level),
            bar_value=creature.trust_level,
        )

    def _format_session_time(self) -> str:
        """Format session time as MM:SS or HH:MM:SS."""
        total_seconds = int(self._session_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def render(
        self,
        surface: pygame.Surface,
        creature: CreatureState,
        tank: TankEnvironment,
    ) -> None:
        """Render the HUD overlay onto the given surface.

        Args:
            surface: Pygame surface to draw on.
            creature: Current creature state for need bars and mood.
            tank: Current tank environment for gauges.
        """
        self._ensure_fonts()
        if self._font is None or self._title_font is None:
            return

        # Draw top bar
        self._render_top_bar(surface, creature)

        # Draw need bars (left side below top bar)
        need_metrics = self._build_need_metrics(creature)
        trust_metric = self._build_trust_metric(creature)
        tank_metrics = self._build_tank_metrics(tank)

        # Position: left column for needs + trust, right column for tank
        y_start = _TOP_BAR_HEIGHT + _SECTION_PADDING
        spacing = _BAR_SPACING_COMPACT if self.compact else _BAR_SPACING

        # Left column: needs + trust
        x_left = _SECTION_PADDING
        y = y_start
        for metric in need_metrics:
            self._render_metric_bar(surface, x_left, y, metric)
            y += spacing

        # Trust bar
        self._render_metric_bar(surface, x_left, y, trust_metric)

        # Right column: tank indicators
        bar_w = _BAR_WIDTH_COMPACT if self.compact else _BAR_WIDTH
        label_w = 0 if self.compact else _LABEL_WIDTH
        value_w = 0 if self.compact else _VALUE_WIDTH
        icon_w = 16
        right_block_w = icon_w + label_w + bar_w + value_w + 12
        x_right = surface.get_width() - right_block_w - _SECTION_PADDING
        y = y_start
        for metric in tank_metrics:
            self._render_metric_bar(surface, x_right, y, metric)
            y += spacing

    def _render_top_bar(
        self,
        surface: pygame.Surface,
        creature: CreatureState,
    ) -> None:
        """Render the top status bar with name/stage, mood, and timer."""
        if self._font is None or self._title_font is None:
            return

        w = surface.get_width()

        # Background
        bar_surface = pygame.Surface((w, _TOP_BAR_HEIGHT), pygame.SRCALPHA)
        bar_surface.fill(_TOP_BAR_BG)
        surface.blit(bar_surface, (0, 0))

        # Creature name/stage (left)
        stage_name = _STAGE_NAMES.get(creature.stage, creature.stage.value)
        title_text = f"Seaman - {stage_name}"
        title_surf = self._title_font.render(title_text, True, _TEXT_COLOR)
        surface.blit(title_surf, (10, (_TOP_BAR_HEIGHT - title_surf.get_height()) // 2))

        # Mood indicator (center)
        mood_name = creature.mood.capitalize()
        mood_color = _MOOD_COLORS.get(creature.mood, _TEXT_DIM)
        mood_surf = self._font.render(mood_name, True, mood_color)
        mood_x = (w - mood_surf.get_width()) // 2
        mood_y = (_TOP_BAR_HEIGHT - mood_surf.get_height()) // 2
        surface.blit(mood_surf, (mood_x, mood_y))

        # Session timer (right)
        timer_text = self._format_session_time()
        timer_surf = self._font.render(timer_text, True, _TEXT_DIM)
        timer_x = w - timer_surf.get_width() - 10
        timer_y = (_TOP_BAR_HEIGHT - timer_surf.get_height()) // 2
        surface.blit(timer_surf, (timer_x, timer_y))

    def _render_metric_bar(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        metric: HUDMetric,
    ) -> None:
        """Render a single metric bar (icon + optional label + bar + optional value).

        Args:
            surface: Surface to draw on.
            x: Left x position.
            y: Top y position.
            metric: The metric data to render.
        """
        if self._font is None:
            return

        bar_h = _BAR_HEIGHT_COMPACT if self.compact else _BAR_HEIGHT
        bar_w = _BAR_WIDTH_COMPACT if self.compact else _BAR_WIDTH

        cursor_x = x

        # Icon
        icon_surf = self._font.render(metric.icon, True, metric.color)
        surface.blit(icon_surf, (cursor_x, y))
        cursor_x += 16

        # Label (expanded mode only)
        if not self.compact:
            label_surf = self._font.render(metric.label, True, _LABEL_COLOR)
            surface.blit(label_surf, (cursor_x, y))
            cursor_x += _LABEL_WIDTH

        # Bar background
        bar_y = y + (self._font_height - bar_h) // 2
        pygame.draw.rect(surface, _BAR_BG, (cursor_x, bar_y, bar_w, bar_h))

        # Bar fill
        fill_w = max(0, int(bar_w * max(0.0, min(1.0, metric.bar_value))))
        if fill_w > 0:
            pygame.draw.rect(surface, metric.color, (cursor_x, bar_y, fill_w, bar_h))

        # Bar border
        pygame.draw.rect(surface, _BAR_BORDER, (cursor_x, bar_y, bar_w, bar_h), 1)

        cursor_x += bar_w + 4

        # Value text (expanded mode only)
        if not self.compact:
            value_surf = self._font.render(metric.display_text, True, _TEXT_COLOR)
            surface.blit(value_surf, (cursor_x, y))
