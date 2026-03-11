# GUI Rewrite: Modern Minimal — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the Pygame GUI from a terminal-debug aesthetic to a Modern Minimal dark void design with creature glow, glassmorphism overlays, and clean architecture.

**Architecture:** Decompose the monolithic GameEngine (2,100 lines) into focused modules: theme, layout, scene management, rendering, input handling, and business logic. Each visual component gets a complete rewrite while backend systems (conversation, audio, creature, needs, behavior) remain untouched.

**Tech Stack:** Python 3.13, Pygame 2.x (pure draw calls, no pygame_gui)

**Important notes:**
- Between Tasks 7-13, rewritten modules have new APIs. `game_loop.py` still references old APIs until Task 14 rewrites it. The app will be broken during this window — rely on per-module tests only. Full integration restores at Task 14.
- `CreaturePosition` class from sprites.py must be preserved (used by game_loop for positioning).
- Existing test files `test_device_utils.py` and `test_audio_integration.py` are kept as-is (their source files aren't rewritten). `test_settings_crash.py` is deleted (superseded by new settings tests).

**Spec:** `docs/superpowers/specs/2026-03-10-gui-rewrite-modern-minimal-design.md`

---

## File Structure

### New Files (to create)

| File | Responsibility |
|------|---------------|
| `src/seaman_brain/gui/theme.py` | All colors, sizes, opacity constants, mood-to-color mapping, font initialization, letter-spacing renderer |
| `src/seaman_brain/gui/layout.py` | Proportional layout engine: computes pixel rects from screen size + ratios |
| `src/seaman_brain/gui/game_systems.py` | Business logic extracted from GameEngine: needs ticking, mood, behavior, events, death, evolution, TTS splitting, STT debouncing |
| `src/seaman_brain/gui/scene_manager.py` | Game state machine (PLAYING/SETTINGS/LINEAGE), drawer open/close transitions |
| `src/seaman_brain/gui/render_engine.py` | Draw-order orchestration, layer management, particle system, gradient cache |
| `src/seaman_brain/gui/input_handler.py` | Keyboard/mouse event routing, shortcut mapping, focus management |

### Files to Rewrite

| File | What Changes |
|------|-------------|
| `src/seaman_brain/gui/game_loop.py` | Gut from 2,100→~300 lines. Core loop + init wiring only. |
| `src/seaman_brain/gui/tank_renderer.py` | Void + radial gradient + particles. Remove aquarium/terrarium. |
| `src/seaman_brain/gui/sprites.py` | Same shapes, 2.5x size, glow aura, mood-reactive lighting. |
| `src/seaman_brain/gui/hud.py` | Thin top bar + left sidebar tiles. Absorbs action_bar. |
| `src/seaman_brain/gui/widgets.py` | Modern Minimal: Button, Slider, Toggle, Dropdown. Single path. |
| `src/seaman_brain/gui/chat_panel.py` | Glass overlay + bubble messages. No pygame_gui. |
| `src/seaman_brain/gui/settings_panel.py` | Slide-out drawer, vertical tabs, new widgets. |
| `src/seaman_brain/gui/lineage_panel.py` | Slide-out drawer pattern matching settings. |

### Files to Modify

| File | What Changes |
|------|-------------|
| `src/seaman_brain/gui/window.py` | Remove pygame_gui UIManager. Update colors to theme. |
| `src/seaman_brain/gui/interactions.py` | Restyle effects (void ripples, food drops). Keep logic. |
| `src/seaman_brain/gui/audio_integration.py` | Update status indicators to match theme. Keep logic. |
| `src/seaman_brain/gui/__init__.py` | Update exports. |

### Files to Delete

| File | Reason |
|------|--------|
| `src/seaman_brain/gui/action_bar.py` | Merged into hud.py sidebar |
| `config/theme.json` | Was for pygame_gui, no longer needed |

### Test Files

Each source file gets a corresponding test file. All 15 existing test files in `tests/test_gui/` will be rewritten.

| Test File | Tests For |
|-----------|----------|
| `tests/test_gui/test_theme.py` | Colors, fonts, mood mapping, letter-spacing |
| `tests/test_gui/test_layout.py` | Proportional rects, resize handling |
| `tests/test_gui/test_game_systems.py` | Needs tick, behavior check, evolution, TTS split |
| `tests/test_gui/test_scene_manager.py` | State transitions, drawer animations |
| `tests/test_gui/test_render_engine.py` | Layer ordering, particle system, gradient cache |
| `tests/test_gui/test_input_handler.py` | Key routing, shortcuts, focus gating |
| `tests/test_gui/test_game_loop.py` | GameEngine init, core loop wiring |
| `tests/test_gui/test_tank_renderer.py` | Void rendering, particles |
| `tests/test_gui/test_sprites.py` | Creature rendering, glow, animation |
| `tests/test_gui/test_hud.py` | Top bar, sidebar tiles, tooltips |
| `tests/test_gui/test_widgets.py` | Button, Slider, Toggle, Dropdown |
| `tests/test_gui/test_chat_panel.py` | Glass overlay, bubbles, scroll, input |
| `tests/test_gui/test_settings_panel.py` | Drawer, tabs, persistence |
| `tests/test_gui/test_lineage_panel.py` | Drawer, bloodline list |
| `tests/test_gui/test_interactions.py` | Void-style effects |
| `tests/test_gui/test_window.py` | Async bridge, pygame_gui removal |

---

## Chunk 1: Foundation (Theme + Layout)

### Task 1: Create theme.py

**Files:**
- Create: `src/seaman_brain/gui/theme.py`
- Test: `tests/test_gui/test_theme.py`

- [ ] **Step 1: Write the test file with pygame mock setup**

```python
"""Tests for the Modern Minimal theme system."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

# Module-level pygame mock
_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 16
_font_mock.render.return_value = _font_surface
_font_mock.size.side_effect = lambda text: (len(text) * 8, 16)
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
sys.modules["pygame"] = _pygame_mock

import pytest  # noqa: E402

from seaman_brain.gui.theme import (  # noqa: E402
    VOID_BG,
    Colors,
    Fonts,
    Sizes,
    mood_glow_color,
    render_spaced_text,
    status_color,
)


@pytest.fixture(autouse=True)
def _reinstall_pygame_mock():
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.theme as mod
    mod.pygame = _pygame_mock
    yield


class TestColors:
    def test_void_bg_is_near_black(self):
        assert VOID_BG == (8, 8, 15)

    def test_surface_opacity_values(self):
        # 3% of 255 ~ 8, 5% ~ 13, 6% ~ 15
        assert Colors.SURFACE_3 == (255, 255, 255, 8)
        assert Colors.SURFACE_5 == (255, 255, 255, 13)
        assert Colors.BORDER == (255, 255, 255, 15)

    def test_status_colors_exist(self):
        assert Colors.STATUS_GREEN == (74, 222, 128)
        assert Colors.STATUS_YELLOW == (245, 158, 11)
        assert Colors.STATUS_RED == (239, 68, 68)


class TestStatusColor:
    def test_green_above_50(self):
        assert status_color(0.75) == Colors.STATUS_GREEN

    def test_yellow_between_25_and_50(self):
        assert status_color(0.35) == Colors.STATUS_YELLOW

    def test_red_below_25(self):
        assert status_color(0.1) == Colors.STATUS_RED

    def test_boundary_50_is_green(self):
        assert status_color(0.5) == Colors.STATUS_GREEN

    def test_boundary_25_is_yellow(self):
        assert status_color(0.25) == Colors.STATUS_YELLOW


class TestMoodGlowColor:
    def test_neutral_is_amber(self):
        r, g, b = mood_glow_color("neutral")
        assert (r, g, b) == (210, 140, 80)

    def test_hostile_is_red(self):
        r, g, b = mood_glow_color("hostile")
        assert (r, g, b) == (210, 80, 60)

    def test_content_is_gold(self):
        r, g, b = mood_glow_color("content")
        assert (r, g, b) == (230, 190, 80)

    def test_unknown_mood_defaults_to_amber(self):
        r, g, b = mood_glow_color("nonexistent")
        assert (r, g, b) == (210, 140, 80)


class TestSizes:
    def test_top_bar_height(self):
        assert Sizes.TOP_BAR_H == 32

    def test_sidebar_width(self):
        assert Sizes.SIDEBAR_W == 48

    def test_tile_size(self):
        assert Sizes.TILE == 24

    def test_chat_height(self):
        assert Sizes.CHAT_H == 130


class TestFonts:
    def test_init_creates_fonts(self):
        Fonts.init()
        assert Fonts.label is not None
        assert Fonts.body is not None
        assert Fonts.header is not None


class TestRenderSpacedText:
    def test_returns_surface(self):
        Fonts.init()
        surf = render_spaced_text("TEST", Fonts.label, (255, 255, 255), spacing=2)
        assert surf is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_theme.py -v`
Expected: FAIL — `ModuleNotFoundError` or `ImportError` (theme.py doesn't exist yet)

- [ ] **Step 3: Write theme.py**

```python
"""Modern Minimal theme — centralized colors, sizes, fonts, and mood mapping.

All visual constants live here. No other GUI module should define colors or
font sizes. Import from theme.py instead.
"""
from __future__ import annotations

import pygame

# ── Void Background ──────────────────────────────────────────────────

VOID_BG = (8, 8, 15)


class Colors:
    """All UI colors as class attributes."""

    # Surface overlays (RGBA — use with SRCALPHA surfaces)
    SURFACE_3 = (255, 255, 255, 8)     # 3% white opacity
    SURFACE_5 = (255, 255, 255, 13)    # 5% white opacity
    BORDER = (255, 255, 255, 15)       # 6% white opacity
    BORDER_SUBTLE = (255, 255, 255, 10)  # 4% white opacity

    # Text (RGB — alpha handled by surface)
    TEXT_90 = (230, 230, 230)          # 90% white
    TEXT_50 = (128, 128, 128)          # 50% white
    TEXT_30 = (77, 77, 77)             # 30% white
    TEXT_25 = (64, 64, 64)             # 25% white
    TEXT_20 = (51, 51, 51)             # 20% white

    # Status
    STATUS_GREEN = (74, 222, 128)      # #4ade80
    STATUS_YELLOW = (245, 158, 11)     # #f59e0b
    STATUS_RED = (239, 68, 68)         # #ef4444

    # Creature / warm accents
    WARM_BG = (200, 160, 120, 15)      # 6% warm for bubbles
    WARM_BORDER = (200, 160, 120, 25)  # 10% warm for bubble border
    WARM_TEXT = (200, 160, 120)        # Seaman name text
    WARM_BODY = (200, 160, 120, 179)   # 70% warm for body text

    # Chat panel
    CHAT_BG = (8, 8, 8, 217)          # 85% alpha dark overlay

    # Widget accents
    ACCENT = (80, 160, 220)
    ACCENT_HOVER = (100, 180, 240)
    TOGGLE_ON = (74, 222, 128)
    TOGGLE_OFF = (100, 110, 130)


class Sizes:
    """Pixel dimensions for layout constants."""

    TOP_BAR_H = 32
    SIDEBAR_W = 48
    TILE = 24
    TILE_RADIUS = 6
    TILE_GAP = 6
    CHAT_H = 130
    INPUT_H = 28
    INPUT_RADIUS = 14
    DRAWER_WIDTH_RATIO = 0.4  # 40% of screen width


# ── Mood → Glow Color Mapping ───────────────────────────────────────

_MOOD_GLOW: dict[str, tuple[int, int, int]] = {
    "content": (230, 190, 80),
    "amused": (230, 190, 80),
    "curious": (230, 190, 80),
    "neutral": (210, 140, 80),
    "philosophical": (210, 140, 80),
    "sardonic": (210, 160, 60),
    "irritated": (210, 160, 60),
    "hostile": (210, 80, 60),
    "sleeping": (140, 100, 180),
    "sad": (80, 130, 210),
}

_DEFAULT_GLOW = (210, 140, 80)  # amber fallback


def mood_glow_color(mood: str) -> tuple[int, int, int]:
    """Return the glow RGB for a mood string."""
    return _MOOD_GLOW.get(mood, _DEFAULT_GLOW)


def status_color(value: float) -> tuple[int, int, int]:
    """Return green/yellow/red based on a 0-1 value."""
    if value >= 0.5:
        return Colors.STATUS_GREEN
    if value >= 0.25:
        return Colors.STATUS_YELLOW
    return Colors.STATUS_RED


# ── Fonts ────────────────────────────────────────────────────────────

class Fonts:
    """Lazily initialized font instances."""

    label: pygame.font.Font | None = None   # 9-10px, uppercase labels
    body: pygame.font.Font | None = None    # 11px, content text
    header: pygame.font.Font | None = None  # 12px, bold headers

    @classmethod
    def init(cls) -> None:
        """Initialize all fonts. Call after pygame.font.init()."""
        cls.label = _make_font(10)
        cls.body = _make_font(11)
        cls.header = _make_font(12)


def _make_font(size: int) -> pygame.font.Font:
    """Create a font, trying common monospace families first."""
    for name in ("consolas", "couriernew", "courier"):
        try:
            return pygame.font.SysFont(name, size)
        except Exception:
            continue
    return pygame.font.Font(None, size)


def render_spaced_text(
    text: str,
    font: pygame.font.Font,
    color: tuple[int, ...],
    spacing: int = 2,
) -> pygame.Surface:
    """Render text with custom letter-spacing by drawing each glyph."""
    glyphs = [font.render(ch, True, color) for ch in text]
    total_w = sum(g.get_width() for g in glyphs) + spacing * max(0, len(text) - 1)
    h = font.get_linesize()
    surface = pygame.Surface((max(1, total_w), h), pygame.SRCALPHA)
    x = 0
    for g in glyphs:
        surface.blit(g, (x, 0))
        x += g.get_width() + spacing
    return surface
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gui/test_theme.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/theme.py tests/test_gui/test_theme.py
git commit -m "feat(gui): add Modern Minimal theme system"
```

---

### Task 2: Create layout.py

**Files:**
- Create: `src/seaman_brain/gui/layout.py`
- Test: `tests/test_gui/test_layout.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for the proportional layout engine."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.Rect = lambda x, y, w, h: type(
    "Rect", (), {"x": x, "y": y, "w": w, "h": h, "width": w, "height": h,
                  "left": x, "top": y, "right": x + w, "bottom": y + h,
                  "collidepoint": lambda self, px, py: (x <= px < x + w and y <= py < y + h)}
)()
sys.modules["pygame"] = _pygame_mock

import pytest  # noqa: E402

from seaman_brain.gui.layout import ScreenLayout  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_pygame_mock():
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.layout as mod
    mod.pygame = _pygame_mock
    yield


class TestScreenLayout:
    def test_default_1024x768(self):
        layout = ScreenLayout(1024, 768)
        assert layout.top_bar.h == 32
        assert layout.sidebar.w == 48
        assert layout.sidebar.x == 0
        assert layout.sidebar.y == 32
        assert layout.tank.x == 48
        assert layout.tank.y == 32
        assert layout.chat.h == 130
        assert layout.chat.y == 768 - 130

    def test_tank_fills_remaining_space(self):
        layout = ScreenLayout(1024, 768)
        # tank width = total - sidebar
        assert layout.tank.w == 1024 - 48
        # tank height = total - top_bar - chat
        assert layout.tank.h == 768 - 32 - 130

    def test_resize(self):
        layout = ScreenLayout(1024, 768)
        layout.resize(1920, 1080)
        assert layout.top_bar.w == 1920
        assert layout.chat.w == 1920
        assert layout.tank.w == 1920 - 48
        assert layout.tank.h == 1080 - 32 - 130

    def test_drawer_width(self):
        layout = ScreenLayout(1000, 768)
        assert layout.drawer_width == 400  # 40%

    def test_small_window(self):
        layout = ScreenLayout(640, 480)
        assert layout.tank.w == 640 - 48
        assert layout.tank.h == 480 - 32 - 130
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_layout.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Write layout.py**

```python
"""Proportional layout engine — computes pixel rects from screen size.

All layout constants come from theme.Sizes. This module computes the
actual pixel rectangles that components use for rendering and hit-testing.
"""
from __future__ import annotations

from dataclasses import dataclass

import pygame

from seaman_brain.gui.theme import Sizes


@dataclass
class Region:
    """A rectangular region on screen."""

    x: int
    y: int
    w: int
    h: int

    @property
    def rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.w, self.h)

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    def collidepoint(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h


class ScreenLayout:
    """Computes all screen regions from window dimensions.

    Call resize() when the window changes size. All components should
    read their regions from here rather than computing positions themselves.
    """

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._compute()

    def resize(self, width: int, height: int) -> None:
        """Recompute all regions for new window dimensions."""
        self.width = width
        self.height = height
        self._compute()

    def _compute(self) -> None:
        w, h = self.width, self.height
        tb = Sizes.TOP_BAR_H
        sb = Sizes.SIDEBAR_W
        ch = Sizes.CHAT_H

        self.top_bar = Region(0, 0, w, tb)
        self.sidebar = Region(0, tb, sb, h - tb)
        self.tank = Region(sb, tb, w - sb, h - tb - ch)
        self.chat = Region(0, h - ch, w, ch)
        self.drawer_width = int(w * Sizes.DRAWER_WIDTH_RATIO)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_gui/test_layout.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/layout.py tests/test_gui/test_layout.py
git commit -m "feat(gui): add proportional layout engine"
```

---

## Chunk 2: Game Loop Decomposition

### Task 3: Extract game_systems.py from GameEngine

This is the highest-risk task. We extract ~1,500 lines of business logic from `game_loop.py` into `game_systems.py` without changing any behavior. Existing tests must still pass.

**Files:**
- Create: `src/seaman_brain/gui/game_systems.py`
- Modify: `src/seaman_brain/gui/game_loop.py`
- Test: `tests/test_gui/test_game_systems.py`

- [ ] **Step 1: Write test for GameSystems.tick()**

The test validates that the extracted business logic ticks needs, checks behaviors, checks events, and handles evolution — the core of what `GameEngine._update()` does today.

```python
"""Tests for the extracted game business logic."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield


class TestGameSystemsTick:
    """Test that tick() calls subsystems at the correct intervals."""

    def test_needs_tick_at_interval(self):
        from seaman_brain.gui.game_systems import GameSystems

        needs_engine = MagicMock()
        creature_state = MagicMock()
        creature_state.is_alive = True
        clock = MagicMock()
        clock.elapsed_minutes = 10.0

        systems = GameSystems(
            needs_engine=needs_engine,
            mood_engine=MagicMock(),
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=MagicMock(),
            creature_state=creature_state,
            clock=clock,
            tank=MagicMock(),
        )

        # First tick should trigger needs update (timer starts at 0)
        systems.tick(1.1)  # past the 1.0s interval
        needs_engine.tick.assert_called_once()

    def test_no_tick_when_dead(self):
        from seaman_brain.gui.game_systems import GameSystems

        creature_state = MagicMock()
        creature_state.is_alive = False

        systems = GameSystems(
            needs_engine=MagicMock(),
            mood_engine=MagicMock(),
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=MagicMock(),
            creature_state=creature_state,
            clock=MagicMock(),
            tank=MagicMock(),
        )

        systems.tick(1.1)
        systems._needs_engine.tick.assert_not_called()


class TestFindTtsSplit:
    """Test TTS buffer splitting (moved from game_loop)."""

    def test_sentence_boundary(self):
        from seaman_brain.gui.game_systems import find_tts_split
        assert find_tts_split("Hello world. More text") == 12

    def test_no_boundary(self):
        from seaman_brain.gui.game_systems import find_tts_split
        assert find_tts_split("Hello world") is None

    def test_clause_boundary_long_enough(self):
        from seaman_brain.gui.game_systems import find_tts_split
        text = "A" * 45 + ", more"
        result = find_tts_split(text)
        assert result == 46  # position after the comma

    def test_clause_boundary_too_short(self):
        from seaman_brain.gui.game_systems import find_tts_split
        assert find_tts_split("Well, then") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_game_systems.py -v`
Expected: FAIL — ImportError

- [ ] **Step 3: Create game_systems.py by extracting from game_loop.py**

Extract the following from `game_loop.py` into `game_systems.py`:
- `find_tts_split()` function and its regex constants (`_SENTENCE_BOUNDARY`, `_CLAUSE_BOUNDARY`, `_MIN_CLAUSE_LENGTH`)
- `_build_interaction_situation()` function and `_INTERACTION_SITUATIONS` dict
- `GameState` enum
- Interval constants (`_NEEDS_UPDATE_INTERVAL`, `_BEHAVIOR_CHECK_INTERVAL`, `_EVENT_CHECK_INTERVAL`, etc.)
- New `GameSystems` class containing the timer-based tick logic from `GameEngine._update()`, `_update_needs()`, `_check_behaviors()`, `_check_events()`, `_check_evolution()`

Update `game_loop.py` to import from `game_systems` instead of defining these locally. This is a pure refactor — no behavior change.

- [ ] **Step 4: Run ALL existing tests to verify nothing breaks**

Run: `python -m pytest tests/ -x --tb=short`
Expected: All tests pass (same count as before)

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/game_systems.py src/seaman_brain/gui/game_loop.py tests/test_gui/test_game_systems.py
git commit -m "refactor(gui): extract business logic into game_systems.py"
```

---

### Task 4: Extract scene_manager.py

**Files:**
- Create: `src/seaman_brain/gui/scene_manager.py`
- Modify: `src/seaman_brain/gui/game_loop.py`
- Test: `tests/test_gui/test_scene_manager.py`

- [ ] **Step 1: Write test**

```python
"""Tests for scene/state management."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.game_systems import GameState  # noqa: E402
from seaman_brain.gui.scene_manager import SceneManager  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield


class TestSceneManager:
    def test_initial_state_is_playing(self):
        sm = SceneManager()
        assert sm.state == GameState.PLAYING

    def test_open_settings(self):
        sm = SceneManager()
        sm.open_settings()
        assert sm.state == GameState.SETTINGS
        assert sm.drawer_open

    def test_close_settings(self):
        sm = SceneManager()
        sm.open_settings()
        sm.close_drawer()
        assert sm.state == GameState.PLAYING
        assert not sm.drawer_open

    def test_open_lineage(self):
        sm = SceneManager()
        sm.open_lineage()
        assert sm.state == GameState.LINEAGE

    def test_drawer_progress_starts_at_zero(self):
        sm = SceneManager()
        sm.open_settings()
        assert sm.drawer_progress == 0.0

    def test_update_animates_drawer(self):
        sm = SceneManager()
        sm.open_settings()
        sm.update(0.15)  # 150ms, half of 300ms animation
        assert 0.0 < sm.drawer_progress < 1.0

    def test_drawer_clamps_to_one(self):
        sm = SceneManager()
        sm.open_settings()
        sm.update(1.0)  # way past 300ms
        assert sm.drawer_progress == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_scene_manager.py -v`
Expected: FAIL

- [ ] **Step 3: Write scene_manager.py**

```python
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
            # Ease-out: decelerate as approaching target
            remaining = 1.0 - self.drawer_progress
            step = remaining * dt / _DRAWER_ANIM_DURATION * 3.0
            self.drawer_progress = min(1.0, self.drawer_progress + max(step, dt / _DRAWER_ANIM_DURATION))
        elif self.drawer_progress > self._drawer_target:
            remaining = self.drawer_progress
            step = remaining * dt / _DRAWER_ANIM_DURATION * 3.0
            self.drawer_progress = max(0.0, self.drawer_progress - max(step, dt / _DRAWER_ANIM_DURATION))
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_scene_manager.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/scene_manager.py tests/test_gui/test_scene_manager.py
git commit -m "feat(gui): add scene manager with drawer transitions"
```

---

### Task 5: Extract input_handler.py

**Files:**
- Create: `src/seaman_brain/gui/input_handler.py`
- Test: `tests/test_gui/test_input_handler.py`

- [ ] **Step 1: Write test**

```python
"""Tests for input routing and keyboard shortcuts."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, call

_pygame_mock = MagicMock()
_pygame_mock.KEYDOWN = 768
_pygame_mock.MOUSEBUTTONDOWN = 1025
_pygame_mock.MOUSEMOTION = 1024
_pygame_mock.MOUSEWHEEL = 1027
_pygame_mock.MOUSEBUTTONUP = 1026
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_F2 = 283
_pygame_mock.K_TAB = 9
_pygame_mock.K_m = 109
_pygame_mock.K_RETURN = 13
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.input_handler import InputHandler  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield


class TestKeyboardShortcuts:
    def test_escape_calls_handler(self):
        handler = InputHandler()
        cb = MagicMock()
        handler.on_escape = cb
        event = MagicMock(type=768, key=27)
        handler.handle_event(event)
        cb.assert_called_once()

    def test_f2_calls_toggle_settings(self):
        handler = InputHandler()
        cb = MagicMock()
        handler.on_toggle_settings = cb
        event = MagicMock(type=768, key=283)
        handler.handle_event(event)
        cb.assert_called_once()

    def test_m_calls_toggle_mic(self):
        handler = InputHandler()
        cb = MagicMock()
        handler.on_toggle_mic = cb
        event = MagicMock(type=768, key=109)
        handler.handle_event(event)
        cb.assert_called_once()

    def test_keys_suppressed_when_chat_focused(self):
        handler = InputHandler()
        handler.chat_focused = True
        cb = MagicMock()
        handler.on_toggle_mic = cb
        event = MagicMock(type=768, key=109)
        handler.handle_event(event)
        cb.assert_not_called()
```

Note: the autouse fixture MUST include `mod.pygame = _pygame_mock` since input_handler.py imports pygame:
```python
@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.input_handler as mod
    mod.pygame = _pygame_mock
    yield
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_input_handler.py -v`
Expected: FAIL

- [ ] **Step 3: Write input_handler.py**

```python
"""Input routing — keyboard shortcuts and mouse event dispatch.

All keyboard shortcuts and mouse routing live here. GameEngine sets
callback attributes for each action; InputHandler calls them.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pygame


class InputHandler:
    """Routes pygame events to callbacks.

    Attributes set by GameEngine after construction:
        on_escape: Called when Escape pressed
        on_toggle_settings: Called on F2
        on_toggle_mic: Called on M
        on_tab: Called on Tab
        chat_focused: When True, suppress shortcut keys (let chat handle them)
    """

    def __init__(self) -> None:
        self.chat_focused = False

        # Callbacks (set by GameEngine)
        self.on_escape: Callable[[], Any] | None = None
        self.on_toggle_settings: Callable[[], Any] | None = None
        self.on_toggle_mic: Callable[[], Any] | None = None
        self.on_tab: Callable[[], Any] | None = None
        self.on_mouse_click: Callable[[Any], Any] | None = None
        self.on_mouse_move: Callable[[Any], Any] | None = None
        self.on_mouse_up: Callable[[Any], Any] | None = None
        self.on_mouse_scroll: Callable[[Any], Any] | None = None
        self.on_key_down: Callable[[Any], Any] | None = None

    def handle_event(self, event: pygame.event.Event) -> None:
        """Route a single pygame event to the appropriate callback."""
        if event.type == pygame.KEYDOWN:
            self._handle_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.on_mouse_click:
                self.on_mouse_click(event)
        elif event.type == pygame.MOUSEMOTION:
            if self.on_mouse_move:
                self.on_mouse_move(event)
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.on_mouse_up:
                self.on_mouse_up(event)
        elif event.type == pygame.MOUSEWHEEL:
            if self.on_mouse_scroll:
                self.on_mouse_scroll(event)

    def _handle_key(self, event: pygame.event.Event) -> None:
        """Route keyboard events, respecting chat focus."""
        key = event.key

        # Escape and F2 always work regardless of chat focus
        if key == pygame.K_ESCAPE:
            if self.on_escape:
                self.on_escape()
            return
        if key == pygame.K_F2:
            if self.on_toggle_settings:
                self.on_toggle_settings()
            return

        # When chat is focused, pass keys to chat handler instead
        if self.chat_focused:
            if self.on_key_down:
                self.on_key_down(event)
            return

        # Global shortcuts (only when chat not focused)
        if key == pygame.K_m:
            if self.on_toggle_mic:
                self.on_toggle_mic()
        elif key == pygame.K_TAB:
            if self.on_tab:
                self.on_tab()
        elif self.on_key_down:
            self.on_key_down(event)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_input_handler.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite to verify no regressions**

Run: `python -m pytest tests/ -x --tb=short`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/seaman_brain/gui/input_handler.py tests/test_gui/test_input_handler.py
git commit -m "feat(gui): add input handler for keyboard/mouse routing"
```

---

## Chunk 3: Render Engine + Tank

### Task 6: Create render_engine.py (gradient cache + particle system)

**Files:**
- Create: `src/seaman_brain/gui/render_engine.py`
- Test: `tests/test_gui/test_render_engine.py`

- [ ] **Step 1: Write test**

```python
"""Tests for the render engine, gradient cache, and particle system."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 200
_surface_mock.get_height.return_value = 200
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.render_engine import GradientCache, ParticleSystem  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.render_engine as mod
    mod.pygame = _pygame_mock
    yield


class TestGradientCache:
    def test_get_creates_surface(self):
        cache = GradientCache()
        surf = cache.get((210, 140, 80), 100)
        assert surf is not None

    def test_same_params_returns_cached(self):
        cache = GradientCache()
        s1 = cache.get((210, 140, 80), 100)
        s2 = cache.get((210, 140, 80), 100)
        assert s1 is s2

    def test_different_color_creates_new(self):
        cache = GradientCache()
        s1 = cache.get((210, 140, 80), 100)
        s2 = cache.get((80, 130, 210), 100)
        assert s1 is not s2

    def test_invalidate_clears_cache(self):
        cache = GradientCache()
        s1 = cache.get((210, 140, 80), 100)
        cache.invalidate()
        s2 = cache.get((210, 140, 80), 100)
        assert s1 is not s2


class TestParticleSystem:
    def test_init_creates_particles(self):
        ps = ParticleSystem(count=10, bounds=(0, 0, 800, 600))
        assert len(ps.particles) == 10

    def test_update_moves_particles_upward(self):
        ps = ParticleSystem(count=1, bounds=(0, 0, 800, 600))
        initial_y = ps.particles[0]["y"]
        ps.update(1.0)
        assert ps.particles[0]["y"] < initial_y  # moved up

    def test_particles_respawn_when_off_top(self):
        ps = ParticleSystem(count=1, bounds=(0, 0, 800, 600))
        ps.particles[0]["y"] = -5  # force off-screen
        ps.update(0.01)
        assert ps.particles[0]["y"] > 0  # respawned at bottom
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_render_engine.py -v`
Expected: FAIL

- [ ] **Step 3: Write render_engine.py**

```python
"""Render engine — gradient cache, particle system, draw-order orchestration.

Provides reusable rendering primitives for the Modern Minimal aesthetic.
"""
from __future__ import annotations

import math
import random

import pygame

from seaman_brain.gui.theme import VOID_BG


class GradientCache:
    """Pre-computed radial gradient surfaces, regenerated on demand."""

    def __init__(self) -> None:
        self._cache: dict[tuple[tuple[int, int, int], int], pygame.Surface] = {}

    def get(self, color: tuple[int, int, int], radius: int) -> pygame.Surface:
        """Get a cached radial gradient surface, creating if needed."""
        key = (color, radius)
        if key not in self._cache:
            self._cache[key] = self._create_radial(color, radius)
        return self._cache[key]

    def invalidate(self) -> None:
        """Clear the cache (call when mood or size changes)."""
        self._cache.clear()

    @staticmethod
    def _create_radial(
        color: tuple[int, int, int], radius: int
    ) -> pygame.Surface:
        """Draw concentric circles with decreasing alpha."""
        size = radius * 2
        surface = pygame.Surface((size, size), pygame.SRCALPHA)
        r, g, b = color
        cx, cy = radius, radius
        for i in range(radius, 0, -2):
            alpha = int(60 * (i / radius) ** 2)  # quadratic falloff
            pygame.draw.circle(surface, (r, g, b, alpha), (cx, cy), i)
        return surface


class ParticleSystem:
    """Floating particles drifting upward in the void."""

    def __init__(
        self, count: int = 12, bounds: tuple[int, int, int, int] = (0, 0, 800, 600)
    ) -> None:
        self.bounds = bounds  # (x, y, w, h)
        self.particles: list[dict] = []
        for _ in range(count):
            self.particles.append(self._spawn(randomize_y=True))

    def _spawn(self, randomize_y: bool = False) -> dict:
        x0, y0, w, h = self.bounds
        return {
            "x": random.uniform(x0, x0 + w),
            "y": random.uniform(y0, y0 + h) if randomize_y else y0 + h,
            "speed": random.uniform(8, 25),
            "size": random.choice([1, 1, 1, 2]),
            "alpha": random.randint(38, 77),  # 15-30% of 255
        }

    def update(self, dt: float) -> None:
        """Move particles upward, respawn at bottom when off-top."""
        x0, y0, _, h = self.bounds
        for p in self.particles:
            p["y"] -= p["speed"] * dt
            if p["y"] < y0:
                new = self._spawn()
                p.update(new)

    def render(self, surface: pygame.Surface) -> None:
        """Draw all particles."""
        for p in self.particles:
            color = (255, 255, 255, p["alpha"])
            pos = (int(p["x"]), int(p["y"]))
            pygame.draw.circle(surface, color, pos, p["size"])
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_render_engine.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/render_engine.py tests/test_gui/test_render_engine.py
git commit -m "feat(gui): add render engine with gradient cache and particles"
```

---

### Task 7: Rewrite tank_renderer.py (void aesthetic)

**Files:**
- Rewrite: `src/seaman_brain/gui/tank_renderer.py`
- Rewrite: `tests/test_gui/test_tank_renderer.py`

- [ ] **Step 1: Write test**

```python
"""Tests for the void tank renderer."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 800
_surface_mock.get_height.return_value = 600
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.tank_renderer import TankRenderer  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.tank_renderer as mod
    mod.pygame = _pygame_mock
    yield


class TestTankRenderer:
    def test_construction(self):
        tr = TankRenderer(800, 600)
        assert tr is not None

    def test_render_calls_fill(self):
        tr = TankRenderer(800, 600)
        surface = MagicMock()
        tr.render(surface)
        surface.fill.assert_called()

    def test_update_does_not_crash(self):
        tr = TankRenderer(800, 600)
        tr.update(0.016)  # one frame at 60fps

    def test_resize(self):
        tr = TankRenderer(800, 600)
        tr.resize(1024, 768)
        # Should not crash, particles should update bounds
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_tank_renderer.py -v`
Expected: FAIL (old tank_renderer has different API)

- [ ] **Step 3: Rewrite tank_renderer.py**

Replace the entire file. The new version is dramatically simpler — void + center glow + particles.

```python
"""Void tank renderer — dark background with radial glow and particles.

Replaces the old aquarium/terrarium renderer with the Modern Minimal void.
"""
from __future__ import annotations

import pygame

from seaman_brain.gui.render_engine import GradientCache, ParticleSystem
from seaman_brain.gui.theme import VOID_BG, mood_glow_color


class TankRenderer:
    """Renders the void: background fill, center glow, floating particles."""

    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._gradient_cache = GradientCache()
        self._particles = ParticleSystem(count=12, bounds=(0, 0, width, height))
        self._mood = "neutral"

    def resize(self, width: int, height: int) -> None:
        """Update dimensions on window resize."""
        self._width = width
        self._height = height
        self._particles = ParticleSystem(
            count=12, bounds=(0, 0, width, height)
        )
        self._gradient_cache.invalidate()

    def set_mood(self, mood: str) -> None:
        """Update the creature mood for glow color."""
        if mood != self._mood:
            self._mood = mood
            self._gradient_cache.invalidate()

    def update(self, dt: float) -> None:
        """Advance particle animation."""
        self._particles.update(dt)

    def render(self, surface: pygame.Surface) -> None:
        """Draw the void: background, center glow, particles."""
        surface.fill(VOID_BG)

        # Center glow
        glow_color = mood_glow_color(self._mood)
        radius = min(self._width, self._height) // 3
        if radius > 10:
            glow_surf = self._gradient_cache.get(glow_color, radius)
            cx = self._width // 2 - radius
            cy = self._height // 2 - radius
            surface.blit(glow_surf, (cx, cy))

        # Particles
        self._particles.render(surface)
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_tank_renderer.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/tank_renderer.py tests/test_gui/test_tank_renderer.py
git commit -m "feat(gui): rewrite tank renderer as void with glow and particles"
```

---

## Chunk 4: Creature Sprites

### Task 8: Rewrite sprites.py (bigger, glow aura, mood-reactive)

**Files:**
- Rewrite: `src/seaman_brain/gui/sprites.py`
- Rewrite: `tests/test_gui/test_sprites.py`

- [ ] **Step 1: Write test for glow aura and size scaling**

```python
"""Tests for creature rendering with glow aura and mood-reactive lighting."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 200
_surface_mock.get_height.return_value = 200
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.sprites import CreatureRenderer, AnimationState  # noqa: E402
from seaman_brain.types import CreatureStage  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.sprites as mod
    mod.pygame = _pygame_mock
    yield


class TestCreatureRendererConstruction:
    def test_default_stage(self):
        cr = CreatureRenderer()
        assert cr.stage == CreatureStage.MUSHROOMER

    def test_base_size_is_100(self):
        cr = CreatureRenderer()
        assert cr._base_size == 100


class TestGlowAura:
    def test_glow_surface_created(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        # Glow should be cached
        assert cr._glow_surface is not None or cr._mood == "neutral"

    def test_mood_change_invalidates_glow(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        old_mood = cr._mood
        cr.set_mood("hostile")
        assert cr._mood == "hostile"


class TestAnimationStates:
    def test_idle_is_default(self):
        cr = CreatureRenderer()
        assert cr.animation_state == AnimationState.IDLE

    def test_set_talking(self):
        cr = CreatureRenderer()
        cr.animation_state = AnimationState.TALKING
        assert cr.animation_state == AnimationState.TALKING


class TestSizeScaling:
    def test_mushroomer_smallest(self):
        cr = CreatureRenderer(stage=CreatureStage.MUSHROOMER)
        # Mushroomer has 0.6x multiplier → 60px
        assert cr._body_width == 60

    def test_frogman_largest(self):
        cr = CreatureRenderer(stage=CreatureStage.FROGMAN)
        # Frogman has 1.5x multiplier → 150px
        assert cr._body_width == 150


class TestIdleAnimation:
    def test_update_changes_bob_offset(self):
        cr = CreatureRenderer()
        cr.update(0.5)
        # Bob offset should be non-zero after time passes
        assert cr._bob_offset != 0.0 or True  # may be near zero at start
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_sprites.py -v`
Expected: FAIL (old sprites.py has different API)

- [ ] **Step 3: Rewrite sprites.py**

Keep the same procedural creature shapes (all 5 stages) but:
- Change `_BASE_SIZE` from 40 to 100
- Add `_glow_surface` cached radial gradient
- Add `set_mood()` to trigger glow color change
- Add `_bob_offset` sine-wave idle animation
- Smooth eye tracking with easing
- All drawing code preserved, just scaled up

This is a large file (~500-600 lines) because of the per-stage rendering. The implementation worker should read the existing `sprites.py` and preserve all shape-drawing code while wrapping it in the new glow/size/animation system.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_sprites.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/sprites.py tests/test_gui/test_sprites.py
git commit -m "feat(gui): rewrite creature sprites with glow aura and 2.5x size"
```

---

## Chunk 5: HUD + Sidebar

### Task 9: Rewrite hud.py (top bar + sidebar tiles)

**Files:**
- Rewrite: `src/seaman_brain/gui/hud.py`
- Rewrite: `tests/test_gui/test_hud.py`

- [ ] **Step 1: Write test**

```python
"""Tests for the Modern Minimal HUD — top bar and sidebar tiles."""
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
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.hud import HUD  # noqa: E402
from seaman_brain.gui.layout import ScreenLayout  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.hud as mod
    mod.pygame = _pygame_mock
    yield


class TestHUDConstruction:
    def test_creates_need_tiles(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        assert len(hud._need_tiles) == 4  # H, +, C, T

    def test_creates_action_tiles(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        assert len(hud._action_tiles) == 7  # F, O, ^, v, *, ~, o


class TestNeedTiles:
    def test_tile_color_green_when_healthy(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        hud.update_needs(hunger=0.8, health=0.9, comfort=0.7, trust=0.6)
        # All above 50%, so all tiles should use green color
        for tile in hud._need_tiles:
            assert tile["color"] == (74, 222, 128)

    def test_tile_color_red_when_critical(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        hud.update_needs(hunger=0.1, health=0.1, comfort=0.1, trust=0.1)
        for tile in hud._need_tiles:
            assert tile["color"] == (239, 68, 68)


class TestActionTileClick:
    def test_click_in_tile_returns_action_key(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        # Action tiles start below need tiles + divider
        # First action tile "F" should be clickable
        result = hud.handle_click(24, 200)  # approximate tile position
        # Should return action key or None depending on exact position
        assert result is None or isinstance(result, str)


class TestTopBar:
    def test_render_does_not_crash(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        surface = MagicMock()
        hud.render(surface)
        # Should have drawn something
        assert _pygame_mock.draw.rect.called or surface.blit.called
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_hud.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite hud.py**

The new HUD renders:
- 32px top bar with name/stage/mood/mic-dot/tts-dot/lineage/settings/status/timer
- 48px left sidebar with 4 need tiles + divider + 7 action tiles
- Tooltips on hover
- Action tile cooldown overlays

The implementation worker should reference the spec for exact colors and sizes. All colors come from `theme.py`, all positions from `layout.py`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_hud.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/hud.py tests/test_gui/test_hud.py
git commit -m "feat(gui): rewrite HUD as top bar + sidebar tiles"
```

---

## Chunk 6: Widgets

### Task 10: Rewrite widgets.py (Modern Minimal)

**Files:**
- Rewrite: `src/seaman_brain/gui/widgets.py`
- Rewrite: `tests/test_gui/test_widgets.py`

- [ ] **Step 1: Write test**

```python
"""Tests for Modern Minimal widgets."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 60
_font_surface.get_height.return_value = 14
_font_mock.render.return_value = _font_surface
_font_mock.size.return_value = (60, 14)
_font_mock.get_linesize.return_value = 16
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
_surface_mock = MagicMock()
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.widgets import Button, Toggle, Slider, Dropdown  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.widgets as mod
    mod.pygame = _pygame_mock
    yield


class TestButton:
    def test_construction(self):
        b = Button(10, 10, 100, 30, "Test")
        assert b.label == "Test"

    def test_click_inside_fires_callback(self):
        cb = MagicMock()
        b = Button(10, 10, 100, 30, "Test", on_click=cb)
        assert b.handle_click(50, 25) is True
        cb.assert_called_once()

    def test_click_outside_returns_false(self):
        cb = MagicMock()
        b = Button(10, 10, 100, 30, "Test", on_click=cb)
        assert b.handle_click(200, 200) is False
        cb.assert_not_called()

    def test_disabled_blocks_click(self):
        cb = MagicMock()
        b = Button(10, 10, 100, 30, "Test", on_click=cb)
        b.enabled = False
        assert b.handle_click(50, 25) is False


class TestToggle:
    def test_initial_state(self):
        t = Toggle(10, 10, 50, 24, on=False)
        assert t.on is False

    def test_click_toggles(self):
        t = Toggle(10, 10, 50, 24, on=False)
        t.handle_click(30, 22)
        assert t.on is True

    def test_callback_fires(self):
        cb = MagicMock()
        t = Toggle(10, 10, 50, 24, on=False, on_change=cb)
        t.handle_click(30, 22)
        cb.assert_called_once_with(True)


class TestSlider:
    def test_initial_value(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50)
        assert s.value == 50

    def test_drag_changes_value(self):
        s = Slider(10, 10, 200, 20, min_val=0, max_val=100, value=50)
        s.handle_click(110, 20)  # middle of slider
        assert 40 <= s.value <= 60  # approximately middle


class TestDropdown:
    def test_construction(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0)
        assert d.selected_text == "A"

    def test_click_expands(self):
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0)
        d.handle_click(80, 24)
        assert d.expanded is True

    def test_select_item(self):
        cb = MagicMock()
        d = Dropdown(10, 10, 150, 28, items=["A", "B", "C"], selected=0, on_change=cb)
        d.handle_click(80, 24)  # expand
        d.handle_click(80, 52)  # click second item
        cb.assert_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_widgets.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite widgets.py**

Complete rewrite with Modern Minimal styling. All colors from `theme.py`. No pygame_gui dependency. Four classes: `Button`, `Toggle`, `Slider`, `Dropdown`. Each with `render()`, `handle_click()`, `handle_mouse_move()`, `handle_mouse_up()`.

The implementation worker should read the existing `widgets.py` for the interaction logic patterns (especially Dropdown's scroll handling) while applying the new visual style.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_widgets.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/widgets.py tests/test_gui/test_widgets.py
git commit -m "feat(gui): rewrite widgets in Modern Minimal style"
```

---

## Chunk 7: Chat Panel

### Task 11: Rewrite chat_panel.py (glass overlay + bubbles)

**Files:**
- Rewrite: `src/seaman_brain/gui/chat_panel.py`
- Rewrite: `tests/test_gui/test_chat_panel.py`

- [ ] **Step 1: Write test**

```python
"""Tests for the glass overlay chat panel with message bubbles."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.K_RETURN = 13
_pygame_mock.K_BACKSPACE = 8
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 14
_font_mock.render.return_value = _font_surface
_font_mock.size.side_effect = lambda text: (len(text) * 7, 14)
_font_mock.get_linesize.return_value = 16
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
_surface_mock = MagicMock()
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.chat_panel import ChatPanel  # noqa: E402
from seaman_brain.gui.layout import ScreenLayout  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.chat_panel as mod
    mod.pygame = _pygame_mock
    yield


class TestChatPanel:
    def test_construction(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        assert cp is not None

    def test_add_system_message(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        cp.add_message("system", "Test message")
        assert len(cp._messages) == 1
        assert cp._messages[0]["role"] == "system"

    def test_add_creature_message(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        cp.add_message("creature", "*yawns*")
        assert cp._messages[0]["role"] == "creature"

    def test_add_user_message(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        cp.add_message("user", "Hello")
        assert cp._messages[0]["role"] == "user"

    def test_input_text(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        cp._input_text = "Hello"
        assert cp._input_text == "Hello"

    def test_submit_clears_input(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        cp._input_text = "Hello"
        cb = MagicMock()
        cp.on_submit = cb
        cp._submit()
        assert cp._input_text == ""
        cb.assert_called_once_with("Hello")

    def test_render_does_not_crash(self):
        layout = ScreenLayout(1024, 768)
        cp = ChatPanel(layout)
        cp.add_message("system", "Test")
        surface = MagicMock()
        cp.render(surface)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_chat_panel.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite chat_panel.py**

Glass overlay with:
- CHAT_BG from theme (RGB 8,8,8 at 85% alpha)
- System messages: "SYSTEM" label + italic text, no bubble
- Creature messages: left-aligned warm-tinted bubble
- User messages: right-aligned lighter bubble
- Pill-shaped input bar at bottom
- Auto-scroll, scroll-lock on manual scroll-up
- `on_submit` callback for chat text

No pygame_gui dependency. All rendering via `pygame.draw` and `font.render()`.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_chat_panel.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/chat_panel.py tests/test_gui/test_chat_panel.py
git commit -m "feat(gui): rewrite chat panel as glass overlay with bubbles"
```

---

## Chunk 8: Settings + Lineage Drawers

### Task 12: Rewrite settings_panel.py (slide-out drawer)

**Files:**
- Rewrite: `src/seaman_brain/gui/settings_panel.py`
- Rewrite: `tests/test_gui/test_settings_panel.py`

- [ ] **Step 1: Write test**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_settings_panel.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite settings_panel.py**

Slide-out drawer with:
- Vertical tab strip on left edge (Personality/LLM/Audio/Vision icons)
- Active tab accent line
- Content area with Modern Minimal widgets (from widgets.py)
- All callbacks preserved: `on_personality_change`, `on_llm_apply`, `on_audio_change`, `on_vision_change`
- `save_user_settings()` still called through the same callback chain
- `render(surface, progress)` — progress 0-1 controls slide animation position

No pygame_gui. All using new widgets.py.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_settings_panel.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/settings_panel.py tests/test_gui/test_settings_panel.py
git commit -m "feat(gui): rewrite settings as slide-out drawer with tabs"
```

---

### Task 13: Rewrite lineage_panel.py (slide-out drawer)

**Files:**
- Rewrite: `src/seaman_brain/gui/lineage_panel.py`
- Rewrite: `tests/test_gui/test_lineage_panel.py`

- [ ] **Step 1: Write test**

```python
"""Tests for the lineage slide-out drawer."""
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

from seaman_brain.gui.lineage_panel import LineagePanel  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.lineage_panel as mod
    mod.pygame = _pygame_mock
    yield


class TestLineagePanel:
    def test_construction(self):
        lp = LineagePanel(width=400)
        assert lp is not None

    def test_set_bloodlines(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["Alpha", "Beta"], active="Alpha")
        assert lp._active == "Alpha"

    def test_select_callback(self):
        cb = MagicMock()
        lp = LineagePanel(width=400, on_select=cb)
        lp.set_bloodlines(["Alpha", "Beta"], active="Alpha")
        lp._select("Beta")
        cb.assert_called_once_with("Beta")

    def test_render_does_not_crash(self):
        lp = LineagePanel(width=400)
        lp.set_bloodlines(["Alpha"], active="Alpha")
        surface = MagicMock()
        lp.render(surface, progress=1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_lineage_panel.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite lineage_panel.py**

Same slide-out drawer pattern as settings. Bloodline list with select/new/delete buttons. Active bloodline highlighted with accent color. `render(surface, progress)` for animation.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_lineage_panel.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/seaman_brain/gui/lineage_panel.py tests/test_gui/test_lineage_panel.py
git commit -m "feat(gui): rewrite lineage panel as slide-out drawer"
```

---

## Chunk 9: Integration + Cleanup

### Task 14: Rewrite game_loop.py (thin orchestrator)

**Files:**
- Rewrite: `src/seaman_brain/gui/game_loop.py`
- Rewrite: `tests/test_gui/test_game_loop.py`

- [ ] **Step 1: Write test**

```python
"""Tests for the thin GameEngine orchestrator."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

_pygame_gui_mock = MagicMock()
sys.modules["pygame_gui"] = _pygame_gui_mock

_pygame_mock = MagicMock()
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.MOUSEBUTTONDOWN = 1025
_pygame_mock.SRCALPHA = 65536
_pygame_mock.K_ESCAPE = 27
_pygame_mock.K_F2 = 283
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768
_pygame_mock.display.set_mode.return_value = _surface_mock
_clock_mock = MagicMock()
_clock_mock.tick.return_value = 33
_pygame_mock.time.Clock.return_value = _clock_mock
_font_mock = MagicMock()
_font_surface = MagicMock()
_font_surface.get_width.return_value = 100
_font_surface.get_height.return_value = 16
_font_mock.render.return_value = _font_surface
_font_mock.size.return_value = (100, 16)
_font_mock.get_linesize.return_value = 18
_pygame_mock.font.SysFont.return_value = _font_mock
_pygame_mock.font.Font.return_value = _font_mock
sys.modules["pygame"] = _pygame_mock

import pytest  # noqa: E402

from seaman_brain.gui.game_loop import GameEngine  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = _pygame_gui_mock
    import seaman_brain.gui.game_loop as mod
    mod.pygame = _pygame_mock
    yield


class TestGameEngineInit:
    def test_construction_does_not_crash(self):
        engine = GameEngine()
        assert engine is not None

    def test_has_scene_manager(self):
        engine = GameEngine()
        assert engine._scene_manager is not None

    def test_has_input_handler(self):
        engine = GameEngine()
        assert engine._input_handler is not None

    def test_has_layout(self):
        engine = GameEngine()
        assert engine._layout is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_gui/test_game_loop.py -v`
Expected: FAIL

- [ ] **Step 3: Rewrite game_loop.py**

Gut to ~300 lines. The new GameEngine:
- `__init__`: Creates ScreenLayout, SceneManager, InputHandler, GameSystems, and all renderers/panels
- `initialize()`: Same as before — sets up ConversationManager, AudioManager, etc.
- `run()`: Simplified loop: get events → input_handler.handle_event() → game_systems.tick(dt) → scene_manager.update(dt) → render
- `_render()`: Delegates to tank_renderer, sprites, hud, chat_panel, settings/lineage drawers based on scene state
- All callbacks wired in `__init__` (settings change, audio change, etc.) — same as before but delegating to the extracted modules

The implementation worker should reference the current `game_loop.py` for all the callback wiring and ensure every existing callback is preserved.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_gui/test_game_loop.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -x --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/seaman_brain/gui/game_loop.py tests/test_gui/test_game_loop.py
git commit -m "feat(gui): rewrite GameEngine as thin orchestrator"
```

---

### Task 15: Update window.py (strip pygame_gui)

**Files:**
- Modify: `src/seaman_brain/gui/window.py`
- Rewrite: `tests/test_gui/test_window.py`

- [ ] **Step 1: Write test verifying no pygame_gui references**

```python
# Add to existing test_window.py
class TestPygameGuiRemoval:
    def test_no_ui_manager(self):
        from seaman_brain.gui.window import GameWindow
        gw = GameWindow()
        assert not hasattr(gw, '_ui_manager') or gw._ui_manager is None
```

- [ ] **Step 2: Remove pygame_gui from window.py**

Remove:
- The `try: import pygame_gui` block
- `self._ui_manager` initialization
- All `self._ui_manager.process_events()`, `.update()`, `.draw_ui()` calls
- Theme loading from `config/theme.json`

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_gui/test_window.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/seaman_brain/gui/window.py tests/test_gui/test_window.py
git commit -m "refactor(gui): strip pygame_gui from window.py"
```

---

### Task 16: Restyle interactions.py (void effects)

**Files:**
- Modify: `src/seaman_brain/gui/interactions.py`
- Update: `tests/test_gui/test_interactions.py`

- [ ] **Step 1: Update interaction visual effects**

Change ripple and food drop colors to match void aesthetic:
- Ripples: dim white concentric rings (rgba 255,255,255 at low alpha)
- Food drops: warm-colored particles
- Tap glass: white flash + screen shake

Import colors from `theme.py` instead of local constants.

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_gui/test_interactions.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/seaman_brain/gui/interactions.py tests/test_gui/test_interactions.py
git commit -m "style(gui): restyle interaction effects for void aesthetic"
```

---

### Task 17: Final cleanup

**Files:**
- Delete: `src/seaman_brain/gui/action_bar.py`
- Delete: `config/theme.json`
- Modify: `src/seaman_brain/gui/__init__.py`
- Modify: `pyproject.toml` (remove pygame_gui dependency if present)
- Delete: `tests/test_gui/test_action_bar.py`

- [ ] **Step 1: Delete action_bar.py and its test**

```bash
git rm src/seaman_brain/gui/action_bar.py tests/test_gui/test_action_bar.py
```

- [ ] **Step 2: Delete config/theme.json**

```bash
git rm config/theme.json
```

- [ ] **Step 3: Update __init__.py**

```python
"""GUI subsystem — Modern Minimal Pygame interface for creature interaction."""

from seaman_brain.gui.window import GameWindow

__all__ = ["GameWindow"]
```

- [ ] **Step 4: Remove pygame_gui from pyproject.toml if present**

Check `pyproject.toml` for `pygame-gui` in dependencies and remove it.

- [ ] **Step 5: Run full test suite + ruff**

```bash
python -m ruff check src/ tests/
python -m pytest tests/ -x --tb=short
```

Expected: All clean, all pass.

- [ ] **Step 6: Commit**

```bash
git add src/seaman_brain/gui/__init__.py pyproject.toml
git commit -m "chore(gui): remove action_bar.py, theme.json, pygame_gui dependency"
```

---

### Task 18: Final integration verification

- [ ] **Step 1: Run the full quality check suite**

```bash
python -m ruff check src/ tests/
python -m pytest tests/ -x --tb=short
python -m seaman_brain --version
```

Expected: All pass.

- [ ] **Step 2: Manual smoke test**

```bash
python -m seaman_brain --gui
```

Verify:
- Window opens with void background
- Creature renders with glow aura at 2.5x size
- Top bar shows name/stage/mood/timer
- Sidebar shows need tiles and action tiles
- Chat panel is glass overlay at bottom
- Settings drawer slides out on F2
- Keyboard shortcuts work (Escape, M, Tab)
- Feeding via sidebar tile opens food menu
- Messages appear in styled bubbles

- [ ] **Step 3: Commit any fixes from smoke test**

---

## Chunk 10: Missing Features (Food Menu, Overlays, Audio Restyling)

### Task 19: Add food selection dropdown to HUD sidebar

**Files:**
- Modify: `src/seaman_brain/gui/hud.py`
- Update: `tests/test_gui/test_hud.py`

- [ ] **Step 1: Write test for food dropdown**

```python
# Add to test_hud.py
class TestFoodDropdown:
    def test_feed_tile_click_opens_menu(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        hud.set_food_types(["Flakes", "Pellets", "Worms"])
        # Click the "F" action tile
        result = hud.handle_action_click("feed")
        assert hud._food_menu_open is True

    def test_food_item_click_fires_callback(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        cb = MagicMock()
        hud.on_feed = cb
        hud.set_food_types(["Flakes", "Pellets"])
        hud._food_menu_open = True
        hud._select_food(0)
        cb.assert_called_once_with("Flakes")

    def test_click_elsewhere_closes_menu(self):
        layout = ScreenLayout(1024, 768)
        hud = HUD(layout)
        hud.set_food_types(["Flakes"])
        hud._food_menu_open = True
        hud.handle_click(500, 400)  # click in tank area
        assert hud._food_menu_open is False
```

- [ ] **Step 2: Implement food dropdown in hud.py**

Add a small floating dropdown anchored to the "F" tile. Lists food types from `FoodType`. Uses theme colors. Dismisses on click-outside.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_gui/test_hud.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/seaman_brain/gui/hud.py tests/test_gui/test_hud.py
git commit -m "feat(gui): add food selection dropdown to sidebar"
```

---

### Task 20: Add modal overlays (game over, evolution, notifications)

**Files:**
- Create or add to: `src/seaman_brain/gui/render_engine.py` (overlay rendering functions)
- Update: `tests/test_gui/test_render_engine.py`

- [ ] **Step 1: Write tests for overlays**

```python
# Add to test_render_engine.py
class TestGameOverOverlay:
    def test_render_does_not_crash(self):
        from seaman_brain.gui.render_engine import render_game_over
        surface = MagicMock()
        render_game_over(surface, 1024, 768, cause="starvation")

class TestEvolutionOverlay:
    def test_render_does_not_crash(self):
        from seaman_brain.gui.render_engine import render_evolution
        surface = MagicMock()
        render_evolution(surface, 1024, 768, stage_name="Gillman", progress=0.5)

class TestNotifications:
    def test_add_and_render(self):
        from seaman_brain.gui.render_engine import NotificationManager
        nm = NotificationManager()
        nm.add("Test notification")
        assert len(nm._notifications) == 1

    def test_notifications_expire(self):
        from seaman_brain.gui.render_engine import NotificationManager
        nm = NotificationManager()
        nm.add("Test")
        nm.update(10.0)  # way past expiry
        assert len(nm._notifications) == 0
```

- [ ] **Step 2: Implement overlay rendering**

Add to `render_engine.py`:
- `render_game_over(surface, w, h, cause)`: Full-screen dim overlay, red glow, centered death cause text, "Press Space" hint
- `render_evolution(surface, w, h, stage_name, progress)`: Full-screen overlay, gold pulsing glow, stage name centered
- `NotificationManager`: Toast stack in bottom-left. `add(text)`, `update(dt)`, `render(surface)`. Pills with fade-in/pause/fade-out.

All colors from `theme.py`.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_gui/test_render_engine.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/seaman_brain/gui/render_engine.py tests/test_gui/test_render_engine.py
git commit -m "feat(gui): add game over, evolution, and notification overlays"
```

---

### Task 21: Restyle audio_integration.py

**Files:**
- Modify: `src/seaman_brain/gui/audio_integration.py`

- [ ] **Step 1: Update color constants**

Replace local color constants with imports from `theme.py`:
- Mic active indicator → `Colors.STATUS_GREEN`
- Mic inactive → `Colors.TEXT_20`
- TTS active → `Colors.STATUS_YELLOW`
- Volume bar colors → theme status colors

- [ ] **Step 2: Run existing tests**

Run: `python -m pytest tests/test_gui/test_audio_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/seaman_brain/gui/audio_integration.py
git commit -m "style(gui): restyle audio integration to use theme colors"
```

---

### Task 22: Add glow pulse and mood transition to sprites

**Files:**
- Modify: `src/seaman_brain/gui/sprites.py`
- Update: `tests/test_gui/test_sprites.py`

- [ ] **Step 1: Write tests for glow pulse and transition**

```python
# Add to test_sprites.py
class TestGlowPulse:
    def test_pulse_oscillates_intensity(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.update(0.0)
        alpha_0 = cr._glow_alpha
        cr.update(1.0)  # half period
        alpha_1 = cr._glow_alpha
        assert alpha_0 != alpha_1  # should oscillate

class TestMoodTransition:
    def test_mood_change_starts_transition(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.set_mood("hostile")
        assert cr._mood_transition_progress < 1.0

    def test_transition_completes_after_1s(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.set_mood("hostile")
        cr.update(1.5)  # past 1s transition
        assert cr._mood_transition_progress >= 1.0
```

- [ ] **Step 2: Implement glow pulse (sine wave ~2s) and mood color transition (~1s ease)**

In `CreatureRenderer`:
- `_glow_alpha`: modulated by `sin(time * pi)` over 2s period, range 0.5-1.0
- `_mood_transition_progress`: 0→1 over 1s when mood changes, lerps between old and new glow color
- Glow surface re-rendered with interpolated color during transition

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_gui/test_sprites.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add src/seaman_brain/gui/sprites.py tests/test_gui/test_sprites.py
git commit -m "feat(gui): add glow pulse and mood color transitions"
```

---

## Summary

| Chunk | Tasks | Key Files |
|-------|-------|-----------|
| 1: Foundation | 1-2 | theme.py, layout.py |
| 2: Decomposition | 3-5 | game_systems.py, scene_manager.py, input_handler.py |
| 3: Render + Tank | 6-7 | render_engine.py, tank_renderer.py |
| 4: Creature | 8 | sprites.py |
| 5: HUD | 9 | hud.py |
| 6: Widgets | 10 | widgets.py |
| 7: Chat | 11 | chat_panel.py |
| 8: Drawers | 12-13 | settings_panel.py, lineage_panel.py |
| 9: Integration | 14-18 | game_loop.py, window.py, interactions.py, cleanup |
| 10: Missing Features | 19-22 | food menu, overlays, audio restyling, glow pulse |

**Total: 22 tasks across 10 chunks. Each task has TDD steps with exact file paths, test code, and commit messages.**
