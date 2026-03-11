# GUI Rewrite: Modern Minimal — Design Spec

## Overview

Full rewrite of the Seaman Reborn Pygame GUI (14 files, ~10,600 lines). Replaces the current "terminal debug" aesthetic with a Modern Minimal design: dark void, creature glow, glassmorphism overlays, thin sidebar. Same Pygame backend, completely new presentation layer. Backend systems (conversation, audio, creature, needs, behavior) remain untouched.

## Visual Language

- **Palette**: Near-black void (#08080f). UI surfaces at 2-5% white opacity. Accent colors only for status indicators and creature glow.
- **Typography**: System font. Small caps for labels (9-10px). Regular for content (11px). High letter-spacing on section headers.
- **Borders**: 1px at ~6% white opacity. No hard edges — elements blend into the void.
- **Depth**: Radial gradients and soft shadows, not flat fills. Semi-transparent tinted overlays (SRCALPHA surfaces) for glass-panel effect — not Gaussian blur, just colored transparency. Radial gradients pre-computed on cached surfaces (regenerated only on mood/size change, not per-frame).
- **Animation**: All state transitions ease — no instant jumps. Smooth interpolation everywhere.
- **Small caps**: Simulated via uppercase text rendered at 9-10px size (Pygame has no OpenType small-caps support).
- **Letter-spacing**: Custom per-character rendering for spaced headers (render each glyph individually with manual advance).

## Screen Layout

```
┌──────────────────────────────────────────────────┐
│ SEAMAN  Mushroomer  ◆ Neutral     ● 00:38  ⚙    │ 32px top bar
├────┬─────────────────────────────────────────────┤
│ H  │                                             │
│ +  │                                             │
│ C  │           THE VOID                          │
│ T  │                                             │
│────│        ·                                    │
│ F  │              ✦ creature ✦                   │ 48px sidebar
│ O  │                  (glow)                     │
│ ^  │                                ·            │
│ v  │      ·                                      │
│ *  │                                             │
│ ~  │                                             │
│ o  │                                             │
├────┴─────────────────────────────────────────────┤
│ ┌─glass overlay──────────────────────────────┐   │
│ │ SYSTEM  Something stirs within you...      │   │ ~130px chat
│ │ ┌─warm bubble─────────────────────────┐    │   │
│ │ │ Seaman: *dozes off briefly*         │    │   │
│ │ └─────────────────────────────────────┘    │   │
│ │  (━━━━━━━ Say something... ━━━━━━━━━━━━)   │   │
│ └────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────┘
```

### Top Bar (32px)

- Left: "SEAMAN" (bold, 11px, 90% white) · Stage name (30% white) · Mood diamond (colored, 60% opacity)
- Center-right: Mic indicator dot (green=active, dim=off) · TTS indicator dot
- Right: Lineage icon (20% white, click opens lineage drawer) · Settings gear icon (20% white) · Status dot (6px circle, green=#4ade80 with glow shadow) · Session timer (25% white)
- Background: 3% white, 1px bottom border at 6% white

### Left Sidebar (48px)

Upper section — Need indicator tiles (24x24px, 6px border-radius):
- H (Hunger), + (Health), C (Comfort), T (Trust)
- Color-coded borders: green (#4ade80) > 50%, yellow (#f59e0b) 25-50%, red (#ef4444) < 25%
- Background: 10% of the status color
- Tooltip on hover: full name + numeric value

Divider line (1px, 6% white)

Lower section — Action tiles (same 24x24px):
- F (Feed), O (Aerate), ^ (Temp+), v (Temp-), * (Clean), ~ (Drain), o (Tap)
- Default: 4% white bg, 6% white border
- Hover: brighten to 8% white bg
- Active click: accent glow
- Cooldown: radial wipe overlay with countdown number

Sidebar background: 2% white, 1px right border at 4% white

### Tank Area (fills remaining space)

- Background: #08080f (the void)
- Center glow: radial gradient (rgba 20,40,80 at 25% opacity, fading to transparent at ~65% radius)
- Particles: 10-15 dots (1-2px), slowly drifting upward, random spawn positions, slight opacity variation (15-30% white)
- No substrate, no plants, no rocks, no water line — just the void and the creature

### Chat Panel (~130px, glass overlay)

- Background: RGB(8,8,8) at 85% alpha (~217/255) — dark but translucent, tank dimly visible through
- 1px top border at 6% white
- Positioned at bottom, overlays the tank

Message types:
- **System**: No bubble. "SYSTEM" label (9px, uppercase, 30% white). Message text (11px, 50% white, italic).
- **Seaman**: Left-aligned bubble. Warm tint (rgba 200,160,120 at 6% bg, 10% border). "Seaman" label (9px, 50% warm). Content (10px, 70% warm, italic for actions).
- **User**: Right-aligned bubble. Slightly brighter (5% white bg, 8% border).

Behavior:
- Auto-scroll to newest message
- Scroll-lock if user scrolls up manually
- Messages fade opacity as they age (optional, subtle)

Input bar:
- Pill-shaped (14px border-radius), 28px height
- 4% white bg, 8% white border
- Placeholder: "Say something..." at 20% white
- Positioned 8px from bottom, 12px horizontal margin

### Settings Drawer

- Slides in from right edge, ~40% of screen width
- Semi-transparent background (5% white)
- Smooth slide animation (ease-out, ~300ms)
- Tank dims slightly when drawer is open (dark overlay on tank)

Tab navigation:
- Vertical icon strip on the left edge of the drawer
- 4 tabs: Personality, LLM, Audio, Vision
- Active tab: bright accent line on left edge
- Inactive: 20% white icon

Content area:
- Scrollable per tab
- All controls (sliders, dropdowns, toggles) use Modern Minimal styling: dark fills, thin borders, accent colors on interaction
- Personality tab: trait sliders with labels
- LLM tab: model dropdown, temperature slider
- Audio tab: TTS/STT provider dropdowns, device dropdowns, volume sliders, AEC/barge-in toggles
- Vision tab: camera selection, source toggle

### Lineage Panel

- Same slide-out drawer pattern as Settings (from right, ~40% width)
- Bloodline list with select/new/delete
- Active bloodline highlighted with accent color

## Creature Rendering

- **Size**: Base body width 100px (was 40px). Per-stage multipliers still apply (0.6x-1.5x → 60-150px range).
- **Shapes**: Same procedural shapes per stage (Mushroomer/Gillman/Podfish/Tadman/Frogman) — no redesign
- **Glow aura**: Pre-computed radial gradient surface (cached, regenerated on mood change). 2x creature size. Color shifts with mood:
  - content/amused/curious → warm gold (rgba 230,190,80)
  - neutral/philosophical → soft amber (rgba 210,140,80)
  - sardonic/irritated → cool orange (rgba 210,160,60)
  - hostile → red pulse (rgba 210,80,60)
  - sleeping → dim purple (rgba 140,100,180)
  - sad → cool blue (rgba 80,130,210)
- **Glow pulse**: Sine wave intensity oscillation, ~2s period
- **Idle animation**: Gentle bob/sway even when idle (not static)
- **Eye tracking**: Smooth mouse following (already exists, refine easing)
- **Mood-reactive**: Glow color transition animates smoothly (~1s ease)

## Keyboard Shortcuts

Preserved from existing code, routed through `input_handler.py`:
- **Escape**: Close active drawer (settings/lineage), or exit
- **Tab**: Toggle chat panel focus
- **F2**: Toggle settings drawer
- **M**: Toggle microphone
- **F/O/^/v/etc**: Action shortcuts (when no text input focused)

## Food Selection

Clicking the "F" (Feed) sidebar tile opens a small floating dropdown menu anchored to the tile. Lists available food types. Clicking a food type triggers the feed interaction. Clicking elsewhere dismisses it. Same FoodType data as current InteractionManager.

## Modal Overlays

These overlay the entire screen with a dimmed void:
- **Game Over**: Death cause text centered, "Press Space to restart" below. Dim red glow.
- **Evolution Celebration**: Stage name with pulsing glow effect, ~3 second display. Gold accent.
- **Notifications**: Toast messages stack in bottom-left. Fade in, pause, fade out. 20% white bg pill with text.

## Interaction Effects (Void Style)

- **Ripples**: Dim white concentric rings expanding from click point, fading to transparent
- **Food drops**: Small warm-colored particles falling with gravity, dissolving on contact
- **Tap glass**: Brief white flash at tap point, subtle screen shake (1-2px offset, 200ms)

## Architecture

### Files to Create (new)

| File | Purpose | Est. Lines |
|------|---------|------------|
| `theme.py` | Centralized colors, sizes, opacity constants, mood-color mapping, font initialization (single source) | ~200 |
| `layout.py` | Proportional layout engine (percentages, handles resize) | ~200 |
| `scene_manager.py` | Game state machine (PLAYING, SETTINGS, LINEAGE), transition animations | ~150 |
| `render_engine.py` | Draw order orchestration, layer management, particle system, gradient cache | ~300 |
| `input_handler.py` | Keyboard/mouse event routing to subsystems | ~150 |
| `game_systems.py` | Business logic extracted from GameEngine: needs ticking, mood updates, behavior/event checks, death/evolution, autonomous remarks, TTS sentence splitting, STT debouncing | ~400 |

### Files to Rewrite

| File | Changes |
|------|---------|
| `game_loop.py` | Gut to ~300 lines. Core loop + GameEngine.__init__ wiring. Delegates rendering to render_engine, input to input_handler, state to scene_manager. Business logic (needs ticking, mood updates, behavior checks, event checks, death/evolution, autonomous remarks, TTS sentence splitting, STT debouncing) moves to a new `game_systems.py` (~400 lines) that GameEngine calls via `systems.tick(dt)`. |
| `widgets.py` | Complete rewrite in Modern Minimal style. Drop pygame_gui dual-path. One clean implementation: Button, Slider, Toggle, Dropdown. |
| `tank_renderer.py` | Simplify to void + radial gradient + particle system. Remove aquarium/terrarium complexity. |
| `sprites.py` | Same creature shapes. Add glow aura system, smoother animation, mood-reactive lighting. |
| `hud.py` | Becomes thin top bar + left sidebar tiles. Remove old bar rendering. |
| `chat_panel.py` | Glass overlay with bubble messages. Drop pygame_gui UITextBox path. |
| `settings_panel.py` | Slide-out drawer with vertical tabs. Rewrite all controls using new widgets. |
| `action_bar.py` | Merge into left sidebar (part of hud.py). Remove as standalone file. |
| `lineage_panel.py` | Restyle as slide-out drawer (same pattern as settings). |

### Files to Keep (logic intact, reskin visuals)

| File | Changes |
|------|---------|
| `audio_integration.py` | Keep all logic. Update visual elements (volume indicators, status dots) to match theme. |
| `device_utils.py` | No changes. Hardware enumeration is UI-agnostic. |
| `interactions.py` | Keep interaction logic. Restyle visual effects (ripples, food drops) to match void aesthetic. |
| `window.py` | Keep async bridge and event loop. Remove all pygame_gui UIManager calls (process_events, update, draw_ui). Update init to use new theme/layout. |

### Dependencies Dropped

- `pygame_gui` — no longer needed. All widgets hand-rolled in Modern Minimal style.
- Dual-path rendering code throughout — single clean path everywhere.

### Dependencies Unchanged

- `pygame` — still the rendering backend
- All backend systems: ConversationManager, AudioManager, NeedsSystem, BehaviorEngine, CreatureState, etc.

## Migration Strategy

1. **Game loop decomposition FIRST** — extract `game_systems.py`, `scene_manager.py`, `input_handler.py` from `game_loop.py`. This is highest-risk, so do it before visual changes. Existing tests validate nothing breaks.
2. **Theme + Layout** — `theme.py`, `layout.py` (visual foundation, no rendering changes yet)
3. **Render engine + Tank** — `render_engine.py`, `tank_renderer.py` (void + gradient + particles)
4. **Creature** — `sprites.py` rewrite (bigger, glow aura, mood-reactive)
5. **Sidebar + HUD** — `hud.py` rewrite (top bar + sidebar tiles, replaces old HUD + action_bar)
6. **Widgets** — `widgets.py` rewrite (Modern Minimal Button/Slider/Toggle/Dropdown)
7. **Chat panel** — `chat_panel.py` (glass overlay + bubbles, uses new widgets)
8. **Settings drawer** — `settings_panel.py` (slide-out, vertical tabs, uses new widgets)
9. **Lineage panel** — `lineage_panel.py` (same drawer pattern)
10. **Window + cleanup** — strip pygame_gui from `window.py`, delete `action_bar.py`, remove pygame_gui dependency, update `__init__.py`

Each step is independently testable. Backend interfaces never change.

## Test Strategy

- **Delete**: All 15 existing test files in `tests/test_gui/` are replaced
- **Rewrite alongside each step**: Each new/rewritten file gets a corresponding test file
- **New test files**: `test_theme.py`, `test_layout.py`, `test_scene_manager.py`, `test_render_engine.py`, `test_input_handler.py`, `test_game_systems.py`
- **Same mock pattern**: Module-level `sys.modules["pygame"] = mock` + autouse fixture re-install (as documented in CLAUDE.md)
- **Settings persistence**: Test that `save_user_settings()` is still called correctly through the new settings drawer

## Cleanup

- Delete `config/theme.json` (was for pygame_gui, no longer needed)
- Delete `action_bar.py` (merged into hud.py sidebar)
- Remove `pygame_gui` from dependencies in pyproject.toml
- Update `__init__.py` exports

## What's NOT Changing

- Entry points (`python -m seaman_brain --gui`)
- Config loading (TOML → Pydantic)
- ConversationManager, AudioManager, NeedsSystem, BehaviorEngine
- Creature state, evolution, genetics, lineage logic
- TTS/STT pipeline
- WebSocket API bridge
- CLI mode
