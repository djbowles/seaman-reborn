# Seaman Reborn — UE5 Integration Handoff

## Project Status

The Python "brain" backend is **feature-complete**: 2155 tests passing, ruff clean, all 52 user stories implemented across 14 subpackages (llm, personality, memory, creature, conversation, cli, audio, environment, needs, behavior, gui, api, vision).

- **Repo**: https://github.com/djbowles/seaman-reborn (private)
- **Branch**: `ralph/ai-brain-core` (all work), `main` (base)
- **Entry points**: `python -m seaman_brain` (terminal), `--gui` (Pygame), `--api` (WebSocket server)
- **Hardware**: RTX 5090 (32GB VRAM), Ollama with qwen3-coder:30b + all-minilm:l6-v2 + qwen3-vl:8b

## What Exists (Python Brain)

| Subsystem | Status | Key Files |
|-----------|--------|-----------|
| LLM inference (Ollama, Anthropic, OpenAI) | Done | `llm/` — factory pattern, streaming support |
| 5-stage personality evolution | Done | `personality/` — traits, constraints, prompt_builder |
| Hybrid RAG memory (episodic + semantic) | Done | `memory/` — LanceDB vectors, embedding pipeline |
| Creature state + genome + genetics | Done | `creature/` — state, evolution, genome, lineage |
| Biological needs + death mechanics | Done | `needs/` — hunger, comfort, health, stimulation |
| Autonomous behavior + mood engine | Done | `behavior/` — mood calculation, idle behaviors |
| Environment simulation | Done | `environment/` — real-time clock, tank state |
| Conversation orchestration | Done | `conversation/` — context assembly, full pipeline |
| TTS/STT audio | Done | `audio/` — pyttsx3 TTS, speech recognition |
| Pygame GUI (playable) | Done | `gui/` — tank, sprites, chat, HUD, action bar |
| **FastAPI WebSocket bridge** | **Done** | `api/` — **this is the UE5 connection point** |

## WebSocket API Contract (for UE5 Client)

**Server**: `ws://127.0.0.1:8420/ws/brain` | REST: `http://127.0.0.1:8420/api/`

### REST Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/health` | Liveness check (`{"status": "ok", "initialized": bool}`) |
| GET | `/api/state` | Full state snapshot (polling alternative) |
| POST | `/api/reset` | Reset creature to MUSHROOMER defaults |

### WebSocket Protocol (v1.0.0)

**Client → Server:**
```json
{
  "type": "input",
  "text": "Hello Seaman",
  "protocol_version": "1.0.0"
}
```

**Server → Client (response to input):**
```json
{
  "type": "response",
  "text": "Go away, human.",
  "state": {
    "creature_state": {
      "stage": "gillman", "age": 3600.0, "mood": "sardonic",
      "trust_level": 0.35, "hunger": 0.2, "health": 0.95,
      "comfort": 0.8, "interaction_count": 42
    },
    "needs": { "hunger": 0.2, "comfort": 0.8, "health": 0.95, "stimulation": 0.5 },
    "tank": {
      "temperature": 24.5, "cleanliness": 0.9, "oxygen_level": 0.95,
      "water_level": 0.8, "environment_type": "aquarium"
    },
    "active_traits": {
      "cynicism": 0.7, "wit": 0.8, "patience": 0.3, "curiosity": 0.6,
      "warmth": 0.2, "verbosity": 0.5, "formality": 0.2, "aggression": 0.4
    },
    "mood": "sardonic",
    "current_stage": "gillman"
  }
}
```

**Server → Client (periodic state diff, every 500ms on change):**
```json
{
  "type": "state_update",
  "state": { "..." },
  "changed_fields": ["creature_state.mood", "needs.hunger"]
}
```

**Server → Client (event notification — defined, not yet triggered):**
```json
{
  "type": "event",
  "event_type": "evolution_ready",
  "severity": "critical",
  "message": "Seaman has evolved to Gillman!",
  "effects": { "new_stage": "gillman" }
}
```

**Server → Client (error):**
```json
{
  "type": "error",
  "message": "Invalid JSON: ..."
}
```

## UE5 Integration Plan (from PRD)

The PRD (`Seaman 2_ Modern Tech Blueprint.txt`) defines UE5 as the rendering frontend that connects to the Python brain via WebSocket. The architecture splits cleanly:

### Python Brain (backend — this repo)
- LLM cognition, personality, memory, creature lifecycle
- Serves state over WebSocket at `:8420`
- Runs as a background process alongside UE5

### UE5 Client (frontend — new repo needed)
- Procedural creature rendering + animation
- Tank environment with Niagara fluid simulation
- Player I/O (microphone, webcam, controller)
- Connects to Python brain's WebSocket API

### PRD-Specified UE5 Components

| Component | PRD Tech Stack | Notes |
|-----------|---------------|-------|
| **Creature rendering** | Skeletal mesh + morph targets | Procedural mesh for 5 evolutionary stages |
| **Locomotion** | Control Rig + Full Body IK | Spine IK chain, procedural swimming/walking |
| **Fluid simulation** | Niagara 3D Flip Hose | Water displacement, wake, bubbles |
| **Lip sync** | MetaHuman Lip Sync or NVIDIA Audio2Face-3D | Phoneme→viseme on non-humanoid mesh |
| **STT** | Epic Runtime Speech Recognizer plugin | Offline, packaged in executable |
| **TTS** | Piper/Kokoro via Runtime TTS plugin (CPU) | Frees GPU for rendering |
| **Webcam emotion** | Google MediaPipe (CPU) | 468 facial landmarks, hand gestures |
| **Tank environment** | Niagara + physics | Aquarium → terrarium transition |

## What Needs to Happen Next

### Phase 1: Python API Hardening (before UE5 work)

1. **Wire up EventBroadcaster** — `streaming.py` has a full per-channel subscription system (MOOD, NEEDS, EVOLUTION, BEHAVIOR, TANK, DEATH) that isn't connected to `BrainServer` yet. Wire it in so UE5 gets push notifications for lifecycle events.

2. **Add streaming LLM responses** — Currently buffers full response before sending. UE5 needs token-by-token streaming for "creature is talking" animation sync.

3. **Add subscribe/unsubscribe messages** — Let UE5 client opt into specific event channels over WebSocket.

4. **Trigger EventNotifications** — Hook creature subsystems (evolution, death, mood shifts) to broadcast `event` messages.

5. **Add action messages** — UE5 needs to send actions beyond text chat:
   - `feed` — trigger feeding
   - `tap_glass` — poke the creature
   - `adjust_temperature` — tank controls
   - `drain_tank` / `fill_tank` — environment transitions

### Phase 2: UE5 Project Setup

1. **Create UE5 project** (separate repo) with WebSocket client plugin
2. **Implement WebSocket client** — connect to `ws://127.0.0.1:8420/ws/brain`
3. **Build state-driven UI** — parse `BrainStateSnapshot` to drive creature visuals
4. **Prototype creature mesh** — basic skeletal mesh with morph targets for MUSHROOMER stage
5. **Tank environment** — basic aquarium with Niagara water

### Phase 3: Creature Rendering Pipeline

1. **5-stage creature models** — MUSHROOMER → GILLMAN → PODFISH → TADMAN → FROGMAN
2. **Control Rig IK** — procedural locomotion driven by AI decisions
3. **Lip sync integration** — route TTS audio through Audio2Face or MetaHuman plugin
4. **Stage transition animations** — smooth interpolation between evolutionary forms

### Phase 4: Player Sensory Input

1. **Microphone → STT** — Epic Runtime Speech Recognizer → send as `InputMessage`
2. **TTS playback** — receive creature text → Piper/Kokoro → speaker output
3. **Webcam tracking** — MediaPipe face/hand tracking → inject as environmental context
4. **Physical interactions** — tap glass, feed, temperature controls

## Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                    PLAYER                            │
│         Microphone │ Webcam │ Controller             │
└────────────┬───────┴────────┴──────┬────────────────┘
             │                       │
             ▼                       ▼
┌─────────────────────────────────────────────────────┐
│              UNREAL ENGINE 5 CLIENT                  │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │ STT      │ │ MediaPipe│ │ Niagara Fluid     │   │
│  │ (offline)│ │ (CPU)    │ │ Simulation        │   │
│  └────┬─────┘ └────┬─────┘ └───────────────────┘   │
│       │             │                                │
│  ┌────▼─────────────▼─────────────────────────┐     │
│  │         WebSocket Client                    │     │
│  │    ws://127.0.0.1:8420/ws/brain            │     │
│  └────────────────┬───────────────────────────┘     │
│                   │                                  │
│  ┌────────────────▼───────────────────────────┐     │
│  │  State-Driven Rendering                     │     │
│  │  • Creature mesh + morph targets            │     │
│  │  • Control Rig IK locomotion                │     │
│  │  • Audio2Face lip sync                      │     │
│  │  • Piper/Kokoro TTS (CPU)                   │     │
│  └─────────────────────────────────────────────┘     │
└──────────────────────┬──────────────────────────────┘
                       │ WebSocket (JSON)
                       ▼
┌─────────────────────────────────────────────────────┐
│              PYTHON BRAIN (this repo)                 │
│              FastAPI @ :8420                          │
│                                                      │
│  ┌──────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │ Ollama   │ │ LanceDB  │ │ Creature State    │   │
│  │ LLM      │ │ Vectors  │ │ + Genome          │   │
│  └──────────┘ └──────────┘ └───────────────────┘   │
│  ┌──────────┐ ┌──────────┐ ┌───────────────────┐   │
│  │Personality│ │ Needs    │ │ Behavior Engine   │   │
│  │+ Prompts │ │ System   │ │ + Mood            │   │
│  └──────────┘ └──────────┘ └───────────────────┘   │
└─────────────────────────────────────────────────────┘
```

## Open Questions

1. **Separate repo for UE5?** — PRD implies single integrated project, but Python brain + UE5 are fundamentally different build systems. Recommend separate repos with shared protocol spec.

2. **LLM location** — PRD suggests TensorRT-LLM in-engine for minimum latency. Current arch uses Ollama externally. Could move to in-engine llama.cpp for production, keep Ollama for dev.

3. **TTS ownership** — PRD puts TTS in UE5 (Piper/Kokoro plugin). Current Python brain has pyttsx3 TTS. For UE5 integration, Python sends text only and UE5 handles voice synthesis + lip sync.

4. **Webcam data injection** — Need a new message type for UE5 to send player emotion/gesture data to the brain for inclusion in LLM prompts.

5. **Creature mesh pipeline** — Need concept art / 3D modeling for the 5 evolutionary stages before UE5 rendering work begins.

## GUI Overhaul

The `--gui` mode now launches `GameEngine` (was launching bare `GameWindow` with no subsystems).

**What was done:**
- Entry point fixed: `__main__.py` now uses `GameEngine` instead of bare `GameWindow`
- Window status overlay only renders when no subsystem renderers are registered
- Right-side **ActionBar** (160px panel) with 6 large labeled buttons: Feed, Temp+, Temp-, Clean, Drain, Tap
- **HUD** bars all stacked in left column (no right-column overlap with action bar), compact mode default
- Settings hotkey changed **F10 → F1** (Windows intercepts F10 for menu bar activation)
- Settings button changed from `[F10]` text to visible `[Settings]` button with background/border
- **Chat panel**: increased background opacity (180→220), added "Chat" header bar, added `[Send]` button
- `TankRenderer.set_render_area()` for flexible layout
- Old tiny interaction buttons disabled when ActionBar is active (spatial tank clicks still work)

## Settings Stabilization & Device Settings (latest)

### Settings Crash Fixes
- **Vision source change**: `_on_vision_change` now creates `VisionBridge` on-demand when source changes to "webcam"/"tank" (was only triggering on `"enabled"` key which the panel never sent)
- **Thread-safe model list**: `_load_model_list_async` callback now queues model list via `_pending_model_list`, applied on the main game loop thread (fixes pygame thread-safety)
- **Try/except wrappers**: All settings callbacks (`_on_personality_change`, `_on_llm_apply`, `_on_audio_change`, `_on_vision_change`) wrapped to prevent crashes from propagating to game loop
- **Change detection**: Vision source dropdown only fires callback on actual value change (not re-selection of same source)

### Granular Device Settings
- **Config**: `AudioConfig` has new `audio_output_device` and `audio_input_device` fields (empty = system default)
- **New `gui/device_utils.py`**: Enumerates audio output/input devices (via `sounddevice`), webcams (via `cv2` + `pygrabber`), and TTS voices (via `pyttsx3`) — all gracefully handle missing libraries
- **Audio tab**: 3 new dropdowns — Output Device, Input Device, TTS Voice
- **Vision tab**: New Camera dropdown for webcam index selection
- **`config/default.toml`**: Added `audio_output_device` and `audio_input_device` fields

### Lineage Manager
- **New overlay**: `gui/lineage_panel.py` — accessible via HUD `[Lineage]` button or **F2** shortcut
- **Bloodline discovery**: Scans `data/saves/` for subdirectories containing `creature.json`
- **Migration**: On first launch, if `data/saves/creature.json` exists at root (old flat layout), auto-migrates into `data/saves/default/` subdirectory
- **Save structure**: Each bloodline is a named subdirectory (`data/saves/<name>/creature.json`)
- **Active tracking**: `data/saves/_active.txt` stores the active bloodline name
- **Panel features**: List view with name/stage/generation, New (creates fresh creature), Load (switches active), Delete (with confirmation, can't delete active)
- **`creature/persistence.py`**: Added `BloodlineInfo` dataclass, `migrate_flat_saves()`, `list_bloodlines()`, `get_active_bloodline()`, `set_active_bloodline()` class methods
- **`gui/hud.py`**: Added `[Lineage]` button with `lineage_rect` for click detection
- **`gui/game_loop.py`**: Added `GameState.LINEAGE`, F2 key binding, ESC closes lineage, mouse routing to lineage panel

### New Tests (66 tests, 4 files)
- `tests/test_gui/test_settings_crash.py` — 12 tests for crash fixes, thread safety, change detection
- `tests/test_gui/test_device_utils.py` — 10 tests for device enumeration with mocked backends (generic + friendly name paths)
- `tests/test_gui/test_lineage_panel.py` — 25 tests for panel lifecycle, new/load/delete bloodlines
- `tests/test_creature/test_persistence_bloodlines.py` — 19 tests for migration, list_bloodlines, multi-directory saves

## Device Enumeration & Vision Fixes (latest)

### Audio Device Enumeration
- **`sounddevice` added to base dependencies** (`pyproject.toml`) — was missing, causing only "System Default" to appear
- **WASAPI filtering**: On Windows, filters to WASAPI host API to avoid duplicate entries from MME/DirectSound/WDM-KS (same physical device shows 4x otherwise)
- **Skip aliases**: Filters out Windows system aliases ("Microsoft Sound Mapper", "Primary Sound Driver")
- Detected hardware: Speakers (CA DacMagic 200M 2.0), Realtek Digital Output, Speakers (Portacapture X6), LG ULTRAFINE, Microphone (Portacapture X6)

### Webcam Friendly Names
- **`pygrabber` added to vision optional dependencies** — uses DirectShow via `comtypes` (already installed) to enumerate video capture device names
- `list_webcams()` now shows real device names (e.g. "OBSBOT Virtual Camera") instead of generic "Camera 0"
- Falls back to "Camera N" if `pygrabber` is not installed or DirectShow enumeration fails

### Camera Index Mapping Fix
- **Bug**: Camera dropdown was passing dropdown list index as `webcam_index` instead of actual OpenCV device index (off-by-one since "System Default" occupies index 0)
- **Fix**: Settings panel now stores `_cam_device_indices` mapping from `list_webcams()` tuples and maps correctly on selection

### "Look Now" Button Wired Up
- **Was broken**: `_on_vision_change` handler had no case for `key == "look_now"` — button did nothing
- **Now works**: Creates `VisionBridge` on-demand if needed, calls `trigger_observation()`, polls for result
- **Settings mode fix**: `_vision_bridge._check_pending()` now runs even during settings overlay (was blocked by early return in `_update()`)
- Shows "Vision source is off" notification if no source is configured
- Result displayed in both notification toast and settings panel "Last:" text

### Embedding Model
- `all-minilm:l6-v2` was not pulled in Ollama — semantic memory embeddings were returning 404
- Model is now pulled and available

### Crash Diagnostics
- All `logger.error()` calls in event handlers, update/render callbacks now include `exc_info=True` for full stack traces in `data/seaman.log`
- Top-level crash catcher added to `__main__.py` — logs `CRITICAL` with full traceback before re-raising
- Run with `--debug` for DEBUG-level logging to diagnose intermittent issues

**Remaining GUI issues to investigate:**
- Verify creature sprite is visible in the resized tank area (864px wide)
- Verify chat panel Send button click works end-to-end with ConversationManager
- Chat panel input may need focus management (currently always captures keys when visible)
- Action bar drain/fill button may need visual state indicator (drained vs filled)
- No food selection submenu from ActionBar — Feed button auto-picks first available food type
- HUD compact mode may be too small on high-DPI displays — may need scaling support
- Lineage manager: rename bloodline not yet implemented (panel has list/new/load/delete only)
- Lineage manager: switching bloodline does not yet reinitialize ConversationManager with new save path (callbacks are stubs that log + notify)
- Device dropdowns: selecting a device updates config but doesn't reinitialize TTS/STT providers at runtime
- Intermittent crash on settings exit — traceback logging now in place, needs reproduction to diagnose

## Vision + Audio Pipeline Fixes (482bcd2)

Fixed 6 bugs preventing vision and audio from functioning at runtime:

1. **Webcam index off-by-one** — `list_webcams()` returned `idx+1` instead of actual OpenCV index; OBSBOT camera (device 0) was requested as device 1 and failed. Fixed to return actual indices, System Default uses `-1` sentinel.
2. **Webcam index not propagated** — Changing camera in settings only updated config; live `VisionBridge` kept old `WebcamCapture`. Added `set_webcam_index()` that recreates the capture object.
3. **TTS WAV header parsed as raw audio** — `pygame.mixer.Sound(buffer=wav_bytes)` treated WAV file bytes as raw PCM. Fixed to `Sound(file=io.BytesIO(wav_bytes))`.
4. **TTS routing** — Changed `play_voice()` to prefer `AudioManager.speak()` (pyttsx3 native audio output) instead of broken synthesize→mixer path. TTS now produces audible output.
5. **STT results discarded** — `_listen_async()` was fire-and-forget with no callback. Changed to continuous listen loop with `on_stt_result` callback that auto-submits transcriptions to chat.
6. **STT input device ignored** — `sr.Microphone()` always used system default. Added `_resolve_mic_index()` to map device names to PyAudio indices.

### Confirmed Working
- TTS produces audible speech (uses pyttsx3 SAPI5 native output)
- Webcam captures work with correct device index
- Vision pipeline ("Look Now" button) functional

### Fixed Runtime Bugs
- **Settings UI not reflecting saved values** — Device dropdowns (output, input, TTS voice, camera) always initialized with `selected_index=0` ("System Default") regardless of saved config. Fixed `settings_panel.py` to find and select the saved value on build. Added `_find_saved_index()` helper.
- **TTS voice not applied** — `Pyttsx3TTSProvider._create_engine()` matched `config.tts_voice` against `voice.id` (Windows registry key) but the settings panel stored `voice.name` (display name). Fixed to match against both `voice.name` and `voice.id`.
- **STT non-functional** — Two fixes: (1) PyAudio was missing (installed via `pip install pyaudio`); (2) When STT enabled at runtime, `NoopSTTProvider` wasn't replaced. Added `_try_upgrade_stt()` to `AudioManager` that recreates the STT provider when `stt_enabled` is set to True while using a noop provider.
- **Creature age stuck at 0.0** — `creature_state.age` was never incremented anywhere in the game loop or needs engine. Creature died with `age=0.0` in death records. Fixed: `game_loop.py` now increments `self._creature_state.age += elapsed` in the periodic needs-update block.
- **Empty shutdown error log** — `window.py` logged `TimeoutError` with `%s` format which produces an empty string (TimeoutError has no message). Fixed: changed to `%r` format with `exc_info=True` for full traceback.

## Minor Code Issues (non-blocking)

- `callable` lowercase type hints in `needs/feeding.py:93`, `needs/care.py:90`, `gui/chat_panel.py:77`
- `needs/system.py:128` uses `creature_state.comfort` as proxy for stimulation
- `behavior/autonomous.py:225` takes `creature_state: dict[str, Any]` instead of typed `CreatureState`
