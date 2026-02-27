# Seaman Reborn — UE5 Integration Handoff

## Project Status

The Python "brain" backend is **feature-complete**: 2278 tests passing, ruff clean, all 52 user stories implemented across 14 subpackages (llm, personality, memory, creature, conversation, cli, audio, environment, needs, behavior, gui, api, vision).

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
| TTS/STT audio | Done | `audio/` — pyttsx3/Kokoro TTS, speech_recognition/Faster-Whisper STT |
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

### Phase 1: Python API Hardening (before UE5 work) — COMPLETE

All Phase 1 items are implemented and tested (4 test files in `tests/test_api/`).

1. **Wire up EventBroadcaster** — Done. `streaming.py` per-channel subscription system wired into `BrainServer`. Event notifications broadcast for death, mood shifts, needs changes, evolution, and autonomous behaviors. ✓

2. **Add streaming LLM responses** — Done. Token-by-token streaming via `process_input_stream()` in ConversationManager, relayed through WebSocket `response_stream` messages. ✓

3. **Add subscribe/unsubscribe messages** — Done. Client sends `subscribe`/`unsubscribe` messages with channel names. Server manages per-connection channel sets. ✓

4. **Trigger EventNotifications** — Done. Creature subsystems (evolution, death, mood shifts, needs, behavior events) fire `event` messages through the broadcaster. ✓

5. **Add action messages** — Done. 7 action types dispatched: `feed`, `tap_glass`, `adjust_temperature`, `drain_tank`, `fill_tank`, `clean_tank`, `toggle_aerator`. ✓

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

## Forge-Audit Optimizations (latest)

Four-phase infrastructure upgrade addressing LLM token misconfiguration, VRAM scheduling, neural TTS, and local STT.

### Phase 1: LLM Token Configuration Fix (CRITICAL)

**Problem**: `LLMConfig.max_tokens = 512` was mapped to Ollama's `num_ctx` (context window), meaning the 30B model could only see ~512 tokens total — less than the system prompt alone. Additionally, `num_predict` (output token limit) was never set, so Qwen3's invisible thinking tokens (~700) consumed the default budget before generating actual responses.

**Fix**: Split the single `max_tokens` field into proper Ollama-specific parameters:

| File | Change |
|------|--------|
| `config.py` (LLMConfig) | Added `context_window: int = 8192` (Ollama `num_ctx`) and `max_response_tokens: int = 4096` (Ollama `num_predict`). Kept `max_tokens = 512` for OpenAI/Anthropic backward compat. |
| `llm/ollama_provider.py` | `num_ctx` uses `cfg.context_window`, added `num_predict` to both `chat()` and `stream()` options |
| `conversation/manager.py` | `ContextAssembler(max_tokens=cfg.llm.context_window)` (was `cfg.llm.max_tokens`) |
| `config/default.toml` | Added `context_window = 8192`, `max_response_tokens = 4096` under `[llm]` |

**Critical follow-up fix — empty autonomous responses**: Expanding context from 512→8192 caused episodic ASSISTANT messages to now fit in context. When `generate_autonomous_remark()` appended the situation directive to the system prompt, the assembled context was SYSTEM + ASSISTANT messages with no trailing USER message. Qwen3 returns empty content in this pattern. **Fixed** by sending the situation directive as a trailing USER ChatMessage after context assembly instead of injecting into the system prompt.

### Phase 2: Kokoro Neural TTS Provider

**Problem**: pyttsx3 uses Windows SAPI5 which sounds robotic. Kokoro is a neural TTS producing natural speech (~2GB VRAM).

| File | Change |
|------|--------|
| `audio/tts.py` | New `KokoroTTSProvider` class — lazy model loading, 24kHz output, voice/speed config. Factory updated: kokoro → pyttsx3 fallback. |
| `config.py` (AudioConfig) | Added `tts_speed: float = 1.0` (Kokoro speed multiplier 0.5-2.0) |
| `gui/device_utils.py` | Added `list_kokoro_voices()` with known voice IDs; `list_tts_voices()` now accepts `provider` param |
| `gui/settings_panel.py` | Passes `config.audio.tts_provider` to `list_tts_voices()` |
| `pyproject.toml` | Added `tts-neural = ["kokoro>=0.9.0", "soundfile>=0.12.0"]` optional deps |

**Activation**: Set `tts_provider = "kokoro"` in config and `pip install seaman-brain[tts-neural]`. Falls back to pyttsx3 if kokoro not installed.

### Phase 3: Faster-Whisper Local STT Provider

**Problem**: `speech_recognition` uses Google's cloud API (network latency, rate limits, privacy). Faster-Whisper runs locally on GPU with CTranslate2 — 5.4x realtime, better accuracy, no network dependency (~3GB VRAM).

| File | Change |
|------|--------|
| `audio/stt.py` | New `FasterWhisperSTTProvider` — lazy CUDA model loading, RMS-based VAD (silence detection), 16kHz sounddevice capture, 15s max phrase cutoff. Factory updated: faster_whisper → speech_recognition fallback. |
| `config.py` (AudioConfig) | Added `stt_model: str = "large-v3-turbo"`, `stt_silence_threshold: float = 0.01`, `stt_silence_duration: float = 1.5` |
| `pyproject.toml` | Added `stt-local = ["faster-whisper>=1.1.0", "sounddevice>=0.5.0"]` optional deps |

**Activation**: Set `stt_provider = "faster_whisper"` in config and `pip install seaman-brain[stt-local]`.

### Phase 4: VRAM-Aware Model Scheduling

**Problem**: Qwen3-Coder-30B (~18GB) and Qwen3-VL-8B (~5-8GB) can collide in 32GB VRAM. Ollama auto-swaps but the 3-6s penalty causes invisible timeouts. No VRAM monitoring existed.

| File | Change |
|------|--------|
| `llm/scheduler.py` | **New file** — `ModelScheduler` with thread-safe slot-based mutual exclusion. Chat and vision are mutually exclusive heavy slots. `acquire(slot) -> bool`, `release(slot)`, `is_active(slot)`. |
| `vision/observer.py` | Fixed: reuses persistent `AsyncClient` instead of creating new one per `observe()` call |
| `vision/bridge.py` | Accepts optional `ModelScheduler`; `_do_capture()` gates on `scheduler.acquire("vision")`; all failure paths release slot |
| `gui/game_loop.py` | Creates `ModelScheduler` in `__init__`; passes to VisionBridge; acquires/releases `"chat"` slot around all LLM calls (`_submit_chat`, `_check_pending_response`, `_request_autonomous_remark`, `_check_pending_autonomous`, `_request_interaction_reaction`) |

**VRAM budget:**

| Combination | VRAM | Fits 32GB? |
|-------------|------|------------|
| Whisper + Kokoro (always loaded) | ~5GB | Yes |
| + Coder (during chat) | ~23GB | Yes |
| + Vision (swaps Coder out) | ~19GB | Yes |
| Coder + Vision simultaneously | ~26-31GB | Risky — scheduler prevents this |

### New Tests (123 tests across 6 files)

- `tests/test_llm/test_ollama_provider.py` — Updated for `num_ctx`/`num_predict` config fields
- `tests/test_config.py` — Updated for new LLM/Audio config fields
- `tests/test_llm/test_scheduler.py` — 8 new tests (mutual exclusion, thread safety, double-acquire)
- `tests/test_audio/test_tts.py` — Kokoro provider tests (init, synthesize, speak, factory, fallback)
- `tests/test_audio/test_stt.py` — Faster-Whisper provider tests (init, listen, factory, fallback)
- `tests/test_conversation/test_manager.py` — Updated autonomous remark test for USER message pattern

## Minor Code Issues (non-blocking)

- `callable` lowercase type hints in `needs/feeding.py:93`, `needs/care.py:90`, `gui/chat_panel.py:77`
- `needs/system.py:128` uses `creature_state.comfort` as proxy for stimulation
- `behavior/autonomous.py:225` takes `creature_state: dict[str, Any]` instead of typed `CreatureState`

---

## Runtime Failure Analysis (2026-02-26)

Systematic 5-agent audit traced **all user flows where audio or visual output permanently stops functioning**. 47 unique failure paths identified across audio pipeline, render pipeline, async bridge, settings mutations, and needs/behavior/events. Organized below by fix priority for a staged hardening plan.

### Already Fixed (this session)

| Bug | Fix | Commit |
|-----|-----|--------|
| LLM cold start (~60s unresponsive) | Warmup call in `manager.py:initialize()` | `1c84c6a` |
| Settings model change doesn't reach OllamaProvider | `update_llm_settings()` in manager, called from `game_loop._on_llm_apply()` | `1c84c6a` |
| TTS task destroyed on shutdown | Future tracking in `audio_integration.py`, cancel in `shutdown()` | `1c84c6a` |

### TIER 1 — Permanent, Unrecoverable Failures (6 issues)

**1. Async event loop thread dies silently**
- **File:** `window.py:186-189` — `_run_loop()` has NO try/except
- **Trigger:** Any unhandled exception in a scheduled task
- **Result:** Loop thread dies, all async subsystems (chat, TTS, STT, vision, behavior) permanently dead. Visual rendering continues but creature is braindead. No error shown.
- **Fix:** Wrap `run_forever()` in try/except with logging + optional loop restart

**2. LLM calls hang indefinitely (no timeout)**
- **Files:** `manager.py:271` (process_input), `manager.py:134` (warmup), `manager.py:508` (autonomous)
- **Trigger:** Ollama hangs, network dead, GPU deadlock
- **Result:** `await self._llm.chat()` blocks forever. `_pending_response` never completes. Chat permanently locked.
- **Fix:** `asyncio.wait_for(self._llm.chat(...), timeout=120.0)` on all LLM calls, `timeout=60.0` for warmup

**3. TTS executor thread pool saturation**
- **Files:** `audio/tts.py:24` — `ThreadPoolExecutor(max_workers=1)`, `audio/manager.py:154`
- **Trigger:** `pyttsx3.init()` or `engine.runAndWait()` blocks (SAPI5 deadlock on Windows)
- **Result:** Single-worker pool saturated, all subsequent TTS calls queue forever. No timeout.
- **Fix:** Add timeout wrapper around `loop.run_in_executor()` calls

**4. AudioManager creation fails → permanent silence**
- **Files:** `game_loop.py:232-237`
- **Trigger:** pyttsx3 import fails, no audio device
- **Result:** `_audio_manager = None` for entire session. No retry mechanism.
- **Fix:** Retry mechanism or lazy re-creation when settings change

**5. Pygame mixer invalidation → no SFX/ambient**
- **Files:** `audio_integration.py:103-128`
- **Trigger:** Mixer uninitialized externally (display mode change, hardware disconnect)
- **Result:** `_mixer_initialized` never re-checked. SFX and ambient permanently silent.
- **Fix:** Health-check in `update()`, re-initialize mixer if `get_init()` returns False

**6. Kokoro TTS lazy load fails → permanent voice silence**
- **Files:** `audio/tts.py:214-246`
- **Trigger:** kokoro not installed, CUDA OOM on first `synthesize()`
- **Result:** `_available = False` set permanently. All voice output becomes empty bytes.
- **Fix:** Retry after delay, or auto-fallback to pyttsx3 with notification

### TIER 2 — High Impact, Conditional (11 issues)

**7. Race: user chats before manager finishes initializing**
- **File:** `game_loop.py:802-803`
- **Root cause:** Checks `manager is not None` but NOT `manager.is_initialized`
- **Fix:** Add `and manager.is_initialized` guard

**8. Submit to dead/stopped loop → silent future hang**
- **File:** `window.py:218-232`
- **Root cause:** `run_coroutine_threadsafe()` returns Future that never completes if loop is dead
- **Fix:** Check `loop.is_running()` before submitting

**9. Audio output device change not propagated at runtime**
- **Files:** `game_loop.py:1088-1107`
- **Root cause:** Saved to config but mixer/pyttsx3 never re-initialized
- **Fix:** Implement `update_audio_output_device()` in AudioManager, reinit mixer

**10. STT provider upgrade fails silently**
- **Files:** `audio/manager.py:85-101`
- **Root cause:** `_try_upgrade_stt()` catches all exceptions, stays on NoopSTTProvider
- **Fix:** UI notification on failure, retry mechanism

**11. Settings config desynchronization (TTS/STT state)**
- **Files:** `game_loop.py:1098-1107`
- **Root cause:** `audio_bridge._config` and `audio_manager._tts_enabled` are separate state
- **Fix:** Single source of truth, or sync both on every settings change

**12. STT device change ignored for FasterWhisper**
- **Files:** `game_loop.py:1125-1132`
- **Root cause:** Only handles `SpeechRecognitionSTTProvider`, `isinstance()` fails for others
- **Fix:** Protocol method `set_input_device()` on all STT providers

**13. Creature death → complete audio/visual halt**
- **Files:** `game_loop.py:360-366, 667-679`
- **Root cause:** `game_over=True` stops all updates; only death screen renders
- **Note:** This is intentional but needs polish (death screen interactivity)

**14. Unguarded render sub-calls in `_render()`**
- **Files:** `game_loop.py:1168-1211`
- **Root cause:** tank_renderer, creature_renderer, settings_panel, lineage_panel renders are NOT individually wrapped in try-except. One throw kills the entire frame.
- **Fix:** Wrap each sub-render in try-except

**15. `_pending_response` / `_pending_autonomous` flags stuck**
- **Files:** `game_loop.py:851-881, 556-590`
- **Root cause:** If LLM future never completes, flags block all subsequent chat/behavior
- **Fix:** Timeout guard that force-clears stuck flags after N seconds

**16. Interaction reactions silently skipped when LLM busy**
- **Files:** `game_loop.py:592-620`
- **Root cause:** `if _pending_response is not None: return` — no canned fallback
- **Fix:** Queue reaction or use canned fallback

**17. NeedsEngine exception cascades**
- **Files:** `game_loop.py:354-358`
- **Root cause:** No try-except around `_update_needs()`. Exception kills mood/behavior/event updates.
- **Fix:** Wrap in try-except, log error, continue with stale state

### TIER 3 — Lower Impact / Edge Cases (10 issues)

**18. Chat panel text wrapping DoS**
- **File:** `chat_panel.py:175-207`
- **Trigger:** Very long spaceless message from LLM
- **Fix:** Max message length or chunked wrapping

**19. Surface creation OOM**
- **Files:** `chat_panel.py:427`, `tank_renderer.py:483`, `game_loop.py:1262`
- **Trigger:** Large window + VRAM exhaustion
- **Fix:** Pre-allocate surfaces, validate dimensions

**20. Font initialization cascade failure**
- **Files:** Multiple (`window.py:162`, `hud.py:190`, `chat_panel.py:141`, etc.)
- **Trigger:** "consolas" not installed
- **Fix:** Robust fallback chain (`consolas` → `courier` → `pygame.font.Font(None, size)`)

**21. Rapid settings toggles → TOML corruption**
- **Files:** `config.py:275-326`
- **Trigger:** Concurrent writes from rapid toggles
- **Fix:** File locking or debounced save

**22. Bloodline switch → creature state None**
- **Files:** `game_loop.py:1407-1414`
- **Root cause:** Switch doesn't reload creature state (incomplete implementation)
- **Fix:** Full bloodline switch with ConversationManager reinitialization

**23. Evolution during active behavior → stage/audio mismatch**
- **Files:** `game_loop.py:641-642`
- **Root cause:** Pending LLM call was built with old stage context
- **Fix:** Cancel pending autonomous on evolution trigger

**24. `_cancel_and_stop()` hangs in `gather()` on shutdown**
- **Files:** `window.py:387-412`
- **Trigger:** Task ignores CancelledError or blocks on I/O
- **Fix:** Add timeout to `_drain()` gather

**25. pyttsx3 silent file failure**
- **Files:** `audio/tts.py:144-155`
- **Trigger:** SAPI5 error state produces empty file
- **Fix:** Raise exception on empty output instead of returning empty WAV

**26. Webcam device index mismatch after hardware change**
- **Files:** `settings_panel.py:366-383`
- **Trigger:** Camera unplugged while settings panel open
- **Fix:** Rebuild device list on panel open, validate indices

**27. Personality trait changes not applied at runtime**
- **Files:** `game_loop.py:1072-1078`
- **Root cause:** Config saved but ConversationManager still uses old TraitProfile
- **Fix:** Add `update_personality_traits()` to ConversationManager

### Cross-Cutting Patterns

| Pattern | Occurrences | Fix |
|---------|------------|-----|
| No timeout on async I/O | LLM chat, warmup, vision, STT listen | `asyncio.wait_for()` |
| No try-except around sub-renders | tank, creature, settings, lineage | Individual wrapping |
| Silent fallback to None (no retry) | AudioManager, STT, Kokoro, mixer | Health check + retry |
| Shared mutable config without locks | 5+ mutation sites across threads | Immutable updates or lock |
| No loop-alive validation | submit_async, play_voice, start_listening | `loop.is_running()` check |
| Stuck flags blocking without timeout | `_pending_response`, `_pending_autonomous` | Timeout + forced clear |

### Suggested Fix Stages (for superplan)

**Stage 1 — Async Safety Net** (highest ROI, prevents total system death)
- Issues: #1, #2, #7, #8, #15
- Files: `window.py`, `manager.py`, `game_loop.py`
- Core: exception-proof the event loop, add timeouts to all LLM calls, add `is_initialized` guard, add `is_running()` check, add stuck-flag timeout

**Stage 2 — Audio Pipeline Resilience**
- Issues: #3, #4, #5, #6, #9, #10, #11, #12, #25
- Files: `audio/tts.py`, `audio/manager.py`, `audio_integration.py`, `game_loop.py`
- Core: TTS executor timeout, AudioManager retry/lazy-recreate, mixer health check, device propagation, STT provider upgrade feedback, config sync

**Stage 3 — Render Pipeline Hardening**
- Issues: #14, #18, #19, #20
- Files: `game_loop.py`, `chat_panel.py`, `tank_renderer.py`, multiple font sites
- Core: try-except per sub-render, text length limits, surface dimension validation, font fallback chain

**Stage 4 — Game State Safety**
- Issues: #13, #16, #17, #22, #23, #27
- Files: `game_loop.py`, `creature/persistence.py`, `conversation/manager.py`
- Core: needs exception handling, interaction fallbacks, bloodline switch completion, evolution cancellation, personality hot-swap

**Stage 5 — Config & Settings Robustness**
- Issues: #21, #24, #26
- Files: `config.py`, `settings_panel.py`, `window.py`
- Core: debounced TOML save, device list refresh, shutdown drain timeout
