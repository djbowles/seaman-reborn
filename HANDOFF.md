# Seaman Reborn — UE5 Integration Handoff

## Project Status

The Python "brain" backend is **feature-complete**: 2041 tests passing, ruff clean, all 52 user stories implemented across 14 subpackages (llm, personality, memory, creature, conversation, cli, audio, environment, needs, behavior, gui, api, vision).

- **Repo**: https://github.com/djbowles/seaman-reborn (private)
- **Branch**: `ralph/ai-brain-core` (all work), `main` (base)
- **Entry points**: `python -m seaman_brain` (terminal), `--gui` (Pygame), `--api` (WebSocket server)
- **Hardware**: RTX 5090 (32GB VRAM), Ollama with qwen3-coder:30b + all-minilm:l6-v2

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

## GUI Overhaul (latest)

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

**Remaining GUI issues to investigate:**
- Verify creature sprite is visible in the resized tank area (864px wide)
- Verify chat panel Send button click works end-to-end with ConversationManager
- Chat panel input may need focus management (currently always captures keys when visible)
- Action bar drain/fill button may need visual state indicator (drained vs filled)
- No food selection submenu from ActionBar — Feed button auto-picks first available food type
- HUD compact mode may be too small on high-DPI displays — may need scaling support
- Settings panel content (LLM model, personality sliders) not verified after F1 change

## Minor Code Issues (non-blocking)

- `callable` lowercase type hints in `needs/feeding.py:93`, `needs/care.py:90`, `gui/chat_panel.py:77`
- `needs/system.py:128` uses `creature_state.comfort` as proxy for stimulation
- `behavior/autonomous.py:225` takes `creature_state: dict[str, Any]` instead of typed `CreatureState`
