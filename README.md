# Seaman Reborn

A sardonic AI pet simulation featuring evolving creature cognition, local LLM inference, persistent vector memory, biological needs, full-duplex audio, and a Modern Minimal Pygame interface.

Raise, converse with, and care for a creature that starts as a grotesque mushroom-like larva and evolves through five distinct life stages — each with its own personality, speech patterns, and appearance. It remembers your conversations, judges your care, and will absolutely let you know when you're being boring.

## Features

- **5-stage personality evolution** — Mushroomer → Gillman → Podfish → Tadman → Frogman, each with unique traits, speech constraints, and sardonic charm
- **Local LLM inference** via Ollama (qwen3:8b) with optional Anthropic Claude and OpenAI cloud providers
- **Persistent hybrid memory** — episodic buffer + semantic vector search (LanceDB) with temporal weighting
- **Biological needs simulation** — hunger, comfort, health, stimulation, trust; neglect leads to death
- **Genetic inheritance** — lineage tracking across generations with trait mutation
- **Modern Minimal GUI** — dark void aesthetic, creature glow auras, glassmorphism chat, slide-out settings/lineage drawers
- **Full-duplex audio** — TTS/STT with echo cancellation, barge-in support, and NVIDIA Riva acceleration
- **Vision** — webcam input with Ollama qwen3-vl for visual observations
- **WebSocket API** — FastAPI bridge for external clients (designed for UE5 integration)

## Quick Start

```bash
# Clone and install
git clone https://github.com/djbowles/seaman-reborn.git
cd seaman-reborn
pip install -e ".[dev]"

# Make sure Ollama is running with required models
ollama pull qwen3:8b
ollama pull all-minilm:l6-v2

# Launch the GUI
python -m seaman_brain --gui
```

## Launch Modes

```bash
python -m seaman_brain          # Interactive terminal (async prompt-toolkit CLI)
python -m seaman_brain --gui    # Pygame GUI (Modern Minimal)
python -m seaman_brain --api    # WebSocket API server (ws://127.0.0.1:8420/ws/brain)
python -m seaman_brain --debug  # Enable debug logging (combinable with any mode)
```

On Windows, `launch.bat` starts the GUI directly.

## Installation

**Core only:**
```bash
pip install -e .
```

**With optional features:**
```bash
pip install -e ".[all]"                  # Everything
pip install -e ".[cloud]"               # Anthropic + OpenAI providers
pip install -e ".[riva,aec]"            # NVIDIA Riva TTS/STT + echo cancellation
pip install -e ".[vision]"              # Webcam + qwen3-vl vision
pip install -e ".[stt-local]"           # Faster-Whisper local STT
pip install -e ".[tts-neural]"          # Kokoro neural TTS
```

### Prerequisites

- Python 3.13+
- [Ollama](https://ollama.ai/) running locally with `qwen3:8b` and `all-minilm:l6-v2`
- (Optional) NVIDIA GPU for Riva audio acceleration
- (Optional) Webcam for vision features

## Architecture

```
src/seaman_brain/
├── llm/            # LLM provider abstraction (Ollama, Anthropic, OpenAI)
├── personality/    # 5-stage trait system + prompt assembly
├── memory/         # Hybrid episodic + semantic RAG (LanceDB)
├── creature/       # State, genetics, evolution, lineage
├── conversation/   # Central orchestrator wiring all subsystems
├── audio/          # TTS/STT providers, full-duplex pipeline, AEC
├── environment/    # Game clock + tank simulation
├── needs/          # Hunger, health, comfort, feeding, care, death
├── behavior/       # Mood engine, autonomous actions, events
├── gui/            # Modern Minimal Pygame interface
├── cli/            # Terminal interface (prompt-toolkit)
├── api/            # FastAPI WebSocket server
├── vision/         # Webcam capture + LLM vision bridge
├── config.py       # Pydantic v2 config loader (TOML)
└── types.py        # Shared enums and dataclasses
```

### Evolutionary Stages

| Stage | Threshold | Personality | Max Words |
|-------|-----------|-------------|-----------|
| Mushroomer | Start | Primitive, suspicious | 15 |
| Gillman | 20 interactions, 0.3 trust | Developing speech, growing cynicism | 30 |
| Podfish | 50 interactions, 0.5 trust | Articulate, sardonic | — |
| Tadman | 100 interactions, 0.6 trust | Philosophical, curious | — |
| Frogman | 200 interactions, 0.8 trust | Fully evolved, wise | — |

Stage-specific traits are configured in `config/stages/*.toml`.

## GUI

The Modern Minimal interface uses a dark void aesthetic with creature glow auras and glassmorphism overlays.

```
┌─────────── Top Bar (32px) ─────────────────────────┐
│ SEAMAN  Stage  ◆  ●  Timer  ⚙                      │
├──┬─────────────────────────────────────────────────┤
│  │                                                 │
│S │           ✦ creature glow aura ✦                │
│i │                                                 │
│d │              THE VOID                           │
│e │                                                 │
│b │                  ·  ·                           │
│a │                ·                                │
│r │                                                 │
├──┴─────────────────────────────────────────────────┤
│  [glass chat overlay with message bubbles]         │
│  Say something...                                  │
└────────────────────────────────────────────────────┘
```

- **Left sidebar**: Need tiles (hunger, health, comfort, trust) + action tiles (feed, aerator, temp, clean, drain, fill)
- **Creature**: 2.5x scaled with mood-reactive glow aura (gold → amber → orange → red → purple → blue)
- **Chat**: Glass overlay with warm bubbles (creature) and bright bubbles (user)
- **Drawers**: Settings and lineage panels slide out from the right

## Audio

Three TTS providers (independent, no fallback chains):

| Provider | Engine | Notes |
|----------|--------|-------|
| `pyttsx3` | System TTS | CPU, works everywhere |
| `kokoro` | Neural TTS | Local GPU inference |
| `riva` | NVIDIA NIM Magpie 1.7.0 | WSL2 Docker, RTF ~0.6x |

Three STT providers:

| Provider | Engine | Notes |
|----------|--------|-------|
| `speech_recognition` | Google Cloud | Cloud-based |
| `faster_whisper` | Whisper | Local CPU/GPU |
| `riva` | NVIDIA Quickstart 2.19.0 | WSL2 Docker |

Full-duplex pipeline with VAD (webrtcvad), NLMS echo cancellation, and barge-in support. Providers can be swapped at runtime from the Settings drawer.

## WebSocket API

```bash
python -m seaman_brain --api
```

**Endpoints:**
- `GET /api/health` — Health check
- `GET /api/state` — Full state snapshot
- `POST /api/reset` — Reset creature
- `WS /ws/brain` — Bidirectional conversation + state updates

**Client → Server:**
```json
{ "type": "input", "text": "Hello Seaman", "protocol_version": "1.0.0" }
```

**Server → Client:**
```json
{
  "type": "response",
  "text": "Go away, human.",
  "state": {
    "creature_state": { "stage": "gillman", "mood": "sardonic", "trust_level": 0.35 },
    "needs": { "hunger": 0.2, "comfort": 0.8 },
    "tank": { "temperature": 24.5, "cleanliness": 0.9 }
  }
}
```

## Configuration

Default config lives in `config/default.toml`. User overrides are auto-saved to `data/user_settings.toml`.

Key sections:

```toml
[llm]
provider = "ollama"       # or "anthropic", "openai"
model = "qwen3:8b"
temperature = 0.8

[audio]
tts_provider = "kokoro"   # or "riva", "pyttsx3"
stt_provider = "speech_recognition"

[creature]
initial_stage = "mushroomer"
auto_save = true

[gui]
window_width = 1024
window_height = 768
fps = 30
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests (2500+)
python -m pytest tests/ -x --tb=short

# Lint
python -m ruff check src/ tests/

# Version check
python -m seaman_brain --version
```

## License

MIT
