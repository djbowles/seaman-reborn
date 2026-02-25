# Seaman Reborn - Module Documentation

## Package: `seaman_brain`

### `types.py` — Shared Types
- `CreatureStage` enum: MUSHROOMER, GILLMAN, PODFISH, TADMAN, FROGMAN
- `MessageRole` enum: SYSTEM, USER, ASSISTANT
- `ChatMessage` dataclass: role + content + timestamp
- `MemoryRecord` dataclass: text, embedding, timestamp, importance, source
- `DeathCause` enum: STARVATION, SUFFOCATION, HYPOTHERMIA, HYPERTHERMIA, ILLNESS, OLD_AGE
- `CreatureMood` enum: HOSTILE, IRRITATED, SARDONIC, NEUTRAL, CURIOUS, AMUSED, PHILOSOPHICAL, CONTENT
- `FoodType` enum: PELLET, WORM, INSECT, NAUTILUS

### `config.py` — Configuration
- `SeamanConfig` Pydantic model: top-level config aggregating all sub-configs
- `LLMConfig`: provider name, model, temperature, max_tokens, base_url
- `MemoryConfig`: buffer_size, vector_dims, top_k, extraction_interval
- `PersonalityConfig`: base traits, stage overrides path
- `AudioConfig`: tts_provider, stt_provider, tts_model, stt_model, volumes
- `EnvironmentConfig`: default tank temperature, cleanliness rates, oxygen rates
- `NeedsConfig`: hunger_rate, comfort_decay, health_threshold, stimulation_decay
- `GUIConfig`: window_width, window_height, fps, theme
- `APIConfig`: host, port, cors_origins
- Loads from `config/default.toml`, merges stage-specific TOML overrides

---

### Subpackage: `llm/`

#### `base.py` — LLM Provider Protocol
- `LLMProvider` Protocol: `async chat(messages: list[ChatMessage]) -> str`
- `async stream(messages: list[ChatMessage]) -> AsyncIterator[str]`

#### `ollama_provider.py` — Local Inference
- Wraps `ollama` Python client
- Configurable model, temperature, context window
- Streaming support for terminal display

#### `openai_provider.py` — OpenAI Cloud Fallback
- Uses `openai` SDK, reads API key from env or config
- Same interface as Ollama provider

#### `anthropic_provider.py` — Anthropic Cloud Fallback
- Uses `anthropic` SDK, reads API key from env or config
- Handles Anthropic's message format differences

#### `factory.py` — Provider Factory
- `create_provider(config: LLMConfig) -> LLMProvider`
- Selects provider by config name, validates availability

---

### Subpackage: `personality/`

#### `traits.py` — Trait System
- 8 trait dimensions: cynicism, wit, patience, curiosity, warmth, verbosity, formality, aggression
- Each is a float 0.0-1.0
- `TraitProfile` dataclass with defaults per `CreatureStage`
- Traits shift based on interaction history and evolution

#### `prompt_builder.py` — System Prompt Assembly (CRITICAL)
- Builds the system prompt that defines Seaman's personality
- Incorporates: stage-specific traits, remembered facts about the user, creature mood, time context
- Strict negative constraints suppressing AI assistant tone
- This is the most important file for achieving authentic Seaman personality

#### `constraints.py` — Output Filtering
- Forbidden phrase list (breaks character immersion)
- Response length enforcement
- Personality consistency checks
- Strips AI assistant cliches ("As an AI...", "I'd be happy to...", etc.)

---

### Subpackage: `memory/`

#### `episodic.py` — Rolling Buffer
- `EpisodicMemory`: fixed-size deque of recent `ChatMessage`s
- Provides conversation context window for LLM
- Configurable buffer size (default: 20 messages)

#### `embeddings.py` — Embedding Provider
- `EmbeddingProvider`: wraps Ollama's `all-minilm:l6-v2` model
- `async embed(text: str) -> list[float]`
- `async embed_batch(texts: list[str]) -> list[list[float]]`

#### `semantic.py` — LanceDB Vector Store
- `SemanticMemory`: persistent vector store using LanceDB
- `async add(record: MemoryRecord)` — store a memory
- `async search(vector: list[float], top_k: int) -> list[MemoryRecord]`
- Schema: text, vector, timestamp, importance, source

#### `extractor.py` — Background Fact Extraction
- `MemoryExtractor`: uses LLM to extract facts from conversation
- Runs every N messages (configurable)
- Extracts: user preferences, personal info, topics discussed, creature observations
- Stores extracted facts in SemanticMemory

#### `retriever.py` — Hybrid Search
- `HybridRetriever`: combines semantic similarity + temporal recency
- Retrieves relevant memories for current conversation context
- Re-ranks by weighted combination of similarity score and recency

---

### Subpackage: `creature/`

#### `state.py` — Creature State
- `CreatureState` dataclass: stage, age, interaction_count, mood, trust_level, hunger, health, comfort, last_fed, last_interaction, birth_time
- Serializable to/from JSON

#### `evolution.py` — Stage Transitions
- `EvolutionEngine`: checks if creature should evolve
- Conditions: interaction count thresholds, trust level, time elapsed
- Transitions: MUSHROOMER -> GILLMAN -> PODFISH -> TADMAN -> FROGMAN

#### `persistence.py` — Save/Load
- `StatePersistence`: JSON file save/load for CreatureState
- Auto-saves after each interaction
- Creates backups before stage transitions

---

### Subpackage: `conversation/`

#### `context_assembler.py` — Context Assembly
- Combines: system prompt + retrieved memories + episodic buffer
- Manages token budget across components
- Formats memories as "things I remember about you" block

#### `manager.py` — MAIN ORCHESTRATOR
- `ConversationManager.process_input(user_input: str) -> str`
- Coordinates all subsystems per the architecture flow
- Handles graceful degradation (no embeddings? skip RAG)
- Entry point for all conversation logic

---

### Subpackage: `cli/`

#### `terminal.py` — Rich Terminal UI
- Uses `rich` for styled output and `prompt_toolkit` for input
- Displays creature stage, mood indicators
- Streaming response display
- Handles Ctrl+C gracefully

#### `commands.py` — Debug Commands
- `/state` — show creature state
- `/memory` — show recent memories
- `/stage` — show/set creature stage
- `/traits` — show current personality traits
- `/reset` — reset creature to initial state
- `/quit` — save and exit

---

### Subpackage: `audio/`

#### `tts.py` — Text-to-Speech
- `TTSProvider` Protocol: `async synthesize(text) -> bytes`, `speak(text) -> None`
- `Pyttsx3TTSProvider`: offline cross-platform TTS via pyttsx3
- Configurable voice, rate, volume

#### `stt.py` — Speech-to-Text
- `STTProvider` Protocol: `async listen() -> str`
- `SpeechRecognitionSTTProvider`: wraps SpeechRecognition library
- Configurable backend (vosk offline, google online)

#### `manager.py` — Audio Manager
- `AudioManager`: coordinates TTS, STT, and SFX
- Per-channel enable/disable (tts, stt, sfx)
- Thread-safe for concurrent GUI usage

---

### Subpackage: `environment/`

#### `clock.py` — Real-Time Clock
- `GameClock`: time-of-day, day-of-week, session tracking
- Detects long absences with severity levels
- `get_time_context() -> dict` for prompt injection

#### `tank.py` — Tank Environment
- `TankEnvironment`: temperature, cleanliness, oxygen, water_level, environment_type
- `update(elapsed)` degrades conditions over time
- `drain()` transitions aquarium -> terrarium
- `is_habitable()` checks survival conditions

---

### Subpackage: `needs/`

#### `system.py` — Creature Needs
- `CreatureNeeds`: hunger, comfort, health, stimulation (all 0-1)
- `NeedsEngine.update()` calculates needs from time + tank + interaction
- `get_urgent_needs()` returns needs requiring attention

#### `feeding.py` — Feeding Mechanics
- `FoodType` enum, `FeedingResult` dataclass
- `FeedingEngine.feed()` applies food effects, validates stage-appropriate food
- Overfeeding penalty, cooldown system

#### `care.py` — Tank Care
- `TankCareEngine`: temperature adjustment, cleaning, drain
- Stage-specific requirements (sprinkler for Frogman)
- Warning generation for urgent maintenance

#### `death.py` — Death Mechanics
- `DeathEngine.check_death()` monitors for death conditions
- `DeathCause` enum with cause-specific messages
- Revival: reset to new egg state

---

### Subpackage: `behavior/`

#### `mood.py` — Dynamic Mood Engine
- `MoodEngine.calculate_mood()` from needs, trust, time, interactions, traits
- Gradual mood transitions (no instant jumps)
- `get_mood_modifiers()` returns prompt modifiers

#### `autonomous.py` — Autonomous Behavior
- `BehaviorEngine`: idle behaviors (swim, tap glass, complain, observe, sleep)
- Weighted by mood and needs
- Cooldown system, optional LLM-powered idle speech

#### `events.py` — Event System
- `EventSystem`: scheduled and random events
- Time-triggered (holidays, 3am visits), stage-triggered (breeding, cannibalism)
- Events modify creature state and trigger dialogue

---

### Subpackage: `gui/`

#### `window.py` — Pygame Window & Main Loop
- `GameWindow`: Pygame init, event loop, update/render cycle
- Configurable size and FPS
- Async bridge for ConversationManager

#### `tank_renderer.py` — Tank Environment Renderer
- Aquarium mode: animated water, bubbles, gravel
- Terrarium mode: dry substrate, rocks, moisture
- Temperature/cleanliness visual indicators

#### `sprites.py` — Creature Sprite System
- Procedural art for 5 stages (no external assets)
- Animation states: IDLE, SWIMMING, TALKING, EATING, SLEEPING
- Face tracks mouse cursor

#### `chat_panel.py` — Chat Panel Overlay
- Semi-transparent overlay, scrollable history
- Text input with cursor, streaming response display
- Toggle with Tab key

#### `hud.py` — HUD & Status Display
- Need bars, mood indicator, trust meter
- Tank condition indicators
- Compact and expanded modes

#### `interactions.py` — Interactive Elements
- Feeding (click to drop food), temperature controls
- Glass tapping, tank cleaning button
- Visual feedback (ripples, food animation)

#### `audio_integration.py` — Pygame Audio Bridge
- Ambient sound loops, creature voice via TTS
- UI sound effects, microphone toggle
- Per-channel volume controls

#### `game_loop.py` — Full Game Integration
- `GameEngine`: orchestrates all GUI subsystems
- Async conversation bridge, needs/mood/behavior ticking
- Evolution and death sequences

---

### Subpackage: `api/`

#### `server.py` — FastAPI WebSocket Server
- WebSocket endpoint `/ws/brain` for real-time communication
- REST endpoints: `/api/state`, `/api/health`, `/api/reset`
- `python -m seaman_brain --api` launches server

#### `protocol.py` — State Protocol Schema
- Pydantic models: InputMessage, ResponseMessage, StateUpdate, EventNotification
- BrainStateSnapshot: full serializable state
- Protocol versioning

#### `streaming.py` — Real-Time Event Streaming
- `EventBroadcaster`: pushes state changes to WebSocket clients
- Subscribable channels: mood, needs, evolution, behavior, tank, death
- State diff streaming, configurable intervals
