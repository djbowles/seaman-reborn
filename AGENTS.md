# Seaman Reborn - Module Documentation

## Package: `seaman_brain`

### `types.py` — Shared Types
- `CreatureStage` enum: MUSHROOMER, GILLMAN, PODFISH, TADMAN, FROGMAN
- `MessageRole` enum: SYSTEM, USER, ASSISTANT
- `ChatMessage` dataclass: role + content + timestamp
- `MemoryRecord` dataclass: text, embedding, timestamp, importance, source

### `config.py` — Configuration
- `SeamanConfig` Pydantic model: top-level config aggregating all sub-configs
- `LLMConfig`: provider name, model, temperature, max_tokens, base_url
- `MemoryConfig`: buffer_size, vector_dims, top_k, extraction_interval
- `PersonalityConfig`: base traits, stage overrides path
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
- This is the most important file for achieving authentic Seaman personality

#### `constraints.py` — Output Filtering
- Forbidden phrase list (breaks character immersion)
- Response length enforcement
- Personality consistency checks
- Strips AI assistant clichés ("As an AI...", "I'd be happy to...", etc.)

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
- `CreatureState` dataclass: stage, age, interaction_count, mood, trust_level, last_fed, etc.
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
