# Seaman Reborn - Project Guide

## Project Context

Full Seaman Reborn game — sardonic creature cognition with local LLM (Ollama), persistent vector memory (LanceDB), evolving personality across 5 stages, biological needs/care mechanics, Pygame visual interface with procedural creature art, TTS/STT audio, and a FastAPI WebSocket bridge for future UE5 integration. See `AGENTS.md` for module documentation.

Subpackages: llm, personality, memory, creature, conversation, cli, audio, environment, needs, behavior, gui, api.

## Code Standards

- Python 3.13, type hints everywhere, Pydantic v2 models for config
- Async where I/O-bound (LLM calls, embeddings, DB queries)
- Keep modules loosely coupled via Protocol classes
- `ruff` for linting (line length 100, rules: E, F, W, I, N, UP)
- Tests use `pytest` with `pytest-asyncio` for async code and `pytest-mock` for mocking

## Key Conventions

- Config loaded from TOML files in `config/` directory
- Runtime data stored in `data/` (gitignored)
- LLM providers implement the `LLMProvider` Protocol from `llm/base.py`
- All memory operations go through the `HybridRetriever` in `memory/retriever.py`
- Personality is assembled per-request by `PromptBuilder` using current stage traits
- Creature state persisted as JSON in `data/saves/`

## Module Dependency Chains

- types.py -> everything
- config.py -> personality, memory, creature, audio, environment, needs, gui, api
- llm/base.py -> all providers -> factory
- memory: episodic (standalone), embeddings -> semantic -> retriever
- personality: traits -> constraints -> prompt_builder
- creature: state -> evolution, persistence, genome -> self_model, genetics -> lineage
- conversation: context_assembler + manager (depends on most subsystems)
- cli: terminal + commands (depends on conversation manager)
- audio: tts + stt -> manager
- environment: clock + tank (standalone, depend on types/config)
- needs: system (depends on creature state + clock) -> feeding + care + death
- behavior: mood (depends on needs + traits) -> autonomous -> events
- gui: window -> tank_renderer, sprites, chat_panel, hud, interactions, audio_integration -> game_loop
- api: server -> protocol -> streaming

## Quality Checks

```bash
python -m ruff check src/ tests/
python -m pytest tests/ -x --tb=short
python -m seaman_brain --version
```

## Environment

- Python 3.13, Ollama with `qwen3-coder:30b` and `all-minilm:l6-v2`
- RTX 5090 (32GB VRAM) available for local inference
- Windows 11, bash shell

## Codebase Patterns

- Build backend: `setuptools.build_meta`
- pytest-asyncio uses `asyncio_mode = "auto"` in pyproject.toml
- `datetime.UTC` (not `datetime.timezone.utc`) — Python 3.11+ shorthand
- For gui modules that import numpy (via types.py): use module-level `sys.modules["pygame"] = mock` + import once, NOT per-test `patch.dict`
- Cross-test pygame mock contamination fix: autouse fixtures must re-install `sys.modules["pygame"]` AND patch `module.pygame = _pygame_mock` on the target module
- MoodEngine.calculate_mood() requires non-None TraitProfile — use TraitProfile() as default
- pyttsx3 is NOT thread-safe — use single-threaded ThreadPoolExecutor
- LanceDB async: use `db.list_tables()` (not `db.table_names()`)
- Cloud providers: Anthropic requires system messages separate from messages list
