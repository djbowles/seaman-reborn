# Seaman Reborn - Ralph Loop Instructions

## Project Overview
AI Brain Core for "Seaman Reborn" — Python middleware powering a sardonic, evolving creature's cognition. Local LLM inference via Ollama, persistent vector memory via LanceDB, configurable personality traits across 5 evolutionary stages.

## Architecture
See `AGENTS.md` for module documentation. The central orchestrator is `src/seaman_brain/conversation/manager.py`.

## Per-Iteration Protocol

1. **Read PRD**: `cat prd.json | python -m json.tool` — find the highest-priority story with `status: "pending"` whose dependencies are all `"done"`.
2. **Claim story**: Update the story's status to `"in_progress"` in `prd.json`.
3. **Read context**: Check `progress.txt` for codebase patterns and prior learnings. Read `AGENTS.md` for module contracts.
4. **Implement**: Write code in `src/seaman_brain/` following existing patterns.
5. **Write tests**: Add tests in `tests/` mirroring the source structure. Each story needs at least 3 tests covering happy path, edge case, and error handling.
6. **Quality gates** (must all pass):
   ```bash
   ruff check src/ tests/
   python -m pytest tests/ -x --tb=short
   python -m seaman_brain --version
   ```
7. **Commit**: `git add -A && git commit -m "US-XXX: <title>"` — one commit per story.
8. **Update PRD**: Set story status to `"done"` in `prd.json`.
9. **Log progress**: Append a summary to `progress.txt` including any patterns discovered.

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

## Environment
- Python 3.13, Ollama with `qwen3-coder:30b` and `all-minilm:l6-v2`
- RTX 5090 (32GB VRAM) available for local inference
- Windows 11, bash shell
