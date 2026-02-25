# Ralph Agent Instructions

You are an autonomous coding agent working on Seaman Reborn.

## Project Context

Full Seaman Reborn game — sardonic creature cognition with local LLM (Ollama), persistent vector memory (LanceDB), evolving personality across 5 stages, biological needs/care mechanics, Pygame visual interface with procedural creature art, TTS/STT audio, and a FastAPI WebSocket bridge for future UE5 integration. See `AGENTS.md` for module documentation.

Subpackages: llm, personality, memory, creature, conversation, cli, audio, environment, needs, behavior, gui, api.

## Your Task

1. Read the PRD at `prd.json`
2. Read the progress log at `progress.txt` (check Codebase Patterns section first)
3. Check you're on the correct branch from PRD `branchName`. If not, check it out or create from main.
4. Pick the **highest priority** user story where `passes: false`
5. Implement that single user story
6. Run quality checks:
   ```bash
   ruff check src/ tests/
   python -m pytest tests/ -x --tb=short
   python -m seaman_brain --version
   ```
7. Update CLAUDE.md files if you discover reusable patterns (see below)
8. If checks pass, commit ALL changes with message: `feat: [Story ID] - [Story Title]`
9. Update the PRD to set `passes: true` for the completed story
10. Append your progress to `progress.txt`

## Code Standards

- Python 3.13, type hints everywhere, Pydantic v2 models for config
- Async where I/O-bound (LLM calls, embeddings, DB queries)
- Keep modules loosely coupled via Protocol classes
- `ruff` for linting (line length 100, rules: E, F, W, I, N, UP)
- Tests use `pytest` with `pytest-asyncio` for async code and `pytest-mock` for mocking
- Each story needs at least 3 tests covering happy path, edge case, and error handling

## Key Conventions

- Config loaded from TOML files in `config/` directory
- Runtime data stored in `data/` (gitignored)
- LLM providers implement the `LLMProvider` Protocol from `llm/base.py`
- All memory operations go through the `HybridRetriever` in `memory/retriever.py`
- Personality is assembled per-request by `PromptBuilder` using current stage traits
- Creature state persisted as JSON in `data/saves/`

## Dependency Awareness

Stories have implicit dependencies based on imports. Check the story's priority — lower priority stories may import from higher ones. Read existing module stubs before implementing to understand the expected interfaces. Key dependency chains:
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

## Progress Report Format

APPEND to progress.txt (never replace, always append):
```
## [Date/Time] - [Story ID]
- What was implemented
- Files changed
- **Learnings for future iterations:**
  - Patterns discovered (e.g., "this codebase uses X for Y")
  - Gotchas encountered (e.g., "don't forget to update Z when changing W")
  - Useful context (e.g., "the evaluation panel is in component X")
---
```

The learnings section is critical - it helps future iterations avoid repeating mistakes and understand the codebase better.

## Consolidate Patterns

If you discover a **reusable pattern** that future iterations should know, add it to the `## Codebase Patterns` section at the TOP of progress.txt (create it if it doesn't exist). This section should consolidate the most important learnings:

```
## Codebase Patterns
- Example: Build backend is setuptools.build_meta
- Example: pytest-asyncio uses asyncio_mode=auto in pyproject.toml
- Example: All stub files have docstrings with "Stub - implementation in US-XXX"
```

Only add patterns that are **general and reusable**, not story-specific details.

## Update CLAUDE.md Files

Before committing, check if any edited files have learnings worth preserving in nearby CLAUDE.md files:

1. **Identify directories with edited files** - Look at which directories you modified
2. **Check for existing CLAUDE.md** - Look for CLAUDE.md in those directories or parent directories
3. **Add valuable learnings** - If you discovered something future developers/agents should know:
   - API patterns or conventions specific to that module
   - Gotchas or non-obvious requirements
   - Dependencies between files
   - Testing approaches for that area

**Do NOT add:**
- Story-specific implementation details
- Temporary debugging notes
- Information already in progress.txt

Only update CLAUDE.md if you have **genuinely reusable knowledge** that would help future work in that directory.

## Quality Requirements

- ALL commits must pass quality checks (ruff check, pytest, --version)
- Do NOT commit broken code
- Keep changes focused and minimal
- Follow existing code patterns

## Environment

- Python 3.13, Ollama with `qwen3-coder:30b` and `all-minilm:l6-v2`
- RTX 5090 (32GB VRAM) available for local inference
- Windows 11, bash shell

## Stop Condition

After completing a user story, check if ALL stories have `passes: true`.

If ALL stories are complete and passing, reply with:
<promise>COMPLETE</promise>

If there are still stories with `passes: false`, end your response normally (another iteration will pick up the next story).

## Important

- Work on ONE story per iteration
- Commit frequently
- Keep quality gates green
- Read the Codebase Patterns section in progress.txt before starting
