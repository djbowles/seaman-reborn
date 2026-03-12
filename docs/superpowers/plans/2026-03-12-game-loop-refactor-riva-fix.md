# GameEngine Refactor + Riva TTS Fix

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the bloated 1031-line `game_loop.py` back into its intended thin-orchestrator architecture (~400 lines), and fix the Riva NIM Magpie TTS crash-loop so both STT and TTS work end-to-end.

**Architecture:** `game_loop.py` was a 317-line thin orchestrator after the GUI rewrite but accreted 714 lines of inlined business logic. We extract that logic into the modules that already exist for it — `game_systems.py` (behavior/events/evolution/death), `interactions.py` (action dispatch), and a new `response_handler.py` (streaming chat + TTS). We also fix Riva TTS by updating the NIM container to resolve an ONNX IR version mismatch.

**Tech Stack:** Python 3.13, Pygame, pytest, NVIDIA Riva NIM (gRPC), Ollama

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `gui/game_systems.py` | Modify | Add behavior/event/evolution/death tick logic; add TickResult dataclass |
| `gui/response_handler.py` | Create | Extract streaming chat, TTS sentence splitting, pending response management |
| `gui/game_loop.py` | Modify | Thin down to ~400 lines, delegate to GameSystems + ResponseHandler |
| `gui/hud.py` | Modify | Add on_settings_click / on_lineage_click callbacks |
| `tests/test_gui/test_game_systems.py` | Modify | Add tests for behavior/event/evolution/death |
| `tests/test_gui/test_response_handler.py` | Create | Tests for streaming, TTS split, pending management |
| `tests/test_gui/test_game_loop.py` | Modify | Update for refactored engine |
| `config/default.toml` | Keep | riva_tts_uri already added |
| `docker/nim_upgrade.sh` | Create | Script to upgrade NIM container |

---

## Chunk 1: Extend GameSystems with full tick logic

### Task 1: Add TickResult and extend GameSystems.tick()

GameSystems currently only ticks needs. It should also handle mood, behavior checks, event checks, evolution checks, and death checks — returning results via a dataclass so game_loop can react (show messages, play sounds, trigger animations).

**Files:**
- Modify: `src/seaman_brain/gui/game_systems.py`
- Modify: `tests/test_gui/test_game_systems.py`

- [ ] **Step 1: Write failing tests for extended tick**

Add to `tests/test_gui/test_game_systems.py`:

```python
class TestFullTick:
    """Test that tick() runs mood, behavior, events, evolution, death."""

    def test_mood_calculated_at_needs_interval(self):
        from seaman_brain.gui.game_systems import GameSystems
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="sardonic")
        creature_state = MagicMock()
        creature_state.is_alive = True
        clock = MagicMock()
        clock.get_time_context.return_value = {}

        systems = GameSystems(
            needs_engine=MagicMock(),
            mood_engine=mood_engine,
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=MagicMock(),
            creature_state=creature_state,
            clock=clock,
            tank=MagicMock(),
        )
        result = systems.tick(1.1)
        mood_engine.calculate_mood.assert_called_once()
        assert result is not None

    def test_behavior_checked_at_interval(self):
        from seaman_brain.gui.game_systems import GameSystems
        behavior_engine = MagicMock()
        behavior_engine.get_idle_behavior.return_value = None
        creature_state = MagicMock()
        creature_state.is_alive = True
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="neutral")
        mood_engine.current_mood = MagicMock()
        clock = MagicMock()
        clock.get_time_context.return_value = {}

        systems = GameSystems(
            needs_engine=MagicMock(),
            mood_engine=mood_engine,
            behavior_engine=behavior_engine,
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=MagicMock(),
            creature_state=creature_state,
            clock=clock,
            tank=MagicMock(),
        )
        result = systems.tick(16.0)  # past 15s behavior interval
        behavior_engine.get_idle_behavior.assert_called_once()

    def test_death_check_returns_cause(self):
        from seaman_brain.gui.game_systems import GameSystems
        death_engine = MagicMock()
        death_engine.check_death.return_value = MagicMock(value="starvation")
        creature_state = MagicMock()
        creature_state.is_alive = True
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="neutral")
        clock = MagicMock()
        clock.get_time_context.return_value = {}

        systems = GameSystems(
            needs_engine=MagicMock(),
            mood_engine=mood_engine,
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=death_engine,
            creature_state=creature_state,
            clock=clock,
            tank=MagicMock(),
        )
        result = systems.tick(1.1)
        assert result.death_cause is not None

    def test_evolution_detected(self):
        from seaman_brain.gui.game_systems import GameSystems
        evolution_engine = MagicMock()
        evolution_engine.check_evolution.return_value = MagicMock()
        creature_state = MagicMock()
        creature_state.is_alive = True
        death_engine = MagicMock()
        death_engine.check_death.return_value = None
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="neutral")
        clock = MagicMock()
        clock.get_time_context.return_value = {}

        systems = GameSystems(
            needs_engine=MagicMock(),
            mood_engine=mood_engine,
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=evolution_engine,
            death_engine=death_engine,
            creature_state=creature_state,
            clock=clock,
            tank=MagicMock(),
        )
        result = systems.tick(1.1)
        assert result.new_stage is not None

    def test_tick_result_none_when_dead(self):
        from seaman_brain.gui.game_systems import GameSystems
        creature_state = MagicMock()
        creature_state.is_alive = False

        systems = GameSystems(
            needs_engine=MagicMock(),
            mood_engine=MagicMock(),
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=MagicMock(),
            creature_state=creature_state,
            clock=MagicMock(),
            tank=MagicMock(),
        )
        result = systems.tick(1.1)
        assert result is None
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
python -m pytest tests/test_gui/test_game_systems.py -x --tb=short -v
```

- [ ] **Step 3: Implement TickResult and extended tick()**

In `game_systems.py`, add `TickResult` dataclass and extend `tick()`:

```python
from dataclasses import dataclass, field
from seaman_brain.behavior.autonomous import IdleBehavior

@dataclass
class TickResult:
    """Results from a single game systems tick."""
    behavior: IdleBehavior | None = None
    fired_events: list[Any] = field(default_factory=list)
    new_stage: Any | None = None  # CreatureStage if evolution triggered
    death_cause: Any | None = None  # DeathCause if creature died
    mood_value: str = "neutral"
```

Extend `tick()` to:
1. Update needs (existing)
2. Check death → return TickResult with death_cause
3. Calculate mood
4. Check behaviors at _BEHAVIOR_CHECK_INTERVAL
5. Check events at _EVENT_CHECK_INTERVAL
6. Check evolution
7. Return TickResult with all results

- [ ] **Step 4: Run tests — verify they pass**

```bash
python -m pytest tests/test_gui/test_game_systems.py -x --tb=short -v
```

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest tests/ -x --tb=short
```

- [ ] **Step 6: Commit**

```bash
git add src/seaman_brain/gui/game_systems.py tests/test_gui/test_game_systems.py
git commit -m "feat(gui): extend GameSystems.tick() with behavior/events/evolution/death"
```

---

## Chunk 2: Extract ResponseHandler

### Task 2: Create response_handler.py with streaming chat logic

Extract the ~160 lines of streaming response handling, TTS sentence splitting, and pending future management from game_loop.py into a focused module.

**Files:**
- Create: `src/seaman_brain/gui/response_handler.py`
- Create: `tests/test_gui/test_response_handler.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for ResponseHandler — streaming chat + TTS splitting."""
from __future__ import annotations
import sys
from unittest.mock import MagicMock
_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import queue
import pytest
from seaman_brain.gui.response_handler import ResponseHandler

@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield

class TestStreamDrain:
    def test_drains_tokens_from_queue(self):
        chat = MagicMock()
        handler = ResponseHandler(chat_panel=chat, audio_bridge=None)
        handler.start_stream()
        handler._stream_queue.put("Hello ")
        handler._stream_queue.put("world")
        handler.drain_stream()
        chat.update_streaming.assert_called()
        assert "Hello world" in handler._stream_accumulated

    def test_sentinel_marks_complete(self):
        chat = MagicMock()
        handler = ResponseHandler(chat_panel=chat, audio_bridge=None)
        handler.start_stream()
        handler._stream_queue.put("Hi")
        handler._stream_queue.put(None)  # sentinel
        handler.drain_stream()
        assert handler._stream_complete

class TestTTSSplitting:
    def test_sentence_triggers_tts(self):
        bridge = MagicMock()
        chat = MagicMock()
        handler = ResponseHandler(chat_panel=chat, audio_bridge=bridge)
        handler.start_stream()
        handler._stream_queue.put("Hello world. More text")
        handler.drain_stream()
        bridge.play_voice.assert_called_once()

class TestPendingTimeout:
    def test_timeout_cancels_pending(self):
        chat = MagicMock()
        handler = ResponseHandler(chat_panel=chat, audio_bridge=None)
        future = MagicMock()
        future.done.return_value = False
        handler.start_response(future)
        handler._pending_time = 0.0  # force timeout
        handler.check_pending()
        future.cancel.assert_called_once()
```

- [ ] **Step 2: Run tests — verify they fail**

- [ ] **Step 3: Implement ResponseHandler**

Key methods:
- `start_stream()` — reset queue, buffer, accumulated text
- `put_token(token)` — feed tokens from async thread
- `drain_stream()` — drain queue, update chat panel, trigger TTS on sentence boundaries
- `start_response(future)` — track pending future with timestamp
- `check_pending()` — timeout handling, completion handling
- `finish()` — clean up, speak remaining buffer, release scheduler
- `cancel()` — cancel in-flight response

- [ ] **Step 4: Run tests — verify they pass**

- [ ] **Step 5: Run full test suite**

- [ ] **Step 6: Commit**

```bash
git add src/seaman_brain/gui/response_handler.py tests/test_gui/test_response_handler.py
git commit -m "feat(gui): extract ResponseHandler for streaming chat + TTS"
```

---

## Chunk 3: Slim down game_loop.py

### Task 3: Refactor GameEngine to delegate to GameSystems and ResponseHandler

Replace the inlined logic with calls to the extracted modules.

**Files:**
- Modify: `src/seaman_brain/gui/game_loop.py`
- Modify: `src/seaman_brain/gui/hud.py`
- Modify: `tests/test_gui/test_game_loop.py`

- [ ] **Step 1: Replace _update() body with GameSystems.tick()**

The current `_update()` is ~125 lines with 10 try-except blocks. Replace with:

```python
def _update(self, dt: float) -> None:
    self._scene_manager.update(dt)
    self._hud.update(dt)

    if self._scene_manager.state != GameState.PLAYING:
        return
    if self.game_over:
        return
    if self._evolution_active:
        self._update_evolution_celebration(dt)
        return

    # Tick all game subsystems
    try:
        self._tank_env.update(dt)
    except Exception as exc:
        logger.error("Tank update error: %s", exc, exc_info=True)

    result = self._game_systems.tick(dt)
    if result is not None:
        self._handle_tick_result(result)

    # Animations + renderer sync
    self._update_renderers(dt)
    # Audio bridge
    self._update_audio(dt)
    # HUD data sync
    self._sync_hud()
    # STT + pending responses
    self._update_responses()
```

- [ ] **Step 2: Move _on_action dispatch to use InteractionManager directly**

Simplify the 70-line `_on_action()` to use InteractionManager's existing APIs.

- [ ] **Step 3: Wire HUD callbacks for settings/lineage buttons**

Add `on_settings_click` and `on_lineage_click` callbacks to HUD.
Remove manual rect-checking from `_on_mouse_click`.

- [ ] **Step 4: Use ResponseHandler for all chat/STT/pending logic**

Replace `_submit_chat`, `_check_pending_response`, `_check_pending_autonomous`, `_check_stt_queue` with ResponseHandler calls.

- [ ] **Step 5: Add creature mood sync (bug fix)**

Add `self._creature_renderer.set_mood(mood_value)` to renderer sync — currently missing.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -x --tb=short
python -m ruff check src/ tests/
```

- [ ] **Step 7: Commit**

```bash
git add src/seaman_brain/gui/game_loop.py src/seaman_brain/gui/game_systems.py \
        src/seaman_brain/gui/hud.py src/seaman_brain/gui/response_handler.py \
        tests/test_gui/
git commit -m "refactor(gui): slim game_loop.py from 1031 to ~400 lines via delegation"
```

---

## Chunk 4: Fix Riva NIM Magpie TTS

### Task 4: Upgrade NIM container and verify TTS

The NIM Magpie 1.7.0 container crashes due to ONNX IR version mismatch (model uses IR v11, runtime supports max v10). Upgrade to latest tag.

**Files:**
- Create: `docker/nim_upgrade.sh`
- Modify: `config/default.toml` (if voice names change)

- [ ] **Step 1: Pull latest NIM Magpie container** (already running in background)

- [ ] **Step 2: Recreate the TTS container with latest image**

```bash
docker stop riva-nim-tts
docker rm riva-nim-tts
# Recreate with same volume mounts + config as before
docker run -d --name riva-nim-tts \
  --runtime=nvidia --shm-size=8GB \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -e LD_LIBRARY_PATH=/usr/lib/wsl/lib \
  -e NIM_CACHE_PATH=/opt/nim/.cache \
  -v nim-tts-cache:/opt/nim/.cache \
  -v nim-tts-workspace:/opt/nim/workspace \
  -p 50052:50051 \
  --restart unless-stopped \
  nvcr.io/nim/nvidia/magpie-tts-multilingual:latest
```

- [ ] **Step 3: Wait for model loading and test gRPC connectivity**

```python
import grpc
ch = grpc.insecure_channel('localhost:50052')
grpc.channel_ready_future(ch).result(timeout=300)
print("TTS REACHABLE")
```

- [ ] **Step 4: Test TTS synthesis end-to-end**

```python
import riva.client
auth = riva.client.Auth(uri="localhost:50052")
service = riva.client.SpeechSynthesisService(auth)
response = service.synthesize(
    text="Hello, I am Seaman.",
    voice_name="Magpie-Multilingual.EN-US.Jason",
    language_code="en-US",
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    sample_rate_hz=22050,
)
print(f"Got {len(response.audio)} bytes of audio")
```

- [ ] **Step 5: Update default.toml voice if needed**

- [ ] **Step 6: Commit**

```bash
git add docker/ config/default.toml
git commit -m "fix(audio): upgrade NIM Magpie TTS to fix ONNX IR v11 crash"
```

---

## Verification

After all chunks complete:

```bash
python -m ruff check src/ tests/
python -m pytest tests/ -x --tb=short
python -m seaman_brain --gui
# Verify: Settings → Audio → TTS Engine: NVIDIA Riva, STT Engine: NVIDIA Riva
# Verify: creature speaks via Riva Magpie TTS, mic input via Riva ASR
```
