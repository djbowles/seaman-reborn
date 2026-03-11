"""Tests for the extracted game business logic."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield


class TestGameSystemsTick:
    """Test that tick() calls subsystems at the correct intervals."""

    def test_needs_tick_at_interval(self):
        from seaman_brain.gui.game_systems import GameSystems

        needs_engine = MagicMock()
        creature_state = MagicMock()
        creature_state.is_alive = True
        clock = MagicMock()
        clock.elapsed_minutes = 10.0

        systems = GameSystems(
            needs_engine=needs_engine,
            mood_engine=MagicMock(),
            behavior_engine=MagicMock(),
            event_system=MagicMock(),
            evolution_engine=MagicMock(),
            death_engine=MagicMock(),
            creature_state=creature_state,
            clock=clock,
            tank=MagicMock(),
        )

        # First tick should trigger needs update (timer starts at 0)
        systems.tick(1.1)  # past the 1.0s interval
        needs_engine.update.assert_called_once()

    def test_no_tick_when_dead(self):
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

        systems.tick(1.1)
        systems._needs_engine.update.assert_not_called()


class TestFindTtsSplit:
    """Test TTS buffer splitting (moved from game_loop)."""

    def test_sentence_boundary(self):
        from seaman_brain.gui.game_systems import find_tts_split

        assert find_tts_split("Hello world. More text") == 12

    def test_no_boundary(self):
        from seaman_brain.gui.game_systems import find_tts_split

        assert find_tts_split("Hello world") is None

    def test_clause_boundary_long_enough(self):
        from seaman_brain.gui.game_systems import find_tts_split

        text = "A" * 45 + ", more"
        result = find_tts_split(text)
        assert result == 46  # position after the comma

    def test_clause_boundary_too_short(self):
        from seaman_brain.gui.game_systems import find_tts_split

        assert find_tts_split("Well, then") is None
