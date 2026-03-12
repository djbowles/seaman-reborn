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


def _make_systems(**overrides):
    """Helper to build GameSystems with reasonable mock defaults."""
    from seaman_brain.gui.game_systems import GameSystems

    creature_state = overrides.pop("creature_state", None)
    if creature_state is None:
        creature_state = MagicMock()
        creature_state.is_alive = True

    mood_engine = overrides.pop("mood_engine", None)
    if mood_engine is None:
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="neutral")

    death_engine = overrides.pop("death_engine", None)
    if death_engine is None:
        death_engine = MagicMock()
        death_engine.check_death.return_value = None

    evolution_engine = overrides.pop("evolution_engine", None)
    if evolution_engine is None:
        evolution_engine = MagicMock()
        evolution_engine.check_evolution.return_value = None

    clock = overrides.pop("clock", None)
    if clock is None:
        clock = MagicMock()
        clock.get_time_context.return_value = {}

    return GameSystems(
        needs_engine=overrides.pop("needs_engine", MagicMock()),
        mood_engine=mood_engine,
        behavior_engine=overrides.pop("behavior_engine", MagicMock()),
        event_system=overrides.pop("event_system", MagicMock()),
        evolution_engine=evolution_engine,
        death_engine=death_engine,
        creature_state=creature_state,
        clock=clock,
        tank=overrides.pop("tank", MagicMock()),
    )


class TestGameSystemsTick:
    """Test that tick() calls subsystems at the correct intervals."""

    def test_needs_tick_at_interval(self):
        needs_engine = MagicMock()
        systems = _make_systems(needs_engine=needs_engine)
        systems.tick(1.1)
        needs_engine.update.assert_called_once()

    def test_no_tick_when_dead(self):
        creature_state = MagicMock()
        creature_state.is_alive = False
        systems = _make_systems(creature_state=creature_state)
        result = systems.tick(1.1)
        assert result is None
        systems._needs_engine.update.assert_not_called()

    def test_returns_tick_result(self):
        systems = _make_systems()
        result = systems.tick(1.1)
        assert result is not None
        assert result.mood_value == "neutral"

    def test_mood_calculated_at_needs_interval(self):
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="sardonic")
        systems = _make_systems(mood_engine=mood_engine)
        result = systems.tick(1.1)
        mood_engine.calculate_mood.assert_called_once()
        assert result.mood_value == "sardonic"

    def test_behavior_checked_at_interval(self):
        behavior_engine = MagicMock()
        behavior_engine.get_idle_behavior.return_value = None
        mood_engine = MagicMock()
        mood_engine.calculate_mood.return_value = MagicMock(value="neutral")
        mood_engine.current_mood = MagicMock()
        systems = _make_systems(
            behavior_engine=behavior_engine,
            mood_engine=mood_engine,
        )
        systems.tick(16.0)  # past 15s behavior interval
        behavior_engine.get_idle_behavior.assert_called_once()

    def test_behavior_not_checked_before_interval(self):
        behavior_engine = MagicMock()
        systems = _make_systems(behavior_engine=behavior_engine)
        systems.tick(5.0)  # under 15s interval
        behavior_engine.get_idle_behavior.assert_not_called()

    def test_death_check_returns_cause(self):
        death_engine = MagicMock()
        death_engine.check_death.return_value = MagicMock(value="starvation")
        systems = _make_systems(death_engine=death_engine)
        result = systems.tick(1.1)
        assert result.death_cause is not None

    def test_evolution_detected(self):
        from seaman_brain.types import CreatureStage
        evolution_engine = MagicMock()
        evolution_engine.check_evolution.return_value = CreatureStage.GILLMAN
        systems = _make_systems(evolution_engine=evolution_engine)
        result = systems.tick(1.1)
        assert result.new_stage == CreatureStage.GILLMAN

    def test_events_checked_at_interval(self):
        event_system = MagicMock()
        event_system.check_events.return_value = []
        systems = _make_systems(event_system=event_system)
        systems.tick(4.0)  # past 3s event interval
        event_system.check_events.assert_called_once()

    def test_events_not_checked_before_interval(self):
        event_system = MagicMock()
        systems = _make_systems(event_system=event_system)
        systems.tick(1.0)  # under 3s interval
        event_system.check_events.assert_not_called()


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
