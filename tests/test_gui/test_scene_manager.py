"""Tests for scene/state management."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.game_systems import GameState  # noqa: E402
from seaman_brain.gui.scene_manager import SceneManager  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    yield


class TestSceneManager:
    def test_initial_state_is_playing(self):
        sm = SceneManager()
        assert sm.state == GameState.PLAYING

    def test_open_settings(self):
        sm = SceneManager()
        sm.open_settings()
        assert sm.state == GameState.SETTINGS
        assert sm.drawer_open

    def test_close_settings(self):
        sm = SceneManager()
        sm.open_settings()
        sm.close_drawer()
        assert sm.state == GameState.PLAYING
        assert not sm.drawer_open

    def test_open_lineage(self):
        sm = SceneManager()
        sm.open_lineage()
        assert sm.state == GameState.LINEAGE

    def test_drawer_progress_starts_at_zero(self):
        sm = SceneManager()
        sm.open_settings()
        assert sm.drawer_progress == 0.0

    def test_update_animates_drawer(self):
        sm = SceneManager()
        sm.open_settings()
        sm.update(0.15)  # 150ms, half of 300ms animation
        assert 0.0 < sm.drawer_progress < 1.0

    def test_drawer_clamps_to_one(self):
        sm = SceneManager()
        sm.open_settings()
        sm.update(1.0)  # way past 300ms
        assert sm.drawer_progress == 1.0
