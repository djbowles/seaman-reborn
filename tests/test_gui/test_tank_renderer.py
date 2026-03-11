"""Tests for the void tank renderer."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 800
_surface_mock.get_height.return_value = 600
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.tank_renderer import TankRenderer  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.tank_renderer as mod
    mod.pygame = _pygame_mock
    yield


class TestTankRenderer:
    def test_construction(self):
        tr = TankRenderer(800, 600)
        assert tr is not None

    def test_render_calls_fill(self):
        tr = TankRenderer(800, 600)
        surface = MagicMock()
        tr.render(surface)
        surface.fill.assert_called()

    def test_update_does_not_crash(self):
        tr = TankRenderer(800, 600)
        tr.update(0.016)  # one frame at 60fps

    def test_resize(self):
        tr = TankRenderer(800, 600)
        tr.resize(1024, 768)
        # Should not crash, particles should update bounds
