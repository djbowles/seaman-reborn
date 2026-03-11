"""Tests for the proportional layout engine."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.Rect = lambda x, y, w, h: type(
    "Rect", (), {"x": x, "y": y, "w": w, "h": h, "width": w, "height": h,
                  "left": x, "top": y, "right": x + w, "bottom": y + h,
                  "collidepoint": lambda self, px, py: (x <= px < x + w and y <= py < y + h)}
)()
sys.modules["pygame"] = _pygame_mock

import pytest  # noqa: E402

from seaman_brain.gui.layout import ScreenLayout  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_pygame_mock():
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.layout as mod
    mod.pygame = _pygame_mock
    yield


class TestScreenLayout:
    def test_default_1024x768(self):
        layout = ScreenLayout(1024, 768)
        assert layout.top_bar.h == 32
        assert layout.sidebar.w == 48
        assert layout.sidebar.x == 0
        assert layout.sidebar.y == 32
        assert layout.tank.x == 48
        assert layout.tank.y == 32
        assert layout.chat.h == 130
        assert layout.chat.y == 768 - 130

    def test_tank_fills_remaining_space(self):
        layout = ScreenLayout(1024, 768)
        # tank width = total - sidebar
        assert layout.tank.w == 1024 - 48
        # tank height = total - top_bar - chat
        assert layout.tank.h == 768 - 32 - 130

    def test_resize(self):
        layout = ScreenLayout(1024, 768)
        layout.resize(1920, 1080)
        assert layout.top_bar.w == 1920
        assert layout.chat.w == 1920
        assert layout.tank.w == 1920 - 48
        assert layout.tank.h == 1080 - 32 - 130

    def test_drawer_width(self):
        layout = ScreenLayout(1000, 768)
        assert layout.drawer_width == 400  # 40%

    def test_small_window(self):
        layout = ScreenLayout(640, 480)
        assert layout.tank.w == 640 - 48
        assert layout.tank.h == 480 - 32 - 130
