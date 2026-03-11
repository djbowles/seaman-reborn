"""Tests for the render engine, gradient cache, and particle system."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 200
_surface_mock.get_height.return_value = 200
_pygame_mock.Surface.return_value = _surface_mock
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

import pytest  # noqa: E402

from seaman_brain.gui.render_engine import GradientCache, ParticleSystem  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    # Each Surface() call returns a fresh MagicMock so identity checks work
    _pygame_mock.Surface.side_effect = lambda *a, **k: MagicMock()
    import seaman_brain.gui.render_engine as mod
    mod.pygame = _pygame_mock
    yield
    _pygame_mock.Surface.side_effect = None


class TestGradientCache:
    def test_get_creates_surface(self):
        cache = GradientCache()
        surf = cache.get((210, 140, 80), 100)
        assert surf is not None

    def test_same_params_returns_cached(self):
        cache = GradientCache()
        s1 = cache.get((210, 140, 80), 100)
        s2 = cache.get((210, 140, 80), 100)
        assert s1 is s2

    def test_different_color_creates_new(self):
        cache = GradientCache()
        s1 = cache.get((210, 140, 80), 100)
        s2 = cache.get((80, 130, 210), 100)
        assert s1 is not s2

    def test_invalidate_clears_cache(self):
        cache = GradientCache()
        s1 = cache.get((210, 140, 80), 100)
        cache.invalidate()
        s2 = cache.get((210, 140, 80), 100)
        assert s1 is not s2


class TestParticleSystem:
    def test_init_creates_particles(self):
        ps = ParticleSystem(count=10, bounds=(0, 0, 800, 600))
        assert len(ps.particles) == 10

    def test_update_moves_particles_upward(self):
        ps = ParticleSystem(count=1, bounds=(0, 0, 800, 600))
        initial_y = ps.particles[0]["y"]
        ps.update(1.0)
        assert ps.particles[0]["y"] < initial_y  # moved up

    def test_particles_respawn_when_off_top(self):
        ps = ParticleSystem(count=1, bounds=(0, 0, 800, 600))
        ps.particles[0]["y"] = -5  # force off-screen
        ps.update(0.01)
        assert ps.particles[0]["y"] > 0  # respawned at bottom
