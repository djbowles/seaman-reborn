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

from seaman_brain.gui.render_engine import (  # noqa: E402
    GradientCache,
    NotificationManager,
    ParticleSystem,
    render_evolution,
    render_game_over,
)


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


class TestGameOverOverlay:
    def test_render_does_not_crash(self):
        surface = MagicMock()
        surface.get_width.return_value = 1024
        surface.get_height.return_value = 768
        render_game_over(surface, 1024, 768, cause="starvation")

    def test_render_with_different_cause(self):
        surface = MagicMock()
        render_game_over(surface, 800, 600, cause="neglect")
        assert surface.blit.called


class TestEvolutionOverlay:
    def test_render_does_not_crash(self):
        surface = MagicMock()
        render_evolution(surface, 1024, 768, stage_name="Gillman", progress=0.5)

    def test_render_at_full_progress(self):
        surface = MagicMock()
        render_evolution(surface, 800, 600, stage_name="Podman", progress=1.0)
        assert surface.blit.called


class TestNotificationManager:
    def test_add_notification(self):
        nm = NotificationManager()
        nm.add("Test notification")
        assert len(nm._notifications) == 1

    def test_notifications_expire(self):
        nm = NotificationManager()
        nm.add("Test")
        nm.update(10.0)  # way past expiry
        assert len(nm._notifications) == 0

    def test_multiple_notifications_stack(self):
        nm = NotificationManager()
        nm.add("First")
        nm.add("Second")
        nm.add("Third")
        assert len(nm._notifications) == 3

    def test_render_does_not_crash(self):
        nm = NotificationManager()
        nm.add("Toast message")
        surface = MagicMock()
        nm.render(surface, 800, 600)

    def test_max_notifications_enforced(self):
        nm = NotificationManager()
        for i in range(10):
            nm.add(f"Msg {i}")
        assert len(nm._notifications) <= 5
