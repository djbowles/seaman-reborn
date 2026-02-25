"""Tests for the TankRenderer (US-036).

Pygame is mocked to avoid requiring a display server in CI.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

# ── Pygame Mock Setup ─────────────────────────────────────────────────

_pygame_mock = MagicMock()
_pygame_mock.SRCALPHA = 65536
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None

# Surface mock
_surface_mock = MagicMock()
_surface_mock.get_width.return_value = 1024
_surface_mock.get_height.return_value = 768

# Overlay surface returned by pygame.Surface()
_overlay_mock = MagicMock()
_pygame_mock.Surface.return_value = _overlay_mock

# Clock mock
_clock_mock = MagicMock()
_clock_mock.tick.return_value = 33
_pygame_mock.time.Clock.return_value = _clock_mock

# Draw mock — returns None
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.ellipse.return_value = None
_pygame_mock.draw.lines.return_value = None
_pygame_mock.draw.polygon.return_value = None

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)


@pytest.fixture(autouse=True)
def _mock_pygame():
    """Patch pygame for all tests."""
    with patch.dict(sys.modules, {"pygame": _pygame_mock}):
        # Force reimport to pick up mock
        for mod in list(sys.modules):
            if mod.startswith("seaman_brain.gui.tank_renderer"):
                del sys.modules[mod]
        yield


# ── Construction Tests ────────────────────────────────────────────────


class TestTankRendererConstruction:
    """Tests for TankRenderer initialization."""

    def test_default_config(self):
        """TankRenderer uses default configs when none provided."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        assert renderer.config.window_width == 1024
        assert renderer.config.window_height == 768

    def test_custom_gui_config(self):
        """TankRenderer accepts custom GUIConfig."""
        from seaman_brain.config import GUIConfig
        from seaman_brain.gui.tank_renderer import TankRenderer

        cfg = GUIConfig(window_width=800, window_height=600)
        renderer = TankRenderer(gui_config=cfg)
        assert renderer.config.window_width == 800
        assert renderer.config.window_height == 600

    def test_custom_env_config(self):
        """TankRenderer accepts custom EnvironmentConfig."""
        from seaman_brain.config import EnvironmentConfig
        from seaman_brain.gui.tank_renderer import TankRenderer

        cfg = EnvironmentConfig(optimal_temp_min=18.0, optimal_temp_max=30.0)
        renderer = TankRenderer(env_config=cfg)
        assert renderer.env_config.optimal_temp_min == 18.0

    def test_initial_transition_zero(self):
        """Transition progress starts at 0 (aquarium)."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        assert renderer.transition_progress == 0.0

    def test_render_area_dimensions(self):
        """Render area matches window size minus top margin."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        x, y, w, h = renderer.render_area
        assert x == 0
        assert y == 45  # top margin
        assert w == 1024
        assert h == 768 - 45


# ── Aquarium Rendering Tests ─────────────────────────────────────────


class TestAquariumRendering:
    """Tests for aquarium mode rendering."""

    def test_render_aquarium_calls_draw(self):
        """Rendering an aquarium makes draw calls on the surface."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)

        _pygame_mock.draw.reset_mock()
        renderer.render(_surface_mock, tank)

        # Should make rect calls (background, ground)
        assert _pygame_mock.draw.rect.called
        # Should make circle calls (gravel, bubbles)
        assert _pygame_mock.draw.circle.called

    def test_render_aquarium_draws_water_surface(self):
        """Aquarium mode draws water surface wave line."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)

        _pygame_mock.draw.reset_mock()
        renderer.render(_surface_mock, tank)

        # Water surface uses draw.lines for the wave
        assert _pygame_mock.draw.lines.called

    def test_render_aquarium_draws_bubbles(self):
        """Aquarium mode draws bubbles as circles."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)

        # Ensure some bubbles exist
        renderer._init_decorations()
        assert len(renderer._bubbles) > 0

        _pygame_mock.draw.reset_mock()
        renderer.render(_surface_mock, tank)

        # Circles should be drawn for bubbles + gravel
        assert _pygame_mock.draw.circle.call_count > 0


# ── Terrarium Rendering Tests ────────────────────────────────────────


class TestTerrariumRendering:
    """Tests for terrarium mode rendering."""

    def test_render_terrarium_no_water_surface(self):
        """Terrarium mode does not draw water surface."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        # Force transition to complete terrarium
        renderer._transition_progress = 1.0

        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        _pygame_mock.draw.reset_mock()
        renderer.render(_surface_mock, tank)

        # Water surface uses draw.lines — should NOT be called in full terrarium
        assert not _pygame_mock.draw.lines.called

    def test_render_terrarium_draws_rocks(self):
        """Terrarium mode draws rock decorations."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._transition_progress = 1.0

        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        _pygame_mock.draw.reset_mock()
        renderer.render(_surface_mock, tank)

        # Rocks are drawn as ellipses
        assert _pygame_mock.draw.ellipse.called

    def test_render_terrarium_draws_moisture(self):
        """Terrarium mode draws moisture particles."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._transition_progress = 1.0

        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        # Ensure moisture exists
        renderer._init_decorations()
        assert len(renderer._moisture) > 0

        _pygame_mock.draw.reset_mock()
        renderer.render(_surface_mock, tank)

        # Moisture particles drawn as circles
        assert _pygame_mock.draw.circle.called


# ── Temperature Overlay Tests ────────────────────────────────────────


class TestTemperatureOverlay:
    """Tests for temperature-based color overlay."""

    def test_no_overlay_in_optimal_range(self):
        """No temperature overlay when temperature is optimal."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(temperature=24.0)  # Default optimal

        _pygame_mock.Surface.reset_mock()
        _surface_mock.blit.reset_mock()

        # Render just temperature overlay
        renderer._render_temperature_overlay(_surface_mock, tank.temperature)

        # No overlay surface created
        _pygame_mock.Surface.assert_not_called()

    def test_cold_overlay_below_optimal(self):
        """Blue tint overlay when temperature is below optimal."""
        from seaman_brain.config import EnvironmentConfig
        from seaman_brain.gui.tank_renderer import TankRenderer

        cfg = EnvironmentConfig(optimal_temp_min=20.0, lethal_temp_min=10.0)
        renderer = TankRenderer(env_config=cfg)
        renderer._init_decorations()

        _pygame_mock.Surface.reset_mock()
        _overlay_mock.reset_mock()
        _surface_mock.blit.reset_mock()

        renderer._render_temperature_overlay(_surface_mock, 15.0)

        # Overlay surface created and blitted
        _pygame_mock.Surface.assert_called_once()
        _overlay_mock.fill.assert_called_once()

        # Check the fill color — should be blue-ish (r=30, g=60, b=180, a>0)
        fill_args = _overlay_mock.fill.call_args[0][0]
        assert fill_args[2] > fill_args[0]  # blue > red

    def test_hot_overlay_above_optimal(self):
        """Red tint overlay when temperature is above optimal."""
        from seaman_brain.config import EnvironmentConfig
        from seaman_brain.gui.tank_renderer import TankRenderer

        cfg = EnvironmentConfig(optimal_temp_max=28.0, lethal_temp_max=38.0)
        renderer = TankRenderer(env_config=cfg)
        renderer._init_decorations()

        _pygame_mock.Surface.reset_mock()
        _overlay_mock.reset_mock()
        _surface_mock.blit.reset_mock()

        renderer._render_temperature_overlay(_surface_mock, 34.0)

        _pygame_mock.Surface.assert_called_once()
        _overlay_mock.fill.assert_called_once()

        # Check the fill color — should be red-ish (r=200, g=50, b=20, a>0)
        fill_args = _overlay_mock.fill.call_args[0][0]
        assert fill_args[0] > fill_args[2]  # red > blue


# ── Cleanliness Overlay Tests ────────────────────────────────────────


class TestCleanlinessOverlay:
    """Tests for cleanliness-based murkiness overlay."""

    def test_no_overlay_when_clean(self):
        """No cleanliness overlay when tank is spotless."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._init_decorations()

        _pygame_mock.Surface.reset_mock()
        _surface_mock.blit.reset_mock()

        renderer._render_cleanliness_overlay(_surface_mock, 1.0, 0.0)

        _pygame_mock.Surface.assert_not_called()

    def test_murky_overlay_when_dirty(self):
        """Murky overlay appears when tank is dirty."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._init_decorations()

        _pygame_mock.Surface.reset_mock()
        _overlay_mock.reset_mock()
        _surface_mock.blit.reset_mock()

        renderer._render_cleanliness_overlay(_surface_mock, 0.2, 0.0)

        # Overlay should be created and applied
        _pygame_mock.Surface.assert_called_once()
        _overlay_mock.fill.assert_called_once()
        _surface_mock.blit.assert_called()

    def test_dirtiest_has_strongest_overlay(self):
        """Filthy tank (cleanliness=0) has strongest overlay alpha."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._init_decorations()

        # Moderately dirty
        _overlay_mock.reset_mock()
        _pygame_mock.Surface.reset_mock()
        renderer._render_cleanliness_overlay(_surface_mock, 0.5, 0.0)
        moderate_fill = _overlay_mock.fill.call_args[0][0]

        # Very dirty
        _overlay_mock.reset_mock()
        _pygame_mock.Surface.reset_mock()
        renderer._render_cleanliness_overlay(_surface_mock, 0.0, 0.0)
        filthy_fill = _overlay_mock.fill.call_args[0][0]

        # Filthy should have higher alpha
        assert filthy_fill[3] > moderate_fill[3]


# ── Animation Update Tests ───────────────────────────────────────────


class TestAnimationUpdate:
    """Tests for update() animation logic."""

    def test_update_advances_time(self):
        """update() advances internal time counter."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment()

        renderer.update(0.5, tank)
        assert renderer._time == pytest.approx(0.5)
        renderer.update(0.3, tank)
        assert renderer._time == pytest.approx(0.8)

    def test_transition_moves_toward_terrarium(self):
        """Transition progress increases when tank is terrarium."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        assert renderer.transition_progress == 0.0

        # Update several times to animate transition
        for _ in range(20):
            renderer.update(0.1, tank)

        # Transition should have progressed toward 1.0
        assert renderer.transition_progress > 0.0
        assert renderer.transition_progress <= 1.0

    def test_transition_moves_toward_aquarium(self):
        """Transition progress decreases when tank is aquarium."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._transition_progress = 1.0  # Start as terrarium

        tank = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)

        for _ in range(20):
            renderer.update(0.1, tank)

        assert renderer.transition_progress < 1.0
        assert renderer.transition_progress >= 0.0

    def test_bubbles_respawn_after_rising(self):
        """Bubbles respawn when they rise off screen."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment()

        renderer._init_decorations()
        initial_count = len(renderer._bubbles)
        assert initial_count > 0

        # Update many frames to move bubbles off-screen and trigger respawn
        for _ in range(200):
            renderer.update(0.1, tank)

        # Should still have bubbles (respawning keeps pool alive)
        assert len(renderer._bubbles) > 0

    def test_moisture_respawns_in_terrarium(self):
        """Moisture particles respawn when floating off-screen."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        renderer._transition_progress = 1.0

        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)
        renderer._init_decorations()

        initial_count = len(renderer._moisture)
        assert initial_count > 0

        for _ in range(200):
            renderer.update(0.1, tank)

        assert len(renderer._moisture) > 0


# ── Transition Smoothness Tests ──────────────────────────────────────


class TestTransitionSmoothness:
    """Tests for smooth aquarium <-> terrarium transition."""

    def test_transition_is_gradual(self):
        """Transition doesn't jump instantly — it interpolates smoothly."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        renderer.update(0.1, tank)

        # After one frame, should NOT be at 1.0 yet (gradual)
        assert 0.0 < renderer.transition_progress < 1.0

    def test_transition_reaches_target(self):
        """Transition eventually reaches target value."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        # Run many frames
        for _ in range(100):
            renderer.update(0.1, tank)

        assert renderer.transition_progress == pytest.approx(1.0)

    def test_transition_does_not_overshoot(self):
        """Transition clamps to [0, 1] and does not overshoot."""
        from seaman_brain.environment.tank import EnvironmentType, TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(environment_type=EnvironmentType.TERRARIUM)

        for _ in range(200):
            renderer.update(0.5, tank)

        assert renderer.transition_progress == pytest.approx(1.0)
        assert renderer.transition_progress <= 1.0

        # Now switch back to aquarium
        tank_aq = TankEnvironment(environment_type=EnvironmentType.AQUARIUM)
        for _ in range(200):
            renderer.update(0.5, tank_aq)

        assert renderer.transition_progress == pytest.approx(0.0)
        assert renderer.transition_progress >= 0.0


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case tests."""

    def test_render_with_zero_cleanliness(self):
        """Rendering doesn't crash with cleanliness=0."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment(cleanliness=0.0)
        renderer.render(_surface_mock, tank)  # No exception

    def test_render_with_extreme_temperature(self):
        """Rendering doesn't crash with extreme temperatures."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()

        tank_cold = TankEnvironment(temperature=5.0)
        renderer.render(_surface_mock, tank_cold)  # No exception

        tank_hot = TankEnvironment(temperature=45.0)
        renderer.render(_surface_mock, tank_hot)  # No exception

    def test_multiple_renders_without_update(self):
        """Calling render multiple times without update is safe."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment()

        for _ in range(5):
            renderer.render(_surface_mock, tank)

    def test_update_with_zero_dt(self):
        """update() with dt=0 is a no-op for time."""
        from seaman_brain.environment.tank import TankEnvironment
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()
        tank = TankEnvironment()
        renderer.update(0.0, tank)
        assert renderer._time == 0.0

    def test_init_decorations_idempotent(self):
        """_init_decorations() only runs once."""
        from seaman_brain.gui.tank_renderer import TankRenderer

        renderer = TankRenderer()

        renderer._init_decorations()
        gravel_count = len(renderer._gravel_points)
        rocks_count = len(renderer._rocks)

        renderer._init_decorations()
        assert len(renderer._gravel_points) == gravel_count
        assert len(renderer._rocks) == rocks_count
