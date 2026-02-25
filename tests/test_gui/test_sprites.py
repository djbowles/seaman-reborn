"""Tests for the creature sprite and animation system (US-037, US-052).

Pygame is mocked to avoid requiring a display server in CI.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

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

# Draw mock — returns None
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.ellipse.return_value = None
_pygame_mock.draw.line.return_value = None
_pygame_mock.draw.lines.return_value = None
_pygame_mock.draw.polygon.return_value = None

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)

# Install pygame mock before importing sprites
sys.modules["pygame"] = _pygame_mock

# Now import the module under test (pygame is mocked)
from seaman_brain.config import GUIConfig  # noqa: E402
from seaman_brain.creature.genome import ALL_TRAITS, CreatureGenome  # noqa: E402
from seaman_brain.gui.sprites import (  # noqa: E402
    AnimationState,
    CreaturePosition,
    CreatureRenderer,
)
from seaman_brain.types import CreatureStage  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset draw mocks and re-install pygame mock between tests."""
    sys.modules["pygame"] = _pygame_mock
    import seaman_brain.gui.sprites as sprites_mod
    sprites_mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)
    yield


# ── Construction Tests ────────────────────────────────────────────────


class TestCreatureRendererConstruction:
    """Tests for CreatureRenderer initialization."""

    def test_default_stage_is_mushroomer(self):
        """CreatureRenderer defaults to MUSHROOMER stage."""
        renderer = CreatureRenderer()
        assert renderer.stage == CreatureStage.MUSHROOMER

    def test_default_animation_is_idle(self):
        """Default animation state is IDLE."""
        renderer = CreatureRenderer()
        assert renderer.animation_state == AnimationState.IDLE

    def test_custom_stage(self):
        """CreatureRenderer accepts custom stage."""
        renderer = CreatureRenderer(stage=CreatureStage.FROGMAN)
        assert renderer.stage == CreatureStage.FROGMAN

    def test_default_position_centered(self):
        """Creature starts at center of render area."""
        renderer = CreatureRenderer()
        assert renderer.position.x == pytest.approx(512.0)  # 1024 / 2
        assert renderer.position.y > 0  # Not at origin

    def test_custom_gui_config(self):
        """CreatureRenderer accepts custom GUIConfig."""
        cfg = GUIConfig(window_width=800, window_height=600)
        renderer = CreatureRenderer(gui_config=cfg)
        assert renderer.gui_config.window_width == 800
        assert renderer.position.x == pytest.approx(400.0)  # 800 / 2


# ── Stage Rendering Tests ────────────────────────────────────────────


class TestStageRendering:
    """Tests for each stage's visual rendering."""

    def test_render_mushroomer(self):
        """Mushroomer renders without error and makes draw calls."""
        renderer = CreatureRenderer(stage=CreatureStage.MUSHROOMER)
        renderer.render(_surface_mock)

        # Mushroomer uses ellipse (cap, body, eye) and circle (eye)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.circle.called

    def test_render_gillman(self):
        """Gillman renders with fish body, fins, and eyes."""
        renderer = CreatureRenderer(stage=CreatureStage.GILLMAN)
        renderer.render(_surface_mock)

        # Gillman uses ellipse (body, mouth), polygon (tail, dorsal fin), line (gills)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.polygon.called
        assert _pygame_mock.draw.line.called

    def test_render_podfish(self):
        """Podfish renders with body, legs, and fins."""
        renderer = CreatureRenderer(stage=CreatureStage.PODFISH)
        renderer.render(_surface_mock)

        # Podfish uses ellipse (body, legs), polygon (tail, fins)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.polygon.called

    def test_render_tadman(self):
        """Tadman renders with round head, limbs, and tail."""
        renderer = CreatureRenderer(stage=CreatureStage.TADMAN)
        renderer.render(_surface_mock)

        # Tadman uses circle (head), line (arms, legs), polygon (tail)
        assert _pygame_mock.draw.circle.called
        assert _pygame_mock.draw.line.called
        assert _pygame_mock.draw.polygon.called

    def test_render_frogman(self):
        """Frogman renders with body, bulging eyes, wide mouth, legs."""
        renderer = CreatureRenderer(stage=CreatureStage.FROGMAN)
        renderer.render(_surface_mock)

        # Frogman uses ellipse (body, belly, mouth), line (limbs), circle (eyes)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.line.called
        assert _pygame_mock.draw.circle.called

    def test_all_stages_render_without_error(self):
        """Every stage renders without exceptions."""
        for stage in CreatureStage:
            renderer = CreatureRenderer(stage=stage)
            renderer.render(_surface_mock)  # No exception


# ── Animation State Tests ────────────────────────────────────────────


class TestAnimationStates:
    """Tests for animation state transitions."""

    def test_set_animation_talking(self):
        """Setting animation to TALKING opens the mouth."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.TALKING)
        assert renderer.animation_state == AnimationState.TALKING
        assert renderer._mouth_open > 0

    def test_set_animation_idle(self):
        """Setting animation to IDLE is accepted."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.SWIMMING)
        renderer.set_animation(AnimationState.IDLE)
        assert renderer.animation_state == AnimationState.IDLE

    def test_set_animation_sleeping(self):
        """Setting animation to SLEEPING is accepted."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.SLEEPING)
        assert renderer.animation_state == AnimationState.SLEEPING

    def test_sleeping_renders_zs(self):
        """Sleeping creature renders Z letters."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.SLEEPING)
        renderer._time = 1.0  # Advance time for Z animation

        renderer.render(_surface_mock)

        # Z's are drawn as lines (3 lines per Z, up to 3 Z's)
        assert _pygame_mock.draw.line.call_count > 0

    def test_talking_mouth_oscillates(self):
        """Mouth oscillates during TALKING animation."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.TALKING)

        mouth_values = []
        for _ in range(10):
            renderer.update(0.05)
            mouth_values.append(renderer._mouth_open)

        # Mouth should vary (not all the same value)
        assert len(set(round(v, 3) for v in mouth_values)) > 1

    def test_mouth_closes_after_talking_stops(self):
        """Mouth gradually closes when animation changes from TALKING."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.TALKING)
        renderer.update(0.1)
        assert renderer._mouth_open > 0

        renderer.set_animation(AnimationState.IDLE)
        # Update enough for mouth to close
        for _ in range(20):
            renderer.update(0.1)
        assert renderer._mouth_open == pytest.approx(0.0, abs=0.01)

    def test_animation_state_enum_values(self):
        """AnimationState enum has all required values."""
        assert AnimationState.IDLE.value == "idle"
        assert AnimationState.SWIMMING.value == "swimming"
        assert AnimationState.TALKING.value == "talking"
        assert AnimationState.EATING.value == "eating"
        assert AnimationState.SLEEPING.value == "sleeping"


# ── Position Interpolation Tests ─────────────────────────────────────


class TestPositionInterpolation:
    """Tests for smooth position movement."""

    def test_move_toward_target(self):
        """Position smoothly moves toward target."""
        pos = CreaturePosition(x=0.0, y=0.0, target_x=100.0, target_y=0.0)
        pos.move_toward_target(0.5)

        # Should have moved toward target
        assert pos.x > 0.0
        assert pos.x < 100.0

    def test_reaches_target(self):
        """Position reaches target after enough updates."""
        pos = CreaturePosition(x=0.0, y=0.0, target_x=50.0, target_y=50.0)

        for _ in range(100):
            pos.move_toward_target(0.1)

        assert pos.at_target()
        assert pos.x == pytest.approx(50.0, abs=1.0)
        assert pos.y == pytest.approx(50.0, abs=1.0)

    def test_at_target_initially_true(self):
        """at_target() is True when position equals target."""
        pos = CreaturePosition(x=100.0, y=200.0, target_x=100.0, target_y=200.0)
        assert pos.at_target()

    def test_set_target(self):
        """set_target() updates target coordinates."""
        pos = CreaturePosition()
        pos.set_target(300.0, 400.0)
        assert pos.target_x == 300.0
        assert pos.target_y == 400.0

    def test_zero_dt_no_movement(self):
        """No movement when dt is zero."""
        pos = CreaturePosition(x=0.0, y=0.0, target_x=100.0, target_y=0.0)
        pos.move_toward_target(0.0)
        assert pos.x == 0.0

    def test_speed_affects_movement(self):
        """Higher speed means more movement per frame."""
        slow = CreaturePosition(
            x=0.0, y=0.0, target_x=100.0, target_y=0.0, speed=30.0
        )
        fast = CreaturePosition(
            x=0.0, y=0.0, target_x=100.0, target_y=0.0, speed=120.0
        )

        slow.move_toward_target(0.1)
        fast.move_toward_target(0.1)

        assert fast.x > slow.x


# ── Face Tracking Tests ──────────────────────────────────────────────


class TestFaceTracking:
    """Tests for mouse cursor face tracking."""

    def test_set_mouse_position(self):
        """set_mouse_position stores cursor coordinates."""
        renderer = CreatureRenderer()
        renderer.set_mouse_position(300.0, 400.0)
        assert renderer._mouse_x == 300.0
        assert renderer._mouse_y == 400.0

    def test_eye_tracks_mouse_right(self):
        """Eye pupil shifts toward mouse position on the right."""
        renderer = CreatureRenderer()
        renderer.position.x = 512.0
        renderer.position.y = 400.0
        renderer.set_mouse_position(1000.0, 400.0)

        renderer.render(_surface_mock)

        # Eye should have been drawn (circle calls exist)
        assert _pygame_mock.draw.circle.called

    def test_eye_tracks_mouse_left(self):
        """Eye pupil shifts toward mouse position on the left."""
        renderer = CreatureRenderer()
        renderer.position.x = 512.0
        renderer.position.y = 400.0
        renderer.set_mouse_position(10.0, 400.0)

        renderer.render(_surface_mock)

        assert _pygame_mock.draw.circle.called


# ── Blinking Tests ───────────────────────────────────────────────────


class TestBlinking:
    """Tests for eye blinking animation."""

    def test_not_blinking_initially(self):
        """Creature is not blinking at initialization."""
        renderer = CreatureRenderer()
        assert not renderer._is_blinking

    def test_blink_eventually_triggers(self):
        """After enough time, a blink triggers."""
        renderer = CreatureRenderer()
        # Force blink by advancing blink timer past threshold
        renderer._blink_timer = 6.0  # > max 5.0 uniform
        renderer.update(0.01)

        assert renderer._is_blinking

    def test_blink_ends_after_duration(self):
        """Blink ends after the blink duration (~0.15s)."""
        renderer = CreatureRenderer()
        renderer._is_blinking = True
        renderer._blink_timer = 0.0

        renderer.update(0.2)  # > 0.15 blink duration
        assert not renderer._is_blinking


# ── Stage Switching Tests ────────────────────────────────────────────


class TestStageSwitching:
    """Tests for changing creature stage."""

    def test_set_stage(self):
        """set_stage changes the creature's stage."""
        renderer = CreatureRenderer()
        assert renderer.stage == CreatureStage.MUSHROOMER

        renderer.set_stage(CreatureStage.GILLMAN)
        assert renderer.stage == CreatureStage.GILLMAN

    def test_render_after_stage_change(self):
        """Render works correctly after changing stage."""
        renderer = CreatureRenderer()
        renderer.set_stage(CreatureStage.PODFISH)

        renderer.render(_surface_mock)

        # Should still draw (no error)
        assert _pygame_mock.draw.ellipse.called


# ── Update Loop Tests ────────────────────────────────────────────────


class TestUpdateLoop:
    """Tests for the update tick."""

    def test_update_advances_time(self):
        """update() advances internal time counter."""
        renderer = CreatureRenderer()
        renderer.update(0.5)
        assert renderer._time == pytest.approx(0.5)
        renderer.update(0.3)
        assert renderer._time == pytest.approx(0.8)

    def test_update_moves_position(self):
        """update() interpolates position toward target."""
        renderer = CreatureRenderer()
        start_x = renderer.position.x
        renderer.position.set_target(start_x + 200, renderer.position.y)

        renderer.update(0.5)
        assert renderer.position.x > start_x

    def test_idle_wander_picks_new_target(self):
        """Idle creature picks new wander target after timer expires."""
        renderer = CreatureRenderer()
        renderer.set_animation(AnimationState.IDLE)

        # Force at target and timer expired
        renderer.position.target_x = renderer.position.x
        renderer.position.target_y = renderer.position.y
        renderer._idle_wander_timer = 3.5  # > 3.0 threshold

        renderer.update(0.1)

        # The wander timer should have reset after picking a new target
        assert renderer._idle_wander_timer < 3.5


# ── Eating Effect Tests ──────────────────────────────────────────────


class TestEatingEffect:
    """Tests for eating visual effect."""

    def test_render_eating_effect(self):
        """render_eating_effect draws food particles."""
        renderer = CreatureRenderer()
        renderer._time = 1.0

        renderer.render_eating_effect(_surface_mock, 200.0, 300.0)

        # Food particles drawn as circles
        assert _pygame_mock.draw.circle.called
        assert _pygame_mock.draw.circle.call_count == 4  # 4 particles


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge case and error handling tests."""

    def test_render_at_zero_position(self):
        """Rendering at (0, 0) doesn't crash."""
        renderer = CreatureRenderer()
        renderer.position.x = 0.0
        renderer.position.y = 0.0
        renderer.render(_surface_mock)  # No exception

    def test_render_with_zero_time(self):
        """Rendering with _time=0 doesn't crash."""
        renderer = CreatureRenderer()
        renderer._time = 0.0
        renderer.render(_surface_mock)  # No exception

    def test_update_with_zero_dt(self):
        """update() with dt=0 doesn't crash."""
        renderer = CreatureRenderer()
        renderer.update(0.0)
        assert renderer._time == 0.0

    def test_update_with_large_dt(self):
        """update() with large dt doesn't crash or produce NaN."""
        renderer = CreatureRenderer()
        renderer.update(100.0)
        assert renderer._time == pytest.approx(100.0)

    def test_set_bounds(self):
        """set_bounds updates wander area."""
        renderer = CreatureRenderer()
        renderer.set_bounds(10, 50, 800, 600)
        assert renderer._bounds_x == 10
        assert renderer._bounds_y == 50
        assert renderer._bounds_w == 800
        assert renderer._bounds_h == 600

    def test_mouse_at_creature_position(self):
        """Mouse at exact creature position doesn't crash (zero dist)."""
        renderer = CreatureRenderer()
        renderer.set_mouse_position(renderer.position.x, renderer.position.y)
        renderer.render(_surface_mock)  # No exception (zero direction handled)


# ── Genome-Driven Rendering Tests (US-052) ───────────────────────────


class TestGenomeDrivenRendering:
    """Tests for genome-parameterized visual variations."""

    def test_different_genomes_produce_different_renders(self):
        """Two different genomes produce visually different draw calls."""
        g1 = CreatureGenome(traits={
            "body_size": 0.1, "hue": 0.1, "saturation": 0.1,
            "eye_size": 0.1, "fin_length": 0.1, "limb_proportion": 0.1,
        })
        g2 = CreatureGenome(traits={
            "body_size": 0.9, "hue": 0.9, "saturation": 0.9,
            "eye_size": 0.9, "fin_length": 0.9, "limb_proportion": 0.9,
        })

        r1 = CreatureRenderer(stage=CreatureStage.PODFISH, genome=g1)
        _pygame_mock.draw.reset_mock()
        r1.render(_surface_mock)
        calls_1 = _pygame_mock.draw.method_calls.copy()

        r2 = CreatureRenderer(stage=CreatureStage.PODFISH, genome=g2)
        _pygame_mock.draw.reset_mock()
        r2.render(_surface_mock)
        calls_2 = _pygame_mock.draw.method_calls.copy()

        # Draw calls should differ (different sizes, colors, pattern elements)
        assert calls_1 != calls_2

    def test_genome_extremes_do_not_break_rendering(self):
        """Extreme genome values (all 0.0 and all 1.0) don't crash."""
        for val in (0.0, 1.0):
            traits = {t: val for t in ALL_TRAITS}
            genome = CreatureGenome(traits=traits)
            for stage in CreatureStage:
                renderer = CreatureRenderer(stage=stage, genome=genome)
                renderer.render(_surface_mock)  # No exception

    def test_default_genome_matches_original_draw_count(self):
        """Default genome (all 0.5) produces same draw call count as no genome."""
        # Without genome
        r1 = CreatureRenderer(stage=CreatureStage.GILLMAN)
        _pygame_mock.draw.reset_mock()
        r1.render(_surface_mock)
        count_no_genome = len(_pygame_mock.draw.method_calls)

        # With default genome (all traits = 0.5)
        default_genome = CreatureGenome()
        r2 = CreatureRenderer(stage=CreatureStage.GILLMAN, genome=default_genome)
        _pygame_mock.draw.reset_mock()
        r2.render(_surface_mock)
        count_with_genome = len(_pygame_mock.draw.method_calls)

        assert count_no_genome == count_with_genome

    def test_body_size_scales_rendering(self):
        """body_size genome trait changes the base size used for drawing."""
        small = CreatureGenome(traits={"body_size": 0.0})
        large = CreatureGenome(traits={"body_size": 1.0})

        r_small = CreatureRenderer(stage=CreatureStage.MUSHROOMER, genome=small)
        _pygame_mock.draw.reset_mock()
        r_small.render(_surface_mock)
        small_calls = _pygame_mock.draw.method_calls.copy()

        r_large = CreatureRenderer(stage=CreatureStage.MUSHROOMER, genome=large)
        _pygame_mock.draw.reset_mock()
        r_large.render(_surface_mock)
        large_calls = _pygame_mock.draw.method_calls.copy()

        # Different body sizes produce different draw arguments
        assert small_calls != large_calls

    def test_hue_saturation_shift_colors(self):
        """Hue/saturation genome traits shift the color palette."""
        cool = CreatureGenome(traits={"hue": 0.0, "saturation": 0.0})
        warm = CreatureGenome(traits={"hue": 1.0, "saturation": 1.0})

        r_cool = CreatureRenderer(stage=CreatureStage.FROGMAN, genome=cool)
        _pygame_mock.draw.reset_mock()
        r_cool.render(_surface_mock)
        cool_calls = _pygame_mock.draw.method_calls.copy()

        r_warm = CreatureRenderer(stage=CreatureStage.FROGMAN, genome=warm)
        _pygame_mock.draw.reset_mock()
        r_warm.render(_surface_mock)
        warm_calls = _pygame_mock.draw.method_calls.copy()

        assert cool_calls != warm_calls

    def test_pattern_complexity_adds_draw_calls(self):
        """High pattern_complexity adds extra draw calls (spots/stripes)."""
        plain = CreatureGenome(traits={"pattern_complexity": 0.0})
        fancy = CreatureGenome(traits={"pattern_complexity": 1.0})

        r_plain = CreatureRenderer(stage=CreatureStage.GILLMAN, genome=plain)
        _pygame_mock.draw.reset_mock()
        r_plain.render(_surface_mock)
        plain_count = len(_pygame_mock.draw.method_calls)

        r_fancy = CreatureRenderer(stage=CreatureStage.GILLMAN, genome=fancy)
        _pygame_mock.draw.reset_mock()
        r_fancy.render(_surface_mock)
        fancy_count = len(_pygame_mock.draw.method_calls)

        # Fancy genome should produce more draw calls (spots + stripes)
        assert fancy_count > plain_count

    def test_set_genome_method(self):
        """set_genome() updates the genome used for rendering."""
        renderer = CreatureRenderer()
        assert renderer.genome is None

        genome = CreatureGenome(traits={"body_size": 0.8})
        renderer.set_genome(genome)
        assert renderer.genome is genome

        renderer.set_genome(None)
        assert renderer.genome is None

    def test_face_expressiveness_affects_mouth_amplitude(self):
        """face_expressiveness genome trait modifies mouth animation range."""
        stoic = CreatureGenome(traits={"face_expressiveness": 0.0})
        expressive = CreatureGenome(traits={"face_expressiveness": 1.0})

        r_stoic = CreatureRenderer(genome=stoic)
        r_stoic.set_animation(AnimationState.TALKING)
        r_expressive = CreatureRenderer(genome=expressive)
        r_expressive.set_animation(AnimationState.TALKING)

        # Collect mouth values over time
        stoic_max = 0.0
        expressive_max = 0.0
        for _ in range(20):
            r_stoic.update(0.05)
            r_expressive.update(0.05)
            stoic_max = max(stoic_max, r_stoic._mouth_open)
            expressive_max = max(expressive_max, r_expressive._mouth_open)

        # Expressive creature should have larger mouth amplitude
        assert expressive_max > stoic_max

    def test_all_stages_render_with_genome(self):
        """Every stage renders without error when a genome is set."""
        genome = CreatureGenome(traits={
            "body_size": 0.7, "eye_size": 0.3, "fin_length": 0.8,
            "limb_proportion": 0.6, "hue": 0.3, "saturation": 0.7,
            "pattern_complexity": 0.8, "face_expressiveness": 0.6,
        })
        for stage in CreatureStage:
            renderer = CreatureRenderer(stage=stage, genome=genome)
            renderer.render(_surface_mock)  # No exception
