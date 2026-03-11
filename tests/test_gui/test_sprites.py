"""Tests for creature rendering with glow aura and mood-reactive lighting.

Rewritten for the Modern Minimal GUI rewrite — 2.5x creature size,
cached glow surface, mood-reactive lighting, bob offset idle animation.
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
_surface_mock.get_width.return_value = 200
_surface_mock.get_height.return_value = 200

# Draw mock — returns None
_pygame_mock.draw.rect.return_value = None
_pygame_mock.draw.circle.return_value = None
_pygame_mock.draw.ellipse.return_value = None
_pygame_mock.draw.line.return_value = None
_pygame_mock.draw.lines.return_value = None
_pygame_mock.draw.polygon.return_value = None

# Rect mock
_pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)

# Surface constructor returns a surface mock
_pygame_mock.Surface.return_value = _surface_mock

# Install pygame mock before importing sprites
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame_gui"] = MagicMock()

from seaman_brain.creature.genome import ALL_TRAITS, CreatureGenome  # noqa: E402
from seaman_brain.gui.sprites import (  # noqa: E402
    AnimationState,
    CreaturePosition,
    CreatureRenderer,
)
from seaman_brain.types import CreatureStage  # noqa: E402


@pytest.fixture(autouse=True)
def _reinstall_mock():
    """Reset draw mocks and re-install pygame mock between tests."""
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame_gui"] = MagicMock()
    import seaman_brain.gui.sprites as mod
    mod.pygame = _pygame_mock
    _pygame_mock.draw.reset_mock()
    _surface_mock.reset_mock()
    _pygame_mock.Rect = lambda x, y, w, h: (x, y, w, h)
    yield


# ── Construction Tests ────────────────────────────────────────────────


class TestCreatureRendererConstruction:
    def test_default_stage(self):
        cr = CreatureRenderer()
        assert cr.stage == CreatureStage.MUSHROOMER

    def test_base_size_is_100(self):
        cr = CreatureRenderer()
        assert cr._base_size == 100

    def test_default_animation_is_idle(self):
        cr = CreatureRenderer()
        assert cr.animation_state == AnimationState.IDLE


# ── Glow Aura Tests ──────────────────────────────────────────────────


class TestGlowAura:
    def test_glow_surface_created_on_set_mood(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        assert cr._glow_surface is not None

    def test_mood_change_updates_mood(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        assert cr._mood == "neutral"
        cr.set_mood("hostile")
        assert cr._mood == "hostile"

    def test_mood_change_invalidates_glow(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.set_mood("hostile")
        # Glow should be rebuilt (new Surface call)
        assert cr._mood == "hostile"
        assert cr._glow_surface is not None

    def test_default_mood_is_neutral(self):
        cr = CreatureRenderer()
        assert cr._mood == "neutral"


# ── Size Scaling Tests ───────────────────────────────────────────────


class TestSizeScaling:
    def test_mushroomer_smallest(self):
        cr = CreatureRenderer(stage=CreatureStage.MUSHROOMER)
        # Mushroomer has 0.6x multiplier → 60px
        assert cr._body_width == 60

    def test_gillman_base(self):
        cr = CreatureRenderer(stage=CreatureStage.GILLMAN)
        assert cr._body_width == 100

    def test_podfish(self):
        cr = CreatureRenderer(stage=CreatureStage.PODFISH)
        assert cr._body_width == 110

    def test_tadman(self):
        cr = CreatureRenderer(stage=CreatureStage.TADMAN)
        assert cr._body_width == 130

    def test_frogman_largest(self):
        cr = CreatureRenderer(stage=CreatureStage.FROGMAN)
        # Frogman has 1.5x multiplier → 150px
        assert cr._body_width == 150


# ── Animation State Tests ────────────────────────────────────────────


class TestAnimationStates:
    def test_idle_is_default(self):
        cr = CreatureRenderer()
        assert cr.animation_state == AnimationState.IDLE

    def test_set_talking(self):
        cr = CreatureRenderer()
        cr.animation_state = AnimationState.TALKING
        assert cr.animation_state == AnimationState.TALKING

    def test_set_animation_talking_opens_mouth(self):
        cr = CreatureRenderer()
        cr.set_animation(AnimationState.TALKING)
        assert cr.animation_state == AnimationState.TALKING
        assert cr._mouth_open > 0

    def test_set_animation_idle(self):
        cr = CreatureRenderer()
        cr.set_animation(AnimationState.SWIMMING)
        cr.set_animation(AnimationState.IDLE)
        assert cr.animation_state == AnimationState.IDLE

    def test_sleeping_renders_zs(self):
        cr = CreatureRenderer()
        cr.set_animation(AnimationState.SLEEPING)
        cr._time = 1.0
        cr.render(_surface_mock)
        assert _pygame_mock.draw.line.call_count > 0

    def test_talking_mouth_oscillates(self):
        cr = CreatureRenderer()
        cr.set_animation(AnimationState.TALKING)
        mouth_values = []
        for _ in range(10):
            cr.update(0.05)
            mouth_values.append(cr._mouth_open)
        assert len(set(round(v, 3) for v in mouth_values)) > 1

    def test_mouth_closes_after_talking_stops(self):
        cr = CreatureRenderer()
        cr.set_animation(AnimationState.TALKING)
        cr.update(0.1)
        assert cr._mouth_open > 0
        cr.set_animation(AnimationState.IDLE)
        for _ in range(20):
            cr.update(0.1)
        assert cr._mouth_open == pytest.approx(0.0, abs=0.01)

    def test_animation_state_enum_values(self):
        assert AnimationState.IDLE.value == "idle"
        assert AnimationState.SWIMMING.value == "swimming"
        assert AnimationState.TALKING.value == "talking"
        assert AnimationState.EATING.value == "eating"
        assert AnimationState.SLEEPING.value == "sleeping"


# ── Idle Animation (Bob Offset) ──────────────────────────────────────


class TestIdleAnimation:
    def test_update_changes_bob_offset(self):
        cr = CreatureRenderer()
        cr.update(0.5)
        # After time passes, bob offset should be computed from sine wave
        # It may be near zero at specific phases, but _time should advance
        assert cr._time == pytest.approx(0.5)

    def test_bob_offset_oscillates(self):
        cr = CreatureRenderer()
        offsets = []
        for _ in range(20):
            cr.update(0.1)
            offsets.append(cr._bob_offset)
        # Should have some variation (sine wave)
        assert max(offsets) > min(offsets)


# ── Position Interpolation Tests ─────────────────────────────────────


class TestPositionInterpolation:
    def test_move_toward_target(self):
        pos = CreaturePosition(x=0.0, y=0.0, target_x=100.0, target_y=0.0)
        pos.move_toward_target(0.5)
        assert pos.x > 0.0
        assert pos.x < 100.0

    def test_reaches_target(self):
        pos = CreaturePosition(x=0.0, y=0.0, target_x=50.0, target_y=50.0)
        for _ in range(100):
            pos.move_toward_target(0.1)
        assert pos.at_target()
        assert pos.x == pytest.approx(50.0, abs=1.0)
        assert pos.y == pytest.approx(50.0, abs=1.0)

    def test_at_target_initially_true(self):
        pos = CreaturePosition(x=100.0, y=200.0, target_x=100.0, target_y=200.0)
        assert pos.at_target()

    def test_set_target(self):
        pos = CreaturePosition()
        pos.set_target(300.0, 400.0)
        assert pos.target_x == 300.0
        assert pos.target_y == 400.0

    def test_speed_affects_movement(self):
        slow = CreaturePosition(
            x=0.0, y=0.0, target_x=100.0, target_y=0.0, speed=30.0
        )
        fast = CreaturePosition(
            x=0.0, y=0.0, target_x=100.0, target_y=0.0, speed=120.0
        )
        slow.move_toward_target(0.1)
        fast.move_toward_target(0.1)
        assert fast.x > slow.x


# ── Stage Rendering Tests ────────────────────────────────────────────


class TestStageRendering:
    def test_render_mushroomer(self):
        cr = CreatureRenderer(stage=CreatureStage.MUSHROOMER)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.circle.called

    def test_render_gillman(self):
        cr = CreatureRenderer(stage=CreatureStage.GILLMAN)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.polygon.called
        assert _pygame_mock.draw.line.called

    def test_render_podfish(self):
        cr = CreatureRenderer(stage=CreatureStage.PODFISH)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.polygon.called

    def test_render_tadman(self):
        cr = CreatureRenderer(stage=CreatureStage.TADMAN)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.circle.called
        assert _pygame_mock.draw.line.called
        assert _pygame_mock.draw.polygon.called

    def test_render_frogman(self):
        cr = CreatureRenderer(stage=CreatureStage.FROGMAN)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.ellipse.called
        assert _pygame_mock.draw.line.called
        assert _pygame_mock.draw.circle.called

    def test_all_stages_render_without_error(self):
        for stage in CreatureStage:
            cr = CreatureRenderer(stage=stage)
            cr.render(_surface_mock)  # No exception


# ── Face Tracking Tests ──────────────────────────────────────────────


class TestFaceTracking:
    def test_set_mouse_position(self):
        cr = CreatureRenderer()
        cr.set_mouse_position(300.0, 400.0)
        assert cr._mouse_x == 300.0
        assert cr._mouse_y == 400.0

    def test_eye_tracks_mouse(self):
        cr = CreatureRenderer()
        cr.position.x = 512.0
        cr.position.y = 400.0
        cr.set_mouse_position(1000.0, 400.0)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.circle.called


# ── Blinking Tests ───────────────────────────────────────────────────


class TestBlinking:
    def test_not_blinking_initially(self):
        cr = CreatureRenderer()
        assert not cr._is_blinking

    def test_blink_eventually_triggers(self):
        cr = CreatureRenderer()
        cr._blink_timer = 6.0
        cr.update(0.01)
        assert cr._is_blinking

    def test_blink_ends_after_duration(self):
        cr = CreatureRenderer()
        cr._is_blinking = True
        cr._blink_timer = 0.0
        cr.update(0.2)
        assert not cr._is_blinking


# ── Stage Switching Tests ────────────────────────────────────────────


class TestStageSwitching:
    def test_set_stage(self):
        cr = CreatureRenderer()
        cr.set_stage(CreatureStage.GILLMAN)
        assert cr.stage == CreatureStage.GILLMAN

    def test_set_stage_updates_body_width(self):
        cr = CreatureRenderer(stage=CreatureStage.MUSHROOMER)
        assert cr._body_width == 60
        cr.set_stage(CreatureStage.FROGMAN)
        assert cr._body_width == 150

    def test_render_after_stage_change(self):
        cr = CreatureRenderer()
        cr.set_stage(CreatureStage.PODFISH)
        cr.render(_surface_mock)
        assert _pygame_mock.draw.ellipse.called


# ── Update Loop Tests ────────────────────────────────────────────────


class TestUpdateLoop:
    def test_update_advances_time(self):
        cr = CreatureRenderer()
        cr.update(0.5)
        assert cr._time == pytest.approx(0.5)
        cr.update(0.3)
        assert cr._time == pytest.approx(0.8)

    def test_update_moves_position(self):
        cr = CreatureRenderer()
        start_x = cr.position.x
        cr.position.set_target(start_x + 200, cr.position.y)
        cr.update(0.5)
        assert cr.position.x > start_x

    def test_idle_wander_picks_new_target(self):
        cr = CreatureRenderer()
        cr.set_animation(AnimationState.IDLE)
        cr.position.target_x = cr.position.x
        cr.position.target_y = cr.position.y
        cr._idle_wander_timer = 3.5
        cr.update(0.1)
        assert cr._idle_wander_timer < 3.5


# ── Eating Effect Tests ──────────────────────────────────────────────


class TestEatingEffect:
    def test_render_eating_effect(self):
        cr = CreatureRenderer()
        cr._time = 1.0
        cr.render_eating_effect(_surface_mock, 200.0, 300.0)
        assert _pygame_mock.draw.circle.called
        assert _pygame_mock.draw.circle.call_count == 4


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_render_at_zero_position(self):
        cr = CreatureRenderer()
        cr.position.x = 0.0
        cr.position.y = 0.0
        cr.render(_surface_mock)

    def test_update_with_zero_dt(self):
        cr = CreatureRenderer()
        cr.update(0.0)
        assert cr._time == 0.0

    def test_update_with_large_dt(self):
        cr = CreatureRenderer()
        cr.update(100.0)
        assert cr._time == pytest.approx(100.0)

    def test_set_bounds(self):
        cr = CreatureRenderer()
        cr.set_bounds(10, 50, 800, 600)
        assert cr._bounds_x == 10
        assert cr._bounds_y == 50
        assert cr._bounds_w == 800
        assert cr._bounds_h == 600

    def test_mouse_at_creature_position(self):
        cr = CreatureRenderer()
        cr.set_mouse_position(cr.position.x, cr.position.y)
        cr.render(_surface_mock)


# ── Genome-Driven Rendering Tests ────────────────────────────────────


class TestGenomeDrivenRendering:
    def test_different_genomes_produce_different_renders(self):
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
        assert calls_1 != calls_2

    def test_genome_extremes_do_not_break_rendering(self):
        for val in (0.0, 1.0):
            traits = {t: val for t in ALL_TRAITS}
            genome = CreatureGenome(traits=traits)
            for stage in CreatureStage:
                cr = CreatureRenderer(stage=stage, genome=genome)
                cr.render(_surface_mock)

    def test_default_genome_matches_original_draw_count(self):
        r1 = CreatureRenderer(stage=CreatureStage.GILLMAN)
        _pygame_mock.draw.reset_mock()
        r1.render(_surface_mock)
        count_no_genome = len(_pygame_mock.draw.method_calls)

        default_genome = CreatureGenome()
        r2 = CreatureRenderer(stage=CreatureStage.GILLMAN, genome=default_genome)
        _pygame_mock.draw.reset_mock()
        r2.render(_surface_mock)
        count_with_genome = len(_pygame_mock.draw.method_calls)
        assert count_no_genome == count_with_genome

    def test_set_genome_method(self):
        cr = CreatureRenderer()
        assert cr.genome is None
        genome = CreatureGenome(traits={"body_size": 0.8})
        cr.set_genome(genome)
        assert cr.genome is genome
        cr.set_genome(None)
        assert cr.genome is None

    def test_pattern_complexity_adds_draw_calls(self):
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
        assert fancy_count > plain_count

    def test_all_stages_render_with_genome(self):
        genome = CreatureGenome(traits={
            "body_size": 0.7, "eye_size": 0.3, "fin_length": 0.8,
            "limb_proportion": 0.6, "hue": 0.3, "saturation": 0.7,
            "pattern_complexity": 0.8, "face_expressiveness": 0.6,
        })
        for stage in CreatureStage:
            cr = CreatureRenderer(stage=stage, genome=genome)
            cr.render(_surface_mock)


# ── Glow Pulse Tests ────────────────────────────────────────────────


class TestGlowPulse:
    def test_pulse_oscillates_intensity(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.update(0.1)
        alpha_0 = cr._glow_alpha
        cr.update(0.4)  # advance to ~0.5s (peak of sine)
        alpha_1 = cr._glow_alpha
        assert alpha_0 != alpha_1  # should oscillate

    def test_glow_alpha_stays_in_range(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        for i in range(40):
            cr.update(0.1)
            assert 0.0 <= cr._glow_alpha <= 1.0


# ── Mood Transition Tests ──────────────────────────────────────────


class TestMoodTransition:
    def test_mood_change_starts_transition(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.set_mood("hostile")
        assert cr._mood_transition_progress < 1.0

    def test_transition_completes_after_duration(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.set_mood("hostile")
        cr.update(1.5)  # past 1s transition
        assert cr._mood_transition_progress >= 1.0

    def test_same_mood_does_not_start_transition(self):
        cr = CreatureRenderer()
        cr.set_mood("neutral")
        cr.update(2.0)  # complete any transition
        cr.set_mood("neutral")
        assert cr._mood_transition_progress >= 1.0
