"""Tests for the NLMS echo canceller."""

from __future__ import annotations

import numpy as np

from seaman_brain.audio.aec import FRAME_SAMPLES, SAMPLE_RATE, NLMSEchoCanceller


class TestNLMSEchoCancellerInit:
    """Test initialization and properties."""

    def test_default_construction(self):
        aec = NLMSEchoCanceller()
        assert aec.filter_length == 2048
        assert aec.step_size == 0.01

    def test_custom_params(self):
        aec = NLMSEchoCanceller(filter_length=512, step_size=0.05)
        assert aec.filter_length == 512
        assert aec.step_size == 0.05

    def test_initial_weights_zero(self):
        aec = NLMSEchoCanceller(filter_length=64)
        assert np.all(aec._weights == 0.0)
        assert np.all(aec._ref_buffer == 0.0)


class TestNLMSEchoCancellerProcessing:
    """Test echo cancellation processing."""

    def test_no_reference_passthrough(self):
        """With zero reference, mic signal passes through unchanged."""
        aec = NLMSEchoCanceller(filter_length=32, step_size=0.01)
        mic = np.random.randn(FRAME_SAMPLES).astype(np.float64)
        ref = np.zeros(FRAME_SAMPLES, dtype=np.float64)

        output = aec.process_frame(mic, ref)

        # With zero reference, output should equal input
        np.testing.assert_allclose(output, mic, atol=1e-10)

    def test_echo_convergence(self):
        """NLMS filter converges to reduce echo over multiple frames."""
        aec = NLMSEchoCanceller(filter_length=64, step_size=0.1)

        # Simulate echo: mic = reference * 0.5 (simple delay-free echo)
        n_frames = 100
        initial_power = None
        final_power = None

        for i in range(n_frames):
            ref = np.random.randn(FRAME_SAMPLES).astype(np.float64) * 0.1
            echo = ref * 0.5  # Simple echo model
            mic = echo  # Mic captures only echo (no desired signal)

            output = aec.process_frame(mic, ref)

            power = float(np.mean(output ** 2))
            if i == 0:
                initial_power = power
            if i == n_frames - 1:
                final_power = power

        # After convergence, output power should be much less than initial
        assert final_power < initial_power * 0.1, (
            f"Echo not converging: initial={initial_power:.6f}, final={final_power:.6f}"
        )

    def test_output_shape_matches_input(self):
        """Output has same shape as input frame."""
        aec = NLMSEchoCanceller(filter_length=32)
        frame_size = 80
        mic = np.random.randn(frame_size).astype(np.float64)
        ref = np.random.randn(frame_size).astype(np.float64)

        output = aec.process_frame(mic, ref)
        assert output.shape == mic.shape

    def test_frame_samples_constant(self):
        """FRAME_SAMPLES matches 10ms at 16kHz."""
        assert FRAME_SAMPLES == 160
        assert SAMPLE_RATE == 16000

    def test_process_single_sample_frame(self):
        """Process a frame with a single sample."""
        aec = NLMSEchoCanceller(filter_length=8)
        mic = np.array([0.5], dtype=np.float64)
        ref = np.array([0.3], dtype=np.float64)

        output = aec.process_frame(mic, ref)
        assert output.shape == (1,)


class TestNLMSEchoCancellerReset:
    """Test reset functionality."""

    def test_reset_clears_state(self):
        """reset() zeros out weights and reference buffer."""
        aec = NLMSEchoCanceller(filter_length=32, step_size=0.1)

        # Process some frames to build up state
        for _ in range(10):
            mic = np.random.randn(FRAME_SAMPLES).astype(np.float64)
            ref = np.random.randn(FRAME_SAMPLES).astype(np.float64)
            aec.process_frame(mic, ref)

        # Verify state is non-zero
        assert not np.all(aec._weights == 0.0)

        # Reset
        aec.reset()
        assert np.all(aec._weights == 0.0)
        assert np.all(aec._ref_buffer == 0.0)

    def test_reset_allows_reuse(self):
        """After reset, AEC can process frames normally again."""
        aec = NLMSEchoCanceller(filter_length=32)

        mic = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.1
        ref = np.zeros(FRAME_SAMPLES, dtype=np.float64)

        aec.reset()
        output = aec.process_frame(mic, ref)
        np.testing.assert_allclose(output, mic, atol=1e-10)
