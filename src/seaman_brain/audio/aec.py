"""NLMS adaptive echo canceller for full-duplex audio.

Implements a Normalized Least Mean Squares (NLMS) filter to remove
acoustic echo from the microphone signal. The reference signal (TTS
playback) is subtracted from the mic input, allowing simultaneous
speak and listen without feedback loops.

Requires only numpy — no external DSP libraries needed.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_MS = 10
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 160 samples per frame


class NLMSEchoCanceller:
    """Normalized Least Mean Squares adaptive echo canceller.

    For each sample, estimates echo as ``w^T * x_ref``, subtracts from
    mic, then updates weights via ``w += mu * error * x / (x^T*x + eps)``.

    Args:
        filter_length: Number of NLMS filter taps.
        step_size: NLMS learning rate (mu). Smaller = more stable, slower.
    """

    def __init__(
        self,
        filter_length: int = 2048,
        step_size: float = 0.01,
    ) -> None:
        self._filter_length = filter_length
        self._step_size = step_size
        self._eps = 1e-10  # Regularization to avoid division by zero

        # Adaptive filter weights (float64 for numerical stability)
        self._weights = np.zeros(filter_length, dtype=np.float64)
        # Reference signal buffer (sliding window of past reference samples)
        self._ref_buffer = np.zeros(filter_length, dtype=np.float64)

    @property
    def filter_length(self) -> int:
        """Number of NLMS filter taps."""
        return self._filter_length

    @property
    def step_size(self) -> float:
        """NLMS learning rate."""
        return self._step_size

    def process_frame(
        self,
        mic_frame: np.ndarray,
        ref_frame: np.ndarray,
    ) -> np.ndarray:
        """Process one frame of audio through the echo canceller.

        Args:
            mic_frame: Microphone input samples (float64, shape ``(N,)``).
            ref_frame: Reference (TTS playback) samples (float64, shape ``(N,)``).

        Returns:
            Cleaned microphone signal with echo subtracted.
        """
        n_samples = len(mic_frame)
        output = np.empty(n_samples, dtype=np.float64)

        for i in range(n_samples):
            # Shift reference buffer and insert new sample
            self._ref_buffer = np.roll(self._ref_buffer, 1)
            self._ref_buffer[0] = ref_frame[i]

            # Estimate echo: w^T * x
            echo_estimate = np.dot(self._weights, self._ref_buffer)

            # Error = mic - estimated echo
            error = mic_frame[i] - echo_estimate
            output[i] = error

            # Update weights: w += mu * error * x / (x^T*x + eps)
            power = np.dot(self._ref_buffer, self._ref_buffer) + self._eps
            self._weights += (self._step_size * error / power) * self._ref_buffer

        return output

    def reset(self) -> None:
        """Clear filter state (call when audio config changes)."""
        self._weights[:] = 0.0
        self._ref_buffer[:] = 0.0
