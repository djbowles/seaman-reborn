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
        # Reference signal circular buffer (replaces O(N) np.roll per sample)
        self._ref_buffer = np.zeros(filter_length, dtype=np.float64)
        self._ref_idx = 0  # Write pointer into circular buffer
        # Running power estimate (avoids recomputing dot product each sample)
        self._ref_power = 0.0

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
        fl = self._filter_length
        buf = self._ref_buffer
        weights = self._weights
        idx = self._ref_idx
        power = self._ref_power
        mu = self._step_size
        eps = self._eps

        for i in range(n_samples):
            # Insert new sample into circular buffer (O(1), replaces np.roll)
            old_val = buf[idx]
            new_val = ref_frame[i]
            buf[idx] = new_val
            # Update running power incrementally
            power += new_val * new_val - old_val * old_val
            idx = (idx + 1) % fl

            # Compute echo estimate via dot product on circular buffer.
            # weights[0] aligns with newest sample (buf[idx-1]), weights[fl-1]
            # with oldest (buf[idx]). Split into two reversed slices.
            n_recent = idx  # buf[0..idx-1] (recent samples)
            n_old = fl - idx  # buf[idx..fl-1] (older samples)

            echo_estimate = 0.0
            if n_recent > 0:
                echo_estimate += np.dot(
                    weights[:n_recent], buf[idx - 1::-1]
                )
            if n_old > 0:
                echo_estimate += np.dot(
                    weights[n_recent:], buf[fl - 1:idx - 1:-1] if idx > 0 else buf[::-1]
                )

            error = mic_frame[i] - echo_estimate
            output[i] = error

            # Update weights: w += mu * error * x / (x^T*x + eps)
            norm_power = max(power, 0.0) + eps
            step = mu * error / norm_power
            if n_recent > 0:
                weights[:n_recent] += step * buf[idx - 1::-1]
            if n_old > 0:
                if idx > 0:
                    weights[n_recent:] += step * buf[fl - 1:idx - 1:-1]
                else:
                    weights[n_recent:] += step * buf[::-1]

        self._ref_idx = idx
        self._ref_power = power
        return output

    def reset(self) -> None:
        """Clear filter state (call when audio config changes)."""
        self._weights[:] = 0.0
        self._ref_buffer[:] = 0.0
        self._ref_idx = 0
        self._ref_power = 0.0
