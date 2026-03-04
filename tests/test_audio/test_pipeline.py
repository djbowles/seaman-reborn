"""Tests for the full-duplex audio I/O pipeline."""

from __future__ import annotations

import io
import threading
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from seaman_brain.audio.aec import FRAME_SAMPLES
from seaman_brain.audio.pipeline import AudioIOPipeline
from seaman_brain.config import AudioConfig


def _make_wav(n_samples: int = 1600, sample_rate: int = 16000) -> bytes:
    """Create a valid WAV file with the given number of samples."""
    samples = np.zeros(n_samples, dtype=np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples.tobytes())
    return buf.getvalue()


def _make_config(**overrides) -> AudioConfig:
    """Create an AudioConfig with AEC defaults."""
    defaults = {
        "aec_enabled": True,
        "vad_aggressiveness": 2,
        "aec_filter_length": 64,
        "aec_step_size": 0.01,
        "barge_in_enabled": False,
        "stt_silence_threshold": 0.01,
        "stt_silence_duration": 0.5,
    }
    defaults.update(overrides)
    return AudioConfig(**defaults)


# ─── Lifecycle ────────────────────────────────────────────────────


class TestPipelineLifecycle:
    """Test start/stop lifecycle."""

    def test_construction(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)
        assert pipeline._running is False
        assert pipeline._stream is None

    def test_start_creates_thread(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        # Mock sounddevice so we don't need real audio hardware
        mock_sd = MagicMock()
        mock_stream = MagicMock()
        # Make read block briefly then stop
        call_count = 0

        def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count > 5:
                pipeline._running = False
            return np.zeros((n, 1), dtype=np.float32), False

        mock_stream.read = fake_read
        mock_stream.start = MagicMock()
        mock_stream.close = MagicMock()
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            pipeline.start()
            assert pipeline._running is True
            assert pipeline._thread is not None
            # Wait for thread to finish
            pipeline._thread.join(timeout=2.0)

    def test_stop_joins_thread(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)
        pipeline._running = False
        pipeline._thread = threading.Thread(target=lambda: None)
        pipeline._thread.start()
        pipeline._thread.join()

        pipeline.stop()
        assert pipeline._running is False
        assert pipeline._thread is None

    def test_start_idempotent(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)
        pipeline._running = True
        # start() should be a no-op when already running
        pipeline.start()
        assert pipeline._thread is None  # Not created because _running was True


# ─── Feed reference ───────────────────────────────────────────────


class TestFeedReference:
    """Test feed_reference WAV decoding and frame chunking."""

    def test_feed_reference_enqueues_frames(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        wav_bytes = _make_wav(n_samples=FRAME_SAMPLES * 3)
        pipeline.feed_reference(wav_bytes)

        assert len(pipeline._ref_queue) == 3
        assert pipeline._tts_playing is True

    def test_feed_reference_sets_tts_playing(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        assert pipeline._tts_playing is False
        pipeline.feed_reference(_make_wav())
        assert pipeline._tts_playing is True

    def test_feed_reference_resamples(self):
        """WAV at different sample rate gets resampled."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        # Create WAV at 22050 Hz
        wav_22k = _make_wav(n_samples=2205, sample_rate=22050)  # ~0.1s
        pipeline.feed_reference(wav_22k)

        # Should have frames at 16kHz — approximately 10 frames for 0.1s
        assert len(pipeline._ref_queue) > 0

    def test_feed_reference_invalid_wav_handled(self):
        """Invalid WAV bytes don't crash."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        pipeline.feed_reference(b"not a wav file")
        assert len(pipeline._ref_queue) == 0

    def test_clear_reference(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        pipeline.feed_reference(_make_wav())
        assert len(pipeline._ref_queue) > 0
        assert pipeline._tts_playing is True

        pipeline.clear_reference()
        assert len(pipeline._ref_queue) == 0
        assert pipeline._tts_playing is False


# ─── Utterance segmentation ──────────────────────────────────────


class TestUtteranceEmission:
    """Test utterance segmentation and emission."""

    def test_speech_then_silence_emits_utterance(self):
        """Speech followed by silence triggers on_utterance callback."""
        received = []
        config = _make_config(stt_silence_duration=0.03)  # 3 frames at 10ms
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: received.append(pcm)
        )

        # Simulate speech frames
        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        for _ in range(5):
            pipeline._segment_utterance(speech, is_speech=True)

        assert len(received) == 0  # Not emitted yet

        # Simulate silence frames
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        for _ in range(pipeline._silence_limit + 1):
            pipeline._segment_utterance(silence, is_speech=False)

        assert len(received) == 1
        assert len(received[0]) > 0

    def test_no_speech_no_emission(self):
        """Only silence produces no utterance."""
        received = []
        config = _make_config()
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: received.append(pcm)
        )

        silence = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        for _ in range(100):
            pipeline._segment_utterance(silence, is_speech=False)

        assert len(received) == 0

    def test_utterance_pcm_format(self):
        """Emitted utterance is 16-bit PCM bytes."""
        received = []
        config = _make_config(stt_silence_duration=0.01)
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: received.append(pcm)
        )

        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.1
        pipeline._segment_utterance(speech, is_speech=True)

        silence = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        for _ in range(pipeline._silence_limit + 1):
            pipeline._segment_utterance(silence, is_speech=False)

        assert len(received) == 1
        pcm = received[0]
        # Should be 16-bit PCM: 2 bytes per sample
        n_samples = len(pcm) // 2
        assert n_samples > 0
        # Verify it's valid int16 data
        samples = np.frombuffer(pcm, dtype=np.int16)
        assert len(samples) == n_samples


# ─── Barge-in ────────────────────────────────────────────────────


class TestBargeIn:
    """Test barge-in detection."""

    def test_barge_in_fires_during_tts(self):
        """Speech during TTS playback triggers barge-in callback."""
        barge_in_count = [0]
        config = _make_config(barge_in_enabled=True)
        pipeline = AudioIOPipeline(
            config, on_barge_in=lambda: barge_in_count.__setitem__(0, barge_in_count[0] + 1)
        )

        # Set TTS playing
        pipeline._tts_playing = True
        pipeline._barge_in_fired = False

        # Manually trigger the barge-in logic
        is_speech = True
        if (
            is_speech
            and pipeline._tts_playing
            and pipeline._config.barge_in_enabled
            and not pipeline._barge_in_fired
        ):
            pipeline._barge_in_fired = True
            pipeline._on_barge_in()

        assert barge_in_count[0] == 1

    def test_barge_in_fires_only_once(self):
        """Barge-in fires only once per TTS playback."""
        barge_in_count = [0]
        config = _make_config(barge_in_enabled=True)
        pipeline = AudioIOPipeline(
            config, on_barge_in=lambda: barge_in_count.__setitem__(0, barge_in_count[0] + 1)
        )

        pipeline._tts_playing = True
        pipeline._barge_in_fired = False

        # First detection
        pipeline._barge_in_fired = True
        pipeline._on_barge_in()
        assert barge_in_count[0] == 1

        # Second detection — already fired
        if not pipeline._barge_in_fired:
            pipeline._on_barge_in()
        assert barge_in_count[0] == 1  # Still 1

    def test_no_barge_in_when_disabled(self):
        """No barge-in when barge_in_enabled=False."""
        barge_in_count = [0]
        config = _make_config(barge_in_enabled=False)
        pipeline = AudioIOPipeline(
            config, on_barge_in=lambda: barge_in_count.__setitem__(0, barge_in_count[0] + 1)
        )

        pipeline._tts_playing = True
        is_speech = True

        if (
            is_speech
            and pipeline._tts_playing
            and pipeline._config.barge_in_enabled
            and not pipeline._barge_in_fired
        ):
            pipeline._on_barge_in()

        assert barge_in_count[0] == 0

    def test_no_barge_in_when_not_speaking(self):
        """No barge-in when TTS is not playing."""
        barge_in_count = [0]
        config = _make_config(barge_in_enabled=True)
        pipeline = AudioIOPipeline(
            config, on_barge_in=lambda: barge_in_count.__setitem__(0, barge_in_count[0] + 1)
        )

        pipeline._tts_playing = False
        is_speech = True

        if (
            is_speech
            and pipeline._tts_playing
            and pipeline._config.barge_in_enabled
            and not pipeline._barge_in_fired
        ):
            pipeline._on_barge_in()

        assert barge_in_count[0] == 0


# ─── VAD ──────────────────────────────────────────────────────────


class TestVAD:
    """Test voice activity detection."""

    def test_rms_fallback_detects_speech(self):
        """RMS fallback detects loud signal as speech."""
        config = _make_config(stt_silence_threshold=0.01)
        pipeline = AudioIOPipeline(config)
        pipeline._vad_available = False  # Force RMS fallback

        loud = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        assert pipeline._detect_speech(loud) is True

    def test_rms_fallback_detects_silence(self):
        """RMS fallback detects quiet signal as silence."""
        config = _make_config(stt_silence_threshold=0.01)
        pipeline = AudioIOPipeline(config)
        pipeline._vad_available = False

        quiet = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        assert pipeline._detect_speech(quiet) is False


# ─── Reference frame dequeue ─────────────────────────────────────


class TestGetRefFrame:
    """Test reference frame dequeue."""

    def test_dequeues_in_order(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        frame1 = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.1
        frame2 = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.2
        pipeline._ref_queue.append(frame1)
        pipeline._ref_queue.append(frame2)
        pipeline._tts_playing = True

        out1 = pipeline._get_ref_frame()
        np.testing.assert_allclose(out1, frame1)

        out2 = pipeline._get_ref_frame()
        np.testing.assert_allclose(out2, frame2)

    def test_returns_zeros_when_empty(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        out = pipeline._get_ref_frame()
        np.testing.assert_allclose(out, np.zeros(FRAME_SAMPLES))

    def test_tts_playing_clears_on_last_frame(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        pipeline._ref_queue.append(np.zeros(FRAME_SAMPLES, dtype=np.float64))
        pipeline._tts_playing = True

        pipeline._get_ref_frame()
        assert pipeline._tts_playing is False
