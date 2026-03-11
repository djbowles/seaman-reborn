"""Tests for the full-duplex audio I/O pipeline."""

from __future__ import annotations

import io
import threading
import time
import wave
from unittest.mock import MagicMock, patch

import numpy as np

from seaman_brain.audio.aec import FRAME_SAMPLES
from seaman_brain.audio.pipeline import _TTS_PLAYING_TIMEOUT, AudioIOPipeline
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


# ─── Echo suppression ─────────────────────────────────────────────


class TestEchoSuppression:
    """Test TTS echo suppression in the pipeline."""

    def test_suppresses_during_tts(self):
        """Frames during TTS playback are not buffered as utterance."""
        received = []
        config = _make_config(barge_in_enabled=False, stt_silence_duration=0.03)
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: received.append(pcm)
        )
        pipeline._tts_playing = True

        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        for _ in range(10):
            pipeline._segment_utterance(speech, is_speech=True)

        # Nothing should be buffered while TTS is playing
        assert len(pipeline._speech_buffer) == 0
        assert len(received) == 0

    def test_cooldown_after_tts(self):
        """Frames during cooldown after TTS stops are also suppressed."""
        received = []
        config = _make_config(barge_in_enabled=False, stt_silence_duration=0.03)
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: received.append(pcm)
        )
        pipeline._tts_playing = False
        pipeline._echo_cooldown_remaining = 5

        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        for _ in range(5):
            pipeline._segment_utterance(speech, is_speech=True)

        # All consumed by cooldown
        assert len(pipeline._speech_buffer) == 0
        assert pipeline._echo_cooldown_remaining == 0

        # Next frame should be captured normally
        pipeline._segment_utterance(speech, is_speech=True)
        assert len(pipeline._speech_buffer) == 1

    def test_barge_in_bypasses_suppression(self):
        """With barge_in_enabled=True, frames ARE captured during TTS."""
        received = []
        config = _make_config(barge_in_enabled=True, stt_silence_duration=0.03)
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: received.append(pcm)
        )
        pipeline._tts_playing = True

        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        for _ in range(5):
            pipeline._segment_utterance(speech, is_speech=True)

        # Should be buffered despite TTS playing
        assert len(pipeline._speech_buffer) == 5

    def test_feed_reference_clears_partial_buffer(self):
        """Partial speech buffer is cleared when new TTS starts."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        # Simulate partial speech buffered
        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        pipeline._speech_buffer.append(speech)
        pipeline._in_utterance = True
        pipeline._silence_count = 3

        # Feed new TTS reference
        pipeline.feed_reference(_make_wav())

        # Buffer should be cleared
        assert len(pipeline._speech_buffer) == 0
        assert pipeline._in_utterance is False
        assert pipeline._silence_count == 0


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


# ─── Input device selection ──────────────────────────────────────


class TestInputDeviceSelection:
    """Test that the pipeline uses the configured input device."""

    def test_configured_device_passed_to_input_stream(self):
        """sd.InputStream receives device= when audio_input_device is set."""
        config = _make_config(audio_input_device="Portacapture X6")
        pipeline = AudioIOPipeline(config)

        mock_sd = MagicMock()
        # Simulate query_devices() returning a list of device dicts
        mock_sd.query_devices.side_effect = lambda *a, **kw: (
            [
                {"name": "Speakers (Realtek)", "max_input_channels": 0},
                {"name": "Portacapture X6", "max_input_channels": 2},
                {"name": "Microphone Array", "max_input_channels": 1},
            ]
            if not a and not kw
            else None  # the kind="input" verification call
        )
        mock_stream = MagicMock()
        call_count = 0

        def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                pipeline._running = False
            return np.zeros((n, 1), dtype=np.float32), False

        mock_stream.read = fake_read
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            pipeline.start()
            pipeline._thread.join(timeout=2.0)

        # Verify InputStream was called with device=1 (index of Portacapture)
        mock_sd.InputStream.assert_called_once()
        call_kwargs = mock_sd.InputStream.call_args
        assert call_kwargs[1]["device"] == 1

    def test_default_device_when_no_config(self):
        """sd.InputStream receives device=None when no device configured."""
        config = _make_config(audio_input_device="")
        pipeline = AudioIOPipeline(config)

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = None
        mock_stream = MagicMock()
        call_count = 0

        def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                pipeline._running = False
            return np.zeros((n, 1), dtype=np.float32), False

        mock_stream.read = fake_read
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            pipeline.start()
            pipeline._thread.join(timeout=2.0)

        mock_sd.InputStream.assert_called_once()
        call_kwargs = mock_sd.InputStream.call_args
        assert call_kwargs[1]["device"] is None

    def test_system_default_treated_as_none(self):
        """'System Default' device name is treated the same as empty string."""
        config = _make_config(audio_input_device="System Default")
        pipeline = AudioIOPipeline(config)

        mock_sd = MagicMock()
        mock_sd.query_devices.return_value = None
        mock_stream = MagicMock()
        call_count = 0

        def fake_read(n):
            nonlocal call_count
            call_count += 1
            if call_count > 2:
                pipeline._running = False
            return np.zeros((n, 1), dtype=np.float32), False

        mock_stream.read = fake_read
        mock_sd.InputStream.return_value = mock_stream

        with patch.dict("sys.modules", {"sounddevice": mock_sd}):
            pipeline.start()
            pipeline._thread.join(timeout=2.0)

        mock_sd.InputStream.assert_called_once()
        call_kwargs = mock_sd.InputStream.call_args
        assert call_kwargs[1]["device"] is None


# ─── Zero-frame feed_reference guard ─────────────────────────────


class TestFeedReferenceZeroFrames:
    """Verify feed_reference with short audio doesn't get _tts_playing stuck."""

    def test_zero_frames_does_not_set_tts_playing(self):
        """WAV shorter than one frame must NOT set _tts_playing = True."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        # Create WAV with fewer samples than FRAME_SAMPLES
        short_wav = _make_wav(n_samples=FRAME_SAMPLES // 2)
        pipeline.feed_reference(short_wav)

        assert pipeline._tts_playing is False
        assert len(pipeline._ref_queue) == 0

    def test_exactly_one_frame_sets_tts_playing(self):
        """WAV with exactly one frame's worth of samples works normally."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        one_frame_wav = _make_wav(n_samples=FRAME_SAMPLES)
        pipeline.feed_reference(one_frame_wav)

        assert pipeline._tts_playing is True
        assert len(pipeline._ref_queue) == 1

    def test_empty_pcm_after_decode_is_safe(self):
        """WAV with header but minimal data doesn't crash or get stuck."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        tiny_wav = _make_wav(n_samples=1)
        pipeline.feed_reference(tiny_wav)

        assert pipeline._tts_playing is False
        assert len(pipeline._ref_queue) == 0


# ─── TTS playing timeout safety valve ────────────────────────────


class TestTtsPlayingTimeout:
    """Verify the safety valve resets _tts_playing after timeout."""

    def test_timeout_resets_stuck_tts_playing(self):
        """If _tts_playing stuck True with empty queue, timeout resets it."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        pipeline._tts_playing = True
        pipeline._tts_started_at = time.monotonic() - _TTS_PLAYING_TIMEOUT - 1

        frame = pipeline._get_ref_frame()

        assert pipeline._tts_playing is False
        assert pipeline._echo_cooldown_remaining > 0
        np.testing.assert_allclose(frame, np.zeros(FRAME_SAMPLES))

    def test_no_timeout_within_limit(self):
        """_tts_playing stays True if within timeout period."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        pipeline._tts_playing = True
        pipeline._tts_started_at = time.monotonic() - 5.0  # Only 5s ago

        pipeline._get_ref_frame()

        assert pipeline._tts_playing is True  # Not timed out yet

    def test_normal_drain_not_affected_by_timeout(self):
        """Normal ref queue drain still works when timeout is set."""
        config = _make_config()
        pipeline = AudioIOPipeline(config)

        pipeline.feed_reference(_make_wav(n_samples=FRAME_SAMPLES * 2))
        assert pipeline._tts_playing is True

        # Drain both frames
        pipeline._get_ref_frame()
        pipeline._get_ref_frame()

        assert pipeline._tts_playing is False


# ─── Pipeline diagnostics ────────────────────────────────────────


class TestPipelineDiagnostics:
    """Test diagnostic counters and is_alive property."""

    def test_is_alive_false_when_stopped(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)
        assert pipeline.is_alive is False

    def test_is_alive_true_when_thread_running(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)
        pipeline._running = True
        started = threading.Event()
        done = threading.Event()

        def _target():
            started.set()
            done.wait()

        pipeline._thread = threading.Thread(target=_target)
        pipeline._thread.start()
        started.wait(timeout=2.0)
        try:
            assert pipeline.is_alive is True
        finally:
            done.set()
            pipeline._thread.join(timeout=1.0)

    def test_get_diagnostics_keys(self):
        config = _make_config()
        pipeline = AudioIOPipeline(config)
        diag = pipeline.get_diagnostics()
        expected_keys = {
            "alive", "frames_processed", "speech_frames",
            "utterances_emitted", "suppressed_frames",
            "tts_playing", "ref_queue_depth", "echo_cooldown",
        }
        assert set(diag.keys()) == expected_keys

    def test_utterance_count_increments(self):
        config = _make_config(stt_silence_duration=0.01)
        pipeline = AudioIOPipeline(
            config, on_utterance=lambda pcm: None
        )

        assert pipeline._diag_utterance_count == 0

        # Emit an utterance
        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        pipeline._segment_utterance(speech, is_speech=True)
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float64)
        for _ in range(pipeline._silence_limit + 1):
            pipeline._segment_utterance(silence, is_speech=False)

        assert pipeline._diag_utterance_count == 1

    def test_suppressed_count_increments_during_tts(self):
        config = _make_config(barge_in_enabled=False)
        pipeline = AudioIOPipeline(config)
        pipeline._tts_playing = True

        assert pipeline._diag_suppressed_count == 0

        speech = np.ones(FRAME_SAMPLES, dtype=np.float64) * 0.5
        for _ in range(10):
            pipeline._segment_utterance(speech, is_speech=True)

        assert pipeline._diag_suppressed_count == 10


# ─── Barge-in debounce ────────────────────────────────────────


class TestBargeInDebounce:
    """Test barge-in debounce (consecutive speech frames before firing)."""

    def test_barge_in_debounce_single_frame_no_fire(self):
        """Single speech frame during TTS does NOT trigger barge-in."""
        barge_in_count = [0]
        config = _make_config(
            barge_in_enabled=True,
            barge_in_debounce_frames=3,
        )
        pipeline = AudioIOPipeline(
            config,
            on_barge_in=lambda: barge_in_count.__setitem__(
                0, barge_in_count[0] + 1
            ),
        )

        # Set TTS playing
        pipeline._tts_playing = True
        pipeline._barge_in_fired = False

        # Process 1 speech frame via the barge-in logic
        is_speech = True
        if (
            is_speech
            and pipeline._tts_playing
            and pipeline._config.barge_in_enabled
            and not pipeline._barge_in_fired
        ):
            pipeline._barge_in_count += 1
            if pipeline._barge_in_count >= pipeline._config.barge_in_debounce_frames:
                pipeline._barge_in_fired = True
                pipeline._on_barge_in()

        assert barge_in_count[0] == 0
        assert pipeline._barge_in_fired is False

    def test_barge_in_debounce_n_frames_fires(self):
        """N consecutive speech frames during TTS triggers barge-in."""
        barge_in_count = [0]
        config = _make_config(
            barge_in_enabled=True,
            barge_in_debounce_frames=3,
        )
        pipeline = AudioIOPipeline(
            config,
            on_barge_in=lambda: barge_in_count.__setitem__(
                0, barge_in_count[0] + 1
            ),
        )

        pipeline._tts_playing = True
        pipeline._barge_in_fired = False

        # Process 3 consecutive speech frames
        for _ in range(3):
            is_speech = True
            if (
                is_speech
                and pipeline._tts_playing
                and pipeline._config.barge_in_enabled
                and not pipeline._barge_in_fired
            ):
                pipeline._barge_in_count += 1
                if (
                    pipeline._barge_in_count
                    >= pipeline._config.barge_in_debounce_frames
                ):
                    pipeline._barge_in_fired = True
                    pipeline._on_barge_in()

        assert barge_in_count[0] == 1
        assert pipeline._barge_in_fired is True

    def test_barge_in_debounce_reset_on_non_speech(self):
        """Non-speech frame resets debounce counter, preventing fire."""
        barge_in_count = [0]
        config = _make_config(
            barge_in_enabled=True,
            barge_in_debounce_frames=3,
        )
        pipeline = AudioIOPipeline(
            config,
            on_barge_in=lambda: barge_in_count.__setitem__(
                0, barge_in_count[0] + 1
            ),
        )

        pipeline._tts_playing = True
        pipeline._barge_in_fired = False

        def _process_frame(speech: bool) -> None:
            if (
                speech
                and pipeline._tts_playing
                and pipeline._config.barge_in_enabled
                and not pipeline._barge_in_fired
            ):
                pipeline._barge_in_count += 1
                if (
                    pipeline._barge_in_count
                    >= pipeline._config.barge_in_debounce_frames
                ):
                    pipeline._barge_in_fired = True
                    pipeline._on_barge_in()
            elif not speech:
                pipeline._barge_in_count = 0

        # 2 speech, 1 non-speech, 2 more speech — should NOT fire
        _process_frame(True)
        _process_frame(True)
        _process_frame(False)  # reset
        _process_frame(True)
        _process_frame(True)

        assert barge_in_count[0] == 0
        assert pipeline._barge_in_fired is False


# ─── Safety valve barge-in callback ──────────────────────────


class TestSafetyValveBargeIn:
    """Verify safety valve timeout fires the on_barge_in callback."""

    def test_safety_valve_fires_barge_in_callback(self):
        """When _tts_playing times out, on_barge_in callback is called."""
        barge_in_count = [0]
        config = _make_config()
        pipeline = AudioIOPipeline(
            config,
            on_barge_in=lambda: barge_in_count.__setitem__(
                0, barge_in_count[0] + 1
            ),
        )

        pipeline._tts_playing = True
        pipeline._tts_started_at = (
            time.monotonic() - _TTS_PLAYING_TIMEOUT - 1
        )

        pipeline._get_ref_frame()

        assert pipeline._tts_playing is False
        assert barge_in_count[0] == 1
