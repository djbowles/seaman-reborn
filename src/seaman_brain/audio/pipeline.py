"""Full-duplex audio I/O pipeline with AEC, VAD, and barge-in.

Opens a continuous microphone input stream and processes each frame
through the echo canceller and voice activity detector. Complete
utterances are emitted via callback for STT transcription.

When ``aec_enabled=True``, the pipeline runs alongside TTS playback,
subtracting the TTS reference signal from the mic input so the user's
voice is cleanly captured even while the creature is speaking.
"""

from __future__ import annotations

import io
import logging
import threading
import wave
from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from seaman_brain.audio.aec import FRAME_SAMPLES, SAMPLE_RATE, NLMSEchoCanceller
from seaman_brain.config import AudioConfig

logger = logging.getLogger(__name__)

_SILENCE_FRAMES_DEFAULT = 150  # 1.5s at 10ms/frame


class AudioIOPipeline:
    """Full-duplex audio pipeline with AEC and VAD.

    Args:
        config: Audio configuration.
        on_utterance: Callback fired with complete utterance PCM bytes
            (16kHz, 16-bit, mono) when end-of-speech is detected.
        on_barge_in: Callback fired (once) when speech is detected while
            TTS is playing and ``barge_in_enabled`` is True.
    """

    def __init__(
        self,
        config: AudioConfig,
        on_utterance: Callable[[bytes], None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ) -> None:
        self._config = config
        self._on_utterance = on_utterance
        self._on_barge_in = on_barge_in

        # AEC
        self._aec = NLMSEchoCanceller(
            filter_length=config.aec_filter_length,
            step_size=config.aec_step_size,
        )

        # VAD
        self._vad: Any = None
        self._vad_available = False
        self._init_vad()

        # Reference signal queue (TTS playback frames for AEC alignment)
        self._ref_queue: deque[np.ndarray] = deque()
        self._ref_lock = threading.Lock()

        # Stream state
        self._stream: Any = None
        self._thread: threading.Thread | None = None
        self._running = False

        # TTS playback state (for barge-in detection)
        self._tts_playing = False
        self._barge_in_fired = False

        # Utterance segmentation
        silence_frames = int(config.stt_silence_duration / 0.01)  # 10ms per frame
        self._silence_limit = max(silence_frames, 10)
        self._speech_buffer: list[np.ndarray] = []
        self._silence_count = 0
        self._in_utterance = False

        # RMS fallback threshold for VAD
        self._rms_threshold = config.stt_silence_threshold

    def _init_vad(self) -> None:
        """Try to initialize webrtcvad."""
        try:
            import webrtcvad

            self._vad = webrtcvad.Vad(self._config.vad_aggressiveness)
            self._vad_available = True
        except ImportError:
            self._vad_available = False
            logger.debug("webrtcvad not available, using RMS fallback for VAD")
        except Exception as exc:
            self._vad_available = False
            logger.warning("webrtcvad init failed: %s", exc)

    def start(self) -> None:
        """Open continuous microphone input stream in a daemon thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._processing_loop,
            name="audio-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("Audio pipeline started")

    def stop(self) -> None:
        """Close stream and join processing thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        logger.info("Audio pipeline stopped")

    def feed_reference(self, wav_bytes: bytes) -> None:
        """Decode TTS WAV and enqueue reference frames for AEC.

        Resamples to 16kHz if needed, then chops into FRAME_SAMPLES-sized
        chunks and enqueues them for the processing loop.

        Args:
            wav_bytes: Raw WAV audio bytes from TTS synthesis.
        """
        try:
            pcm, src_rate = self._decode_wav(wav_bytes)
            if src_rate != SAMPLE_RATE:
                pcm = self._resample(pcm, src_rate, SAMPLE_RATE)

            # Chop into frames
            frames = []
            for i in range(0, len(pcm) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
                frames.append(pcm[i:i + FRAME_SAMPLES].astype(np.float64))

            with self._ref_lock:
                self._ref_queue.extend(frames)
                self._tts_playing = True
                self._barge_in_fired = False

        except Exception as exc:
            logger.warning("Failed to feed reference audio: %s", exc)

    def clear_reference(self) -> None:
        """Flush reference queue (called on barge-in/TTS cancel)."""
        with self._ref_lock:
            self._ref_queue.clear()
            self._tts_playing = False
            self._barge_in_fired = False

    @staticmethod
    def _decode_wav(wav_bytes: bytes) -> tuple[np.ndarray, int]:
        """Decode WAV bytes to float64 numpy array.

        Returns:
            Tuple of (samples as float64 array, sample rate).
        """
        buf = io.BytesIO(wav_bytes)
        with wave.open(buf, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())

        if sampwidth == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
        elif sampwidth == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
        else:
            # Fallback: assume 8-bit unsigned
            samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0

        # Downmix to mono if stereo
        if n_channels > 1:
            samples = samples.reshape(-1, n_channels).mean(axis=1)

        return samples, framerate

    @staticmethod
    def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
        """Resample audio to target sample rate.

        Args:
            pcm: Input samples.
            src_rate: Source sample rate.
            dst_rate: Target sample rate.

        Returns:
            Resampled audio array.
        """
        try:
            from scipy.signal import resample

            n_samples = int(len(pcm) * dst_rate / src_rate)
            return resample(pcm, n_samples)
        except ImportError:
            # Linear interpolation fallback
            ratio = dst_rate / src_rate
            n_out = int(len(pcm) * ratio)
            indices = np.linspace(0, len(pcm) - 1, n_out)
            return np.interp(indices, np.arange(len(pcm)), pcm)

    def _processing_loop(self) -> None:
        """Main processing loop running in background thread."""
        try:
            import sounddevice as sd

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                blocksize=FRAME_SAMPLES,
            )
            self._stream.start()

            while self._running:
                try:
                    data, overflowed = self._stream.read(FRAME_SAMPLES)
                    if overflowed:
                        logger.debug("Audio pipeline: input overflow")

                    mic_frame = data.flatten().astype(np.float64)

                    # Get reference frame for AEC
                    ref_frame = self._get_ref_frame()

                    # AEC processing
                    cleaned = self._aec.process_frame(mic_frame, ref_frame)

                    # VAD
                    is_speech = self._detect_speech(cleaned)

                    # Barge-in detection
                    if (
                        is_speech
                        and self._tts_playing
                        and self._config.barge_in_enabled
                        and not self._barge_in_fired
                    ):
                        self._barge_in_fired = True
                        if self._on_barge_in is not None:
                            self._on_barge_in()

                    # Utterance segmentation
                    self._segment_utterance(cleaned, is_speech)

                except Exception as exc:
                    if self._running:
                        logger.debug("Pipeline frame error: %s", exc)

        except Exception as exc:
            logger.warning("Audio pipeline loop failed: %s", exc)
        finally:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

    def _get_ref_frame(self) -> np.ndarray:
        """Dequeue a reference frame, or return zeros if none available."""
        with self._ref_lock:
            if self._ref_queue:
                frame = self._ref_queue.popleft()
                if not self._ref_queue:
                    self._tts_playing = False
                return frame
        return np.zeros(FRAME_SAMPLES, dtype=np.float64)

    def _detect_speech(self, frame: np.ndarray) -> bool:
        """Run VAD on cleaned frame. Falls back to RMS threshold."""
        if self._vad_available and self._vad is not None:
            try:
                pcm_bytes = (frame * 32768.0).clip(-32768, 32767).astype(np.int16).tobytes()
                return self._vad.is_speech(pcm_bytes, SAMPLE_RATE)
            except Exception:
                pass

        # RMS fallback
        rms = float(np.sqrt(np.mean(frame ** 2)))
        return rms > self._rms_threshold

    def _segment_utterance(self, frame: np.ndarray, is_speech: bool) -> None:
        """Buffer speech frames and detect end-of-utterance via silence."""
        if is_speech:
            self._in_utterance = True
            self._silence_count = 0
            self._speech_buffer.append(frame.copy())
        elif self._in_utterance:
            self._silence_count += 1
            self._speech_buffer.append(frame.copy())
            if self._silence_count >= self._silence_limit:
                self._emit_utterance()
        # else: pre-speech silence, skip

    def _emit_utterance(self) -> None:
        """Concatenate buffered frames and fire on_utterance callback."""
        if not self._speech_buffer:
            self._reset_segmentation()
            return

        audio = np.concatenate(self._speech_buffer)
        # Convert to 16-bit PCM bytes
        pcm_int16 = (audio * 32768.0).clip(-32768, 32767).astype(np.int16)
        pcm_bytes = pcm_int16.tobytes()

        self._reset_segmentation()

        if self._on_utterance is not None:
            try:
                self._on_utterance(pcm_bytes)
            except Exception as exc:
                logger.warning("on_utterance callback error: %s", exc)

    def _reset_segmentation(self) -> None:
        """Reset utterance segmentation state."""
        self._speech_buffer.clear()
        self._silence_count = 0
        self._in_utterance = False
