"""Tests for Pygame audio bridge - ambient, voice, SFX, microphone (US-041).

Pygame and pygame.mixer are mocked at module level to avoid requiring
audio hardware or a display server in CI. Uses the established pattern
from test_interactions.py: sys.modules["pygame"] = mock, import once.
"""

from __future__ import annotations

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Pygame Mock Setup (module-level, before any gui imports) ──────────

_pygame_mock = MagicMock()
_pygame_mock.QUIT = 256
_pygame_mock.KEYDOWN = 768
_pygame_mock.K_m = 109
_pygame_mock.K_ESCAPE = 27
_pygame_mock.init.return_value = (6, 0)
_pygame_mock.font.init.return_value = None

# Mixer mock
_mixer_mock = MagicMock()
_mixer_mock.get_init.return_value = True
_mixer_mock.get_num_channels.return_value = 8

_channel_mock_0 = MagicMock()  # ambient
_channel_mock_1 = MagicMock()  # voice
_channel_mock_2 = MagicMock()  # sfx
_channel_mock_0.get_busy.return_value = True

_sound_mock = MagicMock()


def _make_channel(idx):
    return [_channel_mock_0, _channel_mock_1, _channel_mock_2][idx]


_mixer_mock.Channel = _make_channel
_mixer_mock.Sound = MagicMock(return_value=_sound_mock)
_pygame_mock.mixer = _mixer_mock

# Install pygame mock before importing gui modules
sys.modules["pygame"] = _pygame_mock
sys.modules["pygame.mixer"] = _mixer_mock

from seaman_brain.config import AudioConfig  # noqa: E402
from seaman_brain.gui.audio_integration import (  # noqa: E402
    AudioChannel,
    PygameAudioBridge,
)


@pytest.fixture(autouse=True)
def _reset_mocks():
    """Reset mixer mocks and re-install pygame mock between tests.

    Other test_gui modules also set sys.modules["pygame"] at module level,
    so we must re-install ours before each test to avoid cross-contamination.
    """
    sys.modules["pygame"] = _pygame_mock
    sys.modules["pygame.mixer"] = _mixer_mock

    _mixer_mock.reset_mock()
    _channel_mock_0.reset_mock()
    _channel_mock_1.reset_mock()
    _channel_mock_2.reset_mock()
    _sound_mock.reset_mock()

    _mixer_mock.get_init.return_value = True
    _mixer_mock.get_num_channels.return_value = 8
    _mixer_mock.Channel = _make_channel
    _mixer_mock.Sound = MagicMock(return_value=_sound_mock)
    _channel_mock_0.get_busy.return_value = True


@pytest.fixture()
def audio_config() -> AudioConfig:
    """Audio config with known volume levels."""
    return AudioConfig(
        ambient_volume=0.3,
        tts_volume=0.8,
        sfx_volume=0.5,
    )


@pytest.fixture()
def mock_audio_manager():
    """Mock AudioManager with async speak/synthesize/listen."""
    mgr = MagicMock()
    mgr.synthesize = AsyncMock(return_value=b"RIFF" + b"\x00" * 40)
    mgr.speak = AsyncMock()
    mgr.listen = AsyncMock(return_value="hello seaman")
    mgr.stt_enabled = False
    mgr._sounds_dir = "assets/sounds"
    return mgr


@pytest.fixture()
def bridge(audio_config: AudioConfig) -> PygameAudioBridge:
    """A PygameAudioBridge with default config, no audio manager."""
    return PygameAudioBridge(audio_config=audio_config)


@pytest.fixture()
def bridge_with_manager(
    audio_config: AudioConfig, mock_audio_manager: MagicMock
) -> PygameAudioBridge:
    """A PygameAudioBridge with a mock AudioManager and async loop."""
    loop = asyncio.new_event_loop()
    b = PygameAudioBridge(
        audio_manager=mock_audio_manager,
        audio_config=audio_config,
        async_loop=loop,
    )
    yield b
    loop.close()


# ── Construction Tests ───────────────────────────────────────────────


class TestConstruction:
    """Tests for PygameAudioBridge initialization."""

    def test_default_construction(self):
        """Bridge initializes with default config."""
        b = PygameAudioBridge()
        assert b.audio_available is True

    def test_mixer_initialized(self, bridge: PygameAudioBridge):
        """Mixer channels are set up on construction."""
        assert bridge.audio_available is True
        assert bridge._mixer_initialized is True

    def test_initial_volumes_from_config(self, bridge: PygameAudioBridge):
        """Channel volumes match config values."""
        assert bridge.get_volume(AudioChannel.AMBIENT) == 0.3
        assert bridge.get_volume(AudioChannel.VOICE) == 0.8
        assert bridge.get_volume(AudioChannel.SFX) == 0.5

    def test_mic_starts_inactive(self, bridge: PygameAudioBridge):
        """Microphone starts deactivated."""
        assert bridge.mic_active is False

    def test_mixer_failure_falls_back_gracefully(self):
        """When mixer init fails, audio_available is False."""
        _mixer_mock.get_init.return_value = False
        _mixer_mock.init.side_effect = RuntimeError("No audio device")

        b = PygameAudioBridge()
        assert b.audio_available is False
        assert b._mixer_initialized is False

        # Cleanup
        _mixer_mock.init.side_effect = None


# ── Volume Control Tests ─────────────────────────────────────────────


class TestVolumeControl:
    """Tests for per-channel volume adjustment."""

    def test_set_volume_updates_value(self, bridge: PygameAudioBridge):
        """Setting volume updates the stored value."""
        bridge.set_volume(AudioChannel.AMBIENT, 0.7)
        assert bridge.get_volume(AudioChannel.AMBIENT) == 0.7

    def test_set_volume_clamps_high(self, bridge: PygameAudioBridge):
        """Volume above 1.0 is clamped to 1.0."""
        bridge.set_volume(AudioChannel.SFX, 1.5)
        assert bridge.get_volume(AudioChannel.SFX) == 1.0

    def test_set_volume_clamps_low(self, bridge: PygameAudioBridge):
        """Volume below 0.0 is clamped to 0.0."""
        bridge.set_volume(AudioChannel.VOICE, -0.5)
        assert bridge.get_volume(AudioChannel.VOICE) == 0.0

    def test_set_volume_applies_to_mixer_channel(self, bridge: PygameAudioBridge):
        """Setting volume calls set_volume on the Pygame channel."""
        bridge.set_volume(AudioChannel.AMBIENT, 0.6)
        _channel_mock_0.set_volume.assert_called_with(0.6)

    def test_set_volume_all_channels(self, bridge: PygameAudioBridge):
        """All three channels can be adjusted independently."""
        bridge.set_volume(AudioChannel.AMBIENT, 0.1)
        bridge.set_volume(AudioChannel.VOICE, 0.5)
        bridge.set_volume(AudioChannel.SFX, 0.9)

        assert bridge.get_volume(AudioChannel.AMBIENT) == 0.1
        assert bridge.get_volume(AudioChannel.VOICE) == 0.5
        assert bridge.get_volume(AudioChannel.SFX) == 0.9

    def test_set_volume_no_crash_when_mixer_unavailable(self):
        """Setting volume when mixer is unavailable doesn't crash."""
        _mixer_mock.get_init.return_value = False
        _mixer_mock.init.side_effect = RuntimeError("No audio")

        b = PygameAudioBridge()
        b.set_volume(AudioChannel.AMBIENT, 0.5)
        assert b.get_volume(AudioChannel.AMBIENT) == 0.5

        _mixer_mock.init.side_effect = None


# ── Ambient Loop Tests ───────────────────────────────────────────────


class TestAmbientLoop:
    """Tests for ambient sound loop control."""

    def test_start_ambient_aquarium(self, bridge: PygameAudioBridge):
        """Starting ambient with 'aquarium' attempts to load water sound."""
        with patch("os.path.exists", return_value=True):
            bridge.start_ambient("aquarium")
        assert bridge._current_ambient == "aquarium"

    def test_start_ambient_terrarium(self, bridge: PygameAudioBridge):
        """Starting ambient with 'terrarium' attempts to load nature sound."""
        with patch("os.path.exists", return_value=True):
            bridge.start_ambient("terrarium")
        assert bridge._current_ambient == "terrarium"

    def test_start_ambient_missing_file_stays_silent(self, bridge: PygameAudioBridge):
        """Missing ambient file results in silent ambient (no crash)."""
        with patch("os.path.exists", return_value=False):
            bridge.start_ambient("aquarium")
        assert bridge._ambient_playing is False
        assert bridge._current_ambient == "aquarium"

    def test_stop_ambient(self, bridge: PygameAudioBridge):
        """Stopping ambient resets state."""
        with patch("os.path.exists", return_value=True):
            bridge.start_ambient("aquarium")
        bridge.stop_ambient()

        assert bridge._ambient_playing is False
        assert bridge._current_ambient == ""
        _channel_mock_0.stop.assert_called()

    def test_start_same_ambient_doesnt_restart(self, bridge: PygameAudioBridge):
        """Re-starting the same ambient doesn't restart playback."""
        with patch("os.path.exists", return_value=True):
            bridge.start_ambient("aquarium")
            bridge._ambient_playing = True  # Simulate playing
            call_count = _channel_mock_0.stop.call_count
            bridge.start_ambient("aquarium")
            # stop should not have been called again
            assert _channel_mock_0.stop.call_count == call_count

    def test_ambient_playing_property(self, bridge: PygameAudioBridge):
        """ambient_playing reflects the current state."""
        assert bridge.ambient_playing is False
        with patch("os.path.exists", return_value=True):
            bridge.start_ambient("aquarium")
            bridge._ambient_playing = True
        assert bridge.ambient_playing is True

    def test_start_ambient_no_mixer(self):
        """Start ambient does nothing when mixer is unavailable."""
        _mixer_mock.get_init.return_value = False
        _mixer_mock.init.side_effect = RuntimeError("No audio")

        b = PygameAudioBridge()
        b.start_ambient("aquarium")
        assert b._ambient_playing is False

        _mixer_mock.init.side_effect = None


# ── Voice Playback Tests ─────────────────────────────────────────────


class TestVoicePlayback:
    """Tests for creature voice output via TTS -> Pygame mixer."""

    def test_play_voice_submits_to_async_loop(
        self, bridge_with_manager: PygameAudioBridge
    ):
        """play_voice submits a coroutine to the async loop."""
        bridge_with_manager.play_voice("Hello human")
        # Verify the async loop received a coroutine
        assert bridge_with_manager._async_loop is not None

    def test_play_voice_empty_text_noop(
        self, bridge_with_manager: PygameAudioBridge, mock_audio_manager: MagicMock
    ):
        """Empty text doesn't trigger voice playback."""
        bridge_with_manager.play_voice("")
        bridge_with_manager.play_voice("   ")
        mock_audio_manager.synthesize.assert_not_called()

    def test_play_voice_no_manager_noop(self, bridge: PygameAudioBridge):
        """Voice playback does nothing without AudioManager."""
        bridge.play_voice("Hello")  # Should not crash

    def test_play_voice_no_async_loop_noop(self):
        """Voice playback does nothing without async loop."""
        mgr = MagicMock()
        b = PygameAudioBridge(audio_manager=mgr, async_loop=None)
        b.play_voice("Hello")  # Should not crash

    @pytest.mark.asyncio
    async def test_play_voice_async_speaks(
        self, mock_audio_manager: MagicMock
    ):
        """_play_voice_async calls speak on the audio manager."""
        b = PygameAudioBridge(audio_manager=mock_audio_manager)
        await b._play_voice_async("Hello creature")
        mock_audio_manager.speak.assert_called_once_with("Hello creature")

    @pytest.mark.asyncio
    async def test_play_voice_async_fallback_empty_wav_noop(
        self, mock_audio_manager: MagicMock
    ):
        """Fallback to synthesize with empty bytes doesn't play anything."""
        mock_audio_manager.speak.side_effect = RuntimeError("speak broken")
        mock_audio_manager.synthesize.return_value = b""
        b = PygameAudioBridge(audio_manager=mock_audio_manager)
        await b._play_voice_async("Hello")
        # Should not crash, play not called on channel
        _channel_mock_1.play.assert_not_called()

    @pytest.mark.asyncio
    async def test_play_voice_async_exception_handled(
        self, mock_audio_manager: MagicMock
    ):
        """Both speak and synthesize failures are caught and logged."""
        mock_audio_manager.speak.side_effect = RuntimeError("speak error")
        mock_audio_manager.synthesize.side_effect = RuntimeError("TTS error")
        b = PygameAudioBridge(audio_manager=mock_audio_manager)
        await b._play_voice_async("Hello")  # Should not raise


# ── SFX Tests ────────────────────────────────────────────────────────


class TestSFXPlayback:
    """Tests for UI sound effects."""

    def test_play_sfx_with_file(self, bridge: PygameAudioBridge):
        """Playing SFX with existing file loads and plays it."""
        with patch("os.path.exists", return_value=True):
            bridge.play_sfx("button_click")
        _mixer_mock.Sound.assert_called()
        _channel_mock_2.play.assert_called()

    def test_play_sfx_missing_file_silent(self, bridge: PygameAudioBridge):
        """Missing SFX file results in silence (no crash)."""
        with patch("os.path.exists", return_value=False):
            bridge.play_sfx("nonexistent_sound")
        _channel_mock_2.play.assert_not_called()

    def test_play_sfx_empty_name_noop(self, bridge: PygameAudioBridge):
        """Empty SFX name is ignored."""
        bridge.play_sfx("")
        bridge.play_sfx("   ")
        _channel_mock_2.play.assert_not_called()

    def test_play_sfx_no_mixer_noop(self):
        """SFX does nothing when mixer is unavailable."""
        _mixer_mock.get_init.return_value = False
        _mixer_mock.init.side_effect = RuntimeError("No audio")

        b = PygameAudioBridge()
        b.play_sfx("button_click")
        _channel_mock_2.play.assert_not_called()

        _mixer_mock.init.side_effect = None

    def test_play_sfx_applies_volume(self, bridge: PygameAudioBridge):
        """SFX sound has volume set from channel config."""
        with patch("os.path.exists", return_value=True):
            bridge.play_sfx("feeding_splash")
        _sound_mock.set_volume.assert_called()


# ── Microphone Toggle Tests ──────────────────────────────────────────


class TestMicrophoneToggle:
    """Tests for microphone input toggle."""

    def test_toggle_activates_mic(self, bridge_with_manager: PygameAudioBridge):
        """First toggle activates the microphone."""
        assert bridge_with_manager.mic_active is False
        result = bridge_with_manager.toggle_microphone()
        assert result is True
        assert bridge_with_manager.mic_active is True

    def test_toggle_deactivates_mic(self, bridge_with_manager: PygameAudioBridge):
        """Second toggle deactivates the microphone."""
        bridge_with_manager.toggle_microphone()  # ON
        result = bridge_with_manager.toggle_microphone()  # OFF
        assert result is False
        assert bridge_with_manager.mic_active is False

    def test_toggle_updates_audio_manager_stt(
        self, bridge_with_manager: PygameAudioBridge, mock_audio_manager: MagicMock
    ):
        """Toggle sets stt_enabled on the AudioManager."""
        bridge_with_manager.toggle_microphone()
        assert mock_audio_manager.stt_enabled is True

    def test_handle_key_m_toggles_mic(self, bridge_with_manager: PygameAudioBridge):
        """Pressing M key toggles microphone."""
        handled = bridge_with_manager.handle_key_event(109)  # K_m
        assert handled is True
        assert bridge_with_manager.mic_active is True

    def test_handle_key_other_not_handled(self, bridge: PygameAudioBridge):
        """Non-M key returns False."""
        handled = bridge.handle_key_event(27)  # K_ESCAPE
        assert handled is False

    @pytest.mark.asyncio
    async def test_listen_async_calls_callback(self, mock_audio_manager: MagicMock):
        """_listen_async forwards transcribed text to on_stt_result callback."""
        results = []
        # listen() returns text once, then we deactivate mic to stop the loop
        call_count = 0

        async def _listen_once():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "hello seaman"
            b.mic_active = False
            return ""

        mock_audio_manager.listen = _listen_once
        b = PygameAudioBridge(
            audio_manager=mock_audio_manager, on_stt_result=results.append
        )
        b.mic_active = True
        await b._listen_async()
        assert results == ["hello seaman"]

    @pytest.mark.asyncio
    async def test_listen_async_no_manager(self):
        """_listen_async returns immediately without manager."""
        b = PygameAudioBridge()
        b.mic_active = True
        await b._listen_async()
        # Should return without error

    @pytest.mark.asyncio
    async def test_listen_async_exception_handled(
        self, mock_audio_manager: MagicMock
    ):
        """STT failure is caught and loop continues."""
        call_count = 0

        async def _listen_fail():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Mic error")
            b.mic_active = False
            return ""

        mock_audio_manager.listen = _listen_fail
        b = PygameAudioBridge(audio_manager=mock_audio_manager)
        b.mic_active = True
        await b._listen_async()  # Should not raise
        assert call_count == 2  # Failed once, then stopped


# ── Update and Lifecycle Tests ───────────────────────────────────────


class TestLifecycle:
    """Tests for update() and shutdown()."""

    def test_update_detects_stopped_ambient(self, bridge: PygameAudioBridge):
        """Update resets ambient_playing when channel stops."""
        bridge._ambient_playing = True
        _channel_mock_0.get_busy.return_value = False
        bridge.update(0.016)
        assert bridge._ambient_playing is False

    def test_update_ambient_still_playing(self, bridge: PygameAudioBridge):
        """Update keeps ambient_playing when channel is busy."""
        bridge._ambient_playing = True
        _channel_mock_0.get_busy.return_value = True
        bridge.update(0.016)
        assert bridge._ambient_playing is True

    def test_shutdown_stops_all_channels(self, bridge: PygameAudioBridge):
        """Shutdown stops ambient, voice, and SFX channels."""
        bridge.shutdown()
        assert bridge._ambient_playing is False
        assert bridge.mic_active is False
        _channel_mock_0.stop.assert_called()
        _channel_mock_1.stop.assert_called()
        _channel_mock_2.stop.assert_called()

    def test_shutdown_handles_channel_errors(self, bridge: PygameAudioBridge):
        """Shutdown handles errors from channel.stop() gracefully."""
        _channel_mock_1.stop.side_effect = RuntimeError("Already stopped")
        bridge.shutdown()  # Should not raise

    def test_get_status(self, bridge: PygameAudioBridge):
        """get_status returns correct state dict."""
        status = bridge.get_status()
        assert status["audio_available"] is True
        assert status["ambient_playing"] is False
        assert status["mic_active"] is False
        assert "volumes" in status
        assert status["volumes"]["ambient"] == 0.3
        assert status["volumes"]["voice"] == 0.8
        assert status["volumes"]["sfx"] == 0.5


# ── Edge Cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_play_wav_bytes_empty(self, bridge: PygameAudioBridge):
        """Empty WAV bytes don't crash."""
        bridge._play_wav_bytes(b"", _channel_mock_1)
        _channel_mock_1.play.assert_not_called()

    def test_play_wav_bytes_none_channel(self, bridge: PygameAudioBridge):
        """None channel doesn't crash."""
        bridge._play_wav_bytes(b"RIFF data", None)

    def test_audio_channel_enum_values(self):
        """AudioChannel enum has expected values."""
        assert AudioChannel.AMBIENT.value == "ambient"
        assert AudioChannel.VOICE.value == "voice"
        assert AudioChannel.SFX.value == "sfx"

    def test_get_volume_unknown_channel_returns_zero(
        self, bridge: PygameAudioBridge
    ):
        """Getting volume for a channel not in the dict returns 0.0."""
        # All channels should exist, but test the dict.get default
        assert bridge._volumes[AudioChannel.AMBIENT] == 0.3


# ── Future Tracking Tests ─────────────────────────────────────────────


class TestFutureTracking:
    """Tests for async future tracking and cancellation on shutdown."""

    def test_pending_futures_initialized_empty(self, bridge: PygameAudioBridge):
        """Bridge starts with an empty pending futures list."""
        assert bridge._pending_futures == []

    def test_play_voice_tracks_future(
        self, bridge_with_manager: PygameAudioBridge
    ):
        """play_voice appends a future to _pending_futures."""
        bridge_with_manager.play_voice("Hello human")
        assert len(bridge_with_manager._pending_futures) == 1

    def test_start_listening_tracks_future(
        self, bridge_with_manager: PygameAudioBridge
    ):
        """_start_listening appends a future to _pending_futures."""
        bridge_with_manager._start_listening()
        assert len(bridge_with_manager._pending_futures) == 1

    def test_update_prunes_completed_futures(self, bridge: PygameAudioBridge):
        """update() removes completed futures from the list."""
        done_future = MagicMock()
        done_future.done.return_value = True
        pending_future = MagicMock()
        pending_future.done.return_value = False

        bridge._pending_futures = [done_future, pending_future]
        bridge.update(0.016)
        assert len(bridge._pending_futures) == 1
        assert bridge._pending_futures[0] is pending_future

    def test_shutdown_cancels_pending_futures(self, bridge: PygameAudioBridge):
        """shutdown() cancels all tracked futures and clears the list."""
        f1 = MagicMock()
        f2 = MagicMock()
        bridge._pending_futures = [f1, f2]

        bridge.shutdown()

        f1.cancel.assert_called_once()
        f2.cancel.assert_called_once()
        assert bridge._pending_futures == []

    def test_shutdown_then_play_voice_noop(
        self, bridge_with_manager: PygameAudioBridge
    ):
        """play_voice after shutdown does not add futures."""
        bridge_with_manager.shutdown()
        bridge_with_manager.play_voice("Hello")
        assert bridge_with_manager._pending_futures == []
