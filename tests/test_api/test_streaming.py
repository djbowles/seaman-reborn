"""Tests for the real-time event streaming system (US-045).

Covers:
- EventChannel enum and ALL_CHANNELS constant
- ClientSubscription management (subscribe, unsubscribe, channel filtering)
- State-diff computation (compute_state_diff, diff_to_channels)
- EventBroadcaster: client lifecycle, state update broadcasting, event broadcasting
- Channel-filtered delivery (only subscribed clients receive updates)
- Graceful client disconnect handling
- Periodic broadcast loop
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from seaman_brain.api.protocol import (
    BrainStateSnapshot,
    CreatureStateSnapshot,
    EventSeverity,
    NeedsSnapshot,
    TankSnapshot,
)
from seaman_brain.api.streaming import (
    ALL_CHANNELS,
    ClientSubscription,
    EventBroadcaster,
    EventChannel,
    _flatten_dict,
    compute_state_diff,
    diff_to_channels,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_ws() -> AsyncMock:
    """Create a mock WebSocket with async send_text."""
    ws = AsyncMock()
    ws.send_text = AsyncMock()
    return ws


def _default_snapshot() -> BrainStateSnapshot:
    return BrainStateSnapshot()


def _changed_snapshot(**creature_kw: object) -> BrainStateSnapshot:
    """Return a BrainStateSnapshot with customised CreatureStateSnapshot fields."""
    cs = CreatureStateSnapshot(**creature_kw)
    return BrainStateSnapshot(
        creature_state=cs,
        mood=str(creature_kw.get("mood", "neutral")),
        current_stage=str(creature_kw.get("stage", "mushroomer")),
    )


# ===========================================================================
# EventChannel enum
# ===========================================================================


class TestEventChannel:
    """Tests for the EventChannel StrEnum."""

    def test_all_six_channels_exist(self) -> None:
        assert len(EventChannel) == 6

    def test_channel_values(self) -> None:
        assert EventChannel.MOOD == "mood"
        assert EventChannel.NEEDS == "needs"
        assert EventChannel.EVOLUTION == "evolution"
        assert EventChannel.BEHAVIOR == "behavior"
        assert EventChannel.TANK == "tank"
        assert EventChannel.DEATH == "death"

    def test_all_channels_constant_matches_enum(self) -> None:
        assert ALL_CHANNELS == frozenset(ch.value for ch in EventChannel)

    def test_channel_is_str(self) -> None:
        for ch in EventChannel:
            assert isinstance(ch, str)


# ===========================================================================
# ClientSubscription
# ===========================================================================


class TestClientSubscription:
    """Tests for per-client channel subscription management."""

    def test_default_subscribes_all_channels(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws)
        assert sub.channels == set(ALL_CHANNELS)
        for ch in EventChannel:
            assert sub.is_subscribed(ch)

    def test_custom_channel_subset(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws, channels={"mood", "needs"})
        assert sub.is_subscribed("mood")
        assert sub.is_subscribed("needs")
        assert not sub.is_subscribed("tank")

    def test_subscribe_adds_channel(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws, channels=set())
        sub.subscribe("mood")
        assert sub.is_subscribed("mood")

    def test_subscribe_unknown_channel_raises(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws, channels=set())
        with pytest.raises(ValueError, match="Unknown channel"):
            sub.subscribe("invalid_channel")

    def test_unsubscribe_removes_channel(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws)
        sub.unsubscribe("mood")
        assert not sub.is_subscribed("mood")

    def test_unsubscribe_nonexistent_is_noop(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws, channels=set())
        sub.unsubscribe("mood")  # should not raise
        assert not sub.is_subscribed("mood")

    def test_subscribe_all(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws, channels=set())
        sub.subscribe_all()
        assert sub.channels == set(ALL_CHANNELS)

    def test_unsubscribe_all(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws)
        sub.unsubscribe_all()
        assert len(sub.channels) == 0


# ===========================================================================
# State-diff helpers
# ===========================================================================


class TestFlattenDict:
    """Tests for the _flatten_dict helper."""

    def test_flat_input(self) -> None:
        assert _flatten_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}

    def test_nested_input(self) -> None:
        result = _flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        assert result == {"a.b": 1, "a.c": 2, "d": 3}

    def test_deeply_nested(self) -> None:
        result = _flatten_dict({"x": {"y": {"z": 42}}})
        assert result == {"x.y.z": 42}

    def test_empty_dict(self) -> None:
        assert _flatten_dict({}) == {}


class TestComputeStateDiff:
    """Tests for compute_state_diff."""

    def test_identical_dicts_no_diff(self) -> None:
        snap = _default_snapshot().model_dump()
        assert compute_state_diff(snap, snap) == []

    def test_top_level_change(self) -> None:
        old = {"mood": "neutral", "current_stage": "mushroomer"}
        new = {"mood": "sardonic", "current_stage": "mushroomer"}
        diff = compute_state_diff(old, new)
        assert "mood" in diff

    def test_nested_change(self) -> None:
        old = {"needs": {"hunger": 0.0, "health": 1.0}}
        new = {"needs": {"hunger": 0.5, "health": 1.0}}
        diff = compute_state_diff(old, new)
        assert "needs.hunger" in diff
        assert "needs.health" not in diff

    def test_added_key(self) -> None:
        old = {"a": 1}
        new = {"a": 1, "b": 2}
        diff = compute_state_diff(old, new)
        assert "b" in diff

    def test_removed_key(self) -> None:
        old = {"a": 1, "b": 2}
        new = {"a": 1}
        diff = compute_state_diff(old, new)
        assert "b" in diff

    def test_full_snapshot_diff(self) -> None:
        old = _default_snapshot().model_dump()
        new_snap = BrainStateSnapshot(
            mood="sardonic",
            needs=NeedsSnapshot(hunger=0.8),
        )
        new = new_snap.model_dump()
        diff = compute_state_diff(old, new)
        assert "mood" in diff
        assert "needs.hunger" in diff


class TestDiffToChannels:
    """Tests for diff_to_channels mapping."""

    def test_mood_field(self) -> None:
        assert EventChannel.MOOD in diff_to_channels(["mood"])

    def test_creature_state_mood(self) -> None:
        assert EventChannel.MOOD in diff_to_channels(["creature_state.mood"])

    def test_needs_fields(self) -> None:
        channels = diff_to_channels(["needs.hunger", "needs.health"])
        assert EventChannel.NEEDS in channels

    def test_tank_fields(self) -> None:
        channels = diff_to_channels(["tank.temperature", "tank.cleanliness"])
        assert EventChannel.TANK in channels

    def test_evolution_stage(self) -> None:
        channels = diff_to_channels(["current_stage"])
        assert EventChannel.EVOLUTION in channels

    def test_creature_state_stage(self) -> None:
        channels = diff_to_channels(["creature_state.stage"])
        assert EventChannel.EVOLUTION in channels

    def test_active_traits_maps_to_mood(self) -> None:
        channels = diff_to_channels(["active_traits.cynicism"])
        assert EventChannel.MOOD in channels

    def test_creature_state_other_maps_to_needs(self) -> None:
        channels = diff_to_channels(["creature_state.hunger"])
        assert EventChannel.NEEDS in channels

    def test_empty_diff(self) -> None:
        assert diff_to_channels([]) == set()

    def test_multiple_channels(self) -> None:
        channels = diff_to_channels(["mood", "needs.hunger", "tank.temperature"])
        assert EventChannel.MOOD in channels
        assert EventChannel.NEEDS in channels
        assert EventChannel.TANK in channels


# ===========================================================================
# EventBroadcaster — client management
# ===========================================================================


class TestBroadcasterClientManagement:
    """Tests for EventBroadcaster add/remove/subscribe/unsubscribe."""

    def test_add_client_default_channels(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        sub = eb.add_client(ws)
        assert eb.client_count == 1
        assert sub.channels == set(ALL_CHANNELS)

    def test_add_client_custom_channels(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        sub = eb.add_client(ws, channels={"mood"})
        assert sub.channels == {"mood"}

    def test_remove_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        eb.remove_client(ws)
        assert eb.client_count == 0

    def test_remove_unknown_client_is_noop(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.remove_client(ws)  # should not raise
        assert eb.client_count == 0

    def test_get_subscription(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        sub = eb.add_client(ws)
        assert eb.get_subscription(ws) is sub

    def test_get_subscription_unknown_returns_none(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        assert eb.get_subscription(ws) is None

    def test_subscribe_channel(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels=set())
        eb.subscribe(ws, "mood")
        sub = eb.get_subscription(ws)
        assert sub is not None
        assert sub.is_subscribed("mood")

    def test_subscribe_unregistered_client_raises(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        with pytest.raises(KeyError, match="Client not registered"):
            eb.subscribe(ws, "mood")

    def test_unsubscribe_channel(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        eb.unsubscribe(ws, "mood")
        sub = eb.get_subscription(ws)
        assert sub is not None
        assert not sub.is_subscribed("mood")

    def test_unsubscribe_unregistered_client_raises(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        with pytest.raises(KeyError, match="Client not registered"):
            eb.unsubscribe(ws, "mood")

    def test_multiple_clients(self) -> None:
        eb = EventBroadcaster()
        ws1, ws2, ws3 = _mock_ws(), _mock_ws(), _mock_ws()
        eb.add_client(ws1)
        eb.add_client(ws2)
        eb.add_client(ws3)
        assert eb.client_count == 3
        eb.remove_client(ws2)
        assert eb.client_count == 2


# ===========================================================================
# EventBroadcaster — state update broadcasting
# ===========================================================================


class TestBroadcasterStateUpdates:
    """Tests for EventBroadcaster.broadcast_state_update."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_subscribed_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"mood"})
        snapshot = BrainStateSnapshot(mood="sardonic")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )
        ws.send_text.assert_called_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "state_update"
        assert "mood" in payload["changed_fields"]

    @pytest.mark.asyncio
    async def test_broadcast_skips_unsubscribed_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"tank"})  # only tank
        snapshot = BrainStateSnapshot(mood="sardonic")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_no_diff_no_send(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        snapshot = _default_snapshot()
        # Set last snapshot to same
        eb._last_snapshot = snapshot.model_dump()
        await eb.broadcast_state_update(snapshot)
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_auto_computes_diff(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        # Set last snapshot to default
        eb._last_snapshot = _default_snapshot().model_dump()
        # Broadcast with changed mood
        new_snap = BrainStateSnapshot(mood="sardonic")
        await eb.broadcast_state_update(new_snap)
        ws.send_text.assert_called_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert "mood" in payload["changed_fields"]

    @pytest.mark.asyncio
    async def test_broadcast_updates_last_snapshot(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        snapshot = BrainStateSnapshot(mood="hostile")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )
        assert eb._last_snapshot == snapshot.model_dump()

    @pytest.mark.asyncio
    async def test_broadcast_empty_changed_fields_no_send(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        snapshot = _default_snapshot()
        await eb.broadcast_state_update(snapshot, changed_fields=[])
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_broadcast_multiple_clients_selective(self) -> None:
        eb = EventBroadcaster()
        ws_mood = _mock_ws()
        ws_tank = _mock_ws()
        ws_all = _mock_ws()
        eb.add_client(ws_mood, channels={"mood"})
        eb.add_client(ws_tank, channels={"tank"})
        eb.add_client(ws_all)
        snapshot = BrainStateSnapshot(mood="amused")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )
        ws_mood.send_text.assert_called_once()
        ws_tank.send_text.assert_not_called()
        ws_all.send_text.assert_called_once()


# ===========================================================================
# EventBroadcaster — event broadcasting
# ===========================================================================


class TestBroadcasterEventNotifications:
    """Tests for EventBroadcaster.broadcast_event."""

    @pytest.mark.asyncio
    async def test_event_sent_to_subscribed_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"evolution"})
        await eb.broadcast_event(
            event_type="evolution_ready",
            message="Ready to evolve!",
            severity=EventSeverity.WARNING,
        )
        ws.send_text.assert_called_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["type"] == "event"
        assert payload["event_type"] == "evolution_ready"
        assert payload["severity"] == "warning"
        assert payload["message"] == "Ready to evolve!"

    @pytest.mark.asyncio
    async def test_event_skips_unsubscribed_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"tank"})
        await eb.broadcast_event(
            event_type="evolution_ready",
            message="Ready to evolve!",
        )
        ws.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_explicit_channel(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"death"})
        await eb.broadcast_event(
            event_type="creature_died",
            message="The creature has perished.",
            severity=EventSeverity.CRITICAL,
            channel="death",
        )
        ws.send_text.assert_called_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_event_with_effects(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws)
        await eb.broadcast_event(
            event_type="tank_cleaned",
            message="Tank cleaned",
            effects={"cleanliness": 1.0},
            channel="tank",
        )
        ws.send_text.assert_called_once()
        payload = json.loads(ws.send_text.call_args[0][0])
        assert payload["effects"]["cleanliness"] == 1.0

    @pytest.mark.asyncio
    async def test_event_unknown_type_broadcasts_to_all(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels=set())  # no subscriptions
        # Unknown event type with no explicit channel → _infer_channel returns None → send to all
        await eb.broadcast_event(
            event_type="custom_unknown",
            message="Something happened.",
        )
        ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_infer_death_channel(self) -> None:
        eb = EventBroadcaster()
        ws_death = _mock_ws()
        ws_mood = _mock_ws()
        eb.add_client(ws_death, channels={"death"})
        eb.add_client(ws_mood, channels={"mood"})
        await eb.broadcast_event(
            event_type="creature_death",
            message="RIP",
            severity=EventSeverity.CRITICAL,
        )
        ws_death.send_text.assert_called_once()
        ws_mood.send_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_event_infer_behavior_channel(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"behavior"})
        await eb.broadcast_event(
            event_type="idle_behavior",
            message="Creature is swimming idly.",
        )
        ws.send_text.assert_called_once()


# ===========================================================================
# EventBroadcaster — client disconnect handling
# ===========================================================================


class TestBroadcasterDisconnectHandling:
    """Tests for graceful client disconnection handling."""

    @pytest.mark.asyncio
    async def test_disconnect_during_state_broadcast_removes_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        ws.send_text.side_effect = Exception("Connection lost")
        eb.add_client(ws)
        snapshot = BrainStateSnapshot(mood="hostile")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )
        assert eb.client_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_during_event_broadcast_removes_client(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        ws.send_text.side_effect = Exception("Connection closed")
        eb.add_client(ws, channels={"death"})
        await eb.broadcast_event(
            event_type="creature_death",
            message="Dead",
            channel="death",
        )
        assert eb.client_count == 0

    @pytest.mark.asyncio
    async def test_healthy_clients_survive_peer_disconnect(self) -> None:
        eb = EventBroadcaster()
        ws_bad = _mock_ws()
        ws_good = _mock_ws()
        ws_bad.send_text.side_effect = Exception("Gone")
        eb.add_client(ws_bad)
        eb.add_client(ws_good)
        snapshot = BrainStateSnapshot(mood="irritated")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )
        assert eb.client_count == 1
        ws_good.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_all_clients_disconnect(self) -> None:
        eb = EventBroadcaster()
        ws1 = _mock_ws()
        ws2 = _mock_ws()
        ws1.send_text.side_effect = Exception("Gone")
        ws2.send_text.side_effect = Exception("Gone")
        eb.add_client(ws1)
        eb.add_client(ws2)
        await eb.broadcast_event(
            event_type="custom_unknown",
            message="Bye",
        )
        assert eb.client_count == 0


# ===========================================================================
# EventBroadcaster — periodic loop
# ===========================================================================


class TestBroadcasterPeriodicLoop:
    """Tests for the periodic broadcast loop (start/stop/is_running)."""

    @pytest.mark.asyncio
    async def test_start_and_stop(self) -> None:
        eb = EventBroadcaster(broadcast_interval_ms=50)
        ws = _mock_ws()
        eb.add_client(ws)
        snapshot = BrainStateSnapshot(mood="sardonic")
        eb.start(lambda: snapshot)
        assert eb.is_running
        await asyncio.sleep(0.15)  # allow a couple ticks
        eb.stop()
        assert not eb.is_running

    @pytest.mark.asyncio
    async def test_periodic_loop_detects_changes(self) -> None:
        eb = EventBroadcaster(broadcast_interval_ms=50)
        ws = _mock_ws()
        eb.add_client(ws)
        # First tick: default -> sardonic (should send)
        call_count = 0

        def snapshot_fn():
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _default_snapshot()
            return BrainStateSnapshot(mood="sardonic")

        eb.start(snapshot_fn)
        await asyncio.sleep(0.2)
        eb.stop()
        # Should have sent at least once when mood changed
        assert ws.send_text.call_count >= 1

    @pytest.mark.asyncio
    async def test_periodic_loop_handles_errors(self) -> None:
        eb = EventBroadcaster(broadcast_interval_ms=50)
        ws = _mock_ws()
        eb.add_client(ws)

        def bad_snapshot():
            raise RuntimeError("snapshot failed")

        eb.start(bad_snapshot)
        await asyncio.sleep(0.15)  # should not crash
        assert eb.is_running
        eb.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self) -> None:
        eb = EventBroadcaster()
        eb.stop()  # should not raise
        assert not eb.is_running

    @pytest.mark.asyncio
    async def test_periodic_loop_with_async_snapshot_fn(self) -> None:
        eb = EventBroadcaster(broadcast_interval_ms=50)
        ws = _mock_ws()
        eb.add_client(ws)

        async def async_snapshot():
            return BrainStateSnapshot(mood="philosophical")

        eb.start(async_snapshot)
        await asyncio.sleep(0.15)
        eb.stop()
        # Should have sent at least once
        assert ws.send_text.call_count >= 1


# ===========================================================================
# EventBroadcaster — infer channel
# ===========================================================================


class TestInferChannel:
    """Tests for EventBroadcaster._infer_channel."""

    def test_evolution_keywords(self) -> None:
        assert EventBroadcaster._infer_channel("evolution_ready") == "evolution"
        assert EventBroadcaster._infer_channel("stage_change") == "evolution"

    def test_death_keywords(self) -> None:
        assert EventBroadcaster._infer_channel("creature_death") == "death"
        assert EventBroadcaster._infer_channel("die_of_starvation") == "death"

    def test_mood_keyword(self) -> None:
        assert EventBroadcaster._infer_channel("mood_shift") == "mood"

    def test_needs_keywords(self) -> None:
        assert EventBroadcaster._infer_channel("need_alert") == "needs"
        assert EventBroadcaster._infer_channel("hunger_warning") == "needs"
        assert EventBroadcaster._infer_channel("health_critical") == "needs"

    def test_tank_keywords(self) -> None:
        assert EventBroadcaster._infer_channel("tank_dirty") == "tank"
        assert EventBroadcaster._infer_channel("temperature_high") == "tank"
        assert EventBroadcaster._infer_channel("clean_tank") == "tank"

    def test_behavior_keywords(self) -> None:
        assert EventBroadcaster._infer_channel("idle_behavior") == "behavior"
        assert EventBroadcaster._infer_channel("behaviour_update") == "behavior"

    def test_unknown_returns_none(self) -> None:
        assert EventBroadcaster._infer_channel("custom_event") is None
        assert EventBroadcaster._infer_channel("random_thing") is None


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge case tests for streaming module."""

    def test_broadcaster_default_interval(self) -> None:
        eb = EventBroadcaster()
        assert eb._broadcast_interval_ms == 500

    def test_broadcaster_custom_interval(self) -> None:
        eb = EventBroadcaster(broadcast_interval_ms=100)
        assert eb._broadcast_interval_ms == 100

    @pytest.mark.asyncio
    async def test_broadcast_with_no_clients(self) -> None:
        eb = EventBroadcaster()
        # Should not raise
        snapshot = BrainStateSnapshot(mood="hostile")
        await eb.broadcast_state_update(
            snapshot, changed_fields=["mood"]
        )

    @pytest.mark.asyncio
    async def test_event_broadcast_with_no_clients(self) -> None:
        eb = EventBroadcaster()
        # Should not raise
        await eb.broadcast_event(
            event_type="evolution_ready",
            message="Ready!",
        )

    def test_client_subscription_ws_reference(self) -> None:
        ws = _mock_ws()
        sub = ClientSubscription(ws)
        assert sub.ws is ws

    @pytest.mark.asyncio
    async def test_broadcast_needs_change_to_needs_subscriber(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"needs"})
        snapshot = BrainStateSnapshot(
            needs=NeedsSnapshot(hunger=0.9),
        )
        await eb.broadcast_state_update(
            snapshot, changed_fields=["needs.hunger"]
        )
        ws.send_text.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_tank_change_to_tank_subscriber(self) -> None:
        eb = EventBroadcaster()
        ws = _mock_ws()
        eb.add_client(ws, channels={"tank"})
        snapshot = BrainStateSnapshot(
            tank=TankSnapshot(temperature=30.0),
        )
        await eb.broadcast_state_update(
            snapshot, changed_fields=["tank.temperature"]
        )
        ws.send_text.assert_called_once()
