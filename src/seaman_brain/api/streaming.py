"""Real-time event streaming over WebSocket to connected clients.

Provides :class:`EventBroadcaster` which manages per-channel subscriptions,
state-diff detection, and push delivery to all connected WebSocket clients.

Channels
--------
- ``mood``      – creature mood changes
- ``needs``     – biological need updates (hunger, comfort, health, stimulation)
- ``evolution`` – stage transitions
- ``behavior``  – autonomous idle behaviours
- ``tank``      – tank environment changes (temperature, cleanliness, oxygen)
- ``death``     – creature death events
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum
from typing import Any

from fastapi import WebSocket

from seaman_brain.api.protocol import (
    BrainStateSnapshot,
    EventNotification,
    EventSeverity,
    StateUpdate,
    serialize_response,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel enum
# ---------------------------------------------------------------------------

class EventChannel(StrEnum):
    """Subscribable event channels for real-time streaming."""

    MOOD = "mood"
    NEEDS = "needs"
    EVOLUTION = "evolution"
    BEHAVIOR = "behavior"
    TANK = "tank"
    DEATH = "death"


ALL_CHANNELS: frozenset[str] = frozenset(ch.value for ch in EventChannel)
"""Complete set of valid channel names."""


# ---------------------------------------------------------------------------
# Client subscription wrapper
# ---------------------------------------------------------------------------

class ClientSubscription:
    """Tracks a single WebSocket client and its active channel subscriptions.

    Attributes:
        ws: The underlying WebSocket connection.
        channels: Set of channel names this client receives updates for.
    """

    def __init__(
        self,
        ws: WebSocket,
        channels: set[str] | None = None,
    ) -> None:
        self.ws = ws
        self.channels: set[str] = channels if channels is not None else set(ALL_CHANNELS)

    def is_subscribed(self, channel: str) -> bool:
        """Return ``True`` if this client is subscribed to *channel*."""
        return channel in self.channels

    def subscribe(self, channel: str) -> None:
        """Add *channel* to this client's subscriptions."""
        if channel not in ALL_CHANNELS:
            msg = f"Unknown channel: {channel!r}"
            raise ValueError(msg)
        self.channels.add(channel)

    def unsubscribe(self, channel: str) -> None:
        """Remove *channel* from this client's subscriptions."""
        self.channels.discard(channel)

    def subscribe_all(self) -> None:
        """Subscribe to every available channel."""
        self.channels = set(ALL_CHANNELS)

    def unsubscribe_all(self) -> None:
        """Clear all channel subscriptions."""
        self.channels.clear()


# ---------------------------------------------------------------------------
# State-diff helpers
# ---------------------------------------------------------------------------

def _flatten_dict(
    d: dict[str, Any],
    prefix: str = "",
) -> dict[str, Any]:
    """Flatten a nested dict into dot-separated keys.

    >>> _flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
    {'a.b': 1, 'a.c': 2, 'd': 3}
    """
    items: dict[str, Any] = {}
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_flatten_dict(value, full_key))
        else:
            items[full_key] = value
    return items


def compute_state_diff(
    old: dict[str, Any],
    new: dict[str, Any],
) -> list[str]:
    """Return the list of dot-separated field paths that differ between *old* and *new*.

    Both inputs should be the ``model_dump()`` output of a
    :class:`BrainStateSnapshot`.
    """
    flat_old = _flatten_dict(old)
    flat_new = _flatten_dict(new)
    all_keys = set(flat_old) | set(flat_new)
    return sorted(k for k in all_keys if flat_old.get(k) != flat_new.get(k))


def diff_to_channels(changed_fields: list[str]) -> set[str]:
    """Map a list of changed dot-paths to the relevant :class:`EventChannel` names.

    Mapping rules
    ~~~~~~~~~~~~~
    - ``creature_state.mood`` or ``mood`` → ``mood``
    - ``needs.*`` → ``needs``
    - ``tank.*`` → ``tank``
    - ``creature_state.stage`` or ``current_stage`` → ``evolution``
    - ``active_traits.*`` → ``mood`` (traits affect mood rendering)
    - Other ``creature_state.*`` → ``needs`` (hunger, health, comfort, etc.)
    """
    channels: set[str] = set()
    for path in changed_fields:
        if path in ("mood", "creature_state.mood"):
            channels.add(EventChannel.MOOD)
        elif path.startswith("needs.") or path == "needs":
            channels.add(EventChannel.NEEDS)
        elif path.startswith("tank.") or path == "tank":
            channels.add(EventChannel.TANK)
        elif path in ("current_stage", "creature_state.stage"):
            channels.add(EventChannel.EVOLUTION)
        elif path.startswith("active_traits."):
            channels.add(EventChannel.MOOD)
        elif path.startswith("creature_state."):
            channels.add(EventChannel.NEEDS)
    return channels


# ---------------------------------------------------------------------------
# EventBroadcaster
# ---------------------------------------------------------------------------

class EventBroadcaster:
    """Collects state changes and pushes updates to subscribed WebSocket clients.

    Parameters
    ----------
    broadcast_interval_ms:
        How often (in milliseconds) periodic state-diff updates are pushed
        for *needs* and *tank* channels.  Defaults to ``500``.
    """

    def __init__(self, broadcast_interval_ms: int = 500) -> None:
        self._clients: dict[WebSocket, ClientSubscription] = {}
        self._broadcast_interval_ms = broadcast_interval_ms
        self._last_snapshot: dict[str, Any] = {}
        self._broadcast_task: asyncio.Task[None] | None = None

    # -- Client management ---------------------------------------------------

    @property
    def client_count(self) -> int:
        """Number of currently connected clients."""
        return len(self._clients)

    def add_client(
        self,
        ws: WebSocket,
        channels: set[str] | None = None,
    ) -> ClientSubscription:
        """Register a WebSocket client, optionally with a channel subset.

        Returns the :class:`ClientSubscription` wrapper.
        """
        sub = ClientSubscription(ws, channels)
        self._clients[ws] = sub
        logger.info(
            "Streaming client added (%d total, channels=%s)",
            len(self._clients),
            sub.channels,
        )
        return sub

    def remove_client(self, ws: WebSocket) -> None:
        """Unregister a client.  No-op if not present."""
        if ws in self._clients:
            del self._clients[ws]
            logger.info(
                "Streaming client removed (%d remaining)", len(self._clients)
            )

    def get_subscription(self, ws: WebSocket) -> ClientSubscription | None:
        """Return the :class:`ClientSubscription` for *ws*, or ``None``."""
        return self._clients.get(ws)

    def subscribe(self, ws: WebSocket, channel: str) -> None:
        """Subscribe an existing client to *channel*.

        Raises :class:`KeyError` if the client is not registered.
        """
        sub = self._clients.get(ws)
        if sub is None:
            msg = "Client not registered"
            raise KeyError(msg)
        sub.subscribe(channel)

    def unsubscribe(self, ws: WebSocket, channel: str) -> None:
        """Unsubscribe an existing client from *channel*.

        Raises :class:`KeyError` if the client is not registered.
        """
        sub = self._clients.get(ws)
        if sub is None:
            msg = "Client not registered"
            raise KeyError(msg)
        sub.unsubscribe(channel)

    # -- Broadcasting --------------------------------------------------------

    async def broadcast_state_update(
        self,
        snapshot: BrainStateSnapshot,
        changed_fields: list[str] | None = None,
    ) -> None:
        """Push a :class:`StateUpdate` to all clients subscribed to affected channels.

        Parameters
        ----------
        snapshot:
            Current full brain state.
        changed_fields:
            Explicit list of changed field paths.  If *None*, the broadcaster
            compares against its last cached snapshot to compute the diff.
        """
        snapshot_dict = snapshot.model_dump()
        if changed_fields is None:
            changed_fields = compute_state_diff(self._last_snapshot, snapshot_dict)
        if not changed_fields:
            return  # nothing changed
        self._last_snapshot = snapshot_dict

        affected_channels = diff_to_channels(changed_fields)
        if not affected_channels:
            return

        update = StateUpdate(
            state=snapshot,
            changed_fields=changed_fields,
        )
        payload = serialize_response(update)
        await self._send_to_subscribed(payload, affected_channels)

    async def broadcast_event(
        self,
        event_type: str,
        message: str,
        severity: EventSeverity = EventSeverity.INFO,
        effects: dict[str, Any] | None = None,
        channel: str | None = None,
    ) -> None:
        """Push an :class:`EventNotification` instantly to subscribed clients.

        Parameters
        ----------
        event_type:
            Event category string (e.g. ``"evolution_ready"``).
        message:
            Human-readable description.
        severity:
            Urgency level.
        effects:
            Optional side-effect dict.
        channel:
            Channel to filter by.  If ``None``, the event is inferred from
            *event_type* or sent to all clients.
        """
        notification = EventNotification(
            event_type=event_type,
            severity=severity,
            message=message,
            effects=effects or {},
        )
        payload = serialize_response(notification)

        if channel is not None:
            await self._send_to_subscribed(payload, {channel})
        else:
            inferred = self._infer_channel(event_type)
            if inferred:
                await self._send_to_subscribed(payload, {inferred})
            else:
                await self._send_to_all(payload)

    # -- Periodic diff loop --------------------------------------------------

    def start(self, snapshot_fn: Any) -> None:
        """Start the periodic broadcast loop.

        Parameters
        ----------
        snapshot_fn:
            A callable (sync or async) returning a :class:`BrainStateSnapshot`.
        """
        self._broadcast_task = asyncio.create_task(
            self._periodic_loop(snapshot_fn)
        )

    def stop(self) -> None:
        """Cancel the periodic broadcast task if running."""
        if self._broadcast_task is not None:
            self._broadcast_task.cancel()
            self._broadcast_task = None

    @property
    def is_running(self) -> bool:
        """Whether the periodic loop is active."""
        return self._broadcast_task is not None and not self._broadcast_task.done()

    async def _periodic_loop(self, snapshot_fn: Any) -> None:
        """Background loop that periodically diffs and broadcasts state."""
        interval = self._broadcast_interval_ms / 1000.0
        while True:
            await asyncio.sleep(interval)
            try:
                snapshot = snapshot_fn()
                if asyncio.iscoroutine(snapshot):
                    snapshot = await snapshot
                if snapshot is not None:
                    await self.broadcast_state_update(snapshot)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Error in periodic broadcast loop")

    # -- Internal send helpers -----------------------------------------------

    async def _send_to_subscribed(
        self,
        payload: str,
        channels: set[str],
    ) -> None:
        """Send *payload* to all clients subscribed to at least one of *channels*."""
        disconnected: list[WebSocket] = []
        for ws, sub in list(self._clients.items()):
            if any(sub.is_subscribed(ch) for ch in channels):
                try:
                    await ws.send_text(payload)
                except Exception:
                    logger.info("Client disconnected during send, removing")
                    disconnected.append(ws)
        for ws in disconnected:
            self.remove_client(ws)

    async def _send_to_all(self, payload: str) -> None:
        """Send *payload* to every connected client regardless of subscriptions."""
        disconnected: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await ws.send_text(payload)
            except Exception:
                logger.info("Client disconnected during send, removing")
                disconnected.append(ws)
        for ws in disconnected:
            self.remove_client(ws)

    @staticmethod
    def _infer_channel(event_type: str) -> str | None:
        """Try to map an *event_type* string to a channel name."""
        et = event_type.lower()
        if "evolution" in et or "stage" in et:
            return EventChannel.EVOLUTION
        if "death" in et or "die" in et:
            return EventChannel.DEATH
        if "mood" in et:
            return EventChannel.MOOD
        if "need" in et or "hunger" in et or "health" in et:
            return EventChannel.NEEDS
        if "tank" in et or "temperature" in et or "clean" in et:
            return EventChannel.TANK
        if "behavior" in et or "idle" in et or "behaviour" in et:
            return EventChannel.BEHAVIOR
        return None
