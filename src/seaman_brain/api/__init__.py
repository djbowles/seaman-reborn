"""API subsystem - FastAPI WebSocket server for UE5 bridge."""

from seaman_brain.api.protocol import (
    PROTOCOL_VERSION,
    BrainStateSnapshot,
    CreatureStateSnapshot,
    ErrorMessage,
    EventNotification,
    EventSeverity,
    InputMessage,
    MessageType,
    NeedsSnapshot,
    ResponseMessage,
    StateUpdate,
    TankSnapshot,
    TraitsSnapshot,
    check_protocol_version,
    parse_client_message,
    serialize_response,
)
from seaman_brain.api.server import BrainServer

__all__ = [
    "PROTOCOL_VERSION",
    "BrainServer",
    "BrainStateSnapshot",
    "CreatureStateSnapshot",
    "ErrorMessage",
    "EventNotification",
    "EventSeverity",
    "InputMessage",
    "MessageType",
    "NeedsSnapshot",
    "ResponseMessage",
    "StateUpdate",
    "TankSnapshot",
    "TraitsSnapshot",
    "check_protocol_version",
    "parse_client_message",
    "serialize_response",
]
