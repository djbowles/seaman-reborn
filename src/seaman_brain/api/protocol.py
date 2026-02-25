"""Brain state protocol schema - Pydantic models for API communication.

Defines the JSON protocol for communication between the Seaman brain server
and external clients (UE5, web dashboards, etc.).  Every message exchanged
over the ``/ws/brain`` WebSocket or returned by REST endpoints conforms to
one of the models defined here.

Protocol version: ``1.0.0``
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROTOCOL_VERSION: str = "1.0.0"
"""Semantic version of the wire protocol."""


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MessageType(StrEnum):
    """Discriminator for top-level WebSocket messages."""

    INPUT = "input"
    RESPONSE = "response"
    STATE_UPDATE = "state_update"
    EVENT = "event"
    ERROR = "error"


class EventSeverity(StrEnum):
    """How urgent / impactful an event notification is."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ---------------------------------------------------------------------------
# Snapshot sub-models
# ---------------------------------------------------------------------------

class NeedsSnapshot(BaseModel):
    """Current biological needs of the creature.

    All values are floats in [0.0, 1.0].

    Attributes:
        hunger: 0.0 = full, 1.0 = starving.
        comfort: 0.0 = miserable, 1.0 = comfortable.
        health: 0.0 = dead, 1.0 = healthy.
        stimulation: 0.0 = bored, 1.0 = engaged.
    """

    hunger: float = Field(0.0, ge=0.0, le=1.0, description="0.0=full, 1.0=starving")
    comfort: float = Field(1.0, ge=0.0, le=1.0, description="0.0=miserable, 1.0=comfortable")
    health: float = Field(1.0, ge=0.0, le=1.0, description="0.0=dead, 1.0=healthy")
    stimulation: float = Field(1.0, ge=0.0, le=1.0, description="0.0=bored, 1.0=engaged")


class TankSnapshot(BaseModel):
    """Current state of the creature's tank / habitat.

    Attributes:
        temperature: Celsius temperature of the tank.
        cleanliness: 0.0=filthy, 1.0=spotless.
        oxygen_level: 0.0=none, 1.0=saturated.
        water_level: 0.0=empty, 1.0=full (0 in terrarium mode).
        environment_type: ``"aquarium"`` or ``"terrarium"``.
    """

    temperature: float = Field(24.0, description="Tank temperature in Celsius")
    cleanliness: float = Field(1.0, ge=0.0, le=1.0, description="0.0=filthy, 1.0=spotless")
    oxygen_level: float = Field(1.0, ge=0.0, le=1.0, description="0.0=none, 1.0=saturated")
    water_level: float = Field(1.0, ge=0.0, le=1.0, description="0.0=empty, 1.0=full")
    environment_type: str = Field("aquarium", description="'aquarium' or 'terrarium'")

    @field_validator("environment_type")
    @classmethod
    def _validate_env_type(cls, v: str) -> str:
        allowed = {"aquarium", "terrarium"}
        if v not in allowed:
            msg = f"environment_type must be one of {allowed}, got {v!r}"
            raise ValueError(msg)
        return v


class TraitsSnapshot(BaseModel):
    """Active personality trait values.  Each is a float in [0.0, 1.0].

    Attributes:
        cynicism: Distrust / sardonic worldview.
        wit: Clever humour and wordplay.
        patience: Tolerance before irritation.
        curiosity: Interest in learning / asking questions.
        warmth: Affection toward the human.
        verbosity: Length and detail in responses.
        formality: Politeness and structure in speech.
        aggression: Hostility and combativeness.
    """

    cynicism: float = Field(0.5, ge=0.0, le=1.0)
    wit: float = Field(0.5, ge=0.0, le=1.0)
    patience: float = Field(0.5, ge=0.0, le=1.0)
    curiosity: float = Field(0.5, ge=0.0, le=1.0)
    warmth: float = Field(0.5, ge=0.0, le=1.0)
    verbosity: float = Field(0.5, ge=0.0, le=1.0)
    formality: float = Field(0.5, ge=0.0, le=1.0)
    aggression: float = Field(0.5, ge=0.0, le=1.0)


class CreatureStateSnapshot(BaseModel):
    """Full creature state as exposed to API clients.

    Attributes:
        stage: Current evolutionary stage name (e.g. ``"mushroomer"``).
        age: Total seconds the creature has existed.
        interaction_count: Lifetime user interactions.
        mood: Named mood string (e.g. ``"sardonic"``, ``"content"``).
        trust_level: 0.0–1.0 trust meter.
        hunger: 0.0=full, 1.0=starving.
        health: 0.0=dead, 1.0=healthy.
        comfort: 0.0=miserable, 1.0=comfortable.
        last_fed: ISO-8601 timestamp of last feeding.
        last_interaction: ISO-8601 timestamp of last user interaction.
        birth_time: ISO-8601 timestamp of creature birth.
    """

    stage: str = Field("mushroomer", description="Evolutionary stage name")
    age: float = Field(0.0, ge=0.0, description="Seconds since birth")
    interaction_count: int = Field(0, ge=0, description="Lifetime interactions")
    mood: str = Field("neutral", description="Named mood state")
    trust_level: float = Field(0.0, ge=0.0, le=1.0, description="Trust meter 0-1")
    hunger: float = Field(0.0, ge=0.0, le=1.0, description="0=full, 1=starving")
    health: float = Field(1.0, ge=0.0, le=1.0, description="0=dead, 1=healthy")
    comfort: float = Field(1.0, ge=0.0, le=1.0, description="0=miserable, 1=happy")
    last_fed: str = Field(default="", description="ISO-8601 last feeding timestamp")
    last_interaction: str = Field(default="", description="ISO-8601 last interaction timestamp")
    birth_time: str = Field(default="", description="ISO-8601 birth timestamp")

    @field_validator("stage")
    @classmethod
    def _validate_stage(cls, v: str) -> str:
        allowed = {"mushroomer", "gillman", "podfish", "tadman", "frogman"}
        if v not in allowed:
            msg = f"stage must be one of {allowed}, got {v!r}"
            raise ValueError(msg)
        return v


class BrainStateSnapshot(BaseModel):
    """Aggregate snapshot of the entire brain state.

    This is the top-level state object sent in ``state_update`` messages
    and returned by ``GET /api/state``.

    Attributes:
        creature_state: Creature vitals and metadata.
        needs: Biological needs breakdown.
        tank: Tank environment conditions.
        mood: Current mood string (convenience duplicate of creature_state.mood).
        active_traits: Current personality trait values.
        current_stage: Current stage name (convenience duplicate of creature_state.stage).
    """

    creature_state: CreatureStateSnapshot = Field(
        default_factory=CreatureStateSnapshot,
        description="Creature vitals and metadata",
    )
    needs: NeedsSnapshot = Field(
        default_factory=NeedsSnapshot,
        description="Biological needs breakdown",
    )
    tank: TankSnapshot = Field(
        default_factory=TankSnapshot,
        description="Tank environment conditions",
    )
    mood: str = Field("neutral", description="Current mood string")
    active_traits: TraitsSnapshot = Field(
        default_factory=TraitsSnapshot,
        description="Active personality trait values",
    )
    current_stage: str = Field("mushroomer", description="Current evolutionary stage")


# ---------------------------------------------------------------------------
# Client -> Server messages
# ---------------------------------------------------------------------------

class InputMessage(BaseModel):
    """Message sent by the client to deliver user text to the brain.

    Attributes:
        type: Must be ``"input"``.
        text: The user's chat message text.
        protocol_version: Wire protocol version for compatibility checks.
        timestamp: ISO-8601 timestamp when the message was created.
    """

    type: MessageType = Field(MessageType.INPUT, description="Must be 'input'")
    text: str = Field(..., min_length=1, description="User chat message text")
    protocol_version: str = Field(
        PROTOCOL_VERSION, description="Wire protocol version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 message creation timestamp",
    )


# ---------------------------------------------------------------------------
# Server -> Client messages
# ---------------------------------------------------------------------------

class ResponseMessage(BaseModel):
    """Creature response sent back to the client after an input message.

    Attributes:
        type: Always ``"response"``.
        text: The creature's reply text.
        state: Full brain state snapshot at time of response.
        protocol_version: Wire protocol version.
        timestamp: ISO-8601 timestamp of response generation.
    """

    type: MessageType = Field(MessageType.RESPONSE, description="Always 'response'")
    text: str = Field(..., description="Creature reply text")
    state: BrainStateSnapshot = Field(
        default_factory=BrainStateSnapshot,
        description="Full brain state snapshot",
    )
    protocol_version: str = Field(
        PROTOCOL_VERSION, description="Wire protocol version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 response timestamp",
    )


class StateUpdate(BaseModel):
    """Unsolicited state update pushed to connected clients.

    Sent periodically or when significant state changes occur (mood shift,
    need alert, evolution event).

    Attributes:
        type: Always ``"state_update"``.
        state: Full or partial brain state snapshot.
        changed_fields: List of top-level field paths that changed since last push.
        protocol_version: Wire protocol version.
        timestamp: ISO-8601 timestamp of the update.
    """

    type: MessageType = Field(
        MessageType.STATE_UPDATE, description="Always 'state_update'"
    )
    state: BrainStateSnapshot = Field(
        default_factory=BrainStateSnapshot,
        description="Full brain state snapshot",
    )
    changed_fields: list[str] = Field(
        default_factory=list,
        description="Top-level field paths that changed since last push",
    )
    protocol_version: str = Field(
        PROTOCOL_VERSION, description="Wire protocol version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 update timestamp",
    )


class EventNotification(BaseModel):
    """Notification about a game event (evolution, milestone, holiday, etc.).

    Attributes:
        type: Always ``"event"``.
        event_type: Category of event (e.g. ``"evolution_ready"``, ``"holiday"``).
        severity: Urgency level — ``"info"``, ``"warning"``, or ``"critical"``.
        message: Human-readable event description.
        effects: Dict of side-effects applied to game state.
        protocol_version: Wire protocol version.
        timestamp: ISO-8601 timestamp of event occurrence.
    """

    type: MessageType = Field(MessageType.EVENT, description="Always 'event'")
    event_type: str = Field(
        ..., description="Event category (e.g. 'evolution_ready', 'holiday')"
    )
    severity: EventSeverity = Field(
        EventSeverity.INFO, description="Urgency level"
    )
    message: str = Field("", description="Human-readable event description")
    effects: dict[str, Any] = Field(
        default_factory=dict,
        description="Side-effects applied to game state",
    )
    protocol_version: str = Field(
        PROTOCOL_VERSION, description="Wire protocol version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 event timestamp",
    )


class ErrorMessage(BaseModel):
    """Error message sent when the server cannot process a client request.

    Attributes:
        type: Always ``"error"``.
        message: Human-readable error description.
        protocol_version: Wire protocol version.
        timestamp: ISO-8601 error timestamp.
    """

    type: MessageType = Field(MessageType.ERROR, description="Always 'error'")
    message: str = Field(..., description="Human-readable error description")
    protocol_version: str = Field(
        PROTOCOL_VERSION, description="Wire protocol version"
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(),
        description="ISO-8601 error timestamp",
    )


# ---------------------------------------------------------------------------
# Serialization / deserialization utilities
# ---------------------------------------------------------------------------

def parse_client_message(raw: str | dict[str, Any]) -> InputMessage:
    """Parse and validate a raw client message into an ``InputMessage``.

    Args:
        raw: JSON string or already-parsed dict.

    Returns:
        Validated ``InputMessage``.

    Raises:
        ValueError: If the message is malformed or has wrong type.
    """
    import json as _json

    data: dict[str, Any]
    if isinstance(raw, str):
        try:
            data = _json.loads(raw)
        except _json.JSONDecodeError as exc:
            msg = f"Invalid JSON: {exc}"
            raise ValueError(msg) from exc
    else:
        data = raw

    if not isinstance(data, dict):
        msg = f"Expected a JSON object, got {type(data).__name__}"
        raise ValueError(msg)

    return InputMessage.model_validate(data)


ServerMessage = ResponseMessage | StateUpdate | EventNotification | ErrorMessage


def serialize_response(msg: ServerMessage) -> str:
    """Serialize a server message to a JSON string.

    Args:
        msg: Any server-to-client message model.

    Returns:
        JSON string ready for ``WebSocket.send_text()``.
    """
    return msg.model_dump_json()


def check_protocol_version(version: str) -> bool:
    """Check if a client-supplied protocol version is compatible.

    Currently performs exact-match on the major version.

    Args:
        version: Semantic version string from the client.

    Returns:
        ``True`` if the client version is compatible with the server.
    """
    try:
        client_major = int(version.split(".")[0])
        server_major = int(PROTOCOL_VERSION.split(".")[0])
    except (ValueError, IndexError):
        return False
    return client_major == server_major
