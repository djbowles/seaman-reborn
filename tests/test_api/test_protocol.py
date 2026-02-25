"""Tests for api.protocol — brain state protocol schema (US-044)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

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

# =========================================================================
# NeedsSnapshot
# =========================================================================

class TestNeedsSnapshot:
    """Tests for NeedsSnapshot model."""

    def test_defaults(self) -> None:
        snap = NeedsSnapshot()
        assert snap.hunger == 0.0
        assert snap.comfort == 1.0
        assert snap.health == 1.0
        assert snap.stimulation == 1.0

    def test_custom_values(self) -> None:
        snap = NeedsSnapshot(hunger=0.6, comfort=0.4, health=0.8, stimulation=0.2)
        assert snap.hunger == 0.6
        assert snap.comfort == 0.4

    def test_out_of_range_raises(self) -> None:
        with pytest.raises(ValidationError):
            NeedsSnapshot(hunger=1.5)
        with pytest.raises(ValidationError):
            NeedsSnapshot(health=-0.1)

    def test_serialization_roundtrip(self) -> None:
        snap = NeedsSnapshot(hunger=0.3, comfort=0.7, health=0.9, stimulation=0.5)
        data = snap.model_dump()
        restored = NeedsSnapshot.model_validate(data)
        assert restored == snap


# =========================================================================
# TankSnapshot
# =========================================================================

class TestTankSnapshot:
    """Tests for TankSnapshot model."""

    def test_defaults(self) -> None:
        snap = TankSnapshot()
        assert snap.temperature == 24.0
        assert snap.environment_type == "aquarium"

    def test_terrarium(self) -> None:
        snap = TankSnapshot(environment_type="terrarium", water_level=0.0)
        assert snap.environment_type == "terrarium"
        assert snap.water_level == 0.0

    def test_invalid_environment_type(self) -> None:
        with pytest.raises(ValidationError):
            TankSnapshot(environment_type="space_station")

    def test_cleanliness_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TankSnapshot(cleanliness=2.0)


# =========================================================================
# TraitsSnapshot
# =========================================================================

class TestTraitsSnapshot:
    """Tests for TraitsSnapshot model."""

    def test_defaults_all_half(self) -> None:
        snap = TraitsSnapshot()
        for name in (
            "cynicism", "wit", "patience", "curiosity",
            "warmth", "verbosity", "formality", "aggression",
        ):
            assert getattr(snap, name) == 0.5

    def test_custom_traits(self) -> None:
        snap = TraitsSnapshot(cynicism=0.9, wit=0.1)
        assert snap.cynicism == 0.9
        assert snap.wit == 0.1

    def test_trait_out_of_range(self) -> None:
        with pytest.raises(ValidationError):
            TraitsSnapshot(aggression=1.5)
        with pytest.raises(ValidationError):
            TraitsSnapshot(warmth=-0.2)


# =========================================================================
# CreatureStateSnapshot
# =========================================================================

class TestCreatureStateSnapshot:
    """Tests for CreatureStateSnapshot model."""

    def test_defaults(self) -> None:
        snap = CreatureStateSnapshot()
        assert snap.stage == "mushroomer"
        assert snap.mood == "neutral"
        assert snap.trust_level == 0.0

    def test_all_stages_valid(self) -> None:
        for stage in ("mushroomer", "gillman", "podfish", "tadman", "frogman"):
            snap = CreatureStateSnapshot(stage=stage)
            assert snap.stage == stage

    def test_invalid_stage_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreatureStateSnapshot(stage="goblin")

    def test_negative_age_raises(self) -> None:
        with pytest.raises(ValidationError):
            CreatureStateSnapshot(age=-1.0)

    def test_serialization_roundtrip(self) -> None:
        snap = CreatureStateSnapshot(
            stage="gillman", age=3600.0, interaction_count=42,
            mood="sardonic", trust_level=0.7,
        )
        data = snap.model_dump()
        restored = CreatureStateSnapshot.model_validate(data)
        assert restored == snap


# =========================================================================
# BrainStateSnapshot
# =========================================================================

class TestBrainStateSnapshot:
    """Tests for BrainStateSnapshot aggregate model."""

    def test_defaults(self) -> None:
        snap = BrainStateSnapshot()
        assert snap.mood == "neutral"
        assert snap.current_stage == "mushroomer"
        assert isinstance(snap.creature_state, CreatureStateSnapshot)
        assert isinstance(snap.needs, NeedsSnapshot)
        assert isinstance(snap.tank, TankSnapshot)
        assert isinstance(snap.active_traits, TraitsSnapshot)

    def test_full_snapshot(self) -> None:
        snap = BrainStateSnapshot(
            creature_state=CreatureStateSnapshot(stage="podfish", mood="amused"),
            needs=NeedsSnapshot(hunger=0.4),
            tank=TankSnapshot(temperature=22.0),
            mood="amused",
            active_traits=TraitsSnapshot(wit=0.8),
            current_stage="podfish",
        )
        assert snap.creature_state.stage == "podfish"
        assert snap.needs.hunger == 0.4
        assert snap.tank.temperature == 22.0
        assert snap.active_traits.wit == 0.8

    def test_json_roundtrip(self) -> None:
        snap = BrainStateSnapshot(
            creature_state=CreatureStateSnapshot(
                stage="tadman", age=7200.0, mood="curious",
            ),
            needs=NeedsSnapshot(hunger=0.5, health=0.8),
            tank=TankSnapshot(environment_type="terrarium", water_level=0.0),
            mood="curious",
            current_stage="tadman",
        )
        json_str = snap.model_dump_json()
        data = json.loads(json_str)
        restored = BrainStateSnapshot.model_validate(data)
        assert restored.creature_state.stage == "tadman"
        assert restored.tank.environment_type == "terrarium"


# =========================================================================
# InputMessage
# =========================================================================

class TestInputMessage:
    """Tests for client -> server InputMessage."""

    def test_valid_input(self) -> None:
        msg = InputMessage(text="Hello creature")
        assert msg.type == MessageType.INPUT
        assert msg.text == "Hello creature"
        assert msg.protocol_version == PROTOCOL_VERSION
        assert msg.timestamp  # non-empty

    def test_empty_text_raises(self) -> None:
        with pytest.raises(ValidationError):
            InputMessage(text="")

    def test_from_dict(self) -> None:
        data = {"type": "input", "text": "Hi there"}
        msg = InputMessage.model_validate(data)
        assert msg.text == "Hi there"
        assert msg.type == MessageType.INPUT


# =========================================================================
# ResponseMessage
# =========================================================================

class TestResponseMessage:
    """Tests for server -> client ResponseMessage."""

    def test_basic_response(self) -> None:
        msg = ResponseMessage(text="Go away.")
        assert msg.type == MessageType.RESPONSE
        assert msg.text == "Go away."
        assert isinstance(msg.state, BrainStateSnapshot)
        assert msg.protocol_version == PROTOCOL_VERSION

    def test_with_state(self) -> None:
        state = BrainStateSnapshot(mood="sardonic", current_stage="gillman")
        msg = ResponseMessage(text="What do you want?", state=state)
        assert msg.state.mood == "sardonic"

    def test_json_roundtrip(self) -> None:
        msg = ResponseMessage(
            text="Fine.",
            state=BrainStateSnapshot(mood="irritated"),
        )
        json_str = msg.model_dump_json()
        data = json.loads(json_str)
        restored = ResponseMessage.model_validate(data)
        assert restored.text == "Fine."
        assert restored.state.mood == "irritated"


# =========================================================================
# StateUpdate
# =========================================================================

class TestStateUpdate:
    """Tests for unsolicited StateUpdate messages."""

    def test_defaults(self) -> None:
        msg = StateUpdate()
        assert msg.type == MessageType.STATE_UPDATE
        assert msg.changed_fields == []
        assert msg.protocol_version == PROTOCOL_VERSION

    def test_with_changed_fields(self) -> None:
        msg = StateUpdate(
            state=BrainStateSnapshot(mood="hostile"),
            changed_fields=["mood", "needs.hunger"],
        )
        assert "mood" in msg.changed_fields
        assert msg.state.mood == "hostile"

    def test_serialization(self) -> None:
        msg = StateUpdate(changed_fields=["tank.temperature"])
        data = json.loads(msg.model_dump_json())
        assert data["type"] == "state_update"
        assert data["changed_fields"] == ["tank.temperature"]


# =========================================================================
# EventNotification
# =========================================================================

class TestEventNotification:
    """Tests for game event notifications."""

    def test_minimal_event(self) -> None:
        msg = EventNotification(event_type="evolution_ready")
        assert msg.type == MessageType.EVENT
        assert msg.event_type == "evolution_ready"
        assert msg.severity == EventSeverity.INFO
        assert msg.effects == {}

    def test_critical_event(self) -> None:
        msg = EventNotification(
            event_type="environmental",
            severity=EventSeverity.CRITICAL,
            message="Tank oxygen critically low!",
            effects={"health_change": -0.1},
        )
        assert msg.severity == EventSeverity.CRITICAL
        assert "oxygen" in msg.message
        assert msg.effects["health_change"] == -0.1

    def test_all_event_types(self) -> None:
        for etype in (
            "evolution_ready", "breeding", "holiday",
            "milestone", "random_observation", "environmental",
        ):
            msg = EventNotification(event_type=etype)
            assert msg.event_type == etype

    def test_json_roundtrip(self) -> None:
        msg = EventNotification(
            event_type="milestone",
            severity=EventSeverity.WARNING,
            message="100 interactions reached!",
            effects={"trust_change": 0.05},
        )
        restored = EventNotification.model_validate_json(msg.model_dump_json())
        assert restored.event_type == "milestone"
        assert restored.effects["trust_change"] == 0.05


# =========================================================================
# ErrorMessage
# =========================================================================

class TestErrorMessage:
    """Tests for error messages."""

    def test_basic_error(self) -> None:
        msg = ErrorMessage(message="Invalid JSON")
        assert msg.type == MessageType.ERROR
        assert msg.message == "Invalid JSON"

    def test_json_roundtrip(self) -> None:
        msg = ErrorMessage(message="Unknown message type: foo")
        data = json.loads(msg.model_dump_json())
        assert data["type"] == "error"
        assert data["message"] == "Unknown message type: foo"


# =========================================================================
# Serialization utilities
# =========================================================================

class TestParseClientMessage:
    """Tests for parse_client_message()."""

    def test_parse_json_string(self) -> None:
        raw = json.dumps({"type": "input", "text": "Hello"})
        msg = parse_client_message(raw)
        assert isinstance(msg, InputMessage)
        assert msg.text == "Hello"

    def test_parse_dict(self) -> None:
        msg = parse_client_message({"type": "input", "text": "Hi"})
        assert msg.text == "Hi"

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_client_message("{bad json")

    def test_missing_text_raises(self) -> None:
        with pytest.raises((ValueError, ValidationError)):
            parse_client_message({"type": "input"})

    def test_non_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="JSON object"):
            parse_client_message(json.dumps([1, 2, 3]))


class TestSerializeResponse:
    """Tests for serialize_response()."""

    def test_serialize_response_message(self) -> None:
        msg = ResponseMessage(text="Hmph.")
        result = serialize_response(msg)
        data = json.loads(result)
        assert data["type"] == "response"
        assert data["text"] == "Hmph."

    def test_serialize_state_update(self) -> None:
        msg = StateUpdate(changed_fields=["mood"])
        result = serialize_response(msg)
        data = json.loads(result)
        assert data["type"] == "state_update"

    def test_serialize_event(self) -> None:
        msg = EventNotification(event_type="holiday", message="Happy New Year!")
        result = serialize_response(msg)
        data = json.loads(result)
        assert data["type"] == "event"
        assert data["message"] == "Happy New Year!"

    def test_serialize_error(self) -> None:
        msg = ErrorMessage(message="Oops")
        result = serialize_response(msg)
        data = json.loads(result)
        assert data["type"] == "error"


# =========================================================================
# Protocol version checking
# =========================================================================

class TestCheckProtocolVersion:
    """Tests for check_protocol_version()."""

    def test_exact_match(self) -> None:
        assert check_protocol_version(PROTOCOL_VERSION) is True

    def test_compatible_minor(self) -> None:
        assert check_protocol_version("1.1.0") is True
        assert check_protocol_version("1.99.0") is True

    def test_incompatible_major(self) -> None:
        assert check_protocol_version("2.0.0") is False
        assert check_protocol_version("0.9.0") is False

    def test_garbage_input(self) -> None:
        assert check_protocol_version("not_a_version") is False
        assert check_protocol_version("") is False

    def test_partial_version(self) -> None:
        assert check_protocol_version("1") is True
        assert check_protocol_version("2") is False


# =========================================================================
# MessageType enum
# =========================================================================

class TestMessageType:
    """Tests for the MessageType discriminator."""

    def test_values(self) -> None:
        assert MessageType.INPUT == "input"
        assert MessageType.RESPONSE == "response"
        assert MessageType.STATE_UPDATE == "state_update"
        assert MessageType.EVENT == "event"
        assert MessageType.ERROR == "error"

    def test_from_string(self) -> None:
        assert MessageType("input") == MessageType.INPUT
        assert MessageType("response") == MessageType.RESPONSE


# =========================================================================
# EventSeverity enum
# =========================================================================

class TestEventSeverity:
    """Tests for EventSeverity enum."""

    def test_values(self) -> None:
        assert EventSeverity.INFO == "info"
        assert EventSeverity.WARNING == "warning"
        assert EventSeverity.CRITICAL == "critical"

    def test_from_string(self) -> None:
        assert EventSeverity("warning") == EventSeverity.WARNING


# =========================================================================
# Cross-model integration
# =========================================================================

class TestCrossModelIntegration:
    """Integration tests across multiple protocol models."""

    def test_full_response_chain(self) -> None:
        """Simulate a full input -> response flow with protocol models."""
        # Client sends input
        input_msg = InputMessage(text="How are you feeling?")
        assert input_msg.protocol_version == PROTOCOL_VERSION

        # Server builds response with full state
        state = BrainStateSnapshot(
            creature_state=CreatureStateSnapshot(
                stage="gillman", age=86400.0, interaction_count=150,
                mood="sardonic", trust_level=0.6,
                hunger=0.3, health=0.9, comfort=0.7,
            ),
            needs=NeedsSnapshot(hunger=0.3, comfort=0.7, health=0.9, stimulation=0.5),
            tank=TankSnapshot(temperature=23.5, cleanliness=0.8),
            mood="sardonic",
            active_traits=TraitsSnapshot(cynicism=0.8, wit=0.7),
            current_stage="gillman",
        )
        response = ResponseMessage(text="Oh, you actually care? Shocking.", state=state)

        # Serialize and deserialize
        wire = serialize_response(response)
        data = json.loads(wire)
        restored = ResponseMessage.model_validate(data)

        assert restored.text == "Oh, you actually care? Shocking."
        assert restored.state.creature_state.stage == "gillman"
        assert restored.state.needs.hunger == 0.3
        assert restored.state.active_traits.cynicism == 0.8

    def test_event_notification_full(self) -> None:
        """Evolution event with full effects dict."""
        evt = EventNotification(
            event_type="evolution_ready",
            severity=EventSeverity.WARNING,
            message="Your creature is ready to evolve to Podfish!",
            effects={
                "mood_change": 0.2,
                "trust_change": 0.1,
                "trigger_dialogue": True,
            },
        )
        wire = serialize_response(evt)
        data = json.loads(wire)
        assert data["event_type"] == "evolution_ready"
        assert data["severity"] == "warning"
        assert data["effects"]["trigger_dialogue"] is True

    def test_protocol_version_in_all_messages(self) -> None:
        """Every message model includes a protocol_version field."""
        models = [
            InputMessage(text="test"),
            ResponseMessage(text="reply"),
            StateUpdate(),
            EventNotification(event_type="test"),
            ErrorMessage(message="err"),
        ]
        for msg in models:
            assert msg.protocol_version == PROTOCOL_VERSION
