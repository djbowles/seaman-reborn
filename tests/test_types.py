"""Tests for shared types, enums, and dataclasses."""

from datetime import UTC, datetime

import numpy as np

from seaman_brain.types import (
    ChatMessage,
    CreatureStage,
    MemoryRecord,
    MessageRole,
)


class TestCreatureStage:
    """Tests for the CreatureStage enum."""

    def test_has_five_stages(self):
        assert len(CreatureStage) == 5

    def test_stage_values(self):
        assert CreatureStage.MUSHROOMER.value == "mushroomer"
        assert CreatureStage.GILLMAN.value == "gillman"
        assert CreatureStage.PODFISH.value == "podfish"
        assert CreatureStage.TADMAN.value == "tadman"
        assert CreatureStage.FROGMAN.value == "frogman"

    def test_stage_from_value(self):
        assert CreatureStage("mushroomer") is CreatureStage.MUSHROOMER

    def test_invalid_stage_raises(self):
        import pytest

        with pytest.raises(ValueError):
            CreatureStage("invalid")


class TestMessageRole:
    """Tests for the MessageRole enum."""

    def test_has_three_roles(self):
        assert len(MessageRole) == 3

    def test_role_values(self):
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


class TestChatMessage:
    """Tests for the ChatMessage dataclass."""

    def test_create_message(self):
        msg = ChatMessage(role=MessageRole.USER, content="hello")
        assert msg.role is MessageRole.USER
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, datetime)

    def test_timestamp_default_is_utc(self):
        msg = ChatMessage(role=MessageRole.SYSTEM, content="test")
        assert msg.timestamp.tzinfo is UTC

    def test_custom_timestamp(self):
        ts = datetime(2025, 1, 1, tzinfo=UTC)
        msg = ChatMessage(role=MessageRole.ASSISTANT, content="hi", timestamp=ts)
        assert msg.timestamp == ts

    def test_empty_content(self):
        msg = ChatMessage(role=MessageRole.USER, content="")
        assert msg.content == ""


class TestMemoryRecord:
    """Tests for the MemoryRecord dataclass."""

    def test_create_record(self):
        emb = np.zeros(384, dtype=np.float32)
        ts = datetime.now(UTC)
        rec = MemoryRecord(
            text="something happened",
            embedding=emb,
            timestamp=ts,
            importance=0.8,
            source="conversation",
        )
        assert rec.text == "something happened"
        assert rec.embedding.shape == (384,)
        assert rec.importance == 0.8
        assert rec.source == "conversation"

    def test_importance_boundary_values(self):
        emb = np.zeros(1, dtype=np.float32)
        ts = datetime.now(UTC)
        rec_low = MemoryRecord(text="x", embedding=emb, timestamp=ts, importance=0.0, source="s")
        rec_high = MemoryRecord(text="x", embedding=emb, timestamp=ts, importance=1.0, source="s")
        assert rec_low.importance == 0.0
        assert rec_high.importance == 1.0

    def test_embedding_dtype(self):
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        ts = datetime.now(UTC)
        rec = MemoryRecord(text="t", embedding=emb, timestamp=ts, importance=0.5, source="s")
        assert rec.embedding.dtype == np.float32
