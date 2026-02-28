"""Shared types, enums, and dataclasses for Seaman Brain."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class CreatureStage(Enum):
    """The five evolutionary stages of a Seaman creature."""

    MUSHROOMER = "mushroomer"
    GILLMAN = "gillman"
    PODFISH = "podfish"
    TADMAN = "tadman"
    FROGMAN = "frogman"


class MessageRole(Enum):
    """Role of a participant in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class MemoryRecord:
    """A single memory entry stored in the vector database."""

    text: str
    embedding: NDArray[np.float32]
    timestamp: datetime
    importance: float
    source: str
