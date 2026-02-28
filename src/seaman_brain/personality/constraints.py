"""Output filtering and personality constraints.

Strips AI assistant cliches from LLM output and enforces response length
limits based on the creature's verbosity trait. Ensures Seaman never breaks
character with phrases like "As an AI..." or "I'd be happy to help!".
"""

from __future__ import annotations

import re

from seaman_brain.personality.traits import TraitProfile

# AI assistant cliches that must be stripped from creature output.
# These are compiled as case-insensitive regex patterns for flexible matching.
FORBIDDEN_PHRASES: list[str] = [
    r"as an ai\b",
    r"as a language model\b",
    r"as an artificial intelligence\b",
    r"i would be happy to\b",
    r"i'd be happy to\b",
    r"i'm happy to\b",
    r"i am happy to\b",
    r"i cannot help but\b",
    r"i cannot assist\b",
    r"i'm here to help\b",
    r"i am here to help\b",
    r"how can i assist you\b",
    r"how may i help you\b",
    r"is there anything else\b",
    r"feel free to ask\b",
    r"don't hesitate to\b",
    r"do not hesitate to\b",
    r"i hope this helps\b",
    r"i hope that helps\b",
    r"let me know if you need\b",
    r"i'd love to help\b",
    r"great question\b",
    r"that's a great question\b",
    r"excellent question\b",
    r"wonderful question\b",
    r"thank you for asking\b",
    r"thanks for asking\b",
    r"i appreciate your\b",
    r"absolutely! i\b",
    r"certainly! i\b",
    r"of course! i\b",
    r"sure thing!\b",
]

# Pre-compiled patterns for performance.
_FORBIDDEN_PATTERNS: list[re.Pattern[str]] = [
    re.compile(phrase, re.IGNORECASE) for phrase in FORBIDDEN_PHRASES
]

# Response length bounds (in characters).
_MIN_LENGTH = 10
_MAX_LENGTH = 500
# Verbosity 0.0 -> _MIN_LENGTH, verbosity 1.0 -> _MAX_LENGTH
# Linear interpolation between the two.


def _strip_forbidden(text: str) -> str:
    """Remove all forbidden AI-assistant phrases from text."""
    for pattern in _FORBIDDEN_PATTERNS:
        text = pattern.sub("", text)
    return text


def _clean_whitespace(text: str) -> str:
    """Normalize whitespace: collapse runs and strip leading/trailing."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+", "", text, count=1)
    text = re.sub(r"\s+$", "", text, count=1)
    # Strip leading punctuation/space left after phrase removal (e.g. ", I think...")
    text = re.sub(r"^[,;:\s]+", "", text)
    return text


def _max_length_for_verbosity(verbosity: float) -> int:
    """Calculate max response length from verbosity trait (0.0-1.0)."""
    v = max(0.0, min(1.0, verbosity))
    return int(_MIN_LENGTH + (_MAX_LENGTH - _MIN_LENGTH) * v)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len, breaking at the last sentence or word boundary."""
    if len(text) <= max_len:
        return text

    truncated = text[:max_len]

    # Try to break at last sentence boundary.
    last_sentence = max(
        truncated.rfind(". "),
        truncated.rfind("! "),
        truncated.rfind("? "),
        truncated.rfind(".\n"),
        truncated.rfind("!\n"),
        truncated.rfind("?\n"),
    )
    if last_sentence > max_len // 3:
        return truncated[: last_sentence + 1]

    # Try to break at last word boundary.
    last_space = truncated.rfind(" ")
    if last_space > max_len // 3:
        return truncated[:last_space]

    # Hard truncate as last resort.
    return truncated


def _strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks emitted by Qwen3 and similar models."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)


def apply_constraints(response: str, profile: TraitProfile) -> str:
    """Filter LLM output through personality constraints.

    1. Strips ``<think>`` reasoning blocks from Qwen3-style models.
    2. Strips forbidden AI-assistant phrases.
    3. Cleans up leftover whitespace artifacts.
    4. Enforces max response length based on verbosity trait.
    5. Returns empty string for empty/whitespace-only input.

    Args:
        response: Raw LLM response text.
        profile: Current creature's trait profile (uses verbosity for length).

    Returns:
        Filtered, length-constrained response string.
    """
    if not response or not response.strip():
        return ""

    text = _strip_think_blocks(response)
    text = text.replace("*", "")
    text = _strip_forbidden(text)
    text = _clean_whitespace(text)

    if not text:
        return ""

    max_len = _max_length_for_verbosity(profile.verbosity)
    text = _truncate(text, max_len)

    return text
