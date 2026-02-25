"""Real-time clock and time awareness system.

Wraps the real-time clock with game-relevant properties: time of day, day of week,
session tracking, and absence detection. Feeds into the prompt builder and needs system.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any


class TimeOfDay(Enum):
    """Time-of-day periods matching real-world clock."""

    MORNING = "morning"       # 6:00 - 11:59
    AFTERNOON = "afternoon"   # 12:00 - 17:59
    EVENING = "evening"       # 18:00 - 21:59
    NIGHT = "night"           # 22:00 - 5:59


class AbsenceSeverity(Enum):
    """Severity level for player absence."""

    NONE = "none"           # < 24h
    MILD = "mild"           # 24h - 48h
    MODERATE = "moderate"   # 48h - 7d
    SEVERE = "severe"       # > 7d


class GameClock:
    """Real-time clock wrapper with game-relevant time properties.

    Tracks session start/end, detects player absences, and provides
    time context for prompt injection and needs calculations.

    Args:
        last_session_end: When the previous session ended (None if first session).
        now_func: Injectable clock function for testing (default: datetime.now(UTC)).
    """

    def __init__(
        self,
        last_session_end: datetime | None = None,
        now_func: Any = None,
    ) -> None:
        self._now_func = now_func or (lambda: datetime.now(UTC))
        self._session_start: datetime = self._now_func()
        self._last_session_end: datetime | None = last_session_end

    @property
    def now(self) -> datetime:
        """Current time from the clock."""
        return self._now_func()

    @property
    def session_start(self) -> datetime:
        """When the current session started."""
        return self._session_start

    @property
    def time_of_day(self) -> TimeOfDay:
        """Current time-of-day period."""
        hour = self.now.hour
        if 6 <= hour < 12:
            return TimeOfDay.MORNING
        if 12 <= hour < 18:
            return TimeOfDay.AFTERNOON
        if 18 <= hour < 22:
            return TimeOfDay.EVENING
        return TimeOfDay.NIGHT

    @property
    def day_of_week(self) -> str:
        """Current day name (e.g. 'Monday')."""
        return self.now.strftime("%A")

    @property
    def is_weekend(self) -> bool:
        """Whether the current day is Saturday or Sunday."""
        return self.now.weekday() >= 5

    @property
    def session_duration_minutes(self) -> float:
        """Minutes elapsed since session started."""
        delta = self.now - self._session_start
        return delta.total_seconds() / 60.0

    @property
    def hours_since_last_session(self) -> float | None:
        """Hours since the previous session ended.

        Returns None if this is the first session (no last_session_end).
        """
        if self._last_session_end is None:
            return None
        delta = self._session_start - self._last_session_end
        return max(0.0, delta.total_seconds() / 3600.0)

    @property
    def absence_severity(self) -> AbsenceSeverity:
        """Severity of player absence since last session."""
        hours = self.hours_since_last_session
        if hours is None or hours < 24:
            return AbsenceSeverity.NONE
        if hours < 48:
            return AbsenceSeverity.MILD
        if hours < 168:  # 7 days
            return AbsenceSeverity.MODERATE
        return AbsenceSeverity.SEVERE

    def end_session(self) -> datetime:
        """Mark the current session as ended. Returns the end timestamp."""
        return self.now

    def get_time_context(self) -> dict[str, Any]:
        """Return all time data as a dict for prompt injection.

        Returns:
            Dict with keys: time_of_day, day_of_week, is_weekend,
            hour, minute, session_duration_minutes, hours_since_last_session,
            absence_severity.
        """
        current = self.now
        return {
            "time_of_day": self.time_of_day.value,
            "day_of_week": self.day_of_week,
            "is_weekend": self.is_weekend,
            "hour": current.hour,
            "minute": current.minute,
            "session_duration_minutes": round(self.session_duration_minutes, 1),
            "hours_since_last_session": (
                round(self.hours_since_last_session, 1)
                if self.hours_since_last_session is not None
                else None
            ),
            "absence_severity": self.absence_severity.value,
        }
