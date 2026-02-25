"""Tests for the real-time clock and time awareness system."""

from __future__ import annotations

from datetime import UTC, datetime

from seaman_brain.environment.clock import AbsenceSeverity, GameClock, TimeOfDay

# --- Happy path tests ---


class TestTimeOfDay:
    """Test time-of-day detection."""

    def test_morning(self) -> None:
        morning = datetime(2026, 2, 25, 8, 30, tzinfo=UTC)
        clock = GameClock(now_func=lambda: morning)
        assert clock.time_of_day == TimeOfDay.MORNING

    def test_afternoon(self) -> None:
        afternoon = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: afternoon)
        assert clock.time_of_day == TimeOfDay.AFTERNOON

    def test_evening(self) -> None:
        evening = datetime(2026, 2, 25, 19, 45, tzinfo=UTC)
        clock = GameClock(now_func=lambda: evening)
        assert clock.time_of_day == TimeOfDay.EVENING

    def test_night_late(self) -> None:
        night = datetime(2026, 2, 25, 23, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: night)
        assert clock.time_of_day == TimeOfDay.NIGHT

    def test_night_early(self) -> None:
        early = datetime(2026, 2, 25, 3, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: early)
        assert clock.time_of_day == TimeOfDay.NIGHT


class TestDayOfWeek:
    """Test day-of-week and weekend detection."""

    def test_day_of_week_name(self) -> None:
        # 2026-02-25 is a Wednesday
        wed = datetime(2026, 2, 25, 12, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: wed)
        assert clock.day_of_week == "Wednesday"

    def test_is_weekend_saturday(self) -> None:
        sat = datetime(2026, 2, 28, 12, 0, tzinfo=UTC)  # Saturday
        clock = GameClock(now_func=lambda: sat)
        assert clock.is_weekend is True

    def test_is_weekend_sunday(self) -> None:
        sun = datetime(2026, 3, 1, 12, 0, tzinfo=UTC)  # Sunday
        clock = GameClock(now_func=lambda: sun)
        assert clock.is_weekend is True

    def test_not_weekend_weekday(self) -> None:
        mon = datetime(2026, 2, 23, 12, 0, tzinfo=UTC)  # Monday
        clock = GameClock(now_func=lambda: mon)
        assert clock.is_weekend is False


class TestSessionTracking:
    """Test session duration tracking."""

    def test_session_duration_minutes(self) -> None:
        start = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        later = datetime(2026, 2, 25, 10, 45, tzinfo=UTC)
        times = iter([start, later])
        clock = GameClock(now_func=lambda: next(times))
        assert clock.session_duration_minutes == 45.0

    def test_session_start_recorded(self) -> None:
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: now)
        assert clock.session_start == now

    def test_end_session_returns_timestamp(self) -> None:
        now = datetime(2026, 2, 25, 15, 30, tzinfo=UTC)
        clock = GameClock(now_func=lambda: now)
        end = clock.end_session()
        assert end == now


# --- Absence detection tests ---


class TestAbsenceDetection:
    """Test hours_since_last_session and absence severity."""

    def test_hours_since_last_session_normal(self) -> None:
        last_end = datetime(2026, 2, 24, 22, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.hours_since_last_session == 12.0

    def test_hours_since_last_session_first_session(self) -> None:
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=None, now_func=lambda: now)
        assert clock.hours_since_last_session is None

    def test_absence_severity_none(self) -> None:
        last_end = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 14, 0, tzinfo=UTC)  # 6h
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.NONE

    def test_absence_severity_mild(self) -> None:
        last_end = datetime(2026, 2, 23, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 24, 14, 0, tzinfo=UTC)  # 28h
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.MILD

    def test_absence_severity_moderate(self) -> None:
        last_end = datetime(2026, 2, 20, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 23, 10, 0, tzinfo=UTC)  # 72h
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.MODERATE

    def test_absence_severity_severe(self) -> None:
        last_end = datetime(2026, 2, 10, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)  # 15 days = 360h
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.SEVERE

    def test_absence_severity_first_session(self) -> None:
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=None, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.NONE


# --- get_time_context tests ---


class TestGetTimeContext:
    """Test the time context dict for prompt injection."""

    def test_context_dict_keys(self) -> None:
        now = datetime(2026, 2, 25, 14, 30, tzinfo=UTC)
        clock = GameClock(now_func=lambda: now)
        ctx = clock.get_time_context()
        expected_keys = {
            "time_of_day", "day_of_week", "is_weekend", "hour", "minute",
            "session_duration_minutes", "hours_since_last_session",
            "absence_severity",
        }
        assert set(ctx.keys()) == expected_keys

    def test_context_values_afternoon(self) -> None:
        now = datetime(2026, 2, 25, 14, 30, tzinfo=UTC)
        clock = GameClock(now_func=lambda: now)
        ctx = clock.get_time_context()
        assert ctx["time_of_day"] == "afternoon"
        assert ctx["day_of_week"] == "Wednesday"
        assert ctx["is_weekend"] is False
        assert ctx["hour"] == 14
        assert ctx["minute"] == 30

    def test_context_with_absence(self) -> None:
        last_end = datetime(2026, 2, 23, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)  # 48h
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        ctx = clock.get_time_context()
        assert ctx["hours_since_last_session"] == 48.0
        assert ctx["absence_severity"] == "moderate"

    def test_context_first_session(self) -> None:
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=None, now_func=lambda: now)
        ctx = clock.get_time_context()
        assert ctx["hours_since_last_session"] is None
        assert ctx["absence_severity"] == "none"

    def test_context_session_duration_rounded(self) -> None:
        start = datetime(2026, 2, 25, 10, 0, 0, tzinfo=UTC)
        later = datetime(2026, 2, 25, 10, 15, 30, tzinfo=UTC)
        # First call is __init__ (session_start), subsequent calls return later
        calls = [0]

        def mock_now() -> datetime:
            calls[0] += 1
            return start if calls[0] == 1 else later

        clock = GameClock(now_func=mock_now)
        ctx = clock.get_time_context()
        assert ctx["session_duration_minutes"] == 15.5


# --- Edge case tests ---


class TestEdgeCases:
    """Test boundary conditions and edge cases."""

    def test_time_of_day_boundary_6am(self) -> None:
        boundary = datetime(2026, 2, 25, 6, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: boundary)
        assert clock.time_of_day == TimeOfDay.MORNING

    def test_time_of_day_boundary_noon(self) -> None:
        boundary = datetime(2026, 2, 25, 12, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: boundary)
        assert clock.time_of_day == TimeOfDay.AFTERNOON

    def test_time_of_day_boundary_6pm(self) -> None:
        boundary = datetime(2026, 2, 25, 18, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: boundary)
        assert clock.time_of_day == TimeOfDay.EVENING

    def test_time_of_day_boundary_10pm(self) -> None:
        boundary = datetime(2026, 2, 25, 22, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: boundary)
        assert clock.time_of_day == TimeOfDay.NIGHT

    def test_time_of_day_boundary_midnight(self) -> None:
        midnight = datetime(2026, 2, 25, 0, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: midnight)
        assert clock.time_of_day == TimeOfDay.NIGHT

    def test_time_of_day_boundary_559am(self) -> None:
        pre_morning = datetime(2026, 2, 25, 5, 59, tzinfo=UTC)
        clock = GameClock(now_func=lambda: pre_morning)
        assert clock.time_of_day == TimeOfDay.NIGHT

    def test_absence_boundary_exactly_24h(self) -> None:
        last_end = datetime(2026, 2, 24, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.MILD

    def test_absence_boundary_exactly_48h(self) -> None:
        last_end = datetime(2026, 2, 23, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.MODERATE

    def test_absence_boundary_exactly_7d(self) -> None:
        last_end = datetime(2026, 2, 18, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.absence_severity == AbsenceSeverity.SEVERE

    def test_hours_since_last_session_negative_clamped(self) -> None:
        """If last_session_end is in the future (clock skew), clamp to 0."""
        last_end = datetime(2026, 2, 26, 10, 0, tzinfo=UTC)
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(last_session_end=last_end, now_func=lambda: now)
        assert clock.hours_since_last_session == 0.0

    def test_session_duration_zero_at_start(self) -> None:
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: now)
        assert clock.session_duration_minutes == 0.0

    def test_now_property(self) -> None:
        now = datetime(2026, 2, 25, 10, 0, tzinfo=UTC)
        clock = GameClock(now_func=lambda: now)
        assert clock.now == now


class TestEnumValues:
    """Test enum string values."""

    def test_time_of_day_values(self) -> None:
        assert TimeOfDay.MORNING.value == "morning"
        assert TimeOfDay.AFTERNOON.value == "afternoon"
        assert TimeOfDay.EVENING.value == "evening"
        assert TimeOfDay.NIGHT.value == "night"

    def test_absence_severity_values(self) -> None:
        assert AbsenceSeverity.NONE.value == "none"
        assert AbsenceSeverity.MILD.value == "mild"
        assert AbsenceSeverity.MODERATE.value == "moderate"
        assert AbsenceSeverity.SEVERE.value == "severe"


class TestDefaultClock:
    """Test clock with default (real) now_func."""

    def test_default_clock_creates_session(self) -> None:
        clock = GameClock()
        assert clock.session_start is not None
        assert isinstance(clock.now, datetime)

    def test_default_clock_time_of_day_is_valid(self) -> None:
        clock = GameClock()
        assert clock.time_of_day in list(TimeOfDay)

    def test_default_clock_day_of_week_is_string(self) -> None:
        clock = GameClock()
        assert isinstance(clock.day_of_week, str)
        assert len(clock.day_of_week) > 0
