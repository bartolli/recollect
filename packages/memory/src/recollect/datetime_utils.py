"""Datetime utilities for Memory SDK.

Provides timezone-aware datetime handling, parsing, and conversion
for consistent memory timestamp management.
"""

from datetime import UTC, timedelta
from datetime import datetime as dt
from zoneinfo import ZoneInfo

from pydantic import BaseModel


class TimestampInfo(BaseModel):
    """Standardized timestamp information."""

    datetime: dt
    timezone_name: str
    is_timezone_aware: bool
    utc_datetime: dt
    age_seconds: float
    human_readable_age: str


class DateParseResult(BaseModel):
    """Result of date parsing operation."""

    original_input: str
    parsed_datetime: dt
    format_used: str
    timezone_applied: str | None = None


def get_default_timezone() -> ZoneInfo:
    """Get the default timezone for the application."""
    try:
        tzinfo = dt.now().astimezone(tz=None).tzinfo
        if tzinfo is not None:
            return ZoneInfo(str(tzinfo))
        return ZoneInfo("UTC")
    except (ValueError, KeyError, OSError):
        return ZoneInfo("UTC")


def get_timezone(timezone_name: str) -> ZoneInfo:
    """Get timezone by name with validation."""
    try:
        return ZoneInfo(timezone_name)
    except (ValueError, KeyError, OSError) as e:
        raise ValueError(f"Invalid timezone '{timezone_name}': {e}") from e


def now_utc() -> dt:
    """Get current UTC datetime (timezone-aware)."""
    return dt.now(UTC)


def now_local() -> dt:
    """Get current local datetime (timezone-aware)."""
    return dt.now(get_default_timezone())


def normalize_to_utc(dt_value: dt) -> dt:
    """Normalize datetime to UTC timezone-aware format.

    Args:
        dt_value: Input datetime (timezone-aware or naive)

    Returns:
        UTC timezone-aware datetime
    """
    if dt_value.tzinfo is None:
        local_tz = get_default_timezone()
        dt_value = dt_value.replace(tzinfo=local_tz)
    return dt_value.astimezone(UTC)


def normalize_to_naive_utc(dt_value: dt) -> dt:
    """Normalize datetime to timezone-naive UTC format.

    Used for comparison operations where timezone-naive UTC is preferred.
    """
    utc_dt = normalize_to_utc(dt_value)
    return utc_dt.replace(tzinfo=None)


def parse_flexible_datetime(
    date_str: str, default_timezone: str | None = None
) -> DateParseResult:
    """Parse datetime string in various formats.

    Supports ISO 8601, YYYY-MM-DD, Month DD YYYY, MM/DD/YYYY,
    DD/MM/YYYY, HH:MM, and more.
    """
    datetime_formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%B %d, %Y",
        "%b %d, %Y",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%H:%M:%S",
        "%H:%M",
        "%I:%M %p",
        "%I:%M:%S %p",
    ]

    time_only_formats = {"%H:%M:%S", "%H:%M", "%I:%M %p", "%I:%M:%S %p"}

    parsed_datetime = None
    format_used = None

    for fmt in datetime_formats:
        try:
            parsed_datetime = dt.strptime(date_str, fmt)
            format_used = fmt
            break
        except ValueError:
            continue

    if parsed_datetime is None:
        raise ValueError(
            f"Could not parse datetime '{date_str}'. "
            f"Supported formats: ISO 8601, YYYY-MM-DD, Month DD YYYY, "
            f"MM/DD/YYYY, DD/MM/YYYY, HH:MM, etc."
        )

    if format_used in time_only_formats:
        today = now_utc().date()
        parsed_datetime = dt.combine(today, parsed_datetime.time())

    timezone_applied = None
    if parsed_datetime.tzinfo is None and default_timezone:
        tz = get_timezone(default_timezone)
        parsed_datetime = parsed_datetime.replace(tzinfo=tz)
        timezone_applied = default_timezone
    elif parsed_datetime.tzinfo is None:
        tz = get_default_timezone()
        parsed_datetime = parsed_datetime.replace(tzinfo=tz)
        timezone_applied = str(tz)

    return DateParseResult(
        original_input=date_str,
        parsed_datetime=parsed_datetime,
        format_used=format_used or "",
        timezone_applied=timezone_applied,
    )


def get_timestamp_info(dt_value: dt, reference_time: dt | None = None) -> TimestampInfo:
    """Get comprehensive information about a timestamp."""
    if reference_time is None:
        reference_time = now_utc()

    dt_utc = normalize_to_utc(dt_value)
    ref_utc = normalize_to_utc(reference_time)
    age_delta = ref_utc - dt_utc

    return TimestampInfo(
        datetime=dt_value,
        timezone_name=str(dt_value.tzinfo) if dt_value.tzinfo else "naive",
        is_timezone_aware=dt_value.tzinfo is not None,
        utc_datetime=dt_utc,
        age_seconds=age_delta.total_seconds(),
        human_readable_age=format_time_delta(age_delta),
    )


def format_time_delta(delta: timedelta) -> str:
    """Format timedelta as human-readable string (e.g., '2h 30m ago')."""
    total_seconds = int(abs(delta.total_seconds()))

    if total_seconds == 0:
        return "now"

    days = total_seconds // 86400
    hours = (total_seconds % 86400) // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 and days == 0:
        parts.append(f"{minutes}m")
    if seconds > 0 and days == 0 and hours == 0:
        parts.append(f"{seconds}s")

    result = " ".join(parts) if parts else "0s"
    return f"in {result}" if delta.total_seconds() < 0 else f"{result} ago"


def is_older_than(
    dt_value: dt, threshold: timedelta, reference_time: dt | None = None
) -> bool:
    """Check if datetime is older than a threshold."""
    if reference_time is None:
        reference_time = now_utc()

    dt_utc = normalize_to_utc(dt_value)
    ref_utc = normalize_to_utc(reference_time)
    return (ref_utc - dt_utc) > threshold


def add_timezone_to_naive(dt_value: dt, timezone_name: str) -> dt:
    """Add timezone information to a naive datetime."""
    if dt_value.tzinfo is not None:
        raise ValueError("Datetime is already timezone-aware")
    return dt_value.replace(tzinfo=get_timezone(timezone_name))


def convert_timezone(dt_value: dt, target_timezone: str) -> dt:
    """Convert datetime to different timezone."""
    if dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=get_default_timezone())
    return dt_value.astimezone(get_timezone(target_timezone))


def memory_timestamp_for_storage(dt_value: dt | None = None) -> dt:
    """Create standardized UTC timestamp for memory storage.

    All memory timestamps are stored as timezone-aware UTC.
    """
    if dt_value is None:
        return now_utc()
    return normalize_to_utc(dt_value)


def memory_timestamp_for_comparison(dt_value: dt) -> dt:
    """Create standardized timestamp for comparison operations.

    Returns timezone-naive UTC to avoid comparison issues.
    """
    return normalize_to_naive_utc(dt_value)


# Convenience constants
UTC_TZ = UTC
DEFAULT_TZ = get_default_timezone()
