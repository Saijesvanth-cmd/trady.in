from datetime import datetime, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc
IST = ZoneInfo("Asia/Kolkata")  # needs the `tzdata` package on Windows


def now_utc() -> datetime:
    return datetime.now(UTC)


def iso_ist(dt: datetime) -> str:
    """Store UTC, present Asia/Kolkata."""
    if dt.tzinfo is None:
        raise ValueError("naive datetime not allowed")
    return dt.astimezone(IST).isoformat()
