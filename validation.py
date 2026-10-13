import math

from app.core.errors import CandleValidationError
from app.market_data.base import Candle


def validate_candles(candles: list[Candle]) -> list[Candle]:
    """Sort ascending, then reject (never repair) anything impossible.

    Rejects: naive timestamps, duplicate timestamps, non-finite or non-positive
    prices, high < max(open, close, low), low > min(open, close, high), negative volume.
    """
    problems: list[str] = []
    if any(c.ts.tzinfo is None for c in candles):
        raise CandleValidationError("Rejected candles: naive (timezone-less) timestamp present.")
    ordered = sorted(candles, key=lambda c: c.ts)
    prev_ts = None
    for i, c in enumerate(ordered):
        if prev_ts is not None and c.ts == prev_ts:
            problems.append(f"#{i}: duplicate timestamp {c.ts.isoformat()}")
        prev_ts = c.ts
        px = (c.open, c.high, c.low, c.close)
        if not all(isinstance(p, (int, float)) and math.isfinite(p) and p > 0 for p in px):
            problems.append(f"#{i}: non-finite or non-positive price")
            continue
        if c.high < max(c.open, c.close, c.low):
            problems.append(f"#{i}: high below open/close/low")
        if c.low > min(c.open, c.close, c.high):
            problems.append(f"#{i}: low above open/close/high")
        if c.volume < 0:
            problems.append(f"#{i}: negative volume")
    if problems:
        shown = "; ".join(problems[:5]) + (f" (+{len(problems) - 5} more)" if len(problems) > 5 else "")
        raise CandleValidationError(f"Rejected {len(problems)} bad candle(s): {shown}")
    return ordered
