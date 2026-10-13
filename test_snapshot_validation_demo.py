from datetime import date, datetime, timedelta, timezone

import pytest

from app.core.errors import (CandleValidationError, InstrumentNotFound, UnsupportedInterval)
from app.indicators.core import compute_snapshot
from app.market_data.base import Candle, DataMode
from app.market_data.demo import DemoProvider
from app.market_data.validation import validate_candles

T0 = datetime(2025, 1, 1, 3, 45, tzinfo=timezone.utc)


def c(i, o=100, h=105, l=95, cl=102, v=1000, complete=True, ts=None):
    return Candle(ts or T0 + timedelta(days=i), o, h, l, cl, v, complete)


# ---- validation
def test_validation_sorts_and_accepts_good_data():
    out = validate_candles([c(2), c(0), c(1)])
    assert [x.ts for x in out] == sorted(x.ts for x in out)


@pytest.mark.parametrize("bad", [
    c(0, h=99),                      # high below open/close
    c(0, l=103),                     # low above close
    c(0, o=float("nan")),            # NaN price
    c(0, cl=-1),                     # non-positive
    c(0, v=-5),                      # negative volume
    c(0, ts=datetime(2025, 1, 1)),   # naive timestamp
])
def test_validation_rejects_impossible_candles(bad):
    with pytest.raises(CandleValidationError):
        validate_candles([c(1), bad])


def test_validation_rejects_duplicate_timestamps():
    with pytest.raises(CandleValidationError, match="duplicate"):
        validate_candles([c(0), c(0)])


# ---- snapshot
def test_snapshot_reports_unavailable_instead_of_inventing():
    snap = compute_snapshot([c(i) for i in range(30)])
    assert snap["indicators"]["sma_20"] == 102.0
    assert snap["indicators"]["sma_200"] is None and "sma_200" in snap["unavailable"]
    assert snap["unavailable"]["sma_200"] == {"needed_candles": 200, "have": 30}


def test_snapshot_ignores_incomplete_candles():
    snap = compute_snapshot([c(i) for i in range(25)] + [c(25, cl=104, complete=False)])
    assert snap["candles_used"] == 25 and snap["last_close"] == 102


def test_snapshot_empty():
    assert compute_snapshot([])["candles_used"] == 0


# ---- demo provider
def test_demo_is_deterministic_valid_and_labelled():
    p = DemoProvider()
    ins = p.resolve_symbol("reliance")
    a = p.get_historical_candles(ins.instrument_id, "1d", date(2024, 1, 1), date(2025, 6, 30))
    b = p.get_historical_candles(ins.instrument_id, "1d", date(2024, 1, 1), date(2025, 6, 30))
    assert a == b and len(a) > 300
    validate_candles(a)  # simulated data must pass the same gate as real data
    q = p.get_quote(ins.instrument_id)
    assert q.mode is DataMode.SIMULATED and q.ts < datetime.now(timezone.utc)


def test_demo_errors():
    p = DemoProvider()
    with pytest.raises(InstrumentNotFound):
        p.resolve_symbol("NOTAREALSTOCK")
    with pytest.raises(UnsupportedInterval):
        p.get_historical_candles("DEMO|TCS", "5m", date(2025, 1, 1), date(2025, 2, 1))
    assert p.search_instruments("tcs", "BSE") == []
