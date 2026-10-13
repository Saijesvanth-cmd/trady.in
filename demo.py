"""Demo provider: deterministic SIMULATED data. Prices here are NOT real.

Anchor prices are arbitrary illustrative starting points for a seeded random walk.
Quotes are stamped with the last simulated session close, never with 'now', so
nothing here can pass as a live price.
"""
from __future__ import annotations

import zlib
from datetime import date, datetime, time, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from app.core.errors import InstrumentNotFound, UnsupportedInterval
from app.core.timeutil import IST, UTC, now_utc
from app.market_data.base import Candle, DataMode, Instrument, MarketDataProvider, Quote

ANCHOR_DATE = date(2024, 1, 1)
# symbol: (name, kind, anchor price, base daily volume)
UNIVERSE = {
    "RELIANCE": ("Reliance Industries Ltd", "equity", 2900.0, 6_000_000),
    "TCS": ("Tata Consultancy Services Ltd", "equity", 3900.0, 2_000_000),
    "HDFCBANK": ("HDFC Bank Ltd", "equity", 1650.0, 12_000_000),
    "INFY": ("Infosys Ltd", "equity", 1500.0, 7_000_000),
    "ICICIBANK": ("ICICI Bank Ltd", "equity", 1050.0, 14_000_000),
    "SBIN": ("State Bank of India", "equity", 700.0, 15_000_000),
    "NIFTY 50": ("NIFTY 50 Index", "index", 22000.0, 0),
    "SENSEX": ("S&P BSE SENSEX Index", "index", 72000.0, 0),
}
CLOSE_TIME = time(15, 30)
OPEN_TIME = time(9, 15)


def _tick(x: np.ndarray) -> np.ndarray:
    return np.round(np.round(x / 0.05) * 0.05, 2)


def _last_completed_session(now: datetime) -> date:
    local = now.astimezone(IST)
    d = local.date()
    if local.weekday() >= 5 or local.time() < CLOSE_TIME:
        d -= timedelta(days=1)
    while d.weekday() >= 5:  # holidays are NOT modelled in demo mode
        d -= timedelta(days=1)
    return d


@lru_cache(maxsize=64)
def _series(symbol: str, last_session: date) -> tuple[Candle, ...]:
    _, kind, anchor, base_vol = UNIVERSE[symbol]
    days = pd.bdate_range(ANCHOR_DATE, last_session)
    n = len(days)
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))  # stable across runs
    close = anchor * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    prev = np.concatenate([[anchor], close[:-1]])
    open_ = prev * (1 + rng.normal(0, 0.003, n))
    span = np.abs(rng.normal(0, 0.01, n))
    high = np.maximum(open_, close) * (1 + span * rng.uniform(0.2, 1.0, n))
    low = np.minimum(open_, close) * (1 - span * rng.uniform(0.2, 1.0, n))
    o, h, l, c = _tick(open_), _tick(high), _tick(low), _tick(close)  # rounding is monotonic
    vol = (rng.lognormal(np.log(base_vol), 0.35, n).astype(np.int64) if base_vol else np.zeros(n, np.int64))
    return tuple(
        Candle(
            ts=datetime.combine(d.date(), OPEN_TIME, tzinfo=IST).astimezone(UTC),
            open=float(o[i]), high=float(h[i]), low=float(l[i]), close=float(c[i]),
            volume=int(vol[i]), complete=True,
        )
        for i, d in enumerate(days)
    )


class DemoProvider(MarketDataProvider):
    name = "demo"
    supported_intervals = ("1d",)
    data_mode = DataMode.SIMULATED

    def _instrument(self, symbol: str) -> Instrument:
        name, kind, _, _ = UNIVERSE[symbol]
        return Instrument(symbol, name, "NSE", f"DEMO|{symbol}", kind)

    def search_instruments(self, query: str, exchange: str = "NSE") -> list[Instrument]:
        if exchange.upper() != "NSE":
            return []
        q = query.strip().lower()
        return [
            self._instrument(s) for s, (name, *_rest) in UNIVERSE.items()
            if not q or q in s.lower() or q in name.lower()
        ]

    def _symbol_of(self, instrument_id: str) -> str:
        symbol = instrument_id.removeprefix("DEMO|")
        if symbol not in UNIVERSE:
            raise InstrumentNotFound(f"Unknown demo instrument '{instrument_id}'.")
        return symbol

    def get_quote(self, instrument_id: str) -> Quote:
        symbol = self._symbol_of(instrument_id)
        series = _series(symbol, _last_completed_session(now_utc()))
        last, prev = series[-1], series[-2]
        return Quote(
            symbol=symbol, exchange="NSE", ltp=last.close, open=last.open, high=last.high,
            low=last.low, close=last.close, prev_close=prev.close, volume=last.volume,
            ts=datetime.combine(last.ts.astimezone(IST).date(), CLOSE_TIME, tzinfo=IST).astimezone(UTC),
            mode=DataMode.SIMULATED, source="demo",
        )

    def get_historical_candles(self, instrument_id: str, interval: str, start: date, end: date) -> list[Candle]:
        if interval not in self.supported_intervals:
            raise UnsupportedInterval(
                f"Interval '{interval}' not supported by '{self.name}'. Supported: {', '.join(self.supported_intervals)}."
            )
        symbol = self._symbol_of(instrument_id)
        series = _series(symbol, _last_completed_session(now_utc()))
        return [c for c in series if start <= c.ts.astimezone(IST).date() <= end]

    def get_market_status(self) -> dict:
        local = now_utc().astimezone(IST)
        in_hours = local.weekday() < 5 and OPEN_TIME <= local.time() < CLOSE_TIME
        return {
            "session": "regular_hours" if in_hours else "closed",
            "as_of": local.isoformat(),
            "holiday_calendar_applied": False,
            "data_mode": self.data_mode.value,
        }

    def get_provider_status(self) -> dict:
        return {
            "provider": self.name, "connected": True, "data_mode": self.data_mode.value,
            "detail": "Demo mode: all prices are SIMULATED and not real market data.",
            "supported_intervals": list(self.supported_intervals),
        }
