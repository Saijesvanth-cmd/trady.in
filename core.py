"""Indicator formulas (independent implementations, no pandas-ta).

All functions return a pandas Series aligned to the input; warm-up positions are NaN
(never invented). Conventions:
  SMA(n)    mean of last n values.
  EMA(n)    alpha = 2/(n+1); seeded with the SMA of the first n values, then
            ema[i] = ema[i-1] + alpha * (x[i] - ema[i-1]).
  Wilder    same recursion with alpha = 1/n (used for RSI and ATR).
  RSI(14)   Wilder-smoothed avg gain / avg loss of close-to-close changes;
            RSI = 100 - 100/(1+RS). avg_loss == 0 -> 100 (50 if also flat).
  MACD      EMA12 - EMA26; signal = EMA9 of the MACD line (seeded by SMA of its
            first 9 values); histogram = MACD - signal.
  Bollinger SMA(n) +/- k * population stdev (ddof=0) of the same window.
  ATR(14)   Wilder smoothing of True Range = max(H-L, |H-prevC|, |L-prevC|). TR is
            undefined for the first candle (no previous close), so the first ATR
            appears at the 15th candle (TA-Lib convention).
  ROC(n)    close / close[n bars ago] - 1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.market_data.base import Candle


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _smoothed(values, n: int, alpha: float) -> pd.Series:
    x = _arr(values)
    out = np.full(len(x), np.nan)
    valid = np.flatnonzero(~np.isnan(x))
    if len(valid):
        seed = valid[0] + n - 1
        if seed < len(x):
            out[seed] = x[valid[0] : seed + 1].mean()
            for i in range(seed + 1, len(x)):
                out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return pd.Series(out)


def sma(values, n: int) -> pd.Series:
    return pd.Series(_arr(values)).rolling(n, min_periods=n).mean()


def ema(values, n: int) -> pd.Series:
    return _smoothed(values, n, 2 / (n + 1))


def rsi(close, n: int = 14) -> pd.Series:
    c = _arr(close)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return pd.Series(out)
    d = np.diff(c)
    avg_gain = _smoothed(np.where(d > 0, d, 0.0), n, 1 / n).to_numpy()
    avg_loss = _smoothed(np.where(d < 0, -d, 0.0), n, 1 / n).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        val = 100 - 100 / (1 + avg_gain / avg_loss)
    val = np.where(avg_loss == 0, np.where(avg_gain == 0, 50.0, 100.0), val)
    out[1:] = np.where(np.isnan(avg_gain), np.nan, val)
    return pd.Series(out)


def macd(close, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line.to_numpy(), signal)
    return pd.DataFrame({"macd": line, "signal": sig, "hist": line - sig})


def bollinger(close, n: int = 20, k: float = 2.0) -> pd.DataFrame:
    s = pd.Series(_arr(close))
    mid = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return pd.DataFrame({"upper": mid + k * sd, "mid": mid, "lower": mid - k * sd})


def true_range(high, low, close) -> np.ndarray:
    h, l, c = _arr(high), _arr(low), _arr(close)
    tr = np.full(len(h), np.nan)
    if len(h) > 1:
        tr[1:] = np.maximum.reduce([h[1:] - l[1:], np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])])
    return tr


def atr(high, low, close, n: int = 14) -> pd.Series:
    return _smoothed(true_range(high, low, close), n, 1 / n)


def roc(close, n: int = 10) -> pd.Series:
    s = pd.Series(_arr(close))
    return s / s.shift(n) - 1


def support_resistance(high, low, lookback: int = 20) -> tuple[float | None, float | None]:
    """Crude estimate: lowest low / highest high of the `lookback` candles BEFORE the latest."""
    h, l = _arr(high), _arr(low)
    if len(h) < lookback + 1:
        return None, None
    return float(l[-lookback - 1 : -1].min()), float(h[-lookback - 1 : -1].max())


def classify_trend(close: float | None, sma20: float | None, sma50: float | None) -> str:
    if None in (close, sma20, sma50):
        return "unavailable"
    if close > sma20 > sma50:
        return "uptrend"
    if close < sma20 < sma50:
        return "downtrend"
    return "mixed"


def _last(series: pd.Series, digits: int = 4) -> float | None:
    if len(series) == 0 or pd.isna(series.iloc[-1]):
        return None
    return round(float(series.iloc[-1]), digits)


# candles needed before each indicator group has a value
MIN_CANDLES = {
    "rsi_14": 15, "macd": 34, "sma_20": 20, "sma_50": 50, "sma_200": 200, "ema_9": 9,
    "ema_21": 21, "bollinger_20_2": 20, "atr_14": 15, "volume_sma_20": 20,
    "momentum_10": 11, "support_resistance_20": 21,
}


def compute_snapshot(candles: list[Candle]) -> dict:
    """Latest indicator values from COMPLETED candles only. Unavailable -> None + explanation."""
    done = [c for c in candles if c.complete]
    df = pd.DataFrame(
        {"high": [c.high for c in done], "low": [c.low for c in done],
         "close": [c.close for c in done], "volume": [c.volume for c in done]}
    )
    if df.empty:
        return {"candles_used": 0, "as_of": None, "indicators": {}, "unavailable": dict.fromkeys(MIN_CANDLES, {"have": 0})}

    m, bb = macd(df.close), bollinger(df.close)
    sup, res = support_resistance(df.high, df.low)
    s20, s50 = _last(sma(df.close, 20)), _last(sma(df.close, 50))
    atr14 = _last(atr(df.high, df.low, df.close))
    close = float(df.close.iloc[-1])
    mom = _last(roc(df.close, 10) * 100)
    groups = {
        "rsi_14": {"rsi_14": _last(rsi(df.close))},
        "macd": {"macd": _last(m["macd"]), "macd_signal": _last(m["signal"]), "macd_hist": _last(m["hist"])},
        "sma_20": {"sma_20": s20},
        "sma_50": {"sma_50": s50},
        "sma_200": {"sma_200": _last(sma(df.close, 200))},
        "ema_9": {"ema_9": _last(ema(df.close, 9))},
        "ema_21": {"ema_21": _last(ema(df.close, 21))},
        "bollinger_20_2": {"bb_upper": _last(bb["upper"]), "bb_mid": _last(bb["mid"]), "bb_lower": _last(bb["lower"])},
        "atr_14": {"atr_14": atr14},
        "volume_sma_20": {"volume_sma_20": _last(sma(df.volume, 20), 0)},
        "momentum_10": {"momentum_10_pct": mom},
        "support_resistance_20": {"support_20": sup, "resistance_20": res},
    }
    indicators, unavailable = {}, {}
    for group, vals in groups.items():
        indicators.update(vals)
        if any(v is None for v in vals.values()):
            unavailable[group] = {"needed_candles": MIN_CANDLES[group], "have": len(df)}
    indicators["trend"] = classify_trend(close, s20, s50)
    indicators["atr_pct_of_close"] = round(atr14 / close * 100, 3) if atr14 is not None else None
    return {
        "candles_used": len(df),
        "as_of": done[-1].ts,
        "last_close": close,
        "indicators": indicators,
        "unavailable": unavailable,
    }
