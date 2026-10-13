import math

import numpy as np
import pandas as pd
import pytest

from app.indicators import core as ind

# StockCharts "RSI" worked example (14-period, Wilder smoothing)
SC_CLOSES = [44.3389, 44.0902, 44.1497, 43.6124, 44.3278, 44.8264, 45.0955, 45.4245, 45.8433,
             46.0826, 45.8931, 46.0328, 45.6140, 46.2820, 46.2820, 46.0028, 46.0328, 46.4116,
             46.2222, 45.6439, 46.2122, 46.2521, 45.7137, 46.4515, 45.7835, 45.3548, 44.0288,
             44.1783, 44.2181, 44.5672, 43.4205, 42.6628, 43.1314]


def test_rsi_matches_stockcharts_reference():
    r = ind.rsi(SC_CLOSES, 14)
    assert r.iloc[:14].isna().all()  # warm-up is NaN, not invented
    assert r.iloc[14] == pytest.approx(70.53, abs=0.01)
    assert r.iloc[15] == pytest.approx(66.32, abs=0.01)


def test_rsi_extremes():
    assert ind.rsi(range(1, 40), 14).iloc[-1] == 100
    assert ind.rsi([5.0] * 40, 14).iloc[-1] == 50


def test_sma_and_ema_by_hand():
    assert ind.sma([1, 2, 3, 4, 5], 3).tolist()[2:] == [2.0, 3.0, 4.0]
    e = ind.ema([1, 2, 3, 4, 5], 3)  # seed=SMA=2, alpha=0.5 -> 3, 4
    assert e.isna().tolist() == [True, True, False, False, False]
    assert e.tolist()[2:] == [2.0, 3.0, 4.0]


def test_atr_by_hand_and_warmup():
    a = ind.atr([10, 12, 15, 14], [8, 10, 11, 12], [9, 11, 14, 13], 3)
    assert a.iloc[:3].isna().all()
    assert a.iloc[3] == pytest.approx(3.0)  # mean(TR=3,4,2)


def test_macd_identities_and_warmup():
    close = pd.Series(np.cumsum(np.random.default_rng(1).normal(0, 1, 120)) + 100)
    m = ind.macd(close)
    assert m["macd"].first_valid_index() == 25 and m["signal"].first_valid_index() == 33
    assert (m["macd"] - m["signal"] - m["hist"]).abs().dropna().max() < 1e-12
    assert m["macd"].iloc[-1] == pytest.approx(ind.ema(close, 12).iloc[-1] - ind.ema(close, 26).iloc[-1])


def test_bollinger_constant_series_collapses():
    b = ind.bollinger([10.0] * 25)
    assert b["upper"].iloc[-1] == b["lower"].iloc[-1] == 10.0 and math.isnan(b["mid"].iloc[18])


def test_support_resistance_excludes_latest_candle():
    high = list(range(100, 125)); low = list(range(90, 115))
    assert ind.support_resistance(high, low, 20) == (94.0, 123.0)
    assert ind.support_resistance(high[:10], low[:10], 20) == (None, None)
