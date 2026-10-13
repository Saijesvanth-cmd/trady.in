from __future__ import annotations

from datetime import date
from functools import lru_cache

from app.core.config import get_settings
from app.core.errors import ProviderConfigError, UnsupportedInterval
from app.market_data.base import Candle, Instrument, MarketDataProvider
from app.market_data.demo import DemoProvider
from app.market_data.validation import validate_candles


@lru_cache
def get_provider() -> MarketDataProvider:
    name = get_settings().market_data_provider.lower()
    if name == "demo":
        return DemoProvider()
    # Fail loudly: never fall back to demo data when a real provider was requested.
    raise ProviderConfigError(
        f"MARKET_DATA_PROVIDER='{name}' is not implemented yet. Use 'demo' for now."
    )


def get_validated_candles(
    provider: MarketDataProvider, instrument: Instrument, interval: str, start: date, end: date
) -> list[Candle]:
    if interval not in provider.supported_intervals:
        raise UnsupportedInterval(
            f"Interval '{interval}' not available from '{provider.name}'. "
            f"Supported: {', '.join(provider.supported_intervals)}."
        )
    raw = provider.get_historical_candles(instrument.instrument_id, interval, start, end)
    return validate_candles(raw)
