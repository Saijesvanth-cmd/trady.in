from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum

from app.core.errors import InstrumentNotFound


class DataMode(str, Enum):
    LIVE = "LIVE"
    DELAYED = "DELAYED"
    HISTORICAL = "HISTORICAL"
    SIMULATED = "SIMULATED"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    name: str
    exchange: str  # NSE | BSE
    instrument_id: str  # provider-specific key
    kind: str = "equity"  # equity | index


@dataclass(frozen=True)
class Candle:
    ts: datetime  # timezone-aware, candle START time
    open: float
    high: float
    low: float
    close: float
    volume: int
    complete: bool = True


@dataclass(frozen=True)
class Quote:
    symbol: str
    exchange: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    prev_close: float
    volume: int
    ts: datetime  # exchange/as-of timestamp, NOT the time we fetched it
    mode: DataMode
    source: str


class MarketDataProvider(ABC):
    name: str
    supported_intervals: tuple[str, ...]
    data_mode: DataMode

    @abstractmethod
    def search_instruments(self, query: str, exchange: str = "NSE") -> list[Instrument]: ...

    @abstractmethod
    def get_quote(self, instrument_id: str) -> Quote: ...

    @abstractmethod
    def get_historical_candles(
        self, instrument_id: str, interval: str, start: date, end: date
    ) -> list[Candle]: ...

    @abstractmethod
    def get_market_status(self) -> dict: ...

    @abstractmethod
    def get_provider_status(self) -> dict: ...

    def resolve_symbol(self, symbol: str, exchange: str = "NSE") -> Instrument:
        """Exact symbol match only; never guess."""
        wanted = symbol.strip().upper()
        for ins in self.search_instruments(wanted, exchange):
            if ins.symbol.upper() == wanted:
                return ins
        raise InstrumentNotFound(f"'{symbol}' not found on {exchange} for provider '{self.name}'.")
