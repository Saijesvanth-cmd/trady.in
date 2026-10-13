from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query

from app.core.config import Settings, get_settings
from app.core.errors import BadRequest, ProviderError
from app.core.timeutil import IST, iso_ist, now_utc
from app.market_data.base import MarketDataProvider
from app.market_data.service import get_provider, get_validated_candles
from app.indicators.core import compute_snapshot

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health(settings: Settings = Depends(get_settings)):
    try:
        provider = get_provider().get_provider_status()
    except ProviderError as e:
        provider = {"connected": False, "detail": e.message}
    return {
        "status": "ok",
        "env": settings.app_env,
        "paper_trading_enabled": settings.paper_trading_enabled,
        "live_order_execution_enabled": False,  # hard-wired: no real orders in this app
        "provider": provider,
        "server_time_ist": iso_ist(now_utc()),
    }


@router.get("/market/status")
def market_status(provider: MarketDataProvider = Depends(get_provider)):
    return provider.get_market_status()


@router.get("/market/instruments")
def instruments(q: str = Query("", max_length=60), exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
                provider: MarketDataProvider = Depends(get_provider)):
    found = provider.search_instruments(q, exchange)
    return {"data_mode": provider.data_mode.value, "exchange": exchange,
            "instruments": [vars(i) for i in found]}


@router.get("/market/quote/{symbol}")
def quote(symbol: str, exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
          provider: MarketDataProvider = Depends(get_provider)):
    ins = provider.resolve_symbol(symbol, exchange)
    q = provider.get_quote(ins.instrument_id)
    return {
        "symbol": q.symbol, "exchange": q.exchange, "ltp": q.ltp, "open": q.open, "high": q.high,
        "low": q.low, "close": q.close, "prev_close": q.prev_close, "volume": q.volume,
        "change_pct": round((q.ltp / q.prev_close - 1) * 100, 2),
        "as_of": iso_ist(q.ts), "timezone": "Asia/Kolkata",
        "data_mode": q.mode.value, "source": q.source,
    }


def _candles_for(symbol: str, exchange: str, interval: str, start: date | None, end: date | None,
                 provider: MarketDataProvider, default_days: int):
    end = end or now_utc().astimezone(IST).date()
    start = start or end - timedelta(days=default_days)
    if start > end:
        raise BadRequest("'start' must not be after 'end'.")
    ins = provider.resolve_symbol(symbol, exchange)
    return ins, get_validated_candles(provider, ins, interval, start, end)


@router.get("/market/candles/{symbol}")
def candles(symbol: str, exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
            interval: str = "1d", start: date | None = None, end: date | None = None,
            provider: MarketDataProvider = Depends(get_provider)):
    ins, cs = _candles_for(symbol, exchange, interval, start, end, provider, default_days=365)
    return {
        "symbol": ins.symbol, "exchange": ins.exchange, "interval": interval,
        "timezone": "Asia/Kolkata", "data_mode": provider.data_mode.value, "count": len(cs),
        "candles": [{"t": iso_ist(c.ts), "o": c.open, "h": c.high, "l": c.low, "c": c.close,
                     "v": c.volume, "complete": c.complete} for c in cs],
    }


@router.get("/market/indicators/{symbol}")
def indicators(symbol: str, exchange: str = Query("NSE", pattern="^(NSE|BSE)$"),
               interval: str = "1d", provider: MarketDataProvider = Depends(get_provider)):
    ins, cs = _candles_for(symbol, exchange, interval, None, None, provider, default_days=500)
    snap = compute_snapshot(cs)
    return {
        "symbol": ins.symbol, "exchange": ins.exchange, "interval": interval,
        "data_mode": provider.data_mode.value,
        "as_of": iso_ist(snap["as_of"]) if snap["as_of"] else None,
        "candles_used": snap["candles_used"], "last_close": snap.get("last_close"),
        "indicators": snap["indicators"], "unavailable": snap["unavailable"],
        "computed_by": "python",
    }
