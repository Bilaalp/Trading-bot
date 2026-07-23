"""Candle data sources: CSV files, synthetic data, and live exchanges.

The exchange feed uses ccxt (pip install ccxt) and is imported lazily so the
backtester and tests work with no third-party dependencies installed.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from .models import Candle


def load_csv(path: str | Path) -> list[Candle]:
    """Load candles from a CSV with header: timestamp,open,high,low,close,volume.

    Timestamps must be epoch milliseconds. Rows are returned sorted by time.
    """
    candles: list[Candle] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            candles.append(
                Candle(
                    timestamp=int(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    candles.sort(key=lambda c: c.timestamp)
    return candles


def save_csv(candles: list[Candle], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c.timestamp, c.open, c.high, c.low, c.close, c.volume])


def synthetic_candles(
    n: int = 500,
    start_price: float = 100.0,
    drift: float = 0.0002,
    volatility: float = 0.02,
    seed: int | None = 42,
    start_timestamp: int = 1_700_000_000_000,
    interval_ms: int = 3_600_000,
) -> list[Candle]:
    """Geometric-random-walk candles for demos and tests (deterministic by default)."""
    rng = random.Random(seed)
    candles: list[Candle] = []
    price = start_price
    for i in range(n):
        ret = rng.gauss(drift, volatility)
        close = max(price * (1 + ret), 0.01)
        high = max(price, close) * (1 + abs(rng.gauss(0, volatility / 3)))
        low = min(price, close) * (1 - abs(rng.gauss(0, volatility / 3)))
        candles.append(
            Candle(
                timestamp=start_timestamp + i * interval_ms,
                open=price,
                high=high,
                low=low,
                close=close,
                volume=abs(rng.gauss(1000, 300)),
            )
        )
        price = close
    return candles


def fetch_ohlcv(
    exchange_id: str,
    symbol: str,
    timeframe: str = "1h",
    limit: int = 500,
) -> list[Candle]:
    """Fetch recent candles from a real exchange via ccxt (public endpoint, no keys)."""
    try:
        import ccxt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "ccxt is required for exchange data: pip install ccxt"
        ) from exc
    exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    return [
        Candle(
            timestamp=int(ts),
            open=float(o),
            high=float(h),
            low=float(lo),
            close=float(c),
            volume=float(v),
        )
        for ts, o, h, lo, c, v in raw
    ]
