"""Technical indicators implemented in pure Python.

Each function takes a sequence of closing prices and returns a list the same
length as the input, with None for warm-up bars where the indicator is not
yet defined.
"""

from __future__ import annotations

from collections.abc import Sequence


def sma(prices: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(prices)
    running = 0.0
    for i, price in enumerate(prices):
        running += price
        if i >= period:
            running -= prices[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(prices: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with the SMA of the first window."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(prices)
    if len(prices) < period:
        return out
    alpha = 2.0 / (period + 1)
    seed = sum(prices[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(prices)):
        prev = alpha * prices[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rsi(prices: Sequence[float], period: int = 14) -> list[float | None]:
    """Relative Strength Index using Wilder's smoothing."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(prices)
    if len(prices) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        delta = prices[i] - prices[i - 1]
        if delta >= 0:
            gains += delta
        else:
            losses -= delta
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_value(avg_gain, avg_loss)
    for i in range(period + 1, len(prices)):
        delta = prices[i] - prices[i - 1]
        gain = max(delta, 0.0)
        loss = max(-delta, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_value(avg_gain, avg_loss)
    return out


def _rsi_value(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Average True Range (Wilder), used for volatility-aware stops."""
    n = len(closes)
    if not (len(highs) == len(lows) == n):
        raise ValueError("highs, lows and closes must be the same length")
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * n
    if n <= period:
        return out
    true_ranges = [highs[0] - lows[0]]
    for i in range(1, n):
        true_ranges.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    prev = sum(true_ranges[1 : period + 1]) / period
    out[period] = prev
    for i in range(period + 1, n):
        prev = (prev * (period - 1) + true_ranges[i]) / period
        out[i] = prev
    return out
