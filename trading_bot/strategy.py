"""Strategy interface and built-in strategies.

A strategy sees the full candle history up to "now" and returns a Signal
(or None to do nothing). Strategies never touch the broker directly, which
keeps them backtestable and safe to iterate on.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .indicators import rsi, sma
from .models import Candle, Side, Signal


class Strategy(ABC):
    """Base class. Subclass and implement on_candle()."""

    name = "base"

    @abstractmethod
    def on_candle(self, candles: Sequence[Candle], has_position: bool) -> Signal | None:
        """Called once per closed candle. candles[-1] is the latest bar."""

    @property
    def warmup(self) -> int:
        """Bars needed before the strategy can emit signals."""
        return 0


class SmaCrossover(Strategy):
    """Buy when the fast SMA crosses above the slow SMA, sell on the reverse.

    A classic trend-following baseline: it will not catch tops or bottoms,
    but it keeps you on the right side of sustained moves.
    """

    name = "sma_crossover"

    def __init__(self, fast: int = 10, slow: int = 30):
        if fast >= slow:
            raise ValueError("fast period must be smaller than slow period")
        self.fast = fast
        self.slow = slow

    @property
    def warmup(self) -> int:
        return self.slow + 1

    def on_candle(self, candles: Sequence[Candle], has_position: bool) -> Signal | None:
        if len(candles) < self.warmup:
            return None
        # Only the last slow+1 closes matter; keeps long backtests O(n).
        closes = [c.close for c in candles[-(self.slow + 1):]]
        fast = sma(closes, self.fast)
        slow = sma(closes, self.slow)
        prev_fast, prev_slow = fast[-2], slow[-2]
        cur_fast, cur_slow = fast[-1], slow[-1]
        if None in (prev_fast, prev_slow, cur_fast, cur_slow):
            return None
        crossed_up = prev_fast <= prev_slow and cur_fast > cur_slow
        crossed_down = prev_fast >= prev_slow and cur_fast < cur_slow
        if crossed_up and not has_position:
            return Signal(Side.BUY, f"SMA{self.fast} crossed above SMA{self.slow}")
        if crossed_down and has_position:
            return Signal(Side.SELL, f"SMA{self.fast} crossed below SMA{self.slow}")
        return None


class RsiMeanReversion(Strategy):
    """Buy oversold dips (RSI below the floor), exit when RSI recovers.

    Works best in ranging markets; pair it with the drawdown kill switch,
    because mean reversion loses money in strong downtrends.
    """

    name = "rsi_mean_reversion"

    def __init__(self, period: int = 14, oversold: float = 30.0, exit_level: float = 55.0):
        if not 0 < oversold < exit_level < 100:
            raise ValueError("require 0 < oversold < exit_level < 100")
        self.period = period
        self.oversold = oversold
        self.exit_level = exit_level

    @property
    def warmup(self) -> int:
        return self.period + 1

    def on_candle(self, candles: Sequence[Candle], has_position: bool) -> Signal | None:
        if len(candles) < self.warmup:
            return None
        # Wilder smoothing technically spans all history, but converges fast;
        # a 10-period tail is accurate to well under a point and keeps long
        # backtests O(n).
        closes = [c.close for c in candles[-(self.period * 10):]]
        value = rsi(closes, self.period)[-1]
        if value is None:
            return None
        if value < self.oversold and not has_position:
            return Signal(Side.BUY, f"RSI {value:.1f} below {self.oversold}")
        if value > self.exit_level and has_position:
            return Signal(Side.SELL, f"RSI {value:.1f} above {self.exit_level}")
        return None


STRATEGIES: dict[str, type[Strategy]] = {
    SmaCrossover.name: SmaCrossover,
    RsiMeanReversion.name: RsiMeanReversion,
}


def build_strategy(name: str, params: dict | None = None) -> Strategy:
    """Instantiate a registered strategy by name with keyword params."""
    try:
        cls = STRATEGIES[name]
    except KeyError:
        known = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"unknown strategy {name!r}; available: {known}") from None
    return cls(**(params or {}))
