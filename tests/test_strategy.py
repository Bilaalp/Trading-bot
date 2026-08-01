import pytest

from trading_bot.models import Candle, Side
from trading_bot.strategy import RsiMeanReversion, SmaCrossover, build_strategy


def candles_from_closes(closes: list[float]) -> list[Candle]:
    return [
        Candle(i * 1000, c, c * 1.01, c * 0.99, c, 100.0)
        for i, c in enumerate(closes)
    ]


def test_sma_crossover_buys_on_golden_cross():
    # Flat, then a sharp rally: fast SMA must cross above slow SMA.
    closes = [100.0] * 30 + [100 + 3 * i for i in range(1, 15)]
    strategy = SmaCrossover(fast=3, slow=10)
    signals = []
    candles = candles_from_closes(closes)
    has_position = False
    for i in range(len(candles)):
        sig = strategy.on_candle(candles[: i + 1], has_position)
        if sig:
            signals.append(sig)
            has_position = sig.side is Side.BUY
    assert signals and signals[0].side is Side.BUY


def test_sma_crossover_sells_after_reversal():
    closes = [100.0] * 20 + [100 + 3 * i for i in range(1, 15)] + [142 - 4 * i for i in range(1, 15)]
    strategy = SmaCrossover(fast=3, slow=10)
    candles = candles_from_closes(closes)
    sides = []
    has_position = False
    for i in range(len(candles)):
        sig = strategy.on_candle(candles[: i + 1], has_position)
        if sig:
            sides.append(sig.side)
            has_position = sig.side is Side.BUY
    assert sides == [Side.BUY, Side.SELL]


def test_sma_crossover_validates_periods():
    with pytest.raises(ValueError):
        SmaCrossover(fast=30, slow=10)


def test_rsi_strategy_buys_oversold_sells_recovered():
    closes = [100 - 2 * i for i in range(20)]        # crash -> oversold
    closes += [closes[-1] + 3 * i for i in range(1, 20)]  # recovery
    strategy = RsiMeanReversion(period=14, oversold=30, exit_level=55)
    candles = candles_from_closes(closes)
    sides = []
    has_position = False
    for i in range(len(candles)):
        sig = strategy.on_candle(candles[: i + 1], has_position)
        if sig:
            sides.append(sig.side)
            has_position = sig.side is Side.BUY
    assert sides[:2] == [Side.BUY, Side.SELL]


def test_no_signal_during_warmup():
    strategy = SmaCrossover(fast=3, slow=10)
    candles = candles_from_closes([100.0] * 5)
    assert strategy.on_candle(candles, False) is None


def test_build_strategy_by_name_and_params():
    strategy = build_strategy("sma_crossover", {"fast": 5, "slow": 20})
    assert isinstance(strategy, SmaCrossover)
    assert (strategy.fast, strategy.slow) == (5, 20)
    with pytest.raises(ValueError, match="unknown strategy"):
        build_strategy("does_not_exist")
