import pytest

from trading_bot.indicators import atr, ema, rsi, sma


def test_sma_basic():
    values = sma([1, 2, 3, 4, 5], 3)
    assert values[:2] == [None, None]
    assert values[2:] == [2.0, 3.0, 4.0]


def test_sma_rejects_bad_period():
    with pytest.raises(ValueError):
        sma([1, 2, 3], 0)


def test_ema_seeds_with_sma_and_converges():
    prices = [10.0] * 5 + [20.0] * 50
    values = ema(prices, 5)
    assert values[4] == 10.0
    assert values[-1] == pytest.approx(20.0, abs=0.01)


def test_rsi_bounds_and_direction():
    rising = list(range(1, 40))
    falling = list(range(40, 1, -1))
    up = rsi(rising, 14)[-1]
    down = rsi(falling, 14)[-1]
    assert up == 100.0          # all gains, no losses
    assert down == 0.0          # all losses, no gains
    mixed = rsi([100 + (i % 3) for i in range(40)], 14)[-1]
    assert 0 < mixed < 100


def test_rsi_short_series_all_none():
    assert rsi([1, 2, 3], 14) == [None, None, None]


def test_atr_positive_and_warmup():
    highs = [11, 12, 13, 12, 14, 15, 13]
    lows = [9, 10, 11, 10, 12, 13, 11]
    closes = [10, 11, 12, 11, 13, 14, 12]
    values = atr(highs, lows, closes, 3)
    assert values[:3] == [None, None, None]
    assert all(v > 0 for v in values[3:])


def test_atr_length_mismatch():
    with pytest.raises(ValueError):
        atr([1, 2], [1], [1, 2], 1)
