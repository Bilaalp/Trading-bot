import pytest

from trading_bot.backtest import run_backtest
from trading_bot.data import synthetic_candles
from trading_bot.models import Candle, Side, Signal
from trading_bot.risk import RiskConfig, RiskManager
from trading_bot.strategy import SmaCrossover, Strategy


class BuyOnceStrategy(Strategy):
    """Buys exactly once and never exits (tests stops/kill switch)."""

    name = "buy_once"

    def __init__(self):
        self.bought = False

    def on_candle(self, candles, has_position):
        if not has_position and not self.bought:
            self.bought = True
            return Signal(Side.BUY, "test entry")
        return None


def flat_candles(n: int, price: float = 100.0) -> list[Candle]:
    return [Candle(i * 1000, price, price, price, price, 10.0) for i in range(n)]


def trend_candles(closes: list[float]) -> list[Candle]:
    return [Candle(i * 1000, c, c * 1.001, c * 0.999, c, 10.0) for i, c in enumerate(closes)]


def test_no_trades_on_flat_market():
    result = run_backtest(
        flat_candles(100), SmaCrossover(3, 10), RiskManager(RiskConfig()), 10_000
    )
    assert result.num_trades == 0
    assert result.final_equity == pytest.approx(10_000)


def test_uptrend_is_profitable_for_trend_follower():
    closes = [100.0] * 40 + [100 * 1.01 ** i for i in range(1, 60)]
    result = run_backtest(
        trend_candles(closes),
        SmaCrossover(5, 20),
        RiskManager(RiskConfig(max_position_pct=1.0, risk_per_trade=0.05)),
        10_000,
    )
    assert result.final_equity > 10_000


def test_stop_loss_limits_single_trade_loss():
    # Buy at 100, then crash: the stop at 95 must exit the position.
    closes = [100.0, 100.0, 80.0, 60.0, 40.0]
    candles = [Candle(i * 1000, c, c, c * 0.99, c, 10.0) for i, c in enumerate(closes)]
    config = RiskConfig(stop_loss_pct=0.05, max_position_pct=1.0, max_drawdown_pct=0.99)
    result = run_backtest(candles, BuyOnceStrategy(), RiskManager(config), 10_000)
    assert result.num_trades == 1
    assert result.trades[0].exit.price == pytest.approx(95.0)


def test_kill_switch_halts_backtest():
    closes = [100.0] + [100 - 2 * i for i in range(1, 40)]
    candles = [Candle(i * 1000, c, c, c, c, 10.0) for i, c in enumerate(closes)]
    config = RiskConfig(
        stop_loss_pct=0.9, max_position_pct=1.0, risk_per_trade=1.0, max_drawdown_pct=0.10
    )
    result = run_backtest(candles, BuyOnceStrategy(), RiskManager(config), 10_000)
    assert result.halted_by_drawdown
    # Equity curve is padded to full length even when halted early.
    assert len(result.equity_curve) == len(candles)


def test_fees_reduce_equity():
    closes = [100.0] * 40 + [100 * 1.01 ** i for i in range(1, 60)]
    candles = trend_candles(closes)
    strategy = lambda: SmaCrossover(5, 20)  # noqa: E731
    no_fees = run_backtest(
        candles, strategy(),
        RiskManager(RiskConfig(fee_pct=0.0, max_position_pct=1.0)), 10_000,
    )
    with_fees = run_backtest(
        candles, strategy(),
        RiskManager(RiskConfig(fee_pct=0.005, max_position_pct=1.0)), 10_000,
    )
    assert with_fees.final_equity < no_fees.final_equity


def test_never_spends_more_than_cash():
    config = RiskConfig(risk_per_trade=1.0, stop_loss_pct=0.05, max_position_pct=1.0)
    candles = flat_candles(5)
    result = run_backtest(candles, BuyOnceStrategy(), RiskManager(config), 1_000)
    # Position was capped to available cash; equity only drops by fees.
    assert result.final_equity == pytest.approx(1_000, rel=0.01)


def test_synthetic_smoke_run():
    candles = synthetic_candles(300, seed=7)
    result = run_backtest(candles, SmaCrossover(), RiskManager(RiskConfig()), 10_000)
    assert len(result.equity_curve) == 300
    assert result.final_equity > 0
