import pytest

from trading_bot.models import Account, Candle, Position
from trading_bot.risk import RiskConfig, RiskManager


def make_candle(low: float, close: float | None = None) -> Candle:
    close = close if close is not None else low + 1
    return Candle(0, close, close + 1, low, close, 100.0)


def test_position_size_fixed_fractional():
    risk = RiskManager(RiskConfig(risk_per_trade=0.01, stop_loss_pct=0.05, max_position_pct=1.0))
    # equity 10_000, risk 100, stop distance 5 -> 20 units at price 100
    assert risk.position_size(10_000, 100) == pytest.approx(20.0)


def test_position_size_capped_by_max_position():
    risk = RiskManager(RiskConfig(risk_per_trade=0.10, stop_loss_pct=0.05, max_position_pct=0.25))
    # uncapped would be 200 units (20k notional); cap is 25% of 10k = 25 units
    assert risk.position_size(10_000, 100) == pytest.approx(25.0)


def test_position_size_zero_on_bad_inputs():
    risk = RiskManager(RiskConfig())
    assert risk.position_size(0, 100) == 0.0
    assert risk.position_size(10_000, 0) == 0.0


def test_stop_price_and_hit():
    risk = RiskManager(RiskConfig(stop_loss_pct=0.05))
    stop = risk.stop_price(100.0)
    assert stop == pytest.approx(95.0)
    position = Position(1.0, 100.0, 0, stop_price=stop)
    assert risk.stop_hit(position, make_candle(low=94.0))
    assert not risk.stop_hit(position, make_candle(low=96.0))


def test_drawdown_kill_switch_latches():
    risk = RiskManager(RiskConfig(max_drawdown_pct=0.20))
    assert not risk.update_drawdown(10_000)
    assert not risk.update_drawdown(9_000)   # 10% down: fine
    assert risk.update_drawdown(7_900)       # 21% down: halt
    assert risk.halted
    assert risk.update_drawdown(12_000)      # recovery does not un-halt


def test_can_open_blocked_when_halted_or_in_position():
    risk = RiskManager(RiskConfig())
    account = Account(cash=1_000)
    assert risk.can_open(account, 100)
    account.position = Position(1.0, 100.0, 0)
    assert not risk.can_open(account, 100)
    account.position = None
    risk.halted = True
    assert not risk.can_open(account, 100)


def test_config_validation():
    with pytest.raises(ValueError):
        RiskConfig(risk_per_trade=0)
    with pytest.raises(ValueError):
        RiskConfig(max_drawdown_pct=1.5)
    with pytest.raises(ValueError):
        RiskConfig(fee_pct=-0.1)
