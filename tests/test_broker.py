import pytest

from trading_bot.broker import CcxtBroker, PaperBroker
from trading_bot.risk import RiskConfig, RiskManager


def make_broker(cash: float = 10_000, fee_pct: float = 0.001) -> PaperBroker:
    return PaperBroker(cash, RiskManager(RiskConfig(fee_pct=fee_pct)))


def test_paper_buy_and_sell_roundtrip():
    broker = make_broker(fee_pct=0.0)
    broker.buy(10, 100.0, 0, stop_price=95.0)
    account = broker.account
    assert account.cash == pytest.approx(9_000)
    assert account.position.amount == 10
    assert account.position.stop_price == 95.0
    broker.sell(110.0, 1000)
    assert account.position is None
    assert account.cash == pytest.approx(10_100)


def test_paper_buy_caps_at_available_cash():
    broker = make_broker(cash=500, fee_pct=0.001)
    broker.buy(100, 100.0, 0, stop_price=None)  # would cost 10_000
    account = broker.account
    assert account.cash >= 0
    assert account.position.amount == pytest.approx(500 / (100 * 1.001))


def test_paper_sell_without_position_is_noop():
    broker = make_broker()
    assert broker.sell(100.0, 0) is None
    assert broker.account.cash == 10_000


def test_fees_charged_both_ways():
    broker = make_broker(fee_pct=0.01)
    broker.buy(10, 100.0, 0, stop_price=None)
    broker.sell(100.0, 1)
    # Flat price, but 1% fee each way on 1000 notional = 20 lost.
    assert broker.account.cash == pytest.approx(10_000 - 20)


def test_live_broker_refuses_without_live_flag():
    risk = RiskManager(RiskConfig())
    with pytest.raises(RuntimeError, match="live=True"):
        CcxtBroker("binance", "BTC/USDT", risk, live=False)


def test_live_broker_refuses_without_credentials(monkeypatch):
    monkeypatch.delenv("EXCHANGE_API_KEY", raising=False)
    monkeypatch.delenv("EXCHANGE_API_SECRET", raising=False)
    risk = RiskManager(RiskConfig())
    with pytest.raises(RuntimeError, match="EXCHANGE_API_KEY"):
        CcxtBroker("binance", "BTC/USDT", risk, live=True)
