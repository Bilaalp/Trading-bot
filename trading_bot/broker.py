"""Broker implementations.

PaperBroker simulates fills locally with the same fee model as the
backtester — it is the default everywhere. CcxtBroker places real orders
on a real exchange and refuses to start unless live trading is explicitly
enabled and API keys are present in the environment.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .models import Account, Order, Position, Side
from .risk import RiskManager


class Broker(ABC):
    @abstractmethod
    def buy(self, amount: float, price: float, timestamp: int, stop_price: float | None) -> Order:
        ...

    @abstractmethod
    def sell(self, price: float, timestamp: int) -> Order | None:
        ...

    @property
    @abstractmethod
    def account(self) -> Account:
        ...


class PaperBroker(Broker):
    """Simulated broker: instant fills at the given price, fees applied."""

    def __init__(self, initial_cash: float, risk: RiskManager):
        self._account = Account(cash=initial_cash)
        self._risk = risk

    @property
    def account(self) -> Account:
        return self._account

    def buy(self, amount: float, price: float, timestamp: int, stop_price: float | None) -> Order:
        cost = amount * price
        fee = self._risk.fee(cost)
        spend_all = cost + fee > self._account.cash
        if spend_all:
            amount = self._account.cash / (price * (1 + self._risk.config.fee_pct))
            cost = amount * price
            fee = self._risk.fee(cost)
        order = Order(Side.BUY, amount, price, timestamp, fee)
        # Exact zero when spending everything avoids negative float dust.
        self._account.cash = 0.0 if spend_all else self._account.cash - cost - fee
        self._account.position = Position(amount, price, timestamp, stop_price)
        self._account.orders.append(order)
        return order

    def sell(self, price: float, timestamp: int) -> Order | None:
        position = self._account.position
        if position is None:
            return None
        proceeds = position.amount * price
        fee = self._risk.fee(proceeds)
        order = Order(Side.SELL, position.amount, price, timestamp, fee)
        self._account.cash += proceeds - fee
        self._account.position = None
        self._account.orders.append(order)
        return order


class CcxtBroker(Broker):
    """Live broker backed by ccxt market orders.

    Safety interlocks — ALL of these must be true or __init__ raises:
      * live=True was passed explicitly (the CLI requires --live --yes-i-understand)
      * API credentials exist in the environment (EXCHANGE_API_KEY / EXCHANGE_API_SECRET)
    """

    def __init__(self, exchange_id: str, symbol: str, risk: RiskManager, live: bool = False):
        if not live:
            raise RuntimeError("CcxtBroker requires live=True; use PaperBroker otherwise")
        api_key = os.environ.get("EXCHANGE_API_KEY")
        api_secret = os.environ.get("EXCHANGE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError(
                "Set EXCHANGE_API_KEY and EXCHANGE_API_SECRET in the environment "
                "(see .env.example). Never commit credentials to the repository."
            )
        try:
            import ccxt  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("ccxt is required for live trading: pip install ccxt") from exc
        self._exchange = getattr(ccxt, exchange_id)(
            {"apiKey": api_key, "secret": api_secret, "enableRateLimit": True}
        )
        self._symbol = symbol
        self._risk = risk
        self._account = Account(cash=self._fetch_quote_balance())

    def _fetch_quote_balance(self) -> float:
        quote = self._symbol.split("/")[1]
        balance = self._exchange.fetch_balance()
        return float(balance.get("free", {}).get(quote, 0.0))

    @property
    def account(self) -> Account:
        return self._account

    def buy(self, amount: float, price: float, timestamp: int, stop_price: float | None) -> Order:
        result = self._exchange.create_market_buy_order(self._symbol, amount)
        fill_price = float(result.get("average") or result.get("price") or price)
        filled = float(result.get("filled") or amount)
        fee = self._risk.fee(filled * fill_price)
        order = Order(Side.BUY, filled, fill_price, timestamp, fee)
        self._account.cash = self._fetch_quote_balance()
        self._account.position = Position(filled, fill_price, timestamp, stop_price)
        self._account.orders.append(order)
        return order

    def sell(self, price: float, timestamp: int) -> Order | None:
        position = self._account.position
        if position is None:
            return None
        result = self._exchange.create_market_sell_order(self._symbol, position.amount)
        fill_price = float(result.get("average") or result.get("price") or price)
        filled = float(result.get("filled") or position.amount)
        fee = self._risk.fee(filled * fill_price)
        order = Order(Side.SELL, filled, fill_price, timestamp, fee)
        self._account.cash = self._fetch_quote_balance()
        self._account.position = None
        self._account.orders.append(order)
        return order
