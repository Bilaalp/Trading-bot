"""Core data types shared across the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class Candle:
    """One OHLCV bar. Timestamp is epoch milliseconds (exchange convention)."""

    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class Signal:
    """A strategy's desired action for the current bar."""

    side: Side
    reason: str = ""


@dataclass
class Order:
    side: Side
    amount: float          # base-asset quantity
    price: float           # fill price
    timestamp: int
    fee: float = 0.0


@dataclass
class Position:
    """A single open long position (the bot is long-only by design)."""

    amount: float
    entry_price: float
    entry_timestamp: int
    stop_price: float | None = None

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.amount


@dataclass
class Trade:
    """A completed round trip (entry + exit), used for reporting."""

    entry: Order
    exit: Order

    @property
    def pnl(self) -> float:
        gross = (self.exit.price - self.entry.price) * self.entry.amount
        return gross - self.entry.fee - self.exit.fee

    @property
    def return_pct(self) -> float:
        cost = self.entry.price * self.entry.amount
        return self.pnl / cost if cost else 0.0


@dataclass
class Account:
    """Cash + position book kept by brokers and the backtester."""

    cash: float
    position: Position | None = None
    orders: list[Order] = field(default_factory=list)
    trades: list[Trade] = field(default_factory=list)

    def equity(self, price: float) -> float:
        value = self.position.amount * price if self.position else 0.0
        return self.cash + value
