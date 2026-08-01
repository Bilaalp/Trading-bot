"""Risk management: position sizing, stops, and the drawdown kill switch.

This module is the difference between a strategy and a bot you can trust
with money. Every order passes through RiskManager before it reaches a
broker, in backtests and in live trading alike.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Account, Candle, Position


@dataclass
class RiskConfig:
    """All knobs in one place; see config.example.yaml for docs."""

    risk_per_trade: float = 0.01       # fraction of equity risked per trade
    stop_loss_pct: float = 0.05        # stop distance below entry
    max_position_pct: float = 0.25     # cap on equity deployed in one position
    max_drawdown_pct: float = 0.20     # kill switch: halt if equity falls this far from peak
    fee_pct: float = 0.001             # taker fee assumed on every fill

    def __post_init__(self) -> None:
        for name in ("risk_per_trade", "stop_loss_pct", "max_position_pct", "max_drawdown_pct"):
            value = getattr(self, name)
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1], got {value}")
        if not 0 <= self.fee_pct < 1:
            raise ValueError(f"fee_pct must be in [0, 1), got {self.fee_pct}")


class RiskManager:
    """Sizes entries, tracks stops, and halts trading past max drawdown."""

    def __init__(self, config: RiskConfig):
        self.config = config
        self.peak_equity: float | None = None
        self.halted = False

    def position_size(self, equity: float, price: float) -> float:
        """Base-asset quantity to buy, derived from fixed-fractional risk.

        Risking `risk_per_trade` of equity with a stop `stop_loss_pct` below
        entry gives amount = equity * risk / (price * stop_pct), then capped
        so the position never exceeds `max_position_pct` of equity.
        """
        if price <= 0 or equity <= 0:
            return 0.0
        risk_amount = equity * self.config.risk_per_trade
        amount = risk_amount / (price * self.config.stop_loss_pct)
        max_amount = equity * self.config.max_position_pct / price
        return min(amount, max_amount)

    def stop_price(self, entry_price: float) -> float:
        return entry_price * (1 - self.config.stop_loss_pct)

    def stop_hit(self, position: Position, candle: Candle) -> bool:
        return position.stop_price is not None and candle.low <= position.stop_price

    def update_drawdown(self, equity: float) -> bool:
        """Track peak equity; returns True (and latches halted) past the limit."""
        if self.peak_equity is None or equity > self.peak_equity:
            self.peak_equity = equity
        drawdown = 1 - equity / self.peak_equity if self.peak_equity > 0 else 0.0
        if drawdown >= self.config.max_drawdown_pct:
            self.halted = True
        return self.halted

    def can_open(self, account: Account, price: float) -> bool:
        """Gate for new entries: not halted, no open position, cash available."""
        if self.halted or account.position is not None:
            return False
        return account.cash > 0 and self.position_size(account.equity(price), price) > 0

    def fee(self, notional: float) -> float:
        return abs(notional) * self.config.fee_pct
