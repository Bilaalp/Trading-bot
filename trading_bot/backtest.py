"""Event-driven backtester.

Replays candles through a strategy with the same risk pipeline used live:
signals are sized by RiskManager, stops fire intrabar, fees are charged on
every fill, and the drawdown kill switch halts the run just as it would
halt the live engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .models import Account, Candle, Order, Position, Side, Trade
from .risk import RiskManager
from .strategy import Strategy


_MS_PER_YEAR = 365 * 86_400_000


@dataclass
class BacktestResult:
    initial_cash: float
    final_equity: float
    trades: list[Trade]
    equity_curve: list[float] = field(default_factory=list)
    halted_by_drawdown: bool = False
    buy_hold_return_pct: float = 0.0   # benchmark: hold from first to last close
    bars_per_year: float = 0.0         # derived from candle timestamps

    @property
    def total_return_pct(self) -> float:
        return (self.final_equity / self.initial_cash - 1) * 100

    @property
    def sharpe(self) -> float:
        """Annualized Sharpe ratio of per-bar equity returns (risk-free rate 0)."""
        if len(self.equity_curve) < 3 or not self.bars_per_year:
            return 0.0
        returns = [
            b / a - 1
            for a, b in zip(self.equity_curve, self.equity_curve[1:])
            if a > 0
        ]
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / (n - 1)
        if variance == 0:
            return 0.0
        return mean / math.sqrt(variance) * math.sqrt(self.bars_per_year)

    @property
    def num_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    @property
    def max_drawdown_pct(self) -> float:
        peak = -math.inf
        worst = 0.0
        for equity in self.equity_curve:
            peak = max(peak, equity)
            if peak > 0:
                worst = max(worst, 1 - equity / peak)
        return worst * 100

    def summary(self) -> str:
        lines = [
            f"Initial cash:    {self.initial_cash:,.2f}",
            f"Final equity:    {self.final_equity:,.2f}",
            f"Total return:    {self.total_return_pct:+.2f}%",
            f"Buy & hold:      {self.buy_hold_return_pct:+.2f}%",
            f"Sharpe (ann.):   {self.sharpe:.2f}",
            f"Trades:          {self.num_trades}",
            f"Win rate:        {self.win_rate * 100:.1f}%",
            f"Max drawdown:    {self.max_drawdown_pct:.1f}%",
        ]
        if self.halted_by_drawdown:
            lines.append("NOTE: run halted early by the drawdown kill switch")
        return "\n".join(lines)


def run_backtest(
    candles: list[Candle],
    strategy: Strategy,
    risk: RiskManager,
    initial_cash: float = 10_000.0,
) -> BacktestResult:
    """Replay candles through the strategy and return performance stats.

    Fill model: signals computed on a closed bar are filled at that bar's
    close; stops are filled at the stop price if the bar's low crosses it
    (checked before the strategy sees the bar, mirroring the live engine).
    """
    account = Account(cash=initial_cash)
    open_order: Order | None = None
    equity_curve: list[float] = []
    history: list[Candle] = []  # grows in place; avoids O(n^2) slicing

    for i, candle in enumerate(candles):
        history.append(candle)
        # 1. Stops fire intrabar, before the strategy acts on the close.
        if account.position and risk.stop_hit(account.position, candle):
            open_order = _close_position(
                account, risk, account.position.stop_price, candle.timestamp, open_order
            )

        # 2. Kill switch on marked-to-market equity.
        equity = account.equity(candle.close)
        if risk.update_drawdown(equity) and account.position:
            open_order = _close_position(
                account, risk, candle.close, candle.timestamp, open_order
            )
            equity = account.equity(candle.close)
        equity_curve.append(equity)
        if risk.halted:
            equity_curve.extend(equity for _ in candles[i + 1 :])
            break

        # 3. Strategy sees history up to and including this closed bar.
        signal = strategy.on_candle(history, account.position is not None)
        if signal is None:
            continue
        if signal.side is Side.BUY and risk.can_open(account, candle.close):
            open_order = _open_position(account, risk, candle.close, candle.timestamp)
        elif signal.side is Side.SELL and account.position:
            open_order = _close_position(
                account, risk, candle.close, candle.timestamp, open_order
            )

    final_price = candles[-1].close if candles else 0.0
    buy_hold = (candles[-1].close / candles[0].close - 1) * 100 if candles else 0.0
    bars_per_year = 0.0
    if len(candles) > 1:
        avg_interval = (candles[-1].timestamp - candles[0].timestamp) / (len(candles) - 1)
        if avg_interval > 0:
            bars_per_year = _MS_PER_YEAR / avg_interval
    return BacktestResult(
        initial_cash=initial_cash,
        final_equity=account.equity(final_price),
        trades=account.trades,
        equity_curve=equity_curve,
        halted_by_drawdown=risk.halted,
        buy_hold_return_pct=buy_hold,
        bars_per_year=bars_per_year,
    )


def _open_position(
    account: Account, risk: RiskManager, price: float, timestamp: int
) -> Order:
    equity = account.equity(price)
    amount = risk.position_size(equity, price)
    cost = amount * price
    fee = risk.fee(cost)
    spend_all = cost + fee > account.cash  # never buy on margin
    if spend_all:
        amount = account.cash / (price * (1 + risk.config.fee_pct))
        cost = amount * price
        fee = risk.fee(cost)
    order = Order(Side.BUY, amount, price, timestamp, fee)
    # Exact zero when spending everything avoids negative float dust.
    account.cash = 0.0 if spend_all else account.cash - cost - fee
    account.position = Position(
        amount=amount,
        entry_price=price,
        entry_timestamp=timestamp,
        stop_price=risk.stop_price(price),
    )
    account.orders.append(order)
    return order


def _close_position(
    account: Account,
    risk: RiskManager,
    price: float,
    timestamp: int,
    entry_order: Order | None,
) -> None:
    position = account.position
    assert position is not None
    proceeds = position.amount * price
    fee = risk.fee(proceeds)
    order = Order(Side.SELL, position.amount, price, timestamp, fee)
    account.cash += proceeds - fee
    account.orders.append(order)
    if entry_order is not None:
        account.trades.append(Trade(entry=entry_order, exit=order))
    account.position = None
    return None
