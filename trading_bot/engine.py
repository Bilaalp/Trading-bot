"""The live/paper trading loop.

Polls the exchange for closed candles on the configured timeframe and runs
each new bar through the same pipeline as the backtester: stop check →
drawdown kill switch → strategy signal → risk-sized order.
"""

from __future__ import annotations

import logging
import time

from .broker import Broker
from .data import fetch_ohlcv
from .models import Candle, Side
from .risk import RiskManager
from .strategy import Strategy

log = logging.getLogger("trading_bot")

_TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class TradingEngine:
    def __init__(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        strategy: Strategy,
        risk: RiskManager,
        broker: Broker,
        history_limit: int = 500,
    ):
        if timeframe not in _TIMEFRAME_MS:
            raise ValueError(f"unsupported timeframe {timeframe!r}; use one of {sorted(_TIMEFRAME_MS)}")
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.strategy = strategy
        self.risk = risk
        self.broker = broker
        self.history_limit = max(history_limit, strategy.warmup + 2)
        self.candles: list[Candle] = []

    def run(self, max_iterations: int | None = None) -> None:
        """Main loop. max_iterations is for tests; None means run until halted."""
        log.info(
            "engine starting: %s %s %s strategy=%s",
            self.exchange_id, self.symbol, self.timeframe, self.strategy.name,
        )
        iterations = 0
        while max_iterations is None or iterations < max_iterations:
            iterations += 1
            try:
                fetched = fetch_ohlcv(
                    self.exchange_id, self.symbol, self.timeframe, self.history_limit
                )
                # The most recent bar is still forming; act only on closed bars.
                closed = fetched[:-1]
                new_bars = [
                    c for c in closed
                    if not self.candles or c.timestamp > self.candles[-1].timestamp
                ]
                for candle in new_bars:
                    self.candles.append(candle)
                    self.on_closed_candle(candle)
                if self.risk.halted:
                    log.error("drawdown kill switch tripped — engine halted")
                    return
            except Exception:
                log.exception("engine iteration failed; retrying next cycle")
            self._sleep_until_next_bar()

    def on_closed_candle(self, candle: Candle) -> None:
        account = self.broker.account
        position = account.position

        if position and self.risk.stop_hit(position, candle):
            order = self.broker.sell(position.stop_price or candle.close, candle.timestamp)
            log.warning("stop loss hit: sold %.6f @ %.2f", order.amount, order.price)
            position = None

        equity = account.equity(candle.close)
        if self.risk.update_drawdown(equity):
            if account.position:
                self.broker.sell(candle.close, candle.timestamp)
            return

        signal = self.strategy.on_candle(self.candles, account.position is not None)
        if signal is None:
            return
        log.info("signal: %s (%s)", signal.side.value, signal.reason)
        if signal.side is Side.BUY and self.risk.can_open(account, candle.close):
            amount = self.risk.position_size(equity, candle.close)
            order = self.broker.buy(
                amount, candle.close, candle.timestamp, self.risk.stop_price(candle.close)
            )
            log.info("bought %.6f @ %.2f (equity %.2f)", order.amount, order.price, equity)
        elif signal.side is Side.SELL and account.position:
            order = self.broker.sell(candle.close, candle.timestamp)
            if order:
                log.info("sold %.6f @ %.2f (equity %.2f)", order.amount, order.price, equity)

    def _sleep_until_next_bar(self) -> None:
        interval = _TIMEFRAME_MS[self.timeframe] / 1000
        now = time.time()
        # Wake shortly after the next bar close so the exchange has it ready.
        wait = interval - (now % interval) + 5
        time.sleep(min(wait, interval))
