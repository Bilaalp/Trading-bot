# Trading-bot

A modular, risk-managed trading bot. Backtest strategies on historical or
synthetic data, paper trade against live markets, and only then — behind
explicit safety flags — trade real money on any [ccxt](https://github.com/ccxt/ccxt)-supported
exchange (Binance, Coinbase, Kraken, …).

> **Reality check:** no bot guarantees profits. Most strategies lose money
> after fees. This project's job is to let you find out *cheaply* — in
> backtests and paper trading — before any real money is at stake, and to
> hard-cap losses (stop losses, position limits, a drawdown kill switch)
> if you do go live. Never trade money you can't afford to lose.

## Quick start (no dependencies needed)

```bash
# Run the test suite
pip install pytest
python -m pytest

# Backtest the default SMA-crossover strategy on synthetic data
python -m trading_bot backtest --data synthetic

# List built-in strategies
python -m trading_bot strategies
```

## Backtest on real market data

Real BTC/USD daily candles (Aug 2010 – Jul 2026) ship with the repo, so this
works offline with no dependencies:

```bash
python -m trading_bot backtest --data data/btc_usd_1d.csv
```

The file is aggregated from the hourly OHLCV dataset in
[mouadja02/bitcoin-technical-indicators-dataset](https://github.com/mouadja02/bitcoin-technical-indicators-dataset).
To fetch fresh candles for any symbol yourself:

```bash
pip install ccxt

# Download 500 hourly BTC/USDT candles (public data, no API keys needed)
python -m trading_bot fetch --symbol BTC/USDT --out btc.csv

# Backtest against them
python -m trading_bot backtest --data btc.csv
```

## Configure

```bash
cp config.example.toml config.toml   # then edit
```

Everything lives in `config.toml`: exchange, symbol, timeframe, strategy and
its parameters, and the risk limits. The defaults are conservative on
purpose — loosen them only after you understand what they do.

| Risk setting | Default | Meaning |
|---|---|---|
| `risk_per_trade` | 1% | equity risked per trade (fixed-fractional sizing) |
| `stop_loss_pct` | 5% | stop loss below entry, checked intrabar |
| `max_position_pct` | 25% | max equity in a single position |
| `max_drawdown_pct` | 20% | kill switch: all trading halts past this drawdown |
| `fee_pct` | 0.1% | fee charged on every fill, in backtests too |

## Paper trade (recommended for weeks, not days)

Runs the full live pipeline against real market data with simulated money:

```bash
python -m trading_bot paper
```

## Live trading (real money — read this section fully)

1. Create exchange API keys with **trade permission only** — never enable
   withdrawals for bot keys. Restrict them to your server's IP if the
   exchange supports it.
2. Put them in the environment (see `.env.example`); they are never read
   from the config file and `.env` is gitignored.
3. Both safety flags are required, deliberately:

```bash
export EXCHANGE_API_KEY=... EXCHANGE_API_SECRET=...
python -m trading_bot live --live --yes-i-understand
```

The engine trades long-only spot (no leverage, no shorting), sizes every
entry off your risk config, exits on stops, and latches a permanent halt if
the drawdown kill switch trips.

## Writing your own strategy

Subclass `Strategy`, register it, backtest it:

```python
from trading_bot.strategy import STRATEGIES, Strategy
from trading_bot.models import Side, Signal

class MyStrategy(Strategy):
    name = "my_strategy"

    def on_candle(self, candles, has_position):
        if candles[-1].close > candles[-2].close and not has_position:
            return Signal(Side.BUY, "momentum up")
        return None

STRATEGIES[MyStrategy.name] = MyStrategy
```

Strategies only see candles and return signals — sizing, stops, fees, and
the kill switch are applied outside, identically in backtest and live mode.

## Project layout

```
trading_bot/
  models.py      # Candle, Order, Position, Trade, Account
  indicators.py  # SMA, EMA, RSI, ATR (pure Python)
  strategy.py    # Strategy base + sma_crossover, rsi_mean_reversion
  risk.py        # position sizing, stops, drawdown kill switch
  backtest.py    # event-driven backtester with fees and intrabar stops
  data.py        # CSV / synthetic / ccxt candle sources
  broker.py      # PaperBroker (default) and CcxtBroker (live, gated)
  engine.py      # the live/paper polling loop
  cli.py         # backtest | fetch | paper | live | strategies
tests/           # 38 tests, no network required
```

## Disclaimer

This software is provided for educational purposes, without warranty of any
kind. Trading involves substantial risk of loss. You are solely responsible
for any trades executed with it and for complying with the laws and tax
rules of your jurisdiction.
