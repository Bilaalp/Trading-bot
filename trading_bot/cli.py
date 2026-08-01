"""Command-line interface.

    python -m trading_bot backtest --data synthetic
    python -m trading_bot backtest --data candles.csv
    python -m trading_bot fetch --symbol BTC/USDT --out candles.csv
    python -m trading_bot paper
    python -m trading_bot live --live --yes-i-understand   # real money!
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .backtest import run_backtest
from .broker import CcxtBroker, PaperBroker
from .config import BotConfig, load_config
from .data import fetch_ohlcv, load_csv, save_csv, synthetic_candles
from .engine import TradingEngine
from .optimize import DEFAULT_PARAM_GRIDS, tune
from .risk import RiskManager
from .strategy import STRATEGIES, build_strategy


def _load(args: argparse.Namespace) -> BotConfig:
    path = Path(args.config)
    if path.exists():
        return load_config(path)
    if args.config != "config.toml":
        sys.exit(f"config file not found: {path}")
    return BotConfig()


def cmd_backtest(args: argparse.Namespace) -> None:
    config = _load(args)
    if args.data == "synthetic":
        candles = synthetic_candles(n=args.bars)
    elif args.data == "exchange":
        candles = fetch_ohlcv(config.exchange, config.symbol, config.timeframe, args.bars)
    else:
        candles = load_csv(args.data)
    strategy = build_strategy(config.strategy, config.strategy_params)
    result = run_backtest(candles, strategy, RiskManager(config.risk), config.initial_cash)
    print(f"Strategy: {strategy.name}  |  bars: {len(candles)}")
    print(result.summary())


def cmd_tune(args: argparse.Namespace) -> None:
    config = _load(args)
    strategy = args.strategy or config.strategy
    if strategy not in DEFAULT_PARAM_GRIDS:
        sys.exit(f"no default grid for {strategy!r}; available: {sorted(DEFAULT_PARAM_GRIDS)}")
    if args.data == "synthetic":
        candles = synthetic_candles(n=args.bars)
    else:
        candles = load_csv(args.data)
    report = tune(
        candles,
        strategy,
        train_frac=args.train_frac,
        initial_cash=config.initial_cash,
        evaluate_top=args.top,
    )
    print(report.summary(top=args.top))
    print(
        "\nJudge candidates by the TEST columns; strong train + weak test = overfit.\n"
        "Copy winning params into config.toml [strategy_params] and [risk]."
    )


def cmd_fetch(args: argparse.Namespace) -> None:
    config = _load(args)
    symbol = args.symbol or config.symbol
    candles = fetch_ohlcv(config.exchange, symbol, config.timeframe, args.bars)
    save_csv(candles, args.out)
    print(f"saved {len(candles)} candles for {symbol} to {args.out}")


def cmd_run(args: argparse.Namespace) -> None:
    config = _load(args)
    strategy = build_strategy(config.strategy, config.strategy_params)
    risk = RiskManager(config.risk)
    if args.mode == "live":
        if not (args.live and args.yes_i_understand):
            sys.exit(
                "Refusing to trade real money without BOTH --live and "
                "--yes-i-understand. Run paper mode first and read the README."
            )
        broker = CcxtBroker(config.exchange, config.symbol, risk, live=True)
        print(f"LIVE trading {config.symbol} on {config.exchange} — Ctrl-C to stop")
    else:
        broker = PaperBroker(config.initial_cash, risk)
        print(f"paper trading {config.symbol} on {config.exchange} — Ctrl-C to stop")
    engine = TradingEngine(
        config.exchange, config.symbol, config.timeframe, strategy, risk, broker
    )
    try:
        engine.run()
    except KeyboardInterrupt:
        account = broker.account
        price = engine.candles[-1].close if engine.candles else 0.0
        print(f"\nstopped. cash={account.cash:.2f} equity={account.equity(price):.2f}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="trading_bot", description=__doc__)
    parser.add_argument("--config", default="config.toml", help="path to TOML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_backtest = sub.add_parser("backtest", help="run a backtest")
    p_backtest.add_argument(
        "--data",
        default="synthetic",
        help="'synthetic', 'exchange', or a path to a candles CSV",
    )
    p_backtest.add_argument("--bars", type=int, default=500)
    p_backtest.set_defaults(func=cmd_backtest)

    p_tune = sub.add_parser("tune", help="grid-search params with a train/test holdout")
    p_tune.add_argument("--data", default="synthetic", help="'synthetic' or a candles CSV path")
    p_tune.add_argument("--strategy", default=None, help="strategy to tune (default: from config)")
    p_tune.add_argument("--bars", type=int, default=1000, help="bars when using synthetic data")
    p_tune.add_argument("--train-frac", type=float, default=0.7)
    p_tune.add_argument("--top", type=int, default=5, help="candidates to evaluate on the holdout")
    p_tune.set_defaults(func=cmd_tune)

    p_fetch = sub.add_parser("fetch", help="download candles to CSV (needs ccxt)")
    p_fetch.add_argument("--symbol", default=None)
    p_fetch.add_argument("--bars", type=int, default=500)
    p_fetch.add_argument("--out", default="candles.csv")
    p_fetch.set_defaults(func=cmd_fetch)

    p_paper = sub.add_parser("paper", help="paper trade with live market data")
    p_paper.set_defaults(func=cmd_run, mode="paper", live=False, yes_i_understand=False)

    p_live = sub.add_parser("live", help="trade real money (requires explicit flags)")
    p_live.add_argument("--live", action="store_true")
    p_live.add_argument("--yes-i-understand", action="store_true")
    p_live.set_defaults(func=cmd_run, mode="live")

    p_list = sub.add_parser("strategies", help="list available strategies")
    p_list.set_defaults(func=lambda a: print("\n".join(sorted(STRATEGIES))))

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
