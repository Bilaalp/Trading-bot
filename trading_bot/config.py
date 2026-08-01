"""Bot configuration loaded from a TOML file (stdlib tomllib, no deps).

See config.example.toml at the repo root. Secrets (API keys) never live in
the config file — they come from environment variables only.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .risk import RiskConfig


@dataclass
class BotConfig:
    exchange: str = "binance"
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy: str = "sma_crossover"
    strategy_params: dict = field(default_factory=dict)
    initial_cash: float = 10_000.0
    risk: RiskConfig = field(default_factory=RiskConfig)


def load_config(path: str | Path) -> BotConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    bot = raw.get("bot", {})
    risk = RiskConfig(**raw.get("risk", {}))
    return BotConfig(
        exchange=bot.get("exchange", "binance"),
        symbol=bot.get("symbol", "BTC/USDT"),
        timeframe=bot.get("timeframe", "1h"),
        strategy=bot.get("strategy", "sma_crossover"),
        strategy_params=raw.get("strategy_params", {}),
        initial_cash=float(bot.get("initial_cash", 10_000.0)),
        risk=risk,
    )
