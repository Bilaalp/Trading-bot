import pytest

from trading_bot.config import load_config
from trading_bot.data import load_csv, save_csv, synthetic_candles


def test_config_roundtrip(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[bot]
exchange = "kraken"
symbol = "ETH/USD"
timeframe = "4h"
strategy = "rsi_mean_reversion"
initial_cash = 5000.0

[strategy_params]
period = 10

[risk]
risk_per_trade = 0.02
max_drawdown_pct = 0.15
"""
    )
    config = load_config(path)
    assert config.exchange == "kraken"
    assert config.symbol == "ETH/USD"
    assert config.strategy == "rsi_mean_reversion"
    assert config.strategy_params == {"period": 10}
    assert config.initial_cash == 5000.0
    assert config.risk.risk_per_trade == 0.02
    assert config.risk.max_drawdown_pct == 0.15
    assert config.risk.stop_loss_pct == 0.05  # default preserved


def test_config_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("")
    config = load_config(path)
    assert config.exchange == "binance"
    assert config.strategy == "sma_crossover"


def test_config_rejects_invalid_risk(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[risk]\nrisk_per_trade = 2.0\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_csv_roundtrip(tmp_path):
    candles = synthetic_candles(20, seed=1)
    path = tmp_path / "candles.csv"
    save_csv(candles, path)
    loaded = load_csv(path)
    assert loaded == candles


def test_synthetic_is_deterministic_and_ordered():
    a = synthetic_candles(50, seed=3)
    b = synthetic_candles(50, seed=3)
    assert a == b
    assert all(x.timestamp < y.timestamp for x, y in zip(a, a[1:]))
    assert all(c.low <= min(c.open, c.close) <= max(c.open, c.close) <= c.high for c in a)
