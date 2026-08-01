import pytest

from trading_bot.data import synthetic_candles
from trading_bot.optimize import expand_grid, split_candles, tune


def test_expand_grid():
    grid = {"a": [1, 2], "b": ["x", "y", "z"]}
    combos = expand_grid(grid)
    assert len(combos) == 6
    assert {"a": 1, "b": "x"} in combos and {"a": 2, "b": "z"} in combos


def test_split_candles_fractions():
    candles = synthetic_candles(100, seed=1)
    train, test = split_candles(candles, 0.7)
    assert len(train) == 70 and len(test) == 30
    assert train[-1].timestamp < test[0].timestamp
    with pytest.raises(ValueError):
        split_candles(candles, 1.5)


def test_tune_ranks_by_train_and_evaluates_top_on_holdout():
    candles = synthetic_candles(400, seed=5)
    report = tune(
        candles,
        "sma_crossover",
        param_grid={"fast": [3, 5], "slow": [10, 20]},
        risk_grid=[{"stop_loss_pct": 0.10, "risk_per_trade": 0.10,
                    "max_position_pct": 1.0, "max_drawdown_pct": 0.5}],
        train_frac=0.75,
        evaluate_top=2,
    )
    assert report.train_bars == 300 and report.test_bars == 100
    assert len(report.candidates) == 4
    returns = [c.train.total_return_pct for c in report.candidates]
    assert returns == sorted(returns, reverse=True)
    assert all(c.test is not None for c in report.candidates[:2])
    assert all(c.test is None for c in report.candidates[2:])
    assert "TEST" in report.summary()


def test_tune_skips_invalid_param_combos():
    candles = synthetic_candles(200, seed=2)
    # fast=30/slow=20 is invalid (fast >= slow) and must be skipped, not crash.
    report = tune(
        candles,
        "sma_crossover",
        param_grid={"fast": [5, 30], "slow": [20]},
        risk_grid=[{"stop_loss_pct": 0.10, "risk_per_trade": 0.10,
                    "max_position_pct": 1.0, "max_drawdown_pct": 0.5}],
        evaluate_top=1,
    )
    assert len(report.candidates) == 1
    assert report.candidates[0].params == {"fast": 5, "slow": 20}


def test_backtest_reports_benchmark_fields():
    from trading_bot.backtest import run_backtest
    from trading_bot.risk import RiskConfig, RiskManager
    from trading_bot.strategy import SmaCrossover

    candles = synthetic_candles(300, seed=9)
    result = run_backtest(candles, SmaCrossover(), RiskManager(RiskConfig()), 10_000)
    expected_bh = (candles[-1].close / candles[0].close - 1) * 100
    assert result.buy_hold_return_pct == pytest.approx(expected_bh)
    # hourly synthetic candles -> ~8760 bars/year
    assert result.bars_per_year == pytest.approx(8760, rel=0.01)
    assert "Buy & hold" in result.summary()
