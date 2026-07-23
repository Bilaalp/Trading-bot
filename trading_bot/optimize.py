"""Parameter tuning with an out-of-sample holdout.

Grid search fits parameters on the TRAIN slice only; the TEST slice is
touched exactly once per candidate for reporting. Judge every candidate by
its TEST columns — a great train score with a poor test score is the
signature of overfitting, and picking parameters by their test score turns
the holdout into a second training set.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from .backtest import BacktestResult, run_backtest
from .models import Candle
from .risk import RiskConfig, RiskManager
from .strategy import build_strategy

# Default search grids per strategy. fast=1 makes the crossover a classic
# "price vs N-bar SMA" regime filter.
DEFAULT_PARAM_GRIDS: dict[str, dict[str, list]] = {
    "sma_crossover": {
        "fast": [1, 5, 10, 20, 50],
        "slow": [20, 50, 100, 150, 200],
    },
    "rsi_mean_reversion": {
        "period": [7, 14, 21],
        "oversold": [20.0, 25.0, 30.0],
        "exit_level": [50.0, 60.0, 70.0],
    },
}

# Risk variants searched alongside strategy params. Sizing is "fully
# deployed": risk_per_trade equals the stop distance, so each entry puts
# ~100% of equity to work — the only sizing that can compete with a
# buy-and-hold benchmark, which is 100% deployed by definition.
DEFAULT_RISK_GRID: list[dict] = [
    {"risk_per_trade": stop, "stop_loss_pct": stop, "max_position_pct": 1.0,
     "max_drawdown_pct": 0.5, "fee_pct": 0.001}
    for stop in (0.05, 0.10, 0.20)
]


@dataclass
class Candidate:
    params: dict
    risk_params: dict
    train: BacktestResult
    test: BacktestResult | None = None

    def row(self) -> str:
        cells = [
            f"{self.params}",
            f"stop={self.risk_params['stop_loss_pct']:.0%}",
            f"train {self.train.total_return_pct:+9.1f}% (B&H {self.train.buy_hold_return_pct:+9.1f}%)",
        ]
        if self.test:
            cells.append(
                f"test {self.test.total_return_pct:+8.1f}% "
                f"(B&H {self.test.buy_hold_return_pct:+8.1f}%) "
                f"sharpe {self.test.sharpe:.2f} dd {self.test.max_drawdown_pct:.0f}%"
            )
        return "  ".join(cells)


@dataclass
class TuneReport:
    strategy: str
    train_bars: int
    test_bars: int
    candidates: list[Candidate] = field(default_factory=list)  # sorted by train score

    def summary(self, top: int = 5) -> str:
        lines = [
            f"strategy: {self.strategy}  train: {self.train_bars} bars  test: {self.test_bars} bars",
            f"top {min(top, len(self.candidates))} of {len(self.candidates)} candidates by TRAIN return "
            "(judge by the TEST columns):",
        ]
        lines += [c.row() for c in self.candidates[:top]]
        return "\n".join(lines)


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*grid.values())]


def split_candles(candles: list[Candle], train_frac: float) -> tuple[list[Candle], list[Candle]]:
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(len(candles) * train_frac)
    return candles[:cut], candles[cut:]


def tune(
    candles: list[Candle],
    strategy_name: str,
    param_grid: dict[str, list] | None = None,
    risk_grid: list[dict] | None = None,
    train_frac: float = 0.7,
    initial_cash: float = 10_000.0,
    evaluate_top: int = 5,
) -> TuneReport:
    """Grid-search on the train slice; run the top candidates on the holdout.

    Only the best `evaluate_top` candidates (by train return) are evaluated
    on the test slice, to limit how much the holdout gets looked at.
    """
    param_grid = param_grid or DEFAULT_PARAM_GRIDS[strategy_name]
    risk_grid = risk_grid or DEFAULT_RISK_GRID
    train, test = split_candles(candles, train_frac)

    candidates: list[Candidate] = []
    for params in expand_grid(param_grid):
        try:
            build_strategy(strategy_name, params)  # validate combo (e.g. fast < slow)
        except (ValueError, TypeError):
            continue
        for risk_params in risk_grid:
            result = run_backtest(
                train,
                build_strategy(strategy_name, params),
                RiskManager(RiskConfig(**risk_params)),
                initial_cash,
            )
            candidates.append(Candidate(params, risk_params, result))

    candidates.sort(key=lambda c: c.train.total_return_pct, reverse=True)
    for candidate in candidates[:evaluate_top]:
        candidate.test = run_backtest(
            test,
            build_strategy(strategy_name, candidate.params),
            RiskManager(RiskConfig(**candidate.risk_params)),
            initial_cash,
        )
    return TuneReport(
        strategy=strategy_name,
        train_bars=len(train),
        test_bars=len(test),
        candidates=candidates,
    )
