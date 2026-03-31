"""
Monte Carlo simulation engine.

Runs ``config.MONTE_CARLO_RUNS`` (minimum 1,000) variations of a backtest
by randomly sampling slippage and fee variables from plausible distributions.
This ensures that the strategy's performance is due to signal quality and not
a lucky parameter combination.

The simulation produces confidence intervals for:
  - Net PnL
  - Win rate
  - Profit factor
  - Maximum drawdown
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
from src.backtesting.backtester import run_backtest, BacktestResult

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Aggregated statistics over all Monte Carlo simulation runs."""

    strategy_name: str
    num_runs: int
    window_start: str
    window_end: str
    net_pnl_mean: float
    net_pnl_std: float
    net_pnl_p10: float   # 10th percentile
    net_pnl_p90: float   # 90th percentile
    win_rate_mean: float
    win_rate_std: float
    profit_factor_mean: float
    profit_factor_std: float
    profit_factor_p10: float
    max_drawdown_mean: float
    max_drawdown_p90: float   # Worst-case drawdown (90th percentile)
    pct_runs_profitable: float   # Fraction of runs ending in profit
    pct_runs_meeting_profit_factor: float
    all_results: list[BacktestResult] = field(default_factory=list, repr=False)

    @property
    def confidence_level_met(self) -> bool:
        """True if ≥80% of runs are profitable (config.CONFIDENCE_LEVEL)."""
        return self.pct_runs_profitable >= config.CONFIDENCE_LEVEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "num_runs": self.num_runs,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "net_pnl_mean": round(self.net_pnl_mean, 2),
            "net_pnl_std": round(self.net_pnl_std, 2),
            "net_pnl_p10": round(self.net_pnl_p10, 2),
            "net_pnl_p90": round(self.net_pnl_p90, 2),
            "win_rate_mean": round(self.win_rate_mean, 4),
            "win_rate_std": round(self.win_rate_std, 4),
            "profit_factor_mean": round(self.profit_factor_mean, 4),
            "profit_factor_std": round(self.profit_factor_std, 4),
            "profit_factor_p10": round(self.profit_factor_p10, 4),
            "max_drawdown_mean": round(self.max_drawdown_mean, 4),
            "max_drawdown_p90": round(self.max_drawdown_p90, 4),
            "pct_runs_profitable": round(self.pct_runs_profitable, 4),
            "pct_runs_meeting_profit_factor": round(
                self.pct_runs_meeting_profit_factor, 4
            ),
            "confidence_level_met": self.confidence_level_met,
        }


def run_monte_carlo(
    strategy_name: str,
    signals: list[dict[str, Any]],
    window_start: str,
    window_end: str,
    initial_capital: float = 100_000.0,
    num_runs: int | None = None,
    slippage_mean: float = 0.001,
    slippage_std: float = 0.0005,
    fee_mean: float = 5.0,
    fee_std: float = 2.0,
    seed: int | None = None,
) -> MonteCarloResult:
    """
    Run Monte Carlo stress tests on a strategy.

    Parameters
    ----------
    strategy_name:
        Human-readable name for logging / reporting.
    signals:
        Pre-generated signal list from any strategy module.
    window_start, window_end:
        ISO date strings defining the historical window.
    initial_capital:
        Starting portfolio value in USD.
    num_runs:
        Number of simulation runs.  Defaults to ``config.MONTE_CARLO_RUNS``
        (minimum 1,000).
    slippage_mean:
        Mean one-way slippage fraction (e.g. 0.001 = 0.1%).
    slippage_std:
        Standard deviation for slippage sampling.
    fee_mean:
        Mean round-trip fee in USD.
    fee_std:
        Standard deviation for fee sampling.
    seed:
        Random seed for reproducibility in tests.

    Returns
    -------
    :class:`MonteCarloResult` with statistical summaries.
    """
    n = max(num_runs or config.MONTE_CARLO_RUNS, 1)
    rng = np.random.default_rng(seed)

    slippages = rng.normal(slippage_mean, slippage_std, n).clip(0, 0.05)
    fees = rng.normal(fee_mean, fee_std, n).clip(0, 100)

    results: list[BacktestResult] = []
    for i in range(n):
        result = run_backtest(
            strategy_name=strategy_name,
            signals=signals,
            window_start=window_start,
            window_end=window_end,
            initial_capital=initial_capital,
            slippage=float(slippages[i]),
            fee_per_trade=float(fees[i]),
        )
        results.append(result)
        if (i + 1) % max(1, n // 10) == 0:
            logger.debug("Monte Carlo progress: %d/%d runs complete", i + 1, n)

    net_pnls = np.array([r.net_pnl for r in results])
    win_rates = np.array([r.win_rate for r in results])
    profit_factors = np.array(
        [min(r.profit_factor, 100.0) for r in results]  # cap inf at 100
    )
    drawdowns = np.array([r.max_drawdown for r in results])

    pct_profitable = float(np.mean(net_pnls > 0))
    pct_meeting_pf = float(
        np.mean(profit_factors >= config.MIN_PROFIT_FACTOR)
    )

    mc_result = MonteCarloResult(
        strategy_name=strategy_name,
        num_runs=n,
        window_start=window_start,
        window_end=window_end,
        net_pnl_mean=float(np.mean(net_pnls)),
        net_pnl_std=float(np.std(net_pnls)),
        net_pnl_p10=float(np.percentile(net_pnls, 10)),
        net_pnl_p90=float(np.percentile(net_pnls, 90)),
        win_rate_mean=float(np.mean(win_rates)),
        win_rate_std=float(np.std(win_rates)),
        profit_factor_mean=float(np.mean(profit_factors)),
        profit_factor_std=float(np.std(profit_factors)),
        profit_factor_p10=float(np.percentile(profit_factors, 10)),
        max_drawdown_mean=float(np.mean(drawdowns)),
        max_drawdown_p90=float(np.percentile(drawdowns, 90)),
        pct_runs_profitable=pct_profitable,
        pct_runs_meeting_profit_factor=pct_meeting_pf,
        all_results=results,
    )

    logger.info(
        "Monte Carlo complete for '%s': %d runs, %.1f%% profitable, "
        "mean profit factor=%.2f, 80%% confidence met=%s",
        strategy_name,
        n,
        pct_profitable * 100,
        mc_result.profit_factor_mean,
        mc_result.confidence_level_met,
    )
    return mc_result
