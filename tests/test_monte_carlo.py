"""
Tests for the Monte Carlo simulation engine.

Uses a small number of runs (10–50) for test speed; validates statistical
outputs and edge cases.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import config
from src.backtesting.monte_carlo import run_monte_carlo, MonteCarloResult
from src.strategies.hybrid import generate_signals as hybrid_sigs
from src.strategies.atr_trend import generate_signals as atr_sigs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trending_price_data(n: int = 100, base: float = 1800.0, drift: float = 0.002) -> list[dict]:
    prices = [base * (1 + drift) ** i for i in range(n)]
    return [
        {
            "date": f"2023-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
            "open": prices[i] * 0.999,
            "high": prices[i] * 1.005,
            "low": prices[i] * 0.995,
            "close": prices[i],
            "adjusted_close": prices[i],
        }
        for i in range(n)
    ]


def _flat_price_data(n: int = 100, price: float = 1800.0) -> list[dict]:
    return [
        {
            "date": f"2023-{i // 28 + 1:02d}-{i % 28 + 1:02d}",
            "open": price,
            "high": price * 1.001,
            "low": price * 0.999,
            "close": price,
            "adjusted_close": price,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# MonteCarloResult properties
# ---------------------------------------------------------------------------

class TestMonteCarloResultProperties:
    def _make_result(self, pct_profitable: float) -> MonteCarloResult:
        return MonteCarloResult(
            strategy_name="Test",
            num_runs=100,
            window_start="2023-01-01",
            window_end="2023-12-31",
            net_pnl_mean=1000.0,
            net_pnl_std=500.0,
            net_pnl_p10=-200.0,
            net_pnl_p90=3000.0,
            win_rate_mean=0.6,
            win_rate_std=0.05,
            profit_factor_mean=1.5,
            profit_factor_std=0.3,
            profit_factor_p10=1.0,
            max_drawdown_mean=0.1,
            max_drawdown_p90=0.2,
            pct_runs_profitable=pct_profitable,
            pct_runs_meeting_profit_factor=0.5,
        )

    def test_confidence_level_met_at_80pct(self):
        result = self._make_result(pct_profitable=0.80)
        assert result.confidence_level_met is True

    def test_confidence_level_not_met_below_80pct(self):
        result = self._make_result(pct_profitable=0.79)
        assert result.confidence_level_met is False

    def test_to_dict_has_required_keys(self):
        result = self._make_result(0.85)
        d = result.to_dict()
        required = [
            "strategy_name", "num_runs", "net_pnl_mean", "win_rate_mean",
            "profit_factor_mean", "max_drawdown_mean", "pct_runs_profitable",
            "confidence_level_met",
        ]
        for key in required:
            assert key in d, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def test_returns_monte_carlo_result(self):
        price_data = _trending_price_data()
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]
        mc = run_monte_carlo(
            strategy_name="Hybrid",
            signals=signals,
            window_start="2023-01-01",
            window_end="2023-12-31",
            num_runs=10,
            seed=42,
        )
        assert isinstance(mc, MonteCarloResult)
        assert mc.num_runs == 10

    def test_empty_signals_all_runs_have_zero_trades(self):
        mc = run_monte_carlo(
            strategy_name="Empty",
            signals=[],
            window_start="2023-01-01",
            window_end="2023-12-31",
            num_runs=5,
            seed=0,
        )
        assert mc.num_runs == 5
        # No trades → PnL is 0 in all runs → 0% profitable
        assert mc.pct_runs_profitable == pytest.approx(0.0)

    def test_minimum_runs_enforced(self):
        """Passing num_runs=0 should still produce at least 1 run."""
        price_data = _flat_price_data()
        signals = atr_sigs(price_data)
        mc = run_monte_carlo(
            strategy_name="ATR",
            signals=signals,
            window_start="2023-01-01",
            window_end="2023-12-31",
            num_runs=0,
            seed=1,
        )
        assert mc.num_runs >= 1

    def test_reproducibility_with_seed(self):
        price_data = _trending_price_data()
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]
        mc1 = run_monte_carlo("H", signals, "2023-01-01", "2023-12-31", num_runs=20, seed=7)
        mc2 = run_monte_carlo("H", signals, "2023-01-01", "2023-12-31", num_runs=20, seed=7)
        assert mc1.net_pnl_mean == pytest.approx(mc2.net_pnl_mean)
        assert mc1.win_rate_mean == pytest.approx(mc2.win_rate_mean)

    def test_higher_slippage_reduces_mean_pnl(self):
        price_data = _trending_price_data()
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]
        mc_low = run_monte_carlo(
            "H", signals, "2023-01-01", "2023-12-31",
            num_runs=20, slippage_mean=0.0, slippage_std=0.0, fee_mean=0.0, fee_std=0.0,
            seed=42,
        )
        mc_high = run_monte_carlo(
            "H", signals, "2023-01-01", "2023-12-31",
            num_runs=20, slippage_mean=0.01, slippage_std=0.001, fee_mean=20.0, fee_std=1.0,
            seed=42,
        )
        assert mc_low.net_pnl_mean >= mc_high.net_pnl_mean

    def test_p10_less_than_p90(self):
        price_data = _trending_price_data()
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]
        mc = run_monte_carlo(
            "H", signals, "2023-01-01", "2023-12-31", num_runs=30, seed=5
        )
        assert mc.net_pnl_p10 <= mc.net_pnl_p90

    def test_all_results_stored(self):
        price_data = _trending_price_data()
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]
        mc = run_monte_carlo(
            "H", signals, "2023-01-01", "2023-12-31", num_runs=15, seed=3
        )
        assert len(mc.all_results) == 15
