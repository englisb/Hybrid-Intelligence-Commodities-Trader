"""
Tests for the backtesting engine.

Uses synthetic deterministic price data to produce reproducible signal
sequences and verifies that BacktestResult metrics are calculated correctly.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import config
from src.backtesting.backtester import (
    run_backtest,
    BacktestResult,
    _simulate_trades,
    _compute_max_drawdown,
)
from src.strategies.hybrid import generate_signals as hybrid_sigs
from src.strategies.atr_trend import generate_signals as atr_sigs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_trending_price_data(n: int = 100, base: float = 1800.0, drift: float = 0.002) -> list[dict]:
    """Generate a smoothly trending price series (deterministic)."""
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


def _make_flat_price_data(n: int = 100, price: float = 1800.0) -> list[dict]:
    """Generate a flat price series (no trend – no signals expected from ATR)."""
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
# BacktestResult metrics
# ---------------------------------------------------------------------------

class TestBacktestResultMetrics:
    def test_profit_factor_no_losses(self):
        result = BacktestResult(
            strategy_name="Test",
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=100_000.0,
            final_equity=110_000.0,
            total_trades=5,
            winning_trades=5,
            losing_trades=0,
            total_gross_profit=10_000.0,
            total_gross_loss=0.0,
            max_drawdown=0.0,
        )
        assert result.profit_factor == float("inf")

    def test_profit_factor_calculation(self):
        result = BacktestResult(
            strategy_name="Test",
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=100_000.0,
            final_equity=104_000.0,
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            total_gross_profit=8_000.0,
            total_gross_loss=-4_000.0,
            max_drawdown=0.05,
        )
        assert result.profit_factor == pytest.approx(2.0)

    def test_win_rate_calculation(self):
        result = BacktestResult(
            strategy_name="Test",
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=100_000.0,
            final_equity=100_000.0,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            total_gross_profit=0.0,
            total_gross_loss=0.0,
            max_drawdown=0.0,
        )
        assert result.win_rate == pytest.approx(0.70)

    def test_win_rate_no_trades(self):
        result = BacktestResult(
            strategy_name="Test",
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=100_000.0,
            final_equity=100_000.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_gross_profit=0.0,
            total_gross_loss=0.0,
            max_drawdown=0.0,
        )
        assert result.win_rate == 0.0

    def test_meets_profit_factor_target(self):
        result = BacktestResult(
            strategy_name="Test",
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=100_000.0,
            final_equity=100_000.0,
            total_trades=10,
            winning_trades=7,
            losing_trades=3,
            total_gross_profit=18_000.0,
            total_gross_loss=-10_000.0,
            max_drawdown=0.1,
        )
        assert result.profit_factor == pytest.approx(1.8)
        assert result.meets_profit_factor_target is True

    def test_net_pnl(self):
        result = BacktestResult(
            strategy_name="Test",
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=100_000.0,
            final_equity=115_000.0,
            total_trades=5,
            winning_trades=4,
            losing_trades=1,
            total_gross_profit=17_000.0,
            total_gross_loss=-2_000.0,
            max_drawdown=0.02,
        )
        assert result.net_pnl == pytest.approx(15_000.0)


# ---------------------------------------------------------------------------
# Max drawdown helper
# ---------------------------------------------------------------------------

class TestComputeMaxDrawdown:
    def test_no_trades(self):
        assert _compute_max_drawdown([], 100_000.0) == 0.0

    def test_all_winning_no_drawdown(self):
        trades = [{"pnl": 1000.0}, {"pnl": 2000.0}, {"pnl": 500.0}]
        assert _compute_max_drawdown(trades, 100_000.0) == 0.0

    def test_single_losing_trade(self):
        trades = [{"pnl": -10_000.0}]
        dd = _compute_max_drawdown(trades, 100_000.0)
        assert dd == pytest.approx(0.10)

    def test_consecutive_losses(self):
        trades = [{"pnl": -5_000.0}, {"pnl": -5_000.0}]
        dd = _compute_max_drawdown(trades, 100_000.0)
        assert dd == pytest.approx(0.10)  # 10k loss / 100k peak


# ---------------------------------------------------------------------------
# run_backtest integration
# ---------------------------------------------------------------------------

class TestRunBacktest:
    def test_returns_backtest_result(self):
        price_data = _make_trending_price_data(n=100)
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]
        result = run_backtest(
            strategy_name="Hybrid",
            signals=signals,
            window_start="2023-01-01",
            window_end="2023-12-31",
        )
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "Hybrid"
        assert result.initial_capital == 100_000.0
        assert result.final_equity > 0

    def test_no_signals_returns_unchanged_equity(self):
        result = run_backtest(
            strategy_name="Empty",
            signals=[],
            window_start="2023-01-01",
            window_end="2023-12-31",
            initial_capital=50_000.0,
        )
        assert result.total_trades == 0
        assert result.final_equity == pytest.approx(50_000.0)

    def test_to_dict_has_required_keys(self):
        price_data = _make_trending_price_data(n=100)
        signals = atr_sigs(price_data)
        result = run_backtest("ATR", signals, "2023-01-01", "2023-12-31")
        d = result.to_dict()
        for key in [
            "strategy_name", "window_start", "window_end", "initial_capital",
            "final_equity", "net_pnl", "total_trades", "win_rate",
            "profit_factor", "max_drawdown",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_slippage_reduces_final_equity(self):
        price_data = _make_trending_price_data(n=100)
        signals = [s.to_dict() for s in hybrid_sigs(price_data, sentiment_level=1)]

        result_no_slip = run_backtest("Hybrid", signals, "2023-01-01", "2023-12-31")
        result_with_slip = run_backtest(
            "Hybrid", signals, "2023-01-01", "2023-12-31", slippage=0.005
        )
        # Slippage should reduce equity (or at worst equal it)
        assert result_with_slip.final_equity <= result_no_slip.final_equity

    def test_defensive_exit_signals_no_trades(self):
        from src.strategies.hybrid import generate_signals, HybridSignal
        price_data = _make_flat_price_data(n=100)
        signals_l5 = generate_signals(price_data, sentiment_level=5)
        signal_dicts = [s.to_dict() for s in signals_l5]
        result = run_backtest("Hybrid-L5", signal_dicts, "2023-01-01", "2023-12-31")
        assert result.total_trades == 0
