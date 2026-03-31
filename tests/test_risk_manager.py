"""
Tests for the RiskManager – validates all three hard-coded constraints:
  1. Capital allocation limit (30%)
  2. 50% Kill Switch
  3. Defensive Exit countdown / cancellation
"""

from __future__ import annotations

import time
import threading
from unittest.mock import MagicMock, patch

import pytest

import config
from src.execution.portfolio import Portfolio, Position
from src.execution.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_portfolio(capital: float = 100_000.0) -> Portfolio:
    return Portfolio(initial_capital=capital)


def make_risk_manager(portfolio: Portfolio, **kwargs) -> RiskManager:
    return RiskManager(portfolio, **kwargs)


def open_long(portfolio: Portfolio, symbol: str, notional: float, stop_pct: float = 0.95) -> None:
    """Open a long position with the given notional value."""
    entry = 100.0
    qty = notional / entry
    pos = Position(
        symbol=symbol,
        quantity=qty,
        entry_price=entry,
        position_type="long",
        stop_loss=entry * stop_pct,
    )
    portfolio.open_position(pos)


# ---------------------------------------------------------------------------
# Constraint 1: Capital allocation
# ---------------------------------------------------------------------------

class TestCapitalAllocation:
    def test_within_limit_accepted(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        # 25% of 100_000 = 25_000 → within 30% limit
        assert rm.check_allocation(25_000.0) is True

    def test_at_limit_accepted(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        # Exactly 30% → should be accepted
        assert rm.check_allocation(30_000.0) is True

    def test_over_limit_rejected(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        # 35% → exceeds 30% limit
        assert rm.check_allocation(35_000.0) is False

    def test_existing_positions_counted(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        # Open a 20% position first
        open_long(portfolio, "XAU", 20_000.0)
        # Adding another 15% would total 35% → reject
        assert rm.check_allocation(15_000.0) is False

    def test_zero_equity_returns_false(self):
        portfolio = Portfolio(initial_capital=1.0)
        # Drain cash manually
        portfolio._cash = 0.0
        portfolio._peak_value = 0.0
        rm = make_risk_manager(portfolio)
        assert rm.check_allocation(1.0) is False


# ---------------------------------------------------------------------------
# Constraint 2: Kill Switch
# ---------------------------------------------------------------------------

class TestKillSwitch:
    def test_no_trigger_below_threshold(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        open_long(portfolio, "XAU", 30_000.0)
        # Loss of 20% (price drops from 100 → 80)
        prices = {"XAU": 80.0}
        assert rm.check_kill_switch(prices) is False
        assert "XAU" in portfolio.positions  # position still open

    def test_triggers_at_50pct_drawdown(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        open_long(portfolio, "XAU", 30_000.0)
        # Price drops so that equity = 50% of peak → trigger
        # notional = 30_000, entry = 100; qty = 300
        # cash = 70_000; need total equity = 50_000
        # total_equity = cash + pnl = 70_000 + (price - 100) * 300 = 50_000
        # (price - 100) * 300 = -20_000 → price = 33.33
        prices = {"XAU": 33.33}
        assert rm.check_kill_switch(prices) is True
        assert "XAU" not in portfolio.positions  # liquidated

    def test_kill_switch_callback_invoked(self):
        callback = MagicMock()
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio, on_kill_switch=callback)
        open_long(portfolio, "XAU", 30_000.0)
        prices = {"XAU": 10.0}
        rm.check_kill_switch(prices)
        callback.assert_called_once()

    def test_kill_switch_not_called_when_not_triggered(self):
        callback = MagicMock()
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio, on_kill_switch=callback)
        prices = {}
        rm.check_kill_switch(prices)
        callback.assert_not_called()


# ---------------------------------------------------------------------------
# Constraint 3: Defensive Exit
# ---------------------------------------------------------------------------

class TestDefensiveExit:
    def test_cancel_defensive_exit_before_timeout(self):
        portfolio = make_portfolio(100_000.0)
        open_long(portfolio, "XAU", 10_000.0)
        rm = RiskManager(portfolio, defensive_exit_timeout=10)  # 10s timeout

        sentiment = MagicMock()
        sentiment.level = 5

        rm.start_defensive_exit_countdown(sentiment)
        # Cancel immediately
        assert rm.cancel_defensive_exit() is True
        # Wait a bit and confirm position still open
        time.sleep(0.1)
        assert "XAU" in portfolio.positions

    def test_cancel_when_no_countdown_returns_false(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        assert rm.cancel_defensive_exit() is False

    def test_defensive_exit_executes_after_timeout(self):
        executed = threading.Event()

        def on_exit(portfolio, prices, sentiment):
            executed.set()

        portfolio = make_portfolio(100_000.0)
        open_long(portfolio, "XAU", 10_000.0)
        rm = RiskManager(portfolio, on_defensive_exit=on_exit, defensive_exit_timeout=1)

        sentiment = MagicMock()
        sentiment.level = 5
        rm.start_defensive_exit_countdown(sentiment)

        # Wait up to 3 seconds for the exit to fire
        assert executed.wait(timeout=3.0), "Defensive Exit did not execute in time"
        assert "XAU" not in portfolio.positions

    def test_duplicate_countdown_is_ignored(self):
        portfolio = make_portfolio(100_000.0)
        rm = RiskManager(portfolio, defensive_exit_timeout=60)
        sentiment = MagicMock()
        rm.start_defensive_exit_countdown(sentiment)
        # Second call while first is active → should be a no-op
        rm.start_defensive_exit_countdown(sentiment)
        rm.cancel_defensive_exit()


# ---------------------------------------------------------------------------
# Stop-loss tightening
# ---------------------------------------------------------------------------

class TestTightenStopLosses:
    def test_tighten_long_stop_loss(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        # entry=100, stop=90 → gap=10; tighten_factor=0.5 → new_stop=95
        pos = Position(
            symbol="XAU",
            quantity=100,
            entry_price=100.0,
            position_type="long",
            stop_loss=90.0,
        )
        portfolio.open_position(pos)
        sentiment = MagicMock()
        sentiment.level = 4
        rm.tighten_stop_losses(sentiment, tighten_factor=0.5)
        updated_stop = portfolio.positions["XAU"].stop_loss
        assert updated_stop == pytest.approx(95.0)

    def test_tighten_short_stop_loss(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        # Short: entry=100, stop=110 → gap=10; tighten_factor=0.5 → new_stop=105
        pos = Position(
            symbol="XAG",
            quantity=50,
            entry_price=100.0,
            position_type="short",
            stop_loss=110.0,
        )
        portfolio.open_position(pos)
        sentiment = MagicMock()
        sentiment.level = 4
        rm.tighten_stop_losses(sentiment, tighten_factor=0.5)
        updated_stop = portfolio.positions["XAG"].stop_loss
        assert updated_stop == pytest.approx(105.0)

    def test_no_positions_does_not_raise(self):
        portfolio = make_portfolio(100_000.0)
        rm = make_risk_manager(portfolio)
        sentiment = MagicMock()
        sentiment.level = 4
        rm.tighten_stop_losses(sentiment)  # Should not raise
