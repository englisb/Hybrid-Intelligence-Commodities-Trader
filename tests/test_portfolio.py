"""
Tests for the Portfolio class.
"""

from __future__ import annotations

import pytest

from src.execution.portfolio import Portfolio, Position


def make_position(
    symbol: str = "XAU",
    qty: float = 10.0,
    entry: float = 100.0,
    stop: float = 90.0,
    pos_type: str = "long",
) -> Position:
    return Position(
        symbol=symbol,
        quantity=qty,
        entry_price=entry,
        position_type=pos_type,
        stop_loss=stop,
    )


class TestPortfolioInit:
    def test_initial_capital_set(self):
        p = Portfolio(50_000.0)
        assert p.cash == 50_000.0
        assert p.initial_capital == 50_000.0
        assert p.peak_value == 50_000.0

    def test_invalid_capital_raises(self):
        with pytest.raises(ValueError):
            Portfolio(0.0)
        with pytest.raises(ValueError):
            Portfolio(-1.0)


class TestPositionNotional:
    def test_notional_value(self):
        pos = make_position(qty=5.0, entry=200.0)
        assert pos.notional_value == pytest.approx(1000.0)

    def test_pnl_long(self):
        pos = make_position(qty=10.0, entry=100.0, pos_type="long")
        assert pos.current_pnl(110.0) == pytest.approx(100.0)
        assert pos.current_pnl(90.0) == pytest.approx(-100.0)

    def test_pnl_short(self):
        pos = make_position(qty=10.0, entry=100.0, pos_type="short")
        assert pos.current_pnl(90.0) == pytest.approx(100.0)
        assert pos.current_pnl(110.0) == pytest.approx(-100.0)


class TestOpenPosition:
    def test_open_deducts_cash(self):
        p = Portfolio(10_000.0)
        pos = make_position(qty=10.0, entry=100.0)  # notional = 1000
        p.open_position(pos)
        assert p.cash == pytest.approx(9_000.0)

    def test_open_duplicate_raises(self):
        p = Portfolio(10_000.0)
        p.open_position(make_position())
        with pytest.raises(ValueError, match="already open"):
            p.open_position(make_position())

    def test_insufficient_cash_raises(self):
        p = Portfolio(100.0)
        with pytest.raises(ValueError, match="Insufficient cash"):
            p.open_position(make_position(qty=1000.0, entry=1000.0))


class TestClosePosition:
    def test_close_credits_cash_and_pnl(self):
        p = Portfolio(10_000.0)
        pos = make_position(qty=10.0, entry=100.0)
        p.open_position(pos)
        pnl = p.close_position("XAU", 110.0)
        assert pnl == pytest.approx(100.0)
        assert p.cash == pytest.approx(10_100.0)

    def test_close_nonexistent_raises(self):
        p = Portfolio(10_000.0)
        with pytest.raises(KeyError):
            p.close_position("XAG", 100.0)

    def test_close_records_trade_history(self):
        p = Portfolio(10_000.0)
        p.open_position(make_position(qty=5.0, entry=200.0))
        p.close_position("XAU", 220.0)
        assert len(p.trade_history) == 1
        assert p.trade_history[0]["realised_pnl"] == pytest.approx(100.0)

    def test_peak_value_updated_after_close(self):
        p = Portfolio(10_000.0)
        p.open_position(make_position(qty=10.0, entry=100.0))
        p.close_position("XAU", 200.0)
        assert p.peak_value == pytest.approx(11_000.0)


class TestCloseAllPositions:
    def test_close_all(self):
        p = Portfolio(20_000.0)
        p.open_position(make_position("XAU", 10.0, 100.0))
        p.open_position(make_position("XAG", 20.0, 50.0, pos_type="long"))
        total_pnl = p.close_all_positions({"XAU": 110.0, "XAG": 55.0})
        assert len(p.positions) == 0
        assert total_pnl == pytest.approx(10 * 10.0 + 20 * 5.0)


class TestEquityAndDrawdown:
    def test_total_equity_no_positions(self):
        p = Portfolio(10_000.0)
        assert p.total_equity() == pytest.approx(10_000.0)

    def test_total_equity_with_unrealised_pnl(self):
        p = Portfolio(10_000.0)
        p.open_position(make_position(qty=10.0, entry=100.0))
        # cash = 9000, unrealised = (110 - 100) * 10 = 100
        assert p.total_equity({"XAU": 110.0}) == pytest.approx(9_100.0)

    def test_drawdown_from_peak(self):
        p = Portfolio(10_000.0)
        assert p.drawdown_from_peak(10_000.0) == pytest.approx(0.0)
        assert p.drawdown_from_peak(9_000.0) == pytest.approx(0.10)
        assert p.drawdown_from_peak(5_000.0) == pytest.approx(0.50)

    def test_allocation_fraction(self):
        p = Portfolio(10_000.0)
        p.open_position(make_position(qty=10.0, entry=100.0))  # notional = 1000
        # After opening: cash = 9000, notional = 1000, total_equity = 9000
        # allocated / total_equity = 1000 / 9000 ≈ 0.1111
        assert p.allocation_fraction() == pytest.approx(1000.0 / 9000.0)
