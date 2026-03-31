"""
Portfolio tracker for the Hybrid Intelligence Commodities Trader.

Tracks open positions, cash balance, and overall portfolio value.
All monetary values are in USD.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """A single open trading position."""

    symbol: str
    quantity: float          # Number of contracts / units
    entry_price: float       # Price paid per unit (USD)
    position_type: str       # "long" or "short"
    stop_loss: float         # Current stop-loss price
    take_profit: float | None = None

    @property
    def notional_value(self) -> float:
        """Current notional value at entry price (used for allocation checks)."""
        return abs(self.quantity) * self.entry_price

    def current_pnl(self, current_price: float) -> float:
        """Unrealised PnL at *current_price*."""
        if self.position_type == "long":
            return (current_price - self.entry_price) * self.quantity
        return (self.entry_price - current_price) * abs(self.quantity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "position_type": self.position_type,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "notional_value": self.notional_value,
        }


class Portfolio:
    """
    Tracks cash, open positions, and portfolio equity.

    Parameters
    ----------
    initial_capital:
        Starting cash balance in USD.
    """

    def __init__(self, initial_capital: float) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self._initial_capital = initial_capital
        self._cash: float = initial_capital
        self._positions: dict[str, Position] = {}  # symbol → Position
        self._peak_value: float = initial_capital
        self._trade_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    @property
    def initial_capital(self) -> float:
        return self._initial_capital

    @property
    def peak_value(self) -> float:
        return self._peak_value

    @property
    def trade_history(self) -> list[dict[str, Any]]:
        return list(self._trade_history)

    # ------------------------------------------------------------------
    # Equity calculations
    # ------------------------------------------------------------------

    def total_equity(self, current_prices: dict[str, float] | None = None) -> float:
        """
        Return total portfolio equity (cash + unrealised PnL).

        Parameters
        ----------
        current_prices:
            Dict mapping symbol → current price.  If omitted, positions are
            valued at their entry price (zero unrealised PnL).
        """
        prices = current_prices or {}
        unrealised = sum(
            pos.current_pnl(prices.get(pos.symbol, pos.entry_price))
            for pos in self._positions.values()
        )
        return self._cash + unrealised

    def allocated_notional(self) -> float:
        """Total notional value of all open positions."""
        return sum(pos.notional_value for pos in self._positions.values())

    def allocation_fraction(self, total_equity: float | None = None) -> float:
        """
        Fraction of total equity currently allocated to open positions.

        Parameters
        ----------
        total_equity:
            If *None*, uses :meth:`total_equity` at entry prices.
        """
        equity = total_equity if total_equity is not None else self.total_equity()
        if equity == 0:
            return 0.0
        return self.allocated_notional() / equity

    def drawdown_from_peak(self, current_equity: float) -> float:
        """
        Return the fractional drawdown from the historical peak equity.

        A value of 0.5 means the portfolio has lost 50% from its peak.
        """
        if self._peak_value == 0:
            return 0.0
        return max(0.0, (self._peak_value - current_equity) / self._peak_value)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def open_position(self, position: Position) -> None:
        """
        Add a new position to the portfolio and debit cash.

        Raises
        ------
        ValueError
            If a position for *symbol* is already open, or insufficient cash.
        """
        if position.symbol in self._positions:
            raise ValueError(
                f"Position for {position.symbol} already open. Close it first."
            )
        cost = position.notional_value
        if cost > self._cash:
            raise ValueError(
                f"Insufficient cash: need {cost:.2f}, have {self._cash:.2f}"
            )
        self._positions[position.symbol] = position
        self._cash -= cost
        logger.info(
            "Opened %s position: %s qty=%.4f entry=%.4f",
            position.position_type,
            position.symbol,
            position.quantity,
            position.entry_price,
        )

    def close_position(self, symbol: str, exit_price: float) -> float:
        """
        Close an open position and credit cash.

        Parameters
        ----------
        symbol:
            The symbol of the position to close.
        exit_price:
            The price at which the position is closed.

        Returns
        -------
        Realised PnL for the closed position.
        """
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol}")
        pos = self._positions.pop(symbol)
        pnl = pos.current_pnl(exit_price)
        proceeds = pos.notional_value + pnl
        self._cash += proceeds

        trade_record = {
            **pos.to_dict(),
            "exit_price": exit_price,
            "realised_pnl": pnl,
        }
        self._trade_history.append(trade_record)

        equity = self.total_equity()
        if equity > self._peak_value:
            self._peak_value = equity

        logger.info(
            "Closed position: %s exit=%.4f pnl=%.4f",
            symbol,
            exit_price,
            pnl,
        )
        return pnl

    def close_all_positions(self, current_prices: dict[str, float]) -> float:
        """
        Close ALL open positions at *current_prices*.

        Returns
        -------
        Total realised PnL across all closed positions.
        """
        symbols = list(self._positions.keys())
        total_pnl = 0.0
        for symbol in symbols:
            price = current_prices.get(symbol, self._positions[symbol].entry_price)
            total_pnl += self.close_position(symbol, price)
        return total_pnl

    def update_stop_loss(self, symbol: str, new_stop_loss: float) -> None:
        """Update the stop-loss for an open position."""
        if symbol not in self._positions:
            raise KeyError(f"No open position for {symbol}")
        self._positions[symbol].stop_loss = new_stop_loss
        logger.info("Updated stop-loss for %s to %.4f", symbol, new_stop_loss)

    def update_peak_value(self, current_prices: dict[str, float]) -> None:
        """Recalculate and update the peak portfolio value if equity is higher."""
        equity = self.total_equity(current_prices)
        if equity > self._peak_value:
            self._peak_value = equity
