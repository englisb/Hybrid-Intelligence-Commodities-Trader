"""
Historical backtesting engine.

Simulates a trading strategy over historical price data and calculates
performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import config

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Aggregated performance metrics for a single backtest run."""

    strategy_name: str
    window_start: str
    window_end: str
    initial_capital: float
    final_equity: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_gross_profit: float
    total_gross_loss: float
    max_drawdown: float          # Fraction, e.g. 0.25 = 25%
    trade_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def profit_factor(self) -> float:
        """Gross profit / gross loss.  Returns inf if no losing trades."""
        if self.total_gross_loss == 0:
            return float("inf")
        return self.total_gross_profit / abs(self.total_gross_loss)

    @property
    def win_rate(self) -> float:
        """Fraction of winning trades."""
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def net_pnl(self) -> float:
        return self.final_equity - self.initial_capital

    @property
    def meets_profit_factor_target(self) -> bool:
        return self.profit_factor >= config.MIN_PROFIT_FACTOR

    @property
    def meets_accuracy_target(self) -> bool:
        return self.win_rate >= config.TARGET_ACCURACY_RATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_name": self.strategy_name,
            "window_start": self.window_start,
            "window_end": self.window_end,
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "net_pnl": self.net_pnl,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "meets_profit_factor_target": self.meets_profit_factor_target,
            "meets_accuracy_target": self.meets_accuracy_target,
        }


def _simulate_trades(
    signals: list[dict[str, Any]],
    initial_capital: float,
    stop_loss_key: str = "stop_loss",
    take_profit_pct: float = 0.04,
    slippage: float = 0.0,
    fee_per_trade: float = 0.0,
) -> tuple[list[dict[str, Any]], float]:
    """
    Simulate sequential trades from a list of signals.

    Each signal triggers an open; the position is closed when:
      - The next ``exit`` / ``hold`` signal arrives, OR
      - The stop-loss is hit (simulated against the next day's low/high).

    Parameters
    ----------
    signals:
        List of signal dicts (must have ``date``, ``signal``, ``entry_price``,
        ``stop_loss`` keys).
    initial_capital:
        Starting capital in USD.
    stop_loss_key:
        Key for the stop-loss price in each signal dict.
    take_profit_pct:
        Fixed take-profit as a fraction of entry price.
    slippage:
        One-way slippage fraction applied to entry/exit prices.
    fee_per_trade:
        Fixed USD fee per trade (round-trip).

    Returns
    -------
    ``(trade_log, final_equity)`` where *trade_log* is a list of closed-trade
    dicts.
    """
    capital = initial_capital
    trade_log: list[dict[str, Any]] = []
    open_position: dict[str, Any] | None = None

    for sig in signals:
        signal = sig.get("signal", "hold")
        entry = sig["entry_price"] * (1 + slippage if signal == "long" else 1 - slippage)
        size_factor = sig.get("position_size_factor", 1.0)

        # Close open position on exit / reversal / defensive signal
        if open_position is not None:
            close_signal = signal in ("exit", "defensive_exit") or (
                open_position["type"] == "long" and signal == "short"
            ) or (
                open_position["type"] == "short" and signal == "long"
            )

            # Check stop-loss hit (using today's entry as proxy for low/high)
            stop = open_position["stop_loss"]
            stop_hit = (
                (open_position["type"] == "long" and entry <= stop) or
                (open_position["type"] == "short" and entry >= stop)
            )

            if close_signal or stop_hit:
                exit_price = stop if stop_hit else entry
                pnl_raw = (exit_price - open_position["entry_price"]) * open_position["qty"]
                if open_position["type"] == "short":
                    pnl_raw = -pnl_raw
                pnl = pnl_raw - fee_per_trade
                capital += open_position["notional"] + pnl
                trade_log.append({
                    "open_date": open_position["date"],
                    "close_date": sig["date"],
                    "type": open_position["type"],
                    "entry_price": open_position["entry_price"],
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "stop_hit": stop_hit,
                })
                open_position = None

        # Open new position
        if open_position is None and signal in ("long", "short") and capital > 0:
            notional = capital * config.MAX_CAPITAL_ALLOCATION * size_factor
            qty = notional / entry if entry > 0 else 0
            if qty > 0:
                take_profit = entry * (1 + take_profit_pct) if signal == "long" \
                    else entry * (1 - take_profit_pct)
                open_position = {
                    "date": sig["date"],
                    "type": signal,
                    "entry_price": entry,
                    "stop_loss": sig[stop_loss_key],
                    "take_profit": take_profit,
                    "qty": qty,
                    "notional": notional,
                }
                capital -= notional

    # Close any remaining open position at last entry price
    if open_position is not None:
        last_price = signals[-1]["entry_price"] if signals else open_position["entry_price"]
        pnl_raw = (last_price - open_position["entry_price"]) * open_position["qty"]
        if open_position["type"] == "short":
            pnl_raw = -pnl_raw
        pnl = pnl_raw - fee_per_trade
        capital += open_position["notional"] + pnl
        trade_log.append({
            "open_date": open_position["date"],
            "close_date": "END",
            "type": open_position["type"],
            "entry_price": open_position["entry_price"],
            "exit_price": last_price,
            "pnl": pnl,
            "stop_hit": False,
        })

    return trade_log, capital


def _compute_max_drawdown(trade_log: list[dict[str, Any]], initial_capital: float) -> float:
    """Compute max drawdown fraction from a sequential trade log."""
    equity = initial_capital
    peak = initial_capital
    max_dd = 0.0
    for trade in trade_log:
        equity += trade["pnl"]
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_backtest(
    strategy_name: str,
    signals: list[dict[str, Any]],
    window_start: str,
    window_end: str,
    initial_capital: float = 100_000.0,
    slippage: float = 0.0,
    fee_per_trade: float = 0.0,
) -> BacktestResult:
    """
    Run a backtest for a given strategy over a list of pre-generated signals.

    Parameters
    ----------
    strategy_name:
        Human-readable name of the strategy being tested.
    signals:
        Pre-generated signal list (from any of the strategy modules).
    window_start, window_end:
        ISO date strings defining the historical window.
    initial_capital:
        Starting portfolio value in USD.
    slippage:
        One-way slippage fraction (e.g. 0.001 = 0.1%).
    fee_per_trade:
        Fixed round-trip fee in USD.

    Returns
    -------
    :class:`BacktestResult` with full performance metrics.
    """
    # Normalise signals to have the expected keys
    normalised = []
    for s in signals:
        normalised.append(
            {
                "date": s.get("date", ""),
                "signal": s.get("signal", "hold"),
                "entry_price": float(s.get("entry_price", 0)),
                "stop_loss": float(s.get("stop_loss", 0)),
                "position_size_factor": float(s.get("position_size_factor", 1.0)),
            }
        )

    trade_log, final_equity = _simulate_trades(
        normalised,
        initial_capital=initial_capital,
        slippage=slippage,
        fee_per_trade=fee_per_trade,
    )

    winning = [t for t in trade_log if t["pnl"] > 0]
    losing = [t for t in trade_log if t["pnl"] <= 0]

    return BacktestResult(
        strategy_name=strategy_name,
        window_start=window_start,
        window_end=window_end,
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_trades=len(trade_log),
        winning_trades=len(winning),
        losing_trades=len(losing),
        total_gross_profit=sum(t["pnl"] for t in winning),
        total_gross_loss=sum(t["pnl"] for t in losing),
        max_drawdown=_compute_max_drawdown(trade_log, initial_capital),
        trade_log=trade_log,
    )
