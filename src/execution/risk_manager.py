"""
Risk manager – hard-coded safety constraints.

Enforces the three core constraints from the problem statement:

1. **Capital Allocation Limit** – No more than 30% of total liquid equity
   may be allocated to open positions at any time.

2. **50% Kill Switch** – If the portfolio value drops 50% from its historical
   peak, all positions are immediately liquidated.

3. **Defensive Exit** – When a Level-5 geopolitical event is detected, a
   5-minute countdown starts.  If the human operator does not cancel within
   that window, the system closes all high-risk options positions.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable

import config
from src.execution.portfolio import Portfolio

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces hard-coded risk constraints on the portfolio.

    Parameters
    ----------
    portfolio:
        The :class:`~src.execution.portfolio.Portfolio` being managed.
    on_kill_switch:
        Optional callback invoked when the kill switch fires.
        Receives ``(portfolio, current_prices)`` as arguments.
    on_defensive_exit:
        Optional callback invoked when the Defensive Exit fires.
        Receives ``(portfolio, current_prices, sentiment_result)`` as arguments.
    defensive_exit_timeout:
        Seconds to wait for human intervention before executing Defensive
        Exit. Defaults to ``config.DEFENSIVE_EXIT_TIMEOUT_SECONDS``.
    """

    def __init__(
        self,
        portfolio: Portfolio,
        on_kill_switch: Callable[..., None] | None = None,
        on_defensive_exit: Callable[..., None] | None = None,
        defensive_exit_timeout: int | None = None,
    ) -> None:
        self._portfolio = portfolio
        self._on_kill_switch = on_kill_switch
        self._on_defensive_exit = on_defensive_exit
        self._timeout = (
            defensive_exit_timeout
            if defensive_exit_timeout is not None
            else config.DEFENSIVE_EXIT_TIMEOUT_SECONDS
        )
        self._defensive_exit_active: bool = False
        self._defensive_exit_cancelled: bool = False
        self._defensive_exit_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Constraint 1: Capital allocation check
    # ------------------------------------------------------------------

    def check_allocation(
        self,
        proposed_notional: float,
        current_prices: dict[str, float] | None = None,
    ) -> bool:
        """
        Return *True* if adding *proposed_notional* would stay within the
        30% capital allocation limit.

        Parameters
        ----------
        proposed_notional:
            Notional value (USD) of the trade being considered.
        current_prices:
            Current market prices for valuing existing positions.
        """
        equity = self._portfolio.total_equity(current_prices)
        if equity <= 0:
            return False
        new_allocated = self._portfolio.allocated_notional() + proposed_notional
        fraction = new_allocated / equity
        within_limit = fraction <= config.MAX_CAPITAL_ALLOCATION
        if not within_limit:
            logger.warning(
                "Capital allocation check FAILED: %.1f%% would exceed limit of %.1f%%",
                fraction * 100,
                config.MAX_CAPITAL_ALLOCATION * 100,
            )
        return within_limit

    # ------------------------------------------------------------------
    # Constraint 2: Kill switch
    # ------------------------------------------------------------------

    def check_kill_switch(self, current_prices: dict[str, float]) -> bool:
        """
        Check whether the 50% drawdown kill switch should fire.

        If the drawdown from peak exceeds ``config.KILL_SWITCH_DRAWDOWN``,
        ALL positions are liquidated immediately.

        Returns
        -------
        *True* if the kill switch fired, *False* otherwise.
        """
        equity = self._portfolio.total_equity(current_prices)
        drawdown = self._portfolio.drawdown_from_peak(equity)
        if drawdown >= config.KILL_SWITCH_DRAWDOWN:
            logger.critical(
                "KILL SWITCH TRIGGERED: drawdown %.1f%% >= limit %.1f%%. "
                "Liquidating ALL positions.",
                drawdown * 100,
                config.KILL_SWITCH_DRAWDOWN * 100,
            )
            self._portfolio.close_all_positions(current_prices)
            if self._on_kill_switch:
                self._on_kill_switch(self._portfolio, current_prices)
            return True
        return False

    # ------------------------------------------------------------------
    # Constraint 3: Defensive Exit
    # ------------------------------------------------------------------

    def start_defensive_exit_countdown(self, sentiment_result: Any) -> None:
        """
        Start the Defensive Exit countdown in a background thread.

        The human operator has ``config.DEFENSIVE_EXIT_TIMEOUT_SECONDS``
        to call :meth:`cancel_defensive_exit` before the system
        automatically closes high-risk positions.
        """
        if self._defensive_exit_active:
            logger.warning("Defensive Exit countdown already in progress.")
            return

        self._defensive_exit_active = True
        self._defensive_exit_cancelled = False

        logger.warning(
            "DEFENSIVE EXIT COUNTDOWN STARTED (%ds). "
            "Call cancel_defensive_exit() to abort.",
            self._timeout,
        )
        self._defensive_exit_thread = threading.Thread(
            target=self._run_defensive_exit_countdown,
            args=(sentiment_result,),
            daemon=True,
        )
        self._defensive_exit_thread.start()

    def cancel_defensive_exit(self) -> bool:
        """
        Cancel a pending Defensive Exit countdown (human operator override).

        Returns
        -------
        *True* if the cancellation was accepted, *False* if no countdown
        was active.
        """
        if not self._defensive_exit_active:
            logger.info("No active Defensive Exit countdown to cancel.")
            return False
        self._defensive_exit_cancelled = True
        logger.info("Defensive Exit countdown CANCELLED by operator.")
        return True

    def _run_defensive_exit_countdown(self, sentiment_result: Any) -> None:
        """Background thread: wait for human intervention, then exit if needed."""
        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            if self._defensive_exit_cancelled:
                logger.info("Defensive Exit aborted – operator intervened.")
                self._defensive_exit_active = False
                return
            time.sleep(0.5)

        if not self._defensive_exit_cancelled:
            logger.critical(
                "Defensive Exit EXECUTING: operator did not intervene within %ds.",
                self._timeout,
            )
            self._execute_defensive_exit(sentiment_result)

        self._defensive_exit_active = False

    def _execute_defensive_exit(self, sentiment_result: Any) -> None:
        """Close all positions and invoke the optional callback."""
        # Obtain best-available prices (entry prices as fallback)
        current_prices = {
            sym: pos.entry_price
            for sym, pos in self._portfolio.positions.items()
        }
        self._portfolio.close_all_positions(current_prices)
        if self._on_defensive_exit:
            self._on_defensive_exit(
                self._portfolio, current_prices, sentiment_result
            )

    # ------------------------------------------------------------------
    # Stop-loss management
    # ------------------------------------------------------------------

    def tighten_stop_losses(
        self,
        sentiment_result: Any,
        tighten_factor: float = 0.5,
    ) -> None:
        """
        Tighten stop-losses on all open positions in response to a Level-4+
        sentiment event.

        The stop-loss is moved halfway between the current stop and the
        current entry price (default *tighten_factor* = 0.5).

        Parameters
        ----------
        sentiment_result:
            The :class:`~src.agents.sentiment_classifier.SentimentResult`
            that triggered this call.
        tighten_factor:
            Fraction by which to close the gap between current stop-loss
            and entry price.  0.5 → move stop to halfway.
        """
        for symbol, pos in self._portfolio.positions.items():
            if pos.position_type == "long":
                new_stop = pos.stop_loss + (pos.entry_price - pos.stop_loss) * tighten_factor
            else:
                new_stop = pos.stop_loss - (pos.stop_loss - pos.entry_price) * tighten_factor
            self._portfolio.update_stop_loss(symbol, new_stop)
            logger.info(
                "Stop-loss tightened for %s: %.4f → %.4f (level=%d)",
                symbol,
                pos.stop_loss,
                new_stop,
                sentiment_result.level,
            )
