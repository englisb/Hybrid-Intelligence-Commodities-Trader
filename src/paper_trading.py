"""
Paper trading simulation.

Runs a live (zero-capital-risk) simulation for a configurable duration
(default 1 week / 7 days) using real market data and the full Hybrid
strategy pipeline.  Calibrates the LLM's news-weighting responses against
real market reactions without risking real capital.

Usage::

    from src.paper_trading import PaperTrader
    trader = PaperTrader(initial_capital=100_000.0)
    trader.run(days=7)
    print(trader.summary())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import config
from src.execution.portfolio import Portfolio
from src.execution.risk_manager import RiskManager
from src.execution.order_manager import OrderManager, TradeSignal
from src.agents.sentiment_classifier import SentimentClassifier
from src.agents.llm_agent import LLMAgent
from src.data.metals_api import get_latest_prices
from src.data.alpha_vantage import get_all_equity_quotes
from src.strategies.hybrid import generate_signals as hybrid_signals

logger = logging.getLogger(__name__)


@dataclass
class PaperTradingSession:
    """Records a completed paper-trading session."""

    start_time: datetime
    end_time: datetime
    initial_capital: float
    final_equity: float
    total_cycles: int
    trade_log: list[dict[str, Any]] = field(default_factory=list)
    sentiment_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def net_pnl(self) -> float:
        return self.final_equity - self.initial_capital

    @property
    def return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return self.net_pnl / self.initial_capital

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "net_pnl": round(self.net_pnl, 2),
            "return_pct": round(self.return_pct * 100, 4),
            "total_cycles": self.total_cycles,
            "total_trades": len(self.trade_log),
            "sentiment_events": len(self.sentiment_log),
        }


class PaperTrader:
    """
    Zero-risk paper trading simulator.

    Executes the full Hybrid strategy pipeline against real market data
    without committing real capital.

    Parameters
    ----------
    initial_capital:
        Simulated starting balance in USD.
    poll_interval_seconds:
        How often to poll for new prices and news.  Default 60 s (1 min).
    classifier:
        Optional injected :class:`~src.agents.sentiment_classifier.SentimentClassifier`.
    """

    def __init__(
        self,
        initial_capital: float = 100_000.0,
        poll_interval_seconds: int = 60,
        classifier: SentimentClassifier | None = None,
    ) -> None:
        self._initial_capital = initial_capital
        self._poll_interval = poll_interval_seconds
        self._portfolio = Portfolio(initial_capital)
        self._risk_manager = RiskManager(self._portfolio)
        self._order_manager = OrderManager(self._portfolio, self._risk_manager)
        self._classifier = classifier or SentimentClassifier()
        self._agent = LLMAgent(
            classifier=self._classifier,
            risk_manager=self._risk_manager,
            poll_interval_seconds=poll_interval_seconds,
        )
        self._sentiment_log: list[dict[str, Any]] = []
        self._price_history: list[dict[str, Any]] = []

    def _fetch_and_record_prices(self) -> dict[str, float]:
        """Fetch latest metal prices and record for signal generation."""
        try:
            prices = get_latest_prices()
        except Exception as exc:
            logger.warning("Failed to fetch live prices: %s. Using last known.", exc)
            prices = (
                {row["symbol"]: row["price"] for row in self._price_history[-1:]}
                if self._price_history
                else {}
            )

        now = datetime.now(timezone.utc).isoformat()
        for symbol, price in prices.items():
            self._price_history.append(
                {"date": now, "symbol": symbol, "price": price}
            )
        return prices

    def _run_cycle(self, cycle: int) -> None:
        """Execute one monitoring / trading cycle."""
        logger.info("Paper trading cycle %d", cycle)

        # 1. Fetch prices
        prices = self._fetch_and_record_prices()

        # 2. Kill switch check
        if prices and self._risk_manager.check_kill_switch(prices):
            logger.warning("Kill switch triggered in paper trading cycle %d.", cycle)
            return

        # 3. Sentiment analysis
        sentiment = self._agent.analyse_news()
        if sentiment is not None:
            self._sentiment_log.append(sentiment.to_dict())

        # 4. Generate and execute hybrid signals if we have price data
        gold_price = prices.get(config.COMMODITIES["gold"])
        if gold_price and len(self._price_history) >= 2:
            # Build a minimal price series for signal generation
            gold_series = [
                {
                    "date": row["date"],
                    "open": row["price"],
                    "high": row["price"] * 1.001,
                    "low": row["price"] * 0.999,
                    "close": row["price"],
                    "adjusted_close": row["price"],
                }
                for row in self._price_history
                if row.get("symbol") == config.COMMODITIES["gold"]
            ]
            sentiment_level = sentiment.level if sentiment else 1
            signals = hybrid_signals(gold_series, sentiment_level=sentiment_level)
            if signals:
                latest = signals[-1]
                if latest.signal == "long":
                    signal_obj = TradeSignal(
                        symbol=config.COMMODITIES["gold"],
                        position_type="long",
                        quantity=(
                            self._portfolio.cash
                            * config.MAX_CAPITAL_ALLOCATION
                            * latest.position_size_factor
                        ) / latest.entry_price,
                        entry_price=latest.entry_price,
                        stop_loss=latest.stop_loss,
                        rationale=latest.rationale,
                    )
                    self._order_manager.submit_order(signal_obj, prices)
                elif latest.signal in ("short", "exit"):
                    self._order_manager.close_order(
                        config.COMMODITIES["gold"],
                        latest.entry_price,
                    )

    def run(self, days: int = 7) -> PaperTradingSession:
        """
        Run the paper trading simulation for *days* calendar days.

        Parameters
        ----------
        days:
            Number of calendar days to run.  Default is 7 (1 week).

        Returns
        -------
        :class:`PaperTradingSession` with session results.
        """
        start = datetime.now(timezone.utc)
        end_epoch = time.time() + days * 86_400  # convert days to seconds

        logger.info(
            "Paper trading started: capital=%.2f, duration=%d days.",
            self._initial_capital,
            days,
        )

        cycle = 0
        try:
            while time.time() < end_epoch:
                cycle += 1
                try:
                    self._run_cycle(cycle)
                except Exception as exc:
                    logger.error("Error in paper trading cycle %d: %s", cycle, exc)
                time.sleep(self._poll_interval)
        except KeyboardInterrupt:
            logger.info("Paper trading interrupted by operator.")

        # Close any open positions at last known prices
        last_prices: dict[str, float] = {}
        for row in reversed(self._price_history):
            sym = row.get("symbol", "")
            if sym and sym not in last_prices:
                last_prices[sym] = row["price"]

        if self._portfolio.positions:
            logger.info("Closing all paper positions at end of session.")
            self._portfolio.close_all_positions(last_prices)

        final_equity = self._portfolio.total_equity(last_prices)
        end = datetime.now(timezone.utc)

        session = PaperTradingSession(
            start_time=start,
            end_time=end,
            initial_capital=self._initial_capital,
            final_equity=final_equity,
            total_cycles=cycle,
            trade_log=self._portfolio.trade_history,
            sentiment_log=self._sentiment_log,
        )
        logger.info("Paper trading session complete: %s", session.to_dict())
        return session

    def summary(self) -> dict[str, Any]:
        """Return a quick summary of the current portfolio state."""
        return {
            "cash": self._portfolio.cash,
            "open_positions": len(self._portfolio.positions),
            "trade_history_count": len(self._portfolio.trade_history),
            "sentiment_events": len(self._sentiment_log),
        }
