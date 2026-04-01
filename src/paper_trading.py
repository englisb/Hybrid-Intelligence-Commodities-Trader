"""
Paper trading simulation and predictive accuracy backtesting.

**Legacy mode** – :class:`PaperTrader`
    Runs a live (zero-capital-risk) simulation for a configurable duration
    using real market data and the full Hybrid strategy pipeline.

**Backtesting mode** – :class:`PredictiveBacktester`
    Replaces the 1-week live simulation with a multi-scale walk-forward
    validation protocol that verifies prediction accuracy using historical
    data from 2000 through December 2025 (the information cutoff).

    Protocol:

    1. **5-year blocks** – Use a 5-year training window to generate up to 200
       forward predictions covering the following 3-year horizon.  Verify
       year-by-year as new data is revealed, record discrepancies, then
       advance the window by one year and repeat until ``DATA_CUTOFF``.

    2. **Weekly blocks** – Refine temporal resolution to 52-week training
       windows with 1-week verification steps, capturing short-term
       volatility patterns.

    3. **Daily blocks** – Finest resolution: 30-day training windows with
       1-day verification steps, acting as the minimum batch size for
       training data.

Usage::

    from src.paper_trading import PredictiveBacktester
    backtester = PredictiveBacktester()
    results = backtester.run()
    print(results["five_year_blocks"].to_dict())
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np

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


# ===========================================================================
# Predictive Accuracy Backtester
# ===========================================================================


def _build_price_series(
    start_date: str, end_date: str, seed: int = 42
) -> list[dict[str, Any]]:
    """
    Generate synthetic daily price data between two ISO dates.

    NOTE: Replace with real historical data from Alpha Vantage / Metals-API
    for production backtesting.
    """
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    n_days = (end - start).days + 1

    rng = np.random.default_rng(seed)
    prices = 300.0 * np.cumprod(1 + rng.normal(0.0003, 0.012, n_days))

    result: list[dict[str, Any]] = []
    for i in range(n_days):
        d = start + timedelta(days=i)
        p = float(prices[i])
        result.append(
            {
                "date": d.isoformat(),
                "open": p * 0.999,
                "high": p * 1.005,
                "low": p * 0.995,
                "close": p,
                "adjusted_close": p,
            }
        )
    return result


def _slice_by_date(
    data: list[dict[str, Any]], start: str, end: str
) -> list[dict[str, Any]]:
    """Return rows where start <= row['date'] <= end."""
    return [row for row in data if start <= row["date"] <= end]


def _add_years(date_str: str, years: int) -> str:
    """Add *years* to an ISO date string, clamping to DATA_CUTOFF."""
    d = date.fromisoformat(date_str)
    try:
        result = d.replace(year=d.year + years)
    except ValueError:
        result = d.replace(year=d.year + years, day=28)
    cutoff = date.fromisoformat(config.DATA_CUTOFF)
    return min(result, cutoff).isoformat()


def _add_weeks(date_str: str, weeks: int) -> str:
    """Add *weeks* to an ISO date string, clamping to DATA_CUTOFF."""
    result = date.fromisoformat(date_str) + timedelta(weeks=weeks)
    cutoff = date.fromisoformat(config.DATA_CUTOFF)
    return min(result, cutoff).isoformat()


def _add_days(date_str: str, days: int) -> str:
    """Add *days* to an ISO date string, clamping to DATA_CUTOFF."""
    result = date.fromisoformat(date_str) + timedelta(days=days)
    cutoff = date.fromisoformat(config.DATA_CUTOFF)
    return min(result, cutoff).isoformat()


@dataclass
class PredictionRecord:
    """Records a single forward prediction and its eventual verification."""

    training_start: str
    training_end: str
    prediction_index: int
    signal: str          # "long" or "short"
    entry_price: float
    stop_loss: float
    verified: bool = False
    predicted_correct: bool | None = None
    discrepancy_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_start": self.training_start,
            "training_end": self.training_end,
            "prediction_index": self.prediction_index,
            "signal": self.signal,
            "entry_price": round(self.entry_price, 4),
            "stop_loss": round(self.stop_loss, 4),
            "verified": self.verified,
            "predicted_correct": self.predicted_correct,
            "discrepancy_note": self.discrepancy_note,
        }


@dataclass
class VerificationResult:
    """Outcome of checking one block's predictions against a period of new data."""

    training_start: str
    training_end: str
    verification_start: str
    verification_end: str
    total_predictions: int
    correct_predictions: int
    discrepancies: list[dict[str, Any]] = field(default_factory=list)

    @property
    def accuracy_rate(self) -> float:
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_window": f"{self.training_start} \u2192 {self.training_end}",
            "verification_period": (
                f"{self.verification_start} \u2192 {self.verification_end}"
            ),
            "total_predictions": self.total_predictions,
            "correct_predictions": self.correct_predictions,
            "accuracy_rate_pct": round(self.accuracy_rate * 100, 2),
            "discrepancy_count": len(self.discrepancies),
            "discrepancies": self.discrepancies,
        }


@dataclass
class PredictiveBacktestSession:
    """Complete results from a single-phase predictive accuracy run."""

    window_type: str            # "5-year", "weekly", or "daily"
    data_start: str
    data_end: str               # Never exceeds DATA_CUTOFF
    training_window_label: str  # e.g. "5-year", "52-week", "30-day"
    predictions_per_block: int
    verification_results: list[VerificationResult] = field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        total = sum(v.total_predictions for v in self.verification_results)
        correct = sum(v.correct_predictions for v in self.verification_results)
        return correct / total if total > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_type": self.window_type,
            "data_start": self.data_start,
            "data_end": self.data_end,
            "training_window": self.training_window_label,
            "predictions_per_block": self.predictions_per_block,
            "overall_accuracy_pct": round(self.overall_accuracy * 100, 2),
            "total_verification_blocks": len(self.verification_results),
            "verification_results": [v.to_dict() for v in self.verification_results],
        }


class PredictiveBacktester:
    """
    Multi-scale predictive accuracy backtester.

    Replaces the 1-week live simulation with a walk-forward validation
    protocol that verifies prediction accuracy across three temporal scales,
    acting as a method to alter the batch size of training data.

    Phase 1 – 5-year blocks (``run_five_year_blocks``)
        Train on a 5-year window, generate up to ``PREDICTIONS_PER_BLOCK``
        (200) forward signals covering the following ``PREDICTION_HORIZON_YEARS``
        (3), then fact-check one ``VERIFICATION_STEP_YEARS`` (1) year at a
        time as new data is revealed.  Slide the window forward by one year
        and repeat until ``DATA_CUTOFF`` (December 2025).

    Phase 2 – weekly blocks (``run_weekly_blocks``)
        Refine to 52-week training windows with 1-week verification steps
        to capture short-term market volatility patterns.

    Phase 3 – daily blocks (``run_daily_blocks``)
        Finest resolution: 30-day training windows with 1-day verification
        steps — the minimum batch size for training data.

    Parameters
    ----------
    price_data:
        Optional pre-loaded price series.  If omitted, synthetic data is
        generated as a placeholder — replace with real historical API data
        for production use.
    """

    def __init__(self, price_data: list[dict[str, Any]] | None = None) -> None:
        if price_data is not None:
            self._price_data = price_data
        else:
            self._price_data = _build_price_series(
                config.PREDICTIVE_BACKTEST_START,
                config.PREDICTIVE_BACKTEST_END,
            )
        logger.info(
            "PredictiveBacktester initialised: %d price records (%s \u2192 %s).",
            len(self._price_data),
            config.PREDICTIVE_BACKTEST_START,
            config.PREDICTIVE_BACKTEST_END,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_predictions(
        self,
        training_data: list[dict[str, Any]],
        training_start: str,
        training_end: str,
    ) -> list[PredictionRecord]:
        """Generate up to PREDICTIONS_PER_BLOCK signals from training_data."""
        if not training_data:
            return []
        signals = hybrid_signals(training_data, sentiment_level=1)
        actionable = [s for s in signals if s.signal in ("long", "short")]
        selected = actionable[: config.PREDICTIONS_PER_BLOCK]
        return [
            PredictionRecord(
                training_start=training_start,
                training_end=training_end,
                prediction_index=i,
                signal=s.signal,
                entry_price=s.entry_price,
                stop_loss=s.stop_loss,
            )
            for i, s in enumerate(selected)
        ]

    @staticmethod
    def _verify_prediction(
        record: PredictionRecord,
        verification_data: list[dict[str, Any]],
    ) -> PredictionRecord:
        """
        Verify a prediction against a verification period's price movement.

        Checks whether the closing price at the end of the verification
        period moved in the direction implied by the prediction signal.
        A discrepancy is logged when the actual direction contradicts the
        prediction.
        """
        if not verification_data:
            return record

        final_price = verification_data[-1]["close"]
        if record.signal == "long":
            correct = final_price > record.entry_price
        else:
            correct = final_price < record.entry_price

        note = ""
        if not correct:
            direction = "up" if final_price > record.entry_price else "down"
            note = (
                f"Predicted {record.signal} from {record.entry_price:.2f}; "
                f"actual price moved {direction} to {final_price:.2f}."
            )
        return PredictionRecord(
            training_start=record.training_start,
            training_end=record.training_end,
            prediction_index=record.prediction_index,
            signal=record.signal,
            entry_price=record.entry_price,
            stop_loss=record.stop_loss,
            verified=True,
            predicted_correct=correct,
            discrepancy_note=note,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_five_year_blocks(self) -> PredictiveBacktestSession:
        """
        Phase 1 – 5-year training blocks.

        For each 5-year training window, generate up to 200 predictions
        for the following 3-year horizon and then verify one year at a
        time as new data is revealed.  Window slides forward by 1 year
        per iteration until ``DATA_CUTOFF``.

        Example sequence:
          * Train 2000–2004 → predict 2005–2007 → verify 2005, 2006, 2007
          * Train 2001–2005 → predict 2006–2008 → verify 2006, 2007, 2008
          * … continue until December 2025
        """
        tw = config.TRAINING_WINDOW_YEARS    # 5
        ph = config.PREDICTION_HORIZON_YEARS  # 3
        vs = config.VERIFICATION_STEP_YEARS   # 1
        cutoff = config.DATA_CUTOFF

        session = PredictiveBacktestSession(
            window_type="5-year",
            data_start=config.PREDICTIVE_BACKTEST_START,
            data_end=config.PREDICTIVE_BACKTEST_END,
            training_window_label=f"{tw}-year",
            predictions_per_block=config.PREDICTIONS_PER_BLOCK,
        )

        window_start = config.PREDICTIVE_BACKTEST_START
        while window_start < cutoff:
            training_end = _add_years(window_start, tw)
            if training_end >= cutoff:
                break

            training_data = _slice_by_date(
                self._price_data, window_start, training_end
            )
            predictions = self._generate_predictions(
                training_data, window_start, training_end
            )

            if not predictions:
                window_start = _add_years(window_start, vs)
                continue

            # Verify year-by-year for up to PREDICTION_HORIZON_YEARS
            for yr in range(ph):
                v_start = _add_years(training_end, yr)
                v_end = _add_years(training_end, yr + vs)
                if v_start > cutoff:
                    break

                v_data = _slice_by_date(
                    self._price_data, v_start, min(v_end, cutoff)
                )
                if not v_data:
                    break

                verified = [
                    self._verify_prediction(p, v_data) for p in predictions
                ]
                correct = sum(1 for p in verified if p.predicted_correct)
                discrepancies = [
                    p.to_dict() for p in verified if not p.predicted_correct
                ]

                result = VerificationResult(
                    training_start=window_start,
                    training_end=training_end,
                    verification_start=v_start,
                    verification_end=min(v_end, cutoff),
                    total_predictions=len(verified),
                    correct_predictions=correct,
                    discrepancies=discrepancies,
                )
                session.verification_results.append(result)
                logger.info(
                    "[5-year] train %s\u2192%s | verify %s\u2192%s | "
                    "%d/%d correct (%.1f%%)",
                    window_start,
                    training_end,
                    v_start,
                    result.verification_end,
                    correct,
                    len(verified),
                    result.accuracy_rate * 100,
                )

            window_start = _add_years(window_start, vs)

        return session

    def run_weekly_blocks(self) -> PredictiveBacktestSession:
        """
        Phase 2 – weekly training windows.

        Refines temporal resolution by training on a rolling 52-week
        window and verifying predictions against the following week.
        Starts from ``WEEKLY_BLOCKS_START`` to remain computationally
        tractable.
        """
        training_weeks = 52
        cutoff = config.DATA_CUTOFF

        session = PredictiveBacktestSession(
            window_type="weekly",
            data_start=config.WEEKLY_BLOCKS_START,
            data_end=config.PREDICTIVE_BACKTEST_END,
            training_window_label=f"{training_weeks}-week",
            predictions_per_block=config.PREDICTIONS_PER_BLOCK,
        )

        window_start = config.WEEKLY_BLOCKS_START
        while window_start < cutoff:
            training_end = _add_weeks(window_start, training_weeks)
            if training_end >= cutoff:
                break

            v_start = _add_days(training_end, 1)
            v_end = _add_weeks(v_start, 1)
            if v_start > cutoff:
                break

            training_data = _slice_by_date(
                self._price_data, window_start, training_end
            )
            v_data = _slice_by_date(
                self._price_data, v_start, min(v_end, cutoff)
            )
            predictions = self._generate_predictions(
                training_data, window_start, training_end
            )

            if predictions and v_data:
                verified = [
                    self._verify_prediction(p, v_data) for p in predictions
                ]
                correct = sum(1 for p in verified if p.predicted_correct)
                discrepancies = [
                    p.to_dict() for p in verified if not p.predicted_correct
                ]
                result = VerificationResult(
                    training_start=window_start,
                    training_end=training_end,
                    verification_start=v_start,
                    verification_end=min(v_end, cutoff),
                    total_predictions=len(verified),
                    correct_predictions=correct,
                    discrepancies=discrepancies,
                )
                session.verification_results.append(result)
                logger.debug(
                    "[weekly] train %s\u2192%s | verify %s\u2192%s | "
                    "%d/%d correct",
                    window_start,
                    training_end,
                    v_start,
                    result.verification_end,
                    correct,
                    len(verified),
                )

            window_start = _add_weeks(window_start, 1)

        logger.info(
            "[weekly] complete: %d blocks, overall accuracy=%.1f%%",
            len(session.verification_results),
            session.overall_accuracy * 100,
        )
        return session

    def run_daily_blocks(self) -> PredictiveBacktestSession:
        """
        Phase 3 – daily training windows.

        Finest temporal resolution: trains on a 30-day rolling window
        and verifies the next day's direction.  Acts as the minimum batch
        size for training data.  Starts from ``DAILY_BLOCKS_START``.
        """
        training_days = 30
        cutoff = config.DATA_CUTOFF

        session = PredictiveBacktestSession(
            window_type="daily",
            data_start=config.DAILY_BLOCKS_START,
            data_end=config.PREDICTIVE_BACKTEST_END,
            training_window_label=f"{training_days}-day",
            predictions_per_block=config.PREDICTIONS_PER_BLOCK,
        )

        window_start = config.DAILY_BLOCKS_START
        while window_start < cutoff:
            training_end = _add_days(window_start, training_days)
            if training_end >= cutoff:
                break

            v_start = _add_days(training_end, 1)
            v_end = _add_days(v_start, 1)
            if v_start > cutoff:
                break

            training_data = _slice_by_date(
                self._price_data, window_start, training_end
            )
            v_data = _slice_by_date(
                self._price_data, v_start, min(v_end, cutoff)
            )
            predictions = self._generate_predictions(
                training_data, window_start, training_end
            )

            if predictions and v_data:
                verified = [
                    self._verify_prediction(p, v_data) for p in predictions
                ]
                correct = sum(1 for p in verified if p.predicted_correct)
                discrepancies = [
                    p.to_dict() for p in verified if not p.predicted_correct
                ]
                result = VerificationResult(
                    training_start=window_start,
                    training_end=training_end,
                    verification_start=v_start,
                    verification_end=min(v_end, cutoff),
                    total_predictions=len(verified),
                    correct_predictions=correct,
                    discrepancies=discrepancies,
                )
                session.verification_results.append(result)
                logger.debug(
                    "[daily] train %s\u2192%s | verify %s | %d/%d correct",
                    window_start,
                    training_end,
                    v_start,
                    correct,
                    len(verified),
                )

            window_start = _add_days(window_start, 1)

        logger.info(
            "[daily] complete: %d blocks, overall accuracy=%.1f%%",
            len(session.verification_results),
            session.overall_accuracy * 100,
        )
        return session

    def run(self) -> dict[str, PredictiveBacktestSession]:
        """
        Run the full multi-scale predictive accuracy backtest.

        Executes all three phases sequentially:
        1. 5-year blocks (2000 → 2025)
        2. Weekly blocks (``WEEKLY_BLOCKS_START`` → 2025)
        3. Daily blocks  (``DAILY_BLOCKS_START`` → 2025)

        Returns
        -------
        dict mapping phase label to :class:`PredictiveBacktestSession`.
        """
        logger.info("Starting multi-scale predictive accuracy backtest.")
        results = {
            "five_year_blocks": self.run_five_year_blocks(),
            "weekly_blocks": self.run_weekly_blocks(),
            "daily_blocks": self.run_daily_blocks(),
        }
        for label, session in results.items():
            logger.info(
                "Phase '%s' complete: overall_accuracy=%.1f%%, blocks=%d.",
                label,
                session.overall_accuracy * 100,
                len(session.verification_results),
            )
        return results
