"""
Main LLM-Agent orchestrator.

Coordinates data ingestion, sentiment analysis, and execution decisions.
The agent runs a continuous monitoring loop that:
  1. Fetches latest commodity prices and equity quotes.
  2. Scrapes commodity-relevant news headlines.
  3. Classifies the highest-severity article.
  4. Feeds the result into the risk manager / execution layer.

The human operator can always override or halt execution at any point.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config
from src.agents.sentiment_classifier import SentimentClassifier, SentimentResult
from src.data.news_scraper import fetch_commodity_relevant_news, NewsArticle

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    Top-level LLM-Agent that integrates news sentiment with execution logic.

    Parameters
    ----------
    classifier:
        A :class:`~src.agents.sentiment_classifier.SentimentClassifier`
        instance. If *None*, a new one is created.
    risk_manager:
        An optional risk manager instance injected for testability.
    poll_interval_seconds:
        How often (in seconds) the agent polls for new news in
        :meth:`run_monitoring_loop`.
    """

    def __init__(
        self,
        classifier: SentimentClassifier | None = None,
        risk_manager: Any | None = None,
        poll_interval_seconds: int = 60,
    ) -> None:
        self._classifier = classifier or SentimentClassifier()
        self._risk_manager = risk_manager
        self._poll_interval = poll_interval_seconds
        self._last_articles: list[NewsArticle] = []

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse_news(self) -> SentimentResult | None:
        """
        Fetch commodity news, classify the highest-severity article, and
        trigger risk-management actions if required.

        Returns
        -------
        The :class:`~src.agents.sentiment_classifier.SentimentResult` for the
        most severe article found, or *None* if no articles were retrieved.
        """
        articles = fetch_commodity_relevant_news()
        self._last_articles = articles

        if not articles:
            logger.info("No commodity-relevant articles found.")
            return None

        result = self._classifier.highest_level(articles)
        if result is None:
            return None

        logger.info(
            "Highest sentiment level this cycle: %d (%s)",
            result.level,
            result.rationale,
        )
        self._handle_sentiment(result)
        return result

    def _handle_sentiment(self, result: SentimentResult) -> None:
        """Apply risk-management actions based on the sentiment level."""
        if result.requires_defensive_exit:
            logger.warning(
                "LEVEL 5 EVENT DETECTED: '%s'. Starting %d-second Defensive Exit "
                "countdown. Human operator can cancel within this window.",
                result.article_title,
                config.DEFENSIVE_EXIT_TIMEOUT_SECONDS,
            )
            if self._risk_manager is not None:
                self._risk_manager.start_defensive_exit_countdown(result)

        elif result.requires_stop_loss_tighten:
            logger.warning(
                "LEVEL %d EVENT DETECTED: '%s'. Tightening stop-losses.",
                result.level,
                result.article_title,
            )
            if self._risk_manager is not None:
                self._risk_manager.tighten_stop_losses(result)

    # ------------------------------------------------------------------
    # Monitoring loop
    # ------------------------------------------------------------------

    def run_monitoring_loop(self, cycles: int | None = None) -> None:
        """
        Run the continuous news-monitoring loop.

        Parameters
        ----------
        cycles:
            Number of monitoring cycles to run. Pass *None* (the default)
            to run indefinitely. Useful for testing with a finite count.
        """
        logger.info(
            "LLM-Agent monitoring loop started (poll_interval=%ds).",
            self._poll_interval,
        )
        cycle = 0
        try:
            while cycles is None or cycle < cycles:
                cycle += 1
                logger.debug("Monitoring cycle %d", cycle)
                try:
                    self.analyse_news()
                except Exception as exc:  # pragma: no cover
                    logger.error("Error in monitoring cycle %d: %s", cycle, exc)
                if cycles is None or cycle < cycles:
                    time.sleep(self._poll_interval)
        except KeyboardInterrupt:  # pragma: no cover
            logger.info("Monitoring loop interrupted by operator.")
        logger.info("LLM-Agent monitoring loop finished after %d cycles.", cycle)
