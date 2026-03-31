"""
Claude-powered sentiment classifier for commodity news.

Classifies incoming news on a 1-5 scale:
  1 – Negligible / routine market noise
  2 – Minor – mild market impact expected
  3 – Moderate – notable but manageable disruption
  4 – High – significant geopolitical or macro shock; tighten stop-losses
  5 – Extreme / black swan (e.g. kinetic conflict, nuclear event);
       triggers Defensive Exit countdown

Levels 4 and 5 automatically trigger a tightening of stop-losses in the
execution layer (see :mod:`src.execution.risk_manager`).
Level 5 starts the Defensive Exit countdown (see
:mod:`src.execution.risk_manager`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import anthropic

import config
from src.data.news_scraper import NewsArticle

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a senior commodities risk analyst specialising in precious metals and
uranium markets. Your task is to assess the potential impact of a news article
on metal commodity prices (Gold, Silver, Uranium, Diamonds).

Classify the article on a scale of 1 to 5:
  1 = Negligible / routine market noise with no material price impact.
  2 = Minor – slight price movement likely; no change to risk parameters.
  3 = Moderate – notable disruption; enhanced monitoring warranted.
  4 = High – significant geopolitical or macro shock (sanctions, supply
      disruptions, major central bank policy change, large military
      escalation); TIGHTEN stop-losses immediately.
  5 = Extreme / black swan event (active kinetic conflict, nuclear incident,
      systemic financial crisis); DEFENSIVE EXIT protocol required.

Respond with ONLY a JSON object in the following format (no markdown fences,
no additional commentary):
{"level": <1-5>, "rationale": "<one sentence explaining the classification>"}
"""


@dataclass
class SentimentResult:
    """Result from the sentiment classifier."""

    level: int           # 1–5
    rationale: str
    article_title: str
    article_source: str

    @property
    def requires_stop_loss_tighten(self) -> bool:
        """True when level is at or above the configured threshold (≥4)."""
        return self.level >= config.SENTIMENT_TIGHTEN_STOP_LOSS_LEVEL

    @property
    def requires_defensive_exit(self) -> bool:
        """True when level is 5 – starts the Defensive Exit countdown."""
        return self.level >= config.SENTIMENT_DEFENSIVE_EXIT_LEVEL

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "rationale": self.rationale,
            "article_title": self.article_title,
            "article_source": self.article_source,
            "requires_stop_loss_tighten": self.requires_stop_loss_tighten,
            "requires_defensive_exit": self.requires_defensive_exit,
        }


class SentimentClassifier:
    """
    Classifies commodity news articles using the Claude LLM.

    Parameters
    ----------
    client:
        An :class:`anthropic.Anthropic` client instance.  If *None*, a new
        client is created using ``config.ANTHROPIC_API_KEY``.
    model:
        Claude model identifier.  Defaults to ``config.CLAUDE_MODEL``.
    """

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        model: str | None = None,
    ) -> None:
        self._client = client or anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
        self._model = model or config.CLAUDE_MODEL

    def classify(self, article: NewsArticle) -> SentimentResult:
        """
        Classify a single news article and return a :class:`SentimentResult`.

        Parameters
        ----------
        article:
            The news article to classify.

        Returns
        -------
        :class:`SentimentResult` with level 1–5 and a rationale string.
        """
        user_message = (
            f"Source: {article.source}\n"
            f"Title: {article.title}\n"
            f"Summary: {article.summary}"
        )

        import json

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )
            raw_text: str = response.content[0].text.strip()

            # Strip markdown code fences if present
            raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text)
            raw_text = re.sub(r"\n?```$", "", raw_text)

            parsed = json.loads(raw_text)
            level = int(parsed["level"])
            if level < 1 or level > 5:
                raise ValueError(f"Level {level} out of range 1-5")
            rationale = str(parsed.get("rationale", ""))
        except Exception as exc:
            logger.warning(
                "Sentiment classification failed for '%s': %s. Defaulting to level 1.",
                article.title,
                exc,
            )
            level = 1
            rationale = "Classification failed; defaulting to level 1 (negligible)."

        result = SentimentResult(
            level=level,
            rationale=rationale,
            article_title=article.title,
            article_source=article.source,
        )
        logger.info(
            "Sentiment: level=%d source=%s title='%s'",
            result.level,
            article.source,
            article.title[:60],
        )
        return result

    def classify_batch(self, articles: list[NewsArticle]) -> list[SentimentResult]:
        """Classify a list of articles and return results in the same order."""
        return [self.classify(article) for article in articles]

    def highest_level(self, articles: list[NewsArticle]) -> SentimentResult | None:
        """
        Return the highest-severity :class:`SentimentResult` from a batch.

        Returns *None* if *articles* is empty.
        """
        if not articles:
            return None
        results = self.classify_batch(articles)
        return max(results, key=lambda r: r.level)
