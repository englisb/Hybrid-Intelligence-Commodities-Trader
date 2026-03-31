"""
News scraper for Reuters and Bloomberg RSS feeds.

Fetches recent headlines and summaries from public RSS/Atom feeds and
extracts commodity-relevant articles for downstream sentiment analysis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import feedparser

import config

logger = logging.getLogger(__name__)

# Keywords that flag an article as commodity-relevant
_COMMODITY_KEYWORDS = [
    "gold", "silver", "uranium", "diamond", "precious metal",
    "commodity", "commodities", "inflation", "federal reserve",
    "geopoliti", "conflict", "sanction", "tariff", "trade war",
    "energy", "mining", "nuclear", "xau", "precious",
]


@dataclass
class NewsArticle:
    """Represents a single scraped news article."""

    source: str
    title: str
    summary: str
    url: str
    published: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_commodity_relevant(self) -> bool:
        """Return True if the article is likely relevant to metal commodities."""
        text = (self.title + " " + self.summary).lower()
        return any(kw in text for kw in _COMMODITY_KEYWORDS)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "url": self.url,
            "published": self.published.isoformat(),
        }


def _parse_published(entry: Any) -> datetime:
    """Parse publication datetime from a feedparser entry."""
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        try:
            return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def fetch_feed(source_name: str, feed_url: str) -> list[NewsArticle]:
    """
    Fetch and parse a single RSS/Atom feed.

    Parameters
    ----------
    source_name:
        Human-readable name for the source (e.g. ``"reuters"``).
    feed_url:
        Full URL to the RSS/Atom feed.

    Returns
    -------
    List of :class:`NewsArticle` objects, newest first.
    """
    try:
        parsed = feedparser.parse(feed_url)
    except Exception as exc:  # pragma: no cover
        logger.error("Failed to fetch feed %s (%s): %s", source_name, feed_url, exc)
        return []

    articles: list[NewsArticle] = []
    for entry in parsed.entries:
        title: str = getattr(entry, "title", "")
        summary: str = getattr(entry, "summary", getattr(entry, "description", ""))
        url: str = getattr(entry, "link", "")
        published = _parse_published(entry)
        articles.append(
            NewsArticle(
                source=source_name,
                title=title,
                summary=summary,
                url=url,
                published=published,
            )
        )

    articles.sort(key=lambda a: a.published, reverse=True)
    logger.info("Fetched %d articles from %s", len(articles), source_name)
    return articles


def fetch_all_feeds() -> list[NewsArticle]:
    """
    Fetch articles from all configured news feeds (Reuters, Bloomberg).

    Returns
    -------
    Deduplicated list of :class:`NewsArticle` objects, sorted newest-first.
    """
    all_articles: list[NewsArticle] = []
    for source_name, feed_url in config.NEWS_FEEDS.items():
        all_articles.extend(fetch_feed(source_name, feed_url))

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: list[NewsArticle] = []
    for article in all_articles:
        if article.url not in seen_urls:
            seen_urls.add(article.url)
            unique.append(article)

    unique.sort(key=lambda a: a.published, reverse=True)
    return unique


def fetch_commodity_relevant_news() -> list[NewsArticle]:
    """
    Return only commodity-relevant articles from all configured feeds.

    Filters the full article list using :meth:`NewsArticle.is_commodity_relevant`.
    """
    articles = fetch_all_feeds()
    relevant = [a for a in articles if a.is_commodity_relevant()]
    logger.info(
        "%d of %d articles flagged as commodity-relevant",
        len(relevant),
        len(articles),
    )
    return relevant
