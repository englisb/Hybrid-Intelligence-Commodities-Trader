"""
Tests for the SentimentClassifier.

All Anthropic API calls are mocked to avoid network access and API costs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest

from src.agents.sentiment_classifier import SentimentClassifier, SentimentResult
from src.data.news_scraper import NewsArticle


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_article(
    title: str = "Gold prices rise",
    summary: str = "Gold prices hit a new record.",
    source: str = "reuters",
) -> NewsArticle:
    from datetime import datetime, timezone
    return NewsArticle(
        source=source,
        title=title,
        summary=summary,
        url="https://example.com/article",
        published=datetime.now(timezone.utc),
    )


def _mock_claude_response(level: int, rationale: str = "Test rationale.") -> MagicMock:
    """Build a mock Anthropic messages.create response."""
    content_block = MagicMock()
    content_block.text = json.dumps({"level": level, "rationale": rationale})
    response = MagicMock()
    response.content = [content_block]
    return response


# ---------------------------------------------------------------------------
# SentimentResult properties
# ---------------------------------------------------------------------------

class TestSentimentResult:
    def test_level_1_no_actions(self):
        result = SentimentResult(
            level=1, rationale="routine", article_title="Test", article_source="reuters"
        )
        assert result.requires_stop_loss_tighten is False
        assert result.requires_defensive_exit is False

    def test_level_3_no_actions(self):
        result = SentimentResult(
            level=3, rationale="moderate", article_title="Test", article_source="reuters"
        )
        assert result.requires_stop_loss_tighten is False
        assert result.requires_defensive_exit is False

    def test_level_4_tighten_stop_loss(self):
        result = SentimentResult(
            level=4, rationale="high", article_title="Test", article_source="reuters"
        )
        assert result.requires_stop_loss_tighten is True
        assert result.requires_defensive_exit is False

    def test_level_5_defensive_exit(self):
        result = SentimentResult(
            level=5, rationale="extreme", article_title="Test", article_source="reuters"
        )
        assert result.requires_stop_loss_tighten is True
        assert result.requires_defensive_exit is True

    def test_to_dict(self):
        result = SentimentResult(
            level=3, rationale="moderate", article_title="Test", article_source="reuters"
        )
        d = result.to_dict()
        assert d["level"] == 3
        assert d["rationale"] == "moderate"
        assert "requires_stop_loss_tighten" in d
        assert "requires_defensive_exit" in d


# ---------------------------------------------------------------------------
# SentimentClassifier.classify
# ---------------------------------------------------------------------------

class TestSentimentClassifier:
    def _make_classifier(self, response_level: int, rationale: str = "Test."):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_claude_response(
            response_level, rationale
        )
        return SentimentClassifier(client=mock_client)

    def test_classify_level_1(self):
        classifier = self._make_classifier(1)
        result = classifier.classify(make_article())
        assert result.level == 1
        assert result.requires_stop_loss_tighten is False

    def test_classify_level_4_tightens_stop(self):
        classifier = self._make_classifier(4, "Major geopolitical event.")
        result = classifier.classify(make_article(title="War escalation"))
        assert result.level == 4
        assert result.requires_stop_loss_tighten is True
        assert result.requires_defensive_exit is False

    def test_classify_level_5_triggers_exit(self):
        classifier = self._make_classifier(5, "Nuclear incident.")
        result = classifier.classify(make_article(title="Nuclear explosion"))
        assert result.level == 5
        assert result.requires_defensive_exit is True

    def test_classify_sets_article_metadata(self):
        classifier = self._make_classifier(2)
        article = make_article(title="Silver supply cut", source="bloomberg")
        result = classifier.classify(article)
        assert result.article_title == "Silver supply cut"
        assert result.article_source == "bloomberg"

    def test_malformed_response_defaults_to_level_1(self):
        mock_client = MagicMock()
        bad_content = MagicMock()
        bad_content.text = "NOT JSON AT ALL"
        mock_response = MagicMock()
        mock_response.content = [bad_content]
        mock_client.messages.create.return_value = mock_response

        classifier = SentimentClassifier(client=mock_client)
        result = classifier.classify(make_article())
        assert result.level == 1

    def test_out_of_range_level_defaults_to_1(self):
        mock_client = MagicMock()
        bad_content = MagicMock()
        bad_content.text = json.dumps({"level": 99, "rationale": "bad"})
        mock_response = MagicMock()
        mock_response.content = [bad_content]
        mock_client.messages.create.return_value = mock_response

        classifier = SentimentClassifier(client=mock_client)
        result = classifier.classify(make_article())
        assert result.level == 1

    def test_markdown_fences_stripped(self):
        mock_client = MagicMock()
        content = MagicMock()
        content.text = "```json\n" + json.dumps({"level": 3, "rationale": "ok"}) + "\n```"
        mock_response = MagicMock()
        mock_response.content = [content]
        mock_client.messages.create.return_value = mock_response

        classifier = SentimentClassifier(client=mock_client)
        result = classifier.classify(make_article())
        assert result.level == 3


# ---------------------------------------------------------------------------
# Batch classification
# ---------------------------------------------------------------------------

class TestClassifyBatch:
    def test_classify_batch_returns_all(self):
        mock_client = MagicMock()
        mock_client.messages.create.return_value = _mock_claude_response(2)
        classifier = SentimentClassifier(client=mock_client)
        articles = [make_article(title=f"Article {i}") for i in range(3)]
        results = classifier.classify_batch(articles)
        assert len(results) == 3
        assert all(r.level == 2 for r in results)

    def test_highest_level_returns_max(self):
        mock_client = MagicMock()
        responses = [
            _mock_claude_response(1),
            _mock_claude_response(4),
            _mock_claude_response(2),
        ]
        mock_client.messages.create.side_effect = responses
        classifier = SentimentClassifier(client=mock_client)
        articles = [make_article(title=f"Article {i}") for i in range(3)]
        top = classifier.highest_level(articles)
        assert top is not None
        assert top.level == 4

    def test_highest_level_empty_list_returns_none(self):
        mock_client = MagicMock()
        classifier = SentimentClassifier(client=mock_client)
        assert classifier.highest_level([]) is None
