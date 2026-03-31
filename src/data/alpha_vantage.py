"""
Alpha Vantage client for equity and ETF data.

Provides daily adjusted close prices for ETFs and mining stocks that form
the equity feed in the Hybrid Intelligence Commodities Trader.
API docs: https://www.alphavantage.co/documentation/
"""

from __future__ import annotations

import logging
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

_BASE_URL = "https://www.alphavantage.co/query"


def _get(params: dict[str, Any]) -> dict[str, Any]:
    """Execute a GET request against Alpha Vantage and return parsed JSON."""
    params["apikey"] = config.ALPHA_VANTAGE_API_KEY
    response = requests.get(_BASE_URL, params=params, timeout=10)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        logger.warning("Alpha Vantage rate-limit note: %s", data["Note"])
    return data


def get_daily_adjusted(
    symbol: str,
    output_size: str = "compact",
) -> list[dict[str, Any]]:
    """
    Return daily adjusted close prices for an equity or ETF ticker.

    Parameters
    ----------
    symbol:
        Ticker symbol, e.g. ``"GLD"`` or ``"NEM"``.
    output_size:
        ``"compact"`` returns the latest 100 data points;
        ``"full"`` returns up to 20 years of data.

    Returns
    -------
    List of dicts sorted ascending by date, each with keys:
    ``date`` (str), ``open``, ``high``, ``low``, ``close``,
    ``adjusted_close``, ``volume`` (all numeric).
    """
    data = _get(
        {
            "function": "TIME_SERIES_DAILY_ADJUSTED",
            "symbol": symbol,
            "outputsize": output_size,
        }
    )
    raw: dict[str, dict] = data.get("Time Series (Daily)", {})
    records: list[dict[str, Any]] = []
    for date_str, values in raw.items():
        records.append(
            {
                "date": date_str,
                "open": float(values.get("1. open", 0)),
                "high": float(values.get("2. high", 0)),
                "low": float(values.get("3. low", 0)),
                "close": float(values.get("4. close", 0)),
                "adjusted_close": float(values.get("5. adjusted close", 0)),
                "volume": int(float(values.get("6. volume", 0))),
            }
        )
    records.sort(key=lambda r: r["date"])
    return records


def get_quote(symbol: str) -> dict[str, Any]:
    """
    Return the latest global quote for a single equity ticker.

    Returns a dict with keys: ``symbol``, ``open``, ``high``, ``low``,
    ``price``, ``volume``, ``latest_trading_day``, ``previous_close``,
    ``change``, ``change_percent``.
    """
    data = _get({"function": "GLOBAL_QUOTE", "symbol": symbol})
    quote_raw: dict[str, str] = data.get("Global Quote", {})
    if not quote_raw:
        raise RuntimeError(f"Empty quote returned for {symbol}")
    return {
        "symbol": quote_raw.get("01. symbol", symbol),
        "open": float(quote_raw.get("02. open", 0)),
        "high": float(quote_raw.get("03. high", 0)),
        "low": float(quote_raw.get("04. low", 0)),
        "price": float(quote_raw.get("05. price", 0)),
        "volume": int(float(quote_raw.get("06. volume", 0))),
        "latest_trading_day": quote_raw.get("07. latest trading day", ""),
        "previous_close": float(quote_raw.get("08. previous close", 0)),
        "change": float(quote_raw.get("09. change", 0)),
        "change_percent": quote_raw.get("10. change percent", "0%"),
    }


def get_all_equity_quotes() -> dict[str, dict[str, Any]]:
    """
    Return latest quotes for all equity tickers in ``config.EQUITY_TICKERS``.

    Returns
    -------
    Dict mapping ticker symbol → quote dict (see ``get_quote``).
    """
    quotes: dict[str, dict[str, Any]] = {}
    for ticker in config.EQUITY_TICKERS:
        try:
            quotes[ticker] = get_quote(ticker)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to fetch quote for %s: %s", ticker, exc)
    return quotes
