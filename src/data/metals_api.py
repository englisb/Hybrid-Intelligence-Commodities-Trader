"""
Metals-API client for real-time commodity price data.

Fetches live and historical spot prices for Gold (XAU), Uranium (UXF26),
Silver (XAG) and related metals using the Metals-API service.
API docs: https://metals-api.com/documentation
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

_BASE_URL = "https://metals-api.com/api"


def _get(endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
    """Execute a GET request against the Metals-API and return parsed JSON."""
    params["access_key"] = config.METALS_API_KEY
    url = f"{_BASE_URL}/{endpoint}"
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data: dict[str, Any] = response.json()
    if not data.get("success", True):
        raise RuntimeError(
            f"Metals-API error: {data.get('error', {}).get('info', 'Unknown error')}"
        )
    return data


def get_latest_prices(symbols: list[str] | None = None) -> dict[str, float]:
    """
    Return the latest spot prices for the requested metal symbols.

    Parameters
    ----------
    symbols:
        List of metal ticker symbols (e.g. ``["XAU", "XAG"]``).
        Defaults to all commodities in ``config.COMMODITIES``.

    Returns
    -------
    dict mapping symbol → price in USD.
    """
    if symbols is None:
        symbols = list(config.COMMODITIES.values())

    symbols_str = ",".join(symbols)
    data = _get("latest", {"base": "USD", "symbols": symbols_str})
    rates: dict[str, float] = data.get("rates", {})
    # Metals-API returns rates as currency/metal; invert to get price-per-oz
    prices: dict[str, float] = {}
    for symbol in symbols:
        if symbol in rates and rates[symbol] != 0:
            prices[symbol] = 1.0 / rates[symbol]
        else:
            logger.warning("No rate returned for symbol %s", symbol)
    return prices


def get_historical_prices(
    symbol: str,
    start: date | str,
    end: date | str,
) -> list[dict[str, Any]]:
    """
    Return daily historical spot prices for a single metal symbol.

    Parameters
    ----------
    symbol:
        Metal ticker symbol, e.g. ``"XAU"``.
    start:
        Start date (``date`` object or ISO-format string ``"YYYY-MM-DD"``).
    end:
        End date (inclusive).

    Returns
    -------
    List of dicts with keys ``date`` (str) and ``price`` (float).
    """
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")

    records: list[dict[str, Any]] = []
    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        try:
            data = _get(date_str, {"base": "USD", "symbols": symbol})
            rates: dict[str, float] = data.get("rates", {})
            if symbol in rates and rates[symbol] != 0:
                records.append({"date": date_str, "price": 1.0 / rates[symbol]})
            else:
                logger.warning("No rate for %s on %s", symbol, date_str)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to fetch %s on %s: %s", symbol, date_str, exc)

        # Advance by one day
        from datetime import timedelta

        current += timedelta(days=1)

    return records


def get_historical_prices_range(
    symbol: str,
    start: date | str,
    end: date | str,
) -> list[dict[str, Any]]:
    """
    Return daily historical spot prices for a single metal symbol using the
    ``/timeframe`` endpoint (one API call, far more efficient than per-day
    queries for long date ranges).

    Parameters
    ----------
    symbol:
        Metal ticker symbol, e.g. ``"XAU"``.
    start:
        Start date (``date`` object or ISO-format string ``"YYYY-MM-DD"``).
    end:
        End date (inclusive).

    Returns
    -------
    List of dicts with keys ``date`` (str) and ``price`` (float), sorted
    ascending by date.
    """
    if isinstance(start, date):
        start = start.isoformat()
    if isinstance(end, date):
        end = end.isoformat()

    data = _get(
        "timeframe",
        {"start_date": start, "end_date": end, "base": "USD", "symbols": symbol},
    )
    rates: dict[str, dict[str, float]] = data.get("rates", {})

    records: list[dict[str, Any]] = []
    for date_str in sorted(rates.keys()):
        day_rates = rates[date_str]
        if symbol in day_rates and day_rates[symbol] != 0:
            records.append({"date": date_str, "price": 1.0 / day_rates[symbol]})
        else:
            logger.warning("No rate for %s on %s", symbol, date_str)

    return records


def format_as_ohlcv(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Convert Metals-API price records to OHLCV format expected by strategy
    modules.

    Since the API provides a single daily spot price, ``open``, ``high`` and
    ``low`` are approximated using standard intraday range assumptions:
    open = close × 0.999 (−0.1 %), high = close × 1.005 (+0.5 %),
    low = close × 0.995 (−0.5 %).

    Parameters
    ----------
    records:
        List of dicts with keys ``date`` (str) and ``price`` (float), as
        returned by :func:`get_historical_prices` or
        :func:`get_historical_prices_range`.

    Returns
    -------
    List of OHLCV dicts with keys ``date``, ``open``, ``high``, ``low``,
    ``close``, ``adjusted_close``.
    """
    return [
        {
            "date": r["date"],
            "open": r["price"] * 0.999,
            "high": r["price"] * 1.005,
            "low": r["price"] * 0.995,
            "close": r["price"],
            "adjusted_close": r["price"],
        }
        for r in records
    ]
