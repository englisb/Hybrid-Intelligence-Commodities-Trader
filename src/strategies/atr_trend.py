"""
ATR-based trend-following strategy (comparison baseline).

Uses the Average True Range (ATR) to identify high-volatility breakout
entries and set dynamic stop-losses.  This is a purely technical approach
with no sentiment overlay.

Used as one of the two comparison strategies against the Hybrid model
(the Hybrid model must achieve a profit factor > 1.8 relative to this).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _true_range(
    high: np.ndarray, low: np.ndarray, prev_close: np.ndarray
) -> np.ndarray:
    """Vectorised True Range calculation."""
    hl = high - low
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    return np.maximum(hl, np.maximum(hc, lc))


def compute_atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """
    Compute the Average True Range (ATR) using a Wilder-smoothed RMA.

    Parameters
    ----------
    high, low, close:
        Price arrays of equal length (oldest first).
    period:
        ATR look-back period (default 14).

    Returns
    -------
    ATR array of the same length; first *period-1* values are ``NaN``.
    """
    if len(high) < 2:
        return np.full(len(high), np.nan)

    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]

    tr = _true_range(high, low, prev_close)
    atr = np.full(len(tr), np.nan)
    if len(tr) < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, len(tr)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def generate_signals(
    price_data: list[dict[str, Any]],
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    trend_period: int = 50,
) -> list[dict[str, Any]]:
    """
    Generate ATR breakout trade signals from OHLCV data.

    A long signal is generated when:
      - Price closes above its *trend_period*-day SMA (uptrend confirmed).
      - The close is above the prior high plus ``atr_multiplier × ATR``
        (breakout).

    A short signal is generated on the symmetric downside condition.

    Parameters
    ----------
    price_data:
        List of dicts with keys ``date``, ``open``, ``high``, ``low``,
        ``close`` (and optionally ``adjusted_close``).
    atr_period:
        Look-back for ATR calculation.
    atr_multiplier:
        Multiplier applied to ATR for breakout and stop-loss distance.
    trend_period:
        SMA look-back used to determine the prevailing trend direction.

    Returns
    -------
    List of signal dicts with keys:
    ``date``, ``signal`` (``"long"`` / ``"short"`` / ``"hold"``),
    ``entry_price``, ``stop_loss``, ``atr``.
    """
    if not price_data:
        return []

    data = sorted(price_data, key=lambda d: d["date"])
    n = len(data)

    close = np.array([d.get("adjusted_close", d["close"]) for d in data], dtype=float)
    high = np.array([d["high"] for d in data], dtype=float)
    low = np.array([d["low"] for d in data], dtype=float)

    atr = compute_atr(high, low, close, period=atr_period)

    signals: list[dict[str, Any]] = []
    min_idx = max(atr_period, trend_period)

    for i in range(min_idx, n):
        sma = float(np.mean(close[i - trend_period + 1 : i + 1]))
        current_atr = float(atr[i]) if not np.isnan(atr[i]) else 0.0
        entry = close[i]
        signal = "hold"
        stop = entry  # default

        if close[i] > sma and close[i] > close[i - 1] + atr_multiplier * current_atr:
            signal = "long"
            stop = entry - atr_multiplier * current_atr

        elif close[i] < sma and close[i] < close[i - 1] - atr_multiplier * current_atr:
            signal = "short"
            stop = entry + atr_multiplier * current_atr

        signals.append(
            {
                "date": data[i]["date"],
                "signal": signal,
                "entry_price": float(entry),
                "stop_loss": float(stop),
                "atr": current_atr,
            }
        )

    return signals
