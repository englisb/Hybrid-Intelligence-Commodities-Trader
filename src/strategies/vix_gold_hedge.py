"""
VIX-Gold hedge strategy (comparison baseline).

Based on the empirical negative correlation between the CBOE Volatility
Index (VIX, market fear gauge) and Gold (XAU): when fear spikes, gold
tends to rally as a safe-haven asset.

Signal logic:
  - When VIX rises above a threshold AND is trending up → long Gold.
  - When VIX falls sharply below a threshold → exit / short Gold.

Used as one of two comparison baselines against the Hybrid model.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_rolling_correlation(
    series_a: np.ndarray,
    series_b: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    """
    Return a rolling Pearson correlation between two equal-length series.

    Parameters
    ----------
    series_a, series_b:
        1-D arrays of equal length.
    window:
        Rolling window size.

    Returns
    -------
    Array of the same length; first *window-1* values are ``NaN``.
    """
    n = len(series_a)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        a_slice = series_a[i - window + 1 : i + 1]
        b_slice = series_b[i - window + 1 : i + 1]
        if np.std(a_slice) == 0 or np.std(b_slice) == 0:
            result[i] = 0.0
        else:
            result[i] = float(np.corrcoef(a_slice, b_slice)[0, 1])
    return result


def generate_signals(
    gold_data: list[dict[str, Any]],
    vix_data: list[dict[str, Any]],
    vix_threshold: float = 20.0,
    vix_spike_pct: float = 0.10,
    stop_loss_pct: float = 0.02,
) -> list[dict[str, Any]]:
    """
    Generate VIX-Gold hedge trade signals.

    Parameters
    ----------
    gold_data:
        List of dicts with keys ``date``, ``close`` (gold price).
    vix_data:
        List of dicts with keys ``date``, ``close`` (VIX value).
    vix_threshold:
        VIX level above which fear is considered elevated (default 20).
    vix_spike_pct:
        Minimum daily percentage increase in VIX to trigger a long signal.
    stop_loss_pct:
        Stop-loss as a fraction of entry price (default 2%).

    Returns
    -------
    List of signal dicts with keys:
    ``date``, ``signal`` (``"long"`` / ``"exit"`` / ``"hold"``),
    ``entry_price``, ``stop_loss``, ``vix``, ``vix_change_pct``.
    """
    # Align datasets on common dates
    gold_map = {d["date"]: d for d in gold_data}
    vix_map = {d["date"]: d for d in vix_data}
    common_dates = sorted(gold_map.keys() & vix_map.keys())

    if len(common_dates) < 2:
        return []

    signals: list[dict[str, Any]] = []

    for i in range(1, len(common_dates)):
        dt = common_dates[i]
        prev_dt = common_dates[i - 1]

        gold_price = float(gold_map[dt].get("adjusted_close", gold_map[dt]["close"]))
        vix_now = float(vix_map[dt]["close"])
        vix_prev = float(vix_map[prev_dt]["close"])

        vix_change_pct = (vix_now - vix_prev) / vix_prev if vix_prev != 0 else 0.0

        # Long signal: VIX is elevated AND spiking (fear rising → gold safe haven)
        if vix_now >= vix_threshold and vix_change_pct >= vix_spike_pct:
            signal = "long"
            stop = gold_price * (1.0 - stop_loss_pct)
        # Exit signal: VIX drops sharply (fear receding → exit gold hedge)
        elif vix_now < vix_threshold and vix_change_pct <= -vix_spike_pct:
            signal = "exit"
            stop = 0.0
        else:
            signal = "hold"
            stop = gold_price * (1.0 - stop_loss_pct)

        signals.append(
            {
                "date": dt,
                "signal": signal,
                "entry_price": gold_price,
                "stop_loss": stop,
                "vix": vix_now,
                "vix_change_pct": round(vix_change_pct, 4),
            }
        )

    return signals
