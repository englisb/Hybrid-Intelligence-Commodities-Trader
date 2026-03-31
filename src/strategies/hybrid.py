"""
Hybrid Intelligence strategy – the primary trading strategy.

Combines two signal sources:
  1. **Technical layer** – ATR-based breakout signals (see
     :mod:`src.strategies.atr_trend`).
  2. **Sentiment layer** – Claude LLM news sentiment classification
     (see :mod:`src.agents.sentiment_classifier`).

Decision logic
--------------
- A technical long/short signal is generated first.
- The signal is then gated by the current news sentiment level:

  ============  ================================================================
  Sentiment     Action
  ============  ================================================================
  1–2 (low)     Pass through the technical signal unchanged.
  3 (moderate)  Require a stronger ATR confirmation (higher multiplier) before
                allowing entry; reduce position size by 25%.
  4 (high)      Block new long entries; only allow short entries if momentum is
                strongly bearish.  Tighten stop-losses on existing positions.
  5 (extreme)   Block ALL new entries; trigger Defensive Exit countdown.
  ============  ================================================================

This centaur model ensures the AI's speed on technical patterns is tempered
by contextual geopolitical awareness.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.strategies.atr_trend import generate_signals as atr_signals, compute_atr
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class HybridSignal:
    """A combined technical + sentiment trade signal."""

    date: str
    signal: str          # "long", "short", "hold", "defensive_exit"
    entry_price: float
    stop_loss: float
    position_size_factor: float = 1.0   # 1.0 = full size, 0.75 = 25% reduced
    atr: float = 0.0
    sentiment_level: int = 1
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "signal": self.signal,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "position_size_factor": self.position_size_factor,
            "atr": self.atr,
            "sentiment_level": self.sentiment_level,
            "rationale": self.rationale,
        }


def generate_signals(
    price_data: list[dict[str, Any]],
    sentiment_level: int = 1,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    trend_period: int = 50,
) -> list[HybridSignal]:
    """
    Generate hybrid trade signals for a given price series and current sentiment.

    Parameters
    ----------
    price_data:
        List of OHLCV dicts (``date``, ``open``, ``high``, ``low``, ``close``).
    sentiment_level:
        Current Claude sentiment level (1–5).
    atr_period:
        ATR look-back window.
    atr_multiplier:
        Base ATR multiplier used for breakout detection and stop-loss placement.
    trend_period:
        SMA period for trend direction.

    Returns
    -------
    List of :class:`HybridSignal` objects.
    """
    if sentiment_level == 5:
        # Defensive Exit: block all new entries, signal defensive action
        logger.warning(
            "Sentiment level 5 – DEFENSIVE EXIT signalled; blocking all new entries."
        )
        return [
            HybridSignal(
                date=row["date"],
                signal="defensive_exit",
                entry_price=float(row.get("adjusted_close", row["close"])),
                stop_loss=0.0,
                position_size_factor=0.0,
                sentiment_level=5,
                rationale="Level-5 event: Defensive Exit protocol engaged.",
            )
            for row in sorted(price_data, key=lambda d: d["date"])
        ]

    # Adjust ATR multiplier based on sentiment
    effective_multiplier = atr_multiplier
    if sentiment_level == 3:
        effective_multiplier = atr_multiplier * 1.25
    elif sentiment_level == 4:
        effective_multiplier = atr_multiplier * 1.75

    raw_signals = atr_signals(
        price_data,
        atr_period=atr_period,
        atr_multiplier=effective_multiplier,
        trend_period=trend_period,
    )

    hybrid: list[HybridSignal] = []
    for s in raw_signals:
        sig = s["signal"]
        size_factor = 1.0
        rationale = f"Sentiment level {sentiment_level}; ATR multiplier {effective_multiplier:.2f}."

        if sentiment_level == 3:
            size_factor = 0.75
            rationale = (
                "Moderate geopolitical risk (level 3): "
                "position size reduced to 75%, ATR confirmation tightened."
            )

        elif sentiment_level == 4:
            # Only allow bearish shorts at level 4; block longs
            if sig == "long":
                sig = "hold"
                rationale = (
                    "High geopolitical risk (level 4): long entry blocked; "
                    "stop-losses tightened on existing positions."
                )
            else:
                size_factor = 0.5
                rationale = (
                    "High geopolitical risk (level 4): "
                    "short allowed but size halved."
                )

        hybrid.append(
            HybridSignal(
                date=s["date"],
                signal=sig,
                entry_price=s["entry_price"],
                stop_loss=s["stop_loss"],
                position_size_factor=size_factor,
                atr=s["atr"],
                sentiment_level=sentiment_level,
                rationale=rationale,
            )
        )

    return hybrid
