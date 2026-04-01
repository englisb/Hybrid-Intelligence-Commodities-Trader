"""
Central configuration for the Hybrid Intelligence Commodities Trader.
All hard-coded risk constraints and system-wide constants are defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# API credentials
# ---------------------------------------------------------------------------
METALS_API_KEY: str = os.getenv("METALS_API_KEY", "")
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")

# ---------------------------------------------------------------------------
# Commodities tracked
# ---------------------------------------------------------------------------
COMMODITIES = {
    "gold": "XAU",
    "uranium": "UXF26",
    "silver": "XAG",
    "diamonds": "DIAMOND",
}

# ---------------------------------------------------------------------------
# Hard-coded risk constraints (problem statement §Hard Coded Constraints)
# ---------------------------------------------------------------------------

# Maximum fraction of total liquid equity allocated across ALL open trades
MAX_CAPITAL_ALLOCATION: float = float(
    os.getenv("MAX_CAPITAL_ALLOCATION", "0.30")
)

# System-wide kill switch: liquidate everything if portfolio value drops this
# fraction from its historical peak
KILL_SWITCH_DRAWDOWN: float = float(
    os.getenv("KILL_SWITCH_DRAWDOWN", "0.50")
)

# Seconds the human operator has to intervene after a Level-5 geopolitical
# event before the AI executes a Defensive Exit
DEFENSIVE_EXIT_TIMEOUT_SECONDS: int = int(
    os.getenv("DEFENSIVE_EXIT_TIMEOUT_SECONDS", "300")  # 5 minutes
)

# ---------------------------------------------------------------------------
# Sentiment levels (1 = negligible impact, 5 = kinetic conflict / black swan)
# ---------------------------------------------------------------------------
SENTIMENT_TIGHTEN_STOP_LOSS_LEVEL: int = 4   # Level 4 or 5 → tighten stops
SENTIMENT_DEFENSIVE_EXIT_LEVEL: int = 5      # Level 5 → start countdown

# ---------------------------------------------------------------------------
# Strategy thresholds
# ---------------------------------------------------------------------------
TARGET_ACCURACY_RATE: float = 0.75           # 75% accuracy target
MIN_PROFIT_FACTOR: float = 1.8               # Minimum acceptable profit factor

# ---------------------------------------------------------------------------
# Information cutoff – no market data beyond this date is used
# ---------------------------------------------------------------------------
DATA_CUTOFF: str = "2025-12-31"

# ---------------------------------------------------------------------------
# Backtesting windows  (start, end) in ISO format
# ---------------------------------------------------------------------------
BACKTEST_WINDOWS = [
    ("2023-01-01", "2025-12-31"),    # Window 4: Post-pandemic / Geopolitical stress (cutoff Dec 2025)
    ("2019-01-01", "2022-12-31"),    # Window 3: COVID-19 + Recovery
    ("2015-01-01", "2018-12-31"),    # Window 2: Recovery / Low volatility
    ("2007-01-01", "2009-12-31"),    # Window 1: Global Financial Crisis
]

# ---------------------------------------------------------------------------
# Predictive accuracy backtesting (paper-trading mode)
# Multi-scale walk-forward validation: 5-year → weekly → daily windows
# ---------------------------------------------------------------------------
PREDICTIVE_BACKTEST_START: str = "2000-01-01"  # Start of historical data range
# PREDICTIVE_BACKTEST_END mirrors DATA_CUTOFF: both represent the same December 2025
# boundary.  They are kept as separate constants so that if the available historical
# data range were ever extended independently of the information cutoff they could
# diverge without changing semantics elsewhere.
PREDICTIVE_BACKTEST_END: str = DATA_CUTOFF      # Cannot exceed information cutoff
TRAINING_WINDOW_YEARS: int = 5                 # Initial training block length (years)
PREDICTION_HORIZON_YEARS: int = 3             # Forward prediction horizon (years)
PREDICTIONS_PER_BLOCK: int = 200              # Maximum predictions per training block
VERIFICATION_STEP_YEARS: int = 1             # Years of data revealed per verification step
# Narrower start points keep weekly/daily phases computationally tractable
WEEKLY_BLOCKS_START: str = "2020-01-01"       # Weekly-refinement phase start
DAILY_BLOCKS_START: str = "2024-01-01"        # Daily-refinement phase start

# ---------------------------------------------------------------------------
# News sources
# ---------------------------------------------------------------------------
NEWS_FEEDS = {
    "reuters": "https://feeds.reuters.com/reuters/businessNews",
    "bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
}

# ---------------------------------------------------------------------------
# Claude model
# ---------------------------------------------------------------------------
CLAUDE_MODEL: str = "claude-opus-4-5"
