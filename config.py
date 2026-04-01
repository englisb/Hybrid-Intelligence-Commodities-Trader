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
# Backtesting windows  (start, end) in ISO format
# ---------------------------------------------------------------------------
BACKTEST_WINDOWS = [
    ("2023-01-01", "2026-03-31"),
    ("2019-01-01", "2022-12-31"),
    ("2015-01-01", "2018-12-31"),
    ("2007-01-01", "2009-12-31"),
]

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
