"""
Hybrid Intelligence Commodities Trader – main entry point.

Usage::

    python main.py [mode]

Modes:
  monitor      Run continuous news-monitoring loop (default).
  backtest     Run historical backtests over all configured windows.
  paper        Run multi-scale predictive accuracy backtest (5-year → weekly → daily).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _run_monitor() -> None:
    from src.agents.llm_agent import LLMAgent
    from src.execution.portfolio import Portfolio
    from src.execution.risk_manager import RiskManager

    import config

    portfolio = Portfolio(initial_capital=100_000.0)
    risk_manager = RiskManager(portfolio)
    agent = LLMAgent(risk_manager=risk_manager)
    agent.run_monitoring_loop()


def _fetch_gold_ohlcv(start: str, end: str) -> list[dict]:
    """
    Fetch gold (XAU) OHLCV data from Metals-API for the given date range.

    Falls back to a deterministic synthetic series when ``METALS_API_KEY`` is
    not configured or the API call fails, so the application remains runnable
    without live credentials.
    """
    import config
    from src.data.metals_api import get_historical_prices_range, format_as_ohlcv

    if config.METALS_API_KEY:
        try:
            records = get_historical_prices_range(
                config.COMMODITIES["gold"], start, end
            )
            if records:
                logger.info(
                    "Fetched %d days of XAU data from Metals-API (%s→%s).",
                    len(records),
                    start,
                    end,
                )
                return format_as_ohlcv(records)
            logger.warning(
                "Metals-API returned no data for XAU %s→%s; using synthetic fallback.",
                start,
                end,
            )
        except Exception as exc:
            logger.warning(
                "Metals-API request failed for %s→%s (%s). Using synthetic fallback.",
                start,
                end,
                type(exc).__name__,
            )
    else:
        logger.warning(
            "METALS_API_KEY not set; using synthetic price data for %s→%s.",
            start,
            end,
        )

    # Deterministic synthetic fallback
    import numpy as np
    from datetime import datetime as _dt, timedelta as _td

    np.random.seed(42)
    start_dt = _dt.strptime(start, "%Y-%m-%d")
    end_dt = _dt.strptime(end, "%Y-%m-%d")
    n_days = max(1, (end_dt - start_dt).days + 1)
    prices = 1800.0 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n_days))
    return [
        {
            "date": (start_dt + _td(days=i)).strftime("%Y-%m-%d"),
            "open": float(prices[i] * 0.999),
            "high": float(prices[i] * 1.005),
            "low": float(prices[i] * 0.995),
            "close": float(prices[i]),
            "adjusted_close": float(prices[i]),
        }
        for i in range(n_days)
    ]


def _run_backtest() -> None:
    """Run backtests on Metals-API historical data for all configured windows."""
    import config
    from src.backtesting.backtester import run_backtest
    from src.strategies.hybrid import generate_signals as hybrid_sigs

    results = []
    for start, end in config.BACKTEST_WINDOWS:
        price_data = _fetch_gold_ohlcv(start, end)

        signals = hybrid_sigs(price_data, sentiment_level=1)
        signal_dicts = [s.to_dict() for s in signals]
        result = run_backtest(
            strategy_name="Hybrid",
            signals=signal_dicts,
            window_start=start,
            window_end=end,
        )
        results.append(result.to_dict())
        logger.info(
            "Backtest %s→%s: win_rate=%.1f%%, profit_factor=%.2f",
            start,
            end,
            result.win_rate * 100,
            result.profit_factor,
        )

    print(json.dumps(results, indent=2))


def _run_paper() -> None:
    """
    Run the multi-scale predictive accuracy backtest.

    Replaces the 1-week live simulation with a walk-forward validation
    protocol using historical data from 2000 through December 2025:

    1. 5-year blocks  – 200 predictions per window, verified year-by-year
    2. Weekly blocks  – 52-week training, 1-week verification
    3. Daily blocks   – 30-day training, 1-day verification
    """
    from src.paper_trading import PredictiveBacktester

    backtester = PredictiveBacktester()
    sessions = backtester.run()
    output = {label: session.to_dict() for label, session in sessions.items()}
    print(json.dumps(output, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Hybrid Intelligence Commodities Trader"
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="monitor",
        choices=["monitor", "backtest", "paper"],
        help="Operating mode (default: monitor)",
    )
    args = parser.parse_args(argv)

    mode_map = {
        "monitor": _run_monitor,
        "backtest": _run_backtest,
        "paper": _run_paper,
    }
    mode_map[args.mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
