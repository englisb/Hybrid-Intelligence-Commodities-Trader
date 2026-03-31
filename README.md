# Hybrid Intelligence Commodities Trader

A centaur trading system that combines 24/7 AI-driven technical pattern recognition with human strategic oversight for high-stakes metal commodities (Gold, Uranium, Silver, and Diamonds).

## Overview

Quantitative algorithms process historical data at high speeds but lack contextual awareness for black swan geopolitical events. Human traders understand context but are too slow for micro-fluctuations. This system bridges that gap:

- **AI layer**: Continuous ATR-based technical signal generation and Claude LLM news sentiment classification (1–5 scale)
- **Human layer**: Strategic oversight for Level 4/5 geopolitical events with a 5-minute Defensive Exit window
- **Target**: 75% accuracy rate with a profit factor > 1.8 over comparison strategies

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Data Ingestion Layer                    │
│  Metals-API (XAU, UXF26, XAG)  │  Alpha Vantage (ETFs)  │
│           Reuters / Bloomberg RSS News Feeds              │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                  LLM-Agent Engine (Claude)                │
│  Sentiment Classifier (1-5)  │  Monitoring Loop          │
│  Level 4 → Tighten Stops     │  Level 5 → Defensive Exit │
└─────────────────────┬───────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────┐
│                   Execution Logic                         │
│  Portfolio Tracker  │  Risk Manager  │  Order Manager    │
│  30% Alloc Limit    │  50% Kill Switch│  Stop-Loss Mgmt  │
└─────────────────────────────────────────────────────────┘
```

## Hard-Coded Risk Constraints

| Constraint | Value | Description |
|---|---|---|
| Capital Allocation | 30% | Maximum fraction of liquid equity in open positions |
| Kill Switch | 50% drawdown | System-wide liquidation if portfolio drops 50% from peak |
| Defensive Exit Timeout | 5 minutes | Human intervention window after a Level-5 event |

## Sentiment Classification (1–5 Scale)

| Level | Description | Action |
|---|---|---|
| 1 | Negligible / routine noise | No action |
| 2 | Minor market impact | No action |
| 3 | Moderate disruption | Reduce position size 25%, tighten ATR confirmation |
| 4 | Significant shock (sanctions, supply disruption) | Block new longs, tighten stop-losses |
| 5 | Black swan / kinetic conflict | Block all entries, start Defensive Exit countdown |

## Project Structure

```
├── config.py                      # Central configuration & constants
├── main.py                        # CLI entry point
├── requirements.txt
├── .env.example                   # API key template
├── src/
│   ├── data/
│   │   ├── metals_api.py          # Metals-API client (XAU, UXF26, XAG)
│   │   ├── alpha_vantage.py       # Alpha Vantage equity/ETF client
│   │   └── news_scraper.py        # Reuters & Bloomberg RSS scraper
│   ├── agents/
│   │   ├── sentiment_classifier.py # Claude LLM sentiment (1-5)
│   │   └── llm_agent.py           # Monitoring loop orchestrator
│   ├── execution/
│   │   ├── portfolio.py           # Position & equity tracking
│   │   ├── risk_manager.py        # 30% limit, kill switch, defensive exit
│   │   └── order_manager.py       # Trade validation & execution
│   ├── strategies/
│   │   ├── hybrid.py              # Primary: ATR + sentiment overlay
│   │   ├── atr_trend.py           # Comparison: ATR trend-following
│   │   └── vix_gold_hedge.py      # Comparison: VIX-Gold correlation
│   ├── backtesting/
│   │   ├── backtester.py          # Historical simulation engine
│   │   └── monte_carlo.py         # Monte Carlo stress testing (≥1,000 runs)
│   └── paper_trading.py           # 1-week live simulation (zero capital risk)
└── tests/
    ├── test_portfolio.py
    ├── test_risk_manager.py
    ├── test_sentiment_classifier.py
    ├── test_backtester.py
    └── test_monte_carlo.py
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and fill in your API keys:
#   METALS_API_KEY        – https://metals-api.com
#   ALPHA_VANTAGE_API_KEY – https://www.alphavantage.co
#   ANTHROPIC_API_KEY     – https://console.anthropic.com
```

### 3. Run

```bash
# Continuous news monitoring + trading (default)
python main.py monitor

# Historical backtesting (2007-2009, 2015-2018, 2019-2022, 2023-2026)
python main.py backtest

# 1-week paper trading simulation (zero capital risk)
python main.py paper

# Comparative strategy analysis (Hybrid vs ATR vs VIX-Gold)
python main.py compare
```

## Testing & Validation

### Run tests

```bash
pytest tests/ -v
```

### Backtesting windows

| Window | Period | Market Regime |
|---|---|---|
| 1 | 2007–2009 | Global Financial Crisis |
| 2 | 2015–2018 | Recovery / Low volatility |
| 3 | 2019–2022 | COVID-19 + Recovery |
| 4 | 2023–2026 | Post-pandemic / Geopolitical stress |

### Monte Carlo validation

- Minimum **1,000 simulation runs** with randomised slippage and fee variables
- Target: **≥ 80% of runs** end profitably (confidence level)
- Hybrid model must achieve **profit factor > 1.8** vs both comparison strategies

## Comparison Strategies

| Strategy | Description |
|---|---|
| **Hybrid** (primary) | ATR technical signals gated by Claude sentiment analysis |
| **ATR Trend Following** | Pure technical, ATR breakout with SMA trend filter |
| **VIX-Gold Hedge** | Correlation model: VIX spikes → long Gold safe-haven |

## Commodities & Tickers

| Asset | Ticker | Data Source |
|---|---|---|
| Gold | XAU | Metals-API |
| Uranium | UXF26 | Metals-API |
| Silver | XAG | Metals-API |
| GLD ETF | GLD | Alpha Vantage |
| Uranium ETF | URA | Alpha Vantage |
| Newmont Mining | NEM | Alpha Vantage |
| Barrick Gold | GOLD | Alpha Vantage |
| Cameco | CCJ | Alpha Vantage |