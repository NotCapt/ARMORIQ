# research/stock_analyzer.py

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

# Default universe — large-cap tech + growth stocks
TECH_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "META",
    "AMZN", "AMD", "AVGO", "QCOM", "CRM",
    "ADBE", "NOW", "SNOW", "PLTR", "PANW"
]

SECTOR_UNIVERSES = {
    "tech":    ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMD", "AVGO", "QCOM", "ADBE", "CRM"],
    "ai":      ["NVDA", "AMD", "MSFT", "GOOGL", "META", "PLTR", "NOW", "SNOW", "PANW", "AMZN"],
    "cloud":   ["MSFT", "AMZN", "GOOGL", "CRM", "NOW", "SNOW", "ADBE", "PANW", "ZS", "DDOG"],
    "general": TECH_UNIVERSE
}


def analyze_stock(ticker: str) -> Optional[dict]:
    """
    Fetch 1-year data from Yahoo Finance and compute key metrics for a single ticker.
    Returns None if data is unavailable.
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        if hist.empty or len(hist) < 50:
            return None

        # Current price
        current_price = float(hist["Close"].iloc[-1])

        # 1-Year return
        start_price = float(hist["Close"].iloc[0])
        one_year_return = (current_price - start_price) / start_price

        # Monthly consistency — % of months with positive returns
        hist.index = pd.to_datetime(hist.index)
        monthly = hist["Close"].resample("ME").last().pct_change().dropna()
        positive_months = int((monthly > 0).sum())
        total_months = len(monthly)
        consistency = positive_months / total_months if total_months > 0 else 0.0

        # Annualized volatility
        daily_returns = hist["Close"].pct_change().dropna()
        volatility = float(daily_returns.std() * np.sqrt(252))

        # Composite score: reward return + consistency, penalize volatility
        composite_score = (
            0.4 * min(one_year_return, 1.5)   # cap at 150% to avoid outlier dominance
            + 0.4 * consistency
            - 0.2 * min(volatility, 1.0)       # cap penalty at 100% vol
        )

        # Additional fundamentals from yfinance info
        info = {}
        try:
            raw_info = stock.info
            info = {
                "company_name":  raw_info.get("longName", ticker),
                "sector":        raw_info.get("sector", "Technology"),
                "market_cap_b":  round(raw_info.get("marketCap", 0) / 1e9, 1),
                "pe_ratio":      raw_info.get("trailingPE"),
                "52w_high":      raw_info.get("fiftyTwoWeekHigh"),
                "52w_low":       raw_info.get("fiftyTwoWeekLow"),
            }
        except Exception:
            info = {"company_name": ticker}

        return {
            "ticker":           ticker,
            "current_price":    round(current_price, 2),
            "one_year_return":  round(one_year_return * 100, 1),   # as %
            "positive_months":  positive_months,
            "total_months":     total_months,
            "consistency_pct":  round(consistency * 100, 1),       # as %
            "volatility_pct":   round(volatility * 100, 1),        # as %
            "composite_score":  round(composite_score, 4),
            **info
        }

    except Exception as e:
        print(f"   [StockAnalyzer] Warning: Could not fetch {ticker} — {e}")
        return None


def screen_stocks(
    max_price: Optional[float] = None,
    sector: str = "tech",
    top_n: int = 5,
    min_return_pct: float = 0.0,
    custom_universe: Optional[list] = None
) -> dict:
    """
    Screen stocks from the universe, apply filters, rank by composite score,
    and return structured results + a natural-language context string for the Reasoner.
    """
    universe = custom_universe or SECTOR_UNIVERSES.get(sector, TECH_UNIVERSE)

    print(f"\n🔬 RESEARCH PHASE: Screening {len(universe)} stocks from {sector.upper()} universe...")
    if max_price:
        print(f"   Price filter: under ${max_price:.0f} per share")
    print(f"   Fetching 1-year data from Yahoo Finance...\n")

    results = []
    for ticker in universe:
        data = analyze_stock(ticker)
        if data is None:
            continue

        if max_price and data["current_price"] > max_price:
            print(f"   SKIP {ticker:6s} — ${data['current_price']:.0f} > ${max_price:.0f} price limit")
            continue

        if data["one_year_return"] < min_return_pct:
            print(f"   SKIP {ticker:6s} — {data['one_year_return']:.1f}% < {min_return_pct:.0f}% return minimum")
            continue

        results.append(data)
        print(
            f"   OK   {ticker:6s}  "
            f"${data['current_price']:7.2f}  "
            f"1Y: {data['one_year_return']:+6.1f}%  "
            f"Consistency: {data['consistency_pct']:5.1f}%  "
            f"Vol: {data['volatility_pct']:5.1f}%  "
            f"Score: {data['composite_score']:.3f}"
        )

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    top_stocks = results[:top_n]

    if not top_stocks:
        research_context = (
            "No stocks passed the screening filters. "
            "Consider relaxing the price or return filters."
        )
    else:
        lines = [
            "=== STOCK RESEARCH RESULTS (Yahoo Finance, 1-Year Analysis) ===",
            f"Screened {len(results)} stocks. Top {len(top_stocks)} by composite score "
            f"(return + consistency - volatility):\n"
        ]
        for rank, s in enumerate(top_stocks, 1):
            company = s.get("company_name", s["ticker"])
            pe = f"P/E: {s['pe_ratio']:.1f}" if s.get("pe_ratio") else "P/E: N/A"
            lines.append(
                f"  #{rank}. {s['ticker']} ({company})\n"
                f"      Price: ${s['current_price']:.2f}  |  "
                f"1Y Return: {s['one_year_return']:+.1f}%  |  "
                f"Positive months: {s['positive_months']}/{s['total_months']}  |  "
                f"Volatility: {s['volatility_pct']:.1f}%  |  "
                f"{pe}  |  Score: {s['composite_score']:.3f}"
            )
        lines.append(
            "\nUSE ONLY THESE TICKERS IN YOUR PLAN. Do not invent or substitute tickers. "
            "Prefer higher-ranked stocks. Respect the user's budget constraints."
        )
        research_context = "\n".join(lines)

    print(f"\n   Research complete: {len(top_stocks)} stocks qualify for the plan\n")

    return {
        "top_stocks":      top_stocks,
        "screened_count":  len(results),
        "filters_applied": {
            "max_price":      max_price,
            "sector":         sector,
            "min_return_pct": min_return_pct,
            "top_n":          top_n
        },
        "research_context": research_context
    }


def extract_research_params(user_prompt: str) -> dict:
    """
    Parse the user prompt to extract research parameters without calling an LLM.
    Simple regex + keyword heuristics.
    """
    import re
    prompt_lower = user_prompt.lower()

    # Max price extraction
    max_price = None
    price_patterns = [
        r"under\s+\$?([\d,]+)",
        r"below\s+\$?([\d,]+)",
        r"less\s+than\s+\$?([\d,]+)",
        r"max.*?\$?([\d,]+)\s*(?:per share|a share|each)"
    ]
    for pattern in price_patterns:
        match = re.search(pattern, prompt_lower)
        if match:
            try:
                max_price = float(match.group(1).replace(",", ""))
                break
            except Exception:
                pass

    # Sector detection
    sector = "tech"
    if any(w in prompt_lower for w in ["ai", "artificial intelligence", "machine learning"]):
        sector = "ai"
    elif any(w in prompt_lower for w in ["cloud", "saas", "software"]):
        sector = "cloud"

    # Min return threshold
    min_return_pct = 5.0
    if any(w in prompt_lower for w in ["consistent", "steady", "stable", "reliable"]):
        min_return_pct = 8.0
    elif any(w in prompt_lower for w in ["aggressive", "high growth", "best performing"]):
        min_return_pct = 15.0

    return {
        "max_price":      max_price,
        "sector":         sector,
        "min_return_pct": min_return_pct,
        "top_n":          5
    }
