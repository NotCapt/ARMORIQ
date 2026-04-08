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
