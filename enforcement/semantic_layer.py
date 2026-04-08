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
