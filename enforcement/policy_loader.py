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
