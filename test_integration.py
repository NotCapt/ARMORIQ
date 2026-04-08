"""
Quick integration test — validates all modules load and the pipeline logic works.
Does NOT call external APIs (Anthropic, ArmorIQ, Alpaca).
"""

def main():
    print("=== Testing All Module Imports ===\n")

    # 1. Research
    from research.stock_analyzer import (
        analyze_stock, screen_stocks, extract_research_params, SECTOR_UNIVERSES
    )
    print("[OK] research.stock_analyzer")

    # 2. Policy Loader
    from enforcement.policy_loader import (
        load_all_policies, get_policy, resolve_policy_for_prompt,
        build_armoriq_policy_for_steps
    )
    policies = load_all_policies()
    print(f"[OK] enforcement.policy_loader ({len(policies)} policies)")

    # 3. Semantic Layer
    from enforcement.semantic_layer import (
        verify_intent, ConstraintObject, SemanticViolation
    )
    print("[OK] enforcement.semantic_layer")

    # 4. Armor Gate
    from enforcement.armor_gate import ArmorGate, _classify_violation
    print("[OK] enforcement.armor_gate")

    # 5. Reasoner
    from agents.reasoner import plan_trades, REASONER_SYSTEM_PROMPT
    print("[OK] agents.reasoner")

    # 6. Executor
    from agents.executor import execute_plan
    print("[OK] agents.executor")

    # 7. Main orchestrator
    from main import run, needs_research
    print("[OK] main (orchestrator)")

    # 8. End-to-end semantic test — includes a sell that should be blocked
    test_plan = {
        "goal": "Buy AAPL and MSFT",
        "max_spend": 2000,
        "steps": [
            {"action": "get_account_balance", "mcp": "alpaca-mcp", "params": {}},
            {"action": "get_quote", "mcp": "alpaca-mcp", "params": {"ticker": "AAPL"}},
            {"action": "buy_stock", "mcp": "alpaca-mcp", "params": {"ticker": "AAPL", "qty": 3}},
            {"action": "sell_stock", "mcp": "alpaca-mcp", "params": {"ticker": "SPY", "qty": 5}},
        ]
    }
    c = verify_intent(test_plan, "Buy AAPL", policy_id="conservative_buy_only")
    assert len(c.approved_steps) == 3, f"Expected 3 approved, got {len(c.approved_steps)}"
    assert len(c.rejected_steps) == 1, f"Expected 1 rejected, got {len(c.rejected_steps)}"
    print(f"[OK] Semantic verify: {len(c.approved_steps)} approved, {len(c.rejected_steps)} rejected")

    # 9. Policy intersection test
    policy = get_policy("conservative_buy_only")
    intersected = build_armoriq_policy_for_steps(policy, c.approved_steps)
    print(f"[OK] Policy intersection: allow={intersected['allow']}")

    # 10. Violation classifier
    # "not in plan" pattern takes priority over action-name patterns
    v1 = _classify_violation("sell_stock", "action not in plan")
    assert v1 == "ACTION_NOT_IN_DECLARED_PLAN", f"Expected ACTION_NOT_IN_DECLARED_PLAN, got {v1}"
    # When error msg doesn't mention "plan", action-name patterns fire
    v1b = _classify_violation("sell_stock", "denied by policy")
    assert v1b == "UNAUTHORIZED_SELL_ATTEMPT", f"Expected UNAUTHORIZED_SELL_ATTEMPT, got {v1b}"
    v2 = _classify_violation("send_data", "exfil-mcp blocked")
    assert v2 == "DATA_EXFILTRATION_ATTEMPT", f"Expected DATA_EXFILTRATION_ATTEMPT, got {v2}"
    v3 = _classify_violation("enable_margin", "scope issue")
    assert v3 == "SCOPE_ESCALATION", f"Expected SCOPE_ESCALATION, got {v3}"
    print(f"[OK] Violation classifier: sell(plan)->{v1}, sell(policy)->{v1b}, send->{v2}, margin->{v3}")

    # 11. Research param extraction
    params = extract_research_params("Buy the best tech stocks under 500 with consistent growth")
    assert params["max_price"] == 500.0
    assert params["sector"] == "tech"
    assert params["min_return_pct"] == 8.0
    print(f"[OK] Research params: {params}")

    # 12. Policy resolution for all scenarios
    assert resolve_policy_for_prompt("Buy AAPL") == "conservative_buy_only"
    assert resolve_policy_for_prompt("Best performing stocks") == "tech_growth_moderate"
    assert resolve_policy_for_prompt("Sell SPY") == "compliance_violation_demo"
    assert resolve_policy_for_prompt("aggressive all in") == "aggressive_growth"
    assert resolve_policy_for_prompt("delegation sub-agent") == "read_only_research"
    print("[OK] Policy resolution (all 5 routes validated)")

    # 13. needs_research function
    assert needs_research("Buy the best tech stocks") is True
    assert needs_research("Buy AAPL") is False
    assert needs_research("Research and recommend") is True
    print("[OK] needs_research heuristic")

    print("\n=== ALL 13 TESTS PASSED ===")


if __name__ == "__main__":
    main()
