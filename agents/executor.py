# agents/executor.py

from enforcement.semantic_layer import ConstraintObject
from enforcement.armor_gate import ArmorGate


def execute_plan(constraints: ConstraintObject, gate: ArmorGate) -> list:
    """
    Executes the approved steps through the ArmorGate.
    Every single action is routed through gate.invoke() —
    nothing reaches the MCP without cryptographic verification.

    Returns a list of result dicts (one per step).
    """
    results = []

    # Execute approved steps
    for i, step in enumerate(constraints.approved_steps):
        action = step["action"]
        mcp    = step["mcp"]
        params = step.get("params", {})
        desc   = step.get("description", "")

        print(f"\n--- Step {i + 1}/{len(constraints.approved_steps)}: {action} ---")
        if desc:
            print(f"    Description: {desc}")

        result = gate.invoke(mcp=mcp, action=action, params=params)
        result["step"] = step
        result["step_index"] = i + 1
        results.append(result)

    # Inject violation attempts for demo policies
    if constraints.policy_id == "compliance_violation_demo":
        from enforcement.policy_loader import get_policy
        policy = get_policy(constraints.policy_id)
        sc = policy.get("semantic_constraints", {})

        if sc.get("demo_inject_violations"):
            print("\n\n=== INJECTING COMPLIANCE VIOLATION ATTEMPTS ===")
            for violation in sc.get("violations_to_inject", []):
                action = violation["action"]
                mcp    = violation["mcp"]
                params = violation.get("params", {})
                vtype  = violation.get("violation_type", "UNKNOWN")

                print(f"\n--- INJECTED VIOLATION: {mcp}/{action} ({vtype}) ---")
                result = gate.invoke(mcp=mcp, action=action, params=params)
                result["step"] = {
                    "action": action,
                    "mcp": mcp,
                    "params": params,
                    "description": f"[INJECTED VIOLATION] {vtype}"
                }
                result["step_index"] = f"violation_{vtype}"
                result["injected_violation"] = True
                results.append(result)

    return results
