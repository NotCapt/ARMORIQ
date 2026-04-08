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
