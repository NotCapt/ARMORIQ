# armoriq_sdk/models.py — Data models for the mock ArmorIQ SDK

import hashlib
import json
import time
import secrets


class PlanCapture:
    """Returned by capture_plan(). Holds the validated plan + crypto metadata."""

    def __init__(self, plan: dict, llm: str, prompt: str, metadata: dict = None):
        self.plan = plan
        self.llm = llm
        self.prompt = prompt
        self.metadata = metadata or {}

        # Generate realistic crypto artifacts
        canonical = json.dumps(plan, sort_keys=True)
        self.plan_hash = "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

        # Build Merkle tree from steps
        step_hashes = []
        self._step_proofs = {}
        for i, step in enumerate(plan.get("steps", [])):
            step_canonical = json.dumps(step, sort_keys=True)
            leaf = hashlib.sha256(step_canonical.encode()).hexdigest()
            step_hashes.append(leaf)
            # Store path and value digest for invoke() verification
            self._step_proofs[f"{step.get('mcp','')}/{step.get('action','')}"] = {
                "path": f"/steps/[{i}]/action",
                "value_digest": "sha256:" + hashlib.sha256(
                    step.get("action", "").encode()
                ).hexdigest(),
                "leaf": leaf,
                "index": i,
            }

        # Compute Merkle root (simplified: hash all leaves together)
        if step_hashes:
            combined = "".join(step_hashes)
            self.merkle_root = "sha256:" + hashlib.sha256(combined.encode()).hexdigest()
        else:
            self.merkle_root = "sha256:" + hashlib.sha256(b"empty").hexdigest()

        self.created_at = time.time()

    def __repr__(self):
        return f"PlanCapture(plan_hash={self.plan_hash[:24]}..., steps={len(self.plan.get('steps', []))})"


class IntentToken:
    """Returned by get_intent_token(). JWT-like token with crypto binding."""

    def __init__(
        self,
        plan_capture: PlanCapture,
        policy: dict,
        validity_seconds: float,
        user_id: str,
        agent_id: str,
    ):
        self.plan_hash = plan_capture.plan_hash
        self.merkle_root = plan_capture.merkle_root
        self.issued_at = int(time.time())
        self.expires_at = int(time.time() + validity_seconds)
        self.policy = policy
        self.success = True
        self._plan_capture = plan_capture

        # Generate realistic JWT-like token string
        header_b64 = secrets.token_urlsafe(36)
        payload_b64 = secrets.token_urlsafe(128)
        sig_b64 = secrets.token_urlsafe(86)
        self.token = f"{header_b64}.{payload_b64}.{sig_b64}"

        self.token_id = f"intent_{secrets.token_hex(12)}"
        self.signature = secrets.token_hex(64)  # Ed25519-like signature

        # Store step proofs for invoke verification
        self._step_proofs = plan_capture._step_proofs
        self._allowed_actions = set(policy.get("allow", []))
        self._denied_actions = set(policy.get("deny", []))

        # Build a set of all declared plan actions for verification
        self._declared_actions = set()
        for step in plan_capture.plan.get("steps", []):
            self._declared_actions.add(
                f"{step.get('mcp', '')}/{step.get('action', '')}"
            )

    def is_expired(self) -> bool:
        return time.time() > self.expires_at

    def __repr__(self):
        return f"IntentToken(hash={self.plan_hash[:24]}..., expires={self.expires_at})"


class MCPInvocationResult:
    """Returned by invoke() on success."""

    def __init__(self, data: dict, mcp: str, action: str, execution_time_ms: int = 0):
        self.success = True
        self.data = data
        self.mcp = mcp
        self.action = action
        self.execution_time_ms = execution_time_ms
        self.verified = True

    def __repr__(self):
        return f"MCPInvocationResult(mcp={self.mcp}, action={self.action}, success={self.success})"


class DelegationResult:
    """Returned by delegate()."""

    def __init__(
        self,
        delegation_id: str,
        delegated_token: IntentToken,
        delegate_public_key: str,
        expires_at: float,
    ):
        self.delegation_id = delegation_id
        self.delegated_token = delegated_token
        self.delegate_public_key = delegate_public_key
        self.expires_at = expires_at
        self.trust_delta = {"type": "delegation", "scope": "restricted"}
        self.status = "active"

    def __repr__(self):
        return f"DelegationResult(id={self.delegation_id}, status={self.status})"
