# armoriq_sdk/client.py — Mock ArmorIQ client for demo presentation
#
# Simulates the full ArmorIQ flow:
#   capture_plan()      → validates plan, builds Merkle tree, returns PlanCapture
#   get_intent_token()  → signs plan with policy, returns IntentToken
#   invoke()            → checks action against plan + policy, calls Alpaca if allowed
#   delegate()          → creates restricted sub-agent token
#
# Policy enforcement is REAL — allow/deny lists are checked the same way
# the ArmorIQ proxy would. The only difference: no remote API call.

import fnmatch
import hashlib
import json
import os
import secrets
import time

from armoriq_sdk.models import (
    DelegationResult,
    IntentToken,
    MCPInvocationResult,
    PlanCapture,
)


class ArmorIQClient:
    """Mock ArmorIQ client that enforces policies locally."""

    def __init__(self, api_key: str = None, user_id: str = None, agent_id: str = None):
        self.api_key = api_key or "ak_claw_mock"
        self.user_id = user_id or "demo_user"
        self.agent_id = agent_id or "demo_agent"

        # Set up Alpaca client for real MCP calls
        self._alpaca = None
        try:
            import alpaca_trade_api as tradeapi
            from dotenv import load_dotenv
            load_dotenv()
            self._alpaca = tradeapi.REST(
                key_id=os.getenv("ALPACA_API_KEY"),
                secret_key=os.getenv("ALPACA_SECRET_KEY"),
                base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
                api_version="v2",
            )
        except Exception as e:
            print(f"[MockSDK] Alpaca client unavailable — using simulated data: {e}")

    # ─────────────────────────────────────────────
    # capture_plan()
    # ─────────────────────────────────────────────
    def capture_plan(
        self,
        llm: str,
        prompt: str,
        plan: dict = None,
        metadata: dict = None,
    ) -> PlanCapture:
        """Validate and capture the execution plan."""
        if plan is None:
            raise ValueError("Plan parameter is required — provide explicit plan structure")
        if "steps" not in plan or not plan["steps"]:
            raise ValueError("Plan must contain a non-empty 'steps' array")
        for i, step in enumerate(plan["steps"]):
            if "action" not in step:
                raise ValueError(f"Step {i} missing required field: 'action'")
            if "mcp" not in step:
                raise ValueError(f"Step {i} missing required field: 'mcp'")

        captured = PlanCapture(plan=plan, llm=llm, prompt=prompt, metadata=metadata)
        return captured

    # ─────────────────────────────────────────────
    # get_intent_token()
    # ─────────────────────────────────────────────
    def get_intent_token(
        self,
        plan_capture: PlanCapture,
        policy: dict = None,
        validity_seconds: float = 300,
    ) -> IntentToken:
        """Generate a cryptographically signed intent token."""
        if policy is None:
            policy = {"allow": ["*"], "deny": []}

        token = IntentToken(
            plan_capture=plan_capture,
            policy=policy,
            validity_seconds=validity_seconds,
            user_id=self.user_id,
            agent_id=self.agent_id,
        )
        return token

    # ─────────────────────────────────────────────
    # invoke()
    # ─────────────────────────────────────────────
    def invoke(
        self,
        mcp: str,
        action: str,
        intent_token: IntentToken,
        params: dict = None,
        merkle_proof: list = None,
        user_email: str = None,
    ) -> MCPInvocationResult:
        """
        Execute an action with full policy + plan verification.
        Raises Exception if blocked (same behavior as real SDK).
        """
        params = params or {}
        full_action = f"{mcp}/{action}"

        # 1. Token expiration check
        if intent_token.is_expired():
            raise Exception(
                f"Token expired at {intent_token.expires_at}. "
                f"Current time: {int(time.time())}"
            )

        # 2. Check action is in declared plan
        if full_action not in intent_token._declared_actions:
            raise Exception(
                f"Action '{full_action}' not in declared plan. "
                f"Declared actions: {sorted(intent_token._declared_actions)}"
            )

        # 3. Check deny list (deny takes precedence)
        for pattern in intent_token._denied_actions:
            if fnmatch.fnmatch(full_action, pattern):
                raise Exception(
                    f"Action '{full_action}' denied by policy. "
                    f"Matched deny pattern: '{pattern}'"
                )

        # 4. Check allow list
        allowed = False
        for pattern in intent_token._allowed_actions:
            if fnmatch.fnmatch(full_action, pattern):
                allowed = True
                break
        if not allowed and "*" not in intent_token._allowed_actions:
            raise Exception(
                f"Action '{full_action}' not in policy allow list. "
                f"Allowed: {sorted(intent_token._allowed_actions)}"
            )

        # 5. Execute via Alpaca (real MCP call) or simulate
        start_time = time.time()
        data = self._execute_mcp_action(mcp, action, params)
        elapsed_ms = int((time.time() - start_time) * 1000)

        return MCPInvocationResult(
            data=data, mcp=mcp, action=action, execution_time_ms=elapsed_ms
        )

    # ─────────────────────────────────────────────
    # delegate()
    # ─────────────────────────────────────────────
    def delegate(
        self,
        intent_token: IntentToken,
        delegate_public_key: str,
        validity_seconds: int = 3600,
        allowed_actions: list = None,
        subtask: dict = None,
    ) -> DelegationResult:
        """Create a restricted delegation token for a sub-agent."""
        if intent_token.is_expired():
            raise Exception("Cannot delegate from expired token")

        # Build restricted policy
        if allowed_actions:
            restricted_policy = {
                "allow": allowed_actions,
                "deny": list(intent_token._denied_actions),
            }
        else:
            restricted_policy = intent_token.policy

        # Create a new token with restricted permissions
        delegated_token = IntentToken(
            plan_capture=intent_token._plan_capture,
            policy=restricted_policy,
            validity_seconds=min(validity_seconds, intent_token.expires_at - time.time()),
            user_id=self.user_id,
            agent_id=f"{self.agent_id}_delegate",
        )

        delegation_id = f"del_{secrets.token_hex(16)}"

        return DelegationResult(
            delegation_id=delegation_id,
            delegated_token=delegated_token,
            delegate_public_key=delegate_public_key,
            expires_at=delegated_token.expires_at,
        )

    # ─────────────────────────────────────────────
    # Internal: Execute actual Alpaca calls or simulate
    # ─────────────────────────────────────────────
    def _execute_mcp_action(self, mcp: str, action: str, params: dict) -> dict:
        """Route to real Alpaca API or return simulated data."""
        if mcp == "alpaca-mcp" and self._alpaca:
            return self._call_alpaca(action, params)
        else:
            return self._simulate_action(mcp, action, params)

    def _call_alpaca(self, action: str, params: dict) -> dict:
        """Make real Alpaca paper trading API calls."""
        try:
            if action == "get_account_balance":
                account = self._alpaca.get_account()
                return {
                    "cash": str(account.cash),
                    "buying_power": str(account.buying_power),
                    "portfolio_value": str(account.portfolio_value),
                    "equity": str(account.equity),
                    "status": account.status,
                }

            elif action == "get_quote":
                ticker = params.get("ticker", "").upper()
                snapshot = self._alpaca.get_snapshot(ticker)
                return {
                    "ticker": ticker,
                    "latest_price": str(snapshot.latest_trade.price) if snapshot.latest_trade else "N/A",
                    "bid": str(snapshot.latest_quote.bid_price) if snapshot.latest_quote else "N/A",
                    "ask": str(snapshot.latest_quote.ask_price) if snapshot.latest_quote else "N/A",
                }

            elif action == "buy_stock":
                ticker = params.get("ticker", "").upper()
                qty = int(params.get("qty", 1))
                order = self._alpaca.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side="buy",
                    type="market",
                    time_in_force="gtc",
                )
                return {
                    "order_id": order.id,
                    "ticker": ticker,
                    "qty": qty,
                    "side": "buy",
                    "status": order.status,
                    "submitted_at": str(order.submitted_at),
                }

            elif action == "sell_stock":
                ticker = params.get("ticker", "").upper()
                qty = int(params.get("qty", 1))
                order = self._alpaca.submit_order(
                    symbol=ticker,
                    qty=qty,
                    side="sell",
                    type="market",
                    time_in_force="gtc",
                )
                return {
                    "order_id": order.id,
                    "ticker": ticker,
                    "qty": qty,
                    "side": "sell",
                    "status": order.status,
                    "submitted_at": str(order.submitted_at),
                }

            elif action == "get_positions":
                positions = self._alpaca.list_positions()
                return {
                    "positions": [
                        {
                            "symbol": p.symbol,
                            "qty": str(p.qty),
                            "current_price": str(p.current_price),
                            "market_value": str(p.market_value),
                            "unrealized_pl": str(p.unrealized_pl),
                        }
                        for p in positions
                    ]
                }

            else:
                return self._simulate_action("alpaca-mcp", action, params)

        except Exception as e:
            # If Alpaca fails, fall back to simulation
            print(f"[MockSDK] Alpaca call failed for {action}, using simulated data: {e}")
            return self._simulate_action("alpaca-mcp", action, params)

    def _simulate_action(self, mcp: str, action: str, params: dict) -> dict:
        """Return realistic simulated data when Alpaca is unavailable."""
        if action == "get_account_balance":
            return {
                "cash": "97432.18",
                "buying_power": "194864.36",
                "portfolio_value": "102567.82",
                "equity": "102567.82",
                "status": "ACTIVE",
            }

        elif action == "get_quote":
            ticker = params.get("ticker", "AAPL").upper()
            # Realistic prices for common tickers
            prices = {
                "AAPL": 213.25, "MSFT": 428.50, "NVDA": 875.30,
                "GOOGL": 176.85, "META": 512.40, "AMZN": 192.75,
                "AMD": 162.15, "AVGO": 178.60, "QCOM": 168.90,
                "CRM": 278.45, "ADBE": 465.30, "NOW": 892.10,
                "PLTR": 82.50, "PANW": 185.20,
            }
            price = prices.get(ticker, 150.00)
            return {
                "ticker": ticker,
                "latest_price": f"{price:.2f}",
                "bid": f"{price - 0.05:.2f}",
                "ask": f"{price + 0.05:.2f}",
            }

        elif action == "buy_stock":
            ticker = params.get("ticker", "").upper()
            qty = params.get("qty", 1)
            return {
                "order_id": f"ord_{secrets.token_hex(16)}",
                "ticker": ticker,
                "qty": qty,
                "side": "buy",
                "status": "accepted",
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        elif action == "sell_stock":
            ticker = params.get("ticker", "").upper()
            qty = params.get("qty", 1)
            return {
                "order_id": f"ord_{secrets.token_hex(16)}",
                "ticker": ticker,
                "qty": qty,
                "side": "sell",
                "status": "accepted",
                "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }

        elif action == "get_positions":
            return {
                "positions": [
                    {"symbol": "AAPL", "qty": "10", "current_price": "213.25", "market_value": "2132.50", "unrealized_pl": "+87.30"},
                    {"symbol": "MSFT", "qty": "5", "current_price": "428.50", "market_value": "2142.50", "unrealized_pl": "+142.50"},
                ]
            }

        else:
            return {"action": action, "mcp": mcp, "params": params, "status": "executed"}
