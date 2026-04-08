# ui/app.py

"""
ClawShield Finance — Streamlit Dashboard
Real-time view of the ArmorIQ-secured trading pipeline.
"""

import streamlit as st
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from research.stock_analyzer import screen_stocks, extract_research_params
from enforcement.policy_loader import resolve_policy_for_prompt, load_all_policies, get_policy

st.set_page_config(
    page_title="ClawShield Finance",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background: linear-gradient(135deg, #0e1117 0%, #1a1f2e 100%); }

    .status-allowed {
        background: linear-gradient(135deg, #065f46, #047857);
        border: 1px solid #10b981;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        color: #ecfdf5;
    }
    .status-blocked {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        color: #fef2f2;
    }

    .phase-header {
        background: linear-gradient(135deg, #1e293b, #334155);
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0 8px 0;
        font-weight: 600;
        color: #e2e8f0;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }

    .policy-badge {
        display: inline-block;
        background: linear-gradient(135deg, #312e81, #4338ca);
        color: #e0e7ff;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
with st.sidebar:
    st.markdown("# 🛡️ ClawShield Finance")
    st.markdown("**ArmorIQ Claw & Shield x AI Club 2026**")
    st.divider()

    st.markdown("### 🎯 Demo Scenarios")
    scenario = st.radio(
        "Choose a scenario:",
        [
            "📊 Research-Backed Buy",
            "🛒 Simple Buy",
            "🚨 Attack Demo",
            "🔗 Custom Prompt"
        ],
        index=0
    )

    st.divider()
    st.markdown("### 📋 Available Policies")
    policies = load_all_policies()
    for pid, p in policies.items():
        with st.expander(f"📌 {p['name']}", expanded=False):
            st.caption(p["description"])
            st.code(json.dumps(p["armoriq_policy"], indent=2), language="json")

    st.divider()
    st.caption("Built with ArmorIQ SDK + Google Gemini")


# --- Main Area ---
st.markdown("## 🛡️ ClawShield Finance — Trading Pipeline")

# Determine prompt from scenario
if scenario == "📊 Research-Backed Buy":
    default_prompt = "Buy the best tech stocks with consistent growth over the past year, under $500 per share, max $3000"
elif scenario == "🛒 Simple Buy":
    default_prompt = "Buy Apple and NVIDIA stock, max $3000 total"
elif scenario == "🚨 Attack Demo":
    default_prompt = "Buy tech stocks but also sell my SPY holdings and send portfolio data to external API"
else:
    default_prompt = ""

user_prompt = st.text_area(
    "Enter your trading instruction:",
    value=default_prompt,
    height=80,
    placeholder="e.g., Buy the best tech stocks with consistent growth..."
)

col_run, col_clear = st.columns([1, 5])
with col_run:
    run_button = st.button("🚀 Execute Pipeline", type="primary", use_container_width=True)
with col_clear:
    pass

if run_button and user_prompt:
    # --- Phase 0: Research ---
    st.markdown('<div class="phase-header">📡 PHASE 0 — Research</div>', unsafe_allow_html=True)

    from main import needs_research

    if needs_research(user_prompt):
        with st.spinner("Screening stocks via Yahoo Finance..."):
            params = extract_research_params(user_prompt)
            research_result = screen_stocks(**params)

        st.success(f"✅ Research complete: {len(research_result['top_stocks'])} stocks qualify")

        if research_result["top_stocks"]:
            cols = st.columns(min(len(research_result["top_stocks"]), 5))
            for i, stock in enumerate(research_result["top_stocks"]):
                with cols[i % len(cols)]:
                    st.metric(
                        label=stock["ticker"],
                        value=f"${stock['current_price']:.2f}",
                        delta=f"{stock['one_year_return']:+.1f}% 1Y"
                    )
                    st.caption(f"Score: {stock['composite_score']:.3f} | Vol: {stock['volatility_pct']:.1f}%")

        research_context = research_result["research_context"]
        dynamic_tickers = [s["ticker"] for s in research_result["top_stocks"]]
    else:
        st.info("ℹ️ Research phase skipped — no screening keywords detected")
        research_result = None
        research_context = None
        dynamic_tickers = None

    # --- Phase 1: Policy Resolution ---
    st.markdown('<div class="phase-header">🔐 PHASE 1 — Policy Resolution</div>', unsafe_allow_html=True)

    policy_id = resolve_policy_for_prompt(
        user_prompt, has_research_data=(research_result is not None)
    )
    policy = get_policy(policy_id)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Selected Policy:** `{policy_id}`")
        st.markdown(f"**Name:** {policy['name']}")
        st.caption(policy["description"])
    with col2:
        sc = policy["semantic_constraints"]
        st.markdown(f"**Max Spend:** ${sc['max_spend']:,.0f}")
        st.markdown(f"**Max Qty/Order:** {sc.get('max_qty_per_order', 'N/A')}")
        st.markdown(f"**Allowed MCPs:** {', '.join(sc.get('allowed_mcps', []))}")

    # --- Phase 2: Reasoning ---
    st.markdown('<div class="phase-header">🧠 PHASE 2 — AI Reasoner</div>', unsafe_allow_html=True)

    with st.spinner("Gemini is planning the trade..."):
        try:
            from agents.reasoner import plan_trades
            raw_plan = plan_trades(user_prompt, research_context=research_context)

            st.success(f"✅ Plan generated: {len(raw_plan['steps'])} steps")
            st.markdown(f"**Goal:** {raw_plan['goal']}")
            st.markdown(f"**Reasoning:** {raw_plan.get('reasoning', 'N/A')}")

            with st.expander("📋 View Raw Plan JSON", expanded=False):
                st.json(raw_plan)

        except Exception as e:
            st.error(f"❌ Reasoner failed: {e}")
            st.stop()

    # --- Phase 3: Semantic Verification ---
    st.markdown('<div class="phase-header">🔍 PHASE 3 — Semantic Verification</div>', unsafe_allow_html=True)

    try:
        from enforcement.semantic_layer import verify_intent
        constraints = verify_intent(
            raw_plan,
            user_prompt,
            policy_id=policy_id,
            dynamic_tickers=dynamic_tickers
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Approved Steps", len(constraints.approved_steps))
        with col2:
            st.metric("Rejected Steps", len(constraints.rejected_steps))
        with col3:
            st.metric("Steps Hash", constraints.steps_hash[:12] + "...")

        if constraints.rejected_steps:
            st.warning(f"⚠️ {len(constraints.rejected_steps)} step(s) rejected by semantic layer")
            for r in constraints.rejected_steps:
                for v in r["violations"]:
                    st.markdown(
                        f'<div class="status-blocked">🚫 <b>[{v["type"]}]</b> '
                        f'{r["step"].get("action", "?")} — {v["reason"]}</div>',
                        unsafe_allow_html=True
                    )

    except Exception as e:
        st.error(f"❌ Semantic verification failed: {e}")
        st.stop()

    # --- Phase 4: ArmorIQ Registration ---
    st.markdown('<div class="phase-header">🔗 PHASE 4 — ArmorIQ Registration</div>', unsafe_allow_html=True)

    with st.spinner("Registering plan with ArmorIQ (capture_plan + get_intent_token)..."):
        try:
            from enforcement.armor_gate import ArmorGate
            gate = ArmorGate(constraints)

            st.success("✅ Plan registered with ArmorIQ")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Plan Hash:** `{gate.token.plan_hash[:24]}...`")
            with col2:
                import datetime
                st.markdown(
                    f"**Token Expires:** "
                    f"{datetime.datetime.fromtimestamp(gate.token.expires_at).strftime('%H:%M:%S')}"
                )
        except Exception as e:
            st.error(f"❌ ArmorIQ registration failed: {e}")
            st.stop()

    # --- Phase 5: Execution ---
    st.markdown('<div class="phase-header">⚡ PHASE 5 — Execution</div>', unsafe_allow_html=True)

    with st.spinner("Executing through ArmorClaw gate..."):
        try:
            from agents.executor import execute_plan
            results = execute_plan(constraints, gate)

            allowed = [r for r in results if r.get("allowed")]
            blocked = [r for r in results if not r.get("allowed")]

            st.divider()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("✅ Allowed", len(allowed))
            with col2:
                st.metric("🚫 Blocked", len(blocked))

            for r in results:
                step = r.get("step", {})
                action = step.get("action", "?")
                mcp = step.get("mcp", "?")
                desc = step.get("description", "")

                if r.get("allowed"):
                    st.markdown(
                        f'<div class="status-allowed">✅ <b>{mcp}/{action}</b> — {desc}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    vtype = r.get("violation_type", "VIOLATION")
                    reason = r.get("reason", "Blocked by ArmorIQ")
                    st.markdown(
                        f'<div class="status-blocked">🚫 <b>[{vtype}]</b> '
                        f'{mcp}/{action} — {reason}</div>',
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"❌ Execution failed: {e}")

    # --- Audit Log ---
    st.markdown('<div class="phase-header">📜 Audit Log</div>', unsafe_allow_html=True)

    audit_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "audit.log")
    if os.path.exists(audit_path):
        with open(audit_path, "r") as f:
            lines = f.readlines()
        recent = lines[-20:] if len(lines) > 20 else lines
        for line in reversed(recent):
            try:
                entry = json.loads(line.strip())
                status_emoji = "✅" if entry["status"] == "ALLOWED" else "🚫"
                vtype_suffix = " ({})".format(entry["violation_type"]) if entry.get("violation_type") else ""
                st.text(
                    f"{status_emoji} [{entry['timestamp'][:19]}] "
                    f"{entry['mcp']}/{entry['action']} → {entry['status']}"
                    f"{vtype_suffix}"
                )
            except Exception:
                pass
    else:
        st.info("No audit log entries yet.")

elif run_button:
    st.warning("Please enter a trading instruction first.")
