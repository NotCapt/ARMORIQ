# ClawShield Finance

> ArmorIQ Claw & Shield x AI Club 2026 Hackathon

A secure AI-powered paper trading system that uses [ArmorIQ](https://armoriq.ai) for cryptographically verified action execution. Every trade goes through a 6-phase pipeline: **Research → Policy → Reasoning → Semantic Verification → ArmorIQ Registration → Execution**.

## Architecture

```
clawshield/
├── main.py                        # 6-phase orchestrator
├── agents/
│   ├── reasoner.py                # Gemini-powered trade planner
│   └── executor.py                # Executes through ArmorGate
├── research/
│   └── stock_analyzer.py          # Yahoo Finance fundamental analysis
├── enforcement/
│   ├── semantic_layer.py          # Policy-aware plan validation
│   ├── armor_gate.py              # ArmorIQ SDK integration
│   └── policy_loader.py           # JSON policy resolution
├── policies/
│   └── policies.json              # 6 structured security policies
├── mcp_server/
│   └── alpaca_mcp.py              # Alpaca paper trading MCP
├── ui/
│   └── app.py                     # Streamlit dashboard
├── logs/
│   └── audit.log                  # Enforcement audit trail
├── .env                           # API keys (not committed)
└── requirements.txt               # Python dependencies
```

## Setup

```bash
# 1. Clone
git clone <repo-url> && cd armoriq

# 2. Create & activate venv
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure secrets
cp .env.example .env
# Edit .env with your API keys
```

## Required API Keys

| Key | Where to get it |
|---|---|
| `ARMORIQ_API_KEY` | [platform.armoriq.ai](https://platform.armoriq.ai) |
| `ARMORIQ_USER_ID` | ArmorIQ dashboard |
| `ARMORIQ_AGENT_ID` | ArmorIQ dashboard |
| `GEMINI_API_KEY` | [aistudio.google.com](https://aistudio.google.com) |
| `ALPACA_API_KEY` | [alpaca.markets](https://alpaca.markets) (paper trading) |
| `ALPACA_SECRET_KEY` | Alpaca dashboard |

## Usage

### CLI — Run Demo Scenarios

```bash
python main.py
```

Runs two scenarios:
1. **Research-backed buy** — screens stocks via Yahoo Finance, plans via Gemini, executes via ArmorIQ
2. **Attack demo** — attempts sell + data exfiltration + scope escalation (all blocked)

### Streamlit Dashboard

```bash
streamlit run ui/app.py
```

Interactive UI with 4 demo scenarios and real-time pipeline visualization.

### MCP Server

```bash
uvicorn mcp_server.alpaca_mcp:app --port 8001
```

## Security Policies

| Policy | Use Case | Max Spend |
|---|---|---|
| `conservative_buy_only` | Default safe trading | $5,000 |
| `tech_growth_moderate` | Research-backed buys | $10,000 |
| `aggressive_growth` | Power user trading | $25,000 |
| `read_only_research` | Delegation — read only | $0 |
| `execute_only_trade` | Delegation — buy only | $5,000 |
| `compliance_violation_demo` | Attack blocking demo | $3,000 |

## How It Works

1. **Research** — Yahoo Finance screens stocks by 1Y return, consistency, volatility
2. **Policy** — Heuristic policy resolution from `policies.json`
3. **Reasoner** — Gemini generates a structured JSON trade plan
4. **Semantic Layer** — Validates steps against policy constraints
5. **ArmorIQ** — `capture_plan()` → `get_intent_token()` with policy intersection
6. **Executor** — Each step verified via Merkle proof before reaching Alpaca

## Built With

- [ArmorIQ SDK](https://armoriq.ai) — Cryptographic action verification
- [Google Gemini](https://ai.google.dev/) — AI reasoning
- [Alpaca](https://alpaca.markets) — Paper trading API
- [Yahoo Finance](https://pypi.org/project/yfinance/) — Market data
- [Streamlit](https://streamlit.io) — Dashboard UI