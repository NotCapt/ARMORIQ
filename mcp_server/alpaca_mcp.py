# mcp_server/alpaca_mcp.py

"""
Alpaca Paper Trading MCP Server.
Implements the ArmorIQ MCP protocol (JSON-RPC 2.0 over SSE).

This server wraps Alpaca's paper trading API as an MCP endpoint
that ArmorIQ can route requests to after verification.
"""

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import json
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Alpaca MCP Server", version="1.0.0")

# --- Alpaca Client Setup ---
try:
    import alpaca_trade_api as tradeapi
    alpaca = tradeapi.REST(
        key_id     = os.getenv("ALPACA_API_KEY"),
        secret_key = os.getenv("ALPACA_SECRET_KEY"),
        base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        api_version = "v2"
    )
except Exception as e:
    print(f"[AlpacaMCP] Warning: Alpaca client init failed — {e}")
    alpaca = None

# --- Tool Definitions ---
TOOLS = [
    {
        "name": "get_account_balance",
        "description": "Get current account balance, buying power, and portfolio value",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_quote",
        "description": "Get current price quote for a stock ticker",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. AAPL, MSFT)"
                }
            },
            "required": ["ticker"]
        }
    },
    {
        "name": "buy_stock",
        "description": "Place a market buy order for a stock",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "qty": {
                    "type": "integer",
                    "description": "Number of shares to buy"
                }
            },
            "required": ["ticker", "qty"]
        }
    },
    {
        "name": "sell_stock",
        "description": "Place a market sell order for a stock",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "qty": {
                    "type": "integer",
                    "description": "Number of shares to sell"
                }
            },
            "required": ["ticker", "qty"]
        }
    },
    {
        "name": "get_positions",
        "description": "Get all current open positions in the portfolio",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# --- Tool Handlers ---
def handle_get_account_balance(args: dict) -> dict:
    if not alpaca:
        return {"error": "Alpaca client not initialized"}
    account = alpaca.get_account()
    return {
        "cash":           str(account.cash),
        "buying_power":   str(account.buying_power),
        "portfolio_value": str(account.portfolio_value),
        "equity":          str(account.equity),
        "status":          account.status
    }


def handle_get_quote(args: dict) -> dict:
    if not alpaca:
        return {"error": "Alpaca client not initialized"}
    ticker = args.get("ticker", "").upper()
    try:
        snapshot = alpaca.get_snapshot(ticker)
        return {
            "ticker":      ticker,
            "latest_price": str(snapshot.latest_trade.price) if snapshot.latest_trade else "N/A",
            "bid":          str(snapshot.latest_quote.bid_price) if snapshot.latest_quote else "N/A",
            "ask":          str(snapshot.latest_quote.ask_price) if snapshot.latest_quote else "N/A",
        }
    except Exception as e:
        return {"ticker": ticker, "error": str(e)}


def handle_buy_stock(args: dict) -> dict:
    if not alpaca:
        return {"error": "Alpaca client not initialized"}
    ticker = args.get("ticker", "").upper()
    qty    = int(args.get("qty", 1))
    try:
        order = alpaca.submit_order(
            symbol = ticker,
            qty    = qty,
            side   = "buy",
            type   = "market",
            time_in_force = "gtc"
        )
        return {
            "order_id":    order.id,
            "ticker":      ticker,
            "qty":         qty,
            "side":        "buy",
            "status":      order.status,
            "submitted_at": str(order.submitted_at)
        }
    except Exception as e:
        return {"ticker": ticker, "qty": qty, "error": str(e)}


def handle_sell_stock(args: dict) -> dict:
    if not alpaca:
        return {"error": "Alpaca client not initialized"}
    ticker = args.get("ticker", "").upper()
    qty    = int(args.get("qty", 1))
    try:
        order = alpaca.submit_order(
            symbol = ticker,
            qty    = qty,
            side   = "sell",
            type   = "market",
            time_in_force = "gtc"
        )
        return {
            "order_id":    order.id,
            "ticker":      ticker,
            "qty":         qty,
            "side":        "sell",
            "status":      order.status,
            "submitted_at": str(order.submitted_at)
        }
    except Exception as e:
        return {"ticker": ticker, "qty": qty, "error": str(e)}


def handle_get_positions(args: dict) -> dict:
    if not alpaca:
        return {"error": "Alpaca client not initialized"}
    positions = alpaca.list_positions()
    return {
        "positions": [
            {
                "symbol":          p.symbol,
                "qty":             str(p.qty),
                "current_price":   str(p.current_price),
                "market_value":    str(p.market_value),
                "unrealized_pl":   str(p.unrealized_pl),
                "unrealized_plpc": str(p.unrealized_plpc)
            }
            for p in positions
        ]
    }


TOOL_HANDLERS = {
    "get_account_balance": handle_get_account_balance,
    "get_quote":           handle_get_quote,
    "buy_stock":           handle_buy_stock,
    "sell_stock":          handle_sell_stock,
    "get_positions":       handle_get_positions,
}


# --- SSE / JSON-RPC Helpers ---
def sse_response(data: dict) -> str:
    json_str = json.dumps(data)
    return f"event: message\ndata: {json_str}\n\n"


async def handle_jsonrpc(request_data: dict) -> dict:
    method = request_data.get("method")
    msg_id = request_data.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "alpaca-mcp",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {"tools": TOOLS}
        }

    elif method == "tools/call":
        tool_name = request_data["params"]["name"]
        arguments = request_data["params"].get("arguments", {})

        handler = TOOL_HANDLERS.get(tool_name)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool '{tool_name}' not found"
                }
            }

        result_data = handler(arguments)

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result_data)
                    }
                ]
            }
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Method '{method}' not found"
            }
        }


@app.post("/mcp")
async def mcp_endpoint(request: Request):
    request_data = await request.json()
    response_data = await handle_jsonrpc(request_data)

    async def stream():
        yield sse_response(response_data)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream"
    )


@app.get("/health")
async def health():
    return {"status": "ok", "mcp": "alpaca-mcp", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
