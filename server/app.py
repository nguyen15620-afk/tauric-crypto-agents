import asyncio
import json
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pathlib import Path

from config.settings import settings
from data.ccxt_feed import CCXTMarketFeed
from data.screener import MarketScreener
from agents.graph import TradingAgentGraph
from storage.memory_db import MemoryDB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FastAPIServer")

app = FastAPI(
    title="Crypto Multi-Agent Advisory Intelligence System",
    version="3.0.0",
    description="LangGraph multi-agent crypto market analysis, debate, and strategic advisory with real-time WebSocket streaming"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Shared State Instances — initialized once at startup
market_feed = CCXTMarketFeed(exchange_id=settings.DEFAULT_EXCHANGE)
market_screener = MarketScreener(exchange_id=settings.DEFAULT_EXCHANGE)
memory_db = MemoryDB()
trading_graph = TradingAgentGraph(exchange_feed=market_feed)

# Prevent concurrent multi-agent cycle runs that could exhaust rate limits
_cycle_lock = asyncio.Lock()


class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast to all connected clients, removing stale connections."""
        dead_connections = []
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)
        for dc in dead_connections:
            self.active_connections.discard(dc)


manager = ConnectionManager()


# ------------------------------------------------------------------
# REST Endpoints
# ------------------------------------------------------------------

@app.get("/api/status")
async def get_system_status():
    """Returns system status, model configuration, and server metrics."""
    return {
        "status": "online",
        "exchange": settings.DEFAULT_EXCHANGE,
        "default_symbol": settings.DEFAULT_SYMBOL,
        "llm_provider": settings.DEFAULT_LLM_PROVIDER,
        "model_analyst": settings.MODEL_ANALYST,
        "model_debater": getattr(settings, "MODEL_DEBATER", settings.MODEL_ANALYST),
        "model_reasoning": settings.MODEL_REASONING,
        "has_gemini_key": bool(settings.effective_gemini_key),
        "active_ws_clients": len(manager.active_connections)
    }


@app.get("/api/market")
async def get_market_data(symbol: str = settings.DEFAULT_SYMBOL):
    """Returns full MarketSnapshot for a symbol (live price + indicators + sentiment)."""
    try:
        snapshot = await asyncio.to_thread(
            market_feed.get_market_snapshot, symbol=symbol
        )
        return snapshot.model_dump(mode="json")
    except Exception as e:
        logger.error(f"Market data fetch failed for {symbol}: {e}")
        raise HTTPException(status_code=503, detail=f"Market data unavailable: {e}")


@app.get("/api/candles")
async def get_candles(symbol: str = settings.DEFAULT_SYMBOL, timeframe: str = "15m", limit: int = 60):
    """Returns OHLCV candle data formatted for frontend charting."""
    try:
        df = await asyncio.to_thread(
            market_feed.fetch_ohlcv_df, symbol=symbol, timeframe=timeframe, limit=limit
        )
        candles = []
        for _, row in df.iterrows():
            ts = int(row["timestamp"] / 1000) if "timestamp" in row and not row.isna().get("timestamp", False) else int(datetime.now().timestamp())
            candles.append({
                "time": ts,
                "open": round(float(row["open"]), 4),
                "high": round(float(row["high"]), 4),
                "low": round(float(row["low"]), 4),
                "close": round(float(row["close"]), 4),
                "volume": round(float(row["volume"]), 2),
            })
        return {"symbol": symbol, "timeframe": timeframe, "candles": candles}
    except Exception as e:
        logger.error(f"Candles fetch failed for {symbol}: {e}")
        raise HTTPException(status_code=503, detail=f"Candles unavailable: {e}")


@app.get("/api/scan")
async def scan_market():
    """Scans the market and returns top liquid crypto opportunities."""
    try:
        results = await asyncio.to_thread(
            market_screener.scan_top_opportunities, top_n=8
        )
        return results
    except Exception as e:
        logger.error(f"Market scan failed: {e}")
        raise HTTPException(status_code=503, detail=f"Market scan unavailable: {e}")


@app.get("/api/history")
async def get_history(limit: int = 20):
    """Returns recent decision cycle history from the database."""
    return memory_db.get_recent_cycles(limit=limit)


@app.post("/api/trigger-cycle")
async def trigger_cycle(symbol: str = settings.DEFAULT_SYMBOL):
    """
    Triggers one complete multi-agent advisory deliberation cycle.
    Uses asyncio.to_thread() to run the blocking LangGraph call without
    blocking the FastAPI event loop.
    """
    if _cycle_lock.locked():
        raise HTTPException(
            status_code=429,
            detail="A multi-agent cycle is already running. Please wait for it to complete."
        )

    async with _cycle_lock:
        logger.info(f"Triggering multi-agent advisory cycle for {symbol}")

        await manager.broadcast({
            "type": "CYCLE_START",
            "symbol": symbol,
            "message": f"Bắt đầu chu trình phân tích đa agent cho {symbol}..."
        })

        try:
            state = await asyncio.to_thread(
                trading_graph.run_cycle,
                symbol=symbol,
                timeframe=settings.DEFAULT_TIMEFRAME
            )
        except Exception as e:
            logger.error(f"Multi-agent cycle failed for {symbol}: {e}")
            await manager.broadcast({
                "type": "CYCLE_ERROR",
                "symbol": symbol,
                "error": str(e)
            })
            raise HTTPException(status_code=500, detail=f"Agent cycle error: {e}")

        # Save to memory DB
        cycle_id = memory_db.save_decision_cycle(state)
        snapshot = state.get("snapshot")
        risk_val = state.get("risk_validation")

        # Build broadcast payload
        payload = {
            "type": "CYCLE_COMPLETE",
            "cycle_id": cycle_id,
            "symbol": symbol,
            "snapshot": snapshot.model_dump(mode="json") if snapshot else None,
            "analysts": [r.model_dump(mode="json") for r in state.get("analyst_reports", [])],
            "divergence_score": float(state.get("divergence_score", 0.0)),
            "needs_debate": bool(state.get("needs_debate", False)),
            "debate_turns": [d.model_dump(mode="json") for d in state.get("debate_turns", [])],
            "raw_decision": state.get("raw_decision").model_dump(mode="json") if state.get("raw_decision") else None,
            "risk_validation": risk_val.model_dump(mode="json") if risk_val else None,
            "logs": state.get("logs", [])
        }
        await manager.broadcast(payload)
        return payload


# ------------------------------------------------------------------
# WebSocket Endpoint
# ------------------------------------------------------------------

@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status on connect
        await websocket.send_json({
            "type": "CONNECTED",
            "message": "Connected to Tauric AI Crypto Advisory Intelligence System",
            "model_analyst": settings.MODEL_ANALYST,
            "model_debater": getattr(settings, "MODEL_DEBATER", settings.MODEL_ANALYST),
            "model_reasoning": settings.MODEL_REASONING,
        })
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"type": "pong", "ts": asyncio.get_event_loop().time()})
            else:
                await websocket.send_json({"type": "ack", "received": data})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ------------------------------------------------------------------
# HTML Dashboard
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serves the main trading dashboard HTML."""
    dashboard_file = Path(__file__).resolve().parent / "dashboard.html"
    if dashboard_file.exists():
        return dashboard_file.read_text(encoding="utf-8")
    return "<h1>Dashboard file not found. Run with --mode server after building.</h1>"
