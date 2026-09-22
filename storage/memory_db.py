import sqlite3
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from config.settings import settings
from agents.state import TradingAgentState

logger = logging.getLogger("MemoryDB")

from contextlib import contextmanager

class MemoryDB:
    """
    SQLite-backed Decision Log and Auditing Memory:
    - Stores every market state snapshot, analyst report, debate transcript, and risk decision.
    - Enables post-trade analysis, backtesting audit, and vector/similarity queries on past market conditions.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or settings.DATABASE_PATH
        self._init_tables()

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def _init_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Decision Cycles table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS decision_cycles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                timeframe TEXT,
                current_price REAL,
                action TEXT,
                approved_action TEXT,
                conviction INTEGER,
                approved BOOLEAN,
                position_size_pct REAL,
                position_size_usd REAL,
                stop_loss REAL,
                take_profit REAL,
                rationale TEXT,
                divergence_score REAL,
                needs_debate BOOLEAN,
                analysts_json TEXT,
                debate_json TEXT,
                snapshot_json TEXT,
                risk_json TEXT
            )
            """)

            # Executed Trades table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                side TEXT,
                entry_price REAL,
                amount REAL,
                usd_value REAL,
                fee REAL,
                stop_loss REAL,
                take_profit REAL,
                status TEXT, -- 'OPEN', 'CLOSED_TP', 'CLOSED_SL', 'CLOSED_MANUAL'
                exit_price REAL,
                exit_time DATETIME,
                realized_pnl REAL,
                realized_pnl_pct REAL,
                FOREIGN KEY(cycle_id) REFERENCES decision_cycles(id)
            )
            """)
            conn.commit()

    def save_decision_cycle(self, state: TradingAgentState) -> int:
        """
        Saves a completed state cycle into SQLite.
        Returns the inserted cycle ID.
        """
        snapshot = state.get("snapshot")
        decision = state.get("raw_decision")
        risk_val = state.get("risk_validation")
        
        symbol = state.get("symbol", "BTC/USDT")
        timeframe = state.get("timeframe", "15m")
        current_price = snapshot.current_price if snapshot else 0.0

        action = decision.action.value if decision else "HOLD"
        approved_action = risk_val.final_action.value if risk_val else action
        conviction = decision.conviction if decision else 0
        approved = risk_val.approved if risk_val else True
        pos_size_pct = risk_val.approved_position_size_pct if risk_val else 0.0
        pos_size_usd = risk_val.approved_position_usd if risk_val else 0.0
        stop_loss = risk_val.stop_loss if risk_val else (decision.stop_loss if decision else 0.0)
        take_profit = risk_val.take_profit if risk_val else (decision.take_profit if decision else 0.0)
        rationale = decision.rationale if decision else ""

        analysts_data = [r.model_dump(mode="json") for r in state.get("analyst_reports", [])]
        debate_data = [d.model_dump(mode="json") for d in state.get("debate_turns", [])]
        snapshot_data = snapshot.model_dump(mode="json") if snapshot else {}
        risk_data = risk_val.model_dump(mode="json") if risk_val else {}

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO decision_cycles (
                symbol, timeframe, current_price, action, approved_action, conviction,
                approved, position_size_pct, position_size_usd, stop_loss, take_profit,
                rationale, divergence_score, needs_debate, analysts_json, debate_json,
                snapshot_json, risk_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol, timeframe, current_price, action, approved_action, conviction,
                approved, pos_size_pct, pos_size_usd, stop_loss, take_profit,
                rationale, state.get("divergence_score", 0.0), state.get("needs_debate", False),
                json.dumps(analysts_data), json.dumps(debate_data),
                json.dumps(snapshot_data), json.dumps(risk_data)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_cycles(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
            SELECT * FROM decision_cycles ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            
            results = []
            for r in rows:
                item = dict(r)
                item["analysts"] = json.loads(item["analysts_json"]) if item["analysts_json"] else []
                item["debate"] = json.loads(item["debate_json"]) if item["debate_json"] else []
                item["risk"] = json.loads(item["risk_json"]) if item["risk_json"] else {}
                results.append(item)
            return results
