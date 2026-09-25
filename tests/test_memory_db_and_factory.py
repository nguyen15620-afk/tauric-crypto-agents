import unittest
import tempfile
import os
from pydantic import ValidationError
from storage.memory_db import MemoryDB
from execution.paper_engine import PaperExecutionEngine
from core.types import ActionEnum, RiskValidation, MarketSnapshot, DebateTurn, TradeDecision
from core.llm_factory import LLMFactory, AgentRole
from config.settings import settings


class TestMemoryDBAndFactory(unittest.TestCase):

    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        self.db = MemoryDB(db_path=self.temp_db_path)
        self.engine = PaperExecutionEngine(initial_balance=10000.0, db=self.db)

    def tearDown(self):
        os.close(self.temp_db_fd)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_save_and_retrieve_trade(self):
        trade_data = {
            "symbol": "BTC/USDT",
            "side": "BUY",
            "entry_price": 68000.0,
            "exit_price": 70000.0,
            "amount": 0.1,
            "cost_basis": 6800.0,
            "fee": 3.4,
            "stop_loss": 66000.0,
            "take_profit": 72000.0,
            "reason": "CLOSED_TP",
            "closed_at": "2026-09-25T12:00:00",
            "realized_pnl": 196.6,
            "realized_pnl_pct": 2.89
        }
        trade_id = self.db.save_trade(trade_data, cycle_id=1)
        self.assertIsInstance(trade_id, int)
        self.assertGreater(trade_id, 0)

        trades = self.db.get_recent_trades(limit=10)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["symbol"], "BTC/USDT")
        self.assertEqual(trades[0]["status"], "CLOSED_TP")
        self.assertEqual(trades[0]["realized_pnl"], 196.6)
        self.assertEqual(trades[0]["cycle_id"], 1)

    def test_paper_engine_persists_closed_trade_to_db(self):
        # Open position
        snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            current_price=68000.0,
            change_24h_pct=1.0,
            high_24h=69000.0,
            low_24h=67000.0,
            volume_24h=1000.0
        )
        risk_val = RiskValidation(
            approved=True,
            final_action=ActionEnum.BUY,
            original_action=ActionEnum.BUY,
            approved_position_size_pct=10.0,
            approved_position_usd=1000.0,
            stop_loss=66000.0,
            take_profit=72000.0
        )
        open_res = self.engine.execute_validation(risk_val, snapshot, cycle_id=42)
        self.assertEqual(open_res["status"], "FILLED")
        self.assertIn("BTC/USDT", self.engine.open_positions)

        # Close position
        close_res = self.engine.close_position(symbol="BTC/USDT", exit_price=70000.0, reason="MANUAL_TEST")
        self.assertEqual(close_res["status"], "CLOSED")
        self.assertNotIn("BTC/USDT", self.engine.open_positions)

        # Check that trade was saved to DB
        trades = self.db.get_recent_trades(limit=5)
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["symbol"], "BTC/USDT")
        self.assertEqual(trades[0]["status"], "MANUAL_TEST")

    def test_save_and_retrieve_decision_cycle(self):
        state = {
            "symbol": "ETH/USDT",
            "timeframe": "15m",
            "snapshot": MarketSnapshot(
                symbol="ETH/USDT",
                current_price=2750.0,
                change_24h_pct=2.5,
                high_24h=2800.0,
                low_24h=2700.0,
                volume_24h=50000.0
            ),
            "raw_decision": TradeDecision(
                action=ActionEnum.BUY,
                conviction=8,
                current_price=2750.0,
                stop_loss=2650.0,
                take_profit=2950.0,
                suggested_position_size_pct=10.0,
                rationale="Strong technical breakout"
            ),
            "risk_validation": RiskValidation(
                approved=True,
                final_action=ActionEnum.BUY,
                original_action=ActionEnum.BUY,
                approved_position_size_pct=10.0,
                approved_position_usd=0.0,
                stop_loss=2650.0,
                take_profit=2950.0
            ),
            "divergence_score": 0.15,
            "needs_debate": False,
            "analyst_reports": [],
            "debate_turns": []
        }
        cycle_id = self.db.save_decision_cycle(state)
        self.assertIsInstance(cycle_id, int)
        self.assertGreater(cycle_id, 0)

        cycles = self.db.get_recent_cycles(limit=10)
        self.assertEqual(len(cycles), 1)
        self.assertEqual(cycles[0]["symbol"], "ETH/USDT")
        self.assertEqual(cycles[0]["action"], "BUY")
        self.assertEqual(cycles[0]["conviction"], 8)
        self.assertEqual(cycles[0]["current_price"], 2750.0)

    def test_llm_factory_role_routing(self):
        analyst_model = LLMFactory.get_model_name_for_role(AgentRole.ANALYST)
        debater_model = LLMFactory.get_model_name_for_role(AgentRole.DEBATER)
        trader_model = LLMFactory.get_model_name_for_role(AgentRole.TRADER)

        self.assertEqual(analyst_model, settings.MODEL_ANALYST)
        self.assertEqual(debater_model, settings.MODEL_DEBATER)
        self.assertEqual(trader_model, settings.MODEL_REASONING)
        # Verify debater is routed to flash-lite, sparing reasoning quota
        self.assertEqual(debater_model, "gemini-3.5-flash-lite")

    def test_debate_turn_stance_score_validation(self):
        # Valid scores
        valid_turn = DebateTurn(
            round_num=1,
            speaker="Bull Researcher",
            argument="Valid bull thesis",
            stance_score=0.85
        )
        self.assertEqual(valid_turn.stance_score, 0.85)

        # Invalid score > 1.0 should raise ValidationError
        with self.assertRaises(ValidationError):
            DebateTurn(
                round_num=1,
                speaker="Bull Researcher",
                argument="Invalid score",
                stance_score=1.5
            )

        # Invalid score < -1.0 should raise ValidationError
        with self.assertRaises(ValidationError):
            DebateTurn(
                round_num=1,
                speaker="Bear Researcher",
                argument="Invalid score",
                stance_score=-2.0
            )


if __name__ == "__main__":
    unittest.main()
