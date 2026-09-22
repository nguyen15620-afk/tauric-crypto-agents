import unittest
from agents.graph import TradingAgentGraph
from core.types import ActionEnum

class TestGraphIntegration(unittest.TestCase):
    def setUp(self):
        self.graph = TradingAgentGraph()

    def test_full_agent_cycle(self):
        state = self.graph.run_cycle(
            symbol="BTC/USDT",
            timeframe="15m",
            portfolio_state={
                "total_equity": 10000.0,
                "cash_balance": 10000.0,
                "open_positions": [],
                "daily_drawdown_pct": 0.0
            }
        )

        self.assertIsNotNone(state.get("snapshot"))
        self.assertEqual(len(state.get("analyst_reports", [])), 3)
        self.assertIsNotNone(state.get("raw_decision"))
        self.assertIsNotNone(state.get("risk_validation"))
        
        # Verify valid enum action
        self.assertIn(state["raw_decision"].action, [ActionEnum.BUY, ActionEnum.SELL, ActionEnum.HOLD])
        self.assertIn(state["risk_validation"].final_action, [ActionEnum.BUY, ActionEnum.SELL, ActionEnum.HOLD])
        
        # Verify logs recorded
        self.assertGreater(len(state.get("logs", [])), 0)

if __name__ == "__main__":
    unittest.main()
