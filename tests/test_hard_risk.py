import unittest
from core.types import TradeDecision, MarketSnapshot, ActionEnum
from risk.hard_guardrails import HardRiskGuardrails


class TestHardRiskGuardrails(unittest.TestCase):

    def setUp(self):
        self.guard = HardRiskGuardrails({
            "max_portfolio_risk_pct": 2.0,
            "max_position_size_pct": 15.0,
            "max_open_positions": 2,
            "max_daily_drawdown_pct": 4.0,
            "min_risk_reward_ratio": 1.5,
            "default_stop_loss_pct": 2.0,
            "default_take_profit_pct": 4.0,
            "circuit_breaker": {"enabled": True, "max_price_change_15m_pct": 6.0}
        })
        self.snapshot = MarketSnapshot(
            symbol="BTC/USDT",
            current_price=68000.0,
            change_24h_pct=1.5,
            high_24h=69000.0,
            low_24h=67000.0,
            volume_24h=1500.0,
            atr_14=600.0
        )
        self.portfolio = {
            "total_equity": 10000.0,
            "cash_balance": 10000.0,
            "daily_drawdown_pct": 0.0,
            "open_positions": []
        }

    # ------------------------------------------------------------------
    # BUY tests
    # ------------------------------------------------------------------

    def test_approve_valid_buy(self):
        decision = TradeDecision(
            action=ActionEnum.BUY,
            conviction=8,
            current_price=68000.0,
            stop_loss=66640.0,   # 2% SL
            take_profit=70720.0, # 4% TP
            suggested_position_size_pct=10.0,
            rationale="Valid breakout setup"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertTrue(val.approved)
        self.assertEqual(val.final_action, ActionEnum.BUY)
        self.assertLessEqual(val.approved_position_usd, 1500.0)

    def test_reject_when_drawdown_exceeded(self):
        portfolio_dd = self.portfolio.copy()
        portfolio_dd["daily_drawdown_pct"] = 5.2
        decision = TradeDecision(
            action=ActionEnum.BUY,
            conviction=9,
            current_price=68000.0,
            stop_loss=66000.0,
            take_profit=72000.0,
            suggested_position_size_pct=10.0,
            rationale="Aggressive recovery buy"
        )
        val = self.guard.evaluate(decision, self.snapshot, portfolio_dd)
        self.assertFalse(val.approved)
        self.assertEqual(val.final_action, ActionEnum.HOLD)
        self.assertTrue(any("drawdown" in r.lower() for r in val.rejection_reasons))

    def test_auto_correct_missing_sl_tp(self):
        decision = TradeDecision(
            action=ActionEnum.BUY,
            conviction=7,
            current_price=68000.0,
            stop_loss=None,
            take_profit=None,
            suggested_position_size_pct=5.0,
            rationale="Buy without explicit levels"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertTrue(val.approved)
        self.assertLess(val.stop_loss, 68000.0, "SL must be below price for BUY")
        self.assertGreater(val.take_profit, 68000.0, "TP must be above price for BUY")

    def test_reject_low_conviction(self):
        decision = TradeDecision(
            action=ActionEnum.BUY,
            conviction=3,
            current_price=68000.0,
            stop_loss=66640.0,
            take_profit=70720.0,
            suggested_position_size_pct=5.0,
            rationale="Low confidence trade"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertFalse(val.approved)
        self.assertTrue(any("conviction" in r.lower() for r in val.rejection_reasons))

    def test_reject_invalid_zero_price(self):
        bad_snapshot = MarketSnapshot(
            symbol="SCAM/USDT",
            current_price=0.0,
            change_24h_pct=0.0,
            high_24h=0.0,
            low_24h=0.0,
            volume_24h=0.0,
        )
        decision = TradeDecision(
            action=ActionEnum.BUY,
            conviction=8,
            current_price=0.0,
            stop_loss=None,
            take_profit=None,
            rationale="Suspicious zero-price asset"
        )
        val = self.guard.evaluate(decision, bad_snapshot, self.portfolio)
        self.assertFalse(val.approved)
        self.assertTrue(any("price" in r.lower() for r in val.rejection_reasons))

    # ------------------------------------------------------------------
    # SELL tests (previously uncovered)
    # ------------------------------------------------------------------

    def test_approve_valid_sell(self):
        decision = TradeDecision(
            action=ActionEnum.SELL,
            conviction=7,
            current_price=68000.0,
            stop_loss=69360.0,   # 2% above price
            take_profit=65280.0, # 4% below price
            suggested_position_size_pct=8.0,
            rationale="Head and shoulders breakdown"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertTrue(val.approved)
        self.assertEqual(val.final_action, ActionEnum.SELL)
        self.assertGreater(val.stop_loss, 68000.0, "SELL SL must be above price")
        self.assertLess(val.take_profit, 68000.0, "SELL TP must be below price")

    def test_sell_rr_auto_adjust(self):
        """SELL with bad R:R (only 1% TP vs 2% SL) should auto-widen TP."""
        decision = TradeDecision(
            action=ActionEnum.SELL,
            conviction=7,
            current_price=68000.0,
            stop_loss=69360.0,   # 2% SL above
            take_profit=67320.0, # only 1% TP below → R:R = 0.5, below threshold
            suggested_position_size_pct=8.0,
            rationale="Selling with tight target"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertTrue(val.approved)
        # TP should be widened: 68000 - (69360 - 68000) * 1.5 = 65960
        expected_min_tp = 68000.0 - (69360.0 - 68000.0) * 1.5
        self.assertLessEqual(round(val.take_profit, 0), round(expected_min_tp, 0) + 5)
        self.assertTrue(any("r:r" in w.lower() or "sell" in w.lower() for w in val.warnings))

    def test_auto_correct_missing_sl_tp_sell(self):
        decision = TradeDecision(
            action=ActionEnum.SELL,
            conviction=6,
            current_price=68000.0,
            stop_loss=None,
            take_profit=None,
            suggested_position_size_pct=5.0,
            rationale="Sell without levels"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertTrue(val.approved)
        self.assertGreater(val.stop_loss, 68000.0, "SELL SL must be above price")
        self.assertLess(val.take_profit, 68000.0, "SELL TP must be below price")

    # ------------------------------------------------------------------
    # HOLD test
    # ------------------------------------------------------------------

    def test_hold_always_approved_with_zero_size(self):
        decision = TradeDecision(
            action=ActionEnum.HOLD,
            conviction=5,
            current_price=68000.0,
            rationale="Market conditions unclear"
        )
        val = self.guard.evaluate(decision, self.snapshot, self.portfolio)
        self.assertTrue(val.approved)
        self.assertEqual(val.final_action, ActionEnum.HOLD)
        self.assertEqual(val.approved_position_usd, 0.0)
        self.assertEqual(val.approved_position_size_pct, 0.0)


if __name__ == "__main__":
    unittest.main()
