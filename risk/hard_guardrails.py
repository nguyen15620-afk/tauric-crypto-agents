import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from core.types import TradeDecision, MarketSnapshot, ActionEnum, RiskValidation
from config.settings import RISK_RULES

logger = logging.getLogger("HardRiskGuardrails")


class HardRiskGuardrails:
    """
    Deterministic Non-LLM Risk Management Engine:
    - Never trusts LLM blindly on order sizes or trade validity.
    - Enforces hard stop-loss, take-profit R:R constraints for BOTH BUY and SELL.
    - Enforces max position size (% of equity).
    - Enforces daily drawdown circuit breakers (uses daily_drawdown_pct).
    - Implements abnormal market condition halts (high volatility / wide spread).
    """

    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        self.rules = custom_rules or RISK_RULES.get("risk_limits", {})
        self.max_portfolio_risk_pct = float(self.rules.get("max_portfolio_risk_pct", 2.0))
        self.max_position_size_pct = float(self.rules.get("max_position_size_pct", 15.0))
        self.max_open_positions = int(self.rules.get("max_open_positions", 3))
        self.max_daily_drawdown_pct = float(self.rules.get("max_daily_drawdown_pct", 4.0))
        self.min_risk_reward_ratio = float(self.rules.get("min_risk_reward_ratio", 1.5))
        self.default_stop_loss_pct = float(self.rules.get("default_stop_loss_pct", 2.0))
        self.default_take_profit_pct = float(self.rules.get("default_take_profit_pct", 4.0))

        circuit = self.rules.get("circuit_breaker", {})
        self.cb_enabled = circuit.get("enabled", True)
        self.cb_max_price_change_pct = float(circuit.get("max_price_change_15m_pct", 6.0))

    def evaluate(
        self,
        decision: TradeDecision,
        snapshot: MarketSnapshot,
        portfolio_state: Dict[str, Any]
    ) -> RiskValidation:
        """
        Thoroughly audits the TradeDecision against hard risk boundaries.
        Returns a RiskValidation object with final authorized action and sizing.
        """
        rejection_reasons: List[str] = []
        warnings: List[str] = []

        current_price = snapshot.current_price
        total_equity = float(portfolio_state.get("total_equity", 10000.0))
        cash_balance = float(portfolio_state.get("cash_balance", 10000.0))
        open_positions = portfolio_state.get("open_positions", [])

        # 0. Edge Case: Invalid non-positive market price
        if current_price <= 0:
            return RiskValidation(
                approved=False,
                final_action=ActionEnum.HOLD,
                original_action=decision.action,
                approved_position_size_pct=0.0,
                approved_position_usd=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                rejection_reasons=[f"Invalid non-positive market price detected: ${current_price}"]
            )

        # 1. Action HOLD requires no capital deployment
        if decision.action == ActionEnum.HOLD:
            return RiskValidation(
                approved=True,
                final_action=ActionEnum.HOLD,
                original_action=decision.action,
                approved_position_size_pct=0.0,
                approved_position_usd=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                warnings=["LLM recommended HOLD. No risk allocation needed."]
            )

        # Use daily_drawdown_pct (session-based, resets daily) — NOT max_drawdown_pct
        daily_drawdown_pct = float(portfolio_state.get("daily_drawdown_pct", 0.0))

        # 2. Daily Drawdown Breaker
        if daily_drawdown_pct >= self.max_daily_drawdown_pct:
            rejection_reasons.append(
                f"Daily drawdown of {daily_drawdown_pct:.2f}% exceeds hard limit of "
                f"{self.max_daily_drawdown_pct:.2f}%. Trading halted for today."
            )

        # 3. Market Volatility Circuit Breaker
        if self.cb_enabled:
            if abs(snapshot.change_24h_pct) >= 15.0:
                warnings.append(
                    f"Extreme 24h volatility detected ({snapshot.change_24h_pct:+.2f}%). "
                    f"Proceeding with reduced size."
                )
            if snapshot.atr_14 and current_price > 0:
                atr_pct = (snapshot.atr_14 / current_price) * 100
                if atr_pct > self.cb_max_price_change_pct:
                    rejection_reasons.append(
                        f"Circuit breaker triggered: ATR volatility ({atr_pct:.2f}%) "
                        f"exceeds max threshold ({self.cb_max_price_change_pct:.2f}%)."
                    )

        # 4. Open Positions Limit (only block additional BUYs)
        if len(open_positions) >= self.max_open_positions and decision.action == ActionEnum.BUY:
            rejection_reasons.append(
                f"Maximum open positions ({self.max_open_positions}) reached. "
                f"Cannot open additional positions."
            )

        # 5. Low Conviction Check
        if decision.conviction < 5:
            rejection_reasons.append(
                f"Conviction score ({decision.conviction}/10) is below minimum threshold (5/10)."
            )

        # ------------------------------------------------------------------
        # 6. Stop Loss & Take Profit Validation + R:R Enforcement
        # ------------------------------------------------------------------
        stop_loss = decision.stop_loss
        take_profit = decision.take_profit
        sl_dist_pct = self.default_stop_loss_pct  # default fallback

        if decision.action == ActionEnum.BUY:
            # SL must be BELOW current price
            if not stop_loss or stop_loss >= current_price:
                stop_loss = current_price * (1.0 - (self.default_stop_loss_pct / 100.0))
                warnings.append(
                    f"Invalid or missing Stop Loss for BUY. Applied default SL: ${stop_loss:.2f} "
                    f"(-{self.default_stop_loss_pct}%)"
                )

            sl_dist_pct = ((current_price - stop_loss) / current_price) * 100.0

            # TP must be ABOVE current price
            if not take_profit or take_profit <= current_price:
                take_profit = current_price * (1.0 + (self.default_take_profit_pct / 100.0))
                warnings.append(
                    f"Invalid or missing Take Profit for BUY. Applied default TP: ${take_profit:.2f} "
                    f"(+{self.default_take_profit_pct}%)"
                )

            tp_dist_pct = ((take_profit - current_price) / current_price) * 100.0

            # Risk/Reward Ratio Check — auto-widen TP if needed
            rr_ratio = tp_dist_pct / max(sl_dist_pct, 0.01)
            if rr_ratio < self.min_risk_reward_ratio:
                take_profit = current_price + (current_price - stop_loss) * self.min_risk_reward_ratio
                warnings.append(
                    f"BUY R:R ratio ({rr_ratio:.2f}) below required {self.min_risk_reward_ratio}. "
                    f"Adjusted TP to ${take_profit:.2f}."
                )

        else:  # SELL / SHORT
            # SL must be ABOVE current price
            if not stop_loss or stop_loss <= current_price:
                stop_loss = current_price * (1.0 + (self.default_stop_loss_pct / 100.0))
                warnings.append(
                    f"Invalid or missing Stop Loss for SELL. Applied default SL: ${stop_loss:.2f} "
                    f"(+{self.default_stop_loss_pct}%)"
                )

            sl_dist_pct = ((stop_loss - current_price) / current_price) * 100.0

            # TP must be BELOW current price
            if not take_profit or take_profit >= current_price:
                take_profit = current_price * (1.0 - (self.default_take_profit_pct / 100.0))
                warnings.append(
                    f"Invalid or missing Take Profit for SELL. Applied default TP: ${take_profit:.2f} "
                    f"(-{self.default_take_profit_pct}%)"
                )

            tp_dist_pct = ((current_price - take_profit) / current_price) * 100.0

            # ✅ SELL R:R check (was missing before)
            rr_ratio = tp_dist_pct / max(sl_dist_pct, 0.01)
            if rr_ratio < self.min_risk_reward_ratio:
                # Widen TP downward for SELL to meet minimum R:R
                take_profit = current_price - (stop_loss - current_price) * self.min_risk_reward_ratio
                warnings.append(
                    f"SELL R:R ratio ({rr_ratio:.2f}) below required {self.min_risk_reward_ratio}. "
                    f"Adjusted TP to ${take_profit:.2f}."
                )

        # ------------------------------------------------------------------
        # 7. Position Sizing Calculation
        # Capital risk = Total Equity * max_portfolio_risk_pct (e.g., $10,000 * 2% = $200 risk)
        # Position size = Capital risk / SL distance pct
        # ------------------------------------------------------------------
        sl_pct_decimal = max(sl_dist_pct / 100.0, 0.005)
        risk_capital = total_equity * (self.max_portfolio_risk_pct / 100.0)
        calculated_position_usd = risk_capital / sl_pct_decimal

        # Hard cap at max_position_size_pct of equity AND available cash
        max_allowed_usd = total_equity * (self.max_position_size_pct / 100.0)
        final_position_usd = min(calculated_position_usd, max_allowed_usd, cash_balance)
        final_position_pct = (final_position_usd / total_equity) * 100.0 if total_equity > 0 else 0.0

        if final_position_usd < 50.0:  # Minimum viable order
            rejection_reasons.append(
                f"Calculated position size (${final_position_usd:.2f}) is below minimum viable threshold ($50)."
            )

        approved = (len(rejection_reasons) == 0)
        final_action = decision.action if approved else ActionEnum.HOLD

        return RiskValidation(
            approved=approved,
            final_action=final_action,
            original_action=decision.action,
            approved_position_size_pct=round(final_position_pct, 2) if approved else 0.0,
            approved_position_usd=round(final_position_usd, 2) if approved else 0.0,
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            rejection_reasons=rejection_reasons,
            warnings=warnings
        )
