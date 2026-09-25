import logging
from datetime import datetime, date, timezone
from typing import Dict, Any, List, Optional
from core.types import ActionEnum, RiskValidation, MarketSnapshot, Position
from config.settings import RISK_RULES
from storage.memory_db import MemoryDB

logger = logging.getLogger("PaperEngine")


class PaperExecutionEngine:
    """
    Simulates high-fidelity paper trading execution:
    - Tracks cash balance and total portfolio equity.
    - Applies realistic taker fees (0.05%) and slippage (0.03%).
    - Automatically triggers Stop-Loss (SL) and Take-Profit (TP) on price updates.
    - Computes real-time win rate and proper drawdown metrics:
        * max_drawdown_pct   → all-time peak-to-trough drawdown (for reporting)
        * daily_drawdown_pct → intraday drawdown from session-open equity (for circuit-breaker)
    """

    def __init__(self, initial_balance: float = 10000.0, db: Optional[MemoryDB] = None):
        cfg = RISK_RULES.get("paper_trading", {})
        self.initial_balance = float(cfg.get("initial_balance_usdt", initial_balance))
        self.cash_balance = self.initial_balance
        self.taker_fee_pct = float(cfg.get("taker_fee_pct", 0.05)) / 100.0
        self.slippage_pct = float(cfg.get("slippage_pct", 0.03)) / 100.0

        self.open_positions: Dict[str, Position] = {}
        self.closed_trades: List[Dict[str, Any]] = []

        # --- Drawdown tracking ---
        # all-time peak for max_drawdown
        self.peak_equity = self.initial_balance
        # session-open equity (reset each calendar day) for daily circuit-breaker
        self._session_open_equity = self.initial_balance
        self._session_date = date.today()

        self.db = db or MemoryDB()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_equity(self) -> float:
        pos_value = sum([p.amount * p.current_price for p in self.open_positions.values()])
        return round(self.cash_balance + pos_value, 2)

    def _refresh_session(self):
        """Reset session-open equity when a new calendar day starts."""
        today = date.today()
        if today != self._session_date:
            self._session_open_equity = self.total_equity
            self._session_date = today

    @property
    def daily_drawdown_pct(self) -> float:
        """
        Intraday drawdown from session-open equity.
        Used by Hard Guardrails daily circuit-breaker.
        Resets at midnight.
        """
        self._refresh_session()
        eq = self.total_equity
        if self._session_open_equity <= 0:
            return 0.0
        dd = ((self._session_open_equity - eq) / self._session_open_equity) * 100.0
        return round(max(dd, 0.0), 2)

    @property
    def max_drawdown_pct(self) -> float:
        """
        All-time peak-to-trough drawdown since engine start.
        Used for performance reporting.
        """
        eq = self.total_equity
        if eq > self.peak_equity:
            self.peak_equity = eq
        if self.peak_equity <= 0:
            return 0.0
        dd = ((self.peak_equity - eq) / self.peak_equity) * 100.0
        return round(dd, 2)

    def get_state_dict(self) -> Dict[str, Any]:
        return {
            "initial_balance": self.initial_balance,
            "cash_balance": round(self.cash_balance, 2),
            "total_equity": self.total_equity,
            "peak_equity": round(self.peak_equity, 2),
            # Expose both metrics clearly
            "daily_drawdown_pct": self.daily_drawdown_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "open_positions": [p.model_dump(mode="json") for p in self.open_positions.values()],
            "closed_trades": list(reversed(self.closed_trades[-30:])),
            "closed_trades_count": len(self.closed_trades),
            "win_rate_pct": self._calculate_win_rate(),
            "total_realized_pnl": round(sum([t.get("realized_pnl", 0.0) for t in self.closed_trades]), 2)
        }

    def _calculate_win_rate(self) -> float:
        if not self.closed_trades:
            return 0.0
        wins = sum([1 for t in self.closed_trades if t.get("realized_pnl", 0.0) > 0])
        return round((wins / len(self.closed_trades)) * 100.0, 1)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_validation(
        self,
        risk_val: RiskValidation,
        snapshot: MarketSnapshot,
        cycle_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes an approved risk recommendation on the paper portfolio.
        """
        symbol = snapshot.symbol
        current_price = snapshot.current_price

        # 1. Update any existing positions first with latest market price
        self.update_market_price(symbol, current_price)

        action = risk_val.final_action
        usd_size = risk_val.approved_position_usd

        # If HOLD, no execution
        if action == ActionEnum.HOLD or not risk_val.approved:
            return {"status": "NO_ACTION", "reason": "Action is HOLD or Risk Rejected."}

        # BUY (Open Long or Accumulate)
        if action == ActionEnum.BUY:
            if symbol in self.open_positions:
                return {"status": "SKIPPED", "reason": f"Position for {symbol} is already open."}

            if self.cash_balance < usd_size or usd_size <= 0:
                return {"status": "INSUFFICIENT_FUNDS", "reason": f"Required ${usd_size:.2f} > Cash ${self.cash_balance:.2f}"}

            # Calculate fill price including slippage
            fill_price = current_price * (1.0 + self.slippage_pct)
            fee = usd_size * self.taker_fee_pct
            total_cost = usd_size + fee

            if self.cash_balance < total_cost:
                usd_size = self.cash_balance - fee
                total_cost = self.cash_balance

            amount = usd_size / fill_price
            self.cash_balance -= total_cost

            pos = Position(
                symbol=symbol,
                side=ActionEnum.BUY,
                entry_price=round(fill_price, 2),
                amount=round(amount, 6),
                current_price=round(current_price, 2),
                stop_loss=risk_val.stop_loss,
                take_profit=risk_val.take_profit,
                entry_time=datetime.now(timezone.utc),
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0
            )
            self.open_positions[symbol] = pos

            log_msg = (
                f"[Paper Trade] BOUGHT {pos.amount} {symbol} @ ${pos.entry_price:,.2f} "
                f"| Size: ${usd_size:,.2f} | SL: ${pos.stop_loss} | TP: ${pos.take_profit}"
            )
            logger.info(log_msg)

            return {
                "status": "FILLED",
                "side": "BUY",
                "amount": pos.amount,
                "price": pos.entry_price,
                "fee": round(fee, 2),
                "usd_value": round(usd_size, 2)
            }

        # SELL (Close position)
        elif action == ActionEnum.SELL:
            if symbol in self.open_positions:
                return self._close_position(symbol, current_price, reason="CHIEF_TRADER_SELL", cycle_id=cycle_id)
            else:
                return {"status": "SKIPPED", "reason": f"No open position to sell for {symbol}."}

        return {"status": "IGNORED"}

    def update_market_price(self, symbol: str, current_price: float):
        """
        Updates live price for open positions and checks automated SL/TP triggers.
        """
        if symbol not in self.open_positions:
            return

        pos = self.open_positions[symbol]
        pos.current_price = round(current_price, 2)

        # Calculate unrealized PnL
        diff = (current_price - pos.entry_price) * pos.amount
        pos.unrealized_pnl = round(diff, 2)
        pos.unrealized_pnl_pct = round(((current_price - pos.entry_price) / pos.entry_price) * 100.0, 2)

        # Check automated Stop Loss trigger
        if pos.stop_loss and current_price <= pos.stop_loss:
            logger.warning(
                f"[Paper Trade] STOP LOSS TRIGGERED for {symbol} at ${current_price:,.2f} (SL: ${pos.stop_loss:,.2f})"
            )
            self._close_position(symbol, current_price, reason="CLOSED_SL")

        # Check automated Take Profit trigger
        elif pos.take_profit and current_price >= pos.take_profit:
            logger.info(
                f"[Paper Trade] TAKE PROFIT TRIGGERED for {symbol} at ${current_price:,.2f} (TP: ${pos.take_profit:,.2f})"
            )
            self._close_position(symbol, current_price, reason="CLOSED_TP")

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "MANUAL",
        cycle_id: Optional[int] = None
    ) -> Dict[str, Any]:
        pos = self.open_positions.pop(symbol, None)
        if not pos:
            return {"status": "ERROR", "message": f"No position found for {symbol}"}

        # Include slippage & fee on exit
        fill_price = exit_price * (1.0 - self.slippage_pct)
        gross_return = pos.amount * fill_price
        exit_fee = gross_return * self.taker_fee_pct
        net_return = gross_return - exit_fee

        cost_basis = pos.amount * pos.entry_price
        realized_pnl = net_return - cost_basis
        realized_pnl_pct = (realized_pnl / cost_basis) * 100.0 if cost_basis > 0 else 0.0

        self.cash_balance += net_return

        trade_record = {
            "symbol": symbol,
            "side": pos.side.value,
            "entry_price": pos.entry_price,
            "exit_price": round(fill_price, 2),
            "amount": pos.amount,
            "cost_basis": round(cost_basis, 2),
            "net_return": round(net_return, 2),
            "realized_pnl": round(realized_pnl, 2),
            "realized_pnl_pct": round(realized_pnl_pct, 2),
            "fee": round(exit_fee, 2),
            "reason": reason,
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "stop_loss": pos.stop_loss,
            "take_profit": pos.take_profit,
        }
        self.closed_trades.append(trade_record)

        # Persist trade record to SQLite
        if self.db:
            try:
                self.db.save_trade(trade_record, cycle_id=cycle_id)
            except Exception as e:
                logger.warning(f"Failed to persist trade to database: {e}")

        logger.info(
            f"[Paper Trade] Closed {symbol} ({reason}): PnL: ${realized_pnl:+,.2f} ({realized_pnl_pct:+.2f}%)"
        )
        return {"status": "CLOSED", **trade_record}

    def close_position(self, symbol: str, exit_price: Optional[float] = None, reason: str = "MANUAL") -> Dict[str, Any]:
        """Public method to manually close an open position."""
        pos = self.open_positions.get(symbol)
        if not pos:
            return {"status": "ERROR", "message": f"Không tìm thấy vị thế mở cho {symbol}"}
        price = exit_price if exit_price is not None else pos.current_price
        return self._close_position(symbol=symbol, exit_price=price, reason=reason)
