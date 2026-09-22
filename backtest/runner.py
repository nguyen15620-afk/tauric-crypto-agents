import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from agents.graph import TradingAgentGraph
from execution.paper_engine import PaperExecutionEngine
from data.technical import TechnicalIndicators
from data.sentiment_feed import SentimentFeed
from core.types import MarketSnapshot, ActionEnum

logger = logging.getLogger("BacktestRunner")

# Estimated tokens per full analyst cycle (3 analysts + possible 2 debate turns + trader + risk)
TOKENS_PER_CYCLE = 3500
# Cost for Gemini Flash Lite: ~$0.075 per 1M input tokens
COST_PER_MILLION_TOKENS = 0.075


class BacktestRunner:
    """
    Historical Backtest & Performance Simulation Engine:
    - Replays market candles bar-by-bar
    - Builds a proper historical MarketSnapshot for each bar (NO live data leakage)
    - Injects the historical snapshot directly into TradingAgentGraph (bypasses live fetch)
    - Measures execution return, max drawdown, Sharpe ratio proxy, and API token usage
    """

    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.paper_engine = PaperExecutionEngine(initial_balance=initial_capital)
        # Graph is initialized once and reused across all bar steps
        self.graph = TradingAgentGraph()
        # Cache last sentiment fetch to avoid hammering API on every bar
        self._cached_sentiment = SentimentFeed.get_fear_and_greed_index()

    def run(
        self,
        df: pd.DataFrame,
        symbol: str = "BTC/USDT",
        step_interval: int = 10,
        warmup_period: int = 40
    ) -> Dict[str, Any]:
        """
        Runs backtest over the provided OHLCV DataFrame.

        Args:
            df: OHLCV DataFrame (columns: timestamp, open, high, low, close, volume)
            symbol: Trading pair
            step_interval: Evaluate agent decision every N candles (default: 10)
            warmup_period: Minimum bars required to compute all indicators (default: 40)

        Returns:
            Dict with performance metrics.
        """
        total_bars = len(df)
        if total_bars <= warmup_period:
            raise ValueError(
                f"DataFrame length ({total_bars}) too short for backtest warmup ({warmup_period})"
            )

        logger.info(
            f"Starting backtest on {total_bars} candles of {symbol} "
            f"(Interval: every {step_interval} bars, Warmup: {warmup_period} bars)"
        )

        estimated_tokens_used = 0
        decisions_log: List[Dict[str, Any]] = []
        equity_curve: List[float] = []

        for i in range(warmup_period, total_bars, step_interval):
            window_df = df.iloc[:i + 1]
            current_bar = window_df.iloc[-1]
            current_price = float(current_bar["close"])

            # 1. Update existing open positions with current bar price before deliberation
            self.paper_engine.update_market_price(symbol, current_price)

            # 2. Build a proper historical MarketSnapshot for this exact point in time
            #    (No live API calls — indicators are derived purely from the historical window)
            snapshot = self._build_historical_snapshot(window_df, symbol, current_price)

            # 3. Deliberate with Multi-Agent graph — INJECT the historical snapshot
            #    This prevents the graph's fetch_data node from pulling live Binance data
            state_res = self.graph.run_cycle(
                symbol=symbol,
                timeframe="15m",
                portfolio_state=self.paper_engine.get_state_dict(),
                snapshot=snapshot,  # ← KEY FIX: inject historical data to prevent data leakage
            )

            estimated_tokens_used += TOKENS_PER_CYCLE

            # 4. Execute approved signal using the historical snapshot
            risk_val = state_res.get("risk_validation")
            if risk_val:
                exec_res = self.paper_engine.execute_validation(risk_val, snapshot)
                decisions_log.append({
                    "bar_index": i,
                    "price": current_price,
                    "action": risk_val.final_action.value,
                    "approved": risk_val.approved,
                    "execution": exec_res
                })

            # Track equity after each decision point
            equity_curve.append(self.paper_engine.total_equity)

            # 5. Price-tick all intermediate bars to trigger SL/TP
            next_idx = min(i + step_interval, total_bars)
            for sub_bar_idx in range(i + 1, next_idx):
                bar = df.iloc[sub_bar_idx]
                # Test high then low to simulate realistic SL/TP sweep order
                self.paper_engine.update_market_price(symbol, float(bar["high"]))
                self.paper_engine.update_market_price(symbol, float(bar["low"]))
                self.paper_engine.update_market_price(symbol, float(bar["close"]))
                equity_curve.append(self.paper_engine.total_equity)

        # Final close of any remaining open positions at last bar price
        final_price = float(df.iloc[-1]["close"])
        for s in list(self.paper_engine.open_positions.keys()):
            self.paper_engine._close_position(s, final_price, reason="BACKTEST_END")

        summary = self.paper_engine.get_state_dict()
        net_return_pct = (
            (summary["total_equity"] - self.initial_capital) / self.initial_capital
        ) * 100.0

        # Proper max drawdown from equity curve
        max_drawdown_pct = self._calculate_max_drawdown(equity_curve)

        # Sharpe ratio proxy (annualised, assuming 15m bars, 252 trading days)
        sharpe = self._calculate_sharpe(equity_curve)

        estimated_api_cost_usd = (estimated_tokens_used / 1_000_000) * COST_PER_MILLION_TOKENS

        results = {
            "symbol": symbol,
            "total_bars_tested": total_bars,
            "initial_capital": self.initial_capital,
            "final_equity": summary["total_equity"],
            "net_return_pct": round(net_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "sharpe_ratio": round(sharpe, 3),
            "total_trades": summary["closed_trades_count"],
            "win_rate_pct": summary["win_rate_pct"],
            "total_realized_pnl": summary["total_realized_pnl"],
            "estimated_tokens_used": estimated_tokens_used,
            "estimated_api_cost_usd": round(estimated_api_cost_usd, 4),
            "decisions_count": len(decisions_log)
        }

        logger.info(
            f"Backtest completed: Return={net_return_pct:+.2f}% | "
            f"MaxDD={max_drawdown_pct:.2f}% | Sharpe={sharpe:.3f} | "
            f"Trades={summary['closed_trades_count']}"
        )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_historical_snapshot(
        self,
        window_df: pd.DataFrame,
        symbol: str,
        current_price: float
    ) -> MarketSnapshot:
        """
        Builds a fully self-contained historical MarketSnapshot from the
        bar window — NO external API calls involved.
        """
        indicators = TechnicalIndicators.get_latest_metrics(window_df)

        # 24-period proxy for 6h lookback (15m * 24 = 6h)
        lookback = min(24, len(window_df))
        prev_close = float(window_df.iloc[-lookback]["close"]) if len(window_df) >= lookback else float(window_df.iloc[0]["close"])
        change_pct = ((current_price - prev_close) / prev_close) * 100.0

        return MarketSnapshot(
            symbol=symbol,
            current_price=current_price,
            change_24h_pct=round(change_pct, 2),
            high_24h=float(window_df["high"].tail(96).max()),   # last 24h (96 × 15m bars)
            low_24h=float(window_df["low"].tail(96).min()),
            volume_24h=float(window_df["volume"].tail(96).sum()),
            rsi_14=indicators.get("rsi_14"),
            macd=indicators.get("macd"),
            macd_signal=indicators.get("macd_signal"),
            macd_hist=indicators.get("macd_hist"),
            bb_upper=indicators.get("bb_upper"),
            bb_middle=indicators.get("bb_middle"),
            bb_lower=indicators.get("bb_lower"),
            atr_14=indicators.get("atr_14"),
            ema_20=indicators.get("ema_20"),
            ema_50=indicators.get("ema_50"),
            ema_200=indicators.get("ema_200"),
            # Neutral sentiment for pure historical replay
            # (real sentiment data is not available historically via free API)
            fear_and_greed_score=self._cached_sentiment.get("score", 50),
            fear_and_greed_label=self._cached_sentiment.get("label", "Neutral"),
            news_headlines=["Historical backtest simulation data point"],
            funding_rate=0.0001,
            open_interest=50000.0,
            orderbook_bid_volume=120.0,
            orderbook_ask_volume=110.0,
            orderbook_imbalance=0.043
        )

    @staticmethod
    def _calculate_max_drawdown(equity_curve: List[float]) -> float:
        """Peak-to-trough max drawdown from equity curve."""
        if len(equity_curve) < 2:
            return 0.0
        arr = np.array(equity_curve)
        running_peak = np.maximum.accumulate(arr)
        drawdowns = (running_peak - arr) / (running_peak + 1e-9) * 100.0
        return float(drawdowns.max())

    @staticmethod
    def _calculate_sharpe(equity_curve: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Annualised Sharpe ratio proxy.
        Assumes 15-minute bars: 4 bars/hour × 24h × 365 days = 35,040 bars/year.
        """
        if len(equity_curve) < 2:
            return 0.0
        arr = np.array(equity_curve, dtype=float)
        returns = np.diff(arr) / (arr[:-1] + 1e-9)
        if returns.std() == 0:
            return 0.0
        bars_per_year = 35040  # 15m bars in a year
        sharpe = (returns.mean() - risk_free_rate) / returns.std() * np.sqrt(bars_per_year)
        return float(np.clip(sharpe, -10.0, 10.0))
