import logging
from typing import Dict, Any, Literal, Optional
from langgraph.graph import StateGraph, END
from agents.state import TradingAgentState
from data.ccxt_feed import CCXTMarketFeed
from agents.technical_analyst import TechnicalAnalyst
from agents.sentiment_analyst import SentimentAnalyst
from agents.onchain_analyst import OnChainAnalyst
from agents.validator import BeliefValidator
from agents.debate import BullResearcher, BearResearcher
from agents.trader_agent import ChiefTrader
from risk.hard_guardrails import HardRiskGuardrails
from config.settings import RISK_RULES
from core.types import MarketSnapshot

logger = logging.getLogger("TradingGraph")


class TradingAgentGraph:
    """
    Compiles and executes the LangGraph Multi-Agent Trading System:
    Data -> Analysts -> Validator -> (Debate Loop if divergence) -> Trader -> Hard Risk -> Execution

    Supports two data modes:
    - LIVE mode (default): fetches real-time market data from exchange
    - HISTORICAL mode: accepts a pre-built MarketSnapshot (used by backtester)
    """

    def __init__(self, exchange_feed: CCXTMarketFeed = None):
        self.feed = exchange_feed or CCXTMarketFeed()
        self.tech_analyst = TechnicalAnalyst()
        self.sent_analyst = SentimentAnalyst()
        self.flow_analyst = OnChainAnalyst()
        self.validator = BeliefValidator()
        self.bull_researcher = BullResearcher()
        self.bear_researcher = BearResearcher()
        self.trader = ChiefTrader()
        self.hard_risk = HardRiskGuardrails()
        self.max_debate_rounds = int(RISK_RULES.get("debate", {}).get("max_debate_rounds", 2))

        self.workflow = self._build_graph()
        self.app = self.workflow.compile()

    # --- Node Definitions ---

    def node_fetch_market_data(self, state: TradingAgentState) -> Dict[str, Any]:
        """
        Fetches live market data. Skipped if snapshot was pre-injected
        (e.g., during historical backtest to avoid data leakage).
        """
        # If a snapshot was already injected (backtest mode), skip fetching live data
        if state.get("snapshot") is not None:
            logger.info("[Graph] Pre-injected snapshot detected — skipping live data fetch (backtest mode).")
            return {
                "logs": [
                    f"[Backtest] Sử dụng snapshot lịch sử đã inject: "
                    f"{state['snapshot'].symbol} @ ${state['snapshot'].current_price:,.2f}"
                ]
            }

        symbol = state.get("symbol", "BTC/USDT")
        timeframe = state.get("timeframe", "15m")
        logger.info(f"[Graph] Fetching live market snapshot for {symbol} ({timeframe})")
        snapshot = self.feed.get_market_snapshot(symbol=symbol, timeframe=timeframe)
        return {
            "snapshot": snapshot,
            "logs": [
                f"Tải thành công nến {symbol}: Giá ${snapshot.current_price:,.2f} "
                f"({snapshot.change_24h_pct:+.2f}%)"
            ]
        }

    def node_run_analysts(self, state: TradingAgentState) -> Dict[str, Any]:
        snapshot = state["snapshot"]
        logger.info("[Graph] Running Technical, Sentiment, and On-chain Analysts")

        ta_report = self.tech_analyst.analyze(snapshot)
        sent_report = self.sent_analyst.analyze(snapshot)
        flow_report = self.flow_analyst.analyze(snapshot)

        reports = [ta_report, sent_report, flow_report]
        logs = [
            f"Technical Analyst: {ta_report.belief.bias} (Score: {ta_report.belief.score})",
            f"Sentiment Analyst: {sent_report.belief.bias} (Score: {sent_report.belief.score})",
            f"On-chain Analyst: {flow_report.belief.bias} (Score: {flow_report.belief.score})",
        ]
        return {
            "analyst_reports": reports,
            "logs": logs
        }

    def node_validate_consensus(self, state: TradingAgentState) -> Dict[str, Any]:
        reports = state.get("analyst_reports", [])
        divergence_score, needs_debate, explanation = self.validator.evaluate_consensus(reports)

        return {
            "divergence_score": divergence_score,
            "needs_debate": needs_debate,
            "debate_round": 1 if needs_debate else 0,
            "debate_turns": [],
            "logs": [explanation]
        }

    def node_bull_debate(self, state: TradingAgentState) -> Dict[str, Any]:
        snapshot = state["snapshot"]
        reports = state["analyst_reports"]
        current_round = state.get("debate_round", 1)
        turns = list(state.get("debate_turns", []))

        # Get previous bear argument if any
        prev_bear = None
        for t in reversed(turns):
            if t.speaker == "Bear Researcher":
                prev_bear = t.argument
                break

        turn = self.bull_researcher.speak(snapshot, reports, prev_bear, round_num=current_round)
        turns.append(turn)
        return {
            "debate_turns": turns,
            "logs": [f"Bull Researcher (Vòng {current_round}): {turn.argument[:100]}..."]
        }

    def node_bear_debate(self, state: TradingAgentState) -> Dict[str, Any]:
        snapshot = state["snapshot"]
        reports = state["analyst_reports"]
        current_round = state.get("debate_round", 1)
        turns = list(state.get("debate_turns", []))

        # Get latest bull argument
        prev_bull = turns[-1].argument if turns else None
        turn = self.bear_researcher.speak(snapshot, reports, prev_bull, round_num=current_round)
        turns.append(turn)

        return {
            "debate_turns": turns,
            "debate_round": current_round + 1,
            "logs": [f"Bear Researcher (Vòng {current_round}): {turn.argument[:100]}..."]
        }

    def node_synthesize_trader(self, state: TradingAgentState) -> Dict[str, Any]:
        snapshot = state["snapshot"]
        reports = state.get("analyst_reports", [])
        debate_turns = state.get("debate_turns", [])

        decision = self.trader.synthesize(snapshot, reports, debate_turns)
        log = (
            f"Chief Trader ra quyết định: {decision.action.value} | Độ tự tin: {decision.conviction}/10 "
            f"| SL: ${decision.stop_loss} | TP: ${decision.take_profit}"
        )
        return {
            "raw_decision": decision,
            "logs": [log]
        }

    def node_hard_risk_guard(self, state: TradingAgentState) -> Dict[str, Any]:
        decision = state["raw_decision"]
        snapshot = state["snapshot"]
        portfolio = state.get("portfolio_state", {})

        risk_val = self.hard_risk.evaluate(decision, snapshot, portfolio)

        status_txt = "PHÊ DUYỆT" if risk_val.approved else "TỪ CHỐI / ĐIỀU CHỈNH"
        log = (
            f"Risk Guardrails: {status_txt} -> Lệnh cuối: {risk_val.final_action.value} "
            f"| Khối lượng: {risk_val.approved_position_size_pct}% (${risk_val.approved_position_usd:,.2f})"
        )
        return {
            "risk_validation": risk_val,
            "logs": [log]
        }

    # --- Edge Conditionals ---

    def route_after_validator(self, state: TradingAgentState) -> Literal["bull_debate", "chief_trader"]:
        if state.get("needs_debate", False):
            return "bull_debate"
        return "chief_trader"

    def route_after_bear(self, state: TradingAgentState) -> Literal["bull_debate", "chief_trader"]:
        round_num = state.get("debate_round", 1)
        if round_num <= self.max_debate_rounds:
            return "bull_debate"
        return "chief_trader"

    # --- Graph Compilation ---

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(TradingAgentState)

        # Add Nodes
        builder.add_node("fetch_data", self.node_fetch_market_data)
        builder.add_node("analysts", self.node_run_analysts)
        builder.add_node("validator", self.node_validate_consensus)
        builder.add_node("bull_debate", self.node_bull_debate)
        builder.add_node("bear_debate", self.node_bear_debate)
        builder.add_node("chief_trader", self.node_synthesize_trader)
        builder.add_node("hard_risk", self.node_hard_risk_guard)

        # Set Entrypoint
        builder.set_entry_point("fetch_data")

        # Linear transitions
        builder.add_edge("fetch_data", "analysts")
        builder.add_edge("analysts", "validator")

        # Conditional route: Consensus vs Debate
        builder.add_conditional_edges(
            "validator",
            self.route_after_validator,
            {
                "bull_debate": "bull_debate",
                "chief_trader": "chief_trader"
            }
        )

        # Debate Loop
        builder.add_edge("bull_debate", "bear_debate")
        builder.add_conditional_edges(
            "bear_debate",
            self.route_after_bear,
            {
                "bull_debate": "bull_debate",
                "chief_trader": "chief_trader"
            }
        )

        # Final Risk Evaluation
        builder.add_edge("chief_trader", "hard_risk")
        builder.add_edge("hard_risk", END)

        return builder

    def run_cycle(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "15m",
        portfolio_state: Dict[str, Any] = None,
        snapshot: Optional[MarketSnapshot] = None,  # ← Inject for backtest (avoids data leakage)
    ) -> TradingAgentState:
        """
        Executes one full multi-agent deliberation cycle.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '15m')
            portfolio_state: Current portfolio state dict from PaperExecutionEngine
            snapshot: Optional pre-built MarketSnapshot. When provided (backtest mode),
                      the fetch_data node is skipped to prevent future data leakage.
        """
        initial_state: TradingAgentState = {
            "symbol": symbol,
            "timeframe": timeframe,
            # Pre-inject snapshot if provided (backtest mode prevents live data fetch)
            "snapshot": snapshot,
            "portfolio_state": portfolio_state or {
                "total_equity": 10000.0,
                "cash_balance": 10000.0,
                "open_positions": [],
                "daily_drawdown_pct": 0.0,
            },
            "analyst_reports": [],
            "divergence_score": 0.0,
            "needs_debate": False,
            "debate_turns": [],
            "debate_round": 0,
            "raw_decision": None,
            "risk_validation": None,
            "execution_result": None,
            "logs": []
        }

        final_state = self.app.invoke(initial_state)
        return final_state
