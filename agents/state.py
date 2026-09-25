import operator
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from core.types import MarketSnapshot, AnalystReport, DebateTurn, TradeDecision, RiskValidation


class TradingAgentState(TypedDict):
    symbol: str
    timeframe: str
    snapshot: Optional[MarketSnapshot]
    analyst_reports: List[AnalystReport]
    divergence_score: float
    needs_debate: bool
    debate_turns: List[DebateTurn]
    debate_round: int
    raw_decision: Optional[TradeDecision]
    risk_validation: Optional[RiskValidation]
    # Annotated with operator.add so each node APPENDS to logs instead of replacing
    logs: Annotated[List[str], operator.add]

