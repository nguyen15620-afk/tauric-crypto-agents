from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field

def utc_now() -> datetime:
    return datetime.now(timezone.utc)

class ActionEnum(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class BeliefVector(BaseModel):
    score: float = Field(..., ge=-1.0, le=1.0, description="Directional belief: -1.0 (Strong Bear) to +1.0 (Strong Bull)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in this assessment (0.0 to 1.0)")
    bias: str = Field(default="NEUTRAL", description="BULLISH, BEARISH, or NEUTRAL")
    key_drivers: List[str] = Field(default_factory=list, description="Top factors influencing this belief")

class AnalystReport(BaseModel):
    agent_name: str
    role: str
    belief: BeliefVector
    summary: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)

class DebateTurn(BaseModel):
    round_num: int
    speaker: str  # e.g., 'Bull Researcher' or 'Bear Researcher'
    argument: str
    counter_points: Optional[str] = None
    stance_score: float  # -1.0 to 1.0

class TradeDecision(BaseModel):
    action: ActionEnum
    conviction: int = Field(..., ge=1, le=10, description="Conviction score 1 to 10")
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    suggested_position_size_pct: float = Field(default=5.0, description="Percentage of available capital to allocate")
    rationale: str
    bull_case_summary: Optional[str] = None
    bear_case_summary: Optional[str] = None
    risk_assessment: Optional[str] = None
    timestamp: datetime = Field(default_factory=utc_now)

class RiskValidation(BaseModel):
    approved: bool
    final_action: ActionEnum
    original_action: ActionEnum
    approved_position_size_pct: float
    approved_position_usd: float
    stop_loss: float
    take_profit: float
    rejection_reasons: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=utc_now)

class MarketSnapshot(BaseModel):
    symbol: str
    timestamp: datetime = Field(default_factory=utc_now)
    current_price: float
    change_24h_pct: float
    high_24h: float
    low_24h: float
    volume_24h: float
    # Technical Indicators
    rsi_14: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_hist: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr_14: Optional[float] = None
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_200: Optional[float] = None
    # Sentiment & Macro
    fear_and_greed_score: Optional[int] = None
    fear_and_greed_label: Optional[str] = None
    news_headlines: List[str] = Field(default_factory=list)
    # Market Flow & Derivatives proxy
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    orderbook_bid_volume: Optional[float] = None
    orderbook_ask_volume: Optional[float] = None
    orderbook_imbalance: Optional[float] = None  # (bid - ask) / (bid + ask)

class Position(BaseModel):
    symbol: str
    side: ActionEnum  # BUY (Long) or SELL (Short)
    entry_price: float
    amount: float
    current_price: float
    stop_loss: float
    take_profit: float
    entry_time: datetime = Field(default_factory=utc_now)
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
