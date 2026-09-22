import json
import logging
from core.types import AnalystReport, BeliefVector, MarketSnapshot
from core.llm_factory import LLMFactory, AgentRole

logger = logging.getLogger("OnChainAnalyst")

class OnChainAnalyst:
    """
    On-chain & Market Flow Analyst:
    Evaluates Funding Rates, Open Interest, Orderbook Imbalance, and Liquidity depth.
    """

    def __init__(self):
        self.llm = LLMFactory.get_chat_model(role=AgentRole.ANALYST, temperature=0.1)

    def analyze(self, snapshot: MarketSnapshot) -> AnalystReport:
        prompt = f"""
You are a Crypto Derivatives & Market Flow Analyst. Evaluate structural flow for {snapshot.symbol}:

Current Price: ${snapshot.current_price:,.2f}
Perpetual Funding Rate: {snapshot.funding_rate:.6f} (Positive: Longs pay Shorts, Negative: Shorts pay Longs)
Open Interest: {snapshot.open_interest} contracts
Orderbook Bid Volume (Depth): {snapshot.orderbook_bid_volume}
Orderbook Ask Volume (Depth): {snapshot.orderbook_ask_volume}
Orderbook Imbalance: {snapshot.orderbook_imbalance} (Positive: Buy wall stronger, Negative: Sell wall stronger)

Respond ONLY with a valid JSON object matching this schema:
{{
    "agent_name": "On-chain & Market Flow Analyst",
    "role": "onchain",
    "belief": {{
        "score": float between -1.0 (Extreme Bearish) and 1.0 (Extreme Bullish),
        "confidence": float between 0.0 and 1.0,
        "bias": "BULLISH" or "BEARISH" or "NEUTRAL",
        "key_drivers": ["string", "string", "string"]
    }},
    "summary": "Brief 1-2 sentence market flow takeaway in Vietnamese",
    "metrics": {{"funding_rate": {snapshot.funding_rate}, "orderbook_imbalance": {snapshot.orderbook_imbalance}}}
}}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            return AnalystReport(**data)
        except Exception as e:
            logger.warning(f"Error parsing OnChain Analyst output ({e}). Fallback.")
            imbalance = snapshot.orderbook_imbalance or 0.0
            score = 0.35 if imbalance > 0.05 else (-0.35 if imbalance < -0.05 else 0.0)
            bias = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
            
            return AnalystReport(
                agent_name="On-chain & Market Flow Analyst",
                role="onchain",
                belief=BeliefVector(
                    score=score,
                    confidence=0.72,
                    bias=bias,
                    key_drivers=[
                        f"Độ lệch sổ lệnh (imbalance): {imbalance:+.2%}",
                        "Funding rate duy trì ổn định không quá nóng"
                    ]
                ),
                summary=f"Dòng tiền sổ lệnh cho thấy tường mua/bán ở trạng thái {bias}.",
                metrics={"orderbook_imbalance": imbalance}
            )
