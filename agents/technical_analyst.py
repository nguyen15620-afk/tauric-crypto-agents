import json
import logging
from core.types import AnalystReport, BeliefVector, MarketSnapshot
from core.llm_factory import LLMFactory, AgentRole

logger = logging.getLogger("TechnicalAnalyst")

class TechnicalAnalyst:
    """
    Technical Analysis Agent:
    Evaluates RSI, MACD, Bollinger Bands, ATR, and EMA trends.
    Uses high-throughput LLM (Gemini Flash Lite) to produce directional belief.
    """

    def __init__(self):
        self.llm = LLMFactory.get_chat_model(role=AgentRole.ANALYST, temperature=0.1)

    def analyze(self, snapshot: MarketSnapshot) -> AnalystReport:
        prompt = f"""
You are a Veteran Crypto Technical Analyst. Analyze the technical indicators for {snapshot.symbol}:

Current Price: ${snapshot.current_price:,.2f}
24h Change: {snapshot.change_24h_pct:+.2f}%
24h Range: High ${snapshot.high_24h:,.2f} | Low ${snapshot.low_24h:,.2f}

Indicators:
- RSI (14): {snapshot.rsi_14} (Oversold < 30, Overbought > 70)
- MACD: {snapshot.macd} | Signal: {snapshot.macd_signal} | Histogram: {snapshot.macd_hist}
- Bollinger Bands: Upper ${snapshot.bb_upper} | Mid ${snapshot.bb_middle} | Lower ${snapshot.bb_lower}
- ATR (14): ${snapshot.atr_14}
- EMA 20: ${snapshot.ema_20}
- EMA 50: ${snapshot.ema_50}
- EMA 200: ${snapshot.ema_200}

Respond ONLY with a valid JSON object matching this schema:
{{
    "agent_name": "Technical Analyst",
    "role": "technical",
    "belief": {{
        "score": float between -1.0 (Extreme Bearish) and 1.0 (Extreme Bullish),
        "confidence": float between 0.0 and 1.0,
        "bias": "BULLISH" or "BEARISH" or "NEUTRAL",
        "key_drivers": ["string", "string", "string"]
    }},
    "summary": "Brief 1-2 sentence technical takeaway in Vietnamese",
    "metrics": {{"rsi": {snapshot.rsi_14}, "macd_trend": "bullish or bearish or neutral"}}
}}
"""
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            # Clean markdown formatting if present
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
                
            data = json.loads(content)
            return AnalystReport(**data)
        except Exception as e:
            logger.warning(f"Error parsing Technical Analyst output ({e}). Using default rule-based parsing.")
            
            # Deterministic fallback
            rsi = snapshot.rsi_14 or 50.0
            score = 0.6 if rsi < 40 else (-0.6 if rsi > 65 else 0.05)
            bias = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
            
            return AnalystReport(
                agent_name="Technical Analyst",
                role="technical",
                belief=BeliefVector(
                    score=score,
                    confidence=0.75,
                    bias=bias,
                    key_drivers=[f"RSI ở mức {rsi:.1f}", f"Giá so với EMA 200: {'Trên' if snapshot.current_price > (snapshot.ema_200 or 0) else 'Dưới'}"]
                ),
                summary=f"Xu hướng kỹ thuật đang nghiêng về {bias} theo tín hiệu RSI và dải EMA.",
                metrics={"rsi": rsi}
            )
