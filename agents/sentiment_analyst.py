import json
import logging
from core.types import AnalystReport, BeliefVector, MarketSnapshot
from core.llm_factory import LLMFactory, AgentRole

logger = logging.getLogger("SentimentAnalyst")

class SentimentAnalyst:
    """
    Sentiment & News Analyst:
    Evaluates Fear & Greed Index, headline news, social mood.
    """

    def __init__(self):
        self.llm = LLMFactory.get_chat_model(role=AgentRole.ANALYST, temperature=0.1)

    def analyze(self, snapshot: MarketSnapshot) -> AnalystReport:
        news_str = "\n".join([f"- {h}" for h in snapshot.news_headlines])
        prompt = f"""
You are an expert Crypto Sentiment & News Analyst. Evaluate market sentiment for {snapshot.symbol}:

Fear & Greed Index: {snapshot.fear_and_greed_score}/100 ({snapshot.fear_and_greed_label})
24h Price Change: {snapshot.change_24h_pct:+.2f}%

Recent Market News & Narrative:
{news_str}

Respond ONLY with a valid JSON object matching this schema:
{{
    "agent_name": "Sentiment & News Analyst",
    "role": "sentiment",
    "belief": {{
        "score": float between -1.0 (Extreme Bearish) and 1.0 (Extreme Bullish),
        "confidence": float between 0.0 and 1.0,
        "bias": "BULLISH" or "BEARISH" or "NEUTRAL",
        "key_drivers": ["string", "string", "string"]
    }},
    "summary": "Brief 1-2 sentence sentiment takeaway in Vietnamese",
    "metrics": {{"fear_and_greed": {snapshot.fear_and_greed_score}}}
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
            logger.warning(f"Error parsing Sentiment Analyst output ({e}). Fallback.")
            fgi = snapshot.fear_and_greed_score or 50
            score = 0.4 if fgi > 55 else (-0.35 if fgi < 35 else 0.0)
            bias = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
            
            return AnalystReport(
                agent_name="Sentiment & News Analyst",
                role="sentiment",
                belief=BeliefVector(
                    score=score,
                    confidence=0.70,
                    bias=bias,
                    key_drivers=[f"Chỉ số Fear & Greed ở mức {fgi}", "Tâm lý chung thị trường ổn định"]
                ),
                summary=f"Tâm lý thị trường ghi nhận trạng thái {bias} nhẹ dựa trên Fear & Greed Index.",
                metrics={"fear_and_greed": fgi}
            )
