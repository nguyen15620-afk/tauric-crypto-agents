import json
import logging
from typing import List, Optional
from core.types import DebateTurn, AnalystReport, MarketSnapshot
from core.llm_factory import LLMFactory, AgentRole

logger = logging.getLogger("DebateArena")

class BullResearcher:
    """
    Advocates for the Bullish hypothesis during the debate.
    Scrutinizes technical supports, bullish divergence, inflows, and upward catalysts.
    """
    def __init__(self):
        self.llm = LLMFactory.get_chat_model(role=AgentRole.DEBATER, temperature=0.3)

    def speak(
        self,
        snapshot: MarketSnapshot,
        reports: List[AnalystReport],
        previous_bear_argument: Optional[str] = None,
        round_num: int = 1
    ) -> DebateTurn:
        reports_context = "\n".join([f"- {r.agent_name}: {r.summary} (Score: {r.belief.score})" for r in reports])
        counter_context = f"\nĐối thủ Bear lập luận ở vòng trước:\n\"{previous_bear_argument}\"" if previous_bear_argument else ""

        prompt = f"""
Bạn là Bull Researcher (Trưởng nhóm nghiên cứu phe Bò / Mua) trong quỹ phòng hộ Crypto.
Cặp giao dịch: {snapshot.symbol} | Giá hiện tại: ${snapshot.current_price:,.2f}
Vòng tranh biện: #{round_num}

Báo cáo tóm tắt từ các Analyst:
{reports_context}
{counter_context}

Nhiệm vụ của bạn:
1. Đưa ra lập luận mạnh mẽ nhất bảo vệ quan điểm Mua (Long/Accumulation).
2. Phản bác lại các luận điểm rủi ro của phe Gấu (nếu có).
3. Đánh giá stance_score (từ 0.2 đến 1.0).

Trả về DUY NHẤT một JSON hợp lệ:
{{
    "speaker": "Bull Researcher",
    "argument": "Luận điểm trọng tâm của bạn bằng tiếng Việt (2-3 câu ngắn gọn, sắc sảo)",
    "counter_points": "Phản bác luận điểm đối thủ (1-2 câu)",
    "stance_score": float từ 0.2 đến 1.0
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
            return DebateTurn(round_num=round_num, **data)
        except Exception as e:
            logger.warning(f"Bull researcher parsing error: {e}. Fallback used.")
            return DebateTurn(
                round_num=round_num,
                speaker="Bull Researcher",
                argument=f"Giá {snapshot.symbol} giữ vững cấu trúc hỗ trợ tại ${snapshot.current_price:,.2f}. Phe mua đang tích lũy tốt và hấp thụ hết lượng hàng chốt lời.",
                counter_points="Áp lực bán ngắn hạn không kèm khối lượng lớn, không có tín hiệu phân phối mạnh.",
                stance_score=0.70
            )

class BearResearcher:
    """
    Advocates for the Bearish hypothesis during the debate.
    Highlights hidden distribution, liquidity sweep risks, resistance walls, and downside catalysts.
    """
    def __init__(self):
        self.llm = LLMFactory.get_chat_model(role=AgentRole.DEBATER, temperature=0.3)

    def speak(
        self,
        snapshot: MarketSnapshot,
        reports: List[AnalystReport],
        previous_bull_argument: Optional[str] = None,
        round_num: int = 1
    ) -> DebateTurn:
        reports_context = "\n".join([f"- {r.agent_name}: {r.summary} (Score: {r.belief.score})" for r in reports])
        counter_context = f"\nĐối thủ Bull lập luận ở vòng trước:\n\"{previous_bull_argument}\"" if previous_bull_argument else ""

        prompt = f"""
Bạn là Bear Researcher (Trưởng nhóm nghiên cứu phe Gấu / Bán) trong quỹ phòng hộ Crypto.
Cặp giao dịch: {snapshot.symbol} | Giá hiện tại: ${snapshot.current_price:,.2f}
Vòng tranh biện: #{round_num}

Báo cáo tóm tắt từ các Analyst:
{reports_context}
{counter_context}

Nhiệm vụ của bạn:
1. Đưa ra lập luận chỉ rõ các cạm bẫy thanh khoản (bull trap), rủi ro xả hàng và kháng cự cứng.
2. Phản bác lại luận điểm lạc quan quá mức của phe Bò.
3. Đánh giá stance_score (từ -0.2 đến -1.0).

Trả về DUY NHẤT một JSON hợp lệ:
{{
    "speaker": "Bear Researcher",
    "argument": "Luận điểm cảnh báo rủi ro trọng tâm của bạn bằng tiếng Việt (2-3 câu sắc sảo)",
    "counter_points": "Phản bác luận điểm phe Bò (1-2 câu)",
    "stance_score": float từ -0.2 đến -1.0
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
            return DebateTurn(round_num=round_num, **data)
        except Exception as e:
            logger.warning(f"Bear researcher parsing error: {e}. Fallback used.")
            return DebateTurn(
                round_num=round_num,
                speaker="Bear Researcher",
                argument=f"Vùng kháng cự phía trên của {snapshot.symbol} rất dày, nguy cơ bẫy tăng giá (Bull Trap) quét thanh lý trước khi giảm tiếp là rất cao.",
                counter_points="Phe bò đánh giá thấp sự suy yếu của động lượng thị trường.",
                stance_score=-0.65
            )
