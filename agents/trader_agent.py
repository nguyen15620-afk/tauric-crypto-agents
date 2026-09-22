import json
import logging
from typing import List, Optional
from core.types import TradeDecision, ActionEnum, AnalystReport, DebateTurn, MarketSnapshot
from core.llm_factory import LLMFactory, AgentRole

logger = logging.getLogger("ChiefTrader")

class ChiefTrader:
    """
    Chief Investment Officer / Decision Agent:
    Synthesizes analyst reports and researcher debate into a unified trading decision:
    - Action: BUY / SELL / HOLD
    - Conviction Score: 1 to 10
    - Stop Loss, Take Profit targets
    - Comprehensive Rationale
    """

    def __init__(self):
        self.llm = LLMFactory.get_chat_model(role=AgentRole.TRADER, temperature=0.1)

    def synthesize(
        self,
        snapshot: MarketSnapshot,
        reports: List[AnalystReport],
        debate_turns: List[DebateTurn]
    ) -> TradeDecision:
        current_price = snapshot.current_price
        analysts_text = "\n".join([
            f"- {r.agent_name} [{r.belief.bias} (Score: {r.belief.score})]: {r.summary}"
            for r in reports
        ])

        debate_text = "Không có tranh biện (Đồng thuận cao)."
        if debate_turns:
            debate_text = "\n".join([
                f"[Vòng {d.round_num}] {d.speaker}: {d.argument}"
                for d in debate_turns
            ])

        prompt = f"""
Bạn là Giám Đốc Đầu Tư (Chief Investment Officer / Chief Trader) của quỹ định lượng Crypto.
Cặp giao dịch: {snapshot.symbol}
Giá hiện tại: ${current_price:,.2f}
Biến động 24h: {snapshot.change_24h_pct:+.2f}%

BÁO CÁO CÁC CHUYÊN GIA PHÂN TÍCH:
{analysts_text}

DIỄN BIẾN TRANH BIỆN BULL vs BEAR:
{debate_text}

NHIỆM VỤ CỦA BẠN:
Đưa ra quyết định giao dịch cuối cùng có trách nhiệm, tính toán điểm chốt lời (TP) và cắt lỗ (SL) hợp lý:
1. Quyết định: "BUY" (mua), "SELL" (bán/chốt lời), hoặc "HOLD" (đứng ngoài quan sát).
2. Mức độ tự tin (conviction): Điểm từ 1 đến 10.
3. Stop Loss (bắt buộc nếu BUY/SELL): Giá cụ thể.
4. Take Profit: Giá cụ thể.
5. Giải thích lý do (rationale) rõ ràng bằng tiếng Việt.

Trả về DUY NHẤT một JSON hợp lệ:
{{
    "action": "BUY" hoặc "SELL" hoặc "HOLD",
    "conviction": integer từ 1 đến 10,
    "current_price": {current_price},
    "target_price": float hoặc null,
    "stop_loss": float hoặc null,
    "take_profit": float hoặc null,
    "suggested_position_size_pct": float từ 2.0 đến 12.0,
    "rationale": "Giải thích tổng hợp luận điểm ra quyết định bằng tiếng Việt",
    "bull_case_summary": "Tóm tắt điểm mạnh phe Bò",
    "bear_case_summary": "Tóm tắt rủi ro phe Gấu",
    "risk_assessment": "Đánh giá quản trị rủi ro cho vị thế này"
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
            return TradeDecision(**data)
        except Exception as e:
            logger.warning(f"Chief Trader parsing error: {e}. Generating heuristic synthesis.")
            
            # Simple weighted average of analyst scores
            avg_score = sum([r.belief.score for r in reports]) / max(len(reports), 1)
            
            if avg_score > 0.25:
                action = ActionEnum.BUY
                conviction = min(max(int(5 + avg_score * 5), 5), 9)
                sl = current_price * 0.98
                tp = current_price * 1.04
                rationale = f"Đa số chỉ số và chuyên gia nghiêng về kịch bản TĂNG (điểm trung bình {avg_score:+.2f}). Đề xuất mở vị thế Mua thăm dò."
            elif avg_score < -0.25:
                action = ActionEnum.SELL
                conviction = min(max(int(5 + abs(avg_score) * 5), 5), 9)
                sl = current_price * 1.02
                tp = current_price * 0.96
                rationale = f"Đa số tín hiệu kỹ thuật và tâm lý nghiêng về kịch bản GIẢM (điểm {avg_score:+.2f}). Đề xuất Bán/Chốt lời phòng vệ."
            else:
                action = ActionEnum.HOLD
                conviction = 4
                sl = None
                tp = None
                rationale = f"Tín hiệu thị trường đang giằng co trung lập (điểm {avg_score:+.2f}). Đề xuất đứng ngoài quan sát (HOLD)."

            return TradeDecision(
                action=action,
                conviction=conviction,
                current_price=current_price,
                target_price=tp,
                stop_loss=sl,
                take_profit=tp,
                suggested_position_size_pct=6.0 if action != ActionEnum.HOLD else 0.0,
                rationale=rationale,
                bull_case_summary="Hỗ trợ kỹ thuật vững.",
                bear_case_summary="Áp lực điều chỉnh ngắn hạn.",
                risk_assessment="Duy trì tỷ lệ Risk/Reward tối thiểu 1.5:1."
            )
