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
2. Mức độ tự tin (conviction): Điểm số từ 1 đến 10 dựa theo thang đo bắt buộc dưới đây.

QUY TẮC ĐÁNH GIÁ ĐỘ TỰ TIN (CONVICTION SCORE TỪ 1 ĐẾN 10) - TUYỆT ĐỐI KHÔNG MẶC ĐỊNH 7/10:
• 9-10 (Rất Mạnh / High Edge): Cả 3 chuyên gia đồng thuận tuyệt đối cùng chiều (điểm > +0.5 hoặc < -0.5), không có phân kỳ, volume và nến xác nhận rõ rệt.
• 7-8 (Mạnh / Clear Setup): Đa số chuyên gia ủng hộ (+0.3 đến +0.5), không có bất đồng gay gắt, tỷ lệ R:R thuận lợi.
• 5-6 (Trung Bình / Moderate): Tín hiệu phân hóa nhẹ, một chuyên gia trung lập hoặc sau phản biện còn rủi ro, chỉ nên thăm dò nhỏ.
• 3-4 (Thấp / High Conflict): Bất đồng quan điểm lớn (Debate gay gắt), tín hiệu đối nghịch nhau, thị trường giằng co -> NÊN CHỌN "HOLD".
• 1-2 (Rất Thấp / Extreme Risk): Dữ liệu mâu thuẫn nặng nề hoặc biến động bất thường.

3. Stop Loss (bắt buộc nếu BUY/SELL): Giá cụ thể.
4. Take Profit: Giá cụ thể.
5. Giải thích lý do (rationale) rõ ràng bằng tiếng Việt.

Trả về DUY NHẤT một JSON hợp lệ:
{{
    "action": "BUY" hoặc "SELL" hoặc "HOLD",
    "conviction": integer từ 1 đến 10 (đánh giá chuẩn xác theo quy tắc trên),
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
        scores = [r.belief.score for r in reports] if reports else [0.0]
        avg_score = sum(scores) / max(len(scores), 1)
        pos_count = sum(1 for s in scores if s > 0.15)
        neg_count = sum(1 for s in scores if s < -0.15)
        spread = max(scores) - min(scores) if scores else 0.0

        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            if content.startswith("```json"):
                content = content[7:-3].strip()
            elif content.startswith("```"):
                content = content[3:-3].strip()
            data = json.loads(content)
            
            # --- Quantitative Conviction Calibration ---
            # Prevent LLM from getting stuck at arbitrary values like 7:
            # Calibrate conviction dynamically with the real mathematical consensus of analysts
            raw_conv = int(data.get("conviction", 7))
            chosen_action = data.get("action", "HOLD")

            if chosen_action == "BUY":
                if pos_count == 3 and avg_score >= 0.50:
                    base_conv = 9
                elif pos_count >= 2 and neg_count == 0:
                    base_conv = 8 if avg_score >= 0.38 else 7
                elif neg_count > 0:  # Conflict/debate happened
                    base_conv = 5 if avg_score > 0.25 else 6
                else:
                    base_conv = 6
            elif chosen_action == "SELL":
                if neg_count == 3 and abs(avg_score) >= 0.50:
                    base_conv = 9
                elif neg_count >= 2 and pos_count == 0:
                    base_conv = 8 if abs(avg_score) >= 0.38 else 7
                elif pos_count > 0:
                    base_conv = 5
                else:
                    base_conv = 6
            else:  # HOLD
                base_conv = 4 if spread > 0.35 else 5

            # Calibrate: 50% LLM judgment + 50% quant consensus
            calibrated_conv = int(round(raw_conv * 0.4 + base_conv * 0.6))
            data["conviction"] = max(1, min(10, calibrated_conv))

            return TradeDecision(**data)
        except Exception as e:
            logger.warning(f"Chief Trader parsing error: {e}. Generating heuristic synthesis.")
            
            if avg_score > 0.20:
                action = ActionEnum.BUY
                if pos_count == 3 and avg_score >= 0.50:
                    conviction = 9
                elif pos_count >= 2 and neg_count == 0:
                    conviction = 8 if avg_score >= 0.38 else 7
                elif neg_count > 0:
                    conviction = 5
                else:
                    conviction = 6
                sl = current_price * 0.98
                tp = current_price * 1.04
                rationale = f"Đa số chỉ số và chuyên gia nghiêng về kịch bản TĂNG (điểm trung bình {avg_score:+.2f}). Đề xuất mở vị thế Mua thăm dò."
            elif avg_score < -0.20:
                action = ActionEnum.SELL
                if neg_count == 3 and abs(avg_score) >= 0.50:
                    conviction = 9
                elif neg_count >= 2 and pos_count == 0:
                    conviction = 8 if abs(avg_score) >= 0.38 else 7
                elif pos_count > 0:
                    conviction = 5
                else:
                    conviction = 6
                sl = current_price * 1.02
                tp = current_price * 0.96
                rationale = f"Đa số tín hiệu kỹ thuật và tâm lý nghiêng về kịch bản GIẢM (điểm {avg_score:+.2f}). Đề xuất Bán/Chốt lời phòng vệ."
            else:
                action = ActionEnum.HOLD
                conviction = 4 if spread > 0.35 else 5
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
