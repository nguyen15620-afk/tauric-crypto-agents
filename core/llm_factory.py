import os
import json
import time
import logging
import threading
from typing import Optional, Any, Dict
from enum import Enum
from config.settings import settings

logger = logging.getLogger("LLMFactory")


class AgentRole(str, Enum):
    ANALYST = "analyst"        # High RPM / RPD (e.g., Flash Lite) — parallel use
    DEBATER = "debater"        # Balanced speed & reasoning
    TRADER = "trader"          # Deep reasoning & synthesis (e.g., Flash)
    RISK = "risk"              # Critical verification


class _RateLimiter:
    """
    Thread-safe token bucket rate limiter for LLM API calls.
    Prevents 429 errors by spacing out requests to fit within quota limits.

    Gemini Free Tier limits (conservative):
      - Flash Lite: 15 RPM → min 4s between calls per model
      - Flash:       5 RPM → min 12s between calls per model
    """

    def __init__(self, calls_per_minute: int = 10):
        self._min_interval = 60.0 / max(calls_per_minute, 1)
        self._last_call_time = 0.0
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Block the calling thread if we're calling too fast."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call_time
            wait = self._min_interval - elapsed
            if wait > 0:
                logger.debug(f"[RateLimiter] Throttling LLM call by {wait:.2f}s")
                time.sleep(wait)
            self._last_call_time = time.monotonic()


# Shared rate limiters per model tier (singleton per tier)
_ANALYST_LIMITER = _RateLimiter(calls_per_minute=12)   # Flash Lite: 15 RPM, leave buffer
_DEBATER_LIMITER = _RateLimiter(calls_per_minute=12)   # Flash Lite: 15 RPM, leave buffer
_REASONING_LIMITER = _RateLimiter(calls_per_minute=4)  # Flash: 5 RPM, leave buffer

# Shared model cascade chains tailored to user's exact API quotas
ANALYST_MODEL_CASCADE = [
    "gemini-3.5-flash-lite",  # 15 RPM, 500 RPD (Primary workhorse)
    "gemini-3.1-flash-lite",  # 15 RPM, 500 RPD (Backup workhorse)
    "gemini-2.5-flash-lite",  # 10 RPM, 20 RPD
    "gemma-4-31b",            # 30 RPM, 14.4K RPD (High throughput fallback)
    "gemini-2.5-flash",       # Standard Flash
]

DEBATER_MODEL_CASCADE = [
    "gemini-3.5-flash-lite",  # 15 RPM, 500 RPD
    "gemini-3.1-flash-lite",  # 15 RPM, 500 RPD
    "gemini-2.5-flash-lite",  # 10 RPM, 20 RPD
    "gemini-2.5-flash",       # Standard Flash
]

TRADER_MODEL_CASCADE = [
    "gemini-3.8-flash",       # 5 RPM, 20 RPD (Top strategic reasoning)
    "gemini-3.7-flash",       # 5 RPM, 20 RPD (Fallback 1)
    "gemini-3.6-flash",       # 5 RPM, 20 RPD (Fallback 2)
    "gemini-3.5-flash",       # 5 RPM, 20 RPD (Fallback 3)
    "gemini-3-flash",         # 5 RPM, 20 RPD (Fallback 4)
    "gemini-2.5-flash",       # 5 RPM, 20 RPD (Fallback 5)
    "gemini-3.5-flash-lite",  # 15 RPM, 500 RPD (High-capacity fallback)
]


class _CascadingRateLimitedLLM:
    """
    Intelligent Cascading Wrapper around Google Gemini models with rate limiting,
    automatic fallback across model tiers when 429 / RESOURCE_EXHAUSTED occurs,
    and safe fallback to MockLLM if all API models are temporarily exhausted.
    """

    def __init__(
        self,
        api_key: str,
        candidates: list,
        limiter: _RateLimiter,
        role: AgentRole,
        temperature: float = 0.2
    ):
        self._api_key = api_key
        # Preserve candidate order without duplicates
        self._candidates = list(dict.fromkeys(candidates))
        self._limiter = limiter
        self.role = role
        self.temperature = temperature
        self._mock = MockLLM(role=role)
        self._exhausted: set = set()

    def invoke(self, messages: Any) -> Any:
        from langchain_google_genai import ChatGoogleGenerativeAI

        for model_name in self._candidates:
            if model_name in self._exhausted:
                continue

            self._limiter.wait_if_needed()
            try:
                raw_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=self._api_key,
                    temperature=self.temperature,
                    max_retries=1,
                )
                return raw_llm.invoke(messages)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in ["429", "resource_exhausted", "quota", "rate limit"])
                is_not_found = any(kw in err_str for kw in ["404", "not found", "not supported", "is not found"])

                if is_rate_limit or is_not_found:
                    logger.warning(
                        f"[{self.role.value}] Model '{model_name}' hit limit or unavailable ({e}). "
                        f"Auto-switching to next model in cascade..."
                    )
                    self._exhausted.add(model_name)
                    continue
                else:
                    logger.error(f"[{self.role.value}] Model '{model_name}' unexpected error: {e}")
                    raise

        # If all candidates exhausted, gracefully fallback to heuristic MockLLM
        logger.warning(
            f"[{self.role.value}] All models in cascade {self._candidates} temporarily exhausted. "
            f"Gracefully falling back to heuristic MockLLM to avoid system failure."
        )
        return self._mock.invoke(messages)


class LLMFactory:
    """
    Unified LLM Factory supporting:
    - Tiered Google Gemini model routing tailored for rate limits and quotas:
        * Analysts: Gemini Flash Lite (high RPM, 500 RPD) — parallel execution
        * Debaters: Gemini Flash Lite (fast, 500 RPD)
        * Chief Trader: Gemini 3.8/3.7/3.6/3.5/3/2.5 Flash (deep strategic reasoning)
    - Automatic cascading fallback when 429 quota is reached
    - Built-in rate limiter to prevent burst 429 errors
    - Fallback MockLLM for zero-cost dry-run or when all limits are reached
    """

    @staticmethod
    def get_model_name_for_role(role: AgentRole) -> str:
        if role == AgentRole.ANALYST:
            return getattr(settings, "MODEL_ANALYST", "gemini-3.5-flash-lite")
        elif role == AgentRole.DEBATER:
            return getattr(settings, "MODEL_DEBATER", "gemini-3.5-flash-lite")
        else:
            return getattr(settings, "MODEL_REASONING", "gemini-3.8-flash")

    @classmethod
    def get_chat_model(cls, role: AgentRole = AgentRole.ANALYST, temperature: float = 0.2):
        api_key = settings.effective_gemini_key

        if api_key:
            try:
                if role == AgentRole.ANALYST:
                    user_pref = getattr(settings, "MODEL_ANALYST", "gemini-3.5-flash-lite")
                    candidates = [user_pref] + ANALYST_MODEL_CASCADE
                    limiter = _ANALYST_LIMITER
                elif role == AgentRole.DEBATER:
                    user_pref = getattr(settings, "MODEL_DEBATER", "gemini-3.5-flash-lite")
                    candidates = [user_pref] + DEBATER_MODEL_CASCADE
                    limiter = _DEBATER_LIMITER
                else:
                    user_pref = getattr(settings, "MODEL_REASONING", "gemini-3.8-flash")
                    candidates = [user_pref] + TRADER_MODEL_CASCADE
                    limiter = _REASONING_LIMITER

                return _CascadingRateLimitedLLM(
                    api_key=api_key,
                    candidates=candidates,
                    limiter=limiter,
                    role=role,
                    temperature=temperature
                )
            except Exception as e:
                logger.warning(f"Failed to initialize CascadingLLM ({e}), falling back to MockLLM")
                return MockLLM(role=role)
        else:
            logger.info("No GEMINI_API_KEY detected. Using MockLLM for dry-run simulation.")
            return MockLLM(role=role)


class MockLLM:
    """
    Intelligent heuristic mock LLM for testing workflows, backtesting,
    and verifying logic without burning API quota or needing an active internet connection.
    Parses key numeric values from prompts for realistic heuristic responses.
    """

    def __init__(self, role: AgentRole):
        self.role = role

    def invoke(self, messages: Any) -> Any:
        prompt_text = str(messages)
        content = self._generate_heuristic_response(prompt_text)

        class MockAIMessage:
            def __init__(self, text):
                self.content = text

        return MockAIMessage(content)

    def _generate_heuristic_response(self, prompt: str) -> str:
        p_lower = prompt.lower()

        # --- Chief Trader / CIO ---
        if any(kw in p_lower for kw in ["giám đốc đầu tư", "chief trader", "chief investment officer"]):
            import re
            m = re.search(r"giá hiện tại:\s*\$?([0-9,]+\.?[0-9]*)", p_lower)
            cp = 68500.0
            if m:
                try:
                    cp = float(m.group(1).replace(",", ""))
                except Exception:
                    pass

            # Extract analyst scores from prompt
            score_matches = re.findall(r"score:\s*([+-]?[0-9.]+)", p_lower)
            analyst_scores = []
            for sm in score_matches:
                try:
                    analyst_scores.append(float(sm))
                except Exception:
                    pass

            if not analyst_scores:
                analyst_scores = [0.25, 0.35, 0.20]

            avg_score = sum(analyst_scores) / len(analyst_scores)
            pos_count = sum(1 for s in analyst_scores if s > 0.15)
            neg_count = sum(1 for s in analyst_scores if s < -0.15)
            spread = max(analyst_scores) - min(analyst_scores)

            if avg_score > 0.20:
                action = "BUY"
                if pos_count == 3 and avg_score >= 0.50:
                    conviction = 9
                elif pos_count >= 2 and neg_count == 0:
                    conviction = 8 if avg_score >= 0.38 else 7
                elif neg_count > 0:
                    conviction = 5
                else:
                    conviction = 6
                sl = round(cp * 0.98, 4 if cp < 10 else 2)
                tp = round(cp * 1.04, 4 if cp < 10 else 2)
                rationale = f"Đa số chuyên gia phân tích đồng thuận ủng hộ kịch bản TĂNG (điểm trung bình {avg_score:+.2f}). Cấu trúc thanh khoản và dòng tiền thuận lợi."
            elif avg_score < -0.20:
                action = "SELL"
                if neg_count == 3 and abs(avg_score) >= 0.50:
                    conviction = 9
                elif neg_count >= 2 and pos_count == 0:
                    conviction = 8 if abs(avg_score) >= 0.38 else 7
                elif pos_count > 0:
                    conviction = 5
                else:
                    conviction = 6
                sl = round(cp * 1.02, 4 if cp < 10 else 2)
                tp = round(cp * 0.96, 4 if cp < 10 else 2)
                rationale = f"Đa số tín hiệu kỹ thuật và dòng tiền cảnh báo áp lực GIẢM (điểm trung bình {avg_score:+.2f}). Đề xuất cơ cấu phòng vệ hoặc chốt lời."
            else:
                action = "HOLD"
                conviction = 4 if spread > 0.35 else 5
                sl = None
                tp = None
                rationale = f"Thị trường đang trong pha giằng co tích lũy (điểm trung bình {avg_score:+.2f}). Khuyến nghị kiên nhẫn đứng ngoài quan sát."

            return json.dumps({
                "action": action,
                "conviction": conviction,
                "current_price": cp,
                "target_price": tp,
                "stop_loss": sl,
                "take_profit": tp,
                "suggested_position_size_pct": 8.0 if action != "HOLD" else 0.0,
                "rationale": rationale,
                "bull_case_summary": "Hỗ trợ EMA và orderbook vững chắc.",
                "bear_case_summary": "Rủi ro kháng cự ngắn hạn và quét râu thanh khoản.",
                "risk_assessment": "Duy trì tỷ lệ Risk/Reward tối thiểu 1.5:1."
            })

        # --- Bull Researcher ---
        elif "bull researcher" in p_lower:
            return json.dumps({
                "speaker": "Bull Researcher",
                "argument": "Thị trường duy trì cấu trúc đáy sau cao hơn đáy trước, thanh khoản hỗ trợ vững chắc tại các ngưỡng EMA chủ đạo. Cơ hội tích lũy mở vị thế Long có tỷ lệ R:R vượt trội.",
                "counter_points": "Áp lực bán từ phe gấu chủ yếu là chốt lời ngắn hạn, không có khối lượng xả đột biến.",
                "stance_score": 0.75
            })

        # --- Bear Researcher ---
        elif "bear researcher" in p_lower:
            return json.dumps({
                "speaker": "Bear Researcher",
                "argument": "Chỉ số RSI đang tiến gần vùng quá mua, đồng thời thanh khoản tại các vùng kháng cự phía trên khá mỏng. Rủi ro quét thanh lý (liquidation wick) trước khi có xu hướng mới.",
                "counter_points": "Lập luận phe bò chưa tính đến biến động vĩ mô và funding rate có xu hướng nóng lên.",
                "stance_score": -0.60
            })

        # --- Technical Analyst ---
        elif "technical analyst" in p_lower:
            rsi = 50.0
            if "rsi (14):" in p_lower:
                try:
                    rsi = float(p_lower.split("rsi (14):")[1].split("\n")[0].split("(")[0].strip())
                except Exception:
                    pass

            chg_24h = 0.0
            if "24h change:" in p_lower:
                try:
                    chg_24h = float(p_lower.split("24h change:")[1].split("%")[0].replace("+", "").strip())
                except Exception:
                    pass

            # Dynamic technical belief score
            if rsi < 35:
                score = 0.75
                bias = "BULLISH"
                conf = 0.85
            elif rsi > 70:
                score = -0.70
                bias = "BEARISH"
                conf = 0.85
            elif chg_24h >= 7.0 and rsi < 68:
                score = 0.60
                bias = "BULLISH"
                conf = 0.80
            elif chg_24h <= -7.0 and rsi > 35:
                score = -0.60
                bias = "BEARISH"
                conf = 0.80
            elif rsi < 45:
                score = 0.35
                bias = "BULLISH"
                conf = 0.75
            elif rsi > 60:
                score = -0.30
                bias = "BEARISH"
                conf = 0.72
            else:
                score = 0.05
                bias = "NEUTRAL"
                conf = 0.65

            return json.dumps({
                "agent_name": "Technical Analyst",
                "role": "technical",
                "belief": {
                    "score": score,
                    "confidence": conf,
                    "bias": bias,
                    "key_drivers": [
                        f"RSI 14 ở mức {rsi:.1f}",
                        f"Biến động 24h: {chg_24h:+.2f}%",
                        "Dải Bollinger Bands giữ nhịp nén giá"
                    ]
                },
                "summary": f"Xu hướng kỹ thuật nghiêng về {bias} (điểm {score:+.2f}) theo RSI ({rsi:.1f}) và biến động 24h.",
                "metrics": {"rsi": rsi, "trend": bias}
            })

        # --- Sentiment Analyst ---
        elif "sentiment" in p_lower:
            fgi = 52
            if "fear & greed index:" in p_lower:
                try:
                    fgi = int(p_lower.split("fear & greed index:")[1].split("/")[0].strip())
                except Exception:
                    pass

            if fgi >= 75:
                score = 0.65
                bias = "BULLISH"
                conf = 0.82
            elif fgi >= 55:
                score = 0.40
                bias = "BULLISH"
                conf = 0.75
            elif fgi <= 25:
                score = -0.60
                bias = "BEARISH"
                conf = 0.80
            elif fgi <= 45:
                score = -0.30
                bias = "BEARISH"
                conf = 0.70
            else:
                score = 0.05
                bias = "NEUTRAL"
                conf = 0.65

            return json.dumps({
                "agent_name": "Sentiment & News Analyst",
                "role": "sentiment",
                "belief": {
                    "score": score,
                    "confidence": conf,
                    "bias": bias,
                    "key_drivers": [
                        f"Chỉ số Fear & Greed ở mức {fgi} ({'Tham lam' if fgi > 50 else 'Sợ hãi'})",
                        "Dòng tiền quỹ ETF duy trì trạng thái ròng",
                        "Tin tức kinh tế vĩ mô ổn định"
                    ]
                },
                "summary": f"Tâm lý người tham gia thị trường ở vị thế {bias} (điểm {score:+.2f}).",
                "metrics": {"fear_and_greed": fgi}
            })

        # --- OnChain / Derivatives Analyst ---
        elif any(kw in p_lower for kw in ["derivatives", "market flow", "on-chain"]):
            funding_rate = 0.0001
            if "funding rate:" in p_lower:
                try:
                    fr_str = p_lower.split("funding rate:")[1].split("\n")[0].strip()
                    funding_rate = float(fr_str)
                except Exception:
                    pass

            imbalance = 0.0
            if "orderbook imbalance:" in p_lower or "sổ lệnh imbalance:" in p_lower:
                try:
                    target_str = p_lower.split("imbalance:")[1].split("%")[0].replace("+", "").strip()
                    imbalance = float(target_str)
                except Exception:
                    pass

            if imbalance > 40.0:
                score = 0.55
                bias = "BULLISH"
                conf = 0.82
            elif imbalance < -30.0:
                score = -0.50
                bias = "BEARISH"
                conf = 0.80
            else:
                score = -0.25 if funding_rate > 0.001 else (0.25 if funding_rate < -0.0002 else 0.20)
                bias = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")
                conf = 0.75

            return json.dumps({
                "agent_name": "On-chain & Market Flow Analyst",
                "role": "onchain",
                "belief": {
                    "score": score,
                    "confidence": conf,
                    "bias": bias,
                    "key_drivers": [
                        f"Funding rate: {funding_rate:.4%}",
                        f"Sổ lệnh Orderbook Imbalance: {imbalance:+.2f}%",
                        "Thanh khoản dòng tiền thực tế"
                    ]
                },
                "summary": f"Thanh khoản sổ lệnh ({imbalance:+.1f}%) và funding rate phản ánh vị thế {bias}.",
                "metrics": {"funding_rate": funding_rate, "imbalance": imbalance}
            })

        return json.dumps({"status": "ok", "message": "Standard analysis generated."})
