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
_REASONING_LIMITER = _RateLimiter(calls_per_minute=4)  # Flash: 5 RPM, leave buffer


class _RateLimitedLLM:
    """
    Wrapper around a LangChain LLM that enforces rate limiting and
    exponential backoff on 429/resource-exhausted errors.
    """

    MAX_RETRIES = 3
    BACKOFF_BASE = 3.0  # seconds

    def __init__(self, llm, limiter: _RateLimiter, model_name: str):
        self._llm = llm
        self._limiter = limiter
        self._model_name = model_name

    def invoke(self, messages: Any) -> Any:
        for attempt in range(self.MAX_RETRIES):
            self._limiter.wait_if_needed()
            try:
                return self._llm.invoke(messages)
            except Exception as e:
                err_str = str(e).lower()
                is_rate_limit = any(kw in err_str for kw in ["429", "resource_exhausted", "quota", "rate limit"])
                if is_rate_limit and attempt < self.MAX_RETRIES - 1:
                    wait_time = self.BACKOFF_BASE * (2 ** attempt)  # 3s, 6s, 12s
                    logger.warning(
                        f"[{self._model_name}] Rate limit hit (attempt {attempt + 1}/{self.MAX_RETRIES}). "
                        f"Retrying in {wait_time:.0f}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise  # re-raise non-rate-limit errors or final attempt
        raise RuntimeError(f"[{self._model_name}] All {self.MAX_RETRIES} retries exhausted.")


class LLMFactory:
    """
    Unified LLM Factory supporting:
    - Tiered Google Gemini model routing tailored for rate limits and quotas:
        * Analysts: Gemini Flash Lite (high RPM, lower cost) — suitable for parallel calls
        * Debaters & Chief Trader: Gemini Flash (slower, deeper reasoning)
    - Built-in rate limiter to prevent 429 errors on free tier
    - Exponential backoff retry on quota exhaustion
    - Fallback MockLLM for zero-cost dry-run / integration testing
    """

    @staticmethod
    def get_model_name_for_role(role: AgentRole) -> str:
        if role == AgentRole.ANALYST:
            return getattr(settings, "MODEL_ANALYST", "gemini-2.5-flash-lite")
        else:
            return getattr(settings, "MODEL_REASONING", "gemini-2.5-flash")

    @classmethod
    def get_chat_model(cls, role: AgentRole = AgentRole.ANALYST, temperature: float = 0.2):
        api_key = settings.effective_gemini_key
        model_name = cls.get_model_name_for_role(role)

        if api_key:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
                logger.info(f"Initializing Gemini model={model_name} for role={role.value}")
                raw_llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=temperature,
                    max_retries=1,  # We handle retries ourselves in _RateLimitedLLM
                )

                # Choose limiter based on role tier
                if role == AgentRole.ANALYST:
                    limiter = _ANALYST_LIMITER
                else:
                    limiter = _REASONING_LIMITER

                return _RateLimitedLLM(raw_llm, limiter, model_name)

            except Exception as e:
                logger.warning(f"Failed to initialize ChatGoogleGenerativeAI ({e}), falling back to MockLLM")
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
            return json.dumps({
                "action": "BUY",
                "conviction": 7,
                "current_price": 68500.0,
                "target_price": 71200.0,
                "stop_loss": 66800.0,
                "take_profit": 71500.0,
                "suggested_position_size_pct": 8.0,
                "rationale": "Tổng hợp phân tích kỹ thuật và dòng tiền ủng hộ kịch bản tăng. Luận điểm Bull có cơ sở xác thực với hỗ trợ dòng tiền và cấu trúc giá.",
                "bull_case_summary": "Hỗ trợ EMA và orderbook vững chắc.",
                "bear_case_summary": "Rủi ro kháng cự ngắn hạn và quét râu thanh khoản.",
                "risk_assessment": "Stop loss dưới vùng hỗ trợ cục bộ với tỷ lệ R:R tối thiểu 1.8"
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

            score = 0.65 if rsi < 40 else (-0.65 if rsi > 65 else 0.15)
            bias = "BULLISH" if score > 0.2 else ("BEARISH" if score < -0.2 else "NEUTRAL")

            return json.dumps({
                "agent_name": "Technical Analyst",
                "role": "technical",
                "belief": {
                    "score": score,
                    "confidence": 0.82,
                    "bias": bias,
                    "key_drivers": [
                        f"RSI 14 ở mức {rsi:.1f}",
                        "Đường EMA hỗ trợ cấu trúc xu hướng",
                        "Dải Bollinger Bands giữ nhịp nén giá"
                    ]
                },
                "summary": f"Xu hướng kỹ thuật nghiêng về {bias} theo tín hiệu RSI và cụm chỉ báo.",
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

            score = 0.5 if fgi > 55 else (-0.4 if fgi < 35 else 0.1)
            bias = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")

            return json.dumps({
                "agent_name": "Sentiment & News Analyst",
                "role": "sentiment",
                "belief": {
                    "score": score,
                    "confidence": 0.75,
                    "bias": bias,
                    "key_drivers": [
                        f"Chỉ số Fear & Greed ở mức {fgi} ({'Tham lam' if fgi > 50 else 'Sợ hãi'})",
                        "Dòng tiền quỹ ETF duy trì trạng thái ròng dương",
                        "Tin tức kinh tế vĩ mô ổn định không có cú sốc"
                    ]
                },
                "summary": f"Tâm lý người tham gia thị trường ở vị thế {bias} vừa phải.",
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

            # Positive funding (longs pay) = slightly bearish signal
            score = -0.25 if funding_rate > 0.001 else (0.25 if funding_rate < -0.0002 else 0.25)
            bias = "BULLISH" if score > 0.1 else ("BEARISH" if score < -0.1 else "NEUTRAL")

            return json.dumps({
                "agent_name": "On-chain & Market Flow Analyst",
                "role": "onchain",
                "belief": {
                    "score": score,
                    "confidence": 0.78,
                    "bias": bias,
                    "key_drivers": [
                        f"Funding rate: {funding_rate:.4%} ({'longs pay shorts' if funding_rate > 0 else 'shorts pay longs'})",
                        "Độ sâu sổ lệnh có tường mua hỗ trợ tốt",
                        "Không phát hiện dòng tiền nạp xả đột biến"
                    ]
                },
                "summary": f"Thanh khoản sổ lệnh và funding rate phản ánh trạng thái {bias} của cấu trúc thị trường.",
                "metrics": {"funding_rate": funding_rate}
            })

        return json.dumps({"status": "ok", "message": "Standard analysis generated."})
