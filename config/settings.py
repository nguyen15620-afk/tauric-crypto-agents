import os
from pathlib import Path
import yaml
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    # LLM Settings
    GEMINI_API_KEY: Optional[str] = None
    GOOGLE_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    DEFAULT_LLM_PROVIDER: str = "gemini"
    
    # Model Tiering based on user quota & limits
    # Analyst models (high throughput, 500 RPD, 15 RPM): Gemini 3.5 Flash Lite / 3.1 Flash Lite
    MODEL_ANALYST: str = "gemini-3.5-flash-lite"
    
    # Debater models (high throughput, 500 RPD, 15 RPM): Gemini 3.5 Flash Lite
    MODEL_DEBATER: str = "gemini-3.5-flash-lite"

    # Reasoning models (Chief Trader - deep reasoning, 20 RPD, 5 RPM):
    # Gemini 3.8 Flash / Gemini 3.7 Flash / Gemini 3.6 Flash / Gemini 3.5 Flash
    MODEL_REASONING: str = "gemini-3.8-flash"
    
    # Exchange / Market
    DEFAULT_EXCHANGE: str = "binance"
    DEFAULT_SYMBOL: str = "BTC/USDT"
    DEFAULT_TIMEFRAME: str = "15m"
    EXCHANGE_API_KEY: Optional[str] = None
    EXCHANGE_SECRET_KEY: Optional[str] = None
    PAPER_TRADING: bool = True
    
    # Server / App
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    DATABASE_PATH: str = str(BASE_DIR / "crypto_agents.db")

    @property
    def effective_gemini_key(self) -> Optional[str]:
        return self.GEMINI_API_KEY or self.GOOGLE_API_KEY or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def save_api_keys(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        deepseek_api_key: Optional[str] = None,
        default_provider: Optional[str] = None,
        model_analyst: Optional[str] = None,
        model_debater: Optional[str] = None,
        model_reasoning: Optional[str] = None
    ) -> None:
        """
        Saves updated API keys and configurations to .env file and updates runtime memory.
        """
        env_path = BASE_DIR / ".env"
        env_dict = {}
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line_str = line.strip()
                        if line_str and not line_str.startswith("#") and "=" in line_str:
                            k, v = line_str.split("=", 1)
                            env_dict[k.strip()] = v.strip().strip('"').strip("'")
            except Exception:
                pass

        if gemini_api_key is not None:
            val = gemini_api_key.strip()
            self.GEMINI_API_KEY = val if val else None
            self.GOOGLE_API_KEY = val if val else None
            os.environ["GEMINI_API_KEY"] = val
            os.environ["GOOGLE_API_KEY"] = val
            if val:
                env_dict["GEMINI_API_KEY"] = f'"{val}"'
                env_dict["GOOGLE_API_KEY"] = f'"{val}"'
            else:
                env_dict.pop("GEMINI_API_KEY", None)
                env_dict.pop("GOOGLE_API_KEY", None)

        if openai_api_key is not None:
            val = openai_api_key.strip()
            self.OPENAI_API_KEY = val if val else None
            os.environ["OPENAI_API_KEY"] = val
            if val:
                env_dict["OPENAI_API_KEY"] = f'"{val}"'
            else:
                env_dict.pop("OPENAI_API_KEY", None)

        if deepseek_api_key is not None:
            val = deepseek_api_key.strip()
            self.DEEPSEEK_API_KEY = val if val else None
            os.environ["DEEPSEEK_API_KEY"] = val
            if val:
                env_dict["DEEPSEEK_API_KEY"] = f'"{val}"'
            else:
                env_dict.pop("DEEPSEEK_API_KEY", None)

        if default_provider is not None and default_provider.strip():
            self.DEFAULT_LLM_PROVIDER = default_provider.strip()
            env_dict["DEFAULT_LLM_PROVIDER"] = self.DEFAULT_LLM_PROVIDER

        if model_analyst is not None and model_analyst.strip():
            self.MODEL_ANALYST = model_analyst.strip()
            env_dict["MODEL_ANALYST"] = self.MODEL_ANALYST

        if model_debater is not None and model_debater.strip():
            self.MODEL_DEBATER = model_debater.strip()
            env_dict["MODEL_DEBATER"] = self.MODEL_DEBATER

        if model_reasoning is not None and model_reasoning.strip():
            self.MODEL_REASONING = model_reasoning.strip()
            env_dict["MODEL_REASONING"] = self.MODEL_REASONING

        # Write to .env
        with open(env_path, "w", encoding="utf-8") as f:
            f.write("# Tauric AI Crypto Agents - Configuration\n")
            for k, v in env_dict.items():
                f.write(f"{k}={v}\n")

settings = Settings()

def load_risk_rules() -> Dict[str, Any]:
    rules_path = BASE_DIR / "config" / "risk_rules.yaml"
    if rules_path.exists():
        with open(rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

RISK_RULES = load_risk_rules()

