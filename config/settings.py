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
    
    # Reasoning models (Chief Trader & Debate - deep reasoning):
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

settings = Settings()

def load_risk_rules() -> Dict[str, Any]:
    rules_path = BASE_DIR / "config" / "risk_rules.yaml"
    if rules_path.exists():
        with open(rules_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

RISK_RULES = load_risk_rules()
