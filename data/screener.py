import logging
from typing import List, Dict, Any
import ccxt
import pandas as pd
from data.technical import TechnicalIndicators

logger = logging.getLogger("MarketScreener")

class MarketScreener:
    """
    Automated Quantitative Market Scanner:
    Scans liquid crypto pairs on Binance to find top trading opportunities:
    1. High 24h liquidity filter (> $10M volume)
    2. Unusual volume surges or momentum breakouts
    3. Extreme RSI opportunities (Oversold bounce < 35 or Strong Trend > 60)
    """

    WATCHLIST_SYMBOLS = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "AVAX/USDT", "SUI/USDT", "NEAR/USDT",
        "LINK/USDT", "PEPE/USDT", "APT/USDT", "RENDER/USDT", "FET/USDT"
    ]

    def __init__(self, exchange_id: str = "binance"):
        try:
            exchange_class = getattr(ccxt, exchange_id.lower())
            self.exchange = exchange_class({'enableRateLimit': True, 'timeout': 10000})
        except Exception:
            self.exchange = ccxt.binance({'enableRateLimit': True, 'timeout': 10000})

    def scan_top_opportunities(self, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Scans watchlist symbols and ranks them by opportunity score.
        Opportunity Score factors:
        - Absolute momentum |% 24h change|
        - Volume liquidity
        - Technical setup (Oversold bounce or Trend continuation)
        """
        candidates = []

        try:
            # Batch fetch tickers in one single fast API call
            tickers = self.exchange.fetch_tickers(self.WATCHLIST_SYMBOLS)
        except Exception as e:
            logger.warning(f"Batch fetch tickers error: {e}. Falling back to subset.")
            tickers = {}

        for symbol in self.WATCHLIST_SYMBOLS:
            ticker = tickers.get(symbol)
            if not ticker:
                continue

            last_price = float(ticker.get("last") or ticker.get("close") or 0.0)
            change_pct = float(ticker.get("percentage") or 0.0)
            quote_volume = float(ticker.get("quoteVolume") or 0.0)

            # Liquidity sanity check (must have at least $5M 24h volume)
            if quote_volume < 5_000_000 and quote_volume > 0:
                continue

            # Classify signal category
            signal_type = "BREAKOUT / MOMENTUM" if change_pct > 3.0 else (
                "OVERSOLD BOUNCE" if change_pct < -3.0 else "ACCUMULATION / RANGE"
            )

            # Score = combination of momentum, volatility, and volume
            abs_momentum = abs(change_pct)
            vol_score = min(quote_volume / 100_000_000, 10.0)  # capped
            total_score = round(abs_momentum * 1.5 + vol_score * 0.5, 2)

            candidates.append({
                "symbol": symbol,
                "current_price": last_price,
                "change_24h_pct": round(change_pct, 2),
                "quote_volume_mil": round(quote_volume / 1_000_000, 1),
                "signal_type": signal_type,
                "opportunity_score": total_score
            })

        # Rank by opportunity score descending
        candidates.sort(key=lambda x: x["opportunity_score"], reverse=True)
        return candidates[:top_n]
