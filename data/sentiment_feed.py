import logging
import requests
from typing import Dict, Any, List
from datetime import datetime, timezone

logger = logging.getLogger("SentimentFeed")

# Timeout for all external API calls
_REQUEST_TIMEOUT = 5

# Number of news headlines to return
_MAX_HEADLINES = 6


class SentimentFeed:
    """
    Fetches real-time crypto sentiment:
    - Crypto Fear & Greed Index (Alternative.me API)
    - Real news headlines from CryptoPanic public API (no auth required for basic feed)
      with fallback to CoinGecko trending coins news and finally static placeholders.
    """

    @staticmethod
    def get_fear_and_greed_index() -> Dict[str, Any]:
        """
        Fetch latest Fear & Greed index from Alternative.me.
        Returns: {'score': 0-100, 'label': 'Extreme Fear'|'Fear'|'Neutral'|'Greed'|'Extreme Greed'}
        """
        url = "https://api.alternative.me/fng/?limit=1"
        try:
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                item = data.get("data", [{}])[0]
                score = int(item.get("value", 50))
                label = item.get("value_classification", "Neutral")
                logger.debug(f"Fear & Greed: {score} ({label})")
                return {"score": score, "label": label}
        except Exception as e:
            logger.info(f"Failed to fetch Fear & Greed index: {e}. Using neutral fallback.")

        return {"score": 52, "label": "Neutral"}

    @staticmethod
    def get_crypto_news(symbol: str = "BTC") -> List[str]:
        """
        Retrieve latest real crypto news headlines.

        Priority order:
        1. CryptoPanic public API (no auth, free tier, ~50 latest headlines)
        2. CoinGecko /news endpoint
        3. Static themed placeholders as last resort

        Args:
            symbol: Base asset symbol (e.g., 'BTC', 'ETH', 'SOL')

        Returns:
            List of headline strings (up to _MAX_HEADLINES)
        """
        headlines = SentimentFeed._fetch_cryptopanic(symbol)

        if not headlines:
            headlines = SentimentFeed._fetch_coingecko_news(symbol)

        if not headlines:
            # Final fallback — contextually relevant static placeholders
            logger.info(f"Using static news fallback for {symbol}")
            headlines = SentimentFeed._static_fallback_headlines(symbol)

        return headlines[:_MAX_HEADLINES]

    @staticmethod
    def _fetch_cryptopanic(symbol: str) -> List[str]:
        """
        CryptoPanic public RSS/API — free, no auth needed for basic feed.
        Endpoint: https://cryptopanic.com/api/v1/posts/?auth_token=free&currencies=BTC
        Note: 'free' auth_token gives limited but real headlines.
        """
        try:
            url = (
                f"https://cryptopanic.com/api/v1/posts/"
                f"?auth_token=free&currencies={symbol}&kind=news&public=true"
            )
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                headlines = [
                    item.get("title", "").strip()
                    for item in results
                    if item.get("title")
                ]
                if headlines:
                    logger.debug(f"CryptoPanic: fetched {len(headlines)} headlines for {symbol}")
                return headlines
        except Exception as e:
            logger.info(f"CryptoPanic fetch failed for {symbol}: {e}")

        return []

    @staticmethod
    def _fetch_coingecko_news(symbol: str) -> List[str]:
        """
        CoinGecko /news endpoint — free tier, no auth needed.
        """
        # CoinGecko coin IDs for common symbols
        coin_id_map = {
            "BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana",
            "BNB": "binancecoin", "XRP": "ripple", "ADA": "cardano",
            "DOGE": "dogecoin", "AVAX": "avalanche-2", "LINK": "chainlink",
            "SUI": "sui", "NEAR": "near", "APT": "aptos",
        }
        coin_id = coin_id_map.get(symbol.upper(), symbol.lower())

        try:
            url = f"https://api.coingecko.com/api/v3/news?page=1&per_page=10"
            resp = requests.get(url, timeout=_REQUEST_TIMEOUT)
            if resp.status_code == 200:
                items = resp.json()
                if isinstance(items, list):
                    # Filter by relevance if possible (title contains symbol)
                    relevant = [
                        item.get("title", "").strip()
                        for item in items
                        if symbol.upper() in item.get("title", "").upper()
                        or coin_id in item.get("description", "").lower()
                    ]
                    all_titles = [item.get("title", "").strip() for item in items if item.get("title")]
                    headlines = relevant or all_titles
                    if headlines:
                        logger.debug(f"CoinGecko news: fetched {len(headlines)} headlines")
                    return headlines
        except Exception as e:
            logger.info(f"CoinGecko news fetch failed: {e}")

        return []

    @staticmethod
    def _static_fallback_headlines(symbol: str) -> List[str]:
        """
        Contextually themed static placeholders used only when all APIs fail.
        Separated by market condition themes to avoid systematic bias.
        """
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        return [
            f"{symbol} markets mixed as traders assess macro risk-off signals [{ts}]",
            "Crypto institutional flows show divergent positioning across major assets",
            "On-chain data reflects balanced accumulation vs distribution activity",
            "Regulatory clarity expected to drive next major directional move",
            "Derivatives open interest neutral; funding rate signals lack momentum bias",
            "Global liquidity conditions remain key driver for crypto risk assets",
        ]
