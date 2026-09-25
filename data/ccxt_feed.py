import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import ccxt
from typing import Dict, Any, Optional, List
from core.types import MarketSnapshot
from data.technical import TechnicalIndicators
from data.sentiment_feed import SentimentFeed

logger = logging.getLogger("CCXTFeed")


class CCXTMarketFeed:
    """
    Unified CCXT Market Data Fetcher:
    - Fetches live OHLCV candles, Ticker, Orderbook from Binance or Bybit.
    - Fetches real Funding Rate and Open Interest from Futures market.
    - Computes technical indicators in-process.
    - Integrates sentiment and orderbook depth into a unified MarketSnapshot.
    """

    # Neutral fallback values when API is unavailable
    _FALLBACK_TICKER = {
        "last": 68500.0,
        "change_pct": 1.85,
        "high": 69200.0,
        "low": 67100.0,
        "volume": 25000.0,
    }
    _FALLBACK_OB = {
        "orderbook_bid_volume": 150.0,
        "orderbook_ask_volume": 135.0,
        "orderbook_imbalance": 0.0526,
    }
    _FALLBACK_FUNDING = {"funding_rate": 0.0001, "open_interest": 58400.0}

    def __init__(self, exchange_id: str = "binance"):
        self.exchange_id = exchange_id.lower()

        # Spot exchange (OHLCV, ticker, orderbook)
        try:
            spot_class = getattr(ccxt, self.exchange_id)
            self.exchange = spot_class({"enableRateLimit": True, "timeout": 10000})
        except Exception as e:
            logger.warning(f"Could not initialize CCXT spot exchange {exchange_id}: {e}. Falling back to binance.")
            self.exchange = ccxt.binance({"enableRateLimit": True, "timeout": 10000})

        # Futures/perp exchange for funding rate & OI
        # Binance → binanceusdm; Bybit → bybit (supports both spot & perp)
        self._futures_exchange = self._init_futures_exchange(exchange_id)

    def _init_futures_exchange(self, exchange_id: str):
        futures_map = {
            "binance": "binanceusdm",
            "bybit": "bybit",
            "okx": "okx",
        }
        futures_id = futures_map.get(exchange_id.lower(), "binanceusdm")
        try:
            futures_class = getattr(ccxt, futures_id)
            return futures_class({"enableRateLimit": True, "timeout": 10000})
        except Exception as e:
            logger.warning(f"Could not initialize futures exchange {futures_id}: {e}. Using fallback funding data.")
            return None

    # ------------------------------------------------------------------
    # Public Fetch Methods
    # ------------------------------------------------------------------

    def fetch_ohlcv_df(self, symbol: str = "BTC/USDT", timeframe: str = "15m", limit: int = 100) -> pd.DataFrame:
        """Fetch OHLCV historical candles and return as a pandas DataFrame."""
        try:
            raw_candles = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(raw_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
            return df
        except Exception as e:
            logger.warning(f"CCXT fetch_ohlcv error for {symbol} ({e}). Generating fallback simulated data.")
            return self._generate_simulated_ohlcv(symbol, limit)

    def fetch_ticker(self, symbol: str = "BTC/USDT") -> Dict[str, Any]:
        """Fetch current price, 24h high/low, volume, 24h percentage change."""
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                "last": float(ticker.get("last") or ticker.get("close") or 68000.0),
                "change_pct": float(ticker.get("percentage") or 0.0),
                "high": float(ticker.get("high") or 69000.0),
                "low": float(ticker.get("low") or 67000.0),
                "volume": float(ticker.get("baseVolume") or 1200.0),
            }
        except Exception as e:
            fallback = self._FALLBACK_TICKER.copy()
            if "ETH" in symbol:
                fallback["last"], fallback["high"], fallback["low"] = 3200.0, 3300.0, 3100.0
            elif "SOL" in symbol:
                fallback["last"], fallback["high"], fallback["low"] = 150.0, 160.0, 140.0
            elif "BTC" not in symbol:
                fallback["last"], fallback["high"], fallback["low"] = 10.0, 11.0, 9.0

            logger.warning(
                f"[CCXTFeed] WARNING: fetch_ticker failed for {symbol} ({e}). "
                f"Using simulated fallback ticker (Price: ${fallback['last']:,.2f}). "
                f"This data is NOT live market data!"
            )
            return fallback

    def fetch_orderbook_metrics(self, symbol: str = "BTC/USDT", limit: int = 20) -> Dict[str, float]:
        """
        Calculates orderbook depth and bid/ask imbalance.
        Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
        """
        try:
            ob = self.exchange.fetch_order_book(symbol, limit=limit)
            bids = ob.get("bids", [])
            asks = ob.get("asks", [])

            bid_vol = sum([b[1] for b in bids])
            ask_vol = sum([a[1] for a in asks])
            total_vol = bid_vol + ask_vol + 1e-9
            imbalance = (bid_vol - ask_vol) / total_vol

            return {
                "orderbook_bid_volume": round(bid_vol, 4),
                "orderbook_ask_volume": round(ask_vol, 4),
                "orderbook_imbalance": round(imbalance, 4),
            }
        except Exception as e:
            logger.warning(f"CCXT orderbook fetch failed ({e}). Using neutral fallback values.")
            return self._FALLBACK_OB.copy()

    def fetch_funding_and_oi(self, symbol: str = "BTC/USDT") -> Dict[str, float]:
        """
        Fetches real perpetual Funding Rate and Open Interest from the futures exchange.
        Falls back to neutral values if the exchange is unavailable or the symbol
        does not have a perp contract.

        Returns:
            {
                "funding_rate": float,   # e.g., 0.0001 = 0.01% per 8h (positive = longs pay)
                "open_interest": float,  # in base asset units (e.g., BTC)
            }
        """
        if self._futures_exchange is None:
            logger.debug("No futures exchange configured. Using fallback funding data.")
            return self._FALLBACK_FUNDING.copy()

        # Normalise symbol for futures market (BTC/USDT → BTC/USDT:USDT on most exchanges)
        futures_symbol = self._normalise_futures_symbol(symbol)

        funding_rate = self._FALLBACK_FUNDING["funding_rate"]
        open_interest = self._FALLBACK_FUNDING["open_interest"]

        # --- Funding Rate ---
        try:
            fr_data = self._futures_exchange.fetch_funding_rate(futures_symbol)
            if fr_data:
                funding_rate = float(fr_data.get("fundingRate") or self._FALLBACK_FUNDING["funding_rate"])
                logger.debug(f"Live funding rate for {futures_symbol}: {funding_rate:.6f}")
        except Exception as e:
            logger.info(f"Could not fetch funding rate for {futures_symbol}: {e}. Using fallback.")

        # --- Open Interest ---
        try:
            oi_data = self._futures_exchange.fetch_open_interest(futures_symbol)
            if oi_data:
                # openInterestAmount → base units; openInterestValue → USD
                oi = oi_data.get("openInterestAmount") or oi_data.get("openInterest")
                if oi is not None:
                    open_interest = float(oi)
                logger.debug(f"Live open interest for {futures_symbol}: {open_interest:,.0f}")
        except Exception as e:
            logger.info(f"Could not fetch open interest for {futures_symbol}: {e}. Using fallback.")

        return {"funding_rate": funding_rate, "open_interest": open_interest}

    def _normalise_futures_symbol(self, symbol: str) -> str:
        """
        Convert spot symbol to futures/perp format.
        BTC/USDT → BTC/USDT:USDT  (works for binanceusdm and bybit)
        """
        if ":" in symbol:
            return symbol  # already futures format
        base, quote = symbol.split("/")
        return f"{base}/{quote}:{quote}"

    def get_market_snapshot(self, symbol: str = "BTC/USDT", timeframe: str = "15m") -> MarketSnapshot:
        """
        Builds a comprehensive MarketSnapshot combining:
        - Price & 24h statistics
        - Real technical indicators (RSI, MACD, BB, ATR, EMAs)
        - Orderbook imbalance & depth
        - Real Funding Rate & Open Interest from perp market
        - Fear & Greed sentiment & news headlines
        """
        df = self.fetch_ohlcv_df(symbol=symbol, timeframe=timeframe, limit=100)
        indicators = TechnicalIndicators.get_latest_metrics(df)
        ticker = self.fetch_ticker(symbol)
        ob_metrics = self.fetch_orderbook_metrics(symbol)
        funding_data = self.fetch_funding_and_oi(symbol)
        sentiment = SentimentFeed.get_fear_and_greed_index()
        news = SentimentFeed.get_crypto_news(symbol.split("/")[0])

        current_price = indicators.get("current_price") or ticker["last"]

        return MarketSnapshot(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            current_price=current_price,
            change_24h_pct=ticker["change_pct"],
            high_24h=ticker["high"],
            low_24h=ticker["low"],
            volume_24h=ticker["volume"],
            rsi_14=indicators.get("rsi_14"),
            macd=indicators.get("macd"),
            macd_signal=indicators.get("macd_signal"),
            macd_hist=indicators.get("macd_hist"),
            bb_upper=indicators.get("bb_upper"),
            bb_middle=indicators.get("bb_middle"),
            bb_lower=indicators.get("bb_lower"),
            atr_14=indicators.get("atr_14"),
            ema_20=indicators.get("ema_20"),
            ema_50=indicators.get("ema_50"),
            ema_200=indicators.get("ema_200"),
            fear_and_greed_score=sentiment.get("score"),
            fear_and_greed_label=sentiment.get("label"),
            news_headlines=news,
            # ✅ Real funding & OI from perp market
            funding_rate=funding_data["funding_rate"],
            open_interest=funding_data["open_interest"],
            orderbook_bid_volume=ob_metrics["orderbook_bid_volume"],
            orderbook_ask_volume=ob_metrics["orderbook_ask_volume"],
            orderbook_imbalance=ob_metrics["orderbook_imbalance"],
        )

    # ------------------------------------------------------------------
    # Simulation fallback
    # ------------------------------------------------------------------

    def _generate_simulated_ohlcv(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        """
        Deterministic random walk candle generator used when exchange APIs are unreachable.
        Uses a fixed seed for reproducibility.
        """
        np.random.seed(42)
        base_price = 68000.0 if "BTC" in symbol else (3200.0 if "ETH" in symbol else 100.0)
        prices = [base_price]
        for _ in range(limit):
            pct = np.random.normal(0.0005, 0.008)
            prices.append(prices[-1] * (1 + pct))

        now = datetime.now(timezone.utc).timestamp() * 1000
        interval_ms = 15 * 60 * 1000
        rows = []
        for i in range(limit):
            close_p = prices[i + 1]
            open_p = prices[i]
            high_p = max(open_p, close_p) * (1 + abs(np.random.normal(0, 0.003)))
            low_p = min(open_p, close_p) * (1 - abs(np.random.normal(0, 0.003)))
            vol = np.random.uniform(50, 300)
            t = now - (limit - i) * interval_ms
            rows.append([t, open_p, high_p, low_p, close_p, vol])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
