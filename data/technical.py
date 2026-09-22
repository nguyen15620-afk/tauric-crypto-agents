import pandas as pd
import numpy as np
from typing import Dict, Any

class TechnicalIndicators:
    """
    Computes professional-grade technical indicators on OHLCV DataFrames:
    - RSI (Relative Strength Index, default 14)
    - MACD (Moving Average Convergence Divergence: 12, 26, 9)
    - Bollinger Bands (20, 2 stdev)
    - ATR (Average True Range, 14)
    - EMA Ribbon (20, 50, 200)
    """

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        Returns the DataFrame enriched with technical indicator columns.
        """
        df = df.copy()
        
        # Ensure correct datatypes
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = df[col].astype(float)

        close = df['close']
        high = df['high']
        low = df['low']

        # 1. RSI (14)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # 3. Bollinger Bands (20, 2)
        df['bb_middle'] = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # 4. ATR (14)
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()

        # 5. EMAs (20, 50, 200)
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()

        return df

    @classmethod
    def get_latest_metrics(cls, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract the most recent computed indicator values into a dictionary.
        """
        if df.empty:
            return {}
            
        enriched_df = cls.calculate_indicators(df)
        last_row = enriched_df.iloc[-1]
        prev_row = enriched_df.iloc[-2] if len(enriched_df) > 1 else last_row
        
        return {
            "current_price": float(last_row['close']),
            "rsi_14": float(round(last_row['rsi_14'], 2)) if pd.notna(last_row.get('rsi_14')) else None,
            "macd": float(round(last_row['macd'], 4)) if pd.notna(last_row.get('macd')) else None,
            "macd_signal": float(round(last_row['macd_signal'], 4)) if pd.notna(last_row.get('macd_signal')) else None,
            "macd_hist": float(round(last_row['macd_hist'], 4)) if pd.notna(last_row.get('macd_hist')) else None,
            "bb_upper": float(round(last_row['bb_upper'], 2)) if pd.notna(last_row.get('bb_upper')) else None,
            "bb_middle": float(round(last_row['bb_middle'], 2)) if pd.notna(last_row.get('bb_middle')) else None,
            "bb_lower": float(round(last_row['bb_lower'], 2)) if pd.notna(last_row.get('bb_lower')) else None,
            "atr_14": float(round(last_row['atr_14'], 2)) if pd.notna(last_row.get('atr_14')) else None,
            "ema_20": float(round(last_row['ema_20'], 2)) if pd.notna(last_row.get('ema_20')) else None,
            "ema_50": float(round(last_row['ema_50'], 2)) if pd.notna(last_row.get('ema_50')) else None,
            "ema_200": float(round(last_row['ema_200'], 2)) if pd.notna(last_row.get('ema_200')) else None,
            "price_vs_ema200": "ABOVE" if last_row['close'] > last_row['ema_200'] else "BELOW"
        }
