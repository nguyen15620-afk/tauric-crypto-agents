import unittest
import pandas as pd
import numpy as np
from data.technical import TechnicalIndicators

class TestTechnicalIndicators(unittest.TestCase):
    def setUp(self):
        # Create 50 synthetic bars
        np.random.seed(42)
        closes = [100.0]
        for _ in range(50):
            closes.append(closes[-1] * (1 + np.random.normal(0, 0.02)))
            
        self.df = pd.DataFrame({
            'timestamp': [i * 60000 for i in range(50)],
            'open': closes[:-1],
            'high': [c * 1.01 for c in closes[:-1]],
            'low': [c * 0.99 for c in closes[:-1]],
            'close': closes[1:],
            'volume': [1000.0] * 50
        })

    def test_indicators_calculation(self):
        metrics = TechnicalIndicators.get_latest_metrics(self.df)
        self.assertIn("current_price", metrics)
        self.assertIn("rsi_14", metrics)
        self.assertIn("macd", metrics)
        self.assertIn("bb_upper", metrics)
        self.assertIn("atr_14", metrics)
        self.assertIn("ema_200", metrics)
        
        # Verify RSI bounds
        self.assertTrue(0 <= metrics["rsi_14"] <= 100)
        # Verify Bollinger Bands ordering
        self.assertGreater(metrics["bb_upper"], metrics["bb_middle"])
        self.assertGreater(metrics["bb_middle"], metrics["bb_lower"])

if __name__ == "__main__":
    unittest.main()
