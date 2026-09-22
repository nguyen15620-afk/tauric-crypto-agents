import logging
import numpy as np
from typing import List, Tuple
from core.types import AnalystReport
from config.settings import RISK_RULES

logger = logging.getLogger("BeliefValidator")

class BeliefValidator:
    """
    Belief Vector Consensus Validator:
    Inspired by Chivu171/Multi-Agent-Crypto:
    Measures the degree of disagreement among specialized analysts using
    Variance and Maximum Pairwise Distance.
    
    If disagreement is higher than the threshold, triggers a multi-turn
    Bull vs Bear debate before final decision.
    """

    def __init__(self, threshold: float = 0.35):
        configured_threshold = RISK_RULES.get("debate", {}).get("divergence_threshold")
        self.threshold = float(configured_threshold) if configured_threshold is not None else threshold

    def evaluate_consensus(self, reports: List[AnalystReport]) -> Tuple[float, bool, str]:
        """
        Calculates divergence metric across analyst beliefs.
        Returns: (divergence_score, needs_debate, explanation)
        """
        if not reports:
            return 0.0, False, "No analyst reports to evaluate."

        scores = [r.belief.score for r in reports]
        confidences = [r.belief.confidence for r in reports]

        if len(scores) <= 1:
            return 0.0, False, "Single analyst report; no disagreement possible."

        # Compute variance and max pairwise difference
        variance = float(np.var(scores))
        max_diff = float(max(scores) - min(scores))
        
        # Divergence score combining variance and max spread
        divergence_score = float(round(max_diff * 0.6 + np.sqrt(variance) * 0.4, 3))
        needs_debate = bool(divergence_score >= self.threshold)

        if needs_debate:
            exp = (
                f"Phát hiện bất đồng quan điểm giữa các Analyst (Độ lệch {divergence_score:.2f} >= ngưỡng {self.threshold:.2f}). "
                f"Kích hoạt vòng tranh biện giữa Bull Researcher và Bear Researcher."
            )
        else:
            exp = (
                f"Các Analyst đạt độ đồng thuận cao (Độ lệch {divergence_score:.2f} < ngưỡng {self.threshold:.2f}). "
                f"Chuyển thẳng báo cáo sang Chief Trader để tổng hợp lệnh."
            )

        logger.info(exp)
        return divergence_score, needs_debate, exp
