"""Sectional analyzer — finishing speed from sectional times."""

import logging
from typing import Dict, List, Tuple

from analytics import parse_form_positions

logger = logging.getLogger(__name__)


class SectionalAnalyzer:
    """Scores horses based on their last-400m sectional times."""

    # Benchmarks (seconds) for last 400m
    EXCEPTIONAL = 22.5
    GOOD = 23.5
    AVERAGE = 24.5

    @classmethod
    def analyze_finishing_speed(
        cls, sectional_history: List[Dict]
    ) -> Tuple[float, str]:
        """Score finishing kick from historical sectional data.

        Each dict should have 'last_400m' (float, seconds).
        Returns (score 0-100, reason string).
        """
        if not sectional_history:
            return 0.0, "No sectional data"

        times = [
            s["last_400m"]
            for s in sectional_history
            if s.get("last_400m") and s["last_400m"] > 0
        ]
        if not times:
            return 0.0, "No sectional data"

        avg_time = sum(times) / len(times)
        best_time = min(times)

        # Score based on average last-400m
        if avg_time <= cls.EXCEPTIONAL:
            score = 95.0
            label = f"Exceptional finisher ({avg_time:.1f}s avg)"
        elif avg_time <= cls.GOOD:
            score = 80.0
            label = f"Strong finisher ({avg_time:.1f}s avg)"
        elif avg_time <= cls.AVERAGE:
            score = 60.0
            label = f"Average finisher ({avg_time:.1f}s avg)"
        else:
            score = 40.0
            label = f"Slow finisher ({avg_time:.1f}s avg)"

        # Bonus for improving trend (last 3 times getting faster)
        if len(times) >= 3:
            recent = times[-3:]
            if recent[-1] < recent[0]:
                score = min(100.0, score + 8)
                label += " [improving]"

        # Bonus for exceptional best
        if best_time <= cls.EXCEPTIONAL and score < 90:
            score = min(100.0, score + 5)

        return score, label

    @staticmethod
    def estimate_from_form(last_6: str, best_time: str) -> Tuple[float, str]:
        """Rough fallback when no sectional data available.

        Uses best_time string and form to estimate finishing ability.
        """
        if not last_6 and not best_time:
            return 0.0, "No sectional data"

        score = 50.0
        reasons = []

        # If horse frequently improves position (low finishes improving)
        if last_6:
            positions = parse_form_positions(last_6)
            if len(positions) >= 3:
                # positions is most-recent-first (HKJC convention)
                # recent = first half, older = second half
                recent = sum(positions[:len(positions)//2]) / max(1, len(positions)//2)
                older = sum(positions[len(positions)//2:]) / max(1, len(positions) - len(positions)//2)
                if recent < older:
                    # Lower position number = better finish; recent better than older = improving
                    score += 15
                    reasons.append("Improving trend")
                elif recent > older + 2:
                    score -= 10
                    reasons.append("Declining trend")

        if best_time:
            reasons.append(f"Best: {best_time}")

        return score, " | ".join(reasons) if reasons else "Est. sectional"
