"""Weight analyzer — weight carried vs rating analysis."""

import re
import logging

logger = logging.getLogger(__name__)


class WeightAnalyzer:
    """Analyze weight carried relative to rating and field.

    In HKJC handicap races, weight is assigned based on rating.
    Lower weight relative to field = advantage.
    Apprentice jockey claims (3-10 lbs) reduce effective weight.
    """

    APPRENTICE_CLAIMS = {
        10: "10lb claimer",
        7: "7lb claimer",
        5: "5lb claimer",
        3: "3lb claimer",
    }

    @staticmethod
    def parse_weight(weight_str: str) -> float:
        """Parse weight string like '133' or '115(3)' into effective lbs."""
        if not weight_str:
            return 0.0
        w = weight_str.strip()
        m = re.match(r"(\d+)\s*\((\d+)\)", w)
        if m:
            return float(m.group(1)) - float(m.group(2))
        m = re.match(r"(\d+)", w)
        if m:
            return float(m.group(1))
        return 0.0

    @staticmethod
    def analyze_weight(
        weight_str: str,
        rating: int,
        field_weights: list = None,
    ) -> tuple:
        """Score weight advantage. Returns (score 0-100, reason string)."""
        effective = WeightAnalyzer.parse_weight(weight_str)
        if effective <= 0:
            return 50.0, "Weight data unavailable"

        claim = 0
        m = re.match(r"\d+\s*\((\d+)\)", weight_str.strip())
        if m:
            claim = int(m.group(1))

        reasons = []

        # Weight-to-rating ratio
        if rating > 0:
            ratio = rating / effective
            if ratio >= 0.55:
                score = 75.0
                reasons.append(f"Light weight for rating ({effective:.0f}lbs, rated {rating})")
            elif ratio >= 0.45:
                score = 55.0
                reasons.append(f"Fair weight ({effective:.0f}lbs, rated {rating})")
            else:
                score = 35.0
                reasons.append(f"Heavy weight for rating ({effective:.0f}lbs, rated {rating})")
        else:
            score = 50.0

        # Compare to field
        if field_weights:
            field_effective = [
                WeightAnalyzer.parse_weight(w) for w in field_weights
            ]
            field_effective = [w for w in field_effective if w > 0]
            if field_effective:
                avg_weight = sum(field_effective) / len(field_effective)
                diff = effective - avg_weight
                if diff <= -5:
                    score = min(85.0, score + 15)
                    reasons.append(f"{abs(diff):.0f}lbs below field avg")
                elif diff >= 5:
                    score = max(25.0, score - 10)
                    reasons.append(f"{diff:.0f}lbs above field avg")

        # Apprentice claim bonus
        if claim > 0:
            claim_label = WeightAnalyzer.APPRENTICE_CLAIMS.get(
                claim, f"{claim}lb claim"
            )
            score = min(90.0, score + claim * 1.5)
            reasons.append(f"Apprentice {claim_label}")

        if not reasons:
            return 50.0, f"{effective:.0f}lbs carried"

        return score, "; ".join(reasons)
