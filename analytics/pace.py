"""Pace analyzer — running style classification and pace scenario impact."""

import logging
from typing import Dict, List, Tuple

from analytics import parse_form_positions

logger = logging.getLogger(__name__)


class PaceAnalyzer:
    """Classifies running style and models pace scenario impact."""

    STYLES = ("Leader", "Stalker", "Midfield", "Closer")

    @staticmethod
    def classify_running_style(
        running_positions: List[List[int]],
    ) -> str:
        """Classify from historical running-position arrays.

        Each inner list is [1st_call, 2nd_call, ..., finish].
        We look at the 1st-call position across races.
        """
        if not running_positions:
            return "Midfield"

        first_calls = [rp[0] for rp in running_positions if rp]
        if not first_calls:
            return "Midfield"

        avg = sum(first_calls) / len(first_calls)
        if avg <= 2:
            return "Leader"
        elif avg <= 5:
            return "Stalker"
        elif avg <= 9:
            return "Midfield"
        return "Closer"

    @staticmethod
    def estimate_style_from_form(
        last_6: str, draw: int = 0, distance: int = 1200
    ) -> str:
        """Heuristic fallback when no running-position history.

        Frequent 1st finishes + low draw + short race → likely Leader.
        Frequent high finishes + long race → likely Closer.
        """
        if not last_6:
            return "Midfield"

        positions = parse_form_positions(last_6)

        if not positions:
            return "Midfield"

        ones = positions.count(1)
        avg = sum(positions) / len(positions)

        # Frontrunner signals
        if ones >= 2 and draw <= 4 and distance <= 1200:
            return "Leader"
        if ones >= 3:
            return "Leader"
        if avg <= 2.5:
            return "Stalker"
        if avg >= 6 and distance >= 1600:
            return "Closer"
        if avg >= 5:
            return "Midfield"
        return "Stalker"

    @staticmethod
    def build_pace_scenario(
        styles: List[str], distance: int
    ) -> Dict[str, float]:
        """Compute advantage multipliers for each style given field composition.

        Returns dict like {"Leader": 1.15, "Stalker": 1.0, ...}.
        """
        n_leaders = styles.count("Leader")
        n_total = len(styles) or 1

        # Base multipliers (neutral)
        scenario: Dict[str, float] = {
            "Leader": 1.0,
            "Stalker": 1.0,
            "Midfield": 1.0,
            "Closer": 1.0,
        }

        # Distance bias
        if distance <= 1200:
            scenario["Leader"] += 0.12
            scenario["Stalker"] += 0.05
            scenario["Closer"] -= 0.08
        elif distance >= 1800:
            scenario["Closer"] += 0.12
            scenario["Stalker"] += 0.05
            scenario["Leader"] -= 0.05

        # Pace pressure
        if n_leaders == 0:
            # No designated leader — any frontrunner gets an easy lead
            scenario["Leader"] += 0.15
            scenario["Stalker"] += 0.05
        elif n_leaders == 1:
            # Lone leader — ideal scenario for that horse
            scenario["Leader"] += 0.10
        elif n_leaders == 2:
            # Contested lead — pressure but manageable
            scenario["Leader"] -= 0.05
            scenario["Closer"] += 0.05
        else:
            # Hot pace (3+ leaders) — closers thrive
            scenario["Leader"] -= 0.15
            scenario["Stalker"] += 0.05
            scenario["Closer"] += 0.15
            scenario["Midfield"] += 0.08

        return scenario

    @classmethod
    def analyze_pace(
        cls, style: str, scenario: Dict[str, float]
    ) -> Tuple[float, str]:
        """Score a horse's pace advantage given the field scenario.

        Returns (score 0-100, reason string).
        """
        if style not in cls.STYLES:
            style = "Midfield"

        multiplier = scenario.get(style, 1.0)

        # Convert multiplier (typically 0.8-1.3) to 0-100 score
        base = 50.0
        score = base + (multiplier - 1.0) * 200  # ±0.25 → ±50 points
        score = max(15.0, min(95.0, score))

        n_leaders = sum(1 for v in scenario.values() if v > 1.1)  # rough check
        if multiplier >= 1.15:
            pace_label = "Strong pace advantage"
        elif multiplier >= 1.05:
            pace_label = "Moderate pace edge"
        elif multiplier >= 0.95:
            pace_label = "Neutral pace"
        else:
            pace_label = "Pace against"

        reason = f"{style} - {pace_label} (x{multiplier:.2f})"
        return score, reason
