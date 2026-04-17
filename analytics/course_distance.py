"""Course and distance analyzer — C&D win record scoring."""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class CourseDistanceAnalyzer:
    """Scores horses based on their record at the specific course and distance."""

    @staticmethod
    def analyze_cd_form(
        horse_history: List[Dict],
        current_venue: str,
        current_distance: int,
    ) -> Tuple[float, str]:
        """Score C&D form.

        horse_history: list of dicts with keys:
            venue, distance (int), position (int)
        Returns (score 0-100, reason string).
        """
        if not horse_history:
            return 0.0, "No history data"

        # Normalize venue
        venue_lower = current_venue.lower()
        is_st = "sha tin" in venue_lower or "st" in venue_lower
        is_hv = "happy valley" in venue_lower or "hv" in venue_lower

        cd_runs = 0
        cd_wins = 0
        cd_places = 0  # top 3

        for race in horse_history:
            rv = race.get("venue", "").lower()
            rd = race.get("distance", 0)
            rp = race.get("position", 99)

            # Match venue
            if is_st and ("sha tin" not in rv and "st" not in rv):
                continue
            if is_hv and ("happy valley" not in rv and "hv" not in rv):
                continue
            if not is_st and not is_hv:
                continue

            # Match distance (±100m tolerance)
            try:
                rd_int = int(rd)
            except (ValueError, TypeError):
                continue
            if abs(rd_int - current_distance) > 100:
                continue

            cd_runs += 1
            if rp == 1:
                cd_wins += 1
            if rp <= 3:
                cd_places += 1

        if cd_runs == 0:
            return 50.0, "No C&D record"

        if cd_wins >= 3:
            score = 95.0
            label = f"Proven C&D star ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_wins == 2:
            score = 85.0
            label = f"Strong C&D form ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_wins == 1:
            score = 70.0
            label = f"C&D winner ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_places >= 2:
            score = 62.0
            label = f"C&D placer (0W/{cd_places}P from {cd_runs})"
        elif cd_places == 1:
            score = 55.0
            label = f"C&D placed once (from {cd_runs})"
        else:
            score = 42.0
            label = f"C&D: 0W/0P from {cd_runs} runs"

        return score, label
