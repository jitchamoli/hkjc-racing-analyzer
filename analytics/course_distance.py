"""Course and distance analyzer — C&D win record scoring.

Key insight: a horse returning to the exact distance + venue where it
previously won ("winning trip replication") is one of the strongest
signals in HK racing. This gets a significant boost, especially if
the win was recent (within ~120 days).
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class CourseDistanceAnalyzer:
    """Scores horses based on their record at the specific course and distance."""

    @staticmethod
    def analyze_cd_form(
        horse_history: List[Dict],
        current_venue: str,
        current_distance: int,
        current_date: str = None,
    ) -> Tuple[float, str]:
        """Score C&D form with winning trip replication boost.

        horse_history: list of dicts with keys:
            venue, distance (int), position (int), date (str, optional)
        Returns (score 0-100, reason string).
        """
        if not horse_history:
            return 0.0, "No history data"

        # Normalize venue
        venue_lower = current_venue.lower()
        is_st = "sha tin" in venue_lower or venue_lower == "st"
        is_hv = "happy valley" in venue_lower or venue_lower == "hv"

        cd_runs = 0
        cd_wins = 0
        cd_places = 0  # top 3
        recent_cd_win = False  # winning trip replication
        cd_win_days_ago = None

        for race in horse_history:
            rv = race.get("venue", "").lower()
            rd = race.get("distance", 0)
            rp = race.get("position", 99)

            # Match venue
            rv_is_st = "sha tin" in rv or rv == "st"
            rv_is_hv = "happy valley" in rv or rv == "hv"
            if is_st and not rv_is_st:
                continue
            if is_hv and not rv_is_hv:
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
                # Check recency of C&D win
                race_date = race.get("date", "")
                if current_date and race_date:
                    try:
                        r_dt = datetime.strptime(race_date.replace("/", "-"), "%Y-%m-%d")
                        c_dt = datetime.strptime(current_date.replace("/", "-"), "%Y-%m-%d")
                        days = (c_dt - r_dt).days
                        if days <= 180:
                            recent_cd_win = True
                            if cd_win_days_ago is None or days < cd_win_days_ago:
                                cd_win_days_ago = days
                    except Exception:
                        recent_cd_win = True  # assume recent if can't parse
            if rp <= 3:
                cd_places += 1

        if cd_runs == 0:
            return 50.0, "No C&D record"

        # Base score from win/place record
        if cd_wins >= 3:
            score = 95.0
            label = f"Proven C&D star ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_wins == 2:
            score = 88.0
            label = f"Strong C&D form ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_wins == 1:
            score = 78.0
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

        # Winning trip replication boost — horse won at this exact C&D recently
        if recent_cd_win and cd_win_days_ago is not None:
            if cd_win_days_ago <= 90:
                score = min(100, score + 12)
                label += f" | WINNING TRIP ({cd_win_days_ago}d ago)"
            elif cd_win_days_ago <= 180:
                score = min(100, score + 8)
                label += f" | Winning trip ({cd_win_days_ago}d ago)"
        elif recent_cd_win:
            score = min(100, score + 8)
            label += " | Winning trip"

        return score, label
