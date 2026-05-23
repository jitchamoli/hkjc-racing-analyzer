"""Draw bias analyzer — rail position aware draw scoring."""

import re
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class DrawBiasAnalyzer:
    """Scores draw advantage based on rail position, venue, and distance.

    HKJC moves the running rail each meeting. Positions range from
    A (true rail) through A+1, A+2, A+3, B, B+2, C, C+3.
    When the rail is out (e.g. C+3), inside draws lose their advantage
    because the ground near the rail is chewed up.
    """

    # Ideal draw by (venue, distance_bucket, rail_position).
    # Lower ideal_draw = inside draws favored.
    # Higher ideal_draw = middle/wide draws favored.
    # value = (ideal_draw, spread)  where spread controls falloff steepness
    RAIL_BIAS: Dict[str, Dict[str, Dict[str, Tuple[int, int]]]] = {
        "ST": {  # Sha Tin - wide, galloping track
            "chute_short": {  # ST 1000m + 1200m on the straight chute (no turn) —
                              # stand-side high-draw bias, opposite of round-1200m
                "A":   (9, 4),
                "A+1": (9, 4),
                "A+2": (8, 4),
                "A+3": (8, 4),
                "B":   (8, 4),
                "B+2": (7, 4),
                "C":   (7, 5),
                "C+3": (7, 5),
            },
            "short": {  # ≤1200m
                "A":   (2, 3),
                "A+1": (3, 3),
                "A+2": (4, 3),
                "A+3": (5, 4),
                "B":   (5, 4),
                "B+2": (6, 4),
                "C":   (7, 5),
                "C+3": (8, 5),
            },
            "mid": {  # 1400-1600m
                "A":   (4, 4),
                "A+1": (4, 4),
                "A+2": (5, 4),
                "A+3": (6, 5),
                "B":   (6, 5),
                "B+2": (7, 5),
                "C":   (7, 5),
                "C+3": (8, 6),
            },
            "long": {  # ≥1800m
                "A":   (6, 5),
                "A+1": (6, 5),
                "A+2": (7, 5),
                "A+3": (7, 6),
                "B":   (8, 6),
                "B+2": (8, 6),
                "C":   (9, 6),
                "C+3": (9, 7),
            },
        },
        "HV": {  # Happy Valley - tight, turning track, draw much more important
            "short": {
                "A":   (1, 2),
                "A+1": (2, 2),
                "A+2": (3, 3),
                "A+3": (4, 3),
                "B":   (4, 3),
                "B+2": (5, 3),
                "C":   (5, 4),
                "C+3": (6, 4),
            },
            "mid": {
                "A":   (2, 3),
                "A+1": (3, 3),
                "A+2": (4, 3),
                "A+3": (5, 4),
                "B":   (5, 4),
                "B+2": (6, 4),
                "C":   (6, 5),
                "C+3": (7, 5),
            },
            "long": {
                "A":   (4, 4),
                "A+1": (5, 4),
                "A+2": (5, 5),
                "A+3": (6, 5),
                "B":   (6, 5),
                "B+2": (7, 5),
                "C":   (7, 6),
                "C+3": (8, 6),
            },
        },
    }

    @staticmethod
    def _extract_rail(course: str) -> str:
        """Extract rail position from course string like 'Turf - \"A+2\" Course'."""
        m = re.search(r'"?([A-C](?:\+\d)?)"?', course)
        return m.group(1) if m else ""

    @staticmethod
    def _distance_bucket(distance: str) -> str:
        d = int(distance) if distance and distance.isdigit() else 1200
        if d <= 1200:
            return "short"
        elif d <= 1600:
            return "mid"
        return "long"

    @staticmethod
    def _venue_key(venue: str) -> str:
        if "Happy Valley" in venue or "HV" in venue.upper():
            return "HV"
        return "ST"

    @classmethod
    def analyze_draw_with_rail(
        cls, draw: int, distance: str, course: str, venue: str,
        field_size: int = 0, race_no: int = 0,
    ) -> Tuple[float, str]:
        """Score draw advantage considering rail position.

        field_size: when >0, clip ideal to the field — draw 12 of 12 is the
        widest the field has, but draw 12 of 7 doesn't exist post-scratchings.
        race_no: when ≥ 6 on a true-rail (A/A+1) configuration, the inside
        path is chewed up; nudge ideal one path outward.

        Returns (score 0-100, reason string).
        """
        if draw == 0:
            return 50.0, "No draw data"

        rail = cls._extract_rail(course)
        vk = cls._venue_key(venue)
        db = cls._distance_bucket(distance)
        # ST 1000m + 1200m run on the straight chute (no turn) — flip to the
        # stand-side high-draw bias bucket instead of the round-track low-draw.
        if vk == "ST" and db == "short":
            db = "chute_short"

        if not rail or vk not in cls.RAIL_BIAS:
            return cls._basic_draw_score(draw, distance)

        bias = cls.RAIL_BIAS.get(vk, {}).get(db, {})
        ideal, spread = bias.get(rail, (5, 4))
        if field_size and field_size > 0 and ideal > field_size:
            ideal = field_size
        # Rail degradation: by race 6+ on a true-rail configuration the inside
        # path is chewed and the live rail drifts a path or two outward.
        if race_no >= 6 and rail in ("A", "A+1") and ideal <= 3:
            ideal += 1

        diff = abs(draw - ideal)
        if diff == 0:
            score = 95.0
        elif diff <= 1:
            score = 85.0
        elif diff <= spread:
            score = 75.0 - (diff - 1) * 5
        else:
            score = max(30.0, 60.0 - diff * 4)

        # Extra penalty for very wide draws at Happy Valley
        if vk == "HV" and draw >= 10:
            score = max(25.0, score - 10)

        rail_label = f"Rail {rail}" if rail else "Rail ?"
        if score >= 85:
            reason = f"Ideal draw ({draw}) on {rail_label}"
        elif score >= 70:
            reason = f"Good draw ({draw}) on {rail_label}"
        elif score >= 50:
            reason = f"Avg draw ({draw}) on {rail_label}"
        else:
            reason = f"Poor draw ({draw}) on {rail_label}"

        return score, reason

    @staticmethod
    def _basic_draw_score(draw: int, distance: str) -> Tuple[float, str]:
        """Fallback when no rail info available."""
        d = int(distance) if distance and distance.isdigit() else 1200
        ideal = 3 if d <= 1000 else (5 if d <= 1400 else 8)
        diff = abs(draw - ideal)
        if diff == 0:
            return 95.0, f"Ideal barrier ({draw})"
        elif diff <= 2:
            return 80.0, f"Good barrier ({draw})"
        elif draw > 12:
            return 55.0, f"Wide draw ({draw})"
        return 65.0, f"Average barrier ({draw})"
