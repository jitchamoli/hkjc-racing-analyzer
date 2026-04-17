"""Class movement analyzer — class drop/rise detection."""

import re
import logging

logger = logging.getLogger(__name__)


class ClassMovementAnalyzer:
    """Detect class drop/rise advantage.

    HKJC classes: 1 (top) → 5 (bottom), plus Griffin (debut).
    A horse dropping from Class 3 to Class 4 races weaker opposition.
    A horse rising from Class 5 to Class 4 faces tougher competition.
    """

    CLASS_RANK = {
        "1": 1, "class 1": 1,
        "2": 2, "class 2": 2,
        "3": 3, "class 3": 3,
        "4": 4, "class 4": 4,
        "5": 5, "class 5": 5,
        "griffin": 6,
    }

    @staticmethod
    def _parse_class(class_str: str) -> int:
        if not class_str:
            return 0
        cl = class_str.strip().lower()
        if cl in ClassMovementAnalyzer.CLASS_RANK:
            return ClassMovementAnalyzer.CLASS_RANK[cl]
        m = re.search(r"(\d)", cl)
        if m:
            rank = int(m.group(1))
            if 1 <= rank <= 5:
                return rank
        if "griffin" in cl:
            return 6
        return 0

    @staticmethod
    def analyze_class_movement(
        horse_history: list,
        current_class: str,
    ) -> tuple:
        """Score class movement.

        Returns (score 0-100, reason string).
        Higher score = bigger advantage from class drop.
        """
        current_rank = ClassMovementAnalyzer._parse_class(current_class)
        if current_rank == 0:
            return 50.0, "Class info unavailable"

        prev_rank = 0
        for race in horse_history:
            rc = race.get("class", "") or race.get("class_type", "")
            rank = ClassMovementAnalyzer._parse_class(rc)
            if rank > 0:
                prev_rank = rank
                break

        if prev_rank == 0:
            return 50.0, "No previous class data"

        diff = prev_rank - current_rank  # positive = rising, negative = dropping

        if diff == 0:
            return 50.0, f"Same class (Class {current_rank})"
        elif diff < 0:
            drop = abs(diff)
            if drop >= 2:
                return 88.0, f"BIG DROP: Class {prev_rank}→{current_rank} (much weaker field)"
            else:
                return 72.0, f"Class drop: {prev_rank}→{current_rank} (easier opposition)"
        else:
            if diff >= 2:
                return 25.0, f"BIG RISE: Class {prev_rank}→{current_rank} (much tougher)"
            else:
                return 38.0, f"Class rise: {prev_rank}→{current_rank} (harder opposition)"
