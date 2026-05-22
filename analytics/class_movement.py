"""Class movement analyzer — class drop/rise detection."""

import logging

from ml.features import extract_class_rank

logger = logging.getLogger(__name__)


class ClassMovementAnalyzer:
    """Detect class drop/rise advantage.

    HKJC tier ladder (high → low): Group 1/2 → 0, Group 3/Class 1 → 1, …
    Class 5 → 5, GRIFFIN/MAIDEN → 4. A horse dropping from Class 3 to Class 4
    races weaker opposition; rising from Class 5 to Class 4 faces tougher.
    """

    @staticmethod
    def _parse_class(class_str: str):
        """Return the canonical class rank (0-5, lower = higher class) or None
        if unparseable. Delegates to ml.features.extract_class_rank — the
        single source of truth — so the rule engine and ML agree on every
        class string (previously this had a local ladder that mis-parsed G1/G2/
        G3 as Class 1/2/3 and put GRIFFIN at 6 instead of 4)."""
        return extract_class_rank(class_str)

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
        if current_rank is None:
            return 50.0, "Class info unavailable"

        prev_rank = None
        for race in horse_history:
            rc = race.get("class", "") or race.get("class_type", "")
            rank = ClassMovementAnalyzer._parse_class(rc)
            if rank is not None:
                prev_rank = rank
                break

        if prev_rank is None:
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
