"""Jockey change analyzer — strategic trainer signal detection."""

import logging

logger = logging.getLogger(__name__)


class JockeyChangeAnalyzer:
    """Detect jockey changes — a strategic trainer signal.

    When a trainer books a different (usually better) jockey, it often
    signals they expect the horse to run well. Jockey upgrades are a
    strong positive indicator; downgrades are neutral to negative.
    """

    # Tiers refreshed 2026-05-17 against current-season DB stats:
    # - A Atzeni promoted TIER_2 → TIER_1 (527 rides this season at 10.8% wr,
    #   beats current TIER_1 R Moore/J McDonald on big-sample win rate)
    # - K Teetan removed from TIER_2 → defaults to TIER_3 (552 rides at 7.4%;
    #   user memory `project_may06_race_learnings` documents "Teetan blind spot")
    # - Added H Bentley, M Guyon, C L Chau to TIER_2 (all 8-9% wr on >150 rides)
    TIER_1 = {'Z Purton', 'J Moreira', 'H Bowman', 'R Moore', 'J McDonald',
              'A Atzeni'}
    TIER_2 = {'A Badel', 'B Avdulla', 'R Kingscote',
              'L Ferraris', 'D J Whyte', 'M Newnham',
              'H Bentley', 'M Guyon', 'C L Chau'}

    @staticmethod
    def _jockey_tier(name: str) -> int:
        """Return jockey tier: 1=elite, 2=strong, 3=other."""
        if name in JockeyChangeAnalyzer.TIER_1:
            return 1
        if name in JockeyChangeAnalyzer.TIER_2:
            return 2
        return 3

    @staticmethod
    def analyze_jockey_change(current_jockey: str, last_jockey: str) -> tuple:
        """Score jockey change impact. Returns (score 0-100, reason)."""
        if not last_jockey or not current_jockey:
            return 0.0, ""

        if current_jockey == last_jockey:
            return 0.0, ""  # No change — neutral, don't add as factor

        curr_tier = JockeyChangeAnalyzer._jockey_tier(current_jockey)
        last_tier = JockeyChangeAnalyzer._jockey_tier(last_jockey)

        # Any jockey change = trainer making a deliberate move (positive signal).
        # Direction (upgrade/downgrade) is unreliable — "downgrades" win too.
        # Elite booking is still the strongest signal.
        if curr_tier < last_tier:
            if curr_tier == 1:
                score = 82.0
                reason = f"Jockey UPGRADE: {last_jockey} → {current_jockey} (elite booking)"
            else:
                score = 68.0
                reason = f"Jockey upgrade: {last_jockey} → {current_jockey}"
        elif curr_tier > last_tier:
            score = 55.0
            reason = f"Jockey change: {last_jockey} → {current_jockey}"
        else:
            score = 60.0
            reason = f"Jockey change: {last_jockey} → {current_jockey}"

        return score, reason
