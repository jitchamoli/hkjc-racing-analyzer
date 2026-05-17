"""Going preference analyzer — track condition affinity scoring."""

import logging

logger = logging.getLogger(__name__)


class GoingPreferenceAnalyzer:
    """Analyze horse's track condition (going) preference.

    Going is one of the strongest predictors in horse racing — a horse with
    5 wins on Good ground can completely flop on Yielding. In HK, going can
    change mid-meeting if it rains, making this the biggest upset driver.

    Going categories (HKJC):
      GOOD TO FIRM  — fast, dry surface
      GOOD          — standard conditions
      GOOD TO YIELDING — slightly soft, some give
      YIELDING      — soft, rain-affected
      SOFT / HEAVY  — waterlogged (rare in HK)
    """

    # Map going descriptions to wetness level (0=firm, 4=heavy)
    GOING_WETNESS = {
        'GOOD TO FIRM': 0,
        'GOOD': 1,
        'GOOD TO YIELDING': 2,
        'YIELDING': 3,
        'YIELDING TO SOFT': 3,
        'SOFT': 4,
        'HEAVY': 4,
    }

    @staticmethod
    def _classify_going(going: str) -> int:
        """Map going string to wetness level 0-4.

        Iterates keys longest-first so substring overlaps don't mis-classify
        (audit fix 2026-05-17: 'GOOD TO YIELDING' was matching 'GOOD' and
        returning level 1 instead of 2 because dict insertion order put GOOD
        before GOOD TO YIELDING — silently bypassed all wet-track logic on
        GTY meetings).
        """
        if not going:
            return 1  # default to GOOD
        g = going.upper().strip()
        for key, level in sorted(GoingPreferenceAnalyzer.GOING_WETNESS.items(),
                                  key=lambda kv: -len(kv[0])):
            if key in g:
                return level
        return 1  # default GOOD

    @staticmethod
    def analyze_going_preference(
        going_record: dict,
        today_going: str,
        total_runs: int = 0,
    ) -> tuple:
        """Score horse's affinity for today's going.

        going_record: dict with runs, wins, places, avg_pos on today's going
        today_going: today's track condition string
        total_runs: horse's total career runs (for context)

        Returns (score 0-100, reason string).
        """
        runs = going_record.get('runs', 0) or 0
        wins = going_record.get('wins', 0) or 0
        places = going_record.get('places', 0) or 0
        avg_pos = going_record.get('avg_pos', 0) or 0

        if runs == 0:
            # No data on this going — check if it's unusual going
            wetness = GoingPreferenceAnalyzer._classify_going(today_going)
            if wetness >= 3:
                # Wet track with no record = unknown, slight risk
                return 40.0, f"Untested on {today_going} — wet track risk"
            elif wetness == 0:
                return 45.0, f"Untested on {today_going}"
            else:
                return 0.0, ""  # GOOD is standard, no penalty for no data

        win_rate = wins / runs if runs > 0 else 0
        place_rate = places / runs if runs > 0 else 0

        # Score based on performance on this going
        if wins >= 2 and win_rate >= 0.30:
            score = 90.0
            reason = f"Loves {today_going} ({wins}W/{places}P from {runs})"
        elif wins >= 1 and place_rate >= 0.50:
            score = 78.0
            reason = f"Good on {today_going} ({wins}W/{places}P from {runs})"
        elif wins >= 1:
            score = 68.0
            reason = f"Won on {today_going} ({wins}W from {runs})"
        elif places >= 2 and place_rate >= 0.40:
            score = 62.0
            reason = f"Places on {today_going} ({places}P from {runs})"
        elif places >= 1:
            score = 55.0
            reason = f"Placed on {today_going} ({places}P from {runs})"
        elif avg_pos > 0 and avg_pos <= 5:
            score = 50.0
            reason = f"Fair on {today_going} (avg {avg_pos:.1f} from {runs})"
        elif runs >= 3 and avg_pos > 7:
            score = 25.0
            reason = f"Struggles on {today_going} (avg {avg_pos:.1f} from {runs})"
        elif runs >= 2 and avg_pos > 6:
            score = 35.0
            reason = f"Below par on {today_going} (avg {avg_pos:.1f} from {runs})"
        else:
            score = 45.0
            reason = f"Limited {today_going} data ({runs} runs, avg {avg_pos:.1f})"

        # Bonus: wet track specialist (going >= YIELDING with strong record)
        wetness = GoingPreferenceAnalyzer._classify_going(today_going)
        if wetness >= 3 and win_rate >= 0.25 and runs >= 2:
            score = min(95.0, score + 10)
            reason += " — WET TRACK SPECIALIST"

        return score, reason
