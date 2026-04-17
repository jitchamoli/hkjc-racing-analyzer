"""Rolling form analyzer — rolling performance stats over recent races."""

import logging

logger = logging.getLogger(__name__)


class RollingFormAnalyzer:
    """Analyze rolling performance stats over recent races.

    Rolling averages are among the highest-impact ML features — they capture
    whether a horse is trending up, stable, or declining. CodeWorks Paris
    found these among their top features out of 1,618 engineered.
    """

    @staticmethod
    def analyze_rolling_form(rolling_stats: dict) -> tuple:
        """Score based on rolling stats. Returns (score 0-100, reason)."""
        runs = rolling_stats.get('runs', 0)
        if runs == 0:
            return 0.0, ""

        avg_pos = rolling_stats.get('avg_pos', 0)
        win_rate = rolling_stats.get('win_rate', 0)
        place_rate = rolling_stats.get('place_rate', 0)
        trend = rolling_stats.get('trend', 'unknown')

        # Base score from average position
        if avg_pos <= 2.0:
            score = 90.0
        elif avg_pos <= 3.0:
            score = 78.0
        elif avg_pos <= 4.5:
            score = 65.0
        elif avg_pos <= 6.0:
            score = 50.0
        elif avg_pos <= 8.0:
            score = 35.0
        else:
            score = 20.0

        # Win/place rate bonus
        if win_rate >= 0.30:
            score = min(95, score + 10)
        elif place_rate >= 0.50:
            score = min(90, score + 5)

        # Trend adjustment
        if trend == 'improving':
            score = min(95, score + 8)
            trend_label = "IMPROVING"
        elif trend == 'declining':
            score = max(15, score - 8)
            trend_label = "declining"
        else:
            trend_label = "steady"

        reason = f"Rolling {runs}R: avg {avg_pos}, {win_rate:.0%}W, {place_rate:.0%}P ({trend_label})"
        return score, reason
