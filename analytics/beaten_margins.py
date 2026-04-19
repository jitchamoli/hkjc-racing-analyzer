"""Beaten margin analyzer — key professional handicapping signal."""

import logging

logger = logging.getLogger(__name__)


class BeatenMarginAnalyzer:
    """Analyze beaten margins — key professional handicapping signal.

    Horses finishing within 2.75 lengths of the winner are strong
    next-out contenders. This is one of the most reliable predictors
    in horse racing ML models.
    """

    @staticmethod
    def analyze_margins(margin_stats: dict) -> tuple:
        """Score based on beaten margins. Returns (score 0-100, reason)."""
        runs = margin_stats.get('runs', 0)
        if runs == 0:
            return 0.0, ""

        close = margin_stats.get('close_finishes', 0)
        avg_margin = margin_stats.get('avg_margin', 0)
        last_margin = margin_stats.get('last_margin', 0)

        # Last-start margin is the strongest signal
        # Use last_winner flag to distinguish winner (margin=0) from missing data
        last_winner = margin_stats.get('last_winner', False)
        if last_margin == 0 and last_winner:
            score = 88.0
            reason = "Won last start"
        elif last_margin == 0 and not last_winner:
            # Margin=0 but not flagged as winner — data may be missing
            score = 50.0
            reason = ""
        elif last_margin <= 1.0:
            score = 82.0
            reason = f"Narrowly beaten last ({last_margin}L)"
        elif last_margin <= 2.75:
            score = 72.0
            reason = f"Close up last ({last_margin}L) — strong next-out signal"
        elif last_margin <= 5.0:
            score = 55.0
            reason = f"Fair effort last ({last_margin}L)"
        elif last_margin <= 8.0:
            score = 38.0
            reason = f"Well beaten last ({last_margin}L)"
        else:
            score = 22.0
            reason = f"Tailed off last ({last_margin}L)"

        # Bonus for consistently close finishes
        if runs >= 3 and close >= 2:
            score = min(92, score + 5)
            reason += f" — {close}/{runs} close finishes"

        # Average margin context
        if avg_margin > 0 and avg_margin <= 2.0:
            score = min(95, score + 3)

        return score, reason
