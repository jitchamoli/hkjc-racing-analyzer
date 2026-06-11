"""Barrier trial analyzer — debut horse assessment."""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class BarrierTrialAnalyzer:
    """Assesses debut/Griffin horses via barrier trial results."""

    @staticmethod
    def is_debut_horse(
        last_6: str, rating: float, class_type: str
    ) -> bool:
        """Check if horse has no race form (debut/Griffin)."""
        no_form = not last_6 or last_6.strip() == "" or last_6.strip() == "0"
        is_griffin = "griffin" in class_type.lower() if class_type else False
        no_rating = rating <= 0
        return (no_form and no_rating) or is_griffin

    @classmethod
    def analyze_trials(
        cls, trial_results: List[Dict]
    ) -> Tuple[float, str]:
        """Score debut horse from trial performances.

        Each dict: {position: int, margin: str, time: str, date: str}.
        Returns (score 0-100, reason string).
        """
        if not trial_results:
            return 0.0, "No trial data"

        # Use best trial result
        best = min(trial_results, key=lambda t: t.get("position", 99))
        pos = best.get("position", 99)
        margin = best.get("margin", "")

        if pos == 1:
            score = 85.0
            label = "Trial WINNER"
        elif pos == 2:
            score = 72.0
            label = "Trial 2nd"
        elif pos == 3:
            score = 62.0
            label = "Trial 3rd"
        elif pos <= 5:
            score = 50.0
            label = f"Trial {pos}th"
        else:
            score = 35.0
            label = f"Trial {pos}th (poor)"

        # Adjust for margin if available
        if margin:
            try:
                # Replace multi-char tokens before their substrings: "SHD"
                # before "HD"/"SH", "NK" before "N" — otherwise "SHD" → "S0.2"
                # and float() raises (the close-up bonus was silently skipped).
                margin_val = float(margin.replace("SHD", "0.1").replace("NK", "0.3")
                                   .replace("HD", "0.2").replace("SH", "0.1")
                                   .replace("N", "0.3").replace("L", "").strip())
                if pos > 1 and margin_val <= 1.0:
                    score += 8
                    label += f" (close: {margin})"
                elif pos > 1 and margin_val >= 5.0:
                    score -= 5
                    label += f" (beaten {margin})"
            except (ValueError, AttributeError):
                pass

        # Multiple trials give more confidence
        if len(trial_results) >= 2:
            # Average of top 2
            sorted_trials = sorted(trial_results, key=lambda t: t.get("position", 99))
            avg_pos = sum(t.get("position", 99) for t in sorted_trials[:2]) / 2
            if avg_pos <= 2:
                score = min(95.0, score + 10)
                label += " [consistent]"

        return min(95.0, score), label

    @classmethod
    def supplement_form_score(
        cls, form_score: float, trial_results: List[Dict],
        last_6: str, rating: float, class_type: str
    ) -> Tuple[float, str]:
        """For debut horses, replace zero form score with trial-based score.

        Returns (adjusted_form_score, trial_reason).
        """
        if not cls.is_debut_horse(last_6, rating, class_type):
            return form_score, ""

        trial_score, trial_reason = cls.analyze_trials(trial_results)
        if trial_score > 0:
            return trial_score, f"[Trial] {trial_reason}"
        return form_score, ""
