"""Pace analyzer — running style classification and pace scenario impact."""

import logging
from typing import Dict, List, Tuple

from analytics import parse_form_positions

logger = logging.getLogger(__name__)


class PaceAnalyzer:
    """Classifies running style and models pace scenario impact."""

    STYLES = ("Leader", "Stalker", "Midfield", "Closer")

    @staticmethod
    def classify_running_style(
        running_positions: List[List[int]],
    ) -> str:
        """Classify from historical running-position arrays.

        Each inner list is [1st_call, 2nd_call, ..., finish].
        We look at the 1st-call position across races.
        """
        if not running_positions:
            return "Midfield"

        first_calls = [rp[0] for rp in running_positions if rp]
        if not first_calls:
            return "Midfield"

        avg = sum(first_calls) / len(first_calls)
        if avg <= 2:
            return "Leader"
        elif avg <= 5:
            return "Stalker"
        elif avg <= 9:
            return "Midfield"
        return "Closer"

    @classmethod
    def classify_style_from_db(
        cls, horse_name: str, conn=None, before_date=None,
    ) -> str:
        """Derive a style label from the horse's most recent running_position
        in the DB (99.4% coverage) via the ml.features helper. Returns ''
        when no prior data — caller should then fall back to the last_6 +
        draw heuristic in estimate_style_from_form."""
        if not horse_name:
            return ""
        from ml.features import FeatureEngineering
        early, late, style_f = FeatureEngineering.calculate_running_position_features(
            horse_name, conn=conn, before_date=before_date,
        )
        if early == 0.5 and late == 0.5 and style_f == 0.5:
            return ""
        if style_f <= 0.2:
            return "Leader"
        if style_f <= 0.45:
            return "Stalker"
        if style_f >= 0.7:
            return "Closer"
        return "Midfield"

    @staticmethod
    def estimate_style_from_form(
        last_6: str, draw: int = 0, distance: int = 1200
    ) -> str:
        """Heuristic fallback when no running-position history.

        NOTE: last_6 contains FINISH positions, not early running positions.
        We use avg finish position + draw + distance as weak proxies.
        Low draw + short distance → likely Leader.
        Consistently back in the field + long distance → likely Closer.
        """
        if not last_6:
            return "Midfield"

        positions = parse_form_positions(last_6)

        if not positions:
            return "Midfield"

        avg = sum(positions) / len(positions)

        # Draw + distance heuristics (finish positions are weak style signals)
        if draw > 0 and draw <= 3 and distance <= 1200:
            return "Leader"
        if draw > 0 and draw <= 4 and avg <= 3.0 and distance <= 1400:
            return "Stalker"
        if avg >= 8 and distance >= 1600:
            return "Closer"
        if avg >= 6:
            return "Midfield"
        if avg <= 3.0:
            return "Stalker"
        return "Midfield"

    @staticmethod
    def build_pace_scenario(
        styles: List[str], distance: int
    ) -> Dict[str, float]:
        """Compute advantage multipliers for each style given field composition.

        Returns dict like {"Leader": 1.15, "Stalker": 1.0, ...}.
        """
        n_leaders = styles.count("Leader")
        n_total = len(styles) or 1

        # Base multipliers (neutral)
        scenario: Dict[str, float] = {
            "Leader": 1.0,
            "Stalker": 1.0,
            "Midfield": 1.0,
            "Closer": 1.0,
        }

        # Distance bias
        if distance <= 1200:
            scenario["Leader"] += 0.12
            scenario["Stalker"] += 0.05
            scenario["Closer"] -= 0.08
        elif distance >= 1800:
            scenario["Closer"] += 0.12
            scenario["Stalker"] += 0.05
            scenario["Leader"] -= 0.05

        # Pace pressure
        if n_leaders == 0:
            # No designated leader — any frontrunner gets an easy lead
            scenario["Leader"] += 0.15
            scenario["Stalker"] += 0.05
        elif n_leaders == 1:
            # Lone leader — ideal scenario for that horse
            scenario["Leader"] += 0.10
        elif n_leaders == 2:
            # Contested lead — pressure but manageable
            scenario["Leader"] -= 0.05
            scenario["Closer"] += 0.05
        else:
            # Hot pace (3+ leaders) — closers thrive
            scenario["Leader"] -= 0.15
            scenario["Stalker"] += 0.05
            scenario["Closer"] += 0.15
            scenario["Midfield"] += 0.08

        return scenario

    @classmethod
    def analyze_pace(
        cls, style: str, scenario: Dict[str, float]
    ) -> Tuple[float, str]:
        """Score a horse's pace advantage given the field scenario.

        Returns (score 0-100, reason string).
        """
        if style not in cls.STYLES:
            style = "Midfield"

        multiplier = scenario.get(style, 1.0)

        # Convert multiplier (typically 0.8-1.3) to 0-100 score
        base = 50.0
        score = base + (multiplier - 1.0) * 200  # ±0.25 → ±50 points
        score = max(15.0, min(95.0, score))

        if multiplier >= 1.15:
            pace_label = "Strong pace advantage"
        elif multiplier >= 1.05:
            pace_label = "Moderate pace edge"
        elif multiplier >= 0.95:
            pace_label = "Neutral pace"
        else:
            pace_label = "Pace against"

        reason = f"{style} - {pace_label} (x{multiplier:.2f})"
        return score, reason

    @classmethod
    def generate_speed_map(
        cls, horses: List[Dict], distance: int, venue: str = "ST"
    ) -> List[Dict]:
        """Generate predicted running positions for all horses in a race.

        Each horse dict needs: 'number', 'name', 'draw', 'running_style' (or 'last_6').
        Returns list of dicts sorted by predicted early position:
        [{'number': 1, 'name': 'HORSE', 'style': 'Leader', 'predicted_pos': 1, 'lane': 'inside'}, ...]
        """
        entries = []
        for h in horses:
            style = h.get('running_style', '')
            if not style or style == 'Midfield':
                style = cls.estimate_style_from_form(
                    h.get('last_6', ''),
                    h.get('draw', 0),
                    distance,
                )

            draw = h.get('draw', 0)
            field_size = len(horses)

            style_base = {
                'Leader': 1.5,
                'Stalker': 4.0,
                'Midfield': 7.0,
                'Closer': 10.0,
            }.get(style, 7.0)

            draw_adj = (draw - field_size / 2) * 0.15 if draw > 0 else 0
            predicted = max(1.0, min(float(field_size), style_base + draw_adj))

            if draw <= field_size * 0.33:
                lane = 'inside'
            elif draw <= field_size * 0.66:
                lane = 'middle'
            else:
                lane = 'outside'

            entries.append({
                'number': h.get('number', 0),
                'name': h.get('name', ''),
                'style': style,
                'predicted_pos': round(predicted, 1),
                'lane': lane,
                'draw': draw,
            })

        entries.sort(key=lambda x: x['predicted_pos'])
        for i, e in enumerate(entries):
            e['predicted_pos'] = i + 1

        return entries
