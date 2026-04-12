#!/usr/bin/env python3
"""
HKJC Advanced Analytics Module
Provides 8 specialized analyzers for deeper race analysis:
  1. DrawBiasAnalyzer - Rail position + draw advantage
  2. PaceAnalyzer - Running style classification + pace scenarios
  3. SectionalAnalyzer - Finishing speed from sectional times
  4. BarrierTrialAnalyzer - Trial form for debut horses
  5. CourseDistanceAnalyzer - Course & distance win record
  6. ClassMovementAnalyzer - Class drop/rise detection
  7. GearChangeAnalyzer - Equipment change impact
  8. WeightAnalyzer - Weight carried vs rating analysis
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_form_positions(last_6: str) -> List[int]:
    """Parse form string into list of finishing positions.
    Handles both '1/3/10/2/5' (slash-separated) and legacy '13425' (digit-by-digit)."""
    if not last_6:
        return []
    s = str(last_6)
    if "/" in s:
        return [int(p) for p in s.split("/") if p.strip().isdigit()]
    else:
        return [int(ch) for ch in s if ch.isdigit()]


# ---------------------------------------------------------------------------
# 1. Draw Bias Analyzer - Rail position aware draw scoring
# ---------------------------------------------------------------------------

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
        cls, draw: int, distance: str, course: str, venue: str
    ) -> Tuple[float, str]:
        """Score draw advantage considering rail position.

        Returns (score 0-100, reason string).
        """
        if draw == 0:
            return 50.0, "No draw data"

        rail = cls._extract_rail(course)
        vk = cls._venue_key(venue)
        db = cls._distance_bucket(distance)

        if not rail or vk not in cls.RAIL_BIAS:
            # Fallback: basic draw scoring without rail info
            return cls._basic_draw_score(draw, distance)

        bias = cls.RAIL_BIAS.get(vk, {}).get(db, {})
        ideal, spread = bias.get(rail, (5, 4))

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


# ---------------------------------------------------------------------------
# 2. Pace Analyzer - Running style + pace scenario
# ---------------------------------------------------------------------------

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

    @staticmethod
    def estimate_style_from_form(
        last_6: str, draw: int = 0, distance: int = 1200
    ) -> str:
        """Heuristic fallback when no running-position history.

        Frequent 1st finishes + low draw + short race → likely Leader.
        Frequent high finishes + long race → likely Closer.
        """
        if not last_6:
            return "Midfield"

        positions = parse_form_positions(last_6)

        if not positions:
            return "Midfield"

        ones = positions.count(1)
        avg = sum(positions) / len(positions)

        # Frontrunner signals
        if ones >= 2 and draw <= 4 and distance <= 1200:
            return "Leader"
        if ones >= 3:
            return "Leader"
        if avg <= 2.5:
            return "Stalker"
        if avg >= 6 and distance >= 1600:
            return "Closer"
        if avg >= 5:
            return "Midfield"
        return "Stalker"

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

        n_leaders = sum(1 for v in scenario.values() if v > 1.1)  # rough check
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


# ---------------------------------------------------------------------------
# 3. Sectional Analyzer - Finishing speed
# ---------------------------------------------------------------------------

class SectionalAnalyzer:
    """Scores horses based on their last-400m sectional times."""

    # Benchmarks (seconds) for last 400m
    EXCEPTIONAL = 22.5
    GOOD = 23.5
    AVERAGE = 24.5

    @classmethod
    def analyze_finishing_speed(
        cls, sectional_history: List[Dict]
    ) -> Tuple[float, str]:
        """Score finishing kick from historical sectional data.

        Each dict should have 'last_400m' (float, seconds).
        Returns (score 0-100, reason string).
        """
        if not sectional_history:
            return 0.0, "No sectional data"

        times = [
            s["last_400m"]
            for s in sectional_history
            if s.get("last_400m") and s["last_400m"] > 0
        ]
        if not times:
            return 0.0, "No sectional data"

        avg_time = sum(times) / len(times)
        best_time = min(times)

        # Score based on average last-400m
        if avg_time <= cls.EXCEPTIONAL:
            score = 95.0
            label = f"Exceptional finisher ({avg_time:.1f}s avg)"
        elif avg_time <= cls.GOOD:
            score = 80.0
            label = f"Strong finisher ({avg_time:.1f}s avg)"
        elif avg_time <= cls.AVERAGE:
            score = 60.0
            label = f"Average finisher ({avg_time:.1f}s avg)"
        else:
            score = 40.0
            label = f"Slow finisher ({avg_time:.1f}s avg)"

        # Bonus for improving trend (last 3 times getting faster)
        if len(times) >= 3:
            recent = times[-3:]
            if recent[-1] < recent[0]:
                score = min(100.0, score + 8)
                label += " [improving]"

        # Bonus for exceptional best
        if best_time <= cls.EXCEPTIONAL and score < 90:
            score = min(100.0, score + 5)

        return score, label

    @staticmethod
    def estimate_from_form(last_6: str, best_time: str) -> Tuple[float, str]:
        """Rough fallback when no sectional data available.

        Uses best_time string and form to estimate finishing ability.
        """
        if not last_6 and not best_time:
            return 0.0, "No sectional data"

        score = 50.0
        reasons = []

        # If horse frequently improves position (low finishes improving)
        if last_6:
            positions = parse_form_positions(last_6)
            if len(positions) >= 3:
                # Check if recent positions are better than earlier ones
                early = sum(positions[:len(positions)//2]) / max(1, len(positions)//2)
                late = sum(positions[len(positions)//2:]) / max(1, len(positions) - len(positions)//2)
                if late < early:
                    score += 15
                    reasons.append("Improving trend")
                elif late > early + 2:
                    score -= 10
                    reasons.append("Declining trend")

        if best_time:
            reasons.append(f"Best: {best_time}")

        return score, " | ".join(reasons) if reasons else "Est. sectional"


# ---------------------------------------------------------------------------
# 4. Barrier Trial Analyzer - Debut horse assessment
# ---------------------------------------------------------------------------

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
                margin_val = float(margin.replace("L", "").replace("HD", "0.2")
                                   .replace("NK", "0.3").replace("SHD", "0.1")
                                   .replace("SH", "0.1").replace("N", "0.3").strip())
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


# ---------------------------------------------------------------------------
# 5. Course & Distance Analyzer
# ---------------------------------------------------------------------------

class CourseDistanceAnalyzer:
    """Scores horses based on their record at the specific course and distance."""

    @staticmethod
    def analyze_cd_form(
        horse_history: List[Dict],
        current_venue: str,
        current_distance: int,
    ) -> Tuple[float, str]:
        """Score C&D form.

        horse_history: list of dicts with keys:
            venue, distance (int), position (int)
        Returns (score 0-100, reason string).
        """
        if not horse_history:
            return 0.0, "No history data"

        # Normalize venue
        venue_lower = current_venue.lower()
        is_st = "sha tin" in venue_lower or "st" in venue_lower
        is_hv = "happy valley" in venue_lower or "hv" in venue_lower

        cd_runs = 0
        cd_wins = 0
        cd_places = 0  # top 3

        for race in horse_history:
            rv = race.get("venue", "").lower()
            rd = race.get("distance", 0)
            rp = race.get("position", 99)

            # Match venue
            if is_st and ("sha tin" not in rv and "st" not in rv):
                continue
            if is_hv and ("happy valley" not in rv and "hv" not in rv):
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
            if rp <= 3:
                cd_places += 1

        if cd_runs == 0:
            return 50.0, "No C&D record"

        if cd_wins >= 3:
            score = 95.0
            label = f"Proven C&D star ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_wins == 2:
            score = 85.0
            label = f"Strong C&D form ({cd_wins}W/{cd_places}P from {cd_runs})"
        elif cd_wins == 1:
            score = 70.0
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

        return score, label


# ---------------------------------------------------------------------------
# Class Movement Analyzer
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Gear Change Analyzer
# ---------------------------------------------------------------------------

class GearChangeAnalyzer:
    """Analyze equipment changes and their expected impact.

    Common HKJC gear codes:
    B  = Blinkers          TT = Tongue-tie
    V  = Visor             P  = Pacifiers
    H  = Hood              CO = Cross-over noseband
    BO = Blinkers off      SR = Shadow roll
    """

    GEAR_IMPACT = {
        "B":  {"score": 72, "label": "Blinkers ON", "note": "improves focus"},
        "B1": {"score": 78, "label": "Blinkers FIRST TIME", "note": "high strike rate"},
        "BO": {"score": 40, "label": "Blinkers OFF", "note": "removed — mixed signal"},
        "TT": {"score": 68, "label": "Tongue-tie ON", "note": "aids breathing"},
        "V":  {"score": 65, "label": "Visor ON", "note": "partial blinkers"},
        "V1": {"score": 70, "label": "Visor FIRST TIME", "note": "new gear trial"},
        "H":  {"score": 60, "label": "Hood ON", "note": "calming effect"},
        "P":  {"score": 58, "label": "Pacifiers ON", "note": "settling gear"},
        "CO": {"score": 55, "label": "Cross-over noseband", "note": "breathing aid"},
        "SR": {"score": 55, "label": "Shadow roll", "note": "visual aid"},
    }

    @staticmethod
    def parse_gear_string(gear: str) -> list:
        if not gear or gear.strip() in ("--", "-", ""):
            return []
        parts = re.split(r"[/,\s+]+", gear.strip().upper())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def analyze_gear(gear: str, is_first_time: bool = False) -> tuple:
        """Score gear impact. Returns (score 0-100, reason string)."""
        codes = GearChangeAnalyzer.parse_gear_string(gear)
        if not codes:
            return 50.0, "No gear changes"

        best_score = 50
        descriptions = []

        for code in codes:
            ft_code = code + "1"
            if is_first_time and ft_code in GearChangeAnalyzer.GEAR_IMPACT:
                info = GearChangeAnalyzer.GEAR_IMPACT[ft_code]
            elif code in GearChangeAnalyzer.GEAR_IMPACT:
                info = GearChangeAnalyzer.GEAR_IMPACT[code]
            else:
                continue

            if info["score"] > best_score:
                best_score = info["score"]
            descriptions.append(f"{info['label']} ({info['note']})")

        if not descriptions:
            return 50.0, f"Gear: {gear} (no significant impact)"

        return float(best_score), "; ".join(descriptions)


# ---------------------------------------------------------------------------
# Weight Analyzer
# ---------------------------------------------------------------------------

class WeightAnalyzer:
    """Analyze weight carried relative to rating and field.

    In HKJC handicap races, weight is assigned based on rating.
    Lower weight relative to field = advantage.
    Apprentice jockey claims (3-10 lbs) reduce effective weight.
    """

    APPRENTICE_CLAIMS = {
        10: "10lb claimer",
        7: "7lb claimer",
        5: "5lb claimer",
        3: "3lb claimer",
    }

    @staticmethod
    def parse_weight(weight_str: str) -> float:
        """Parse weight string like '133' or '115(3)' into effective lbs."""
        if not weight_str:
            return 0.0
        w = weight_str.strip()
        m = re.match(r"(\d+)\s*\((\d+)\)", w)
        if m:
            return float(m.group(1)) - float(m.group(2))
        m = re.match(r"(\d+)", w)
        if m:
            return float(m.group(1))
        return 0.0

    @staticmethod
    def analyze_weight(
        weight_str: str,
        rating: int,
        field_weights: list = None,
    ) -> tuple:
        """Score weight advantage. Returns (score 0-100, reason string)."""
        effective = WeightAnalyzer.parse_weight(weight_str)
        if effective <= 0:
            return 50.0, "Weight data unavailable"

        claim = 0
        m = re.match(r"\d+\s*\((\d+)\)", weight_str.strip())
        if m:
            claim = int(m.group(1))

        reasons = []

        # Weight-to-rating ratio
        if rating > 0:
            ratio = rating / effective
            if ratio >= 0.55:
                score = 75.0
                reasons.append(f"Light weight for rating ({effective:.0f}lbs, rated {rating})")
            elif ratio >= 0.45:
                score = 55.0
                reasons.append(f"Fair weight ({effective:.0f}lbs, rated {rating})")
            else:
                score = 35.0
                reasons.append(f"Heavy weight for rating ({effective:.0f}lbs, rated {rating})")
        else:
            score = 50.0

        # Compare to field
        if field_weights:
            field_effective = [
                WeightAnalyzer.parse_weight(w) for w in field_weights
            ]
            field_effective = [w for w in field_effective if w > 0]
            if field_effective:
                avg_weight = sum(field_effective) / len(field_effective)
                diff = effective - avg_weight
                if diff <= -5:
                    score = min(85.0, score + 15)
                    reasons.append(f"{abs(diff):.0f}lbs below field avg")
                elif diff >= 5:
                    score = max(25.0, score - 10)
                    reasons.append(f"{diff:.0f}lbs above field avg")

        # Apprentice claim bonus
        if claim > 0:
            claim_label = WeightAnalyzer.APPRENTICE_CLAIMS.get(
                claim, f"{claim}lb claim"
            )
            score = min(90.0, score + claim * 1.5)
            reasons.append(f"Apprentice {claim_label}")

        if not reasons:
            return 50.0, f"{effective:.0f}lbs carried"

        return score, "; ".join(reasons)


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
        """Map going string to wetness level 0-4."""
        if not going:
            return 1  # default to GOOD
        g = going.upper().strip()
        for key, level in GoingPreferenceAnalyzer.GOING_WETNESS.items():
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
