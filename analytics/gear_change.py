"""Gear change analyzer — equipment change impact."""

import re
import logging

logger = logging.getLogger(__name__)


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
