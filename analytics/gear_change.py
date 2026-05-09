"""Gear change analyzer — equipment change impact."""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


class GearChangeAnalyzer:
    """Analyze equipment changes and their expected impact.

    Common HKJC gear codes:
    B  = Blinkers          TT = Tongue-tie
    V  = Visor             P  = Pacifiers
    H  = Hood              CO = Cross-over noseband
    BO = Blinkers off      SR = Shadow roll

    Scores below are conservative defaults (50 = neutral, used as cold-start).
    They are overridden at import time by `models/gear_scores.json` if
    present — that file is produced by `scripts/calibrate_gear.py` and
    reflects observed win rates with shrinkage toward neutral so small
    samples don't dominate. Re-run the calibration after each meeting so
    scores adapt as data grows.
    """

    GEAR_IMPACT = {
        "B":  {"score": 50, "label": "Blinkers ON", "note": "improves focus"},
        "B1": {"score": 50, "label": "Blinkers FIRST TIME", "note": "first-time use"},
        "BO": {"score": 50, "label": "Blinkers OFF", "note": "removed"},
        "TT": {"score": 50, "label": "Tongue-tie ON", "note": "aids breathing"},
        "V":  {"score": 50, "label": "Visor ON", "note": "partial blinkers"},
        "V1": {"score": 50, "label": "Visor FIRST TIME", "note": "first-time use"},
        "H":  {"score": 50, "label": "Hood ON", "note": "calming effect"},
        "P":  {"score": 50, "label": "Pacifiers ON", "note": "settling gear"},
        "CP": {"score": 50, "label": "Cross-over noseband", "note": "breathing aid"},
        "CO": {"score": 50, "label": "Cross-over noseband", "note": "breathing aid"},
        "SR": {"score": 50, "label": "Shadow roll", "note": "visual aid"},
        "XB": {"score": 50, "label": "X-blinkers", "note": "modified blinkers"},
    }

    @classmethod
    def _load_calibrated_scores(cls):
        """Override default scores with calibrated values from JSON if present.

        Only the `score` field is overwritten — labels/notes stay descriptive.
        Silently keeps defaults if the file is missing or malformed.
        """
        path = (Path(__file__).resolve().parent.parent
                / "models" / "gear_scores.json")
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"gear_scores.json unreadable: {e}")
            return
        for code, info in data.items():
            if code.startswith("_") or code not in cls.GEAR_IMPACT:
                continue
            if isinstance(info, dict) and "score" in info:
                cls.GEAR_IMPACT[code] = {
                    **cls.GEAR_IMPACT[code], "score": int(info["score"])
                }

    @staticmethod
    def parse_gear_string(gear: str) -> list:
        if not gear or gear.strip() in ("--", "-", ""):
            return []
        parts = re.split(r"[/,\s+]+", gear.strip().upper())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def analyze_gear(gear: str, prior_gear: str = "",
                     is_first_time: bool = False) -> tuple:
        """Score gear impact. Returns (score 0-100, reason string).

        prior_gear: gear codes from the horse's previous run. Codes present
            in `gear` but not in `prior_gear` are treated as first-time-on-
            this-horse (uses the +"1" variant in GEAR_IMPACT when available).
        is_first_time: True for debutants. All current gear is treated as
            first-time-use, since the horse has no prior run to compare to.
        """
        codes = GearChangeAnalyzer.parse_gear_string(gear)
        if not codes:
            return 50.0, "No gear changes"

        prior_codes_list = GearChangeAnalyzer.parse_gear_string(prior_gear)
        prior_codes = set(prior_codes_list)
        # Only treat codes as "new" when we have evidence:
        #   - debutant (no prior runs) — every current code is first-time
        #   - prior_gear was supplied and differs — real change vs last run
        # If prior_gear is empty and not a debutant, we have no diff signal,
        # so use standard (non-first-time) variants.
        if is_first_time:
            new_codes = set(codes)
        elif prior_codes_list:
            new_codes = set(codes) - prior_codes
        else:
            new_codes = set()

        # Track score with biggest deviation from neutral (50), in either
        # direction — so fade scores (e.g. B1=45) come through, not just
        # positive boosts. Without this, taking max() would cap any negative
        # gear signal at 50.
        dominant_score = 50
        descriptions = []

        for code in codes:
            ft_code = code + "1"
            is_new = code in new_codes
            if is_new and ft_code in GearChangeAnalyzer.GEAR_IMPACT:
                info = GearChangeAnalyzer.GEAR_IMPACT[ft_code]
                prefix = "FIRST TIME (debut) — " if is_first_time else "NEW — "
            elif code in GearChangeAnalyzer.GEAR_IMPACT:
                info = GearChangeAnalyzer.GEAR_IMPACT[code]
                prefix = (
                    "FIRST TIME (debut) — " if (is_new and is_first_time)
                    else ("NEW — " if is_new else "")
                )
            else:
                continue

            if abs(info["score"] - 50) > abs(dominant_score - 50):
                dominant_score = info["score"]
            descriptions.append(f"{prefix}{info['label']} ({info['note']})")

        if not descriptions:
            return 50.0, f"Gear: {gear} (no significant impact)"

        return float(dominant_score), "; ".join(descriptions)


# Apply calibrated scores at import time (no-op if file is missing)
GearChangeAnalyzer._load_calibrated_scores()
