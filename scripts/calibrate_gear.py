"""Calibrate per-gear scores from observed race_results.

Replaces hand-tuned GEAR_IMPACT values with data-driven ones. Re-run after
each meeting (e.g. as part of auto-learn) so scores adapt gradually.

Method
------
For each gear code, compute observed win rate vs field baseline:
  raw_lift = win_rate(code) - win_rate(field)

Apply shrinkage toward neutral so small samples don't dominate:
  effective_lift = raw_lift * N / (N + PRIOR_STRENGTH)

Map to score (50 = neutral):
  score = clamp(50 + effective_lift * 100 * LIFT_TO_SCORE, 20, 80)

This way:
- Cold-start (N=0) → score stays at default
- Small N → pulled toward neutral
- Large N → reflects observed data
- Always bounded so one outlier can't break the model

Output: models/gear_scores.json
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Tunable priors. Larger PRIOR_STRENGTH = more shrinkage toward neutral.
# 200 means a code needs ~200 observations before its raw lift is fully
# trusted; below that, the score is partially pulled toward 50.
PRIOR_STRENGTH = 200
LIFT_TO_SCORE = 2.0   # 1pp lift → 2 score points
SCORE_FLOOR = 20
SCORE_CEIL = 80

# Codes we calibrate. Standard codes look at any presence of the code in
# the gear string. First-time variants (B1, V1) look only at debutants.
STANDARD_CODES = ["B", "TT", "V", "H", "P", "CP", "XB", "BO", "CO", "SR"]
FIRST_TIME_CODES = ["B1", "V1"]


def _shrink_lift(raw_lift: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return raw_lift * n / (n + PRIOR_STRENGTH)


def _lift_to_score(effective_lift: float) -> int:
    raw = 50 + effective_lift * 100 * LIFT_TO_SCORE
    return int(round(max(SCORE_FLOOR, min(SCORE_CEIL, raw))))


def calibrate(db_path: str = "tracking_data/horse_history.db") -> dict:
    """Compute calibrated scores. Returns dict mapping code → score."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Field baseline win rate
    c.execute("SELECT COUNT(*), SUM(is_winner) FROM race_results")
    n_all, w_all = c.fetchone()
    if not n_all:
        logger.warning("No race_results data — returning empty calibration")
        return {}
    baseline = w_all / n_all
    logger.info(f"Field baseline: {baseline*100:.2f}% ({w_all:,}/{n_all:,})")

    out = {"_meta": {
        "baseline_win_rate": round(baseline, 6),
        "n_total_rows": n_all,
        "prior_strength": PRIOR_STRENGTH,
        "lift_to_score": LIFT_TO_SCORE,
    }}

    # Standard codes — any presence in gear string
    for code in STANDARD_CODES:
        # Use word-boundary matching to avoid 'B' matching 'BO', 'B-' etc.
        # Simple approach: split by /,space and check membership
        c.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(is_winner), 0)
            FROM race_results
            WHERE gear IS NOT NULL AND gear != '' AND gear != '-'
              AND ('/' || REPLACE(REPLACE(UPPER(gear), '-', ''), ' ', '') || '/')
                  LIKE '%/' || ? || '/%'
            """,
            (code,),
        )
        n, w = c.fetchone()
        if n == 0:
            continue
        win_rate = w / n
        raw_lift = win_rate - baseline
        eff_lift = _shrink_lift(raw_lift, n)
        score = _lift_to_score(eff_lift)
        out[code] = {
            "score": score,
            "n": n,
            "win_rate": round(win_rate, 4),
            "lift_pp": round(raw_lift * 100, 2),
            "effective_lift_pp": round(eff_lift * 100, 2),
        }
        logger.info(
            f"  {code:4s} N={n:5,} win={win_rate*100:5.2f}% "
            f"lift={raw_lift*100:+5.2f}pp → score={score}"
        )

    # First-time variants — debutant rows only
    for ft_code in FIRST_TIME_CODES:
        base_code = ft_code[:-1]  # 'B1' → 'B'
        c.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(is_winner), 0)
            FROM race_results r
            WHERE r.gear IS NOT NULL AND r.gear != '' AND r.gear != '-'
              AND ('/' || REPLACE(REPLACE(UPPER(r.gear), '-', ''), ' ', '') || '/')
                  LIKE '%/' || ? || '/%'
              AND NOT EXISTS (
                SELECT 1 FROM race_results r2
                WHERE r2.horse_name = r.horse_name
                  AND (r2.race_date < r.race_date
                       OR (r2.race_date = r.race_date
                           AND r2.race_number < r.race_number))
              )
            """,
            (base_code,),
        )
        n, w = c.fetchone()
        if n == 0:
            continue
        win_rate = w / n
        raw_lift = win_rate - baseline
        eff_lift = _shrink_lift(raw_lift, n)
        score = _lift_to_score(eff_lift)
        out[ft_code] = {
            "score": score,
            "n": n,
            "win_rate": round(win_rate, 4),
            "lift_pp": round(raw_lift * 100, 2),
            "effective_lift_pp": round(eff_lift * 100, 2),
        }
        logger.info(
            f"  {ft_code:4s} N={n:5,} win={win_rate*100:5.2f}% "
            f"lift={raw_lift*100:+5.2f}pp → score={score}"
        )

    conn.close()
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_path = Path("models/gear_scores.json")
    out_path.parent.mkdir(exist_ok=True)
    scores = calibrate()
    out_path.write_text(json.dumps(scores, indent=2))
    logger.info(f"Wrote {out_path} ({len(scores)-1} codes calibrated)")


if __name__ == "__main__":
    main()
