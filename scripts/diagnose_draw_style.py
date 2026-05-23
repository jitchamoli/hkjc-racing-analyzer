"""Cheap-diagnostic for the former-jockey's #5 recommendation:
"Low-draw closer at HV = buried on rail, no gap, finishes 7th flying.
Tool currently marks both draw and closing style as positives — needs
an explicit interaction."

Candidate feature: cell_top3_rate
  For each horse-race, look up the historical top-3 rate of the
  (venue × draw_bucket × style_bucket) cell, leak-free via race_date < ?
  filter, Bayesian-smoothed with Beta(2, 8) prior.

Bucketing (chosen so cell-counts stay meaningful: ~3000 rows/cell average):
  venue:        ST or HV  (2 buckets)
  draw_bucket:  low (1-3), mid (4-7), high (8+)  (3 buckets)
  style_bucket: front (style ≤ 0.33), stalker (0.33-0.66), closer (>0.66)
  → 2 × 3 × 3 = 18 cells

Where style comes from: race_results.running_position parsed via
FeatureEngineering.compute_running_position_features (matches Feature 24
encoding so the diagnostic uses the same definition the ML model would see).

Three kill-gates + one keep-gate (same pattern as the sire diagnostic):
  Gate A1 (redundancy with existing draw_advantage feature):
    |r(cell_t3, draw_advantage_proxy)|  > 0.5  → KILL
  Gate A2 (redundancy with running_style):
    |r(cell_t3, running_style)|         > 0.7  → KILL
  Gate B (signal vs outcome):
    |r(cell_t3, is_top3)|               < 0.02 → KILL
    |r(cell_t3, is_top3)|               > 0.03 → PROCEED

Single-pass O(N) — walks rows in (date, race) order, accumulates per-cell
counters AS WE GO. For each row, the feature is computed from the CURRENT
(prior-state) counters — strictly leak-free without per-row subqueries.

If this passes, expose the same cell-rate lookup as Feature 27 in the
ml/feature_schema.py + ml/features.py. The XGB model can then either use
it directly or learn richer interactions from it.
"""
from __future__ import annotations

import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402
from ml.features import FeatureEngineering  # noqa: E402


def draw_bucket(draw: int, field_size: int) -> str:
    """3 buckets: low (1-3), mid (4-7), high (8+). Returns '' if unknown."""
    if not draw or draw < 1:
        return ''
    if draw <= 3:
        return 'low'
    if draw <= 7:
        return 'mid'
    return 'high'


def style_bucket(running_position: str, field_size: int) -> str:
    """Use same parser as Feature 24 (running_style). Returns 'front',
    'stalker', 'closer', or '' if unparseable."""
    if not running_position or not field_size:
        return ''
    try:
        _early, _late, style = (
            FeatureEngineering.compute_running_position_features(
                running_position, field_size
            )
        )
    except Exception:
        return ''
    if style <= 0.33:
        return 'front'
    if style <= 0.66:
        return 'stalker'
    return 'closer'


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    if vx == 0 or vy == 0:
        return 0.0
    return cov / (vx * vy)


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    print("Step 1: scanning race_results in chronological order...")
    # Need a stable venue identifier — race_results has 'venue' but it's
    # often 'Sha Tin' / 'Happy Valley' (or blank for overseas races).
    # Normalise to ST/HV; skip blanks.
    rows = conn.execute("""
        SELECT race_date, race_number, horse_name,
               draw, venue, running_position,
               is_top3, finish_position,
               (SELECT COUNT(*) FROM race_results rr2
                WHERE rr2.race_date = race_results.race_date
                  AND rr2.race_number = race_results.race_number
                  AND rr2.finish_position > 0) AS field_size
        FROM race_results
        WHERE finish_position > 0 AND draw > 0
          AND venue IS NOT NULL AND venue != ''
          AND running_position != ''
        ORDER BY race_date, race_number
    """).fetchall()
    print(f"  {len(rows):,} rows with draw + venue + running_position\n")

    def normalize_venue(v):
        if not v:
            return ''
        u = v.upper()
        if 'SHA' in u or u == 'ST':
            return 'ST'
        if 'HAPPY' in u or u == 'HV':
            return 'HV'
        return ''

    # Single-pass: accumulate (venue, draw_bucket, style_bucket) → [starts, t3].
    cell_counts = defaultdict(lambda: [0, 0])

    xs_cell, xs_draw_proxy, xs_style_z, ys = [], [], [], []
    skipped = 0
    for r in rows:
        v = normalize_venue(r['venue'])
        db = draw_bucket(r['draw'] or 0, r['field_size'] or 0)
        sb = style_bucket(r['running_position'], r['field_size'] or 12)
        if not (v and db and sb):
            skipped += 1
            continue

        key = (v, db, sb)
        starts, t3 = cell_counts[key]
        # Bayesian smoothing — Beta(2, 8) → base 20% top-3, prior weight 10
        cell_rate = (t3 + 2) / (starts + 10)

        # Comparison features for redundancy gates.
        # draw_advantage_proxy: simple normalized draw position (lower=better)
        # — mirrors what the rule engine roughly does.
        field = r['field_size'] or 12
        draw_proxy = 1.0 - (r['draw'] - 1) / max(field - 1, 1)
        draw_proxy = max(0.0, min(1.0, draw_proxy))

        # running_style as float (0.0 front, 0.5 stalker, 1.0 closer)
        style_val = {'front': 0.0, 'stalker': 0.5, 'closer': 1.0}[sb]

        xs_cell.append(cell_rate)
        xs_draw_proxy.append(draw_proxy)
        xs_style_z.append(style_val)
        ys.append(float(r['is_top3'] or 0))

        # Update counter AFTER recording (leak-free).
        starts += 1
        if r['is_top3']:
            t3 += 1
        cell_counts[key] = [starts, t3]

    print(f"  {len(ys):,} aligned rows  ({skipped:,} skipped — bad bucket)\n")

    print("  Per-cell sample sizes:")
    for v in ('ST', 'HV'):
        for db in ('low', 'mid', 'high'):
            for sb in ('front', 'stalker', 'closer'):
                starts, t3 = cell_counts.get((v, db, sb), [0, 0])
                rate = (t3 + 2) / (starts + 10)
                print(f"    {v}/{db:<5}/{sb:<8} {starts:>6,} starts, "
                      f"{t3:>5,} top3,  smoothed rate {rate:.3f}")
        print()

    if len(ys) < 100:
        print("INSUFFICIENT DATA")
        return 1

    # Correlations
    print("=" * 72)
    print("CORRELATION MATRIX")
    print("=" * 72)
    print(f"  n = {len(ys):,} rows\n")
    r_sig = pearson(xs_cell, ys)
    r_red_draw = pearson(xs_cell, xs_draw_proxy)
    r_red_style = pearson(xs_cell, xs_style_z)
    r_draw_vs_top3 = pearson(xs_draw_proxy, ys)
    r_style_vs_top3 = pearson(xs_style_z, ys)

    print("  Gate A1 (redundancy with draw_advantage_proxy):")
    print(f"     r(cell_t3, draw_proxy)        = {r_red_draw:+.4f}")
    print("  Gate A2 (redundancy with running_style):")
    print(f"     r(cell_t3, running_style_val) = {r_red_style:+.4f}")
    print()
    print("  Gate B (signal vs is_top3):")
    print(f"     r(cell_t3,    is_top3)        = {r_sig:+.4f}")
    print()
    print("  Baselines for scale:")
    print(f"     r(draw_proxy,     is_top3)    = {r_draw_vs_top3:+.4f}")
    print(f"     r(running_style,  is_top3)    = {r_style_vs_top3:+.4f}")
    print()

    # Verdict
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    kills = []
    if abs(r_red_draw) > 0.5:
        kills.append(f"KILL — Gate A1: redundant with draw_advantage "
                     f"|r|={abs(r_red_draw):.2f} > 0.5")
    else:
        print(f"  Gate A1 PASS: orthogonal to draw_proxy "
              f"(|r|={abs(r_red_draw):.4f} < 0.5)")
    if abs(r_red_style) > 0.7:
        kills.append(f"KILL — Gate A2: redundant with running_style "
                     f"|r|={abs(r_red_style):.2f} > 0.7")
    else:
        print(f"  Gate A2 PASS: orthogonal to running_style "
              f"(|r|={abs(r_red_style):.4f} < 0.7)")
    if abs(r_sig) < 0.02:
        kills.append(f"KILL — Gate B: no signal |r|={abs(r_sig):.4f} < 0.02")
    elif abs(r_sig) > 0.03:
        print(f"  Gate B PASS: cell_t3 carries place-outcome signal "
              f"(|r|={abs(r_sig):.4f} > 0.03)")
    else:
        print(f"  Gate B MARGINAL: |r|={abs(r_sig):.4f}")
    print()
    if kills:
        for k in kills:
            print(f"  {k}")
        print()
        print("  RECOMMENDATION: do NOT add as a feature; current draw/style")
        print("  encoding already captures the (venue × draw × style) signal.")
        conn.close()
        return 0
    if abs(r_sig) > 0.03:
        baseline = max(abs(r_draw_vs_top3), abs(r_style_vs_top3))
        ratio = abs(r_sig) / baseline if baseline > 0 else float('inf')
        print(f"  RECOMMENDATION: PROCEED to retrain test.")
        print(f"  Signal strength is {ratio:.0%} of the stronger of "
              f"(draw_proxy, running_style)'s own correlation with is_top3 —")
        print(f"  a meaningful add that captures an interaction the model")
        print(f"  doesn't currently see as a single channel.")
        print()
        print(f"  Next step: re-rank test on burned 6-meeting holdout via")
        print(f"  a sibling scripts/test_draw_style_rerank.py before retrain.")
    else:
        print(f"  RECOMMENDATION: MARGINAL. The interaction signal exists but")
        print(f"  is weak. Consider re-rank test before retraining.")
    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
