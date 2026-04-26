"""Per-tier hit-rate + Wilson 95% CI, computed from prediction_log × race_results.

Sliding-window over the last N completed races (default 73 — matches the
backtest holdout size). Cached to tracking_data/tier_stats.json; recomputed
on every results ingestion (see dashboard/results_fetcher.py integration).
"""

import json
import os
import sqlite3
from typing import Dict, Tuple

from statsmodels.stats.proportion import proportion_confint

CACHE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tracking_data', 'tier_stats.json',
)


def wilson_ci_95(hits: int, total: int) -> Tuple[float, float]:
    """95% Wilson confidence interval on a binomial proportion."""
    if total == 0:
        return (0.0, 1.0)
    low, high = proportion_confint(hits, total, alpha=0.05, method='wilson')
    return (float(low), float(high))


def compute_tier_stats(conn: sqlite3.Connection, window: int = 73) -> Dict:
    """Hit-rate + Wilson CI per conviction_tier over the last `window` completed races.

    Only rank-1 predictions counted. Hit = horse finished first (is_winner = 1).
    Predictions without conviction_tier set are skipped.
    """
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT r.race_date, r.race_number
        FROM race_results r
        WHERE r.finish_position > 0
        ORDER BY r.race_date DESC, r.race_number DESC
        LIMIT ?
    """, (window,))
    window_races = c.fetchall()
    if not window_races:
        return {}

    placeholders = ','.join(['(?, ?)'] * len(window_races))
    flat = [x for pair in window_races for x in pair]

    q = f"""
        SELECT p.conviction_tier,
               SUM(CASE WHEN r.is_winner = 1 THEN 1 ELSE 0 END) AS hits,
               COUNT(*) AS total
        FROM prediction_log p
        JOIN race_results r
          ON p.race_date = r.race_date
         AND p.race_number = r.race_number
         AND p.horse_name = r.horse_name
        WHERE p.predicted_rank = 1
          AND p.conviction_tier IS NOT NULL
          AND (p.race_date, p.race_number) IN ({placeholders})
        GROUP BY p.conviction_tier
    """
    c.execute(q, flat)

    stats = {}
    for tier, hits, total in c.fetchall():
        rate = hits / total if total > 0 else 0.0
        ci_low, ci_high = wilson_ci_95(hits, total)
        stats[tier] = {
            'hit': int(hits),
            'total': int(total),
            'rate': round(rate, 4),
            'ci_low': round(ci_low, 4),
            'ci_high': round(ci_high, 4),
        }
    return stats


def load_cached_stats() -> Dict:
    """Read the cache, return {} if missing/corrupt."""
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_cached_stats(stats: Dict) -> None:
    """Overwrite tracking_data/tier_stats.json."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(stats, f, indent=2)


def refresh_cache(conn: sqlite3.Connection, window: int = 73) -> Dict:
    """Recompute from DB and overwrite cache."""
    stats = compute_tier_stats(conn, window=window)
    write_cached_stats(stats)
    return stats
