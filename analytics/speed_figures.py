#!/usr/bin/env python3
"""Adjusted speed figures — Beyer/Timeform-style par time normalization.

Computes par times per (distance, going, class) from historical data,
then measures each horse's performance relative to par. Adjusts for
carried weight using Benter's HK approximation (+0.1s per lb above 120).
"""

import logging
import sqlite3
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Benter's weight adjustment for HK racing: +0.1s per lb above 120
WEIGHT_BASELINE = 120
WEIGHT_ADJUSTMENT_PER_LB = 0.1


def parse_finish_time(ft: str) -> Optional[float]:
    """Parse finish time to seconds. Handles both 'M:SS.ss' (HKJC live tables)
    and 'M.SS.ss' (HKJC LocalResults dump — used in 85% of race_results rows).
    Returns None on failure.
    """
    s = str(ft).strip()
    if not s:
        return None
    try:
        if ':' in s:
            # M:SS.ss format
            parts = s.split(':')
            secs = float(parts[0]) * 60 + float(parts[1])
        else:
            # M.SS.ss format — three dot-separated tokens
            parts = s.split('.')
            if len(parts) >= 3:
                secs = float(parts[0]) * 60 + float(parts[1] + '.' + parts[2])
            else:
                # Single dotted number — already in seconds
                secs = float(s)
        return secs if 0 < secs < 999 else None
    except (ValueError, IndexError):
        return None


def compute_par_times(conn: sqlite3.Connection, before_date: str = None) -> Dict[Tuple, float]:
    """Compute median finish times per (distance, going, class) bucket.

    Falls back to (distance, going) if class bucket has <5 races,
    then (distance) only. Returns dict keyed by (distance, going, class_type).

    Args:
        conn: SQLite connection to horse_history.db
        before_date: Only use races before this date (prevents leakage in training)
    """
    query = """
        SELECT distance, going, class_type, finish_time
        FROM race_results
        WHERE finish_time LIKE '%:%' AND distance > 0
    """
    params = []
    if before_date:
        query += " AND race_date < ?"
        params.append(before_date)

    rows = conn.execute(query, params).fetchall()

    # Group times by (distance, going, class)
    buckets_full = {}    # (dist, going, class) → [times]
    buckets_dg = {}      # (dist, going) → [times]
    buckets_d = {}       # (dist,) → [times]

    for dist, going, cls, ft in rows:
        secs = parse_finish_time(ft)
        if secs is None:
            continue
        going = going or ''
        cls = cls or ''

        buckets_full.setdefault((dist, going, cls), []).append(secs)
        buckets_dg.setdefault((dist, going), []).append(secs)
        buckets_d.setdefault((dist,), []).append(secs)

    # Compute medians with fallback
    par_times = {}
    all_keys = set(buckets_full.keys()) | {(d, g, '') for d, g in buckets_dg} | {(d, '', '') for (d,) in buckets_d}

    for key in buckets_full:
        dist, going, cls = key
        times = buckets_full[key]
        if len(times) >= 5:
            par_times[key] = sorted(times)[len(times) // 2]
        elif len(buckets_dg.get((dist, going), [])) >= 5:
            dg_times = buckets_dg[(dist, going)]
            par_times[key] = sorted(dg_times)[len(dg_times) // 2]
        elif len(buckets_d.get((dist,), [])) >= 5:
            d_times = buckets_d[(dist,)]
            par_times[key] = sorted(d_times)[len(d_times) // 2]

    # Also store fallback lookups
    for dg_key, times in buckets_dg.items():
        dist, going = dg_key
        full_key = (dist, going, '')
        if full_key not in par_times and len(times) >= 5:
            par_times[full_key] = sorted(times)[len(times) // 2]

    for d_key, times in buckets_d.items():
        dist = d_key[0]
        full_key = (dist, '', '')
        if full_key not in par_times and len(times) >= 3:
            par_times[full_key] = sorted(times)[len(times) // 2]

    logger.info(f"Computed {len(par_times)} par time buckets from {len(rows)} results")
    return par_times


def get_par_time(par_times: Dict, distance: int, going: str, class_type: str) -> Optional[float]:
    """Look up par time with fallback: (dist,going,class) → (dist,going,'') → (dist,'','')."""
    going = going or ''
    class_type = class_type or ''

    # Exact match
    pt = par_times.get((distance, going, class_type))
    if pt:
        return pt
    # Fallback: distance + going
    pt = par_times.get((distance, going, ''))
    if pt:
        return pt
    # Fallback: distance only
    return par_times.get((distance, '', ''))


def adjusted_speed_figure(finish_time_str: str, distance: int, going: str,
                          class_type: str, weight: int,
                          par_times: Dict) -> Optional[float]:
    """Compute adjusted speed figure for a single performance.

    Returns value where 0 = par, positive = faster than par, negative = slower.
    Adjusts for carried weight using Benter's approximation.

    Args:
        finish_time_str: 'M:SS.ss' format
        distance: race distance in meters
        going: track condition
        class_type: race class
        weight: carried weight in lbs
        par_times: precomputed par time dict

    Returns:
        Adjusted speed figure (roughly -5 to +5 range), or None if can't compute
    """
    actual_secs = parse_finish_time(finish_time_str)
    if actual_secs is None:
        return None

    par = get_par_time(par_times, distance, going, class_type)
    if par is None:
        return None

    # Weight adjustment: credit heavier horses (they ran that time with more weight)
    # Symmetric around baseline — lighter horses are debited, heavier credited
    try:
        weight_lbs = int(weight) if weight else 120
    except (ValueError, TypeError):
        weight_lbs = 120
    weight_adj = WEIGHT_ADJUSTMENT_PER_LB * (weight_lbs - WEIGHT_BASELINE)
    weight_adjusted_time = actual_secs - weight_adj

    # Speed figure: % faster/slower than par
    figure = (par - weight_adjusted_time) / par * 100

    return figure


def normalize_speed_figure(figure: float) -> float:
    """Normalize adjusted speed figure to 0-1 range for ML feature.

    Maps roughly: -5 → 0.0, 0 → 0.5, +5 → 1.0
    """
    return max(0.0, min(1.0, (figure + 5) / 10))


def normalize_consistency(std: float) -> float:
    """Normalize speed consistency (std dev) to 0-1 range.

    Lower std = more consistent = higher value.
    Maps: std=0 → 1.0, std=3 → 0.0
    """
    return max(0.0, min(1.0, 1.0 - std / 3.0))
