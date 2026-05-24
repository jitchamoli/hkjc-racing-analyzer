"""Cheap-diagnostic for two candidate features from theracingapi.com:

  1. jockey_trainer_ae  — lifetime actual/expected ratio for J×T combo
  2. jockey_distance_ae — lifetime actual/expected ratio for J×distance

Both are STRUCTURAL priors (lifetime, slow-changing). Hypothesis: a/e
captures market-priced sharpness orthogonal to win-rate. Combos where
the jockey wins MORE than implied odds suggest are systematically
underbet — actionable for the rule engine if Gate B passes.

Pipeline (same as scripts/diagnose_sire_going.py):
  Stage 1: pull /jockeys/{id}/analysis/{trainers,distances} for top-N HK
           jockeys; cache to tracking_data/api_cache/.
  Stage 2: build per-row features on race_results since 2023-01-01.
  Stage 3: gates
     A1  redundancy with jockey_win_rate    |r| > 0.5  → KILL
     A2  redundancy with jockey-trainer raw win-rate
         (only for jt_ae)                   |r| > 0.7  → KILL
     B   signal vs is_top3                  |r| < 0.02 → KILL
                                            |r| > 0.03 → PROCEED
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import requests
from requests.auth import HTTPBasicAuth

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

CACHE_DIR = REPO_ROOT / 'tracking_data' / 'api_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_JOCKEYS = 30                # covers ~92% of race_results rows
ROW_CUTOFF    = '2023/01/01'      # diagnostic window (avoid ancient regime)

_AUTH = HTTPBasicAuth(config.RACING_API_USERNAME, config.RACING_API_PASSWORD)
_BASE = config.RACING_API_BASE_URL.rstrip('/')


def _api(path: str, params=None) -> dict:
    r = requests.get(f'{_BASE}{path}', auth=_AUTH, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def _cached(name: str, fn):
    p = CACHE_DIR / f'{name}.json'
    if p.exists():
        return json.loads(p.read_text())
    data = fn()
    p.write_text(json.dumps(data))
    return data


def _shorten(full: str) -> str:
    """API 'James McDonald' → HKJC 'J McDonald'. Mirrors
    data.racing_api_hk_news._shorten_jockey."""
    parts = (full or '').strip().split()
    if len(parts) == 2 and len(parts[0]) > 1:
        return f'{parts[0][0]} {parts[1]}'
    return (full or '').strip()


def resolve_jky_id(db_name: str) -> str | None:
    """Find the API jky_id whose shortened name matches db_name AND has
    HK rides. Cache the mapping."""
    cache_key = f'jky_id_{db_name.replace(" ", "_")}'
    def _resolve():
        # Search by last-name (more permissive than full-name)
        last = db_name.split()[-1] if db_name else ''
        results = _api('/v1/jockeys/search', params={'name': last}).get('search_results', [])
        candidates = [r for r in results if _shorten(r.get('name', '')) == db_name]
        if not candidates:
            return None
        # Pick the candidate with the most HK rides by probing course analysis
        best, best_rides = None, 0
        for c in candidates:
            try:
                courses = _api(f"/v1/jockeys/{c['id']}/analysis/courses",
                               params={'region': 'hk'}).get('courses', [])
                rides = sum(int(x.get('rides', 0) or 0) for x in courses)
                if rides > best_rides:
                    best, best_rides = c['id'], rides
            except requests.HTTPError:
                continue
        return best
    return _cached(cache_key, _resolve)


def fetch_jockey_analyses(jky_id: str) -> dict:
    """Returns {'trainers': {trainer_name → ae}, 'distances': {dist_m_int → ae}}."""
    def _f():
        out = {'trainers': {}, 'distances': {}, 'jt_winrate': {}}
        trs = _api(f'/v1/jockeys/{jky_id}/analysis/trainers',
                   params={'region': 'hk'}).get('trainers', [])
        for t in trs:
            name = (t.get('trainer') or '').strip()
            ae = t.get('a/e')
            rides = int(t.get('rides', 0) or 0)
            if name and ae is not None and rides >= 30:   # min-sample gate
                out['trainers'][name] = float(ae)
                # raw win rate for the combo, for Gate A2
                wins = int(t.get('1st', 0) or 0)
                out['jt_winrate'][name] = wins / rides if rides else 0.0
        dists = _api(f'/v1/jockeys/{jky_id}/analysis/distances',
                     params={'region': 'hk'}).get('distances', [])
        for d in dists:
            dm = d.get('dist_m')
            ae = d.get('a/e')
            rides = int(d.get('rides', 0) or 0)
            if dm and ae is not None and rides >= 50:
                out['distances'][int(float(dm))] = float(ae)
        return out
    return _cached(f'jky_analysis_{jky_id}', _f)


def pearson(xs, ys) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else 0.0


def main() -> int:
    conn = sqlite3.connect(str(config.DB_PATH))

    print(f"Step 1: identify top-{TOP_N_JOCKEYS} HK jockeys from race_results...")
    top = conn.execute(f"""
        SELECT jockey, COUNT(*) AS n
        FROM race_results
        WHERE jockey IS NOT NULL AND jockey != ''
        GROUP BY jockey ORDER BY n DESC LIMIT {TOP_N_JOCKEYS}
    """).fetchall()
    db_names = [j for j, _ in top]
    print(f"  Covered ride-rows: {sum(n for _, n in top):,}")

    print("\nStep 2: resolve API jky_id for each (cached) ...")
    name_to_id = {}
    for nm in db_names:
        jid = resolve_jky_id(nm)
        if jid:
            name_to_id[nm] = jid
        print(f"  {nm:<22} {jid or 'NOT FOUND'}")
    print(f"  Resolved {len(name_to_id)}/{len(db_names)}")

    print("\nStep 3: pull /analysis/{trainers,distances} for each (cached) ...")
    jky_data = {nm: fetch_jockey_analyses(jid)
                for nm, jid in name_to_id.items()}
    n_tr_pairs = sum(len(d['trainers']) for d in jky_data.values())
    n_dist_pairs = sum(len(d['distances']) for d in jky_data.values())
    print(f"  J×T combos (n>=30 rides): {n_tr_pairs}")
    print(f"  J×Dist combos (n>=50 rides): {n_dist_pairs}")

    print(f"\nStep 4: build per-row features on race_results since {ROW_CUTOFF}...")
    rows = conn.execute("""
        SELECT jockey, trainer, distance, is_top3
        FROM race_results
        WHERE race_date >= ? AND jockey IS NOT NULL AND trainer IS NOT NULL
          AND distance IS NOT NULL AND is_top3 IS NOT NULL
    """, (ROW_CUTOFF,)).fetchall()

    # Per-jockey raw win-rate over the same window — for Gate A1
    jky_winrate = {}
    for nm in db_names:
        r = conn.execute("""
            SELECT AVG(CAST(is_winner AS FLOAT))
            FROM race_results
            WHERE jockey=? AND race_date >= ? AND is_winner IS NOT NULL
        """, (nm, ROW_CUTOFF)).fetchone()
        jky_winrate[nm] = float(r[0] or 0.0)

    xs_jt_ae, xs_jd_ae = [], []
    xs_jt_wr, xs_jky_wr = [], []
    ys_jt, ys_jd = [], []
    n_jt_hit, n_jd_hit = 0, 0
    for jky, trn, dist_str, top3 in rows:
        if jky not in jky_data:
            continue
        try:
            dist_m = int(float(dist_str))
        except (ValueError, TypeError):
            continue
        d = jky_data[jky]
        jt_ae = d['trainers'].get(trn)
        jd_ae = d['distances'].get(dist_m)
        jky_wr = jky_winrate.get(jky, 0.0)

        if jt_ae is not None:
            xs_jt_ae.append(jt_ae)
            xs_jt_wr.append(d['jt_winrate'].get(trn, 0.0))
            xs_jky_wr.append(jky_wr)
            ys_jt.append(float(top3 or 0))
            n_jt_hit += 1
        if jd_ae is not None:
            xs_jd_ae.append(jd_ae)
            ys_jd.append(float(top3 or 0))
            n_jd_hit += 1

    print(f"  Rows in window: {len(rows):,}")
    print(f"  Rows with jt_ae coverage:   {n_jt_hit:,} "
          f"({100*n_jt_hit/len(rows):.1f}%)")
    print(f"  Rows with jd_ae coverage:   {n_jd_hit:,} "
          f"({100*n_jd_hit/len(rows):.1f}%)")

    if n_jt_hit < 1000 or n_jd_hit < 1000:
        print("INSUFFICIENT COVERAGE — at least one feature has <1000 rows; abort.")
        return 1

    print("\n" + "=" * 70)
    print("CANDIDATE 1: jockey_trainer_ae (lifetime a/e for J×T combo)")
    print("=" * 70)
    r_b_jt   = pearson(xs_jt_ae, ys_jt)
    r_a1_jt  = pearson(xs_jt_ae, xs_jky_wr)
    r_a2_jt  = pearson(xs_jt_ae, xs_jt_wr)
    print(f"  n = {n_jt_hit:,}")
    print(f"  Gate A1 (vs jockey win-rate)  r = {r_a1_jt:+.4f}  "
          f"{'PASS' if abs(r_a1_jt) < 0.5 else 'KILL'}")
    print(f"  Gate A2 (vs J×T raw win-rate) r = {r_a2_jt:+.4f}  "
          f"{'PASS' if abs(r_a2_jt) < 0.7 else 'KILL'}")
    print(f"  Gate B  (vs is_top3)          r = {r_b_jt:+.4f}  "
          f"{'PASS' if abs(r_b_jt) > 0.03 else ('MARGINAL' if abs(r_b_jt) > 0.02 else 'KILL')}")

    print("\n" + "=" * 70)
    print("CANDIDATE 2: jockey_distance_ae (lifetime a/e for J×distance)")
    print("=" * 70)
    r_b_jd  = pearson(xs_jd_ae, ys_jd)
    print(f"  n = {n_jd_hit:,}")
    print(f"  Gate B  (vs is_top3)          r = {r_b_jd:+.4f}  "
          f"{'PASS' if abs(r_b_jd) > 0.03 else ('MARGINAL' if abs(r_b_jd) > 0.02 else 'KILL')}")

    print("\nVERDICT:")
    verdict = []
    for name, rb, ra1, ra2 in [
        ('jt_ae', r_b_jt, r_a1_jt, r_a2_jt),
        ('jd_ae', r_b_jd, None, None),
    ]:
        if abs(rb) < 0.02:
            verdict.append(f"  {name}: KILL (Gate B fail)")
        elif ra1 is not None and abs(ra1) > 0.5:
            verdict.append(f"  {name}: KILL (Gate A1 — redundant with jockey win-rate)")
        elif ra2 is not None and abs(ra2) > 0.7:
            verdict.append(f"  {name}: KILL (Gate A2 — redundant with raw J×T win-rate)")
        elif abs(rb) > 0.03:
            verdict.append(f"  {name}: PROCEED to Stage 2 (re-rank test)")
        else:
            verdict.append(f"  {name}: MARGINAL — re-rank test optional")
    for v in verdict:
        print(v)

    conn.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
