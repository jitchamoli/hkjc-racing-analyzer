"""Generate shadow predictions for a race date using rule_w=0.9 (vs production 0.6).

Run AFTER `generate_predictions.py` so production output is locked in. This
script re-fetches the same race cards from HKJC, runs the analyzer with
patched RULE_WEIGHT=0.9, and writes a slim shadow JSON capturing only the
bet selections (no HTML, no DB logging, no dashboard side-effects).

Usage:
    python3 -m scripts.shadow_generate '2026/05/17'

Output: results/shadow_race_data_YYYYMMDD.json with structure:
    {
      "rule_w": 0.9,
      "ml_w": 0.1,
      "generated_at": "...",
      "races": [
        {"race_number": N, "bet_type": "QIN"/"BOX Q"/"BOX Q4"/"TRIO"/"WIN"/"SKIP",
         "legs": [{"rank": R, "horse_number": H, "horse_name": "...",
                   "bet_recommendation": "QIN"/...}, ...]},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


async def run(race_date: str, rule_w: float = 0.9) -> int:
    # Patch config BEFORE importing analyzer modules so any module-level reads
    # use the new weights. Most reads are inside functions (call-time) so this
    # is belt-and-braces.
    config.RULE_WEIGHT = rule_w
    config.ML_WEIGHT = 1.0 - rule_w

    from core.fetcher import HKJCDataFetcher
    from core.analyzer import CombinedRaceAnalyzer

    print(f"Shadow generator — rule_w={rule_w}, ml_w={1.0-rule_w} on {race_date}", flush=True)

    fetcher = HKJCDataFetcher()
    analyzer = CombinedRaceAnalyzer()
    # Close the Playwright browser before returning so it doesn't stack with
    # the production run that immediately precedes/follows (audit fix 2026-05-17).
    try:
        # Fetch all race cards concurrently — same pattern as dashboard/generator.py
        fetched = await asyncio.gather(
            *(fetcher.get_race_card(race_date, rn) for rn in range(1, 12)),
            return_exceptions=True,
        )

        shadow_races = []
        for race_no, race in enumerate(fetched, start=1):
            if isinstance(race, Exception):
                print(f"  Race {race_no}: fetch failed ({race}), stopping", flush=True)
                break
            if not race or not race.horses:
                print(f"  Race {race_no}: no data, stopping", flush=True)
                break

            analyses = analyzer.analyze_race(race)
            if not analyses:
                continue

            # Infer race-level bet type from per-horse recommendations (top 4)
            bet_type = _infer_bet_type(analyses)

            # Slim leg payload — only what we need for evaluation
            legs = [{
                "rank": a.rank,
                "horse_number": a.horse.number,
                "horse_name": a.horse.name,
                "win_odds": a.horse.win_odds,
                "bet_recommendation": a.bet_recommendation,
                "calibrated_prob": round(a.combined_probability, 4),
            } for a in analyses[:5]]

            shadow_races.append({
                "race_number": race_no,
                "bet_type": bet_type,
                "legs": legs,
            })
            print(f"  Race {race_no}: {len(race.horses)} horses, "
                  f"shadow bet={bet_type}, top-1={analyses[0].horse.name}",
                  flush=True)

        date_str = race_date.replace('/', '')
        out_path = Path(__file__).resolve().parent.parent / "results" / f"shadow_race_data_{date_str}.json"
        payload = {
            "rule_w": rule_w,
            "ml_w": 1.0 - rule_w,
            "generated_at": datetime.now().isoformat(timespec='seconds'),
            "race_date": race_date,
            "races": shadow_races,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"Wrote {out_path} ({len(shadow_races)} races)", flush=True)
        return 0
    finally:
        await fetcher.close()


def _infer_bet_type(analyses) -> str:
    """Mirror of scripts/backtest._infer_race_bet_type."""
    for a in analyses[:4]:
        rec = a.bet_recommendation
        if rec in ('BANKER WIN', 'WIN'):
            return 'WIN'
        if rec == 'TRIO':
            return 'TRIO'
    qin_count = sum(1 for a in analyses[:4] if a.bet_recommendation == 'QIN')
    skip_count = sum(1 for a in analyses[4:] if a.bet_recommendation == 'SKIP')
    if qin_count == 4 and (len(analyses) <= 4 or skip_count > 0):
        return 'BOX Q4'
    if qin_count == 3:
        return 'BOX Q'
    if qin_count == 2:
        return 'QIN'
    return 'SKIP'


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', help='YYYY/MM/DD')
    p.add_argument('--rule-w', type=float, default=0.9,
                   help='Rule weight for shadow (default: 0.9 vs prod 0.6)')
    args = p.parse_args()
    return asyncio.run(run(args.race_date, rule_w=args.rule_w))


if __name__ == '__main__':
    sys.exit(main())
