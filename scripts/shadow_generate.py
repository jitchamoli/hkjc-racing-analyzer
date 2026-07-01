"""Generate shadow predictions for forward-validation experiments.

Two variants supported:

  rule_w       (default, legacy): patches config.RULE_WEIGHT=0.9 vs prod 0.6.
               Output: results/shadow_race_data_YYYYMMDD.json

  jockey_form: patches config.JOCKEY_FORM_RIDES_MIN=15 (effectively disables
               the small-sample-noisy jockey_form rule-engine factor — the
               SQL LIMIT is 10, so the gate never fires). Output:
               results/shadow_jockey_race_data_YYYYMMDD.json
               Hypothesis: backtest 2026-05-23 showed +6.24pp straight QIN
               ROI and better Brier/BSS/ECE from disabling this factor.
               Shadow-log to forward-validate before shipping.

Run AFTER `generate_predictions.py` so production output is locked in. This
script re-fetches the same race cards from HKJC, runs the analyzer with the
patched config, and writes a slim shadow JSON capturing only the bet
selections (no HTML, no DB logging, no dashboard side-effects).

Usage:
    python3 -m scripts.shadow_generate '2026/05/17'                       # rule_w
    python3 -m scripts.shadow_generate '2026/05/17' --variant jockey_form
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


VARIANTS = {
    'rule_w': {
        'output_basename': 'shadow_race_data',
        'description': 'rule_w=0.9 (vs prod 0.6)',
    },
    'jockey_form': {
        'output_basename': 'shadow_jockey_race_data',
        'description': 'jockey_form factor disabled (rides_min=15 vs prod 3)',
    },
    'tips': {
        'output_basename': 'shadow_tips_race_data',
        'description': 'TIPS factor disabled (weight 0.11 -> 0)',
    },
}


def _apply_variant(variant: str, rule_w: float) -> dict:
    """Patch config for the chosen variant. Returns metadata dict for the
    payload header so the evaluator knows what was tested."""
    if variant == 'rule_w':
        config.RULE_WEIGHT = rule_w
        config.ML_WEIGHT = 1.0 - rule_w
        return {'variant': 'rule_w', 'rule_w': rule_w, 'ml_w': 1.0 - rule_w}
    if variant == 'jockey_form':
        config.JOCKEY_FORM_RIDES_MIN = 15
        return {'variant': 'jockey_form', 'jockey_form_rides_min': 15}
    if variant == 'tips':
        # Zero the TIPS weight so compute_weighted_score renormalizes it out.
        # WEIGHT_MAP is read live by the rule engine, so patching it here (in
        # this isolated shadow subprocess) re-scores every race with TIPS off
        # while production is untouched. TIPS data is on the live card, so the
        # shadow genuinely differs from prod (unlike a backtest, where tips is
        # never stored — hence forward-shadow is the only valid A/B). 2026-06-27.
        from core.scorer import BettingWeights
        BettingWeights.WEIGHT_MAP['tips'] = 0.0
        return {'variant': 'tips', 'tips_weight': 0.0}
    raise ValueError(f"Unknown variant: {variant}. Choose from {list(VARIANTS)}.")


async def run(race_date: str, variant: str = 'rule_w', rule_w: float = 0.9) -> int:
    # Patch config BEFORE importing analyzer modules so any module-level reads
    # use the new values. Most reads are call-time so this is belt-and-braces.
    meta = _apply_variant(variant, rule_w)

    from core.fetcher import HKJCDataFetcher
    from core.analyzer import CombinedRaceAnalyzer

    print(f"Shadow generator — variant={variant} ({VARIANTS[variant]['description']}) "
          f"on {race_date}", flush=True)

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
        # Backfill blank per-race going from the meeting modal so the shadow's
        # CHAOS detection matches production (see dashboard/generator.py).
        from dashboard.generator import backfill_meeting_going
        backfill_meeting_going(fetched)

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
        basename = VARIANTS[variant]['output_basename']
        out_path = Path(__file__).resolve().parent.parent / "results" / f"{basename}_{date_str}.json"
        payload = {
            **meta,
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
    p.add_argument('--variant', choices=list(VARIANTS), default='rule_w',
                   help='Which shadow experiment to run (default: rule_w)')
    p.add_argument('--rule-w', type=float, default=0.9,
                   help='Rule weight for rule_w variant (default: 0.9 vs prod 0.6)')
    args = p.parse_args()
    return asyncio.run(run(args.race_date, variant=args.variant, rule_w=args.rule_w))


if __name__ == '__main__':
    sys.exit(main())
