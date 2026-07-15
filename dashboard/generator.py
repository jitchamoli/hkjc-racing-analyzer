#!/usr/bin/env python3
"""
Generate race predictions and dashboard - non-interactive
"""

import asyncio
import itertools
import re
import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from dashboard.template import generate_html
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase, OddsMovementAnalyzer
from analytics.pace import PaceAnalyzer
from betting.exotic_calculator import find_value_exotics

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preflight_dashboard_ml() -> dict:
    """Fail closed before race-day dashboard generation touches live data."""
    from ml.integrity import check_dashboard_ml_integrity
    return check_dashboard_ml_integrity()


def _atomic_write(path: str, content: str) -> None:
    # Write to a sibling temp file then os.replace so concurrent readers
    # (the dashboard server, a browser polling the file) never see a
    # half-written file. os.replace is atomic on POSIX and Windows ≥ Vista.
    # try/finally cleans up the tmp on write failure so .tmp.{pid} files
    # don't accumulate after disk-full / SIGKILL / etc.
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, 'w') as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


_OVERRIDES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'tracking_data', 'manual_overrides.json',
)


def _load_manual_overrides() -> dict:
    """Tactical per-race overrides for the rank-6 TRIO-6 swap. When the auto
    rank-6 swap picks a poor candidate (e.g., horse off 168-day layoff) but
    racing-API form/RPR signal flags a better hedge, the user can set the
    TRIO-6 rank-6 horse here. Schema documented at the top of the JSON file.
    Returns {} if file missing or unreadable so a misfiled override never
    breaks dashboard generation.
    """
    if not os.path.exists(_OVERRIDES_PATH):
        return {}
    try:
        with open(_OVERRIDES_PATH) as f:
            return json.load(f) or {}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"manual_overrides.json unreadable, skipping: {e}")
        return {}


# Result fields set by results_fetcher that should survive prediction re-runs
_RESULT_KEYS_RACE = {'results_available', 'actual_results', 'actual_winner', 'actual_top3', 'actual_win_odds', 'accuracy'}
_RESULT_KEYS_HORSE = {'actual_position'}

# Pre-race prediction fields that should be FROZEN once a race is finished.
# Re-running generate_predictions after a race ends pulls post-race "odds"
# (HKJC swaps in final SP / dividend prices once the race is settled), which
# corrupts the model blend and produces fake EDGE fires on losing horses.
# Audit 2026-05-17: R1 flipped MED → EDGE post-race because #10's odds
# moved $7.40 → $3.10 in the post-race feed, dragging cal_probs.
_FROZEN_PREDICTION_KEYS_RACE = {'conviction_tier', 'has_edge', 'suggested_stake',
                                 'bankroll', 'odds_age_minutes',
                                 'bet_ticket',
                                 # Scheduled post time. The fresh fetch can drop
                                 # it (Playwright fallback / late race-info header
                                 # — the same source blank that backfill_meeting_
                                 # going patches for `going`). _merge_with_existing
                                 # detects the off with an old_race fallback, but
                                 # finalize_tickets_and_stakes._race_off reads only
                                 # r.get('time'); an unfrozen blank made it read a
                                 # run race as live and rebuild its ticket on
                                 # settled odds (audit 2026-07-12 round 2).
                                 'time',
                                 # Shape stats — must move in lockstep with the
                                 # frozen horse-level calibrated_prob (audit
                                 # 2026-05-17, ML expert finding HIGH-1)
                                 'top_2_share', 'top_3_share',
                                 # EDGE banner identifiers — must point at the
                                 # horse whose frozen edge_fires=True
                                 'edge_horse_rank', 'edge_horse_number',
                                 'edge_horse_name',
                                 # Tactical manual override for TRIO-6 rank-6
                                 # slot (set via tracking_data/manual_overrides.json)
                                 'manual_trio6_horse', 'manual_trio6_reason',
                                 # BOX TRIO-6 take/skip + chaos decision
                                 # (_compute_trio6_decision). MUST be frozen
                                 # alongside bet_ticket — otherwise a post-off
                                 # re-render recomputes the verdict from drifted
                                 # odds while the ticket stays frozen, and the
                                 # displayed verdict disagrees with what was
                                 # staked (audit-2026-05-29 finding A). Locked by
                                 # tests/test_audit_2026_05_29.py::test_trio6_decision_frozen.
                                 'trio6_decision'}
_FROZEN_PREDICTION_KEYS_HORSE = {'rank', 'win_odds', 'calibrated_prob',
                                  'edge_pp', 'ev_win', 'overlay', 'edge_fires',
                                  'pure_prob', 'market_prob', 'model_rank',
                                  'bet_recommendation', 'confidence',
                                  'expected_return', 'value_score',
                                  # Feeds _compute_trio6_decision. Left mutable, it
                                  # was recomputed from POST-race odds in the
                                  # post-off window, flipping the TRIO-6 verdict
                                  # (audit 2026-07-12).
                                  'steam_move'}


def _post_time_passed(race_time_str, race_date: str | None = None) -> bool:
    """True once a race's scheduled post time (HKT) has passed, for TODAY's card.

    The results-derived flags (`results_available`, `actual_position`) only flip
    after results_fetcher runs — but `raceday live` runs generation BEFORE it. So
    for the whole post-off/pre-results window a race that has ALREADY RUN still
    looks unresolved, and anything keyed on those flags will happily rebuild it
    from post-race odds. This is the third "has it run?" signal, and the only one
    available in that window.

    Gated to today's date so regenerating a past or future card never reads a
    time-of-day as "off'd".
    """
    if not race_time_str:
        return False
    today_str = datetime.now().strftime('%Y/%m/%d')
    if race_date is not None and race_date != today_str:
        return False
    try:
        hh, mm = str(race_time_str).split(':')[:2]
        now = datetime.now()
        return (now.hour * 60 + now.minute) >= (int(hh) * 60 + int(mm))
    except (ValueError, AttributeError):
        return False


def _merge_with_existing(json_file: str, new_races: list, race_date: str | None = None) -> list:
    """Preserve result data from a prior prediction file when re-generating.

    If the existing file has actual_position / accuracy fields (set by
    results_fetcher), carry them forward into the new predictions so a
    re-run for fresher odds doesn't wipe scored results.

    race_date (YYYY/MM/DD) gates the L5 race-has-offed restoration to
    today's date only — historical/future regen must not treat past
    times-of-day as if the race has off'd (Finding #3, 2026-05-24).
    """
    # Prefer _with_results variant (has scored results), fall back to base file
    results_file = json_file.replace('.json', '_with_results.json')
    source = None
    for candidate in [results_file, json_file]:
        if os.path.exists(candidate):
            source = candidate
            break

    if not source:
        return new_races

    try:
        with open(source) as f:
            old_data = json.load(f)
        # _with_results wraps races in {"races": [...]}
        old_races = old_data.get('races', old_data) if isinstance(old_data, dict) else old_data
    except (json.JSONDecodeError, OSError):
        return new_races

    # Build lookup: race_number -> old race data
    old_by_num = {r['race_number']: r for r in old_races if isinstance(r, dict)}

    # Sanity check: if all races with results share the same winner, data is likely bogus
    winners_with_results = [
        r.get('actual_winner') for r in old_by_num.values()
        if r.get('results_available') and r.get('actual_winner')
    ]
    if len(winners_with_results) >= 3 and len(set(winners_with_results)) == 1:
        logger.warning(f"  Skipping result merge — all {len(winners_with_results)} races "
                       f"have same winner #{winners_with_results[0]} (likely bogus data)")
        return new_races

    for race in new_races:
        old_race = old_by_num.get(race['race_number'])
        if not old_race:
            continue

        # Post-off / pre-results-merge protection: the original audit-2026-05-17
        # freeze only fires for results_available=True races, leaving a 1-5 min
        # window after off where bookies have settled and odds drift, but
        # results_fetcher hasn't yet flipped results_available. In that window,
        # fresh tier/has_edge/shape stats get computed from post-race-noisy
        # odds. Detect "race has off'd" by checking the scheduled post time
        # (`race['time']`, HKT) against now. If past, restore all the frozen
        # race-level fields from the prior file. Caught on R3/R4/R5 (bet_ticket)
        # and R7 (conviction_tier flipped LOW → EDGE post-off) 2026/05/24.
        # race_date guard added 2026-05-24 (Finding #3): only fires when the
        # file being generated is for today, so historical/future regen doesn't
        # treat past times-of-day as off'd.
        race_has_offed = _post_time_passed(
            race.get('time') or old_race.get('time'), race_date)

        if race_has_offed:
            for key in _FROZEN_PREDICTION_KEYS_RACE:
                if key in old_race:
                    race[key] = old_race[key]
            # Horse-level freeze added 2026-05-24 (Finding #1): the L5 block
            # above protected race-level keys, but `database.log_predictions`
            # later UPDATEs prediction_log.predicted_rank from race['horses'],
            # which was still mutable in the post-off/pre-results window.
            # That corrupted forward-validation rank → poisoned validate_hk.
            old_horses_by_num = {h['number']: h for h in old_race.get('horses', []) if 'number' in h}
            for horse in race.get('horses', []):
                old_h = old_horses_by_num.get(horse.get('number'))
                if old_h:
                    for key in _FROZEN_PREDICTION_KEYS_HORSE:
                        if key in old_h:
                            horse[key] = old_h[key]
            old_horses_by_rank = sorted(
                (h for h in old_race.get('horses', []) if 'rank' in h),
                key=lambda h: h.get('rank', 99)
            )
            if old_horses_by_rank:
                race['horses'] = sorted(race.get('horses', []),
                                         key=lambda h: h.get('rank', 99))
        elif 'bet_ticket' in old_race:
            # Pre-off: restore prior bet_ticket so finalize_tickets_and_stakes
            # has the OLD ticket to compare against. Layer 2 of that function
            # (archive-on-mutate via _merge_ticket_pre_result) needs the prior
            # to know whether a gate flip happened — without this restore the
            # new generation would silently skip archiving. The morning ticket
            # may be replaced; the old one is preserved in
            # bet_ticket.superseded_tickets[] for audit.
            race['bet_ticket'] = old_race['bet_ticket']

        if not old_race.get('results_available'):
            continue

        # Copy race-level result fields
        for key in _RESULT_KEYS_RACE:
            if key in old_race:
                race[key] = old_race[key]

        # Freeze pre-race prediction fields on finished races (audit 2026-05-17).
        # HKJC's post-race odds feed corrupts the model blend on re-run,
        # producing fake EDGE fires on horses that finished outside the money.
        for key in _FROZEN_PREDICTION_KEYS_RACE:
            if key in old_race:
                race[key] = old_race[key]

        # Copy horse-level result fields + freeze pre-race predictions
        old_horses = {h['number']: h for h in old_race.get('horses', []) if 'number' in h}
        # Build a horse-number → predicted_rank-ordered list so we can also
        # restore the original rank ordering (post-race feed can reshuffle).
        old_horses_by_rank = sorted(
            (h for h in old_race.get('horses', []) if 'rank' in h),
            key=lambda h: h.get('rank', 99)
        )
        for horse in race.get('horses', []):
            old_h = old_horses.get(horse['number'])
            if old_h:
                for key in _RESULT_KEYS_HORSE:
                    if key in old_h:
                        horse[key] = old_h[key]
                for key in _FROZEN_PREDICTION_KEYS_HORSE:
                    if key in old_h:
                        horse[key] = old_h[key]
        # Re-sort horses by the frozen rank so downstream renderers see the
        # original pre-race ordering, not the post-race shuffled one.
        if old_horses_by_rank:
            race['horses'] = sorted(race.get('horses', []),
                                     key=lambda h: h.get('rank', 99))

        logger.info(f"  Race {race['race_number']}: FROZEN — preserved pre-race "
                    f"predictions + results from prior run")

    # Carry forward any race the fresh fetch dropped. A transient card-fetch
    # failure makes generate_predictions skip that race; the predictions file is
    # then overwritten WHOLESALE, and results_fetcher iterates the predictions —
    # so a settled race's ticket, results and STRIKE badges vanish from the card
    # and stake_ab silently omits its bets. The DB is unaffected (so replay still
    # sees it), but the on-disk record is a lie (audit 2026-07-12).
    new_nums = {r.get('race_number') for r in new_races}
    dropped = [r for n, r in old_by_num.items() if n not in new_nums]
    if dropped:
        logger.warning(
            f"  {len(dropped)} race(s) missing from this fetch "
            f"({sorted(r.get('race_number') for r in dropped)}) — carrying the "
            f"prior version forward rather than dropping them from the card."
        )
        new_races = sorted(new_races + dropped,
                           key=lambda r: r.get('race_number') or 0)

    return new_races


def _load_previous_race_data(json_file: str) -> list:
    """Return race list from the most recent prior JSON at this path, or []."""
    if not os.path.exists(json_file):
        return []
    try:
        with open(json_file) as f:
            data = json.load(f)
        return data.get('races', data) if isinstance(data, dict) else data
    except (json.JSONDecodeError, OSError):
        return []


def _compute_odds_age_minutes(odds_db, race_date: str, race_number: int):
    """Return minutes elapsed since the most recent odds snapshot for this race.

    Returns None when no snapshot exists OR on any error — caller treats
    None as 'no fresh odds' and FAILS G5. Previously this returned 0, which
    passed the `< 30` gate even when there were zero snapshots in the DB,
    inverting the gate's stated purpose ('stale odds invalidate EV').
    """
    try:
        conn = odds_db._get_conn()
        try:
            row = conn.execute(
                "SELECT MAX(timestamp) FROM odds_snapshots "
                "WHERE race_date = ? AND race_number = ?",
                (race_date, race_number),
            ).fetchone()
            if row and row[0]:
                last_ts = datetime.strptime(row[0], '%Y-%m-%d %H:%M:%S')
                return max(0, int((datetime.now() - last_ts).total_seconds() / 60))
        finally:
            conn.close()
    except Exception:
        pass
    return None


def backfill_meeting_going(fetched: list) -> None:
    """Fill blank per-race going from the meeting's modal going, in place.

    Going is a meeting-wide track property, but the per-page regex in
    core/fetcher.py reads it from each race page independently — so a race
    fetched via the Playwright fallback (or whose race-info header loads late)
    can come back with going='' even though every other race reads 'Yielding'.
    That blank lets the CHAOS soft-going filter (detect_chaos) silently skip
    the race, so it falsely escapes the no-bet override on a soft card
    (2026/06/13 R5/R9 staked bets they should not have). Backfilling the SOURCE
    race object before analyze/build means build_race_data + detect_chaos see the
    correct value with no downstream changes. Shared by the production generator,
    the live monitor (parity), and the shadow generator so all three agree.
    `fetched` items may be race objects or Exceptions (asyncio.gather results).
    """
    goings = [r.going for r in fetched
              if not isinstance(r, Exception) and r and getattr(r, 'going', '')]
    if not goings:
        return
    from collections import Counter
    modal_going = Counter(goings).most_common(1)[0][0]
    for race_no, race in enumerate(fetched, start=1):
        if isinstance(race, Exception) or not race:
            continue
        if not getattr(race, 'going', ''):
            race.going = modal_going
            logger.info(f"  Race {race_no}: blank going backfilled → "
                        f"'{modal_going}' (meeting-wide modal)")


def apply_card_exposure_cap(all_races: list, bankroll: float,
                            cap_frac: float = 0.10) -> float:
    """Cap the day's TOTAL forward suggested-stake exposure at cap_frac of
    bankroll, pro-rating every UNFINISHED race's stake down proportionally.

    Per-race stakes are computed independently, so a busy card can put 15-20%
    of bankroll at risk at once (and the box bets within a race are correlated
    — they win/lose together). Settled races aren't at risk so they're excluded
    from the cap. Mutates each capped race's 'suggested_stake' in place and
    returns the applied scale factor (1.0 = no cap needed).
    """
    if not cap_frac or cap_frac <= 0:
        return 1.0
    pending = [r for r in all_races
               if not r.get('results_available') and r.get('suggested_stake')]
    total_stake = sum((r['suggested_stake'].get('amount') or 0.0) for r in pending)
    cap_amount = bankroll * cap_frac
    if not (total_stake > cap_amount > 0):
        return 1.0
    scale = cap_amount / total_stake
    for r in pending:
        s = r['suggested_stake']
        s['amount'] = round((s.get('amount') or 0.0) * scale, -1)
        if s.get('pct') is not None:
            s['pct'] = round(s['pct'] * scale, 2)
        s['card_capped'] = True
        s['method'] = (s.get('method') or '') + \
            f" · card-capped ×{scale:.2f} ({int(cap_frac*100)}% day limit)"
    return scale


def _ticket_add_bet(bets: list, label: str, pool: str, horses: list,
                    combos: int, reason: str, swap: dict | None = None) -> None:
    """Final ticket-level no-odds guard (GPT-5.5 R13 #1 2026-05-25).

    The recommender skips no-odds horses for PRIMARY bets, but the
    automatic add-ons (BOX QIN-4 PROTECTION, F4 SAVER, BOX TRIO-5,
    F4 BOX-5, BOX TRIO-6) call this helper with raw top-N rank slices
    and never re-validated odds. A horse with win_odds<=0 could enter
    an add-on ticket and be staked.

    Refuse any bet whose legs contain a horse without a live price.
    Log + drop the bet rather than silently strip the bad leg — a
    different combo count would change the ticket semantics."""
    if combos <= 0 or not horses:
        return
    bad_legs = [h for h in horses
                if not (h.get('win_odds') and h.get('win_odds') > 0)]
    if bad_legs:
        bad_names = ', '.join(
            f"#{h.get('number')} {h.get('name')}" for h in bad_legs
        )
        logger.warning(
            f"Refusing to add {label} — leg(s) have no live odds: {bad_names}"
        )
        return
    # Duplicate-leg guard: `combos` is computed from the leg COUNT, so a repeated
    # horse bills a smaller box at a larger box's price (HKJC prices C(distinct,k)).
    # Drop rather than de-dupe — a different combo count is a different bet.
    nums = [h.get('number') for h in horses]
    if len(set(nums)) != len(nums):
        logger.warning(
            f"Refusing to add {label} — duplicate leg(s) in {nums}"
        )
        return
    entry = {
        'label': label,
        'pool': pool,
        'combos': int(combos),
        'legs': [
            {'rank': h.get('rank'), 'number': h.get('number'), 'name': h.get('name')}
            for h in horses
        ],
        'reason': reason,
    }
    if swap:
        entry['swap'] = swap
    bets.append(entry)


def _compute_trio6_decision(*, horses: list, tier: str, field: int,
                             top1_odds: float, top_2_share: float) -> dict:
    """Single source of truth for BOX TRIO-6 take/skip + chaos annotation.

    Returns a dict consumed by:
      - build_bet_ticket (server-side gate to include BOX TRIO-6 leg)
      - dashboard/template.py detail-panel render (verdict + chaos badge)
      - dashboard/template.py WA compact copy (verdict in MAIN/Optional)

    Before consolidation (2026-05-29) all three sites recomputed independently
    and had drifted: WA had a top_2_share<0.40 chaos clause server lacked,
    so the staked ticket and the displayed verdict could disagree on the
    same race. Locked by tests/test_audit_2026_05_29.py::test_trio6_decision_single_source.

    Gate logic (first match wins):
      SKIP — heavy fav (top1 < $2.50)
      TAKE — ≥2 morning-steam horses in top 12
      TAKE — HIGH tier AND field ≥11 AND $2.50-$5.00 favorite
      YOUR CALL — otherwise (and TAKE flag stays False — bet not added)

    Chaos score 0-4 (independent of take/skip — pure annotation):
      +1 field ≥13
      +1 $2.50-$5.00 favorite
      +1 ≥2 morning-steam horses in top 12
      +1 top_2_share < 0.40 (model itself uncertain)
    """
    steam_count = sum(
        1 for h in horses[:12]
        if (h.get('steam_move') or {}).get('is_steam')
    )

    # Gate (whether to add BOX TRIO-6 to the ticket)
    if top1_odds > 0 and top1_odds < 2.5:
        take = False
        verdict = 'SKIP heavy fav'
    elif steam_count >= 2:
        take = True
        verdict = f'★ TAKE ({steam_count} steam)'
    elif tier == 'HIGH' and field >= 11 and 2.5 <= top1_odds <= 5.0:
        take = True
        verdict = '★ TAKE'
    else:
        take = False
        verdict = 'YOUR CALL'

    # Chaos annotation (independent of take/skip)
    factors = []
    chaos_score = 0
    if field >= 13:
        chaos_score += 1
        factors.append(f'large field {field}')
    if 2.5 <= top1_odds <= 5.0:
        chaos_score += 1
        factors.append(f'competitive fav ${top1_odds:.1f}')
    if steam_count >= 2:
        chaos_score += 1
        factors.append(f'{steam_count} steam horses')
    if 0 < top_2_share < 0.40:
        chaos_score += 1
        factors.append(f'top-2 share {top_2_share*100:.0f}% (model uncertain)')

    return {
        'take': bool(take),
        'verdict_label': verdict,
        'chaos_score': int(chaos_score),
        'chaos_factors': factors,
        'top1_odds': float(top1_odds or 0),
        'field_size': int(field),
        'steam_count': int(steam_count),
        'top_2_share': float(top_2_share or 0),
    }


def _pick_trio6_top6(horses: list, manual_trio6_horse, manual_reason: str = ''):
    """Return (top6_horses, swap_meta_dict_or_None) for BOX TRIO-6.

    Mirrors the JS rank-6 swap logic in dashboard/template.py (moved
    server-side 2026-05-25, GPT-5.5 audit Finding #2). Without this, the
    ticket stored horses[:6] while the UI/WA showed a swapped top-6 →
    validate_hk --replay scored different legs than the user staked.

    Priority:
      1. Manual override (race.manual_trio6_horse) wins outright.
      2. Auto swap: scan ranks 7-12 for a horse with (a) odds ≤ rank-6
         × 0.8 AND (b) gear visible. Pick the shortest-priced candidate.
      3. Otherwise return the literal top-6 by rank.
    """
    if len(horses) < 6:
        return horses[:6], None
    top6 = horses[:6]
    # Manual override
    if manual_trio6_horse:
        # Must not already be in the top 5. Ranks move with odds between the AM
        # override and the T-6min re-run, so an override horse that has climbed
        # into the top 5 would be appended a SECOND time — 6 legs, 5 distinct
        # horses, billed as C(6,3)=20 combos for a box HKJC can only accept as
        # C(5,3)=10. The stake doubles and validate_hk (which de-dupes legs to a
        # set but charges combos × unit) scores a TRIO-5 at TRIO-6 money.
        top5_nums = {h.get('number') for h in horses[:5]}
        if (top6[5].get('number') != manual_trio6_horse
                and manual_trio6_horse not in top5_nums):
            manual_h = next((h for h in horses if h.get('number') == manual_trio6_horse), None)
            if manual_h:
                return horses[:5] + [manual_h], {
                    'kind': 'manual',
                    'out_number': top6[5].get('number'),
                    'in_number': manual_h.get('number'),
                    'reason': manual_reason,
                }
        # else: override is already in the top 6 — no swap
    # Auto swap
    rank6 = top6[5]
    rank6_odds = rank6.get('win_odds') or 0
    if rank6_odds > 0:
        best = None
        for cand in horses[6:12]:
            cand_odds = cand.get('win_odds') or 0
            cand_gear = (cand.get('gear') or '').strip()
            if cand_odds > 0 and cand_gear and cand_odds <= rank6_odds * 0.8:
                if best is None or cand_odds < (best.get('win_odds') or 0):
                    best = cand
        if best:
            return horses[:5] + [best], {
                'kind': 'auto',
                'out_number': rank6.get('number'),
                'in_number': best.get('number'),
                'in_rank': best.get('rank'),
                'in_gear': (best.get('gear') or '').strip(),
            }
    return top6, None


def _ticket_signature(t: dict) -> tuple:
    """Identity for `bet_ticket` change detection: combos + sorted
    (label, sorted_leg_numbers) pairs. Catches combo/label/leg drift but
    ignores cosmetic metadata (frozen_at, source, superseded_tickets)."""
    if not t:
        return (0, ())
    combos = t.get('total_combos', 0) or 0
    pairs = []
    for b in (t.get('bets') or []):
        leg_nums = tuple(sorted(
            l.get('number') for l in (b.get('legs') or []) if l.get('number') is not None
        ))
        pairs.append((b.get('label', ''), leg_nums))
    return (combos, tuple(sorted(pairs)))


def _merge_ticket_pre_result(old_ticket: dict | None, new_ticket: dict,
                             generated_at: str) -> dict:
    """Pre-RESULT merge with archive-on-mutate.

    Where this sits in the freeze chain:
      1. `_merge_with_existing` enforces the real post-OFF freeze for
         today's card via scheduled `race['time']` — once past post time
         it restores all frozen race+horse fields (including bet_ticket)
         from the prior file, so this function never sees post-off races
         for today's date.
      2. `finalize_tickets_and_stakes` adds a post-RESULT fallback
         (`results_available` / `actual_position`) for non-today
         scenarios and as belt-and-braces after results land.
      3. THIS helper runs only for races that survived (1) and (2) —
         i.e. truly pre-result. It allows pre-result gate flips and
         archives the old ticket into `superseded_tickets[]`.

    Behavior:
      - first build (old=None) → tag source=pre_off_build, stamp
        frozen_at + odds_snapshot_at, init empty superseded_tickets.
      - same signature → preserve prior source + superseded_tickets,
        refresh timestamps only.
      - different signature (combos / labels / legs) → archive old
        into superseded_tickets[] with superseded_at, replace with
        new, mark source=pre_off_refresh.
    """
    merged = dict(new_ticket)
    merged['frozen_at'] = generated_at
    merged['odds_snapshot_at'] = generated_at

    if old_ticket is None:
        merged['source'] = 'pre_off_build'
        merged['superseded_tickets'] = []
        return merged

    history = list(old_ticket.get('superseded_tickets') or [])
    if _ticket_signature(old_ticket) == _ticket_signature(new_ticket):
        merged['source'] = old_ticket.get('source', 'pre_off_build')
        merged['superseded_tickets'] = history
        return merged

    archived = {k: v for k, v in old_ticket.items() if k != 'superseded_tickets'}
    archived['superseded_at'] = generated_at
    history.append(archived)
    merged['source'] = 'pre_off_refresh'
    merged['superseded_tickets'] = history
    return merged


def finalize_tickets_and_stakes(all_races: list, race_date: str, analyzer) -> None:
    """Compute per-race suggested stake + apply card-level exposure cap +
    build frozen bet_tickets with all freeze guards. Shared between
    generator and monitor so both surfaces emit the same staking +
    ticket structure (GPT-5.5 R13 #2 2026-05-25).

    Mutates `all_races` in place: adds `suggested_stake`, `bankroll`, and
    `bet_ticket` per race. Skips recompute on finished races (frozen
    ticket + stake are restored by _merge_with_existing).

    Freeze chain (upstream → downstream):
      1. `_merge_with_existing` is the primary post-OFF freeze for today:
         once `now >= race['time']` (HKT scheduled post), it restores all
         frozen race+horse fields (incl. bet_ticket) from the prior file.
         So for today's card, this function never sees a post-off race.
      2. Layer 1 here (`results_available=True` or any `actual_position`)
         is the post-RESULT fallback — catches non-today scenarios and
         the small window between off and results landing on dates the
         scheduled-time guard isn't authoritative for.
      3. Layer 2 here (`_merge_ticket_pre_result`) is pre-RESULT
         archive-on-mutate: pre-off gate flips ARE allowed; the old
         ticket goes to `bet_ticket.superseded_tickets[]` for audit.

    Operational rule: don't run `generate_predictions.py` for a date
    other than today after off — the scheduled-time guard only fires
    for today's date by design (see `_merge_with_existing`).
    """
    from betting.optimizer import compute_suggested_stake
    import config as _cfg

    def _race_off(r) -> bool:
        """Has this race run? Three signals, because the first two are blind in
        the post-off/pre-results window that `raceday live` generates inside.

        Keying only on results (as every site here used to) meant a race that
        had already run was treated as live: its stake was recomputed and its
        FROZEN ticket rebuilt from post-race odds. That is how 2026/06/27 R4 and
        2026/07/01 R2 acquired a BOX TRIO-6 that appears in NO pre-race freeze —
        and those phantom legs then hit, crediting `validate_hk --replay` with
        money that was never staked (audit 2026-07-12).
        """
        if r.get('results_available'):
            return True
        if any(h.get('actual_position') for h in (r.get('horses') or [])):
            return True
        return _post_time_passed(r.get('time'), race_date)

    # Per-card suggested stake — bankroll-fraction (BOX) or half-Kelly
    # (WIN/BANKER WIN), tier-capped. See compute_suggested_stake.
    for r in all_races:
        if _race_off(r):
            continue
        top = (r.get('horses') or [{}])[0]
        stake = compute_suggested_stake(
            conviction_tier=r.get('conviction_tier') or 'SKIP',
            bet_type=top.get('bet_recommendation') or '',
            top_prob=top.get('calibrated_prob') or 0.0,
            top_odds=top.get('win_odds') or 0.0,
            bankroll=_cfg.BANKROLL,
            top_2_share=r.get('top_2_share'),
        )
        r['suggested_stake'] = stake
        r['bankroll'] = _cfg.BANKROLL

    # Card-level exposure cap
    apply_card_exposure_cap(all_races, _cfg.BANKROLL,
                            getattr(_cfg, 'CARD_EXPOSURE_CAP', 0.10))

    # Bankroll discipline — enforce the daily stop-loss / exposure cap that
    # betting/bankroll_guard exposes. It was previously advisory-only: no
    # generation path called it, so the system kept recommending fresh money
    # after the day's loss/exposure cap was hit (audit 2026-06-08 #5). When the
    # day is HALTED, zero the suggested stake on every not-yet-resolved race
    # BEFORE the ticket is built — so build_bet_ticket (which only adds bets
    # when amount>0) produces an EMPTY system ticket. The WA-copy + visual
    # ticket render from bet_ticket.bets, so zeroing only suggested_stake would
    # still show/copy a live bet (audit follow-up 2026-06-08 #1). Guarded: a
    # guard failure must never block prediction generation.
    # Only a realized STOP-LOSS blanks the card. The guard ALSO halts on the
    # daily EXPOSURE cap (cumulative pending+resolved stake >= 10%), but after
    # the HK$10 unit floor ~doubled allocation that cap is tripped by a busy
    # card's own pending tickets on every live/finish re-run — blanking the
    # whole card as NO SYSTEM BET with zero losses. That "you've planned a lot"
    # signal is already surfaced honestly by the #1 card-over-cap banner below,
    # so here we react ONLY to a genuine realized loss (chase-prevention).
    halt_reason = None
    try:
        from betting.bankroll_guard import daily_status
        guard = daily_status(race_date, _cfg.BANKROLL)
        # React ONLY to a realized stop-loss, not the exposure cap. Inspect the
        # guard's own halt_reason rather than re-deriving the loss threshold from
        # config — a local getattr fallback could silently diverge from the value
        # the guard actually halted on. daily_status sets halt_reason to
        # "DAILY LOSS CAP ..." or "DAILY EXPOSURE CAP ..." (bankroll_guard.py).
        if guard.halted and 'LOSS CAP' in (guard.halt_reason or ''):
            halt_reason = guard.halt_reason
    except Exception as e:
        logger.debug(f"bankroll guard check skipped: {e}")

    def _is_unresolved(r):
        return not _race_off(r)

    if halt_reason:
        for r in all_races:
            if _is_unresolved(r):
                r['suggested_stake'] = {
                    'amount': 0.0, 'pct': 0.0,
                    'method': f'STOP: {halt_reason}',
                }
                r['bankroll_halt'] = halt_reason
        logger.warning(
            f"Bankroll guard HALTED for {race_date} ({halt_reason}); "
            f"suggested stakes + system tickets zeroed on unresolved races."
        )

    model_ver = (analyzer.ml_analyzer.ml_model.model_version
                  if hasattr(analyzer, 'ml_analyzer') else '')
    ticket_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for r in all_races:
        # Freeze ticket once the race has run — including the post-off window
        # where no result has landed yet (see _race_off).
        if r.get('bet_ticket') and _race_off(r):
            continue
        old_ticket = r.get('bet_ticket')
        new_ticket = build_bet_ticket(
            r, race_date=race_date, model_version=model_ver, generated_at=ticket_ts,
        )
        merged = _merge_ticket_pre_result(old_ticket, new_ticket, ticket_ts)
        # Halted day: force the frozen ticket itself to a zero/empty system
        # ticket (belt-and-braces over the amount=0 build above, in case an
        # older non-empty ticket survives the merge). This is the ticket the
        # WA-copy + visual renderers read, so it must carry no payable bets.
        if halt_reason and _is_unresolved(r):
            merged['bets'] = []
            merged['total_combos'] = 0
            merged['unit_stake'] = 0.0
            merged['allocated_amount'] = 0.0
            merged['over_budget'] = False
            merged['over_amount'] = 0.0
            merged['bankroll_halt'] = halt_reason
            merged['no_bet_reason'] = f'STOP: {halt_reason}'
        elif old_ticket and merged.get('source') == 'pre_off_refresh':
            old_combos = old_ticket.get('total_combos', 0) or 0
            new_combos = merged.get('total_combos', 0) or 0
            logger.info(
                f"  R{r.get('race_number')}: bet_ticket rebuilt "
                f"({old_combos} → {new_combos} combos); prior ticket archived to "
                f"superseded_tickets[{len(merged.get('superseded_tickets') or [])}]."
            )
        r['bet_ticket'] = merged

    # Card-level allocated-spend disclosure (audit 2026-06-08 #1). The
    # pre-build apply_card_exposure_cap pro-rates the SUGGESTED budget, but the
    # HK$10 unit floor + automatic add-ons push the actual frozen allocation
    # above it — and that overshoot was never surfaced, so the card LOOKED
    # capped while real exposure was 2x+ the limit. Per user decision we do NOT
    # drop bets (all MAIN reward-max boxes are kept); instead we compute the
    # true allocated total and flag the residual honestly so the dashboard +
    # bankroll guard show real exposure and the user can trim manually.
    cap_frac = getattr(_cfg, 'CARD_EXPOSURE_CAP', 0.10)
    cap_amount = _cfg.BANKROLL * cap_frac
    pending = [r for r in all_races
               if _is_unresolved(r) and not r.get('bankroll_halt')
               and r.get('bet_ticket')]
    # Scope the exposure banner to the QIN pool the user actually stakes
    # (2026-07-01 expert review). Summing allocated_amount across ALL pools
    # (TRIO/F4 boxes the user never plays) over-warned ~4x on 2026/07/01 and
    # disagreed with bankroll_guard, which counts QIN only (_is_counted_label).
    # Align the two so the one real-wallet number is the one shown.
    from betting.bankroll_guard import _is_counted_label

    def _counted_alloc(ticket):
        return sum((b.get('amount') or 0.0) for b in (ticket.get('bets') or [])
                   if not b.get('paper') and _is_counted_label(b.get('label')))

    card_allocated = sum(_counted_alloc(r['bet_ticket']) for r in pending)
    over_cap = max(0.0, card_allocated - cap_amount) if cap_amount > 0 else 0.0
    for r in pending:
        r['card_allocated_total'] = round(card_allocated, 2)  # QIN-scoped
        r['card_exposure_cap'] = round(cap_amount, 2)
        r['card_exposure_cap_pct'] = round(cap_frac * 100, 1)
        r['card_over_cap'] = over_cap > 0
        r['card_over_cap_amount'] = round(over_cap, 2)
    if over_cap > 0:
        logger.warning(
            f"Card QIN exposure ${card_allocated:.0f} exceeds "
            f"{cap_frac*100:.0f}% cap (${cap_amount:.0f}) by ${over_cap:.0f} on "
            f"{race_date} — bets NOT dropped (MAIN bets kept); flagged for "
            f"manual trim."
        )


def build_bet_ticket(race: dict, *, race_date: str, model_version: str,
                     generated_at: str) -> dict:
    """Server-side frozen bet ticket; dashboard/evaluators should render this.

    The old UI derived extra MAIN bets client-side. Persisting the ticket here
    makes the actual decision auditable before results land.
    """
    horses = sorted(race.get('horses') or [], key=lambda h: h.get('rank') or 99)
    stake = race.get('suggested_stake') or {}
    amount = float(stake.get('amount') or 0.0)
    bets = []

    if amount > 0 and horses:
        by_rec = {}
        for h in horses:
            rec = (h.get('bet_recommendation') or '').strip()
            if rec and rec != 'SKIP':
                by_rec.setdefault(rec, []).append(h)

        for rec, legs in by_rec.items():
            n = len(legs)
            if rec in ('WIN', 'BANKER WIN'):
                _ticket_add_bet(bets, rec, 'WIN', legs[:1], 1, 'recommender')
            elif rec == 'PLACE':
                _ticket_add_bet(bets, 'PLACE', 'PLACE', legs[:1], 1, 'recommender')
            elif rec == 'QIN':
                combos = 1 if n == 2 else n * (n - 1) // 2
                label = 'STRAIGHT QIN' if n == 2 else f'BOX QIN-{n}'
                _ticket_add_bet(bets, label, 'QUINELLA', legs, combos, 'recommender')
            elif rec == 'TRIO':
                combos = 1 if n == 3 else n * (n - 1) * (n - 2) // 6
                label = 'STRAIGHT TRIO' if n == 3 else f'BOX TRIO-{n}'
                _ticket_add_bet(bets, label, 'TRIO', legs, combos, 'recommender')

        # QIN PLACE BANKER — short-favorite races (top1 ≤ $1.8): WIN pays
        # barely above stake. Banker the fav with next 3 picks for 3
        # combos that hit when fav places + any banker places.
        # Mirrors JS at template.py:1947-1959 (moved server-side 2026-05-24
        # to make bet_ticket the single source of truth).
        top1 = horses[0] if horses else None
        if top1 and (top1.get('win_odds') or 99) <= 1.8 and len(horses) >= 4:
            bankers = horses[1:4]
            _ticket_add_bet(
                bets, 'QIN PLACE BANKER', 'QIN PLACE',
                [top1] + bankers, 3, 'qin_place_banker',
            )

        # BOX QIN-4 PROTECTION — always show when top-4 has horses outside
        # primary recommender QIN bets (catches the rank 3-4 spoiler placings).
        # Mirrors JS at template.py:1966-1980.
        top4 = horses[:4]
        if len(top4) == 4:
            # "primary" = the recommender's own QIN bets ONLY. Summing legs over
            # ALL bets swept in the QIN PLACE BANKER added 7 lines above, whose
            # legs are [top1] + horses[1:4] = exactly the top 4 — so on every
            # short-favourite race has_non_primary went False and the protection
            # bet was silently suppressed. QIN PLACE is a different pool, so
            # nothing covered it. Regression from the 2026-05-24 JS→server port
            # (cc699ca); the pre-port JS built primaryNums from the recommender
            # map alone.
            primary_nums = {leg['number']
                            for b in bets if b.get('reason') == 'recommender'
                            for leg in b.get('legs') or []}
            has_non_primary = any(h.get('number') not in primary_nums for h in top4)
            if has_non_primary or not bets:
                _ticket_add_bet(bets, 'BOX QIN-4 PROTECTION', 'QUINELLA',
                                top4, 6, 'qin_protection')

        top5 = horses[:5]
        if len(top5) == 5:
            tier = (race.get('conviction_tier') or 'SKIP').upper()
            top1_odds = top5[0].get('win_odds') or 0
            field = len(horses)
            # F4 SAVER — 1-unit FIRST 4 ANY-ORDER on the model top-4. The
            # HKJC FIRST 4 pool is any-order top-4 (per
            # campaign.hkjc.com/en/qtt-f4/mergedpool_oddscal.aspx); QUARTET
            # is the exact-order product. Earlier copy strings said "exact
            # order" which contradicted the pool we actually bet — corrected
            # 2026-05-25 (GPT-5.5 R12 #8). Same gates as F4 BOX-5
            # EXCEPT heavy-fav-skip is RETAINED (exact-order F4 collapses
            # hard on heavy favs; the box's larger payout-on-hit absorbs
            # the cost, the saver's 1-combo stake doesn't). Mirrors JS at
            # template.py:1988-2018 (moved server-side 2026-05-24).
            f4_saver_skip = (
                (top1_odds > 0 and top1_odds < 2.5) or
                (tier not in ('EDGE', 'HIGH', 'MED')) or
                (field < 8)
            )
            if not f4_saver_skip:
                _ticket_add_bet(bets, 'F4 SAVER', 'FIRST 4', top4, 1,
                                'f4_saver_gate')

            # Heavy-fav-skip dropped 2026-05-24 per user: at flat $10/combo
            # the downside is bounded; user prefers visible-and-staked over
            # the "phantom-hit" recap inconsistency (R8/R11 2026-05-24).
            triple_gate_skip = (
                (tier not in ('EDGE', 'HIGH', 'MED')) or
                (field < 8)
            )
            if not triple_gate_skip:
                _ticket_add_bet(bets, 'BOX TRIO-5', 'TRIO', top5, 10, 'main_trio5_gate')
                _ticket_add_bet(bets, 'F4 BOX-5', 'FIRST 4', top5, 5, 'main_f4_gate')

            # Single source of truth for BOX TRIO-6 take/skip + chaos badge.
            # Both JS render sites (detail panel + WA copy) used to recompute
            # this independently and had already drifted (WA added
            # top_2_share<0.40, server hadn't), so the staked ticket disagreed
            # with the displayed verdict. Now the decision lives once in
            # _compute_trio6_decision() and rides on race_data['trio6_decision'].
            # Locked by tests/test_audit_2026_05_29.py::test_trio6_decision_single_source.
            decision = _compute_trio6_decision(
                horses=horses, tier=tier, field=field, top1_odds=top1_odds,
                top_2_share=race.get('top_2_share') or 0.0,
            )
            race['trio6_decision'] = decision
            if decision['take'] and len(horses) >= 6:
                # Apply rank-6 swap (manual override or auto longshot pick)
                # so the ticket's legs match what UI/WA renders. Otherwise
                # validate_hk --replay scores horses[:6] while the user
                # staked the swapped top-6 (GPT-5.5 Finding #2, 2026-05-25).
                top6_swapped, swap_meta = _pick_trio6_top6(
                    horses,
                    race.get('manual_trio6_horse'),
                    race.get('manual_trio6_reason') or '',
                )
                _ticket_add_bet(
                    bets, 'BOX TRIO-6', 'TRIO', top6_swapped, 20, 'main_trio6_gate',
                    swap=swap_meta,
                )

    # EDGE PLACE (paper) — an EDGE race flags a single-horse MISPRICING, which
    # is a different axis from ticket confidence; on a flat field every QIN/box
    # gate can SKIP, leaving the "top tier" race with no bet at all. The edge
    # horse historically PLACES far more than it WINS (production: 3/5 placed,
    # 0 won), so when the ticket is otherwise empty, surface a paper PLACE bet
    # on it. Tagged `paper` -> displayed/WA-copied/replay-tracked for ROI but
    # NOT staked against the budget and NOT counted toward the QIN-scoped
    # stop-loss (bankroll_guard._is_counted_label excludes PLACE). User
    # decision 2026-06-24.
    if not bets and (race.get('conviction_tier') or '').upper() == 'EDGE':
        edge_num = race.get('edge_horse_number')
        edge_h = next((h for h in horses if h.get('number') == edge_num), None)
        if edge_h is None:
            edge_h = next((h for h in horses if h.get('edge_fires')), None)
        if edge_h and (edge_h.get('win_odds') or 0) > 0:
            _ticket_add_bet(bets, 'EDGE PLACE', 'PLACE', [edge_h], 1,
                            'edge_place_paper')
            for b in bets:
                if b.get('reason') == 'edge_place_paper':
                    b['paper'] = True

    total_combos = sum(b['combos'] for b in bets)
    unit = (amount / total_combos) if amount > 0 and total_combos > 0 else 0.0
    # HKJC minimum bet is HK$10 — WIN / PLACE / QUINELLA / QUINELLA PLACE are
    # not accepted below $10, so a $5 unit produced un-placeable tickets
    # (user-reported 2026-06-08). Floor every pool at HK$10 and round to $10
    # increments, matching the user's actual flat-$10/combo betting. When the
    # budget can't fund every combo at that floor, the ticket allocation
    # EXCEEDS `amount`. Honest disclosure preferred over silent overspend:
    # the ticket carries `over_budget` + `over_amount` so the display can
    # show "$100 budget / $150 allocated (+$50 over)" instead of misleading
    # the user (GPT-5.5 R12 #2 2026-05-25).
    unit_rounded = max(10.0, (unit // 10.0) * 10.0) if unit > 0 else 0.0
    allocated = 0.0
    for b in bets:
        b['unit_stake'] = float(unit_rounded)
        b['amount'] = float(b['combos'] * unit_rounded)
        allocated += b['amount']
    over_amount = max(0.0, allocated - amount) if amount > 0 else 0.0

    # Paper bets (EDGE PLACE) carry a flat HK$10 display stake — the user's
    # standard flat unit — but do NOT consume the real budget (`allocated`),
    # since they are reference/tracking only, not part of the staked ticket.
    for b in bets:
        if b.get('paper'):
            b['unit_stake'] = 10.0
            b['amount'] = float(b['combos'] * 10.0)
    # Real allocated budget excludes paper bets — recompute defensively so a
    # paper bet can never inflate allocated_amount (e.g. a <4-horse field where
    # the first loop assigned it a nonzero unit_rounded before the override).
    allocated = sum(b['amount'] for b in bets if not b.get('paper'))

    return {
        'version': 1,
        'race_date': race_date,
        'race_number': race.get('race_number'),
        'generated_at': generated_at,
        'model_version': model_version,
        'conviction_tier': race.get('conviction_tier'),
        'has_edge': bool(race.get('has_edge')),
        'odds_age_minutes': race.get('odds_age_minutes'),
        'bankroll': race.get('bankroll'),
        'suggested_stake': stake,
        'total_combos': int(total_combos),
        'unit_stake': float(unit_rounded),
        'allocated_amount': float(allocated),
        # Budget over/under disclosure — `over_budget=True` means the HK$10
        # unit-stake floor pushed allocated above the suggested budget.
        # WA + visual ticket surface this so the user sees the real spend.
        'over_budget': bool(over_amount > 0),
        'over_amount': float(over_amount),
        'bets': bets,
    }


def build_race_data(race, analyses, race_date, db, odds_db, analyzer=None):
    """Build the JSON-serializable race data dict from analyses.

    Shared between generator and monitor to avoid duplication.
    Returns race_data dict with horses, speed_map, and exotic_value.
    analyzer: optional CombinedRaceAnalyzer instance — used for ML integrity check (G6).
    """
    # Reset per-run caches so backtests across multiple race dates don't
    # bleed stale jockey/trainer rates between dates.
    from ml.features import clear_feature_caches
    clear_feature_caches()
    race_going = race.going or ''
    race_distance = int(race.distance) if race.distance and str(race.distance).isdigit() else 1200

    # Speed map
    speed_map_input = [
        {'number': h.number, 'name': h.name, 'draw': h.draw,
         'running_style': getattr(h, 'running_style', ''),
         'last_6': h.last_6}
        for h in race.horses
    ]
    speed_map = PaceAnalyzer.generate_speed_map(speed_map_input, race_distance)

    # Odds movement
    odds_summary = odds_db.get_movement_summary(race_date, race.race_number)

    # Loop-invariant: same venue+distance for all horses in this race
    dist_record = db.get_distance_record(race.venue or '', race_distance)

    race_data = {
        "race_number": race.race_number,
        "venue": race.venue,
        "time": race.time,
        "distance": race.distance,
        "class_type": race.class_type,
        "going": race.going,
        "course": race.course,
        "horses": [],
    }

    for i, a in enumerate(analyses):
        h = a.horse
        ml_prob = a.ml_prediction.win_probability if a.ml_prediction else 0
        # Raw (pre-calibration) ML prob — feeds pure_prob per CLAUDE.md spec.
        # Mixing calibrated ML into pure_prob biased edge_pp toward smaller
        # magnitudes; isotonic calibration shifts toward base rate, attenuating
        # strong ML signals on top picks.
        raw_ml_prob = a.ml_prediction.raw_win_probability if a.ml_prediction else 0
        place_prob = a.ml_prediction.place_probability if a.ml_prediction else 0
        calibrated_prob = a.combined_probability if a.combined_probability else 0

        going_form = db.get_horse_going_form(h.name, race_going, race_date)
        jh_stats = db.get_jockey_horse_stats(h.jockey, h.name, race_date)
        dist_pb = db.get_distance_best_time(h.name, race_distance, race_date)
        # Same-format history: prior runs at this exact venue+distance
        # so the dashboard can show "this horse has run here before, here is what happened"
        format_history = db.get_horse_format_history(
            h.name, race.venue or '', race_distance,
            going='',  # going is variable; loose match for now
            before_date=race_date, limit=5,
        )
        # Jockey strike rate (Bayesian-smoothed, prior-only). Reuse the same
        # cached helper the ML feature pipeline uses so we don't re-query.
        from ml.features import FeatureEngineering
        jockey_strike_rate = FeatureEngineering.calculate_jockey_win_rate(
            h.jockey or '', [], conn=db.conn, before_date=race_date,
        )
        jockey_form_30d = FeatureEngineering.calculate_jockey_form_window(
            h.jockey or '', conn=db.conn, before_date=race_date, window_days=30,
        )

        horse_odds_move = odds_summary.get(h.number, {})
        is_steam, steam_severity, steam_label = OddsMovementAnalyzer.detect_steam_move(
            horse_odds_move.get('opening', 0), h.win_odds or 0
        )
        is_cold, cold_label = OddsMovementAnalyzer.detect_cold_horse(
            horse_odds_move.get('opening', 0), h.win_odds or 0, a.rule_score
        )

        race_data["horses"].append({
            "rank": i + 1,
            "number": h.number,
            "name": h.name,
            "jockey": h.jockey,
            "trainer": h.trainer,
            "draw": h.draw,
            "weight": h.weight,
            "gear": h.gear,
            "rating": h.rating,
            "win_odds": h.win_odds,
            "place_odds": h.place_odds,
            "last_6": h.last_6,
            "age": h.age,
            "horse_weight": h.horse_weight,
            "days_since_last_run": h.days_since_last_run,
            "rating_change": h.rating_change,
            "rule_score": a.rule_score,
            "ml_prob": ml_prob,
            "raw_ml_prob": raw_ml_prob,
            "place_prob": place_prob,
            "calibrated_prob": calibrated_prob,
            "expected_return": a.expected_return,
            "bet_recommendation": a.bet_recommendation,
            "confidence": a.combined_confidence,
            "reasoning": a.reasoning,
            "going_form": going_form,
            "jockey_horse_stats": jh_stats,
            "distance_pb": dist_pb,
            "distance_record": dist_record,
            "format_history": format_history,
            "jockey_strike_rate": round(jockey_strike_rate, 4),
            "jockey_form_30d": round(jockey_form_30d, 4),
            "odds_movement": horse_odds_move,
            "steam_move": {"is_steam": is_steam, "severity": steam_severity, "label": steam_label},
            "cold_horse": {"is_cold": is_cold, "label": cold_label},
        })

    race_data["speed_map"] = speed_map

    # === Edge math (pure_prob, market_prob, edge fields, edge_fires) ===
    from betting.edge_calculator import (
        compute_pure_probs, compute_market_probs, compute_edge_fields,
        apply_edge_gates, EdgeGates,
    )
    try:
        from ml.integrity import check_model_integrity
    except ImportError:
        check_model_integrity = None

    # Use raw_ml_prob (pre-calibration) for pure_prob — CLAUDE.md spec compliance.
    # Falls back to ml_prob if raw is missing (legacy JSON, model load failure).
    horses_for_edge = [{
        'rule_score': h['rule_score'],
        'ml_prob': h.get('raw_ml_prob') or h['ml_prob'],
        'win_odds': h['win_odds'] or 0,
    } for h in race_data['horses']]

    pure_probs_list = compute_pure_probs(horses_for_edge, field_size=len(horses_for_edge))
    market_probs_list = compute_market_probs(horses_for_edge)

    # Model integrity — used by G6.
    integrity_ok = False
    try:
        if check_model_integrity is not None:
            ml_model = getattr(analyzer, 'ml_analyzer', None)
            ml_model = getattr(ml_model, 'ml_model', None) if ml_model else None
            metadata = getattr(ml_model, '_metadata', {}) if ml_model else {}
            check_model_integrity(metadata)
            integrity_ok = True
    except Exception as e:
        if getattr(analyzer, 'fail_closed_ml', False):
            raise RuntimeError(
                f"Dashboard ML integrity check failed while building "
                f"race {race.race_number}: {type(e).__name__}: {e}"
            ) from e
        integrity_ok = False

    # Persist the just-fetched odds before computing age — otherwise
    # _compute_odds_age_minutes returns None (no snapshot in DB) and G5
    # fails for every EDGE candidate even though the dashboard is showing
    # the freshest possible odds. This snapshot is only used downstream by
    # the EDGE freshness gate; nothing else reads it on the predict path.
    # Audit fix 2026-05-17.
    # Anti-poison: skip the snapshot if the race already has results in
    # race_results — HKJC post-race odds (final SP / settled) would
    # otherwise overwrite the legitimate pre-race odds we captured earlier
    # and corrupt any future backtest that reads odds_snapshots.
    # 2026-05-25 (GPT-5.5 Finding #3): also skip when the race scheduled
    # post-time has passed for TODAY's card — closes the 1-5 min window
    # between race off and results_fetcher writing race_results. Date-aware
    # so historical/future regen does not falsely treat past-time-of-day
    # races as off'd (same pattern as _merge_with_existing L5).
    try:
        race_finished = False
        if db is not None and hasattr(db, 'conn'):
            row = db.conn.execute(
                "SELECT 1 FROM race_results WHERE race_date=? AND race_number=? LIMIT 1",
                (race_date, race.race_number),
            ).fetchone()
            race_finished = bool(row)
        race_offed = False
        race_time_str = getattr(race, 'time', '') or ''
        today_str = datetime.now().strftime('%Y/%m/%d')
        if race_time_str and race_date == today_str:
            try:
                hh, mm = race_time_str.split(':')[:2]
                now = datetime.now()
                post_minutes = int(hh) * 60 + int(mm)
                now_minutes = now.hour * 60 + now.minute
                race_offed = now_minutes >= post_minutes
            except (ValueError, AttributeError):
                pass
        if not race_finished and not race_offed:
            odds_horses = [
                {'number': h.number, 'name': h.name,
                 'win_odds': h.win_odds or 0, 'place_odds': getattr(h, 'place_odds', 0) or 0}
                for h in race.horses if (h.win_odds or 0) > 0
            ]
            if odds_horses:
                odds_db.record_snapshot(race_date, race.race_number,
                                        odds_horses, source='generator')
    except Exception:
        pass  # snapshot failure must not block prediction generation

    # G5 odds freshness — compute once for the race, reuse per-horse
    odds_age_min = _compute_odds_age_minutes(odds_db, race_date, race.race_number)

    gates = EdgeGates()
    has_edge_race = False
    for i, h in enumerate(race_data['horses']):
        pure = pure_probs_list[i]
        market = market_probs_list[i]
        odds = h['win_odds'] or 0
        fields = compute_edge_fields(pure, market, odds)

        model_rank = h['rank']  # already 1-indexed in the horses loop

        ok, _failed = apply_edge_gates(
            ev_win=fields['ev_win'], pure_prob=pure, edge_pp=fields['edge_pp'],
            model_rank=model_rank, odds_age_min=odds_age_min,
            integrity_ok=integrity_ok, gates=gates,
        )

        h['pure_prob'] = round(pure, 4)
        h['market_prob'] = round(market, 4)
        h['edge_pp'] = round(fields['edge_pp'], 2)
        h['ev_win'] = round(fields['ev_win'], 3)
        h['overlay'] = round(fields['overlay'], 3)
        h['edge_fires'] = ok
        h['model_rank'] = model_rank
        if ok:
            has_edge_race = True

    race_data['has_edge'] = has_edge_race
    race_data['odds_age_minutes'] = odds_age_min

    # Manual TRIO-6 rank-6 override — tactical hedge when racing-API form/RPR
    # signal beats the auto rank-6 swap (e.g., auto-swap picks horse with
    # 168-day layoff). Surface as race-level fields so the template can apply
    # the override without recomputing form data client-side.
    _overrides = _load_manual_overrides()
    _race_overrides = (_overrides.get(race_date) or {}).get(str(race.race_number)) or {}
    if _race_overrides.get('trio6_horse'):
        race_data['manual_trio6_horse'] = _race_overrides['trio6_horse']
        race_data['manual_trio6_reason'] = _race_overrides.get('reason', '')

    # Identify which horse fired EDGE — alert text was misleadingly hardcoded
    # to "the top pick", but G4 allows max_model_rank=2 so rank-2 can fire EDGE
    # while rank-1 doesn't. Surface the lowest-rank firing horse so the
    # template can name it correctly.
    edge_firers = [h for h in race_data['horses'] if h.get('edge_fires')]
    if edge_firers:
        edge_firers.sort(key=lambda h: h['rank'])
        eh = edge_firers[0]
        race_data['edge_horse_rank'] = eh['rank']
        race_data['edge_horse_number'] = eh['number']
        race_data['edge_horse_name'] = eh['name']

    # === Post-bet analytics: Harville dividend + conviction tier + EV ===
    from betting.dividends import (
        estimate_quinella_dividend, quinella_bet_ev, normalize_market_probs
    )
    horses_json = race_data['horses']
    qin_indices = [i for i, h in enumerate(horses_json) if h['bet_recommendation'] == 'QIN']
    win_odds_list = [h.get('win_odds') or 0 for h in horses_json]
    model_probs = [h.get('calibrated_prob') or 0 for h in horses_json]
    market_probs = normalize_market_probs(win_odds_list)

    # Conviction tier (replaces strategy_confidence — spec §4)
    from betting.conviction import assign_conviction_tier, detect_chaos
    sorted_probs = sorted(model_probs, reverse=True)
    top_2_share = sum(sorted_probs[:2]) if len(sorted_probs) >= 2 else 0
    top_3_share = sum(sorted_probs[:3]) if len(sorted_probs) >= 3 else 0
    race_data['conviction_tier'] = assign_conviction_tier(
        has_edge=race_data['has_edge'],
        top_2_share=top_2_share,
        top_3_share=top_3_share,
    )
    race_data['top_2_share'] = round(top_2_share, 4)
    race_data['top_3_share'] = round(top_3_share, 4)

    # CHAOS no-bet override (sits above SKIP). Empirically: soft-going races
    # returned −10.9% ROI vs +34.8% on clean going across 10 backfilled meetings
    # (Bayesian posterior 0.996). Filter suppresses ALL bets — not just exotics.
    is_chaos, chaos_reason = detect_chaos(
        going=race_data.get('going', ''),
        horses=horses_json,
        class_type=race_data.get('class_type', ''),
    )
    if is_chaos:
        # Preserve the pre-chaos intent BEFORE the wipe so
        # scripts/stake_ab._reconstruct_bets can build the flat_min_pre_chaos
        # counterfactual — the whole point of that forward variant. The override
        # sets tier=CHAOS and nulls every bet_recommendation, which otherwise
        # leaves the reconstruction with nothing to read, silently making the
        # CHAOS-filter validation a no-op (5-expert review 2026-07-15).
        race_data['pre_chaos_conviction_tier'] = race_data.get('conviction_tier')
        race_data['conviction_tier'] = 'CHAOS'
        race_data['chaos_reason'] = chaos_reason
        for h in horses_json:
            h['pre_chaos_bet_recommendation'] = h.get('bet_recommendation')
            h['bet_recommendation'] = None

    # Quinella dividend estimate + EV (only when QIN bet is active).
    # Re-derive qin_indices from the CURRENT recommendations: the snapshot taken
    # earlier predates the CHAOS override, which nullifies all bet_recommendations
    # — without this, a CHAOS (no-bet) race would still compute+display QIN EV.
    qin_indices = [i for i, h in enumerate(horses_json)
                   if h.get('bet_recommendation') == 'QIN']
    if len(qin_indices) >= 2:
        pair_combos = list(itertools.combinations(qin_indices, 2))
        # Edge/EV math uses pure_prob (pre-market-blend), NOT calibrated_prob —
        # else our P(hit) is dragged toward the market and the EV is mechanically
        # understated (CLAUDE.md "edge math uses pure_prob"). model_probs above
        # stays calibrated because it drives the conviction-tier shares, which
        # were tuned on calibrated probabilities.
        pure_probs = [h.get('pure_prob') or 0 for h in horses_json]
        ev_result = quinella_bet_ev(pure_probs, market_probs, pair_combos)
        race_data['qin_bet_ev'] = {
            'n_combos': len(pair_combos),
            'total_stake': ev_result['total_stake'],
            'expected_profit': round(ev_result['expected_profit'], 2),
            'ev_pct': round(ev_result['ev_pct'], 1),
            'hit_prob': round(ev_result['hit_prob'], 4),
            'avg_dividend': round(ev_result['avg_dividend'], 2),
        }
        # Top pair dividend — prefer real HKJC payout if race has results
        top_div = None
        top_div_source = 'harville'
        if db and race.date and race.race_number:
            # If results already in, actual_winner + 2nd give us the real pair
            try:
                finishers = db.conn.execute(
                    "SELECT horse_number, finish_position FROM race_results "
                    "WHERE race_date = ? AND race_number = ? AND finish_position IN (1, 2)",
                    (race.date, race.race_number)
                ).fetchall()
                if len(finishers) == 2:
                    h1, h2 = finishers[0][0], finishers[1][0]
                    real_div = db.get_quinella_dividend(race.date, race.race_number, h1, h2)
                    if real_div:
                        top_div = real_div
                        top_div_source = 'real'
            except Exception:
                pass
        if top_div is None:
            # estimate_quinella_dividend returns a per-$1 figure; the real-payout
            # path (db.get_quinella_dividend) returns the raw per-HK$10 HKJC
            # declaration. Scale the estimate ×10 so the DISPLAYED dividend is on
            # the same per-$10 basis punters expect (was silently jumping 10×
            # when real results replaced the estimate).
            est = estimate_quinella_dividend(market_probs, qin_indices[0], qin_indices[1])
            top_div = est * 10 if est else None
        race_data['qin_top_pair_dividend'] = round(top_div, 2) if top_div else None
        race_data['qin_dividend_source'] = top_div_source

    # Exotic bet value detection
    if len(analyses) >= 4:
        # Edge math must use pure_prob (pre-blend); calibrated_prob is
        # market-blended, which mechanically shrinks edge vs the market.
        model_probs = [h['pure_prob'] for h in race_data['horses']]
        market_probs = []
        for h in race_data['horses']:
            wo = h.get('win_odds', 0)
            market_probs.append(1.0 / wo if wo and wo > 0 else 0)
        mp_total = sum(market_probs) or 1.0
        market_probs = [p / mp_total for p in market_probs]
        horse_nums = [h['number'] for h in race_data['horses']]
        place_probs = [h.get('place_prob', 0) for h in race_data['horses']]
        horse_draws = [h.get('draw', 0) for h in race_data['horses']]
        horse_styles = [
            next((r.split(':')[0] for r in (h.get('reasoning') or [])
                  if 'Leader' in r or 'Stalker' in r or 'Midfield' in r or 'Closer' in r), '')
            for h in race_data['horses']
        ]
        exotics = find_value_exotics(
            model_probs, market_probs, horse_nums, min_edge=0.10,
            place_probs=place_probs, draws=horse_draws, styles=horse_styles,
        )
        race_data["exotic_value"] = exotics

    return race_data


async def main():
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
        print(f"Invalid date format: {race_date}. Expected YYYY/MM/DD")
        sys.exit(1)

    logger.info(f"Generating predictions for {race_date}")

    try:
        ml_integrity = preflight_dashboard_ml()
    except Exception as e:
        logger.error(
            "Dashboard ML integrity preflight failed; refusing to generate "
            f"predictions: {type(e).__name__}: {e}"
        )
        raise SystemExit(2) from e
    logger.info(
        "Dashboard ML integrity passed: "
        f"model={ml_integrity.get('model_version')} "
        f"groups={ml_integrity.get('model_fold_groups')} "
        f"blend={ml_integrity.get('blend_version')}"
    )

    fetcher = HKJCDataFetcher()
    analyzer = CombinedRaceAnalyzer(fail_closed_ml=True)
    # Lazy-started Playwright browser sits on fetcher._pw_fetcher. Without an
    # explicit close() at the end the chromium subprocess lingers and stacks
    # across shadow runs that immediately follow (audit fix 2026-05-17).
    try:
        await _run_generation(fetcher, analyzer, race_date)
    finally:
        await fetcher.close()


async def _run_generation(fetcher, analyzer, race_date: str):

    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)

    all_races = []

    db = HorseDatabase()
    odds_db = OddsDatabase()

    # Fetch all race cards concurrently (was sequential 1→11 — the main
    # generate_predictions bottleneck). analyze/build stay sequential below
    # since they touch the DB; only the network fetch is parallelized.
    # Detect venue once up front — otherwise all 11 get_race_card calls each
    # fire their own detect_venue HTTP request and could disagree mid-gather.
    logger.info("Fetching all race cards concurrently...")
    detected_venue = fetcher.api.detect_venue(race_date)
    fetched = await asyncio.gather(
        *(fetcher.get_race_card(race_date, rn, venue=detected_venue) for rn in range(1, 12)),
        return_exceptions=True,
    )

    backfill_meeting_going(fetched)

    for race_no, race in enumerate(fetched, start=1):
        # Skip, do NOT break: all 11 cards are already fetched concurrently, so a
        # transient miss on one race must not drop every later race from the
        # generated dashboard (non-existent races on a <11-race card just skip).
        if isinstance(race, Exception):
            logger.warning(f"Race {race_no} fetch failed: {race} — skipping")
            continue
        if not race or not race.horses:
            logger.info(f"No data for Race {race_no} — skipping")
            continue

        analyses = analyzer.analyze_race(race)
        race_data = build_race_data(race, analyses, race_date, db, odds_db, analyzer=analyzer)
        all_races.append(race_data)
        logger.info(f"  Race {race_no}: {len(race.horses)} horses, bet: {analyses[0].bet_recommendation if analyses else 'N/A'}")

    # Fail loud on a total fetch miss: if EVERY race skipped (e.g. venue
    # detection returned the wrong course — CLAUDE.md's "wrong venue → empty
    # HTML" trap, most likely when re-rendering a historical HV meeting) we must
    # NOT write [] and clobber a good dashboard with green "Done!" logs.
    if not all_races:
        logger.error(
            f"No race data fetched for {race_date} — every race skipped. Venue "
            "detection may have returned the wrong course (HV meeting fetched as "
            "ST?). Refusing to overwrite existing dashboard/JSON with empty output."
        )
        raise SystemExit(3)

    date_str = race_date.replace('/', '')
    json_file = os.path.join(results_dir, f"race_data_{date_str}.json")

    # Merge: preserve result fields from prior run (e.g. actual_position, accuracy)
    all_races = _merge_with_existing(json_file, all_races, race_date=race_date)

    # Tier stats — load from cache (refreshed on results ingestion, Task 8)
    from betting.tier_stats import load_cached_stats
    tier_stats = load_cached_stats()
    for r in all_races:
        r['tier_stats'] = tier_stats

    finalize_tickets_and_stakes(all_races, race_date, analyzer)

    # Alerts — diff current vs previous race_data file
    from betting.alerts import compute_alerts, AlertState, load_alert_state, save_alert_state
    prev_json = _load_previous_race_data(json_file)
    state = load_alert_state(race_date)
    alerts = compute_alerts(all_races, prev_json, state)
    save_alert_state(race_date, state)
    for r in all_races:
        r['alerts'] = [a for a in alerts if a['race'] == r['race_number']]

    _atomic_write(json_file, json.dumps(all_races, indent=2))

    # Log predictions to DB for longitudinal accuracy tracking.
    # model_ver was previously read from finalize_tickets_and_stakes' local
    # scope (worked when this lived inline); after the R13 F2 extraction
    # the scope is no longer shared — re-derive here from the analyzer.
    model_ver = (analyzer.ml_analyzer.ml_model.model_version
                 if hasattr(analyzer, 'ml_analyzer') else '')
    db.log_predictions(race_date, all_races, model_version=model_ver)

    # Snapshot 60-day accuracy for the dashboard's "Recent accuracy" panel +
    # WhatsApp tier-trust footer. Safe-fallback to None when no results yet.
    try:
        accuracy_summary = db.get_accuracy_summary(days=60)
        if 'message' in accuracy_summary:
            accuracy_summary = None
    except Exception:
        accuracy_summary = None

    db.close()

    logger.info(f"Saved predictions to {json_file}")
    logger.info(f"Total races: {len(all_races)}")

    json_data = json.dumps(all_races)
    # Prevent </script> in JSON from breaking the <script> block (XSS defense)
    json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Detect venue from race data
    venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

    # Load EDGE stats (live count from track_edge.py — replaces the
    # hardcoded N=4 banner text). Silent fallback to None if file missing.
    edge_stats = None
    try:
        es_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                'tracking_data', 'edge_stats.json')
        if os.path.exists(es_path):
            with open(es_path) as f:
                edge_stats = json.load(f)
    except (json.JSONDecodeError, IOError):
        pass

    html = generate_html(json_data, now, race_date, venue,
                          accuracy_summary=accuracy_summary,
                          edge_stats=edge_stats)

    html_file = os.path.join(results_dir, f"racing_dashboard_{date_str}.html")
    latest_file = os.path.join(results_dir, "racing_dashboard.html")
    _atomic_write(html_file, html)
    _atomic_write(latest_file, html)

    logger.info(f"Saved dashboard to {html_file}")
    logger.info(f"Saved latest dashboard to {latest_file}")
    logger.info("Done!")


if __name__ == '__main__':
    asyncio.run(main())
