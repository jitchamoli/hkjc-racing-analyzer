"""Forward-validation A/B for staking strategies (same picks, different sizes).

User bets HK$10/combo (HKJC minimum) by default — see memory
[[user-stake-habit-10hkd-minimum]]. The dashboard recommends production
tier-ramped stakes that the user typically ignores. This script logs per-race
P&L under multiple staking variants so the user can see which strategy would
have won over time, given the SAME production picks.

Unlike the model shadows (shadow_generate/evaluate), staking variants don't
change picks — they just re-size the bets the production recommender already
chose. No re-fetch from HKJC needed; reads only results/race_data_*.json
(post-merge with results) + race_dividends DB.

Variants (extensible — add to VARIANTS dict to log another):
  flat_min    — HK$10 per combo, flat (user's actual habit; the baseline)
  production  — current code: tier ramp 3%/2%/1%/0.5% × bankroll, split
                across firing bets by combo-share (mirrors template.py)
  flat_1pct   — 1% × bankroll for every tier, split by combo-share
  invert      — 0.5/1/2/3% ramp (high payouts → bigger stake), split

Per-race per-variant rows append to tracking_data/stake_ab_log.csv.

Usage:
    python3 -m scripts.stake_ab '2026/05/20'        # log one meeting
    python3 -m scripts.stake_ab --report            # cumulative + Bayesian
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from betting.dividends import hkjc_declared_to_unit
from betting.dead_heat import (
    group_by_position, winning_combo_entries, winning_name_sets,
)

# Date the CHAOS no-bet filter shipped. A ticket frozen on or after this date
# already encodes the chaos decision, so re-deriving it here (from the MUTABLE
# `going` field) would delete bets production actually staked. Older tickets
# predate the filter and still need the retrospective overlay.
CHAOS_FILTER_SHIPPED = '2026/05/23'

REPO_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = REPO_ROOT / 'tracking_data' / 'stake_ab_log.csv'
DEFAULT_BANKROLL = float(getattr(config, 'BANKROLL', 10000.0))

# Variant definitions. Extend by adding a new entry — the rest is generic.
# `apply_chaos`: when False, the CHAOS soft-going filter is BYPASSED for this
# variant's bet selection. This gives us the counterfactual baseline needed
# to forward-validate the CHAOS filter at N≥6 meetings (Bayesian per-meeting
# verdict via cumulative ROI delta vs production).
VARIANTS = {
    'flat_min':   {'mode': 'flat_per_combo', 'stake_per_combo': 10.0,
                   'skip_tiers': set(), 'apply_chaos': True,
                   'label': 'flat HK$10/combo (user habit)'},
    'flat_min_pre_chaos': {
                   'mode': 'flat_per_combo', 'stake_per_combo': 10.0,
                   'skip_tiers': set(), 'apply_chaos': False,
                   # Counterfactual: what flat_min would have done WITHOUT
                   # the CHAOS filter shipped 2026-05-23. After N≥6 forward
                   # meetings where CHAOS fires, compare cumulative ROI vs
                   # flat_min via bayesian_verdict to validate/kill the filter.
                   'label': 'flat HK$10/combo, CHAOS filter OFF'},
    'flat_min_skip_aware': {
                   'mode': 'flat_per_combo', 'stake_per_combo': 10.0,
                   # Mirrors production's implicit "no bet on SKIP-tier races"
                   # without copying its tier ramp. Isolates whether
                   # production's edge is the RAMP or the SKIP filter.
                   'skip_tiers': {'SKIP'}, 'apply_chaos': True,
                   'label': 'flat HK$10/combo + skip SKIP-tier'},
    'straight_qin_only': {
                   'mode': 'flat_per_combo', 'stake_per_combo': 10.0,
                   'skip_tiers': set(), 'apply_chaos': True,
                   # Collapse every WIN-quinella bet to a single 1-combo
                   # STRAIGHT QIN and stake nothing on other pools. Holdout
                   # diagnostic 2026-08-03 (634 races, real dividends): box
                   # widening destroys ROI monotonically — straight -1.0%,
                   # box-3 -12.9%, box-4 -15.9%, box-5 -18.6%, because each
                   # widening adds combos faster than hit-rate x dividend.
                   # Paired bootstrap P(straight beats box-4) = 0.899: below
                   # the 0.95 ship bar, hence a forward-logged variant rather
                   # than a production gate change. Stake matches flat_min so
                   # the comparison isolates STRUCTURE, not stake size.
                   'qin_straight_only': True,
                   'label': 'straight QIN only (1 combo)'},
    'production': {'mode': 'race_ramp_split',
                   'ramp': {'EDGE': 0.030, 'HIGH': 0.020, 'MED': 0.010,
                            'LOW': 0.005, 'SKIP': 0.000},
                   'skip_tiers': set(), 'apply_chaos': True,
                   'label': 'production ramp 3/2/1/0.5%'},
    'flat_1pct':  {'mode': 'race_ramp_split',
                   'ramp': {'EDGE': 0.010, 'HIGH': 0.010, 'MED': 0.010,
                            'LOW': 0.010, 'SKIP': 0.000},
                   'skip_tiers': set(), 'apply_chaos': True,
                   'label': 'flat 1% × bankroll'},
    'invert':     {'mode': 'race_ramp_split',
                   'ramp': {'EDGE': 0.005, 'HIGH': 0.010, 'MED': 0.020,
                            'LOW': 0.030, 'SKIP': 0.000},
                   'skip_tiers': set(), 'apply_chaos': True,
                   'label': 'inverted 0.5/1/2/3%'},
}

LOG_HEADERS = [
    'race_date', 'race_number', 'variant', 'bet_label',
    'combos', 'stake_hkd', 'hit', 'dividend_per_1', 'pnl_hkd',
]


# ---------------------------------------------------------------------------
# Firing-bet inference — mirrors dashboard/template.py:799-840
# ---------------------------------------------------------------------------

def firing_bets(race: dict, apply_chaos: bool = True) -> dict[str, int]:
    """Return {bet_label: combo_count} for all bets the user actually staked
    on this race.

    Ticket-first (preferred): reads `race.bet_ticket.bets` — the frozen
    server-side ticket that drove the dashboard, WA copy, and HKJC entries.
    This is the SAME source `validate_hk --replay` consumes, so stake_ab's
    ROI now matches replay exactly. Mirrors the validate_hk shift from
    synthetic-reconstruction to ticket-driven scoring (GPT-5.5 R9 2026-05-25).

    Legacy reconstruction (fallback): pre-bet_ticket JSONs reconstruct the
    fire list from horse-level `bet_recommendation` + production gates.
    This re-derivation drifts whenever production gates change after the
    JSON was saved — caught when stake_ab over-fired BOX TRIO-5/F4 BOX-5
    + missed BOX QIN-4 PROTECTION/F4 SAVER vs the actual ticket.

    apply_chaos: when True (default), suppresses ALL bets on soft-going
    races per the CHAOS filter shipped 2026-05-23. Set False to compute
    the pre-CHAOS counterfactual for forward-validation of the filter."""
    horses = race.get('horses', [])

    # The frozen ticket ALREADY encodes the chaos decision made at bet time, so
    # the production variants must never re-derive it. `going` is not a frozen
    # key: a mid-card going deterioration rewrites it while conviction_tier and
    # bet_ticket stay frozen, so re-running detect_chaos here DELETED races
    # production actually staked (2026/07/04 R1+R2: tier HIGH, HK$410 of frozen
    # bets, going since rewritten to 'Good To Yielding' — dropped from flat_min /
    # production / flat_1pct / invert while validate_hk --replay still scored
    # them). flat_min is the user's real-wallet baseline AND the denominator of
    # every Bayesian verdict here (audit 2026-07-12).
    #
    # The chaos COUNTERFACTUAL (apply_chaos=False) is the one caller that still
    # needs a reconstruction: if the ticket was chaos-suppressed at bet time it
    # is empty, and the legacy fallback below is the only way to recover what
    # would otherwise have been bet.
    bet_ticket = race.get('bet_ticket')
    has_ticket = isinstance(bet_ticket, dict) and 'bets' in bet_ticket

    # A ticket frozen by a build that ALREADY had the CHAOS filter needs no
    # retrospective overlay — the decision is baked in. Older tickets predate the
    # filter and DO need it re-derived, so the pre-CHAOS meetings stay comparable
    # to current production (that is what the overlay is for).
    ticket_date = ''
    if has_ticket:
        ticket_date = (bet_ticket.get('race_date')
                       or race.get('date') or race.get('race_date') or '')
    ticket_is_chaos_aware = bool(ticket_date) and ticket_date >= CHAOS_FILTER_SHIPPED

    def _from_ticket() -> dict:
        out: dict[str, int] = {}
        for b in bet_ticket.get('bets') or []:
            # Paper bets (EDGE PLACE) are reference-only and never staked. Every
            # other consumer excludes them (validate_hk, generator._counted_alloc,
            # template.py); counting them here handed the whole EDGE-race stake to
            # a $10 reference bet under the `production` variant.
            if b.get('paper'):
                continue
            label = (b.get('label') or '').strip()
            combos = int(b.get('combos', 0) or 0)
            if label and combos > 0:
                # If the same label appears twice (defensive — shouldn't
                # happen with current build_bet_ticket), take the larger
                # combo count rather than silently dropping one.
                out[label] = max(out.get(label, 0), combos)
        return out

    def _chaos_fires() -> bool:
        from betting.conviction import detect_chaos
        is_chaos, _ = detect_chaos(
            going=race.get('going', ''), horses=horses,
            class_type=race.get('class_type', ''),
        )
        return is_chaos

    if has_ticket and (ticket_is_chaos_aware or not apply_chaos):
        out = _from_ticket()
        # CHAOS counterfactual (apply_chaos=False): a ticket left EMPTY because
        # chaos fired at freeze time must be reconstructed to recover what would
        # have been bet WITHOUT the filter — otherwise flat_min_pre_chaos silently
        # equals flat_min and the filter reads as a validated no-op on the first
        # soft-going meeting (audit 2026-07-12 round 3). An empty ticket for any
        # non-chaos reason (SKIP tier, no edge), or any non-empty ticket, is
        # authoritative. Under apply_chaos=True the empty ticket is the correct
        # suppression and is returned as-is.
        if out or apply_chaos or not _chaos_fires():
            return out
        return _reconstruct_bets(race, horses)

    if apply_chaos and _chaos_fires():
        # Ticket predates the CHAOS filter (or there is none) — re-derive it so
        # those meetings compare apples-to-apples with current production.
        return {}

    if has_ticket:
        return _from_ticket()

    return _reconstruct_bets(race, horses)


def _reconstruct_bets(race: dict, horses: list) -> dict:
    """Legacy fallback (pre-bet_ticket JSONs, and the CHAOS counterfactual
    recovery): reconstruct the fire list from horse-level `bet_recommendation`
    + production gates. Applies every gate EXCEPT the CHAOS filter, so it is the
    honest "what would we have bet without chaos" for flat_min_pre_chaos."""
    bet_horse_counts = defaultdict(int)
    for h in horses:
        # On a CHAOS race build_race_data has nulled bet_recommendation and
        # stashed the original in pre_chaos_bet_recommendation — read that first
        # so the counterfactual sees what WOULD have been bet without the filter.
        b = (h.get('pre_chaos_bet_recommendation')
             or h.get('bet_recommendation') or '').strip()
        if b and b != 'SKIP':
            bet_horse_counts[b] += 1

    out: dict[str, int] = {}
    for bt, n in bet_horse_counts.items():
        if bt in ('WIN', 'BANKER WIN'):
            out[bt] = 1
        elif bt == 'PLACE':
            out['PLACE'] = 1
        elif bt in ('QIN', 'BOX QIN'):
            if n == 2:
                out['STRAIGHT QIN'] = 1
            elif n >= 3:
                out[f'BOX QIN-{n}'] = n * (n - 1) // 2
        elif bt == 'TRIO':
            if n == 3:
                out['STRAIGHT TRIO'] = 1
            elif n >= 4:
                out[f'BOX TRIO-{n}'] = n * (n - 1) * (n - 2) // 6

    horses_by_rank = sorted(horses, key=lambda h: h.get('rank') or 99)
    if (horses_by_rank
            and (horses_by_rank[0].get('win_odds') or 99) <= 1.8
            and len(horses_by_rank) >= 4):
        out['QIN PLACE BANKER'] = 3

    if len(horses) >= 5:
        # On a CHAOS race tier was overwritten to 'CHAOS'; the pre-chaos tier is
        # what the triple gate would have seen without the filter.
        tier = (race.get('pre_chaos_conviction_tier')
                or race.get('conviction_tier') or 'SKIP').upper()
        field_size = len(horses)
        triple_gate_skip = (tier not in ('EDGE', 'HIGH', 'MED') or
                            field_size < 8)
        if not triple_gate_skip:
            out['BOX TRIO-5'] = 10
            out['F4 BOX-5'] = 5

    return out


# ---------------------------------------------------------------------------
# Stake sizing per variant
# ---------------------------------------------------------------------------

# WIN-quinella pool labels. QIN PLACE BANKER is the QUINELLA PLACE pool and is
# deliberately NOT here — different pool, different dividend, not a substitute.
_WIN_QIN_LABELS = ('STRAIGHT QIN', 'BOX QIN-')


def restrict_to_straight_qin(fires: dict[str, int]) -> dict[str, int]:
    """Collapse every WIN-quinella bet to one 1-combo STRAIGHT QIN.

    Drops all other pools (TRIO / FIRST 4 / QUINELLA PLACE) entirely. Returns
    a NEW dict — the caller's `fires` is shared across variants in the scoring
    loop and must not be mutated.
    """
    has_qin = any(lbl == 'STRAIGHT QIN' or lbl.startswith('BOX QIN-')
                  for lbl in fires)
    return {'STRAIGHT QIN': 1} if has_qin else {}


def variant_stakes(variant: str, race: dict, fires: dict[str, int],
                   bankroll: float) -> dict[str, float]:
    """Return {bet_label: total_stake_HKD} for this variant on this race."""
    cfg = VARIANTS[variant]
    tier = (race.get('conviction_tier') or 'SKIP').upper()

    # Tier-level skip filter (applies to any variant). Used by
    # flat_min_skip_aware to mirror production's SKIP-on-SKIP-tier behavior
    # without copying its tier ramp.
    if tier in cfg.get('skip_tiers', set()):
        return {bt: 0.0 for bt in fires}

    if cfg['mode'] == 'flat_per_combo':
        per = cfg['stake_per_combo']
        return {bt: combos * per for bt, combos in fires.items()}

    # race_ramp_split: race-level stake from tier × bankroll, split by combo
    race_stake = bankroll * cfg['ramp'].get(tier, 0.0)
    if race_stake <= 0 or not fires:
        return {bt: 0.0 for bt in fires}
    total_combos = sum(fires.values())
    return {bt: round(race_stake * combos / total_combos, 2)
            for bt, combos in fires.items()}


# ---------------------------------------------------------------------------
# Hit detection + dividend lookup
# ---------------------------------------------------------------------------

def _finishers(race: dict) -> list[dict]:
    return sorted(
        [h for h in race.get('horses', []) if (h.get('actual_position') or 0) > 0],
        key=lambda h: h['actual_position'],
    )


def _top_picks(race: dict, n: int) -> list[dict]:
    return sorted(race.get('horses', []), key=lambda h: h.get('rank') or 99)[:n]


def _names(horses) -> set[str]:
    return {h['name'] for h in horses}


def _score_key(legs, finishers):
    """Hit-comparison key for ticket scoring: prefer horse_number, fall back
    to a normalized name when numbers are absent (legacy tickets/rows).

    Horse names can drift in casing/spelling between a frozen ticket leg and
    the prediction_log finisher row while the saddle number stays stable, so
    number-keyed comparison avoids false misses (audit 2026-06-08 #3). Numbers
    are stringified to neutralize int-vs-str storage drift; the name fallback
    is upper-cased + stripped."""
    legs_have = bool(legs) and all(l.get('number') is not None for l in legs)
    fin_have = bool(finishers) and all(h.get('number') is not None for h in finishers)
    if legs_have and fin_have:
        return lambda h: str(h.get('number')).strip()
    return lambda h: (h.get('name') or '').strip().upper()


def _lookup_dividend(conn, race_date: str, race_no: int, pool: str,
                     combo: str) -> float:
    """Look up a dividend from race_dividends. Returns 0.0 if absent.
    Dividends in the table are per HK$10; divide by 10 for per-$1 stake."""
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? "
        "AND race_number=? AND pool=? AND combination=?",
        (race_date, race_no, pool, combo),
    ).fetchone()
    div = hkjc_declared_to_unit(row[0]) if row else None
    return div if div is not None else 0.0


def hit_and_dividend(bet_label: str, race: dict, conn,
                     race_date: str) -> tuple[bool, float]:
    """Return (hit, dividend_per_$1_stake) for this bet on this race.
    Returns (False, 0.0) when no result/dividend data available."""
    finishers = _finishers(race)
    if not finishers:
        return False, 0.0

    # DEAD-HEAT AWARE (code review 2026-08-03). These used to be
    # `_names(finishers[:k])` — slicing a position-sorted list, which picks one
    # arbitrary horse out of a tied group and books a genuinely winning box as
    # a miss. HKJC pays EVERY tied combination, so a race can have more than
    # one winning pair/trio. Same enumeration validate_hk uses.
    win2 = winning_name_sets(finishers, 2)
    win3 = winning_name_sets(finishers, 3)
    win4 = winning_name_sets(finishers, 4)
    # PLACE asks "did our pick finish top-3", so a horse tied FOR 3rd counts —
    # filter on position rather than slicing.
    actual_top3 = {h['name'] for h in finishers
                   if (h.get('actual_position') or 99) <= 3}
    top1 = finishers[0]['name']
    top1_won = True  # we already have the winner

    rn = race['race_number']

    # WIN family — top pick wins
    if bet_label in ('WIN', 'BANKER WIN'):
        picks = _top_picks(race, 1)
        if not picks:
            return False, 0.0
        hit = picks[0]['name'] == top1
        odds = picks[0].get('win_odds') or 0  # already per $1
        return (hit, float(odds))

    # PLACE — top pick top-3. race_dividends pool='PLACE' rows are keyed
    # by single horse number (combination = '7' not '7,8'). Lookup the
    # placed pick's per-$1 dividend directly. Prior MVP returned 0.0 and
    # under-counted every PLACE hit (GPT-5.5 R8 Finding #2 2026-05-25).
    if bet_label == 'PLACE':
        picks = _top_picks(race, 1)
        if not picks:
            return False, 0.0
        pick = picks[0]
        hit = pick['name'] in actual_top3
        if not hit:
            return (False, 0.0)
        pick_num = pick.get('number')
        if not pick_num:
            return (True, 0.0)
        return (True, _lookup_dividend(conn, race_date, rn, 'PLACE', str(pick_num)))

    # QIN PLACE BANKER — pivot (top1) + bankers (top2-4); each (pivot, banker)
    # pair is one combo that pays when BOTH horses place top-3. Sum
    # per-$1 dividends across hit pairs — multi-hit cards compound.
    # Mirrors validate_hk._score_meeting_replay QPL branch.
    if bet_label == 'QIN PLACE BANKER':
        picks4 = _top_picks(race, 4)
        if len(picks4) < 4:
            return False, 0.0
        pivot = picks4[0]
        bankers = picks4[1:]
        if pivot['name'] not in actual_top3:
            return (False, 0.0)
        piv_num = pivot.get('number')
        if not piv_num:
            return (False, 0.0)
        div_sum = 0.0
        hit_any = False
        for bk in bankers:
            if bk['name'] in actual_top3:
                bk_num = bk.get('number')
                if not bk_num:
                    continue
                lo, hi = sorted([int(piv_num), int(bk_num)])
                d2 = _lookup_dividend(conn, race_date, rn,
                                      'QUINELLA PLACE', f"{lo},{hi}")
                if d2 > 0:
                    div_sum += d2
                    hit_any = True
        return (hit_any, div_sum)

    # QIN family — straight & box. `win2` holds EVERY winning pair (more than
    # one when a dead heat straddles 2nd), so a straight bet hits if our pair
    # is any of them and a box hits if it contains any of them.
    if bet_label == 'STRAIGHT QIN':
        picks = _top_picks(race, 2)
        hit = _names(picks) in win2
        div = race.get('qin_top_pair_dividend')
        unit_div = hkjc_declared_to_unit(div) if div else None
        return (hit, unit_div if unit_div is not None else 0.0)
    if bet_label.startswith('BOX QIN-'):
        n = int(bet_label.split('-')[1])
        box = _names(_top_picks(race, n))
        hit = any(w <= box for w in win2)
        div = race.get('qin_top_pair_dividend')
        unit_div = hkjc_declared_to_unit(div) if div else None
        return (hit, unit_div if unit_div is not None else 0.0)

    # TRIO family
    if bet_label == 'STRAIGHT TRIO':
        picks = _top_picks(race, 3)
        hit = _names(picks) in win3
        if len(finishers) < 3:
            return (hit, 0.0)
        combo = ','.join(str(h['number']) for h in
                         sorted(finishers[:3], key=lambda h: h['number']))
        return (hit, _lookup_dividend(conn, race_date, rn, 'TRIO', combo))
    if bet_label.startswith('BOX TRIO-'):
        n = int(bet_label.split('-')[1])
        box = _names(_top_picks(race, n))
        hit = any(w <= box for w in win3)
        if len(finishers) < 3:
            return (hit, 0.0)
        combo = ','.join(str(h['number']) for h in
                         sorted(finishers[:3], key=lambda h: h['number']))
        return (hit, _lookup_dividend(conn, race_date, rn, 'TRIO', combo))

    # F4 BOX-5 — a winning top-4 ⊂ our top-5
    if bet_label == 'F4 BOX-5':
        picks5 = _names(_top_picks(race, 5))
        hit = any(w <= picks5 for w in win4)
        if len(finishers) < 4:
            return (hit, 0.0)
        combo = ','.join(str(h['number']) for h in
                         sorted(finishers[:4], key=lambda h: h['number']))
        return (hit, _lookup_dividend(conn, race_date, rn, 'FIRST 4', combo))

    return False, 0.0


def _score_bet_from_ticket(bet: dict, race: dict, conn,
                            race_date: str) -> tuple[bool, float]:
    """Score a single frozen ticket bet using its pool + legs (NOT the
    label). Mirrors validate_hk._score_meeting_replay branch-for-branch
    so stake_ab and replay agree on every hit/miss + dividend lookup.

    Fixes three bugs in the old label-parsing hit_and_dividend (GPT-5.5
    R10 2026-05-25):
      - `BOX QIN-4 PROTECTION` crashed: int('4 PROTECTION') raised
        because `bet_label.split('-')[1]` ate the suffix.
      - `F4 SAVER` had no branch → silently returned (False, 0.0).
      - swapped `BOX TRIO-6` false-missed: scoring read rank-1..6 via
        _top_picks instead of the ticket's actual legs (which include
        the manual/auto rank-6 swap).
    """
    finishers = _finishers(race)
    if not finishers:
        return False, 0.0

    pool = (bet.get('pool') or '').strip()
    legs = bet.get('legs') or []
    if not legs:
        return False, 0.0

    _key = _score_key(legs, finishers)
    leg_ids = {_key(leg) for leg in legs if _key(leg)}
    # PLACE asks "did this leg finish top-3", so tied-for-3rd counts.
    actual_top3 = {_key(h) for h in finishers
                   if (h.get('actual_position') or 99) <= 3}
    top1 = _key(finishers[0])
    rn = race['race_number']

    pos_groups = group_by_position(finishers)

    def _score_pool(k: int, pool_name: str) -> tuple[bool, float]:
        """Sum EVERY winning k-combination our legs hold (dead-heat aware).

        HKJC declares a dividend per tied combination, so a box straddling a
        dead heat can collect more than one. Mirrors validate_hk._score_box.
        """
        combos = winning_combo_entries(pos_groups, k)
        won = [c for c in combos if {_key(e) for e in c} <= leg_ids]
        if not won or len(finishers) < k:
            return (False, 0.0)
        total = 0.0
        for combo in won:
            nums = sorted(int(e['number']) for e in combo
                          if e.get('number') is not None)
            if len(nums) != k:
                continue
            total += _lookup_dividend(conn, race_date, rn, pool_name,
                                      ','.join(str(n) for n in nums))
        return (True, total)

    if pool in ('WIN', 'BANKER WIN'):
        first_id = _key(legs[0]) if legs else None
        if first_id != top1:
            return (False, 0.0)
        winner_num = finishers[0].get('number')
        if winner_num is None:
            return (True, 0.0)
        return (True, _lookup_dividend(conn, race_date, rn, 'WIN', str(winner_num)))

    if pool == 'PLACE':
        placed_div = 0.0
        placed_any = False
        for leg in legs:
            if _key(leg) in actual_top3:
                ln_num = leg.get('number')
                if ln_num is None:
                    continue
                d2 = _lookup_dividend(conn, race_date, rn, 'PLACE', str(ln_num))
                if d2 > 0:
                    placed_div += d2
                    placed_any = True
        return (placed_any, placed_div)

    if pool == 'QUINELLA':
        return _score_pool(2, 'QUINELLA')

    if pool == 'TRIO':
        return _score_pool(3, 'TRIO')

    if pool == 'FIRST 4':
        return _score_pool(4, 'FIRST 4')

    if pool == 'QIN PLACE':
        # QPL BANKER ticket: legs[0] = pivot, legs[1:] = bankers. Each
        # (pivot, banker_i) pair pays when BOTH place top-3. Sum across
        # all hit pairs.
        if len(legs) < 2:
            return (False, 0.0)
        pivot = legs[0]
        if _key(pivot) not in actual_top3:
            return (False, 0.0)
        piv_num = pivot.get('number')
        if piv_num is None:
            return (False, 0.0)
        div_sum = 0.0
        hit_any = False
        for bk in legs[1:]:
            if _key(bk) in actual_top3:
                bk_num = bk.get('number')
                if bk_num is None:
                    continue
                lo, hi = sorted([int(piv_num), int(bk_num)])
                d2 = _lookup_dividend(conn, race_date, rn,
                                       'QUINELLA PLACE', f"{lo},{hi}")
                if d2 > 0:
                    div_sum += d2
                    hit_any = True
        return (hit_any, div_sum)

    return (False, 0.0)


def _pnl(stake: float, combos: int, hit: bool, div_per_1: float) -> float:
    """P&L formula: stake distributed evenly across combos; one combo wins.
       hit:  pnl = stake × (div / combos − 1)
       miss: pnl = −stake
    """
    if stake <= 0:
        return 0.0
    if not hit or combos <= 0 or div_per_1 <= 0:
        return -stake
    return stake * (div_per_1 / combos - 1.0)


# ---------------------------------------------------------------------------
# Per-meeting evaluation
# ---------------------------------------------------------------------------

def evaluate_date(race_date: str, bankroll: float) -> int:
    date_str = race_date.replace('/', '')
    json_path = REPO_ROOT / 'results' / f'race_data_{date_str}_with_results.json'
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run results_fetcher first.",
              file=sys.stderr)
        return 1

    payload = json.loads(json_path.read_text())
    races = payload['races'] if isinstance(payload, dict) else payload
    if not races:
        print(f"No races in {json_path}")
        return 1

    conn = sqlite3.connect(str(config.DB_PATH))

    rows_to_log = []
    print(f"Stake A/B for {race_date} — bankroll={bankroll:.0f} HKD")
    print(f"{'Race':<5} {'Variant':<12} {'Bet':<14} {'Combos':>6} "
          f"{'Stake':>8} {'Hit':>4} {'Div/$1':>8} {'P&L':>9}")
    print('-' * 78)

    for race in races:
        if not race.get('results_available'):
            continue
        rn = race.get('race_number')
        # Ticket-first scoring map: when bet_ticket exists, score each
        # bet from its pool + legs via _score_bet_from_ticket so the
        # numbers agree with validate_hk --replay. Falls back to legacy
        # label parsing only for pre-bet_ticket JSONs. GPT-5.5 R10.
        _ticket = race.get('bet_ticket') or {}
        _bet_by_label = {b.get('label'): b for b in (_ticket.get('bets') or [])
                         if b.get('label')}
        for variant, cfg in VARIANTS.items():
            # Fires now per-variant — `flat_min_pre_chaos` bypasses the
            # CHAOS filter; everyone else applies it (production behavior).
            fires = firing_bets(race, apply_chaos=cfg.get('apply_chaos', True))
            if cfg.get('qin_straight_only'):
                fires = restrict_to_straight_qin(fires)
            if not fires:
                continue
            stakes = variant_stakes(variant, race, fires, bankroll)
            for bet_label, combos in fires.items():
                stake = stakes.get(bet_label, 0.0)
                bet_obj = _bet_by_label.get(bet_label)
                if bet_obj is not None:
                    hit, div = _score_bet_from_ticket(bet_obj, race, conn, race_date)
                else:
                    hit, div = hit_and_dividend(bet_label, race, conn, race_date)
                pnl = _pnl(stake, combos, hit, div) if stake > 0 else 0.0
                rows_to_log.append({
                    'race_date': race_date,
                    'race_number': rn,
                    'variant': variant,
                    'bet_label': bet_label,
                    'combos': combos,
                    'stake_hkd': round(stake, 2),
                    'hit': int(hit),
                    'dividend_per_1': round(div, 3),
                    'pnl_hkd': round(pnl, 2),
                })
                if stake > 0:
                    print(f"R{rn:<4} {variant:<12} {bet_label:<14} {combos:>6} "
                          f"{stake:>8.0f} {('✓' if hit else '✗'):>4} "
                          f"{div:>8.3f} {pnl:>+9.0f}")

    conn.close()
    _append_log(rows_to_log, race_date)
    print(f"\nAppended {len(rows_to_log)} rows to {LOG_PATH}")
    return 0


def _append_log(rows, race_date):
    # Atomic read-modify-rewrite: write to a tmp sibling then os.replace.
    # A SIGKILL/disk-full between truncate and rewrite of LOG_PATH directly
    # would zero the only forward A/B record. Mirrors validate_hk._append_csv.
    #
    # Dedup is on (race_date, variant) restricted to the variants being
    # written, so a partial backfill (e.g. running only flat_min_pre_chaos
    # for a historical date) doesn't strip sibling-variant rows that are
    # not being re-run.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_variants = {r['variant'] for r in rows}
    existing = []
    if LOG_PATH.exists():
        with open(LOG_PATH) as f:
            existing = [r for r in csv.DictReader(f)
                        if not (r['race_date'] == race_date
                                and r['variant'] in new_variants)]
    tmp = LOG_PATH.with_suffix(LOG_PATH.suffix + '.tmp')
    # csv.DictWriter defaults to '\r\n' line endings even with newline=''
    # on the file open. Force LF-only so committed log rows don't carry
    # trailing whitespace flagged by `git show --check` (GPT-5.5 R10).
    with open(tmp, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=LOG_HEADERS, lineterminator='\n')
        w.writeheader()
        for r in existing:
            w.writerow(r)
        for r in rows:
            w.writerow(r)
    tmp.replace(LOG_PATH)


# ---------------------------------------------------------------------------
# Cumulative report — Bayesian posterior vs flat_min baseline
# ---------------------------------------------------------------------------

def _per_meeting_winner(rows, variant, baseline='flat_min') -> list[int]:
    """For each meeting in chronological order, returns 1 if `variant`
    cumulative ROI through that meeting > `baseline` cumulative ROI, else 0.
    """
    # A variant with no staked rows has never bet. Its ROI would evaluate to 0,
    # which spuriously "beats" a losing baseline every meeting — manufacturing
    # a confident posterior out of no evidence. Return no meetings instead.
    if not any(r['variant'] == variant and float(r['stake_hkd']) > 0
               for r in rows):
        return []

    meetings = sorted(set(r['race_date'] for r in rows))
    out = []
    for end_date in meetings:
        through = [r for r in rows if r['race_date'] <= end_date]
        def _roi(v):
            stake = sum(float(r['stake_hkd']) for r in through if r['variant'] == v)
            pnl = sum(float(r['pnl_hkd']) for r in through if r['variant'] == v)
            return (pnl / stake) if stake else 0
        out.append(1 if _roi(variant) > _roi(baseline) else 0)
    return out


from scripts.forward_common import bayesian_verdict as _shared_bayesian_verdict


def _bayesian_verdict(winners: list[int]) -> dict:
    """Thin wrapper around scripts.forward_common.bayesian_verdict that
    renames `n_meetings` → `n` for back-compat with the existing report
    text. Underlying logic + thresholds are identical."""
    out = _shared_bayesian_verdict(winners)
    return {'n': out['n_meetings'], 'wins': out['wins'],
            'posterior': out['posterior'], 'verdict': out['verdict']}


def report() -> int:
    if not LOG_PATH.exists():
        print(f"No {LOG_PATH.name} yet. Run scripts.stake_ab <date> after each meeting.")
        return 0
    with open(LOG_PATH) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("Log empty.")
        return 0

    meetings = sorted(set(r['race_date'] for r in rows))
    print(f"Cumulative stake A/B — {len(meetings)} meetings, {len(rows)} rows")
    print(f"Meetings: {', '.join(meetings)}")
    print()

    # Per-variant totals
    by_var = defaultdict(lambda: {'stake': 0.0, 'pnl': 0.0, 'bets': 0, 'hits': 0})
    for r in rows:
        v = r['variant']
        stake = float(r['stake_hkd'])
        if stake <= 0:
            continue
        by_var[v]['stake'] += stake
        by_var[v]['pnl'] += float(r['pnl_hkd'])
        by_var[v]['bets'] += 1
        by_var[v]['hits'] += int(r['hit'])

    print(f"{'Variant':<12} {'Label':<28} {'Bets':>5} {'Hit%':>6} "
          f"{'Stake HKD':>11} {'P&L HKD':>10} {'ROI':>8}")
    print('-' * 84)
    baseline_roi = None
    for v in VARIANTS:
        e = by_var.get(v)
        if not e or e['stake'] <= 0:
            continue
        roi = e['pnl'] / e['stake'] * 100
        hr = e['hits'] / e['bets'] * 100 if e['bets'] else 0
        if v == 'flat_min':
            baseline_roi = roi
        print(f"{v:<12} {VARIANTS[v]['label']:<28} {e['bets']:>5} {hr:>5.1f}% "
              f"{e['stake']:>11.0f} {e['pnl']:>+10.0f} {roi:>+7.1f}%")

    if baseline_roi is None:
        return 0

    # Bayesian per variant vs flat_min
    print()
    print("Bayesian posterior P(variant beats flat_min over meetings)")
    print("  SHIP if >= 0.95, KILL if <= 0.05, else WATCH")
    print('-' * 84)
    pending = []
    for v in VARIANTS:
        if v == 'flat_min':
            continue
        # A variant with NO logged rows must not be scored. _per_meeting_winner
        # compares cumulative ROI, and a variant that never bet has ROI 0 —
        # which "beats" a losing baseline in most meetings and printed a
        # confident-looking posterior for a variant that has never placed a
        # bet (found 2026-08-03 when straight_qin_only was added and reported
        # meetings-won=14/23, posterior=0.846, on zero rows).
        e = by_var.get(v)
        if not e or e['stake'] <= 0:
            pending.append(v)
            continue
        winners = _per_meeting_winner(rows, v, baseline='flat_min')
        ver = _bayesian_verdict(winners)
        print(f"  {v:<12} meetings-won={ver['wins']}/{ver['n']}  "
              f"posterior={ver['posterior']:.3f}  → {ver['verdict']}")
    if pending:
        print()
        print(f"  Awaiting first data (no bets logged yet): {', '.join(pending)}")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('race_date', nargs='?', help='YYYY/MM/DD (omit if --report)')
    p.add_argument('--bankroll', type=float, default=DEFAULT_BANKROLL,
                   help=f'Bankroll for race_ramp_split variants (default: {DEFAULT_BANKROLL:.0f})')
    p.add_argument('--report', action='store_true',
                   help='Print cumulative comparison from the log')
    args = p.parse_args()
    if args.report:
        return report()
    if not args.race_date:
        p.error('race_date required unless --report')
    return evaluate_date(args.race_date, args.bankroll)


if __name__ == '__main__':
    sys.exit(main())
