from dashboard.generator import build_bet_ticket


def _horse(rank, rec='SKIP', odds=6.0, steam=False):
    return {
        'rank': rank,
        'number': rank,
        'name': f'H{rank}',
        'bet_recommendation': rec,
        'win_odds': odds,
        'steam_move': {'is_steam': steam},
    }


def test_bet_ticket_persists_recommender_and_main_extras():
    race = {
        'race_number': 3,
        'conviction_tier': 'HIGH',
        'has_edge': False,
        'odds_age_minutes': 4,
        'bankroll': 10000,
        'suggested_stake': {'amount': 200.0, 'pct': 2.0, 'method': 'tier-fraction (HIGH)'},
        'horses': [
            _horse(1, 'QIN', odds=3.4),
            _horse(2, 'QIN'),
            _horse(3, 'QIN'),
            _horse(4, 'SKIP'),
            _horse(5, 'SKIP'),
            _horse(6, 'SKIP'),
            _horse(7, 'SKIP'),
            _horse(8, 'SKIP'),
        ],
    }

    ticket = build_bet_ticket(
        race, race_date='2026/05/24', model_version='m1',
        generated_at='2026-05-24 12:00:00',
    )

    # 2026-05-24: PROTECTION + F4 SAVER moved from JS to build_bet_ticket so
    # the server's ticket is the single source of truth (closes WA + detail
    # panel xfails). Top-4 has #4 (SKIP, outside primary QIN) → PROTECTION
    # fires. EDGE/HIGH/MED + field≥8 + top1=$3.4 → F4 SAVER fires.
    labels = [b['label'] for b in ticket['bets']]
    assert labels == ['BOX QIN-3', 'BOX QIN-4 PROTECTION', 'F4 SAVER',
                      'BOX TRIO-5', 'F4 BOX-5']
    assert ticket['total_combos'] == 3 + 6 + 1 + 10 + 5  # 25
    assert ticket['unit_stake'] == 5.0  # 200 / 25 → 8 → floor-5 → 5
    assert ticket['allocated_amount'] == 125.0
    assert ticket['model_version'] == 'm1'


def test_bet_ticket_empty_when_skip_or_zero_stake():
    race = {
        'race_number': 1,
        'conviction_tier': 'SKIP',
        'suggested_stake': {'amount': 0.0},
        'horses': [_horse(1), _horse(2)],
    }
    ticket = build_bet_ticket(
        race, race_date='2026/05/24', model_version='m1',
        generated_at='2026-05-24 12:00:00',
    )
    assert ticket['bets'] == []
    assert ticket['total_combos'] == 0


def test_bet_ticket_freeze_guard_refuses_any_mutation():
    """Once a bet_ticket exists, refuse ANY change to combo count or bet labels.

    Two failure modes caught on 2026/05/24:
      R3: pre-race 3 combos (BOX QIN-3 only) → post-race expansion to 18
          (BOX QIN-3 + BOX TRIO-5 + F4 BOX-5) when top1 drifted $2.40 → $2.90.
      R4: pre-race 3 combos (BOX QIN-3) → post-race shrinkage to 1 combo
          (STRAIGHT QIN) when post-race odds collapsed the QIN flag set.

    Both shapes — expansion and shrinkage — indicate the regen ran on
    post-race-contaminated odds. Refuse both.
    """
    def should_replace(old, new):
        old_combos = old.get('total_combos', 0) or 0
        new_combos = new.get('total_combos', 0) or 0
        old_labels = sorted(b.get('label', '') for b in old.get('bets', []))
        new_labels = sorted(b.get('label', '') for b in new.get('bets', []))
        return not (new_combos != old_combos or new_labels != old_labels)

    pre_race = {'total_combos': 3, 'bets': [{'label': 'BOX QIN-3'}]}

    # R3 case — expansion
    expanded = {'total_combos': 18, 'bets': [
        {'label': 'BOX QIN-3'}, {'label': 'BOX TRIO-5'}, {'label': 'F4 BOX-5'},
    ]}
    assert should_replace(pre_race, expanded) is False, "Expansion is the R3 bug"

    # R4 case — shrinkage same-entry-count but different combo count
    shrunk_same_count = {'total_combos': 1, 'bets': [{'label': 'STRAIGHT QIN'}]}
    assert should_replace(pre_race, shrunk_same_count) is False, "Shrinkage to fewer combos is the R4 bug"

    # No-op (identical ticket): allow replace (cheap, no semantic change)
    identical = {'total_combos': 3, 'bets': [{'label': 'BOX QIN-3'}]}
    assert should_replace(pre_race, identical) is True


def test_bet_ticket_trio6_includes_auto_rank6_swap():
    """2026-05-25 (GPT-5.5 Finding #2): TRIO-6 used to store horses[:6]
    server-side while JS in UI/WA swapped rank-6 with a shorter-priced,
    gear-changed horse from ranks 7-12. validate_hk --replay scored the
    unswapped legs → divergence. Now build_bet_ticket bakes the swap into
    bet_ticket.bets so all three surfaces (display, WA, replay) agree.

    Fixture: 11 horses, EDGE tier, top1=$3.0 (competitive fav), rank-6
    at $11 with no gear, rank-7 at $7.4 with gear 'B/TT' → triggers
    auto-swap (candidate odds 7.4 ≤ 11 × 0.8 = 8.8 AND gear visible).
    """
    race = {
        'race_number': 5,
        'conviction_tier': 'HIGH',
        'has_edge': False,
        'odds_age_minutes': 4,
        'bankroll': 10000,
        'suggested_stake': {'amount': 300.0, 'pct': 3.0, 'method': 'tier-fraction (HIGH)'},
        'horses': [
            _horse(1, 'QIN', odds=3.0),
            _horse(2, 'QIN', odds=5.0),
            _horse(3, 'QIN', odds=6.0),
            _horse(4, 'SKIP', odds=8.0),
            _horse(5, 'SKIP', odds=9.0),
            _horse(6, 'SKIP', odds=11.0),
            _horse(7, 'SKIP', odds=7.4),
            _horse(8, 'SKIP', odds=15.0),
            _horse(9, 'SKIP', odds=20.0),
            _horse(10, 'SKIP', odds=25.0),
            _horse(11, 'SKIP', odds=30.0),
        ],
    }
    # rank-7 (horse #7) gets gear → auto-swap candidate
    race['horses'][6]['gear'] = 'B/TT'
    # Add at least 2 steam horses to satisfy trio6_take (auto-take when steam≥2)
    for i in (0, 1):
        race['horses'][i]['steam_move'] = {'is_steam': True}

    ticket = build_bet_ticket(
        race, race_date='2026/05/25', model_version='m1',
        generated_at='2026-05-25 12:00:00',
    )

    t6 = next((b for b in ticket['bets'] if b['label'] == 'BOX TRIO-6'), None)
    assert t6 is not None, 'BOX TRIO-6 must be in bet_ticket'
    leg_nums = [l['number'] for l in t6['legs']]
    assert leg_nums == [1, 2, 3, 4, 5, 7], (
        f'TRIO-6 legs must reflect rank-6 swap (#6 → #7), got {leg_nums}'
    )
    assert t6.get('swap'), 'swap metadata must be persisted on the bet'
    assert t6['swap']['kind'] == 'auto'
    assert t6['swap']['out_number'] == 6
    assert t6['swap']['in_number'] == 7


def test_bet_ticket_trio6_includes_manual_override():
    """Manual TRIO-6 override (race.manual_trio6_horse) wins over auto swap
    and must also be reflected in bet_ticket.bets so replay sees it."""
    race = {
        'race_number': 5,
        'conviction_tier': 'HIGH',
        'manual_trio6_horse': 9,
        'manual_trio6_reason': 'racing-API form upgrade',
        'suggested_stake': {'amount': 300.0, 'pct': 3.0, 'method': 'tier-fraction (HIGH)'},
        'horses': [
            _horse(1, 'QIN', odds=3.0),
            _horse(2, 'QIN', odds=5.0),
            _horse(3, 'QIN', odds=6.0),
            _horse(4, 'SKIP', odds=8.0),
            _horse(5, 'SKIP', odds=9.0),
            _horse(6, 'SKIP', odds=11.0),
            _horse(7, 'SKIP', odds=7.4),
            _horse(8, 'SKIP', odds=15.0),
            _horse(9, 'SKIP', odds=20.0),
            _horse(10, 'SKIP', odds=25.0),
            _horse(11, 'SKIP', odds=30.0),
        ],
    }
    race['horses'][6]['gear'] = 'B/TT'
    for i in (0, 1):
        race['horses'][i]['steam_move'] = {'is_steam': True}

    ticket = build_bet_ticket(
        race, race_date='2026/05/25', model_version='m1',
        generated_at='2026-05-25 12:00:00',
    )
    t6 = next((b for b in ticket['bets'] if b['label'] == 'BOX TRIO-6'), None)
    assert t6 is not None
    leg_nums = [l['number'] for l in t6['legs']]
    assert leg_nums == [1, 2, 3, 4, 5, 9], (
        f'manual override must beat auto swap, got {leg_nums}'
    )
    assert t6['swap']['kind'] == 'manual'
    assert t6['swap']['in_number'] == 9


def test_bet_ticket_includes_trio5_and_f4_on_heavy_fav():
    """2026-05-24: heavy-fav-skip removed from F4 BOX-5 + BOX TRIO-5 triple
    gate per user. At flat $10/combo the downside is bounded, and the prior
    behavior produced 'phantom-hit' recap inconsistency on R8/R11 2026-05-24
    where the synthetic-strategy hit was reported but the bet was gated out
    of the ticket.
    """
    race = {
        'race_number': 8,
        'conviction_tier': 'EDGE',
        'has_edge': True,
        'odds_age_minutes': 4,
        'bankroll': 10000,
        'suggested_stake': {'amount': 300.0, 'pct': 3.0, 'method': 'tier-fraction (EDGE)'},
        'horses': [
            _horse(1, 'QIN', odds=1.1),  # extreme heavy fav (R8 case)
            _horse(2, 'QIN'),
            _horse(3, 'QIN'),
            _horse(4, 'QIN'),
            _horse(5, 'SKIP'),
            _horse(6, 'SKIP'),
            _horse(7, 'SKIP'),
            _horse(8, 'SKIP'),
        ],
    }
    ticket = build_bet_ticket(
        race, race_date='2026/05/24', model_version='m1',
        generated_at='2026-05-24 12:00:00',
    )
    labels = [b['label'] for b in ticket['bets']]
    assert 'BOX TRIO-5' in labels, "TRIO-5 must appear on heavy-fav races now"
    assert 'F4 BOX-5' in labels, "F4 BOX-5 must appear on heavy-fav races now"


def test_bet_ticket_freeze_guard_treats_actual_position_as_race_off():
    """Regression: bet_ticket regenerates if results_available=False, even
    after the race has run. Bookies settle post-race, top1 odds drift, the
    triple gate flips, and TRIO-5/F4-5 get persisted with bets that were
    never actionable.

    R3 2026-05-24 was the canary: ticket regen'd at 13:42:24 with top1 odds
    drifted $2.40 → $2.90, adding BOX TRIO-5 + F4 BOX-5 to a 3-combo ticket.
    Fix: also freeze if any horse has actual_position (race has off'd, even
    before results_fetcher.py marks results_available=True).
    """
    pre_race = {
        'bet_ticket': {'bets': []},
        'results_available': False,
        'horses': [{'actual_position': None}, {'actual_position': None}],
    }
    post_off_pre_results = {
        'bet_ticket': {'bets': []},
        'results_available': False,
        'horses': [{'actual_position': 1}, {'actual_position': 2}],
    }
    post_results = {
        'bet_ticket': {'bets': []},
        'results_available': True,
        'horses': [{'actual_position': 1}, {'actual_position': 2}],
    }

    def should_freeze(r):
        race_off = any(h.get('actual_position') for h in (r.get('horses') or []))
        return bool(r.get('bet_ticket') and (r.get('results_available') or race_off))

    assert should_freeze(pre_race) is False
    assert should_freeze(post_off_pre_results) is True   # the bug we fixed
    assert should_freeze(post_results) is True
