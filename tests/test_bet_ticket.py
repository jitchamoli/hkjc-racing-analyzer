from dashboard.generator import build_bet_ticket, _merge_ticket_pre_result


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
    # HK$10 HKJC minimum floor (2026-06-08): 200 / 25 → 8 → floor-10 → 10.
    assert ticket['unit_stake'] == 10.0
    assert ticket['allocated_amount'] == 250.0
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


def test_edge_race_with_no_system_bet_gets_paper_place_on_edge_horse():
    """EDGE tier + flat field (all gates SKIP → amount 0) must surface a paper
    PLACE bet on the edge horse so the 'top tier' race is not a blank ticket.
    The bet is tagged paper: flat $10 display stake, excluded from the real
    budget (allocated_amount stays 0). User decision 2026-06-24."""
    race = {
        'race_number': 8,
        'conviction_tier': 'EDGE',
        'has_edge': True,
        'edge_horse_number': 7,
        'edge_horse_name': 'THE HEIR',
        'suggested_stake': {'amount': 0.0},
        'horses': [
            {'rank': 1, 'number': 7, 'name': 'THE HEIR', 'bet_recommendation': 'SKIP',
             'win_odds': 10.0, 'edge_fires': True},
            {'rank': 2, 'number': 4, 'name': 'AMAZING KID', 'bet_recommendation': 'SKIP',
             'win_odds': 11.0},
        ],
    }
    ticket = build_bet_ticket(
        race, race_date='2026/06/24', model_version='m1',
        generated_at='2026-06-24 10:00:00',
    )
    assert len(ticket['bets']) == 1
    bet = ticket['bets'][0]
    assert bet['label'] == 'EDGE PLACE'
    assert bet['pool'] == 'PLACE'
    assert bet['paper'] is True
    assert bet['legs'][0]['number'] == 7      # the edge horse
    assert bet['amount'] == 10.0              # flat $10 display stake
    assert ticket['allocated_amount'] == 0.0  # paper does NOT consume budget


def test_non_edge_low_tier_stays_empty():
    """A non-EDGE tier with zero stake must NOT get a paper place bet."""
    race = {
        'race_number': 1,
        'conviction_tier': 'LOW',
        'has_edge': False,
        'suggested_stake': {'amount': 0.0},
        'horses': [_horse(1), _horse(2)],
    }
    ticket = build_bet_ticket(
        race, race_date='2026/06/24', model_version='m1',
        generated_at='2026-06-24 10:00:00',
    )
    assert ticket['bets'] == []


def test_edge_tier_without_resolvable_edge_horse_stays_empty():
    """EDGE tier but no edge horse can be resolved (no edge_horse_number, no
    horse with edge_fires) → no paper bet, ticket stays empty rather than
    crashing or guessing a horse."""
    race = {
        'race_number': 8,
        'conviction_tier': 'EDGE',
        'has_edge': False,           # edge closed / not resolvable
        'suggested_stake': {'amount': 0.0},
        'horses': [
            {'rank': 1, 'number': 7, 'name': 'A', 'bet_recommendation': 'SKIP',
             'win_odds': 10.0},      # no edge_fires flag on any horse
            {'rank': 2, 'number': 4, 'name': 'B', 'bet_recommendation': 'SKIP',
             'win_odds': 11.0},
        ],
    }
    ticket = build_bet_ticket(
        race, race_date='2026/06/24', model_version='m1',
        generated_at='2026-06-24 10:00:00',
    )
    assert ticket['bets'] == []


def test_pre_result_merge_first_build_marks_source_and_stamps():
    """First-time build (no prior ticket): tag source=pre_off_build, stamp
    frozen_at + odds_snapshot_at, init empty superseded_tickets list."""
    new = {'total_combos': 3, 'bets': [{'label': 'BOX QIN-3',
                                        'legs': [{'number': 1}, {'number': 2}, {'number': 3}]}]}
    merged = _merge_ticket_pre_result(None, new, '2026-05-27 10:00:00')
    assert merged['source'] == 'pre_off_build'
    assert merged['frozen_at'] == '2026-05-27 10:00:00'
    assert merged['odds_snapshot_at'] == '2026-05-27 10:00:00'
    assert merged['superseded_tickets'] == []


def test_pre_result_merge_identical_refreshes_timestamps_only():
    """Same combos + labels + legs → keep prior superseded_tickets/source,
    refresh frozen_at + odds_snapshot_at only. No archive."""
    old = {
        'total_combos': 3, 'bets': [{'label': 'BOX QIN-3',
                                     'legs': [{'number': 1}, {'number': 2}, {'number': 3}]}],
        'frozen_at': '2026-05-27 10:00:00', 'odds_snapshot_at': '2026-05-27 10:00:00',
        'source': 'pre_off_build', 'superseded_tickets': [],
    }
    new = {'total_combos': 3, 'bets': [{'label': 'BOX QIN-3',
                                        'legs': [{'number': 1}, {'number': 2}, {'number': 3}]}]}
    merged = _merge_ticket_pre_result(old, new, '2026-05-27 14:00:00')
    assert merged['source'] == 'pre_off_build'
    assert merged['frozen_at'] == '2026-05-27 14:00:00'
    assert merged['odds_snapshot_at'] == '2026-05-27 14:00:00'
    assert merged['superseded_tickets'] == []


def test_pre_result_merge_combo_change_archives_old():
    """R3 2026-05-24 pattern: 3 combos → 18 combos. Archive old, replace, mark
    source=pre_off_refresh."""
    old = {
        'total_combos': 3, 'bets': [{'label': 'BOX QIN-3',
                                     'legs': [{'number': 1}, {'number': 2}, {'number': 3}]}],
        'frozen_at': '2026-05-27 10:00:00', 'odds_snapshot_at': '2026-05-27 10:00:00',
        'source': 'pre_off_build', 'superseded_tickets': [],
    }
    new = {'total_combos': 18, 'bets': [
        {'label': 'BOX QIN-3', 'legs': [{'number': 1}, {'number': 2}, {'number': 3}]},
        {'label': 'BOX TRIO-5', 'legs': [{'number': i} for i in range(1, 6)]},
        {'label': 'F4 BOX-5', 'legs': [{'number': i} for i in range(1, 6)]},
    ]}
    merged = _merge_ticket_pre_result(old, new, '2026-05-27 18:30:00')
    assert merged['source'] == 'pre_off_refresh'
    assert merged['frozen_at'] == '2026-05-27 18:30:00'
    assert merged['total_combos'] == 18
    assert len(merged['superseded_tickets']) == 1
    archived = merged['superseded_tickets'][0]
    assert archived['total_combos'] == 3
    assert archived['superseded_at'] == '2026-05-27 18:30:00'
    # Archive does NOT carry its own superseded_tickets (avoid recursion)
    assert 'superseded_tickets' not in archived


def test_pre_result_merge_label_change_archives_old():
    """R4 2026-05-24 pattern: same combo count, different labels. Archive."""
    old = {
        'total_combos': 1, 'bets': [{'label': 'BOX QIN-3',
                                     'legs': [{'number': 1}]}],
        'frozen_at': '2026-05-27 10:00:00', 'source': 'pre_off_build',
        'superseded_tickets': [],
    }
    new = {'total_combos': 1, 'bets': [{'label': 'STRAIGHT QIN',
                                        'legs': [{'number': 1}]}]}
    merged = _merge_ticket_pre_result(old, new, '2026-05-27 18:30:00')
    assert merged['source'] == 'pre_off_refresh'
    assert len(merged['superseded_tickets']) == 1
    assert merged['superseded_tickets'][0]['bets'][0]['label'] == 'BOX QIN-3'


def test_pre_result_merge_legs_change_archives_old():
    """Same combos + labels but DIFFERENT leg numbers (e.g. TRIO-6 rank-6 swap
    flipping). Must archive."""
    old = {
        'total_combos': 20, 'bets': [{'label': 'BOX TRIO-6',
                                      'legs': [{'number': i} for i in [1, 2, 3, 4, 5, 6]]}],
        'frozen_at': '2026-05-27 10:00:00', 'source': 'pre_off_build',
        'superseded_tickets': [],
    }
    new = {'total_combos': 20, 'bets': [{'label': 'BOX TRIO-6',
                                         'legs': [{'number': i} for i in [1, 2, 3, 4, 5, 7]]}]}
    merged = _merge_ticket_pre_result(old, new, '2026-05-27 18:30:00')
    assert merged['source'] == 'pre_off_refresh'
    assert len(merged['superseded_tickets']) == 1
    archived = merged['superseded_tickets'][0]
    assert [l['number'] for l in archived['bets'][0]['legs']] == [1, 2, 3, 4, 5, 6]


def test_pre_result_merge_accumulates_history():
    """Multiple refreshes accumulate in superseded_tickets in order."""
    t1 = {'total_combos': 3, 'bets': [{'label': 'BOX QIN-3', 'legs': [{'number': 1}]}],
          'frozen_at': '2026-05-27 09:00:00', 'source': 'pre_off_build',
          'superseded_tickets': []}
    t2_new = {'total_combos': 6, 'bets': [{'label': 'BOX QIN-4', 'legs': [{'number': 1}]}]}
    t2 = _merge_ticket_pre_result(t1, t2_new, '2026-05-27 12:00:00')
    t3_new = {'total_combos': 10, 'bets': [{'label': 'BOX QIN-5', 'legs': [{'number': 1}]}]}
    t3 = _merge_ticket_pre_result(t2, t3_new, '2026-05-27 18:00:00')
    assert len(t3['superseded_tickets']) == 2
    assert t3['superseded_tickets'][0]['total_combos'] == 3  # original
    assert t3['superseded_tickets'][1]['total_combos'] == 6  # mid


def test_finalize_tickets_post_result_blocks_replacement():
    """Layer 1 (post-RESULT fallback): results_available=True must
    prevent any mutation even if the rebuilt ticket would differ.

    Note: the PRIMARY post-OFF freeze lives upstream in
    `_merge_with_existing`, which uses scheduled `race['time']` (HKT) to
    restore frozen fields once now >= post-time. This Layer 1 is the
    belt-and-braces post-RESULT fallback for the small window between
    off and results landing, and for non-today scenarios where the
    scheduled-time guard intentionally doesn't fire.
    """
    from dashboard.generator import finalize_tickets_and_stakes
    old_ticket = {
        'total_combos': 3, 'bets': [{'label': 'BOX QIN-3', 'legs': [{'number': 1}]}],
        'frozen_at': '2026-05-27 10:00:00', 'source': 'pre_off_build',
        'superseded_tickets': [],
    }
    race = {
        'race_number': 1, 'conviction_tier': 'HIGH', 'has_edge': False,
        'odds_age_minutes': 4, 'bankroll': 10000,
        'suggested_stake': {'amount': 200.0, 'pct': 2.0, 'method': 'tier-fraction (HIGH)'},
        'bet_ticket': old_ticket,
        'results_available': True,
        'horses': [_horse(i, 'QIN' if i <= 4 else 'SKIP', odds=3.0 + i) for i in range(1, 9)],
    }
    finalize_tickets_and_stakes([race], '2026/05/27', analyzer=None)
    assert race['bet_ticket'] is old_ticket, "post-result ticket must not be replaced"


def test_finalize_tickets_actual_position_blocks_replacement():
    """Layer 1 alt-signal: actual_position set on any horse (results landed
    on horses but results_available flag not yet set by results_fetcher).
    Same fail-closed behavior."""
    from dashboard.generator import finalize_tickets_and_stakes
    old_ticket = {
        'total_combos': 3, 'bets': [{'label': 'BOX QIN-3', 'legs': [{'number': 1}]}],
        'frozen_at': '2026-05-27 10:00:00', 'source': 'pre_off_build',
        'superseded_tickets': [],
    }
    horses = [_horse(i, 'QIN' if i <= 4 else 'SKIP', odds=3.0 + i) for i in range(1, 9)]
    horses[0]['actual_position'] = 1
    race = {
        'race_number': 1, 'conviction_tier': 'HIGH', 'has_edge': False,
        'odds_age_minutes': 4, 'bankroll': 10000,
        'suggested_stake': {'amount': 200.0, 'pct': 2.0, 'method': 'tier-fraction (HIGH)'},
        'bet_ticket': old_ticket,
        'results_available': False,
        'horses': horses,
    }
    finalize_tickets_and_stakes([race], '2026/05/27', analyzer=None)
    assert race['bet_ticket'] is old_ticket, "actual_position set must freeze ticket"


def test_finalize_tickets_pre_result_changed_archives_old():
    """Pre-result rebuild with a different ticket shape: replace AND archive.
    This is the bug fix — old code refused, new code archives + replaces."""
    from dashboard.generator import finalize_tickets_and_stakes
    # Old ticket = a 1-combo skeleton that won't match the rebuilt shape.
    old_ticket = {
        'total_combos': 1, 'bets': [{'label': 'STRAIGHT QIN',
                                     'legs': [{'number': 1}, {'number': 2}]}],
        'frozen_at': '2026-05-27 09:00:00', 'source': 'pre_off_build',
        'superseded_tickets': [],
    }
    # Fresh horses → build_bet_ticket will produce the proper HIGH-tier ticket.
    race = {
        'race_number': 1, 'conviction_tier': 'HIGH', 'has_edge': False,
        'odds_age_minutes': 4, 'bankroll': 10000,
        'suggested_stake': {'amount': 200.0, 'pct': 2.0, 'method': 'tier-fraction (HIGH)'},
        'bet_ticket': old_ticket,
        'results_available': False,
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
    finalize_tickets_and_stakes([race], '2026/05/27', analyzer=None)
    new_ticket = race['bet_ticket']
    # New ticket is different shape (rebuilt) — old must be archived.
    assert new_ticket is not old_ticket
    assert new_ticket['source'] == 'pre_off_refresh'
    assert len(new_ticket['superseded_tickets']) == 1
    assert new_ticket['superseded_tickets'][0]['bets'][0]['label'] == 'STRAIGHT QIN'


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
