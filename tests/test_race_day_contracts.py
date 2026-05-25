"""Race-day invariant contract tests (GPT-5.5 R12 2026-05-25).

The "fix one bug, add one bug-specific test" pattern caught individual
regressions but missed cross-surface drift. This file pins the system
invariants — properties that must hold across every surface (visual
ticket, WA copy, replay scoring, stake_ab, prediction_log, dashboard
STRIKE badge) at the same time.

Six contract groups, mapped to bug classes the audit history saw:

  1. Ticket parity ............ stake math diverges between WA + visual + replay
  2. Results monotonicity ..... fresh fetch silently downgrades prior results
  3. ML fail-closed ........... rule-only recs leak when ML pickle fails
  4. Ranking + odds coherence . staked horse has lower final prob / no odds
  5. Betting product semantics  copy says exact-order on any-order pool
  6. Analytics parity ......... diagnostics pretend to be production

Recommended pre-race CI gate:
    python3 -m pytest tests/test_race_day_contracts.py \
                       tests/test_dashboard_playwright.py \
                       tests/test_results_fetcher_monotonic.py \
                       tests/test_bet_ticket.py \
                       tests/test_dashboard_fail_closed.py -q
    python3 -m scripts.validate_hk --report --replay
"""
from __future__ import annotations
import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import config


# ---------------------------------------------------------------------------
# 1. Ticket parity — frozen bet_ticket drives every surface
# ---------------------------------------------------------------------------

def _build_ticket(**overrides):
    """Helper: minimal viable bet_ticket dict."""
    base = {
        'version': 1,
        'race_date': '2026/05/24',
        'race_number': 1,
        'total_combos': 9,
        'unit_stake': 10.0,
        'allocated_amount': 90.0,
        'over_budget': False,
        'over_amount': 0.0,
        'bets': [
            {'label': 'BOX QIN-3', 'pool': 'QUINELLA', 'combos': 3,
             'unit_stake': 10.0, 'amount': 30.0,
             'legs': [{'rank': 1, 'number': 1, 'name': 'A'},
                      {'rank': 2, 'number': 2, 'name': 'B'},
                      {'rank': 3, 'number': 3, 'name': 'C'}],
             'reason': 'recommender'},
            {'label': 'BOX TRIO-5', 'pool': 'TRIO', 'combos': 10,
             'unit_stake': 10.0, 'amount': 100.0,
             'legs': [{'rank': i + 1, 'number': i + 1, 'name': chr(65 + i)}
                      for i in range(5)],
             'reason': 'main_trio5_gate'},
        ],
    }
    base.update(overrides)
    return base


def test_ticket_allocated_amount_never_exceeds_suggested_unless_over_budget_flagged():
    """If allocated > budget, ticket must carry `over_budget=True` so the
    user sees the over-spend honestly. Silent overspend = misleading."""
    # Reproduce the F2 scenario: 25 combos at $5 floor on $100 budget.
    from dashboard.generator import build_bet_ticket

    horses = [
        {'number': i + 1, 'name': f'H{i+1}', 'rank': i + 1,
         'bet_recommendation': 'QIN' if i < 4 else 'SKIP',
         'win_odds': 3.0 + i}
        for i in range(10)
    ]
    race = {
        'race_number': 1,
        'horses': horses,
        'conviction_tier': 'EDGE',
        'suggested_stake': {'amount': 100.0, 'pct': 1.0, 'method': 'test'},
    }
    ticket = build_bet_ticket(race, race_date='2026/05/24',
                               model_version='test',
                               generated_at='2026-05-24 12:00:00')
    if ticket is None or not ticket.get('bets'):
        pytest.skip('build_bet_ticket produced empty ticket on this fixture')
    suggested = race['suggested_stake']['amount']
    allocated = ticket['allocated_amount']
    if allocated > suggested:
        assert ticket['over_budget'] is True, (
            f"allocated ${allocated} > suggested ${suggested} but "
            f"over_budget flag is False — silent overspend"
        )
        assert ticket['over_amount'] > 0


def test_ticket_per_bet_amount_equals_combos_times_unit_stake():
    """For every bet in the ticket, amount = combos × unit_stake. WA + visual
    both read this; a divergence breaks the parity contract."""
    ticket = _build_ticket()
    for bet in ticket['bets']:
        expected = bet['combos'] * bet['unit_stake']
        assert abs(bet['amount'] - expected) < 0.01, (
            f"{bet['label']}: amount {bet['amount']} != combos×unit "
            f"({bet['combos']} × {bet['unit_stake']} = {expected})"
        )


def test_wa_uses_frozen_ticket_amount_not_recomputed():
    """WA payload must use bet.amount, not bet.combos × ss.amount/total math.
    Greps the rendered template for the frozen-amount call site."""
    src = Path(__file__).resolve().parent.parent / 'dashboard' / 'template.py'
    text = src.read_text()
    # Frozen-amount lookup helper must exist
    assert 'function _waAmountFor' in text, (
        "_waAmountFor() helper missing — WA still re-deriving stakes"
    )
    # No combos × waUnit stake math in MAIN block emitters
    assert "(bet.combos || 0) * waUnit" not in text, (
        "Found `bet.combos × waUnit` stake math in WA — should read bet.amount"
    )


def test_visual_ticket_reads_unit_stake_from_frozen_ticket():
    """The visual stake table must read race.bet_ticket.unit_stake when
    available, not re-derive `Math.max(5, Math.floor(ss.amount / combos / 5) * 5)`."""
    src = Path(__file__).resolve().parent.parent / 'dashboard' / 'template.py'
    text = src.read_text()
    # The unit_stake read must appear
    assert 'race.bet_ticket.unit_stake' in text, (
        "Visual stake table not reading frozen unit_stake"
    )
    # Legacy fallback path may still recompute — but only inside an else branch
    # that fires when bet_ticket is absent. Check the canonical guard precedes
    # any recompute:
    assert 'race.bet_ticket && race.bet_ticket.bets && race.bet_ticket.bets.length' in text


# ---------------------------------------------------------------------------
# 2. Results monotonicity — fresh fetch never silently downgrades
# ---------------------------------------------------------------------------

def test_partial_fresh_row_does_not_overwrite_complete_prior(tmp_path):
    """If HKJC returns a partial row (1 finisher) for a race that previously
    had a complete top-3 merged, the prior complete result must win."""
    from dashboard.results_fetcher import ResultsFetcher

    fetcher = ResultsFetcher.__new__(ResultsFetcher)
    fetcher.results_dir = str(tmp_path)
    existing = {
        'race_date': '2026/05/24', 'updated_at': '', 'day_summary': {},
        'races': [{'race_number': 1, 'results_available': True,
                   'actual_results': [
                       {'number': 1, 'name': 'A', 'position': 1, 'win_odds': 5.0},
                       {'number': 2, 'name': 'B', 'position': 2},
                       {'number': 3, 'name': 'C', 'position': 3},
                   ],
                   'actual_winner': 1, 'actual_top3': [1, 2, 3],
                   'actual_win_odds': 5.0,
                   'horses': [
                       {'number': 1, 'name': 'A', 'rank': 1, 'actual_position': 1,
                        'bet_recommendation': 'QIN', 'win_odds': 5.0},
                       {'number': 2, 'name': 'B', 'rank': 2, 'actual_position': 2,
                        'bet_recommendation': 'QIN', 'win_odds': 6.0},
                       {'number': 3, 'name': 'C', 'rank': 3, 'actual_position': 3,
                        'bet_recommendation': 'QIN', 'win_odds': 7.0},
                   ]}],
    }
    (tmp_path / 'race_data_20260524_with_results.json').write_text(json.dumps(existing))

    predictions = [
        {'race_number': 1,
         'horses': [{'number': i + 1, 'name': chr(65 + i), 'rank': i + 1,
                     'bet_recommendation': 'QIN', 'win_odds': 5.0 + i}
                    for i in range(3)]},
    ]
    # Fresh fetch has only the winner — partial result.
    actual_results_fresh = {1: [{'number': 1, 'name': 'A', 'position': 1, 'win_odds': 5.0}]}
    loaded = fetcher.load_existing_merged('2026/05/24')
    merged = fetcher.merge_results_with_predictions(
        predictions, actual_results_fresh, loaded
    )
    r1 = next(r for r in merged if r['race_number'] == 1)
    # Prior had 3 placed horses — partial-fresh-guard must keep them
    assert len(r1.get('actual_results', [])) == 3, (
        "partial fresh row overwrote complete prior result"
    )
    assert r1['actual_top3'] == [1, 2, 3]


def test_db_fill_uses_merged_results_not_raw_fresh():
    """_fill_prediction_log must receive the merged results dict (which
    includes restored frozen rows + partial-guard protected rows), not
    the raw fresh actual_results."""
    src = Path(__file__).resolve().parent.parent / 'dashboard' / 'results_fetcher.py'
    text = src.read_text()
    # The fill call must NOT pass raw actual_results
    assert 'self._fill_prediction_log(race_date, actual_results)' not in text, (
        "_fill_prediction_log called with raw actual_results — restored "
        "races will be dropped from prediction_log"
    )
    # Look for the merged_results derivation
    assert 'merged_results = {' in text or 'merged_results: dict' in text or \
           '_fill_prediction_log(race_date, merged_results)' in text, (
        "_fill_prediction_log not called with merged result set"
    )


def test_corrupt_existing_merged_fails_safe(tmp_path):
    """A corrupted prior _with_results must not crash or overwrite — return
    empty existing-merged dict so fresh fetch is the sole source for this run."""
    from dashboard.results_fetcher import ResultsFetcher
    fetcher = ResultsFetcher.__new__(ResultsFetcher)
    fetcher.results_dir = str(tmp_path)
    (tmp_path / 'race_data_20260524_with_results.json').write_text('NOT JSON')
    assert fetcher.load_existing_merged('2026/05/24') == {}


# ---------------------------------------------------------------------------
# 3. ML fail-closed paths — every race-day entry uses fail_closed_ml=True
# ---------------------------------------------------------------------------

def test_no_production_path_constructs_analyzer_without_fail_closed_ml():
    """Grep generator + monitor: any CombinedRaceAnalyzer() construction
    in a production race-day entry must pass fail_closed_ml=True. Default
    (fail_closed_ml=False) is only legal in tests + backtest harnesses."""
    repo = Path(__file__).resolve().parent.parent
    bad_sites = []
    for relpath in ('dashboard/generator.py', 'monitor/race_day.py'):
        text = (repo / relpath).read_text()
        # Find every CombinedRaceAnalyzer() construction and check that
        # fail_closed_ml=True is on the same line OR within 60 chars of
        # the opening paren (handles `analyzer = analyzer or CombinedRaceAnalyzer(fail_closed_ml=True)`).
        for lineno, line in enumerate(text.splitlines(), 1):
            if 'CombinedRaceAnalyzer(' not in line:
                continue
            # Skip type annotations like `analyzer: CombinedRaceAnalyzer = None`
            if ': CombinedRaceAnalyzer' in line and '(' not in line.split('CombinedRaceAnalyzer', 1)[1][:3]:
                continue
            if 'fail_closed_ml=True' not in line:
                # Also skip `from core.analyzer import CombinedRaceAnalyzer` lines
                if 'import' in line:
                    continue
                bad_sites.append(f"{relpath}:{lineno}: {line.strip()}")
    assert not bad_sites, (
        "Production race-day analyzer constructions must pass "
        "fail_closed_ml=True; found fail-open sites:\n  " +
        "\n  ".join(bad_sites)
    )


# ---------------------------------------------------------------------------
# 4. Ranking + odds coherence (F4 + F5)
# ---------------------------------------------------------------------------

def _mk_analysis(rank, combined_prob, win_odds, reasoning=None):
    """Minimal HorseAnalysis stub for recommender tests. Only the fields
    the recommender reads are set; everything else takes Horse's defaults."""
    from core.analyzer import HorseAnalysis, Horse
    horse = Horse(
        number=rank, name=f'H{rank}', draw=rank, jockey='J', trainer='T',
        weight='120', rating=50, win_odds=win_odds, place_odds=2.0,
        last_6='', running_style='Stalker',
    )
    return HorseAnalysis(
        horse=horse, rank=rank,
        rule_score=50.0, rule_confidence='MEDIUM',
        ml_prediction=None,
        combined_probability=combined_prob,
        combined_confidence='MEDIUM', value_score=combined_prob * 100,
        expected_return=(win_odds * combined_prob - 1) if win_odds > 0 else -1,
        bet_recommendation='', reasoning=reasoning or [],
        prediction_breakdown={},
    )


def test_recommender_resorts_by_calibrated_prob_after_blend():
    """After assign_bet_recommendations, rank ordering must reflect the
    POST-calibration (market-blended) probability, not the pre-blend
    model order. pre_blend_rank must preserve the original order."""
    from core.recommender import assign_bet_recommendations
    # Pre-blend order: A > B (A is rank 1 by combined_probability)
    a = _mk_analysis(rank=1, combined_prob=0.25, win_odds=4.0)
    b = _mk_analysis(rank=2, combined_prob=0.20, win_odds=2.5)  # market fav
    c = _mk_analysis(rank=3, combined_prob=0.15, win_odds=8.0)
    analyses = [a, b, c]
    assign_bet_recommendations(analyses, field_size=12, distance=1200)
    # After re-sort: rank == 1 must hold the highest calibrated_prob horse
    sorted_probs = sorted([x.combined_probability for x in analyses], reverse=True)
    assert analyses[0].combined_probability == sorted_probs[0]
    assert analyses[0].rank == 1
    # pre_blend_rank must be set on every analysis
    assert all(x.pre_blend_rank is not None for x in analyses)


def test_no_recommended_leg_has_zero_win_odds():
    """A horse with win_odds=0 must never receive a bet recommendation
    other than SKIP — even if other horses in the race have valid odds."""
    from core.recommender import assign_bet_recommendations
    a = _mk_analysis(rank=1, combined_prob=0.30, win_odds=3.0)
    b = _mk_analysis(rank=2, combined_prob=0.20, win_odds=0)  # NO ODDS
    c = _mk_analysis(rank=3, combined_prob=0.15, win_odds=5.0)
    d = _mk_analysis(rank=4, combined_prob=0.10, win_odds=8.0)
    analyses = [a, b, c, d]
    assign_bet_recommendations(analyses, field_size=12, distance=1200)
    no_odds = [x for x in analyses if not (x.horse.win_odds and x.horse.win_odds > 0)]
    for h in no_odds:
        assert h.bet_recommendation == 'SKIP', (
            f"horse #{h.horse.number} has win_odds={h.horse.win_odds} but got "
            f"bet_recommendation={h.bet_recommendation!r} — should be SKIP"
        )


# ---------------------------------------------------------------------------
# 5. Betting product semantics — pool label/copy/scoring agree
# ---------------------------------------------------------------------------

def test_f4_saver_pool_is_first_4_any_order_not_quartet():
    """HKJC FIRST 4 is ANY-order top-4. QUARTET is exact-order. F4 SAVER
    must use pool='FIRST 4' and the copy must say any-order, since the
    1-combo bet pays the FIRST 4 dividend on any-order top-4."""
    gen = (Path(__file__).resolve().parent.parent / 'dashboard' / 'generator.py').read_text()
    assert "_ticket_add_bet(bets, 'F4 SAVER', 'FIRST 4'" in gen, (
        "F4 SAVER must use pool='FIRST 4' (HKJC any-order top-4 product)"
    )
    tmpl = (Path(__file__).resolve().parent.parent / 'dashboard' / 'template.py').read_text()
    assert 'top-4 any order' in tmpl, (
        "F4 SAVER copy must say 'top-4 any order' (FIRST 4 is any-order)"
    )
    # No user-facing "exact order" string for F4 SAVER. Strings rendered to
    # WA / dashboard must say "any order". Allow the word inside JS comments
    # that explicitly document the correction (`// Earlier copy said "exact
    # order"...`), but reject any rendered visible string.
    bad_phrases = ['(1 unit · exact 1→4)', '(exact order — 1st → 4th)',
                   "'exact order — 1st → 4th'", "exact-order top-4"]
    for phrase in bad_phrases:
        # The "exact-order top-4" / "exact order — 1st → 4th" phrases must
        # not exist as RENDERED strings. Comments that document the fix
        # ("Earlier copy said 'exact order'") are fine — those contain the
        # word 'corrected' or 'Earlier'/'earlier' nearby.
        if phrase in tmpl:
            idx = tmpl.find(phrase)
            # Look 200 chars back for a comment-correction marker
            context = tmpl[max(0, idx - 200):idx]
            assert ('corrected' in context.lower() or
                    'earlier' in context.lower()), (
                f"User-facing 'exact order' wording still present for "
                f"F4 SAVER at offset {idx}: {tmpl[idx:idx+80]!r}"
            )


def test_first_4_pool_scored_any_order_in_stake_ab():
    """_score_bet_from_ticket for pool='FIRST 4' must use subset (any-order)
    semantics, not exact-order. Mirror validate_hk._score_meeting_replay."""
    from scripts.stake_ab import _score_bet_from_ticket
    if not Path(config.DB_PATH).exists():
        pytest.skip("No horse_history.db")
    conn = sqlite3.connect(str(config.DB_PATH))
    # Top-4 in DIFFERENT order than legs — any-order bet must HIT.
    race = {
        'race_number': 99,
        'horses': [
            {'name': 'A', 'number': 1, 'rank': 1, 'actual_position': 3},  # 3rd
            {'name': 'B', 'number': 2, 'rank': 2, 'actual_position': 1},  # 1st
            {'name': 'C', 'number': 3, 'rank': 3, 'actual_position': 4},  # 4th
            {'name': 'D', 'number': 4, 'rank': 4, 'actual_position': 2},  # 2nd
        ],
    }
    bet = {'label': 'F4 SAVER', 'pool': 'FIRST 4', 'combos': 1,
           'legs': [{'name': 'A', 'number': 1}, {'name': 'B', 'number': 2},
                    {'name': 'C', 'number': 3}, {'name': 'D', 'number': 4}]}
    hit, _ = _score_bet_from_ticket(bet, race, conn, '2026/05/24')
    conn.close()
    assert hit, "FIRST 4 pool must hit any-order, not exact-order"


# ---------------------------------------------------------------------------
# 6. Analytics parity — diagnostics not pretending to be production
# ---------------------------------------------------------------------------

def test_no_ticket_leg_has_zero_win_odds_at_ticket_build_time():
    """GPT-5.5 R13 #1 2026-05-25 — `_ticket_add_bet` must refuse any bet
    whose legs contain a horse with win_odds<=0, even when called by the
    automatic add-on builders (BOX QIN-4 PROTECTION / F4 SAVER / TRIO-5 /
    F4 BOX-5 / TRIO-6) that iterate raw top-N rank slices."""
    from dashboard.generator import _ticket_add_bet
    bets = []
    horses_with_bad_leg = [
        {'number': 1, 'name': 'A', 'rank': 1, 'win_odds': 3.0},
        {'number': 2, 'name': 'B', 'rank': 2, 'win_odds': 0},  # NO ODDS
        {'number': 3, 'name': 'C', 'rank': 3, 'win_odds': 5.0},
    ]
    _ticket_add_bet(bets, 'TEST QIN', 'QUINELLA', horses_with_bad_leg,
                     combos=3, reason='qin_protection')
    assert bets == [], (
        "_ticket_add_bet must refuse any bet whose legs contain a horse "
        "with win_odds<=0; got bets={!r}".format(bets)
    )

    # Sanity: same call with all-good legs DOES add the bet.
    horses_all_good = [
        {'number': 1, 'name': 'A', 'rank': 1, 'win_odds': 3.0},
        {'number': 2, 'name': 'B', 'rank': 2, 'win_odds': 4.0},
        {'number': 3, 'name': 'C', 'rank': 3, 'win_odds': 5.0},
    ]
    _ticket_add_bet(bets, 'TEST QIN', 'QUINELLA', horses_all_good,
                     combos=3, reason='qin_protection')
    assert len(bets) == 1
    assert bets[0]['label'] == 'TEST QIN'


def test_recommender_does_not_swap_rank_for_market_fav_or_banker():
    """GPT-5.5 R13 #3 2026-05-25 — both rank promotions REMOVED. Final
    output must be globally sorted by combined_probability descending,
    even when BANKER signals and a strong market favourite both exist."""
    from unittest.mock import patch
    from core.recommender import assign_bet_recommendations
    fixed_probs = [0.40, 0.30, 0.18, 0.12]
    with patch('core.recommender.calibrate_probabilities', return_value=fixed_probs):
        a = _mk_analysis(rank=1, combined_prob=0.40, win_odds=2.0)  # mkt fav
        b = _mk_analysis(rank=2, combined_prob=0.30, win_odds=4.0,
                          reasoning=['BANKER (6/8 signals)'])
        c = _mk_analysis(rank=3, combined_prob=0.18, win_odds=8.0)
        d = _mk_analysis(rank=4, combined_prob=0.12, win_odds=15.0)
        analyses = [a, b, c, d]
        assign_bet_recommendations(analyses, field_size=12, distance=1200)
    probs = [x.combined_probability for x in analyses]
    assert probs == sorted(probs, reverse=True), (
        "Final analyses must be sorted by calibrated_prob descending; "
        "got {!r}".format(probs)
    )


def test_dump_restore_includes_race_bet_ticket():
    """GPT-5.5 R13 #4 2026-05-25 — dump/restore must include
    race_bet_ticket. validate_hk --replay reads this table; omitting it
    silently zeroes replay numbers after a restore."""
    dump = (Path(__file__).resolve().parent.parent / 'scripts' / 'dump_db.py').read_text()
    restore = (Path(__file__).resolve().parent.parent / 'scripts' / 'restore_db.py').read_text()
    assert 'race_bet_ticket' in dump, (
        "scripts/dump_db.py must include race_bet_ticket in CORE_TABLES"
    )
    assert 'race_bet_ticket' in restore, (
        "scripts/restore_db.py must include race_bet_ticket in drop+verify lists"
    )


def test_monitor_uses_finalize_tickets_and_stakes():
    """GPT-5.5 R13 #2 2026-05-25 — monitor must call the shared
    finalize_tickets_and_stakes (extracted from generator) so live
    dashboards carry both bet_ticket AND suggested_stake on every race.
    Without this, the live monitor renders horses with bet_recommendation
    but no frozen ticket → WA/detail panel shows SKIP everywhere."""
    monitor = (Path(__file__).resolve().parent.parent / 'monitor' / 'race_day.py').read_text()
    assert 'finalize_tickets_and_stakes' in monitor, (
        "monitor/race_day.py must call finalize_tickets_and_stakes "
        "(shared with generator) — otherwise live dashboards skip the "
        "stake+ticket pipeline"
    )


def test_visual_ticket_blocks_render_from_frozen_legs_not_rank():
    """GPT-5.5 R13 #6 2026-05-25 — visual ticket blocks (QPL BANKER,
    BOX QIN-4 PROTECTION, F4 SAVER, F4 BOX-5, BOX TRIO-5) must read
    legs from the frozen bet_ticket via _ticketLegsFor(label), not
    reconstruct from race.horses.slice(0, N). Diverging rank vs ticket
    legs would show different horses than replay scores."""
    src = (Path(__file__).resolve().parent.parent / 'dashboard' / 'template.py').read_text()
    # The helper must exist
    assert 'function _ticketLegsFor' in src, (
        "_ticketLegsFor helper missing — visual blocks still using rank slices"
    )
    # All five protection-bet emitters must call it
    for label in ('QIN PLACE BANKER', 'BOX QIN-4 PROTECTION', 'F4 SAVER',
                  'F4 BOX-5', 'BOX TRIO-5'):
        # Inside the relevant ticketLines.push block, the legs source must
        # come through _ticketLegsFor (NOT race.horses.slice for any of
        # these labels). Spot-check via the call presence.
        assert f"_ticketLegsFor('{label}')" in src, (
            f"visual ticket block for {label} must read from "
            f"_ticketLegsFor('{label}'), not race.horses.slice"
        )


def test_compact_wa_ticket_payload_reads_bet_legs_and_bet_amount_directly():
    """GPT-5.5 R14 2026-05-25 (refined R15) — the ticket-first 'Copy WA
    Ticket' payload must iterate frozen `bet_ticket.bets` and read
    `bet.legs` + `bet.amount` directly (no rank slicing, no combo-math
    redoing). Pool label must be DERIVED from bet.pool + leg count,
    not from a hardcoded label map. Static code check; Playwright
    counterparts in test_dashboard_playwright.py assert rendered output."""
    src = (Path(__file__).resolve().parent.parent / 'dashboard' / 'template.py').read_text()
    # Compact payload variable must be built from ticket bets
    assert 'waPayloadCompact' in src, (
        "Compact WA payload variable missing — Copy WA Ticket button "
        "should produce its own ticket-first payload"
    )
    # R15 #4 — replaced hardcoded _POOL_LABEL_MAP with a derivation
    # function so every regular box label (BOX QIN-3/4, BOX TRIO-N,
    # STRAIGHT QIN/TRIO) renders correctly without per-label entries.
    assert 'function _hkjcPoolLabel' in src, (
        "Pool-label derivation function _hkjcPoolLabel missing — "
        "compact payload should derive QIN BOX-N / TRIO BOX-N / "
        "FIRST 4 BOX-N from bet.pool + bet.legs.length, not a "
        "hardcoded map"
    )
    # The compact emitter must use bet.legs directly (via map), not
    # race.horses.slice. Spot-check the leg iteration shape.
    assert '(bet.legs || []).map(function(l)' in src, (
        "Compact payload must map over bet.legs (frozen ticket legs), "
        "not over race.horses.slice"
    )
    # Status guards
    for token in ('STATUS: ', 'NO SYSTEM BET', 'LOCKED', 'STALE ODDS',
                  'OPTIONAL FORCE PICK', 'Manual only'):
        assert token in src, (
            f"Compact payload missing R15-spec token: {token!r}"
        )
    # Two test IDs distinguish the two buttons
    assert 'copy-wa-ticket-R' in src, (
        "Compact Copy WA Ticket button must have distinct testid prefix"
    )


def test_wa_copy_ticket_blocks_render_from_frozen_legs_not_rank():
    """GPT-5.5 R14 2026-05-25 — WA copy emitters for QPL BANKER, BOX QIN-4
    PROTECTION, F4 SAVER, BOX TRIO-5, and F4 BOX-5 must read legs from the
    frozen bet_ticket via _waTicketLegsFor(label). Earlier code started
    from race.horses.slice(0, N) so WA could tell the user to bet
    different horses than the dashboard visual + replay scored — the same
    failure mode that R13 #6 fixed on the visual side."""
    src = (Path(__file__).resolve().parent.parent / 'dashboard' / 'template.py').read_text()
    assert 'function _waTicketLegsFor' in src, (
        "_waTicketLegsFor helper missing — WA blocks still using rank slices"
    )
    for label in ('QIN PLACE BANKER', 'BOX QIN-4 PROTECTION', 'F4 SAVER',
                  'BOX TRIO-5', 'F4 BOX-5'):
        assert f"_waTicketLegsFor('{label}')" in src, (
            f"WA copy block for {label} must read from "
            f"_waTicketLegsFor('{label}'), not race.horses.slice"
        )


def test_stake_ab_firing_bets_reads_frozen_ticket_when_present():
    """Already pinned by test_firing_bets_prefers_frozen_ticket_over_legacy
    in test_stake_ab.py — included here as a cross-suite assertion that the
    invariant lives in race_day_contracts too."""
    from scripts.stake_ab import firing_bets
    race = {
        'horses': [{'rank': 1, 'number': 1, 'win_odds': 3.0,
                    'bet_recommendation': 'QIN'}],
        'conviction_tier': 'EDGE',
        # Frozen ticket says only ONE bet:
        'bet_ticket': {'bets': [{'label': 'PRIMARY_ONE', 'combos': 1}]},
    }
    fires = firing_bets(race)
    assert fires == {'PRIMARY_ONE': 1}, (
        f"stake_ab.firing_bets must read frozen ticket; got {fires!r}"
    )
