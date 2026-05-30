"""Lock the CLV market-agreement split logic (scripts/clv_report.py).

Guards the disambiguation that separates real sharpness (we back a horse the
market did NOT initially favour, and it shortens) from the favourite
late-money mirage. The pure helpers are tested directly; no DB needed.
"""
import scripts.clv_report as clv


def test_market_rank_ranks_by_ascending_opening_odds():
    # shortest opening odds = market favourite = rank 1
    om = {('2026/05/30', 1): {3: 2.5, 7: 5.0, 9: 12.0}}
    assert clv._market_rank(om, '2026/05/30', 1, 3) == 1   # favourite
    assert clv._market_rank(om, '2026/05/30', 1, 7) == 2
    assert clv._market_rank(om, '2026/05/30', 1, 9) == 3   # longshot


def test_market_rank_missing_returns_none():
    om = {('2026/05/30', 1): {3: 2.5}}
    assert clv._market_rank(om, '2026/05/30', 1, 99) is None  # horse absent
    assert clv._market_rank(om, '2026/05/30', 5, 3) is None   # race absent


def test_agg_empty_is_none():
    assert clv._agg([]) is None


def test_agg_computes_rates():
    rows = [
        {'clv': 10.0, 'won': True, 'placed': True},
        {'clv': -5.0, 'won': False, 'placed': True},
    ]
    a = clv._agg(rows)
    assert a['n'] == 2
    assert a['clv'] == 2.5            # mean CLV
    assert a['beat'] == 50.0          # 1 of 2 shortened
    assert a['win'] == 50.0
    assert a['place'] == 100.0


def _bucket(clv_pct, win=0.0, place=0.0, n=5):
    return {'clv': clv_pct, 'win': win, 'place': place, 'n': n}


def test_verdict_real_signal_when_rank23_positive_and_converts():
    # rank 2-3 positive AND placing = the only monetizable signal
    r23 = _bucket(5.0, win=30, place=60)
    tag, _ = clv._verdict(_bucket(2.0), _bucket(4.0), r23, _bucket(40.0, win=0))
    assert tag == 'REAL'  # reported FIRST even though longshots also blow out


def test_verdict_longshot_blowout_is_artifact():
    # rank 4+ huge CLV but 0% wins = noise, not edge
    tag, _ = clv._verdict(_bucket(5.0, place=64), None,
                          _bucket(-12.7, win=33, place=44), _bucket(45.0, win=0, place=17))
    assert tag == 'ARTIFACT'


def test_verdict_weak_rank23_does_not_mask_artifact():
    # REVIEW FINDING 2026-05-30: a sub-threshold positive rank-2-3 CLV (+0.8%)
    # must NOT suppress the ARTIFACT diagnosis when rank-4+ is a 0-win blowout.
    # The old chain printed "Mixed/flat" here — the gap this test locks shut.
    tag, _ = clv._verdict(_bucket(2.0), None,
                          _bucket(0.8, win=10, place=40), _bucket(45.0, win=0))
    assert tag == 'ARTIFACT'


def test_verdict_favourite_mirage():
    # CLV only in favourites, no longshot blowout, no real rank-2-3 signal
    tag, _ = clv._verdict(_bucket(6.0, place=64), _bucket(1.0),
                          _bucket(-3.0), None)
    assert tag == 'MIRAGE'


def test_verdict_inconclusive_when_all_weak():
    tag, _ = clv._verdict(_bucket(0.2), _bucket(0.1), _bucket(0.3), _bucket(2.0, win=20))
    assert tag == 'INCONCLUSIVE'
