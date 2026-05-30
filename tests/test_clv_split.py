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
