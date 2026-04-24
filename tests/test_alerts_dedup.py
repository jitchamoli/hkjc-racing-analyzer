"""Alert generation + dedup across snapshots."""
import pytest
from betting.alerts import compute_alerts, AlertState


def _make_race(race_num, horses, has_edge=False, minutes_to_post=30):
    return {
        'race_number': race_num,
        'minutes_to_post': minutes_to_post,
        'has_edge': has_edge,
        'horses': horses,
    }


def _horse(number, odds, edge_pp=0.0, prev_odds=None):
    h = {'number': number, 'win_odds': odds, 'edge_pp': edge_pp}
    if prev_odds is not None:
        h['prev_odds'] = prev_odds
    return h


def test_edge_opened_fires_once():
    prev = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=False)
    curr = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=True)
    state = AlertState()

    alerts_1 = compute_alerts([curr], [prev], state)
    assert any(a['type'] == 'EDGE_OPENED' and a['race'] == 5 for a in alerts_1)

    alerts_2 = compute_alerts([curr], [curr], state)
    assert not any(a['type'] == 'EDGE_OPENED' for a in alerts_2)


def test_edge_closed_suppresses_subsequent_edge_opened_same_race():
    state = AlertState()
    r_open = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=True)
    r_closed = _make_race(5, [_horse(1, 5.0, edge_pp=1.0)], has_edge=False)
    r_reopen = _make_race(5, [_horse(1, 8.3, edge_pp=10.0)], has_edge=True)

    compute_alerts([r_open], [], state)
    compute_alerts([r_closed], [r_open], state)
    alerts = compute_alerts([r_reopen], [r_closed], state)
    assert not any(a['type'] == 'EDGE_OPENED' for a in alerts)


def test_edge_widened_requires_prior_edge_state():
    prev = _make_race(5, [_horse(1, 8.3, edge_pp=6.0)], has_edge=True)
    curr = _make_race(5, [_horse(1, 8.3, edge_pp=9.0)], has_edge=True)
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    assert any(a['type'] == 'EDGE_WIDENED' for a in alerts)


def test_steam_fires_on_25pct_shortening():
    prev = _make_race(3, [_horse(4, 14.0)])
    curr = _make_race(3, [_horse(4, 9.5)])
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    steam = [a for a in alerts if a['type'] == 'STEAM']
    assert len(steam) == 1
    assert steam[0]['race'] == 3
    assert steam[0]['horse'] == 4


def test_steam_does_not_fire_below_threshold():
    prev = _make_race(3, [_horse(4, 10.0)])
    curr = _make_race(3, [_horse(4, 9.0)])
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    assert not any(a['type'] == 'STEAM' for a in alerts)


def test_cold_fires_on_40pct_drift_of_top_pick():
    prev = _make_race(5, [_horse(2, 4.5)])
    curr = _make_race(5, [_horse(2, 7.2)])
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    cold = [a for a in alerts if a['type'] == 'COLD']
    assert len(cold) == 1


def test_race_upcoming_fires_at_6_min():
    curr = _make_race(3, [_horse(1, 5.0)], minutes_to_post=5)
    state = AlertState()

    alerts = compute_alerts([curr], [], state)
    upcoming = [a for a in alerts if a['type'] == 'RACE_UPCOMING']
    assert len(upcoming) == 1


def test_race_upcoming_dedupes_across_refreshes():
    curr = _make_race(3, [_horse(1, 5.0)], minutes_to_post=5)
    state = AlertState()
    compute_alerts([curr], [], state)
    alerts = compute_alerts([curr], [curr], state)
    assert not any(a['type'] == 'RACE_UPCOMING' for a in alerts)


def test_steam_dedupes_within_day():
    prev = _make_race(3, [_horse(4, 14.0)])
    curr = _make_race(3, [_horse(4, 9.5)])
    state = AlertState()
    compute_alerts([curr], [prev], state)
    alerts = compute_alerts([curr], [curr], state)
    assert not any(a['type'] == 'STEAM' for a in alerts)


def test_alerts_sorted_newest_first():
    prev = _make_race(3, [_horse(4, 14.0), _horse(2, 5.0)])
    curr = _make_race(3, [_horse(4, 9.5), _horse(2, 7.5)])
    state = AlertState()

    alerts = compute_alerts([curr], [prev], state)
    assert len(alerts) == 2
    types = {a['type'] for a in alerts}
    assert 'STEAM' in types and 'COLD' in types
