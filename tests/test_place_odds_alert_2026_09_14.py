"""A silent place-odds outage must be loud.

Place odds were 0 on every runner from at least May to 2026-09-14 and nothing
said so: the dead odds pages returned None, the Tips page kept win odds alive,
and every log line stayed green. The fix now reads PLA from HKJC's GraphQL
endpoint, which rejects any query it has not whitelisted — so if HKJC changes
that query the gap comes straight back.

Rule: when the card has win odds (pools are trading) but NOT ONE race has a
place price, that is an outage. Meeting-level on purpose: a single race with
no place price (closed pool after the off, a scratched-heavy field) is normal
and must not cry wolf during `raceday live`.
"""
from core.fetcher import HorseEntry, RaceCard
from dashboard.generator import meeting_missing_place_odds


def _race(no, *horses):
    return RaceCard(race_number=no, horses=[HorseEntry(number=i + 1, name=f"H{i}", win_odds=w, place_odds=p)
                                            for i, (w, p) in enumerate(horses)])


def test_win_odds_everywhere_but_no_place_odds_anywhere_is_an_outage():
    fetched = [_race(1, (2.5, 0.0), (6.0, 0.0)), _race(2, (3.1, 0.0))]
    assert meeting_missing_place_odds(fetched) == [1, 2]


def test_one_race_without_place_odds_is_not_an_outage():
    fetched = [_race(1, (2.5, 1.3), (6.0, 2.1)), _race(2, (3.1, 0.0))]   # R2 pool closed
    assert meeting_missing_place_odds(fetched) == []


def test_no_win_odds_means_pools_not_open_yet_so_no_alert():
    fetched = [_race(1, (0.0, 0.0)), _race(2, (0.0, 0.0))]
    assert meeting_missing_place_odds(fetched) == []


def test_failed_and_empty_fetches_are_ignored():
    fetched = [RuntimeError("timeout"), None, RaceCard(race_number=3), _race(4, (2.0, 0.0))]
    assert meeting_missing_place_odds(fetched) == [4]
