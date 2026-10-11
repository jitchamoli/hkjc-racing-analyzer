import pytest

from ml.training import drop_unusable_races, looks_truncated


def rows(positions, venue='Sha Tin'):
    return [dict(race_date='2024/06/02', race_number=1, horse_number=i,
                 finish_position=p, venue=venue) for i, p in enumerate(positions, 1)]


def test_overseas_same_date_race_does_not_join_hk_field():
    hk = rows(range(1, 13))
    overseas = dict(hk[0], horse_number=None, venue=None, finish_position=17)
    assert drop_unusable_races(hk + [overseas]) == hk


@pytest.mark.parametrize('positions', [(1, 2, 3, 5, 6, 8, 9, 10, 11, 12),
                                       (2, 3, 4, 5, 6), (1, 2, 3, 4, 8)])
def test_partial_fields_are_quarantined(positions):
    assert looks_truncated(rows(positions))
    assert drop_unusable_races(rows(positions)) == []


@pytest.mark.parametrize('positions', [(1, 2, 2, 4, 5, 6), (1, 1, 3, 4, 5),
                                       (1, 2, 3, 4, 5), (1, 2, 3, 4, 5, 0)])
def test_dead_heats_small_groups_and_nonfinishers_survive(positions):
    assert not looks_truncated(rows(positions))


def test_two_hk_venues_and_duplicate_runner_are_quarantined():
    mixed = rows(range(1, 7))
    mixed[-1]['venue'] = 'Happy Valley'
    assert drop_unusable_races(mixed) == []
    duplicate = rows(range(1, 7))
    duplicate[-1]['horse_number'] = 1
    assert drop_unusable_races(duplicate) == []
