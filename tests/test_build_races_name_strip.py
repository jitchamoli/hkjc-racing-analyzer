"""_build_races_from_json (auto_learn fallback) must key horse names with the
SAME canonical brand-strip the scraper uses, so an imported runner is not
fragmented into two race_results records.

Bug (10-expert review 2026-06-03): the fallback used `name.split('(')[0].strip()`
which truncates country-code names — 'KIWI STAR (NZ)(K123)' -> 'KIWI STAR' —
while the scraper (data/scraper.py) keys the same horse as 'KIWI STAR (NZ)'.
The two ingest paths then write different horse_name keys, splitting that
horse's history and biasing prior-history ML lookups for overseas runners.

Fix: use the canonical regex (strip only trailing digit-bearing brand codes,
keep letters-only country codes) + .upper(), matching core/fetcher.parse_results
and data/scraper.py.
"""
from dashboard.results_fetcher import _build_races_from_json


def _race():
    return [{
        'race_number': 1, 'venue': 'ST', 'distance': 1200,
        'horses': [{'number': 7}, {'number': 9}, {'number': 3}],
        'actual_results': [
            {'number': 7, 'name': 'KIWI STAR (NZ)(K123)', 'position': 1},
            {'number': 9, 'name': 'FRENCH FLYER (FR)', 'position': 2},
            {'number': 3, 'name': 'PODIUM(H173)', 'position': 3},
        ],
    }]


def test_country_code_kept_brand_stripped():
    out = _build_races_from_json(_race(), '2026/06/04')
    names = {e['name'] for e in out[0]['entries']}
    # Brand code (has a digit) stripped; country code (letters only) kept.
    assert 'KIWI STAR (NZ)' in names, f"got {names}"
    assert 'FRENCH FLYER (FR)' in names
    assert 'PODIUM' in names


def test_country_code_name_not_truncated():
    """The exact regression: the (NZ) runner must NOT collapse to 'KIWI STAR'."""
    out = _build_races_from_json(_race(), '2026/06/04')
    names = {e['name'] for e in out[0]['entries']}
    assert 'KIWI STAR' not in names, "country-code name was truncated (history-fragmenting bug)"
    assert 'FRENCH FLYER' not in names


def test_names_uppercased_to_match_scraper():
    races = [{
        'race_number': 1, 'horses': [{'number': 1}],
        'actual_results': [{'number': 1, 'name': 'lower case horse(J001)', 'position': 1}],
    }]
    out = _build_races_from_json(races, '2026/06/04')
    assert out[0]['entries'][0]['name'] == 'LOWER CASE HORSE'
