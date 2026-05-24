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

    labels = [b['label'] for b in ticket['bets']]
    assert labels == ['BOX QIN-3', 'BOX TRIO-5', 'F4 BOX-5']
    assert ticket['total_combos'] == 18
    assert ticket['unit_stake'] == 10.0
    assert ticket['allocated_amount'] == 180.0
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
