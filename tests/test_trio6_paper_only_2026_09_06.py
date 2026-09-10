r"""BOX TRIO-6 is demoted to PAPER — generated and tracked, never staked.

Forward replay over 16 meetings (2026/05/24 .. 2026/09/06), real HKJC
dividends, the user's flat HK$10/combo:

    BOX TRIO-6   stake $12,000   return $5,230   19 hits   -$6,770   -56.4%
    everything else combined                               -$9,030   -44.4%

It is simultaneously the LARGEST single exposure on the card (37% of all money
staked) and the WORST return, and it was net positive in only 2 of 16 meetings.
The structural reason is the conviction/exotic mismatch already documented in
memory: 20 combos x $10 = $200 per race, so a hit must clear the whole card's
TRIO-6 outlay to break even — but the races we hit are the predictable ones,
which pay least. 2026/09/06 is the illustration: TRIO-6 HIT on R8 and still
lost $426 on the meeting ($574 returned against $1,000 staked over 5 races).

PAPER, not deleted. The bet keeps being generated, displayed and scored so the
decision stays reversible on data rather than on memory. Two consequences the
tests below pin:

1. It must not consume the real budget — `allocated_amount` excludes it, the
   same treatment EDGE PLACE already gets.
2. It must NOT silently vanish from tracking. `validate_hk --replay` skips
   paper bets by design (they are not real-money ROI), so demoting TRIO-6
   without more would have deleted it from the only report that measures it.
   Paper bets are therefore accumulated and reported under their own heading.
"""
from __future__ import annotations

import pytest

from dashboard.generator import _PAPER_ONLY_LABELS, build_bet_ticket


def _race(n_horses=12):
    return {
        'race_number': 4,
        'conviction_tier': 'HIGH',
        'has_edge': False,
        'suggested_stake': {'amount': 150.0, 'pct': 1.5, 'method': 'test'},
        'odds_age_minutes': 0,
        'horses': [
            {'number': i, 'name': f'H{i}', 'rank': i, 'win_odds': 3.0 + i,
             'place_odds': 2.0, 'calibrated_prob': 0.3 / i,
             'bet_recommendation': 'QIN' if i <= 4 else 'SKIP'}
            for i in range(1, n_horses + 1)
        ],
    }


def _ticket():
    return build_bet_ticket(_race(), race_date='2026/09/06',
                            model_version='test', generated_at='now')


class TestTrio6IsPaper:
    def test_trio6_is_in_the_paper_only_set(self):
        assert 'BOX TRIO-6' in _PAPER_ONLY_LABELS

    def test_trio6_bet_is_flagged_paper(self):
        t = _ticket()
        six = [b for b in t['bets'] if b['label'] == 'BOX TRIO-6']
        assert six, f"fixture must trigger TRIO-6: {[b['label'] for b in t['bets']]}"
        assert six[0].get('paper') is True, 'BOX TRIO-6 must be flagged paper'

    def test_trio6_does_not_consume_the_real_budget(self):
        t = _ticket()
        six = [b for b in t['bets'] if b['label'] == 'BOX TRIO-6']
        assert six, 'fixture must trigger TRIO-6'
        real = sum(b['amount'] for b in t['bets'] if not b.get('paper'))
        assert t['allocated_amount'] == pytest.approx(real)
        assert six[0]['amount'] > 0, 'paper bet still carries a display stake'
        assert t['allocated_amount'] < real + six[0]['amount']

    def test_it_is_still_generated(self):
        """Paper, not deleted — the decision must stay reversible on data."""
        t = _ticket()
        assert any(b['label'] == 'BOX TRIO-6' for b in t['bets']), (
            'TRIO-6 must still be GENERATED — paper, not deleted')


class TestPaperStaysTracked:
    """validate_hk --replay skips paper bets; they must still be reported."""

    def test_replay_accumulates_paper_separately(self):
        import scripts.validate_hk as vh
        assert hasattr(vh, 'PAPER_LABEL_PREFIX'), (
            'paper bets need a distinct accumulator key or they vanish from '
            'the replay report entirely'
        )

    def test_paper_prefix_is_not_confusable_with_a_real_label(self):
        import scripts.validate_hk as vh
        assert vh.PAPER_LABEL_PREFIX.strip(), 'prefix must be non-empty'
        assert not vh.PAPER_LABEL_PREFIX[0].isalnum(), (
            'prefix must be visually distinct from a real bet label so a '
            'paper row can never be read as staked money'
        )
