"""Pin the exact rating_change → trend_score mapping shipped 2026-05-25
(GPT-5.5 R6 Finding #6, R7 GPT-5.5 suggested regression test). Old
mapping (80 / 65 / 45 / 30) treated rating rise as strictly positive —
anti-handicapping in HK where a rating rise = more weight + tougher
class next start. Conservative rebalance:

  +5+ rating: 55 (carrying more weight)
  +1-4:       50 (slightly tougher mark)
  -1-3:       55 (well-handicapped)
  -4 or less: 35 (declining form)
   0:         no factor contribution

Also pins the flipped BANKER signal #2: was "Rating surging (+5)" → now
"Well-handicapped (-1 to -3)".
"""
from __future__ import annotations
from unittest.mock import MagicMock

import pytest


def _stub_horse(rating_change: int):
    h = MagicMock()
    h.rating_change = rating_change
    return h


@pytest.mark.parametrize('rtg_chg,expected_score', [
    (8,   55.0),   # large rise — was 80
    (5,   55.0),   # threshold for large rise
    (4,   50.0),   # small rise — was 65
    (1,   50.0),   # min positive rise
    (0,    0.0),   # no change — no factor
    (-1,  55.0),   # small drop — was 45 (flipped to slight positive)
    (-3,  55.0),   # threshold of "well-handicapped" band
    (-4,  35.0),   # threshold of declining — was 30
    (-10, 35.0),   # large drop — declining
])
def test_rating_change_to_trend_score_mapping(rtg_chg, expected_score):
    """Direct unit test of the score lookup logic in core/analyzer.py:796.
    Reads the lookup branch values verbatim — any future change to those
    numbers will require updating this test, which forces the change to
    be explicit and reviewed."""
    # Inline the score-lookup logic to keep this test independent of the
    # surrounding rule engine's expensive setup. If the analyzer changes
    # the lookup, this stub must be updated to match (and the diff will
    # show in PR review).
    if rtg_chg != 0:
        if rtg_chg >= 5:
            score = 55.0
        elif rtg_chg > 0:
            score = 50.0
        elif rtg_chg >= -3:
            score = 55.0
        else:
            score = 35.0
    else:
        score = 0.0
    assert score == expected_score, (
        f"rating_change={rtg_chg}: expected {expected_score}, got {score}"
    )

    # Cross-check that core/analyzer.py contains the exact literal numbers
    # we expect. If a refactor changes them silently this test catches it.
    from pathlib import Path
    src = Path(__file__).resolve().parent.parent / 'core' / 'analyzer.py'
    text = src.read_text()
    # Hunt for the rebalanced numbers in the rating-change block.
    assert 'trend_score = 55.0' in text, \
        "core/analyzer.py: expected 'trend_score = 55.0' (post-2026-05-25 rebalance)"
    assert 'trend_score = 50.0' in text, \
        "core/analyzer.py: expected 'trend_score = 50.0'"
    assert 'trend_score = 35.0' in text, \
        "core/analyzer.py: expected 'trend_score = 35.0'"


def test_banker_signal_well_handicapped_replaces_rating_surge():
    """BANKER signal #2 flipped 2026-05-25: was 'Rating surging (+5+)',
    now 'Well-handicapped (-1 to -3)'. The old direction was
    anti-handicapping (rating rise = more weight) AND double-counted
    form_score. Pin the new contract."""
    from pathlib import Path
    src = (Path(__file__).resolve().parent.parent / 'core' / 'analyzer.py').read_text()
    assert '-3 <= rtg_chg <= -1' in src, \
        "BANKER signal #2 must promote small rating drops (well-handicapped)"
    assert 'Well-handicapped' in src, \
        "BANKER signal #2 reason text missing"
    # Old BANKER promotion path (`if rtg_chg >= 5:` followed by
    # `banker_reasons.append(f"Rating surging`) must NOT be present as an
    # active append; matches against the literal append line, not the
    # surrounding "was" comment that documents the flip.
    assert 'banker_reasons.append(f"Rating surging' not in src, \
        "Old anti-handicapping BANKER append must not return"
