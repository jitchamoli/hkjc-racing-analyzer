"""C1 test: missing rating (=0) is excluded from weighted average,
not dragged into it with weight 0.08."""
from core.scorer import compute_weighted_score


def test_missing_rating_excluded_from_weighted_score():
    """A race where all horses have rating=0 (missing data) must produce
    the same ranking as a race where rating is absent entirely from the
    factors dict. Pre-fix: rating=0 contributes 0 × 0.08 = 0 to every
    score, dragging all scores by ~8 points. Post-fix: rating absent from
    weight_map, renormalized out."""
    factors_with_rating_zero = {
        "rating": 0.0,       # missing data, should be excluded
        "form": 60.0,
        "jockey": 55.0,
        "trainer": 50.0,
        "draw": 50.0,
        "pace": 55.0,
    }
    factors_without_rating = {
        "form": 60.0,
        "jockey": 55.0,
        "trainer": 50.0,
        "draw": 50.0,
        "pace": 55.0,
    }
    score_with_zero = compute_weighted_score(factors_with_rating_zero)
    score_without = compute_weighted_score(factors_without_rating)
    assert abs(score_with_zero - score_without) < 0.01, (
        f"rating=0 treated differently than rating absent: "
        f"with_zero={score_with_zero:.4f}, without={score_without:.4f}. "
        f"Pre-fix: with_zero is ~8 points lower due to ALWAYS_INCLUDED drag."
    )


def test_missing_gear_excluded_from_weighted_score():
    """Same assertion for gear (empty string produces gear_score=0)."""
    factors_with_gear_zero = {
        "rating": 65.0,
        "form": 60.0,
        "jockey": 55.0,
        "gear": 0.0,   # missing gear
    }
    factors_without_gear = {
        "rating": 65.0,
        "form": 60.0,
        "jockey": 55.0,
    }
    score_with_zero = compute_weighted_score(factors_with_gear_zero)
    score_without = compute_weighted_score(factors_without_gear)
    assert abs(score_with_zero - score_without) < 0.01, (
        f"gear=0 treated differently: with_zero={score_with_zero:.4f}, "
        f"without={score_without:.4f}"
    )


def test_nonzero_rating_still_contributes():
    """Rating with actual value (e.g. 70) still contributes correctly."""
    factors_with_rating = {
        "rating": 70.0,
        "form": 60.0,
        "jockey": 55.0,
    }
    factors_without_rating = {
        "form": 60.0,
        "jockey": 55.0,
    }
    score_with = compute_weighted_score(factors_with_rating)
    score_without = compute_weighted_score(factors_without_rating)
    # Rating of 70 is above the others' average, so adding it should raise score
    assert score_with > score_without, (
        f"rating=70 should raise score but didn't: "
        f"with={score_with:.4f}, without={score_without:.4f}"
    )
