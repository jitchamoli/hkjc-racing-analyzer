"""Race-card weight must be the ACTUAL weight carried, like the results table.

Bug (code review 2026-08-03): a live race card carries the DECLARED (allotted)
weight in cell [5], while the apprentice's claim sits in the jockey cell as
"(-7)" and was stripped and thrown away:

    jockey = re.sub(r"\\s*\\(-?\\d+\\)", "", jockey).strip()

But `race_results.weight` — what `ml/training.py` trains on — stores the
ACTUAL weight carried (declared minus the claim). So the `weight_vs_field`
feature was trained on post-claim weight and evaluated on pre-claim weight:
a live ML-Rule-#2 (train/predict consistency) violation.

Measured over 25 saved meetings / 3,082 runners: 547 (17.7%) had card weight
!= DB weight, and the delta histogram is exactly the HK claim ladder
(+2 x291, +3 x30, +7 x87, +10 x77) rather than symmetric noise. Broken out by
rider it is unambiguous — P N Wong {0: 22, 7: 87}, H Y Yuen {0: 11, 10: 77},
while Z Purton (no claim) is {0: 179}.

Because the claim also shifts the field MEDIAN, every other horse in an
apprentice-ridden race is skewed the opposite way too.
"""
import pytest

from core.fetcher import HKJCParser


class TestApprenticeClaim:
    @pytest.mark.parametrize("raw,claim,expected", [
        ("P N Wong(-7)", 7, "126"),
        ("H Y Yuen (-10)", 10, "123"),
        ("C L Chau(-2)", 2, "131"),
        ("Y L Chung (-3)", 3, "130"),
    ])
    def test_claim_is_subtracted_from_declared_weight(self, raw, claim, expected):
        """133 declared minus the claim = what the results table will store."""
        assert HKJCParser.apply_apprentice_claim("133", raw) == expected

    def test_no_claim_leaves_weight_untouched(self):
        assert HKJCParser.apply_apprentice_claim("133", "Z Purton") == "133"

    def test_positive_paren_is_not_treated_as_a_claim(self):
        """Only a NEGATIVE parenthesised number is an apprentice claim."""
        assert HKJCParser.apply_apprentice_claim("133", "A Rider (5)") == "133"

    def test_unparseable_weight_is_passed_through(self):
        assert HKJCParser.apply_apprentice_claim("", "P N Wong(-7)") == ""
        assert HKJCParser.apply_apprentice_claim("n/a", "P N Wong(-7)") == "n/a"

    def test_claim_never_drives_weight_implausibly_low(self):
        """Guard against a mis-parse producing a nonsense carried weight."""
        assert HKJCParser.apply_apprentice_claim("100", "X (-90)") == "100"


class TestJockeyNameUnchanged:
    def test_claim_is_still_stripped_from_the_jockey_name(self):
        """Regression guard: the name must stay clean for DB keying."""
        assert HKJCParser.strip_jockey_claim("P N Wong(-7)") == "P N Wong"
        assert HKJCParser.strip_jockey_claim("H Y Yuen (-10)") == "H Y Yuen"
        assert HKJCParser.strip_jockey_claim("Z Purton") == "Z Purton"
