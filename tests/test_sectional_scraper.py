"""Tests for the HKJC sectional-time parser (Path B, Step 1).

parse_sectional_cell + position resolution are tested on synthetic strings;
parse_sectional_html is tested against a real captured page fixture so the
suite stays deterministic and offline.
"""
from pathlib import Path

import pytest

from data.sectional_scraper import (
    parse_sectional_cell,
    parse_sectional_html,
    _is_valid_margin,
    _pos_margin_candidates,
    _resolve_section_positions,
    _split_name_brand,
)

FIXTURE = Path(__file__).parent / "fixtures" / "sectional_20260513_r1.html"


# ---------------------------------------------------------------------------
# parse_sectional_cell — the concatenated cell format
# ---------------------------------------------------------------------------

class TestParseSectionalCell:
    def test_simple_cell_no_subsplits(self):
        # "5227.84" -> prefix "52", time 27.84
        out = parse_sectional_cell("5227.84")
        assert out == {"prefix": "52", "time": 27.84, "sub_splits": []}

    def test_cell_with_subsplits(self):
        # "31-1/424.2212.0012.22" -> prefix "31-1/4", time 24.22, splits 12.00/12.22
        out = parse_sectional_cell("31-1/424.2212.0012.22")
        assert out["prefix"] == "31-1/4"
        assert out["time"] == 24.22
        assert out["sub_splits"] == [12.00, 12.22]

    def test_abbreviation_margin(self):
        # "1SH24.1011.8112.29" -> prefix "1SH" (pos 1, margin short-head)
        out = parse_sectional_cell("1SH24.1011.8112.29")
        assert out["prefix"] == "1SH"
        assert out["time"] == 24.10
        assert out["sub_splits"] == [11.81, 12.29]

    def test_nose_margin(self):
        out = parse_sectional_cell("1N23.6111.4912.12")
        assert out["prefix"] == "1N"
        assert out["time"] == 23.61

    def test_two_digit_position_prefix(self):
        # "11923.49" -> prefix "119" (pos 11, margin 9) — split happens later
        out = parse_sectional_cell("11923.49")
        assert out["prefix"] == "119"
        assert out["time"] == 23.49

    def test_empty_cell_returns_none(self):
        assert parse_sectional_cell("") is None
        assert parse_sectional_cell("   ") is None

    def test_no_time_token_returns_none(self):
        # A garbage cell with no NN.NN token
        assert parse_sectional_cell("ABC") is None

    def test_section_time_is_first_token(self):
        # The FIRST NN.NN is the section time, not a sub-split
        out = parse_sectional_cell("8423.7811.8911.89")
        assert out["time"] == 23.78
        assert out["sub_splits"] == [11.89, 11.89]


# ---------------------------------------------------------------------------
# margin validation + position/margin candidates
# ---------------------------------------------------------------------------

class TestMarginValidation:
    @pytest.mark.parametrize("m", ["", "SH", "HD", "NK", "NS", "N", "DH", "SHD",
                                   "1", "9", "12", "3/4", "1/2",
                                   "1-1/4", "5-3/4", "9-1/2"])
    def test_valid_margins(self, m):
        assert _is_valid_margin(m)

    @pytest.mark.parametrize("m", ["X", "1.5", "-", "/", "ABC", "1-2"])
    def test_invalid_margins(self, m):
        assert not _is_valid_margin(m)


class TestPosMarginCandidates:
    def test_unambiguous_single_digit(self):
        # "52" -> only (5, "2"); 2-digit "52" is out of range
        assert _pos_margin_candidates("52") == [(5, "2")]

    def test_abbreviation_prefix_unambiguous(self):
        assert _pos_margin_candidates("1SH") == [(1, "SH")]

    def test_fraction_prefix(self):
        assert _pos_margin_candidates("31-1/4") == [(3, "1-1/4")]

    def test_ambiguous_two_vs_one_digit(self):
        # "11" -> both (11, "") and (1, "1") are structurally valid
        cands = _pos_margin_candidates("11")
        assert (11, "") in cands
        assert (1, "1") in cands

    def test_two_digit_with_margin(self):
        # "119" -> (11, "9") and (1, "19")? "19" is a valid whole-number margin
        cands = _pos_margin_candidates("119")
        assert (11, "9") in cands


# ---------------------------------------------------------------------------
# _resolve_section_positions — the permutation constraint
# ---------------------------------------------------------------------------

class TestResolveSectionPositions:
    def test_all_unambiguous(self):
        # A 3-horse section — positions must be 1, 2, 3
        out = _resolve_section_positions(["13/4", "2SH", "31-1/4"])
        positions = [p for p, _ in out]
        assert sorted(positions) == [1, 2, 3]

    def test_ambiguous_resolved_by_constraint(self):
        # "11" is ambiguous (1 or 11). With a 2-horse section and the other
        # horse clearly at position 2, "11" must resolve to position 1.
        out = _resolve_section_positions(["11", "2SH"])
        positions = [p for p, _ in out]
        assert 1 in positions and 2 in positions
        assert 11 not in positions

    def test_unresolvable_keeps_raw(self):
        # A prefix with no valid interpretation keeps position=None
        out = _resolve_section_positions(["XYZ"])
        assert out[0][0] is None


# ---------------------------------------------------------------------------
# _split_name_brand
# ---------------------------------------------------------------------------

class TestSplitNameBrand:
    def test_standard(self):
        assert _split_name_brand("PODIUM\xa0(H173)") == ("PODIUM", "H173")

    def test_multiword_name(self):
        assert _split_name_brand("WAH MAY WAI WAI\xa0(J193)") == ("WAH MAY WAI WAI", "J193")

    def test_no_brand(self):
        name, brand = _split_name_brand("SOME HORSE")
        assert name == "SOME HORSE"
        assert brand == ""


# ---------------------------------------------------------------------------
# parse_sectional_html — full page against a real captured fixture
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not FIXTURE.exists(), reason="sectional fixture missing")
class TestParseSectionalHtml:
    @pytest.fixture(scope="class")
    def parsed(self):
        html = FIXTURE.read_text()
        return parse_sectional_html(html, "2026/05/13", 1)

    def test_returns_data(self, parsed):
        assert parsed is not None
        assert parsed["race_date"] == "2026/05/13"
        assert parsed["race_no"] == 1

    def test_twelve_horses(self, parsed):
        assert len(parsed["horses"]) == 12

    def test_four_sections(self, parsed):
        # 2026/05/13 R1 is a 4-section race
        assert parsed["num_sections"] == 4
        assert len(parsed["race_section_times"]) == 4

    def test_winner_parsed_correctly(self, parsed):
        winner = parsed["horses"][0]
        assert winner["finishing_order"] == 1
        assert winner["horse_no"] == 1
        assert winner["horse_name"] == "PODIUM"
        assert winner["horse_brand"] == "H173"
        assert winner["final_time"] == pytest.approx(99.85, abs=0.01)
        # PODIUM ran 5th, 4th, 3rd, 1st
        positions = [s["position"] for s in winner["sections"]]
        assert positions == [5, 4, 3, 1]
        # last_400m == final section time
        assert winner["last_400m"] == winner["sections"][-1]["time"]

    def test_every_section_is_a_valid_permutation(self, parsed):
        """The core correctness check: each section's positions are 1..N."""
        n = len(parsed["horses"])
        for sec in range(1, parsed["num_sections"] + 1):
            positions = sorted(
                s["position"]
                for h in parsed["horses"]
                for s in h["sections"]
                if s["section"] == sec
            )
            assert positions == list(range(1, n + 1)), (
                f"Section {sec} positions not a 1..{n} permutation: {positions}"
            )

    def test_all_section_times_plausible(self, parsed):
        # HK ~400m section times sit roughly in 20-30s; first section can be longer
        for h in parsed["horses"]:
            for s in h["sections"]:
                assert 20.0 < s["time"] < 35.0, f"implausible section time {s['time']}"

    def test_sub_splits_only_on_later_sections(self, parsed):
        # HKJC only publishes 200m sub-splits for the final ~2 sections
        for h in parsed["horses"]:
            assert h["sections"][0]["sub_splits"] == [], "section 1 should have no sub-splits"
            # final section should have two 200m sub-splits
            assert len(h["sections"][-1]["sub_splits"]) == 2


class TestParseSectionalHtmlEdgeCases:
    def test_empty_html(self):
        assert parse_sectional_html("") is None

    def test_no_table(self):
        assert parse_sectional_html("<html><body>nothing</body></html>") is None
