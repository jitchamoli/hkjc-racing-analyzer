"""HKJC results-page parser — strips the brand suffix from horse names.

Regression: HKJC's results table collapses the name cell to e.g.
'PODIUM(H173)'. prediction_log stores 'PODIUM' (clean), so the prior
parser silently broke fill_prediction_results — UPDATE matched 0 rows
on every results fetch, leaving prediction_log results-empty for
months while race_results filled normally (auto_learn uses a different
scraper).
"""
from core.fetcher import HKJCParser


_RESULTS_HTML = """
<html><body>
<table class="tableBorderBlue">
  <tr><th>Pos</th><th>No</th><th>Horse</th><th>Jockey</th><th>Trainer</th><th>Odds</th></tr>
  <tr><td>1</td><td>1</td><td>PODIUM(H173)</td><td>D Eustace</td><td>M Newnham</td><td>8.9</td></tr>
  <tr><td>2</td><td>3</td><td>SETANTA(G095)</td><td>K Teetan</td><td>D J Hall</td><td>20.0</td></tr>
  <tr><td>3</td><td>12</td><td>WAH MAY WAI WAI(J193)</td><td>J Moreira</td><td>A S Cruz</td><td>13.0</td></tr>
  <tr><td>4</td><td>5</td><td>ALREADY CLEAN</td><td>Z Purton</td><td>J Size</td><td>3.6</td></tr>
  <tr><td>5</td><td>7</td><td>KIWI STAR (NZ)(K123)</td><td>H Bowman</td><td>C Fownes</td><td>6.5</td></tr>
  <tr><td>6</td><td>9</td><td>FRENCH FLYER (FR)</td><td>R Maia</td><td>P Yiu</td><td>9.0</td></tr>
</table>
</body></html>
"""


class TestParseResultsNameStripping:
    def test_strips_paren_brand_suffix(self):
        results = HKJCParser.parse_results(_RESULTS_HTML)
        names = [r["name"] for r in results]
        assert "PODIUM" in names
        assert "SETANTA" in names
        assert "WAH MAY WAI WAI" in names  # multi-word names preserved

    def test_no_brand_suffix_remains(self):
        import re
        results = HKJCParser.parse_results(_RESULTS_HTML)
        for r in results:
            # A brand suffix is a trailing paren containing a digit (e.g.
            # '(H173)'); country codes like '(NZ)' are letters-only and kept.
            assert not re.search(r"\([^)]*\d[^)]*\)", r["name"]), \
                f"brand suffix leaked: {r['name']!r}"

    def test_clean_names_unchanged(self):
        results = HKJCParser.parse_results(_RESULTS_HTML)
        names = [r["name"] for r in results]
        assert "ALREADY CLEAN" in names

    def test_country_code_preserved(self):
        """Brand code (has a digit) is stripped; country code (letters
        only) is kept — strip only trailing parens containing a digit."""
        results = HKJCParser.parse_results(_RESULTS_HTML)
        names = [r["name"] for r in results]
        assert "KIWI STAR (NZ)" in names   # (K123) stripped, (NZ) kept
        assert "FRENCH FLYER (FR)" in names  # no brand → country code intact

    def test_position_and_number_preserved(self):
        results = HKJCParser.parse_results(_RESULTS_HTML)
        # First row → position 1, horse_no 1, name PODIUM
        first = next(r for r in results if r["name"] == "PODIUM")
        assert first["position"] == 1
        assert first["number"] == 1

    def test_empty_html_returns_empty(self):
        assert HKJCParser.parse_results("") == []

    def test_finish_time_not_a_trainer_name(self):
        """Regression (10-expert review): finish_time used to read texts[4],
        which is the Trainer column in the results layout — so it was populated
        with a trainer name. The canonical finish time comes from the sectional
        scraper; parse_results must not emit a trainer name as finish_time."""
        results = HKJCParser.parse_results(_RESULTS_HTML)
        trainers = {"M Newnham", "D J Hall", "A S Cruz", "J Size", "C Fownes", "P Yiu"}
        for r in results:
            assert r["finish_time"] not in trainers, (
                f"finish_time leaked a trainer name: {r['finish_time']!r}"
            )

    def test_win_odds_still_parsed_from_last_cell(self):
        """The finish_time fix must not disturb win_odds (last cell)."""
        results = HKJCParser.parse_results(_RESULTS_HTML)
        podium = next(r for r in results if r["name"] == "PODIUM")
        assert podium["win_odds"] == 8.9
