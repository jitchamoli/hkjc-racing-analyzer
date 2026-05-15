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
        results = HKJCParser.parse_results(_RESULTS_HTML)
        for r in results:
            assert "(" not in r["name"], f"brand suffix leaked: {r['name']!r}"
            assert ")" not in r["name"]

    def test_clean_names_unchanged(self):
        results = HKJCParser.parse_results(_RESULTS_HTML)
        names = [r["name"] for r in results]
        assert "ALREADY CLEAN" in names

    def test_position_and_number_preserved(self):
        results = HKJCParser.parse_results(_RESULTS_HTML)
        # First row → position 1, horse_no 1, name PODIUM
        first = next(r for r in results if r["name"] == "PODIUM")
        assert first["position"] == 1
        assert first["number"] == 1

    def test_empty_html_returns_empty(self):
        assert HKJCParser.parse_results("") == []
