r"""Scraper must preserve the running-position call separators.

Bug (code review 2026-08-03): `scrape_meeting_results` built cell text with
`c.get_text(strip=True)` — no separator argument. HKJC renders the
Running-Position cell as separate inline nodes ("<td>13<br>12<br>10</td>"),
so the per-call positions were concatenated into "131210". The downstream
normalisation in `_parse_tables` (`re.sub(r"[^\d\s]", " ", rp)`) cannot
restore separators that were already deleted.

`ml.features.FeatureEngineering.parse_running_position` then falls back to a
per-character parse, so "131210" -> [1,3,1,2,1,0] instead of [13,12,10]. That
INVERTS `early_speed` (a top-5 importance ML feature): a backmarker that ran
13th at the first call is re-encoded as a front-runner.

Measured impact before the fix: every meeting scraped from 2026/05/03 onward
was concatenated (2,576 rows, 0 space-separated), 197 of 207 races had
physically-impossible inconsistent call-counts between finishers of the same
race, and 554 rows parsed out a position 0 (which cannot exist).

`core/fetcher.py::parse_barrier_trials` already did this correctly with
`get_text(" ", strip=True)`; this pins the results scraper to the same rule.
"""
from bs4 import BeautifulSoup

from data.scraper import _extract_table_data, _parse_tables
from ml.features import FeatureEngineering


_HEADER_CELLS = ["Pla", "Horse No", "Horse", "Jockey", "Trainer", "Act.Wt",
                 "Horse Wt", "Dr", "LBW", "RunPos", "Time", "Odds"]


def _results_html(runpos_html: str) -> str:
    """A minimal 3-row HKJC-shaped results table (>=3 rows to pass the filter)."""
    header = "".join(f"<th>{h}</th>" for h in _HEADER_CELLS)
    def row(pla, no, name, rp):
        return (f"<tr><td>{pla}</td><td>{no}</td><td>{name}</td><td>J Doe</td>"
                f"<td>T Smith</td><td>126</td><td>1100</td><td>5</td><td>2</td>"
                f"<td>{rp}</td><td>1:09.50</td><td>4.5</td></tr>")
    return (f"<table><tr>{header}</tr>"
            f"{row(1, 1, 'FRONT RUNNER', '1<br/>1<br/>1')}"
            f"{row(10, 2, 'BACK MARKER', runpos_html)}"
            f"</table>")


class TestRunningPositionSeparators:
    def test_extract_table_data_preserves_br_separated_calls(self):
        """<br>-separated calls must not be concatenated into one token."""
        soup = BeautifulSoup(_results_html("13<br/>12<br/>10"), "html.parser")
        tables = _extract_table_data(soup)

        backmarker_row = tables[0][2]
        assert backmarker_row[9] == "13 12 10", (
            f"running-position cell was mangled: {backmarker_row[9]!r}"
        )

    def test_parse_tables_stores_space_separated_running_position(self):
        """End-to-end: HTML -> cells -> race dict keeps the separators."""
        soup = BeautifulSoup(_results_html("13<br/>12<br/>10"), "html.parser")
        race = _parse_tables(_extract_table_data(soup), "2026/08/03", 1)

        entry = next(e for e in race["entries"] if e["name"] == "BACK MARKER")
        assert entry["running_position"] == "13 12 10"

    def test_double_digit_calls_do_not_invert_early_speed(self):
        """The consequence that matters: a closer must not read as a leader."""
        soup = BeautifulSoup(_results_html("13<br/>12<br/>10"), "html.parser")
        race = _parse_tables(_extract_table_data(soup), "2026/08/03", 1)
        entry = next(e for e in race["entries"] if e["name"] == "BACK MARKER")

        calls = FeatureEngineering.parse_running_position(entry["running_position"])
        assert calls == [13, 12, 10], f"calls mis-parsed: {calls}"

        early_speed, _, _ = FeatureEngineering.compute_running_position_features(
            entry["running_position"], field_size=14
        )
        # early_speed: 0.0 = front-runner, 1.0 = closer.
        assert early_speed > 0.5, (
            f"a horse 13th at the first call scored as a front-runner "
            f"(early_speed={early_speed:.3f})"
        )

    def test_nbsp_separated_calls_also_parse(self):
        """HKJC sometimes uses &nbsp; instead of <br> between calls."""
        soup = BeautifulSoup(_results_html("13&nbsp;12&nbsp;10"), "html.parser")
        race = _parse_tables(_extract_table_data(soup), "2026/08/03", 1)

        entry = next(e for e in race["entries"] if e["name"] == "BACK MARKER")
        assert entry["running_position"] == "13 12 10"

    def test_single_digit_calls_still_parse(self):
        """Regression guard: the common single-digit case is unchanged."""
        soup = BeautifulSoup(_results_html("2<br/>1<br/>1"), "html.parser")
        race = _parse_tables(_extract_table_data(soup), "2026/08/03", 1)

        entry = next(e for e in race["entries"] if e["name"] == "BACK MARKER")
        assert entry["running_position"] == "2 1 1"

    def test_other_columns_unaffected_by_separator_change(self):
        """Adding a separator must not corrupt the plain single-value cells."""
        soup = BeautifulSoup(_results_html("13<br/>12<br/>10"), "html.parser")
        race = _parse_tables(_extract_table_data(soup), "2026/08/03", 1)

        entry = next(e for e in race["entries"] if e["name"] == "BACK MARKER")
        assert entry["jockey"] == "J Doe"
        assert entry["trainer"] == "T Smith"
        assert entry["finish_time"] == "1:09.50"
        assert entry["draw"] == 5
        assert entry["horse_weight"] == 1100
