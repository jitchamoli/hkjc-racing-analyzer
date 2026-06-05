"""Scraper must keep dead-heat finishers.

Bug (data-integrity audit 2026-06-05): HKJC renders a dead-heat placing with
a "DH" marker in the Pla cell, e.g. "9 DH". The results parser gated the row
on `re.match(r"^\\d{1,2}$", pos_text)`, which rejects anything but bare digits
— so BOTH dead-heating horses were silently dropped from race_results (lost
from training data, prior-history lookups, and dividend reconciliation).

Real case: 2026/06/03 HV R9 had a dead heat for 9th (MISTER DAPPER #2 and
CLASS #7). The lenient JSON path (`parse_results`, _safe_int) kept them; the
scraper path dropped both, leaving 10 of 12 runners in the DB.

A whole-DB scan found ~243 races with `max(finish_position) > row_count`,
the signature of dropped finishers — mostly single dead-heat partners.
"""
from data.scraper import _parse_tables


_HEADER = ["Pla", "Horse No", "Horse", "Jockey", "Trainer", "Act.Wt",
           "Horse Wt", "Dr", "LBW", "RunPos", "Time", "Odds"]


def _row(pla, no, name, odds):
    return [pla, str(no), name, "J Doe", "T Smith", "126",
            "1100", "5", "2", "3 3 3", "1:09.50", str(odds)]


def _table_with_dead_heat(pla_text):
    """A 4-horse results table where 2nd and 3rd dead-heat for 2nd."""
    return [[
        _HEADER,
        _row("1", 1, "WINNER", 3.5),
        _row(pla_text, 2, "DEADHEAT A", 6.0),
        _row(pla_text, 3, "DEADHEAT B", 8.0),
        _row("4", 4, "LAST ONE", 20.0),
    ]]


class TestDeadHeatParsing:
    def test_dead_heat_with_dh_suffix_space(self):
        race = _parse_tables(_table_with_dead_heat("2 DH"), "2026/06/03", 9)
        names = {e["name"] for e in race["entries"]}
        assert {"DEADHEAT A", "DEADHEAT B"} <= names, f"dead-heat horses dropped: {names}"
        assert len(race["entries"]) == 4

    def test_dead_heat_with_dh_suffix_nospace(self):
        race = _parse_tables(_table_with_dead_heat("2DH"), "2026/06/03", 9)
        assert len(race["entries"]) == 4

    def test_dead_heat_positions_preserved(self):
        race = _parse_tables(_table_with_dead_heat("2 DH"), "2026/06/03", 9)
        by_name = {e["name"]: e["actual_position"] for e in race["entries"]}
        assert by_name["DEADHEAT A"] == 2
        assert by_name["DEADHEAT B"] == 2

    def test_plain_digits_still_work(self):
        race = _parse_tables(_table_with_dead_heat("2"), "2026/06/03", 9)
        assert len(race["entries"]) == 4

    def test_venue_hint_used_when_header_lacks_venue(self):
        """HKJC moved the venue into a 1-row table dropped by the >=3-row
        filter, so the header no longer proves a venue. A page-level hint
        must fill it instead of storing UNKNOWN."""
        race = _parse_tables(_table_with_dead_heat("2 DH"), "2026/06/03", 9,
                             venue_hint="Happy Valley")
        assert race["venue"] == "Happy Valley"

    def test_real_header_venue_beats_hint(self):
        """If the header DOES prove a venue, it wins over the hint."""
        tables = _table_with_dead_heat("2 DH")
        # A real (>=3-row) header table that proves Sha Tin.
        tables.insert(0, [["Sha Tin"], ["Class 4 - 1200M"], ["Going: Good"]])
        race = _parse_tables(tables, "2026/06/03", 9, venue_hint="Happy Valley")
        assert race["venue"] == "Sha Tin"

    def test_header_and_garbage_rows_still_skipped(self):
        # A non-result row (no leading digit) must NOT become an entry.
        tbl = [[
            _HEADER,
            _row("1", 1, "WINNER", 3.5),
            _row("WV", 2, "WITHDRAWN", 0),   # withdrawn → no placing
            _row("2", 3, "RUNNER UP", 6.0),
        ]]
        race = _parse_tables(tbl, "2026/06/03", 9)
        names = {e["name"] for e in race["entries"]}
        assert "WITHDRAWN" not in names
        assert names == {"WINNER", "RUNNER UP"}
