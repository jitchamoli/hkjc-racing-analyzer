"""Sectional scraper must keep dead-heat horses.

Bug (money-path review 2026-06-05): parse_sectional_html filtered data rows on
`cells[0].isdigit()`, which rejects a dead-heat placing like "9 DH" — so BOTH
tied horses were dropped from the sectional table. This is the identical class
of bug fixed in data/scraper.py the same day; it explains why the 2 dead-heat
horses in 2026/06/03 R9 had no sectional_times even after the results fix.
"""
from data.sectional_scraper import parse_sectional_html


def _row(order, no, name, time="1:09.50"):
    return f"<tr><td>{order}</td><td>{no}</td><td>{name}</td><td>{time}</td></tr>"


_HTML = f"""
<html><body>
<table>
  <tr><th>Finishing Order</th><th>Horse No</th><th>Horse</th><th>Finish Time</th></tr>
  {_row("1", 4, "WITHOUT COMPARE (H1)")}
  {_row("9 DH", 2, "MISTER DAPPER (H2)")}
  {_row("9 DH", 7, "CLASS (H3)")}
  {_row("11", 5, "KEEFY (H4)")}
</table>
</body></html>
"""


def test_dead_heat_horses_kept():
    data = parse_sectional_html(_HTML, "2026/06/03", 9)
    assert data is not None
    nos = {h["horse_no"] for h in data["horses"]}
    assert nos == {4, 2, 7, 5}, f"dead-heat horses dropped: {nos}"


def test_dead_heat_finishing_order_numeric():
    data = parse_sectional_html(_HTML, "2026/06/03", 9)
    by_no = {h["horse_no"]: h["finishing_order"] for h in data["horses"]}
    assert by_no[2] == 9
    assert by_no[7] == 9
    assert by_no[5] == 11
