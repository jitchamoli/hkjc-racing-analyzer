"""Barrier-trial parser (core.fetcher.HKJCParser.parse_barrier_trials).

Rewritten 2026-06-07 for the current BTResult.aspx layout (one table per heat,
header: Horse|Jockey|Trainer|Draw|Gear|LBW|Running Position|Time; runners in
finishing order). Fixtures mirror the live structure verified that day.
"""
from core.fetcher import HKJCParser

_HEAT = """
<table>
  <tr><th>Horse</th><th>Jockey</th><th>Trainer</th><th>Draw</th>
      <th>Gear</th><th>LBW</th><th>Running Position</th><th>Time</th></tr>
  <tr><td>AEROVOLANIC (K123)</td><td>Z Purton</td><td>P C Ng</td><td>07</td>
      <td>B</td><td></td><td>1 1 1</td><td>1.10.80</td></tr>
  <tr><td>WITH A SMILE (V456)</td><td>J Orman</td><td>W Y So</td><td>05</td>
      <td></td><td>3/4L</td><td>2 2 2</td><td>1.10.93</td></tr>
  <tr><td>PRAY FOR JUSTICE (B789)</td><td>L Hewitson</td><td>D Eustace</td><td>02</td>
      <td>XB</td><td>5-1/2L</td><td>5 5 3</td><td>1.11.69</td></tr>
</table>
"""

# A second heat + a non-trial layout table that must be ignored.
_PAGE = f"""
<html><body>
<table><tr><td>Barrier Trials</td></tr><tr><td>navigation junk</td></tr></table>
{_HEAT}
<table>
  <tr><th>Horse</th><th>Jockey</th><th>Trainer</th><th>Draw</th>
      <th>Gear</th><th>LBW</th><th>Running Position</th><th>Time</th></tr>
  <tr><td>SZERYNG (A1)</td><td>B Avdulla</td><td>J Size</td><td>03</td>
      <td></td><td></td><td>1 1 1</td><td>0.57.46</td></tr>
</table>
</body></html>
"""


def test_parses_runners_in_finishing_order():
    rows = HKJCParser.parse_barrier_trials(_HEAT)
    assert len(rows) == 3
    assert [r["position"] for r in rows] == [1, 2, 3]
    assert [r["horse_name"] for r in rows] == [
        "AEROVOLANIC", "WITH A SMILE", "PRAY FOR JUSTICE"]  # brand codes stripped


def test_fields_extracted():
    rows = HKJCParser.parse_barrier_trials(_HEAT)
    winner, second, third = rows
    assert winner["jockey"] == "Z Purton" and winner["gear"] == "B"
    assert winner["margin"] == "" and winner["time"] == "1.10.80"
    assert second["margin"] == "3/4L" and second["trainer"] == "W Y So"
    assert third["draw"] == "02" and third["running_position"] == "5 5 3"


def test_multiple_heats_numbered_and_navtable_skipped():
    rows = HKJCParser.parse_barrier_trials(_PAGE)
    assert len(rows) == 4                       # 3 + 1, nav table ignored
    assert {r["heat"] for r in rows} == {1, 2}
    h2 = [r for r in rows if r["heat"] == 2]
    assert len(h2) == 1 and h2[0]["horse_name"] == "SZERYNG"


def test_trial_date_passthrough_and_empty():
    rows = HKJCParser.parse_barrier_trials(_HEAT, trial_date="2026/06/05")
    assert all(r["date"] == "2026/06/05" for r in rows)
    assert HKJCParser.parse_barrier_trials("") == []
    assert HKJCParser.parse_barrier_trials(None) == []
