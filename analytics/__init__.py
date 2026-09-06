"""Specialized racing analyzers (draw bias, pace, sectionals, etc.)."""

from typing import List


def parse_form_positions(last_6: str) -> List[int]:
    """Parse form string into list of finishing positions.

    Three formats reach this function:
      '1/3/10/2/5'    slash-separated  — live HKJC race card
      '4 11 9 5 0 8'  space-separated  — Renavon backfill (95% of race_results)
      '13425'         concatenated     — legacy, one digit per placing

    The space-separated case MUST split on whitespace before falling back to
    digit-by-digit: '4 11 9 5 0 8' read character-wise yields
    [4, 1, 1, 9, 5, 0, 8] — seven runs, and an 11th placing becomes two
    apparent WINS. 60% of stored last_6 values contain a two-digit placing
    (bug found 2026-09-06).

    Note: the legacy concatenated format still cannot represent positions >= 10;
    that is inherent to the format, not a parsing choice.
    """
    if not last_6:
        return []
    s = str(last_6).strip()
    if "/" in s:
        tokens = s.split("/")
    elif s.split()[1:]:                     # any internal whitespace
        tokens = s.split()
    else:
        return [int(ch) for ch in s if ch.isdigit()]
    return [int(t) for t in (t.strip() for t in tokens) if t.isdigit()]
