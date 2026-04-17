"""Specialized racing analyzers (draw bias, pace, sectionals, etc.)."""

from typing import List


def parse_form_positions(last_6: str) -> List[int]:
    """Parse form string into list of finishing positions.
    Handles both '1/3/10/2/5' (slash-separated) and legacy '13425' (digit-by-digit)."""
    if not last_6:
        return []
    s = str(last_6)
    if "/" in s:
        return [int(p) for p in s.split("/") if p.strip().isdigit()]
    else:
        return [int(ch) for ch in s if ch.isdigit()]
