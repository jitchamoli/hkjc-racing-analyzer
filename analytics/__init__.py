"""Specialized racing analyzers (draw bias, pace, sectionals, etc.)."""

from typing import List


def parse_form_positions(last_6: str) -> List[int]:
    """Parse form string into list of finishing positions.
    Handles '1/3/10/2/5' (slash-separated) and legacy '13425' (digit-by-digit).
    Note: legacy digit-by-digit format cannot represent positions >= 10."""
    if not last_6:
        return []
    s = str(last_6).strip()
    if "/" in s:
        results = []
        for p in s.split("/"):
            p = p.strip()
            if p.isdigit():
                results.append(int(p))
        return results
    else:
        return [int(ch) for ch in s if ch.isdigit()]
