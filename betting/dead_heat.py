#!/usr/bin/env python3
"""Canonical dead-heat-aware winning-combination enumeration.

HKJC declares a dividend for EACH tied combination. A dead heat for 2nd means
there are TWO winning quinellas (1st+A and 1st+B) — 2026/06/07 R1 really has
two declared QUINELLA rows, '1,3' ($37) and '1,8' ($354) — and a box holding
all three horses collects BOTH.

This module is the single source of truth. Every scorer that decides whether a
box "hit" must go through `winning_combo_entries`; slicing a position-sorted
finisher list (`finishers[:2]`) picks one arbitrary member of a tied group and
silently books a genuine winner as a miss.

Extracted from scripts/validate_hk.py (code review 2026-08-03) because the
logic had been re-implemented — and had diverged — across stake_ab,
shadow_evaluate, gate_impact, track_qpl and track_longshot_trio.
"""

from itertools import combinations
from typing import Dict, List, Sequence


def winning_combo_entries(pos_to_entries: Dict[int, Sequence[dict]],
                          k: int) -> List[List[dict]]:
    """Every winning k-horse (unordered) combination, DEAD-HEAT aware.

    Horses finishing above the cutoff are certain; a tie that straddles the
    cutoff expands into one combination per way of filling the open slots.
    Returns [] when there aren't k finishers.
    """
    certain: List[dict] = []
    for pos in sorted(pos_to_entries):
        group = pos_to_entries[pos]
        if not group:
            continue
        if len(certain) + len(group) <= k:
            certain = certain + list(group)
            if len(certain) == k:
                return [certain]
        else:
            slots = k - len(certain)
            if slots <= 0:
                break
            return [certain + list(c) for c in combinations(group, slots)]
    return []


def group_by_position(finishers: Sequence[dict],
                      position_key: str = "actual_position") -> Dict[int, List[dict]]:
    """Bucket finishers by finishing position (ties share a bucket)."""
    pos_to_entries: Dict[int, List[dict]] = {}
    for entry in finishers:
        pos = entry.get(position_key)
        if not pos or int(pos) <= 0:
            continue
        pos_to_entries.setdefault(int(pos), []).append(entry)
    return pos_to_entries


def winning_name_sets(finishers: Sequence[dict], k: int,
                      position_key: str = "actual_position",
                      name_key: str = "name") -> List[set]:
    """Winning k-horse combinations as sets of horse names.

    Convenience wrapper for scorers that compare by name rather than by entry.
    """
    combos = winning_combo_entries(group_by_position(finishers, position_key), k)
    return [{e[name_key] for e in combo} for combo in combos]
