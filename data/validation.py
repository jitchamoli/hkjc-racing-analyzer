#!/usr/bin/env python3
"""Schema validation for parsed HKJC data.

Called after parsing race cards to catch HKJC site layout changes early.
Returns warnings (not fatal) so predictions still run on partial data.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def validate_horse_entry(entry) -> List[str]:
    """Validate a single HorseEntry. Returns list of warning strings."""
    warnings = []
    if not entry.name or len(entry.name) < 2:
        warnings.append(f"Horse #{entry.number}: missing or very short name '{entry.name}'")
    if entry.number < 1 or entry.number > 18:
        warnings.append(f"Horse #{entry.number}: number out of expected range 1-18")
    if entry.draw < 0 or entry.draw > 18:
        warnings.append(f"Horse #{entry.number} {entry.name}: draw {entry.draw} out of range 0-18")
    if entry.rating and (entry.rating < 0 or entry.rating > 140):
        warnings.append(f"Horse #{entry.number} {entry.name}: rating {entry.rating} suspicious (expected 0-140)")
    if entry.jockey and len(entry.jockey) < 2:
        warnings.append(f"Horse #{entry.number} {entry.name}: jockey name too short '{entry.jockey}'")
    if entry.trainer and len(entry.trainer) < 2:
        warnings.append(f"Horse #{entry.number} {entry.name}: trainer name too short '{entry.trainer}'")
    return warnings


def validate_race_card(card) -> List[str]:
    """Validate a full RaceCard after parsing. Returns list of warning strings."""
    warnings = []
    if len(card.horses) < 2:
        warnings.append(f"Race {card.race_number}: only {len(card.horses)} horses parsed (expected 2+)")
    if len(card.horses) > 18:
        warnings.append(f"Race {card.race_number}: {len(card.horses)} horses parsed (expected max 18)")

    distance = 0
    try:
        distance = int(card.distance) if card.distance else 0
    except (ValueError, TypeError):
        pass
    if distance > 0 and (distance < 800 or distance > 2400):
        warnings.append(f"Race {card.race_number}: distance {card.distance}m suspicious (expected 800-2400)")

    for horse in card.horses:
        warnings.extend(validate_horse_entry(horse))

    if warnings:
        for w in warnings:
            logger.warning(f"[Validation] {w}")
    return warnings
