"""Tests for HKJC data validation."""

import pytest
from unittest.mock import MagicMock

from data.validation import validate_horse_entry, validate_race_card


def _make_horse(number=1, name="Test Horse", draw=3, rating=65.0,
                jockey="Z Purton", trainer="J Size"):
    """Create a mock HorseEntry."""
    horse = MagicMock()
    horse.number = number
    horse.name = name
    horse.draw = draw
    horse.rating = rating
    horse.jockey = jockey
    horse.trainer = trainer
    return horse


def _make_card(race_number=1, distance="1200", horses=None):
    """Create a mock RaceCard."""
    card = MagicMock()
    card.race_number = race_number
    card.distance = distance
    card.horses = horses if horses is not None else [_make_horse(i) for i in range(1, 11)]
    return card


class TestValidateHorseEntry:
    def test_valid_entry_no_warnings(self):
        horse = _make_horse()
        warnings = validate_horse_entry(horse)
        assert warnings == []

    def test_missing_name(self):
        horse = _make_horse(name="")
        warnings = validate_horse_entry(horse)
        assert any("missing" in w.lower() or "short" in w.lower() for w in warnings)

    def test_short_name(self):
        horse = _make_horse(name="A")
        warnings = validate_horse_entry(horse)
        assert len(warnings) > 0

    def test_draw_out_of_range(self):
        horse = _make_horse(draw=20)
        warnings = validate_horse_entry(horse)
        assert any("draw" in w.lower() for w in warnings)

    def test_rating_too_high(self):
        horse = _make_horse(rating=200)
        warnings = validate_horse_entry(horse)
        assert any("rating" in w.lower() for w in warnings)

    def test_valid_rating_zero(self):
        """Rating of 0 is valid (unrated horse)."""
        horse = _make_horse(rating=0)
        warnings = validate_horse_entry(horse)
        assert not any("rating" in w.lower() for w in warnings)


class TestValidateRaceCard:
    def test_valid_card_no_warnings(self):
        card = _make_card()
        warnings = validate_race_card(card)
        assert warnings == []

    def test_too_few_horses(self):
        card = _make_card(horses=[_make_horse(1)])
        warnings = validate_race_card(card)
        assert any("only" in w.lower() or "horses" in w.lower() for w in warnings)

    def test_empty_card(self):
        card = _make_card(horses=[])
        warnings = validate_race_card(card)
        assert len(warnings) > 0

    def test_suspicious_distance(self):
        card = _make_card(distance="500")
        warnings = validate_race_card(card)
        assert any("distance" in w.lower() for w in warnings)

    def test_normal_distance(self):
        card = _make_card(distance="1600")
        warnings = validate_race_card(card)
        assert not any("distance" in w.lower() for w in warnings)

    def test_too_many_horses(self):
        horses = [_make_horse(i) for i in range(1, 20)]
        card = _make_card(horses=horses)
        warnings = validate_race_card(card)
        assert any("18" in w for w in warnings)
