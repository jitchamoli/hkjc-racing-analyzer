"""Protocol-conformance tests for core/ports.py.

Catches the "someone renamed a method core/ depends on" class of break.
runtime_checkable Protocols verify attribute existence, not signatures —
that is intentionally what we want here (a strict signature check would
be brittle on a healthy refactor).
"""
from __future__ import annotations

import inspect

import pytest

from core.ports import HorseHistoryStore, RaceDataSource


def test_horse_database_conforms_to_horse_history_store():
    """HorseDatabase must satisfy HorseHistoryStore at runtime."""
    from data.database import HorseDatabase

    db = HorseDatabase()
    try:
        assert isinstance(db, HorseHistoryStore), (
            "HorseDatabase no longer satisfies HorseHistoryStore — "
            "a method or attribute the core/ layer relies on has been "
            "removed or renamed. Either restore it on HorseDatabase or "
            "update the Protocol + audit core/analyzer.py callsites."
        )
    finally:
        db.close()


def test_hkjc_fetcher_conforms_to_race_data_source():
    """HKJCDataFetcher must satisfy RaceDataSource at runtime."""
    from core.fetcher import HKJCDataFetcher

    fetcher = HKJCDataFetcher()
    assert isinstance(fetcher, RaceDataSource), (
        "HKJCDataFetcher no longer satisfies RaceDataSource — a method "
        "the orchestration layer relies on has been removed or renamed."
    )


def test_race_data_source_methods_are_async():
    """All RaceDataSource methods must be coroutines.

    runtime_checkable does not validate sync-vs-async, so we check this
    structurally. A sync replacement would silently pass isinstance but
    break every `await` callsite.
    """
    from core.fetcher import HKJCDataFetcher

    fetcher = HKJCDataFetcher()
    async_methods = [
        'get_race_card', 'get_live_odds', 'get_place_odds',
        'get_results', 'get_meeting_dates', 'get_all_races', 'close',
    ]
    for name in async_methods:
        method = getattr(fetcher, name)
        assert inspect.iscoroutinefunction(method), (
            f"HKJCDataFetcher.{name} is not async — RaceDataSource Protocol "
            f"declares it as `async def`."
        )
