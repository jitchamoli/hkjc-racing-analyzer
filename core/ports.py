"""Domain ports — Protocols the core/ layer depends on.

This file is the documented boundary between domain logic and
infrastructure. Anything in core/ that needs to talk to the outside
world (DB, HKJC website) is expressed here as a Protocol; the concrete
implementations live in adapter modules (`data/database.py`,
`core/fetcher.py`).

Why have this when there is only one implementation?
- Documents the actual surface area core/ uses, so future refactors
  (Postgres swap, AU/UK race adapter, test mocks) are localised.
- Catches accidental method renames at protocol-conformance test time.
- Makes the dependency direction explicit: core/ depends on Protocols
  here, never on `data.database` directly (type-wise).

These Protocols are `runtime_checkable` so tests can assert conformance
via isinstance(). Runtime checking only verifies attribute existence,
not signatures — that is sufficient for catching the "someone deleted
a method core/ relies on" class of break.
"""
from __future__ import annotations

import sqlite3
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

if TYPE_CHECKING:
    from core.fetcher import RaceCard, RaceDay


@runtime_checkable
class HorseHistoryStore(Protocol):
    """Read-side access to per-horse history and jockey/trainer stats.

    `conn` is intentionally exposed because two callers reach past the
    abstraction with raw SQL: `analytics/speed_figures.compute_par_times`
    and the LIKE-based fallback in
    `core.analyzer.BettingStrategy._score_person_from_db`. Cleaning those
    leaks is a follow-up; the Protocol documents the current surface
    rather than an aspirational one.
    """

    conn: sqlite3.Connection

    def get_horse_record(self, name: str) -> Optional[Dict[str, Any]]: ...
    def get_horse_history(self, name: str, limit: int = 5) -> List[Dict[str, Any]]: ...
    def get_horse_cd_history(
        self, name: str, venue: str, distance: int
    ) -> List[Dict[str, Any]]: ...
    def get_horse_going_record(self, name: str, going: str) -> Dict[str, Any]: ...
    def get_rolling_stats(self, name: str, n_races: int = 5) -> Dict[str, Any]: ...
    def get_beaten_margins(self, name: str, n_races: int = 5) -> Dict[str, Any]: ...
    def get_last_jockey(self, name: str) -> str: ...
    def get_jockey_recent_form(
        self, jockey: str, n_rides: int = 10
    ) -> Dict[str, Any]: ...
    def get_trainer_recent_form(
        self, trainer: str, n_runners: int = 10
    ) -> Dict[str, Any]: ...
    def get_jockey_stats(self, name: str) -> Optional[Dict[str, Any]]: ...
    def get_trainer_stats(self, name: str) -> Optional[Dict[str, Any]]: ...
    def close(self) -> None: ...


@runtime_checkable
class RaceDataSource(Protocol):
    """Async read-side access to race cards, odds, and the meeting calendar.

    `HKJCDataFetcher` is today's only implementation. An AU/UK fetcher
    would implement this same surface to plug into existing entry points
    (`scripts/generate.py`, `monitor/`, `dashboard/results_fetcher.py`)
    without touching core/.
    """

    async def get_race_card(
        self, race_date: str, race_no: int
    ) -> Optional["RaceCard"]: ...
    async def get_live_odds(
        self, race_date: str, race_no: int
    ) -> Dict[int, float]: ...
    async def get_place_odds(
        self, race_date: str, race_no: int
    ) -> Dict[int, float]: ...
    async def get_results(
        self, race_date: str, race_no: int
    ) -> List[Dict[str, Any]]: ...
    async def get_meeting_dates(self) -> List[str]: ...
    async def get_all_races(self, race_date: str) -> "RaceDay": ...
    async def close(self) -> None: ...
