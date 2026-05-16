"""HKJC parser canary — touches the live HKJC website.

Catches the "HKJC silently changed their HTML" class of break that
CLAUDE.md flags as a known parser fragility. Default: SKIPPED. Run
manually before a meeting:

    RUN_CANARY=1 pytest tests/test_hkjc_parser_canary.py -v

Skips (does not fail) when HKJC is unreachable so it can be safely
wired into a pre-meeting workflow without false-positive paging.
Fails loudly when HKJC IS reachable but returns empty/malformed data
— that is the exact signal we want.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from core.fetcher import HKJCDataFetcher

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_CANARY") != "1",
    reason="HKJC parser canary — set RUN_CANARY=1 to run (touches live HKJC)",
)


async def _fetch_dates_and_first_race():
    fetcher = HKJCDataFetcher()
    try:
        dates = await fetcher.get_meeting_dates()
        if not dates:
            return [], None, None
        # Try the most recent few dates — the freshest one may be a future
        # meeting whose card is not yet published; iterate until one returns.
        last_err = None
        for d in dates[:5]:
            try:
                rc = await fetcher.get_race_card(d, 1)
            except Exception as e:
                last_err = e
                continue
            if rc and rc.horses:
                return dates, rc, None
        return dates, None, last_err
    finally:
        await fetcher.close()


def test_hkjc_parser_canary():
    try:
        dates, rc, last_err = asyncio.run(_fetch_dates_and_first_race())
    except Exception as e:
        pytest.skip(f"HKJC unreachable ({type(e).__name__}): {e}")

    assert dates, (
        "HKJC returned ZERO meeting dates. fetch_meeting_dates() / "
        "parse may be broken, or HKJC's fixture page format changed."
    )
    for d in dates[:3]:
        assert len(d) == 10 and d[4] == "/" and d[7] == "/", (
            f"Meeting date format wrong: {d!r}. Expected YYYY/MM/DD."
        )

    assert rc is not None, (
        f"Race 1 returned no horses on any of {len(dates[:5])} candidate "
        f"dates ({dates[:5]}). _parse_horse_table() may be reading the "
        f"wrong cell indices, or HKJC changed the table layout. "
        f"Last fetch error: {last_err}"
    )

    assert len(rc.horses) >= 4, (
        f"Suspiciously small field ({len(rc.horses)} horses) on {rc.date} "
        f"Race 1 — parser may be partially broken (some horses dropped)."
    )

    h = rc.horses[0]
    assert h.number > 0, (
        f"First horse has number={h.number!r} — parser cell index for "
        f"horse number is wrong."
    )
    assert h.name, (
        f"First horse has empty name — parser cell index for horse name "
        f"is wrong (was {h!r})."
    )
