"""FastAPI wrapper for HKJC predictions (Clean Architecture Phase 1).

Read-only access to cached prediction artifacts produced by
`generate_predictions.py`. Re-generation is deliberately out of scope here:
a fetch+ML run takes ~30s and would block the request thread. A future
phase can add an async /generate endpoint backed by a job queue.
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI, HTTPException

import config

app = FastAPI(title="HKJC Racing Analyzer API", version="0.1.0")

_URL_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")
_RESULTS_DIR = config.PROJECT_ROOT / "results"


def _date_to_filename(date: str) -> str:
    m = _URL_DATE_RE.match(date)
    if not m:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {date}. Expected YYYY-MM-DD",
        )
    return f"{m.group(1)}{m.group(2)}{m.group(3)}"


def _load_races(date: str) -> list[dict[str, Any]]:
    date_str = _date_to_filename(date)
    # _with_results variant has scored outcomes; prefer it when present
    candidates = [
        _RESULTS_DIR / f"race_data_{date_str}_with_results.json",
        _RESULTS_DIR / f"race_data_{date_str}.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except json.JSONDecodeError as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Corrupt prediction file {path.name}: {e}",
                )
            return data.get("races", data) if isinstance(data, dict) else data
    raise HTTPException(
        status_code=404,
        detail=(
            f"No predictions found for {date}. "
            f"Run: python3 generate_predictions.py '{date.replace('-', '/')}'"
        ),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hkjc-racing-analyzer-api", "version": "0.1.0"}


@app.get("/predictions/{date}")
def get_predictions(date: str) -> dict[str, Any]:
    races = _load_races(date)
    return {"date": date, "race_count": len(races), "races": races}


@app.get("/races/{date}/{race_no}")
def get_race(date: str, race_no: int) -> dict[str, Any]:
    races = _load_races(date)
    for race in races:
        if race.get("race_number") == race_no:
            return race
    raise HTTPException(
        status_code=404,
        detail=f"Race {race_no} not found for {date}",
    )
