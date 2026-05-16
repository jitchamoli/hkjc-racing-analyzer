"""Phase 1 smoke test for the FastAPI wrapper.

Covers: /health, /predictions/{date}, /races/{date}/{race_no}.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import config
from app.main import app

client = TestClient(app)

SYNTHETIC_DATE_URL = "2999-12-31"
SYNTHETIC_DATE_FILE = "29991231"


@pytest.fixture
def synthetic_predictions():
    races = [
        {"race_number": 1, "venue": "Sha Tin", "horses": [{"number": 1, "name": "ALPHA"}]},
        {"race_number": 2, "venue": "Sha Tin", "horses": [{"number": 1, "name": "BETA"}]},
    ]
    path = config.PROJECT_ROOT / "results" / f"race_data_{SYNTHETIC_DATE_FILE}.json"
    path.write_text(json.dumps(races))
    yield races
    path.unlink(missing_ok=True)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_predictions_missing_returns_404():
    response = client.get("/predictions/1999-01-01")
    assert response.status_code == 404
    assert "No predictions" in response.json()["detail"]


def test_predictions_bad_format_returns_400():
    response = client.get("/predictions/2026_05_16")
    assert response.status_code == 400


def test_predictions_returns_full_card(synthetic_predictions):
    response = client.get(f"/predictions/{SYNTHETIC_DATE_URL}")
    assert response.status_code == 200
    body = response.json()
    assert body["date"] == SYNTHETIC_DATE_URL
    assert body["race_count"] == 2
    assert len(body["races"]) == 2
    assert body["races"][0]["race_number"] == 1


def test_single_race_returns_that_race(synthetic_predictions):
    response = client.get(f"/races/{SYNTHETIC_DATE_URL}/2")
    assert response.status_code == 200
    body = response.json()
    assert body["race_number"] == 2
    assert body["horses"][0]["name"] == "BETA"


def test_single_race_unknown_number_returns_404(synthetic_predictions):
    response = client.get(f"/races/{SYNTHETIC_DATE_URL}/99")
    assert response.status_code == 404


def test_single_race_bad_date_returns_400():
    response = client.get("/races/not-a-date/1")
    assert response.status_code == 400
