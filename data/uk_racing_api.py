"""The Racing API client for UK + IRE racing data.

Wraps https://api.theracingapi.com. Uses HTTP Basic Auth — credentials
loaded from config.RACING_API_USERNAME and config.RACING_API_PASSWORD
(both populated by .env via python-dotenv). HK pipeline does NOT use
this — it continues to scrape HKJC directly.

This module is a SCAFFOLD. Endpoint signatures will be finalised once
we have a sample response. Until then, MissingCredentialsError is
raised on instantiation if either credential is empty — that's a
deliberate fail-fast so we never silently make unauthenticated requests.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from requests.auth import HTTPBasicAuth

import config


class MissingCredentialsError(RuntimeError):
    """Raised when RACING_API_USERNAME or RACING_API_PASSWORD is missing.
    Caller should check this and surface the .env setup instructions to
    the user instead of crashing."""


class UKRacingAPI:
    """Thin wrapper around The Racing API.

    Usage (once endpoint paths are finalised):
        api = UKRacingAPI()                              # raises if creds missing
        results = api.get_results_for_date("2026-05-16") # returns list[dict]
        card = api.get_racecard(date, course, time)      # returns dict
    """

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.username = username or config.RACING_API_USERNAME
        self.password = password or config.RACING_API_PASSWORD
        self.base_url = (base_url or config.RACING_API_BASE_URL).rstrip("/")
        if not self.username or not self.password:
            raise MissingCredentialsError(
                "RACING_API_USERNAME or RACING_API_PASSWORD is not set. "
                "Copy .env.example to .env and paste both credentials "
                "from theracingapi.com Dashboard → API Keys."
            )
        self._auth = HTTPBasicAuth(self.username, self.password)

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Internal: authenticated GET request. Returns parsed JSON."""
        url = f"{self.base_url}{path}"
        r = requests.get(url, auth=self._auth, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    # --- Endpoints (stubs — implement once auth pattern + sample response known) ---

    def get_results_for_date(
        self,
        date: str,
        region: Optional[str] = None,
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch all race results for a single date (YYYY-MM-DD).

        Convenience wrapper for get_results_for_range(date, date, ...).
        """
        return self.get_results_for_range(date, date, region=region, page_size=page_size)

    def get_results_for_range(
        self,
        start_date: str,
        end_date: str,
        region: Optional[str] = None,
        page_size: int = 50,
    ) -> List[Dict[str, Any]]:
        """Fetch all race results in [start_date, end_date] inclusive.

        Both dates YYYY-MM-DD. The API has a hard 50-per-response cap;
        this auto-paginates via skip until all rows in the range are
        returned.

        region: optional filter ('gb' = UK only, 'ire' = Ireland, etc).
                None returns all regions.

        Note: the API rejects end_date >= today with 422 "end date cannot be
        in the future". We clamp silently to (today - 1 day) so callers like
        `--since-last` that compute end_date=today don't crash.
        """
        from datetime import date, timedelta
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        if end_date >= date.today().isoformat():
            end_date = yesterday
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": page_size,
            "skip": 0,
        }
        if region:
            params["region"] = region
        all_results: List[Dict[str, Any]] = []
        while True:
            data = self._get("/v1/results", params=params)
            page = data.get("results", []) or []
            all_results.extend(page)
            total = data.get("total")
            if total is None or len(all_results) >= total or not page:
                break
            params["skip"] += page_size
        return all_results

    def get_race_result(self, race_id: str) -> Dict[str, Any]:
        """Fetch a single race result by ID (e.g. 'rac_11931218')."""
        return self._get(f"/v1/results/{race_id}")

    def get_racecards(self, tier: str = "standard") -> List[Dict[str, Any]]:
        """Fetch upcoming racecards (today + tomorrow, depending on tier).

        tier: 'basic' / 'standard' / 'pro' — selects verbosity. Falls back
              gracefully if the account doesn't have the requested tier.
        """
        path = f"/v1/racecards/{tier}" if tier else "/v1/racecards"
        data = self._get(path)
        return data.get("racecards", []) or []

    def get_courses(self) -> List[Dict[str, Any]]:
        """Fetch the global course directory (id, name, region)."""
        data = self._get("/v1/courses")
        return data.get("courses", []) or []
