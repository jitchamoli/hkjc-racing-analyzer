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
        """Fetch all race results for a given date (YYYY-MM-DD).

        region: optional filter — 'gb' for UK only, 'ire' for Ireland, etc.
                None returns all regions (UK + IRE + global).
        page_size: API hard cap appears to be 50 per response; helper
                   auto-paginates via skip param until all rows are read.

        Returns: list of race dicts (see api/results sample response in
        memory). Each race has top-level fields (course, off, race_name,
        class, dist, going, etc.) plus a 'runners' list with full
        per-runner detail (horse, sp, sp_dec, position, jockey, trainer,
        or, btn, weight_lbs, headgear, comment, pedigree, etc.).
        """
        params: Dict[str, Any] = {"start_date": date, "limit": page_size, "skip": 0}
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
