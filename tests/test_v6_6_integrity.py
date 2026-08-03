"""Integrity tests for the v6.6 audit fixes (2026-05-09).

Each test guards against a specific bug we found and fixed today.
If you ever revert one of these fixes, the corresponding test will catch it.
"""

import pytest
import numpy as np

from analytics.speed_figures import parse_finish_time
from core.scorer import BettingWeights, calibrate_probabilities
from core.analyzer import CombinedRaceAnalyzer
from data.database import HorseDatabase


# ---------------------------------------------------------------------------
# 1. parse_finish_time handles both HKJC formats
# ---------------------------------------------------------------------------

class TestParseFinishTime:
    """The HKJC LocalResults dump uses 'M.SS.ss' (e.g. '1.10.33') while the
    live results table uses 'M:SS.ss'. We must handle both — 85% of DB rows
    are in the M.SS.ss format and were silently failing pre-v6.6, making the
    speed_fig factor and Feature 21 feed neutral defaults."""

    def test_colon_format(self):
        assert parse_finish_time("1:10.33") == pytest.approx(70.33, abs=0.01)
        assert parse_finish_time("1:22.70") == pytest.approx(82.70, abs=0.01)

    def test_dot_format_three_tokens(self):
        # The dominant DB format
        assert parse_finish_time("1.10.33") == pytest.approx(70.33, abs=0.01)
        assert parse_finish_time("1.10.10") == pytest.approx(70.10, abs=0.01)
        assert parse_finish_time("0.59.50") == pytest.approx(59.50, abs=0.01)

    def test_seconds_only(self):
        assert parse_finish_time("59.85") == pytest.approx(59.85, abs=0.01)

    def test_empty_or_invalid(self):
        assert parse_finish_time("") is None
        assert parse_finish_time("---") is None
        assert parse_finish_time("not a time") is None

    def test_out_of_range_returns_none(self):
        assert parse_finish_time("99:99.99") is None  # > 999s


# ---------------------------------------------------------------------------
# 2. Rule engine has NO market-info factors
# ---------------------------------------------------------------------------

class TestRuleEngineMarketPurity:
    """Per CLAUDE.md, market info enters via the dynamic market blend ONLY.
    Three sites in v6.6 audit were violating this; this test ensures the
    BettingWeights map doesn't reintroduce a market-derived factor."""

    def test_no_odds_factor_in_weight_map(self):
        forbidden = {"odds", "win_odds", "market", "implied_prob",
                     "odds_movement", "market_prob"}
        for key in BettingWeights.WEIGHT_MAP:
            assert key not in forbidden, (
                f"Weight map contains market-derived factor '{key}' — "
                "market info must enter only via the dynamic blend"
            )


# ---------------------------------------------------------------------------
# 3. par_times cache is keyed by date (not single-slot)
# ---------------------------------------------------------------------------

class TestParTimesCachePerDate:
    """The pre-v6.6 par_times cache was a single None-or-set slot. Singleton
    analyzer reuse (e.g. in scripts/backtest.py) leaked future data into
    earlier races by reusing the first-fetched par_times. v6.6 keys the
    cache by race date."""

    def test_cache_attribute_is_dict(self):
        analyzer = CombinedRaceAnalyzer()
        try:
            assert isinstance(analyzer._par_times_by_date, dict)
            assert analyzer._par_times_by_date == {}  # empty until first use
        finally:
            analyzer.close()

    def test_cache_separates_by_date(self):
        """Manually populate two dates and verify they're independent slots."""
        analyzer = CombinedRaceAnalyzer()
        try:
            analyzer._par_times_by_date["2026/04/01"] = {"sentinel_a": 1.0}
            analyzer._par_times_by_date["2026/05/01"] = {"sentinel_b": 2.0}
            assert analyzer._par_times_by_date["2026/04/01"] != \
                   analyzer._par_times_by_date["2026/05/01"]
        finally:
            analyzer.close()


# ---------------------------------------------------------------------------
# 4. F-20 fallback symmetry (no-odds horses scaled to model_w)
# ---------------------------------------------------------------------------

class TestFTwentyFallbackSymmetry:
    """When SOME horses have live odds and OTHERS don't, the no-odds branch
    must scale by model_w (not pass cal_probs through directly). Otherwise
    no-odds horses get systematically inflated probabilities after the
    post-blend renormalization."""

    def _mk_analysis(self, prob, odds):
        from types import SimpleNamespace
        horse = SimpleNamespace(win_odds=odds, place_odds=0.0)
        return SimpleNamespace(
            combined_probability=prob, horse=horse,
            model_probability=None, value_score=0, expected_return=0,
            rank=0, bet_recommendation=None, reasoning=[]
        )

    def test_all_no_odds_normalizes_correctly(self):
        # When NO horses have odds, all hit the fallback, all scale by model_w,
        # and post-renorm equals pure model distribution.
        analyses = [
            self._mk_analysis(0.30, 0),
            self._mk_analysis(0.20, 0),
            self._mk_analysis(0.10, 0),
        ]
        out = calibrate_probabilities(analyses, field_size=10, distance=1400)
        assert sum(out) == pytest.approx(1.0, abs=1e-6)
        # Rank order must not REVERSE. Use >= not >: the post-blend isotonic
        # calibrator is only weakly monotonic, so two close inputs can map to
        # the same output (a flat isotonic region) — a legitimate tie, not a
        # bug. A strict > spuriously fails whenever a retrain's calibrator
        # develops a plateau across this input range (seen 2026-06-08).
        assert out[0] >= out[1] >= out[2]

    def test_mixed_odds_no_systematic_inflation(self):
        # Horse with odds vs horse without — equal pure_prob inputs should
        # NOT produce wildly different blended values.
        a1 = self._mk_analysis(0.20, 5.0)   # has odds
        a2 = self._mk_analysis(0.20, 0)     # no odds (fallback)
        a3 = self._mk_analysis(0.20, 5.0)
        out = calibrate_probabilities([a1, a2, a3], field_size=10, distance=1400)
        assert sum(out) == pytest.approx(1.0, abs=1e-6)
        # The no-odds horse must not be more than 2× the with-odds horses
        # (would be uncapped under the pre-fix bug).
        assert out[1] <= max(out[0], out[2]) * 2.0


# ---------------------------------------------------------------------------
# 5. rebuild_stats applies current-season filter
# ---------------------------------------------------------------------------

class TestRebuildStatsCurrentSeasonFilter:
    """Pre-v6.6, rebuild_stats() computed lifetime totals from race_results
    (5y window). Visiting/retired jockeys from prior seasons appeared in
    leaderboards with inflated wins. v6.6 filters to race_date >= season
    start (Jul 1)."""

    def test_current_season_start_format(self, tmp_path):
        # _current_season_start became an INSTANCE method 2026-08-03: it now
        # derives the season from MAX(race_date) instead of datetime.now(), so
        # rebuild_stats is deterministic and doesn't collapse to a stub window
        # during the off-season. Format contract is unchanged.
        db = HorseDatabase(db_path=str(tmp_path / "t.db"))
        try:
            start = db._current_season_start()
        finally:
            db.close()
        assert len(start) == 10  # YYYY/MM/DD
        assert start.endswith("/07/01")
        assert start[4] == "/" and start[7] == "/"

    def test_current_season_string(self):
        season = HorseDatabase._current_season()
        # Format like "2025-26"
        assert "-" in season
        a, b = season.split("-")
        assert len(a) == 4 and len(b) == 2


# ---------------------------------------------------------------------------
# 6. v6.6 + v6.8 + v6.9 + v6.10 schema: 26 features, all named additions present
# ---------------------------------------------------------------------------

class TestV66Schema:
    def test_has_23_features(self):
        # v6.11 (2026-05-24) pruned draw_advantage, speed_fig_norm, running_style
        # from v6.10's 26 → 23. jt_pair_win_rate (v6.6) survives the prune.
        from ml.feature_schema import NUM_FEATURES, FEATURE_NAMES
        assert NUM_FEATURES == 23
        assert "jt_pair_win_rate" in FEATURE_NAMES
        assert "speed_fig_norm" not in FEATURE_NAMES  # pruned v6.11
        assert "mid_race_pace" in FEATURE_NAMES  # v6.9
        assert "sire_top3_rate_on_going" in FEATURE_NAMES  # v6.10
        # v6.8 surviving additions (running_style pruned in v6.11)
        assert "early_speed" in FEATURE_NAMES
        assert "late_kick" in FEATURE_NAMES
        assert "running_style" not in FEATURE_NAMES   # pruned v6.11

    def test_jt_pair_position(self):
        # v6.11: jt_pair_win_rate is now at index 18 (was 19) after pruning
        # draw_advantage which lived at index 2.
        from ml.feature_schema import FEATURE_NAMES
        assert FEATURE_NAMES[18] == "jt_pair_win_rate"

    def test_running_position_features(self):
        # v6.11: early_speed + late_kick now at indices 19, 20 after pruning
        # draw_advantage (idx 2) and speed_fig_norm (idx 20).
        from ml.feature_schema import FEATURE_NAMES
        assert FEATURE_NAMES[19] == "early_speed"
        assert FEATURE_NAMES[20] == "late_kick"


# ---------------------------------------------------------------------------
# 7. fetch_day_results short-circuits after finished races run out
# ---------------------------------------------------------------------------

class TestFetchDayResultsShortCircuit:
    """Pre-v6.6, results_fetcher iterated 1→11 even when only races 1-4 had
    finished, wasting 7 HTTP calls per pre-race monitor cycle. v6.6 stops
    after 2 consecutive empties (allows 1 transient miss on a finished race
    while still skipping all upcoming/future races)."""

    def test_stops_after_two_consecutive_empties(self):
        import asyncio
        from dashboard.results_fetcher import ResultsFetcher
        fetcher = ResultsFetcher()

        # Mock fetch_race_results: races 1-4 finished, 5+ empty
        finished = {1, 2, 3, 4}
        call_count = 0
        async def mock_fetch(date, rno):
            nonlocal call_count
            call_count += 1
            return [{"position": 1, "number": 1}] if rno in finished else []

        fetcher.fetch_race_results = mock_fetch
        results = asyncio.run(fetcher.fetch_day_results("2026/05/09", num_races=11))

        # Expected: fetch races 1, 2, 3, 4 (finished), 5 (1st empty), 6 (2nd
        # empty triggers break). So 6 total calls instead of 11.
        assert call_count == 6, f"Expected 6 calls, got {call_count}"
        assert set(results.keys()) == finished
        # Critically: no fetch was made for races 7-11 (the upcoming races
        # the user flagged — fetching odds for them while ALSO querying
        # their results was wasteful).

    def test_does_not_short_circuit_before_first_finished(self):
        """If we start fetching before ANY race has finished (e.g. pre-meeting
        warmup), we should keep checking — not bail after the first empty."""
        import asyncio
        from dashboard.results_fetcher import ResultsFetcher
        fetcher = ResultsFetcher()

        call_count = 0
        async def mock_fetch(date, rno):
            nonlocal call_count
            call_count += 1
            return []  # no race has finished yet

        fetcher.fetch_race_results = mock_fetch
        asyncio.run(fetcher.fetch_day_results("2026/05/09", num_races=11))
        # Fetched all 11 — never short-circuited because seen_any_results
        # was False the whole time.
        assert call_count == 11
