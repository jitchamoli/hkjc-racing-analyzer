#!/usr/bin/env python3
"""
HKJC Horse Racing Analyzer v1.0.0 - Real-Time Data Edition
Uses HKJCDataFetcher for live race cards, odds, and results.
Combined rule-based + ML prediction with real market data.
"""

import asyncio
import re
import sys
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datetime import datetime
import logging

from core.fetcher import HKJCDataFetcher, HorseEntry, RaceCard
from core.ports import HorseHistoryStore

from core.predictor import (
    HKJCRacingMLModel,
    FeatureEngineering,
    CombinedRacingAnalyzer as _MLAnalyzer,
    MLPrediction,
    create_combined_prediction,
    ML_AVAILABLE,
)
from core.scorer import compute_weighted_score, BettingWeights
from core.recommender import assign_bet_recommendations
from ml.features import parse_age_int

from betting.optimizer import BettingOptimizer, BettingReportGenerator

from analytics.draw_bias import DrawBiasAnalyzer
from analytics.pace import PaceAnalyzer
from analytics.sectionals import SectionalAnalyzer
from analytics.barrier_trials import BarrierTrialAnalyzer
from analytics.course_distance import CourseDistanceAnalyzer
from analytics.class_movement import ClassMovementAnalyzer
from analytics.gear_change import GearChangeAnalyzer
from analytics.weight import WeightAnalyzer
from analytics.going_pref import GoingPreferenceAnalyzer
from analytics.rolling_form import RollingFormAnalyzer
from analytics.beaten_margins import BeatenMarginAnalyzer
from analytics.jockey_change import JockeyChangeAnalyzer
from analytics.speed_figures import (
    compute_par_times, adjusted_speed_figure, normalize_speed_figure,
)
from data.odds_tracker import OddsMovementAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain models (kept compatible with v4 for betting_optimizer / ml_predictor)
# ---------------------------------------------------------------------------

@dataclass
class Horse:
    number: int
    name: str
    draw: int
    jockey: str
    trainer: str
    weight: str
    rating: float
    win_odds: float
    place_odds: float
    last_6: str
    career: str = ""
    track: str = ""
    distance: str = ""
    # HKJC Tips Index data
    tips_odds: float = 0.0
    initial_tips_index: float = 0.0
    raceday_tips_index: float = 0.0
    is_top_pick: bool = False
    # Advanced analytics
    running_style: str = ""
    last_400m_avg: float = 0.0
    cd_wins: int = 0
    cd_runs: int = 0
    trial_position: int = 0
    opening_odds: float = 0.0
    gear: str = ""
    # New high-impact fields
    days_since_last_run: int = 0
    rating_change: int = 0
    horse_weight: int = 0
    weight_change: int = 0
    age: str = ""
    season_stakes: str = ""
    import_cat: str = ""
    sex: str = ""


@dataclass
class Race:
    race_number: int
    venue: str
    date: str
    time: str
    distance: str
    class_type: str
    going: str
    track: str = ""
    course: str = ""
    horses: List[Horse] = field(default_factory=list)
    odds_updated_at: str = ""


@dataclass
class HorseAnalysis:
    horse: Horse
    rank: int
    rule_score: float
    rule_confidence: str
    ml_prediction: MLPrediction
    combined_probability: float
    combined_confidence: str
    value_score: float
    expected_return: float
    bet_recommendation: str
    reasoning: List[str]
    prediction_breakdown: Dict
    # F-19: snapshot of pre-swap model probability for calibration analytics.
    # combined_probability may be mutated by BANKER/market-fav rank swaps;
    # model_probability preserves the original value so accuracy.py measures
    # the model, not human overrides.
    model_probability: float = None


# ---------------------------------------------------------------------------
# Conversion: HorseEntry / RaceCard → Horse / Race
# ---------------------------------------------------------------------------

def convert_race(rc: RaceCard) -> Race:
    """Convert fetcher RaceCard into analyzer Race."""
    horses = []
    for h in rc.horses:
        # Estimate place odds from win odds if not available
        place_odds = h.place_odds
        if place_odds <= 0 and h.win_odds > 0:
            if h.win_odds < 3:
                place_odds = 1.1
            elif h.win_odds < 5:
                place_odds = 1.3
            elif h.win_odds < 10:
                place_odds = 1.8
            elif h.win_odds < 20:
                place_odds = 3.0
            else:
                place_odds = 5.0

        horses.append(Horse(
            number=h.number,
            name=h.name,
            draw=h.draw,
            jockey=h.jockey,
            trainer=h.trainer,
            weight=h.weight,
            rating=h.rating,
            win_odds=h.win_odds,
            place_odds=place_odds,
            last_6=h.last_6,
            distance=rc.distance,
            tips_odds=h.tips_odds,
            initial_tips_index=h.initial_tips_index,
            raceday_tips_index=h.raceday_tips_index,
            is_top_pick=h.is_top_pick,
            running_style=h.running_style,
            last_400m_avg=h.last_400m_avg,
            cd_wins=h.cd_wins,
            cd_runs=h.cd_runs,
            trial_position=h.trial_position,
            opening_odds=h.opening_odds,
            gear=h.gear,
            days_since_last_run=h.days_since_last_run,
            rating_change=h.rating_change,
            horse_weight=h.horse_weight,
            weight_change=h.weight_change,
            age=getattr(h, 'age', ''),
            season_stakes=getattr(h, 'season_stakes', ''),
            import_cat=getattr(h, 'import_cat', ''),
            sex=getattr(h, 'sex', ''),
        ))

    return Race(
        race_number=rc.race_number,
        venue=rc.venue,
        date=rc.date,
        time=rc.time,
        distance=rc.distance,
        class_type=rc.class_type,
        going=rc.going,
        track=rc.track,
        course=rc.course,
        horses=horses,
        odds_updated_at=rc.odds_updated_at,
    )


# ---------------------------------------------------------------------------
# Betting Strategy (same logic as v4)
# ---------------------------------------------------------------------------

class BettingStrategy:
    # Weight constants now live in core/scorer.py:BettingWeights

    @staticmethod
    def analyze_form(last_6: str) -> tuple:
        if not last_6 or len(last_6) < 1:
            return 0.0, "No recent form data"

        # Position scores — handles positions beyond 6th
        def pos_score(p):
            if p == 1: return 100
            if p == 2: return 60
            if p == 3: return 40
            if p == 4: return 25
            if p == 5: return 15
            if p == 6: return 10
            if p <= 9: return 5
            return 2  # 10th or worse

        # Parse positions — handles both "1/2/3/10/5" and legacy "12345" formats
        positions = []
        if "/" in str(last_6):
            for part in str(last_6).split("/"):
                part = part.strip()
                if part.isdigit():
                    positions.append(int(part))
        else:
            # Legacy single-digit format (may be inaccurate for 10+)
            for ch in str(last_6):
                if ch.isdigit():
                    positions.append(int(ch))

        if not positions:
            return 0.0, "No recent form data"

        # Recency-weighted scoring: most recent race gets highest weight
        # Weights: [1.0, 0.85, 0.7, 0.55, 0.4, 0.3, ...] (most recent first)
        # HKJC form reads LEFT = oldest, RIGHT = most recent
        weights = []
        n = len(positions)
        for i in range(n):
            w = 0.3 + 0.7 * (i / max(1, n - 1))  # 0.3 for oldest → 1.0 for most recent
            weights.append(w)

        weighted_total = 0.0
        weight_sum = 0.0
        wins = 0
        places = 0  # 2nd or 3rd
        recent_win = False

        for i, pos in enumerate(positions):
            score = pos_score(pos)
            weighted_total += score * weights[i]
            weight_sum += weights[i]
            if pos == 1:
                wins += 1
                if i >= n - 2:  # last 2 runs
                    recent_win = True
            elif pos in (2, 3):
                places += 1

        avg_score = weighted_total / max(0.01, weight_sum)

        # Bonus for recent winner (last 1-2 starts)
        if recent_win:
            avg_score = min(100, avg_score + 15)

        if wins >= 3:
            reason = f"Strong form: {wins}W in last {n} ({avg_score:.0f})"
        elif recent_win:
            reason = f"Recent winner: {wins}W/{places}P ({avg_score:.0f})"
        elif wins >= 1:
            reason = f"Winner in form: {wins}W/{places}P ({avg_score:.0f})"
        elif places >= 2:
            reason = f"Consistent placer: {places}P ({avg_score:.0f})"
        else:
            reason = f"Moderate form ({avg_score:.0f})"

        return avg_score, reason

    @staticmethod
    def _score_person_from_db(name: str, role: str, db, stats_getter,
                              table: str, count_field: str,
                              tiers: dict, tier_scores: tuple,
                              thresholds: list, score_cap: tuple,
                              place_bonuses: list, default_score: float) -> tuple:
        """Shared scoring logic for jockeys and trainers.

        Args:
            thresholds: list of (min_wr, base_score, multiplier, label) tuples, highest first.
            score_cap: (max_score, min_score) for clamping.
            place_bonuses: list of (min_place_rate, bonus, cap) tuples, highest first.
        """
        if db and name:
            stats = stats_getter(name)
            if not stats:
                try:
                    c = db.conn.cursor()
                    c.execute(f"SELECT * FROM {table} WHERE name LIKE ?",
                              (f'%{name.split()[-1]}%',))
                    row = c.fetchone()
                    if row:
                        stats = dict(row)
                except Exception as e:
                    logger.warning(f"DB lookup failed for {role} {name}: {e}")

            if stats and stats.get(count_field, 0) >= 5:
                win_rate = stats.get('win_rate', 0)
                place_rate = stats.get('place_rate', 0)
                count = stats[count_field]
                wins = stats.get('wins', 0)

                score, label = thresholds[-1][1], thresholds[-1][3]  # default to lowest
                for min_wr, base, mult, lbl in thresholds:
                    if win_rate >= min_wr:
                        score = min(score_cap[0], base + win_rate * mult)
                        label = lbl
                        break

                for min_pr, bonus, cap in place_bonuses:
                    if place_rate >= min_pr:
                        score = min(cap, score + bonus)
                        break

                score = min(score_cap[0], max(score_cap[1], score))
                reason = f"{label} {role} - {name} ({wins}W/{count}R, {win_rate:.0%}W)"
                return score, reason

        # Fallback: tier-based defaults
        name_lower = name.lower() if name else ""
        surname = name_lower.split()[-1] if name_lower.split() else ""

        if surname in tiers.get(1, set()):
            return tier_scores[0], f"Elite {role} - {name}"
        elif surname in tiers.get(2, set()):
            return tier_scores[1], f"Top {role} - {name}"
        elif surname in tiers.get(3, set()):
            return tier_scores[2], f"Experienced {role} - {name}"
        return default_score, f"{role.title()}: {name}"

    @staticmethod
    def analyze_jockey(jockey: str, db=None) -> tuple:
        """Score jockey using real DB stats when available, else fallback."""
        return BettingStrategy._score_person_from_db(
            name=jockey, role="jockey", db=db,
            stats_getter=db.get_jockey_stats if db else lambda x: None,
            table="jockey_stats", count_field="rides",
            # Tiers refreshed 2026-05-17 against current-season DB stats:
            # - "atzeni" promoted tier 2 → tier 1 (527 rides at 10.8% wr)
            # - "teetan" demoted tier 2 → tier 3 (552 rides at 7.4%, per
            #   `project_may06_race_learnings` Teetan blind spot)
            # DB-stats path takes precedence when available; this is fallback only.
            tiers={
                1: {"moreira", "purton", "bowman", "moore", "mcdonald", "atzeni"},
                2: {"badel", "hewitson", "shinn", "avdulla", "bentley",
                    "guyon", "chau"},
                3: {"teetan", "schofield", "ho", "kingscote", "ferraris",
                    "borges", "leung"},
            },
            tier_scores=(90.0, 78.0, 70.0),
            thresholds=[
                (0.20, 80, 75, "Elite"),
                (0.14, 78, 50, "Top"),
                (0.08, 65, 80, "Solid"),
                (0.04, 55, 100, "Average"),
                (0.00, 40, 200, "Struggling"),
            ],
            score_cap=(97, 30),
            place_bonuses=[(0.35, 5, 97), (0.25, 3, 95)],
            default_score=60.0,
        )

    @staticmethod
    def analyze_trainer(trainer: str, db=None) -> tuple:
        """Score trainer using real DB stats when available, else fallback."""
        return BettingStrategy._score_person_from_db(
            name=trainer, role="trainer", db=db,
            stats_getter=db.get_trainer_stats if db else lambda x: None,
            table="trainer_stats", count_field="runners",
            # Tiers refreshed 2026-05-17 against current-season DB stats:
            # - "shum" promoted tier 2 → tier 1 (445 runners at 12.1% wr,
            #   #1 by win rate; user memory notes Shum top 2 seasons)
            # - "fownes", "newnham", "lui" all 10-11% wr — added to tier 1
            # - "lor" demoted tier 1 → tier 2 (7.7% wr this season, behind hayes
            #   at 8.8%; per 2026-05-17 audit domain-expert review)
            # DB-stats path takes precedence when available; this is fallback only.
            tiers={
                1: {"size", "cruz", "shum", "fownes", "newnham", "lui"},
                2: {"hall", "whyte", "hayes", "richards", "man", "yiu", "lor"},
                3: {"ting", "eustace", "crawford", "yip", "tsui", "mo"},
            },
            tier_scores=(88.0, 75.0, 68.0),
            thresholds=[
                (0.18, 78, 80, "Elite"),
                (0.12, 72, 60, "Top"),
                (0.06, 58, 100, "Solid"),
                (0.02, 45, 200, "Average"),
                (0.00, 35, 300, "Struggling"),
            ],
            score_cap=(95, 25),
            place_bonuses=[(0.30, 4, 95), (0.20, 2, 93)],
            default_score=58.0,
        )

    @staticmethod
    def analyze_draw(draw: int, distance: str) -> tuple:
        dist = int(distance) if distance and distance.isdigit() else 1200

        if dist <= 1000:
            ideal_range = (1, 4)
        elif dist <= 1400:
            ideal_range = (1, 6)
        else:
            ideal_range = (4, 10)

        if draw == 0:
            return 50.0, "No draw data"

        if ideal_range[0] <= draw <= ideal_range[1]:
            # Within ideal range — score based on closeness to center
            center = (ideal_range[0] + ideal_range[1]) / 2
            dist_from_center = abs(draw - center) / max(1, (ideal_range[1] - ideal_range[0]) / 2)
            score = 95 - dist_from_center * 15  # 95 at center, ~80 at edges
            return score, f"Good barrier ({draw})"
        elif draw > 12:
            return 45.0, f"Wide draw ({draw})"
        elif draw > ideal_range[1]:
            return 60.0, f"Outside ideal range ({draw})"
        else:
            return 65.0, f"Average barrier ({draw})"

    @staticmethod
    def analyze_tips(horse: "Horse", all_horses: List["Horse"]) -> tuple:
        """Score horse based on HKJC Tips Index data.
        Lower tips index = more favoured by experts.
        Returns (score 0-100, reasoning string)."""
        if horse.initial_tips_index <= 0:
            return 0.0, "No tips data"

        ti = horse.initial_tips_index

        # Collect all tips indices to rank relatively
        all_ti = [h.initial_tips_index for h in all_horses if h.initial_tips_index > 0]
        if not all_ti:
            return 0.0, "No tips data"

        min_ti = min(all_ti)
        max_ti = max(all_ti)

        # Normalize: lower tips index = higher score
        if max_ti > min_ti:
            score = (1 - (ti - min_ti) / (max_ti - min_ti)) * 100
        else:
            score = 50.0

        # Bonus for top pick
        if horse.is_top_pick:
            score = min(100, score + 15)

        # Bonus for odds movement (tips odds vs current odds)
        odds_move = ""
        if horse.tips_odds > 0 and horse.win_odds > 0:
            if horse.win_odds < horse.tips_odds:
                odds_move = " (odds shortening - money coming)"
            elif horse.win_odds > horse.tips_odds * 1.2:
                odds_move = " (odds drifting)"

        reason_parts = [f"Tips Index: {ti:.1f}"]
        if horse.is_top_pick:
            reason_parts.append("HKJC TOP PICK")
        if odds_move:
            reason_parts.append(odds_move.strip())

        return score, " | ".join(reason_parts)

    @staticmethod
    def get_rule_confidence(score: float) -> str:
        if score >= 75:
            return "HIGH"
        elif score >= 60:
            return "MEDIUM"
        elif score >= 45:
            return "LOW"
        else:
            return "VERY_LOW"


# ---------------------------------------------------------------------------
# Combined Race Analyzer
# ---------------------------------------------------------------------------

class CombinedRaceAnalyzer:
    def __init__(self):
        self.strategy = BettingStrategy()
        self.ml_analyzer = _MLAnalyzer()
        self.draw_analyzer = DrawBiasAnalyzer()
        self.pace_analyzer = PaceAnalyzer()
        self.sectional_analyzer = SectionalAnalyzer()
        self.trial_analyzer = BarrierTrialAnalyzer()
        self.cd_analyzer = CourseDistanceAnalyzer()
        self.odds_movement = OddsMovementAnalyzer()
        self.class_analyzer = ClassMovementAnalyzer()
        self.gear_analyzer = GearChangeAnalyzer()
        self.weight_analyzer = WeightAnalyzer()
        self.going_analyzer = GoingPreferenceAnalyzer()
        self.rolling_analyzer = RollingFormAnalyzer()
        self.margin_analyzer = BeatenMarginAnalyzer()
        self.jockey_change_analyzer = JockeyChangeAnalyzer()
        # Typed as the HorseHistoryStore Protocol (core/ports.py) so the
        # domain does not depend on the concrete adapter; HorseDatabase
        # duck-types into it.
        self._horse_db: Optional[HorseHistoryStore] = None
        try:
            from data.database import HorseDatabase
            self._horse_db = HorseDatabase()
        except Exception as e:
            logger.warning(f"Could not connect to horse database: {e}")
        # Lazy-computed par-time cache for speed figures, keyed by race date.
        # Recomputing per horse would be O(N) per horse * N par buckets, so we
        # cache. Keyed by date so that singleton analyzer instances (e.g. in
        # backtest) don't reuse stale par_times across different race dates —
        # a date-mismatched cache would leak future data into earlier races.
        self._par_times_by_date = {}

    def close(self):
        """Close DB connections held by this analyzer."""
        if self._horse_db:
            try:
                self._horse_db.close()
            except Exception:
                pass

    def analyze_horse(
        self, horse: Horse, race: Race, all_horses: List[Horse] = None,
        all_results: List = None, pace_scenario: Dict = None,
    ) -> HorseAnalysis:
        factors = {}
        reasoning = []
        # Normalize distance to string (some paths pass int, some str)
        distance = str(race.distance) if race.distance else "1200"
        # Cache horse record once — used by going_pref and career_place sections
        horse_record = self._horse_db.get_horse_record(horse.name) if self._horse_db else None

        # --- Core factors (same as before) ---

        # Rating — now scored relative to field average (pbovard63 approach)
        field_ratings = [h.rating for h in (all_horses or []) if h.rating > 0]
        field_avg_rating = sum(field_ratings) / len(field_ratings) if field_ratings else 0
        if horse.rating > 0 and field_avg_rating > 0:
            rating_diff = horse.rating - field_avg_rating
            # Score based on how far above/below field average
            base = min(100, horse.rating)
            if rating_diff >= 10:
                factors["rating"] = min(100, base + 10)
                rating_label = f"Rating {horse.rating:.0f} (+{rating_diff:.0f} vs field avg {field_avg_rating:.0f})"
            elif rating_diff >= 3:
                factors["rating"] = min(100, base + 5)
                rating_label = f"Rating {horse.rating:.0f} (+{rating_diff:.0f} vs field)"
            elif rating_diff >= -3:
                factors["rating"] = base
                rating_label = f"Rating {horse.rating:.0f} (field avg {field_avg_rating:.0f})"
            elif rating_diff >= -8:
                factors["rating"] = max(10, base - 5)
                rating_label = f"Rating {horse.rating:.0f} ({rating_diff:.0f} vs field)"
            else:
                factors["rating"] = max(5, base - 10)
                rating_label = f"Rating {horse.rating:.0f} ({rating_diff:.0f} below field)"
        else:
            factors["rating"] = min(100, horse.rating)
            rating_label = f"Rating {horse.rating:.0f}"
        reasoning.append(rating_label)

        form_score, form_reason = self.strategy.analyze_form(horse.last_6)
        # No-form fallback intentionally leaves form_score=0 so the factor is
        # excluded from the weighted average and the remaining factors
        # renormalize. Market odds must NOT enter the rule engine — they
        # come in only via the dynamic market blend (see CLAUDE.md).

        # Supplement form for debut/Griffin horses with trial data
        if self.trial_analyzer.is_debut_horse(horse.last_6, horse.rating, race.class_type):
            trial_results = []
            if horse.trial_position > 0:
                trial_results = [{"position": horse.trial_position, "margin": "", "time": "", "date": ""}]
            adj_form, trial_reason = self.trial_analyzer.supplement_form_score(
                form_score, trial_results, horse.last_6, horse.rating, race.class_type
            )
            if trial_reason:
                form_score = adj_form
                form_reason = trial_reason

        factors["form"] = form_score
        reasoning.append(form_reason)

        # Market odds: shown in reasoning for context but NOT scored here
        # (market signal enters via 35-55% dynamic market blend in calibration)
        if horse.win_odds > 0:
            odds_prob = 1 / horse.win_odds
            reasoning.append(
                f"Market: {odds_prob*100:.1f}% implied @ ${horse.win_odds:.1f}"
            )
        else:
            reasoning.append("No live odds available")

        jockey_score, jockey_reason = self.strategy.analyze_jockey(horse.jockey, db=self._horse_db)
        factors["jockey"] = jockey_score
        reasoning.append(jockey_reason)

        trainer_score, trainer_reason = self.strategy.analyze_trainer(horse.trainer, db=self._horse_db)
        factors["trainer"] = trainer_score
        reasoning.append(trainer_reason)

        # --- Enhanced draw with rail bias ---
        draw_score, draw_reason = self.draw_analyzer.analyze_draw_with_rail(
            horse.draw, distance, race.course, race.venue
        )
        factors["draw"] = draw_score
        reasoning.append(draw_reason)

        # --- HKJC Tips Index ---
        tips_score, tips_reason = self.strategy.analyze_tips(
            horse, all_horses or []
        )
        factors["tips"] = tips_score
        if tips_score > 0:
            reasoning.append(tips_reason)

        # --- NEW: Pace Map ---
        style = horse.running_style or self.pace_analyzer.estimate_style_from_form(
            horse.last_6, horse.draw,
            int(distance) if distance and distance.isdigit() else 1200
        )
        if pace_scenario:
            pace_score, pace_reason = self.pace_analyzer.analyze_pace(style, pace_scenario)
        else:
            pace_score, pace_reason = 50.0, f"{style} - No pace data"
        factors["pace"] = pace_score
        reasoning.append(pace_reason)

        # --- NEW: Sectional Times / Finishing Speed ---
        if horse.last_400m_avg > 0:
            sect_score, sect_reason = self.sectional_analyzer.analyze_finishing_speed(
                [{"last_400m": horse.last_400m_avg}]
            )
        else:
            sect_score, sect_reason = self.sectional_analyzer.estimate_from_form(
                horse.last_6, getattr(horse, 'best_time', '')
            )
        factors["sectional"] = sect_score
        if sect_score > 0:
            reasoning.append(f"Finish: {sect_reason}")

        # --- Course & Distance Form (with winning trip replication) ---
        cd_score, cd_reason = 0.0, ""
        if self._horse_db:
            cd_history = self._horse_db.get_horse_cd_history(
                horse.name, race.venue or '', int(race.distance or 1200))
            if cd_history:
                cd_score, cd_reason = self.cd_analyzer.analyze_cd_form(
                    cd_history, race.venue or '', int(race.distance or 1200),
                    current_date=race.date if hasattr(race, 'date') else None)
            elif horse.cd_runs > 0:
                # Fallback to race card data if no DB history
                cd_wins = horse.cd_wins
                cd_runs = horse.cd_runs
                if cd_wins >= 1:
                    cd_score = 78.0
                    cd_reason = f"C&D winner ({cd_wins}W from {cd_runs})"
                else:
                    cd_score = 50.0
                    cd_reason = f"C&D: {cd_runs} runs, no wins"
        factors["cd_form"] = cd_score
        if cd_score > 0:
            reasoning.append(f"C&D: {cd_reason}")

        # Odds movement: shown in reasoning for context but NOT scored here
        # (enters via market blend in calibrate_probabilities + steam/drift dashboard badges)
        if horse.opening_odds > 0 and horse.win_odds > 0:
            _, move_reason = self.odds_movement.analyze_movement(
                horse.opening_odds, horse.win_odds
            )
            if 'steam' in move_reason.lower() or 'drift' in move_reason.lower():
                reasoning.append(f"Odds: {move_reason}")
        elif horse.tips_odds > 0 and horse.win_odds > 0:
            _, move_reason = self.odds_movement.analyze_from_tips(
                horse.tips_odds, horse.win_odds
            )
            if 'steam' in move_reason.lower() or 'drift' in move_reason.lower():
                reasoning.append(f"Odds: {move_reason}")

        # --- Class Movement (now wired to DB history) ---
        class_score = 50.0
        class_reason = "No previous class data"
        if self._horse_db and race.class_type:
            horse_hist = self._horse_db.get_horse_history(horse.name, limit=5)
            if horse_hist:
                class_score, class_reason = self.class_analyzer.analyze_class_movement(
                    horse_hist, race.class_type
                )
        factors["class_movement"] = class_score
        if class_score != 50.0:
            reasoning.append(f"Class: {class_reason}")

        # --- NEW: Gear Changes (with debutant + last-run diff detection) ---
        prior_gear = ""
        is_debutant = False
        if self._horse_db:
            history = self._horse_db.get_horse_history(horse.name, limit=1)
            if not history:
                is_debutant = True
            else:
                prior_gear = history[0].get('gear', '') or ''
        gear_score, gear_reason = self.gear_analyzer.analyze_gear(
            horse.gear, prior_gear=prior_gear, is_first_time=is_debutant
        )
        factors["gear"] = gear_score
        if gear_reason != "No gear changes":
            reasoning.append(f"Gear: {gear_reason}")

        # --- NEW: Speed Figures (Beyer-style) ---
        # Compute par-time-normalised speed figure from the horse's most
        # recent race. Adjusts for carried weight (Benter +0.1s/lb above
        # 120). Score: 0 = at par, +5 = elite, -5 = poor. Mapped to 0-100
        # via normalize_speed_figure (0.0 → 0, 0.5 → 50, 1.0 → 100).
        if self._horse_db and horse.name:
            try:
                cache_key = race.date if race.date else ''
                par_times = self._par_times_by_date.get(cache_key)
                if par_times is None:
                    par_times = compute_par_times(
                        self._horse_db.conn,
                        before_date=race.date if race.date else None,
                    )
                    self._par_times_by_date[cache_key] = par_times
                last_race = self._horse_db.get_horse_history(horse.name, limit=1)
                if last_race:
                    lr = last_race[0]
                    sf = adjusted_speed_figure(
                        lr.get('finish_time') or '',
                        lr.get('distance') or 0,
                        lr.get('going') or '',
                        lr.get('class_type') or '',
                        lr.get('weight') or 120,
                        par_times,
                    )
                    if sf is not None:
                        # sf typically -5 to +5; normalize to 0-1 then scale to 0-100
                        sf_score = normalize_speed_figure(sf) * 100
                        factors["speed_fig"] = sf_score
                        if sf >= 1.0:
                            reasoning.append(f"Speed: +{sf:.1f} above par (last run)")
                        elif sf <= -1.0:
                            reasoning.append(f"Speed: {sf:.1f} below par (last run)")
            except Exception as e:
                logger.debug(f"Speed figure computation failed for {horse.name}: {e}")

        # --- NEW: Weight Analysis ---
        field_weights = [h.weight for h in (all_horses or []) if h.weight]
        weight_score, weight_reason = self.weight_analyzer.analyze_weight(
            horse.weight, int(horse.rating) if horse.rating else 0, field_weights
        )
        factors["weight_carried"] = weight_score
        if weight_reason != "Weight data unavailable":
            reasoning.append(f"Weight: {weight_reason}")

        # --- NEW: Days Since Last Run (Freshness) ---
        days = horse.days_since_last_run
        if days > 0:
            # Optimal: 14-35 days. Too fresh (<14) or too long (>60) is risky
            if 14 <= days <= 35:
                fresh_score = 75.0
                fresh_reason = f"Ideal spacing ({days} days)"
            elif 7 <= days < 14:
                fresh_score = 60.0
                fresh_reason = f"Quick backup ({days} days)"
            elif 35 < days <= 60:
                fresh_score = 55.0
                fresh_reason = f"Fair break ({days} days)"
            elif 60 < days <= 120:
                fresh_score = 40.0
                fresh_reason = f"Long absence ({days} days) — fitness query"
            else:
                fresh_score = 30.0
                fresh_reason = f"Very long layoff ({days} days)"
        else:
            fresh_score = 0.0
            fresh_reason = ""
        factors["freshness"] = fresh_score
        if fresh_score > 0:
            reasoning.append(f"Freshness: {fresh_reason}")

        # --- NEW: Rating Change (Trending) ---
        rtg_chg = horse.rating_change
        if rtg_chg != 0:
            if rtg_chg >= 5:
                trend_score = 80.0
                trend_reason = f"Rating up {rtg_chg}pts — improving fast"
            elif rtg_chg > 0:
                trend_score = 65.0
                trend_reason = f"Rating up {rtg_chg}pts — on the rise"
            elif rtg_chg >= -3:
                trend_score = 45.0
                trend_reason = f"Rating down {rtg_chg}pts — slight decline"
            else:
                trend_score = 30.0
                trend_reason = f"Rating down {rtg_chg}pts — declining"
        else:
            trend_score = 0.0
            trend_reason = ""
        factors["rating_trend"] = trend_score
        if trend_score > 0:
            reasoning.append(f"Trend: {trend_reason}")

        # --- NEW: Season Stakes (Class Indicator) ---
        stakes_val = 0
        if horse.season_stakes and horse.season_stakes not in ("-", "0", ""):
            stakes_val = int(re.sub(r'[^\d]', '', horse.season_stakes) or '0')
        if stakes_val > 0:
            if stakes_val >= 3_000_000:
                stakes_score = 80.0
                stakes_reason = f"Top earner (${stakes_val:,.0f})"
            elif stakes_val >= 1_000_000:
                stakes_score = 65.0
                stakes_reason = f"Solid earner (${stakes_val:,.0f})"
            elif stakes_val >= 300_000:
                stakes_score = 50.0
                stakes_reason = f"Moderate earner (${stakes_val:,.0f})"
            else:
                stakes_score = 35.0
                stakes_reason = f"Low earner (${stakes_val:,.0f})"
        else:
            stakes_score = 0.0
            stakes_reason = ""
        factors["season_stakes"] = stakes_score
        if stakes_score > 0:
            reasoning.append(f"Earnings: {stakes_reason}")

        # --- NEW: Going (Track Condition) Preference ---
        going_score = 0.0
        going_reason = ""
        if race.going and self._horse_db:
            going_record = self._horse_db.get_horse_going_record(horse.name, race.going)
            total_runs = horse_record.get('total_starts', 0) if horse_record else 0
            going_score, going_reason = self.going_analyzer.analyze_going_preference(
                going_record, race.going, total_runs
            )
        factors["going_pref"] = going_score
        if going_score > 0:
            reasoning.append(f"Going: {going_reason}")

        # --- NEW: Rolling Form Stats ---
        rolling_score = 0.0
        rolling_reason = ""
        if self._horse_db:
            rolling_stats = self._horse_db.get_rolling_stats(horse.name, n_races=5)
            if rolling_stats.get('runs', 0) >= 2:
                rolling_score, rolling_reason = self.rolling_analyzer.analyze_rolling_form(rolling_stats)
        factors["rolling_form"] = rolling_score
        if rolling_score > 0:
            reasoning.append(f"Form trend: {rolling_reason}")

        # --- NEW: Beaten Margins ---
        margin_score = 0.0
        margin_reason = ""
        if self._horse_db:
            margin_stats = self._horse_db.get_beaten_margins(horse.name, n_races=5)
            if margin_stats.get('runs', 0) > 0:
                margin_score, margin_reason = self.margin_analyzer.analyze_margins(margin_stats)
        factors["margin"] = margin_score
        if margin_score > 0:
            reasoning.append(f"Margin: {margin_reason}")

        # --- NEW: Jockey Change Detection ---
        jc_score = 0.0
        jc_reason = ""
        if self._horse_db:
            last_jockey = self._horse_db.get_last_jockey(horse.name)
            if last_jockey:
                jc_score, jc_reason = self.jockey_change_analyzer.analyze_jockey_change(
                    horse.jockey, last_jockey
                )
        factors["jockey_change"] = jc_score
        if jc_score > 0:
            reasoning.append(jc_reason)

        # --- NEW: Jockey Recent Form (Hot Streak) ---
        jf_score = 0.0
        jf_reason = ""
        if self._horse_db and horse.jockey:
            jf = self._horse_db.get_jockey_recent_form(horse.jockey, n_rides=10)
            if jf.get('rides', 0) >= 3:
                wr = jf['win_rate']
                pr = jf['place_rate']
                if jf['hot']:
                    jf_score = 85.0
                    jf_reason = f"Jockey HOT: {horse.jockey} {jf['wins']}W/{jf['rides']}R ({wr:.0%})"
                elif wr >= 0.15:
                    jf_score = 65.0
                    jf_reason = f"Jockey solid: {horse.jockey} {jf['wins']}W/{jf['rides']}R ({wr:.0%})"
                elif pr >= 0.30:
                    jf_score = 55.0
                    jf_reason = f"Jockey placing: {horse.jockey} {jf['places']}P/{jf['rides']}R"
                else:
                    jf_score = 35.0
                    jf_reason = f"Jockey cold: {horse.jockey} {jf['wins']}W/{jf['rides']}R ({wr:.0%})"
        factors["jockey_form"] = jf_score
        if jf_score > 0:
            reasoning.append(jf_reason)

        # --- NEW: Trainer Recent Form (Hot Streak) ---
        tf_score = 0.0
        tf_reason = ""
        if self._horse_db and horse.trainer:
            tf = self._horse_db.get_trainer_recent_form(horse.trainer, n_runners=10)
            if tf.get('runners', 0) >= 3:
                wr = tf['win_rate']
                if tf['hot']:
                    tf_score = 80.0
                    tf_reason = f"Trainer HOT: {horse.trainer} {tf['wins']}W/{tf['runners']}R ({wr:.0%})"
                elif wr >= 0.12:
                    tf_score = 60.0
                    tf_reason = f"Trainer solid: {horse.trainer} {tf['wins']}W/{tf['runners']}R"
                else:
                    tf_score = 38.0
                    tf_reason = f"Trainer quiet: {horse.trainer} {tf['wins']}W/{tf['runners']}R"
        factors["trainer_form"] = tf_score
        if tf_score > 0:
            reasoning.append(tf_reason)

        # --- NEW: Field Size Impact ---
        # Pure structural signal — small fields favour any in-form horse,
        # large fields are noisier. Per-horse differentiation across the
        # field comes from the market blend, NOT from win_odds in the rule
        # engine (CLAUDE.md: market info enters via the dynamic market
        # blend only — prevents triple-counting).
        field_n = len(all_horses) if all_horses else 0
        if field_n > 0:
            if field_n <= 8:
                fs_score = 65.0
                fs_reason = f"Small field ({field_n}) — predictable shape"
            elif field_n <= 11:
                fs_score = 55.0
                fs_reason = f"Medium field ({field_n})"
            else:
                fs_score = 48.0
                fs_reason = f"Large field ({field_n}) — chaos risk"
        else:
            fs_score = 0.0
            fs_reason = ""
        factors["field_size"] = fs_score
        if fs_score > 0:
            reasoning.append(f"Field: {fs_reason}")

        # --- NEW: Career Place Rate ---
        cp_score = 0.0
        cp_reason = ""
        if self._horse_db:
            hr = horse_record  # reuse cached result from going_pref section
            if hr and hr.get('total_starts', 0) >= 3:
                starts = hr['total_starts']
                places = hr.get('total_places', 0) or 0
                wins = hr.get('total_wins', 0) or 0
                place_rate = places / starts if starts else 0
                win_rate = wins / starts if starts else 0
                if win_rate >= 0.25:
                    cp_score = 85.0
                    cp_reason = f"Career: {wins}W/{places}P from {starts} ({win_rate:.0%}W)"
                elif place_rate >= 0.40:
                    cp_score = 72.0
                    cp_reason = f"Career: {places}P from {starts} ({place_rate:.0%} top3)"
                elif place_rate >= 0.25:
                    cp_score = 58.0
                    cp_reason = f"Career: {places}P from {starts} ({place_rate:.0%} top3)"
                elif starts >= 5:
                    cp_score = 35.0
                    cp_reason = f"Career: {places}P from {starts} ({place_rate:.0%} top3)"
                else:
                    cp_score = 45.0
                    cp_reason = f"Career: {starts} starts"
        factors["career_place"] = cp_score
        if cp_score > 0:
            reasoning.append(cp_reason)

        # --- NEW: Age vs Field (pbovard63 approach) ---
        age_score = 0.0
        age_reason = ""
        horse_age = parse_age_int(horse.age)  # handles '5' and DB '5.0'
        if horse_age is not None:
            field_ages = [a for a in (parse_age_int(h.age) for h in (all_horses or []))
                          if a is not None]
            if field_ages:
                avg_age = sum(field_ages) / len(field_ages)
                # In HK, 3-4yo = improving, 5-6 = peak, 7+ = declining
                if horse_age <= 4:
                    age_base = 65.0  # young improver
                elif horse_age <= 6:
                    age_base = 60.0  # peak years
                else:
                    age_base = 45.0  # older
                # Relative to field
                age_diff = horse_age - avg_age
                if abs(age_diff) >= 2:
                    if horse_age <= 4 and age_diff < 0:
                        age_score = 72.0
                        age_reason = f"Young improver ({horse_age}yo vs field avg {avg_age:.0f})"
                    elif horse_age >= 7 and age_diff > 0:
                        age_score = 35.0
                        age_reason = f"Older in field ({horse_age}yo vs avg {avg_age:.0f})"
                    else:
                        age_score = age_base
                        age_reason = f"Age {horse_age} (field avg {avg_age:.0f})"
                else:
                    age_score = age_base
                    age_reason = f"Age {horse_age} (field avg {avg_age:.0f})"
        factors["age"] = age_score
        if age_score > 0:
            reasoning.append(age_reason)

        # --- Weighted score calculation (dynamic) ---
        rule_score = compute_weighted_score(factors)

        rule_confidence = self.strategy.get_rule_confidence(rule_score)

        # --- BANKER detection: count strong conviction signals ---
        # When multiple strong signals align, flag as high-conviction bet
        banker_signals = 0
        banker_reasons = []

        # 1. High overall score (top of field)
        if rule_score >= 68:
            banker_signals += 1
            banker_reasons.append("High score")

        # 2. Rating trending up strongly
        if rtg_chg >= 5:
            banker_signals += 1
            banker_reasons.append(f"Rating surging (+{rtg_chg})")

        # 3. Ideal freshness (14-35 days)
        if 14 <= days <= 35:
            banker_signals += 1
            banker_reasons.append("Peak fitness")

        # 4. Strong recent form (2+ wins in last 6)
        if form_score >= 60:
            banker_signals += 1
            banker_reasons.append("Winning form")

        # 5. Pace advantage (leader at sprint or closer at distance)
        if pace_score >= 70:
            banker_signals += 1
            banker_reasons.append("Pace advantage")

        # 6. Good draw for distance
        if draw_score >= 70:
            banker_signals += 1
            banker_reasons.append("Ideal draw")

        # 7. Top jockey
        if jockey_score >= 82:
            banker_signals += 1
            banker_reasons.append("Top jockey")

        # 8. Top earner (proven at level)
        if stakes_val >= 2_000_000:
            banker_signals += 1
            banker_reasons.append("Proven earner")

        # Market confidence intentionally NOT a BANKER signal — it would
        # double-count market info already applied via the dynamic blend
        # downstream. Threshold is 6/8 (was 7/9) so promotion frequency is
        # roughly preserved.
        is_banker = banker_signals >= 6
        if is_banker:
            reasoning.insert(0, f"BANKER ({banker_signals}/8 signals: {', '.join(banker_reasons)})")

        horse_data = {
            "number": horse.number,
            "name": horse.name,
            "rating": horse.rating,
            "draw": horse.draw,
            "jockey": horse.jockey,
            "trainer": horse.trainer,
            "last_6": horse.last_6,
            "distance": distance,
            # New ML features
            "running_style": style,
            "last_400m_avg": horse.last_400m_avg,
            "cd_wins": horse.cd_wins,
            "cd_runs": horse.cd_runs,
            "opening_odds": horse.opening_odds,
            "win_odds": horse.win_odds,
            "tips_odds": horse.tips_odds,
            "course": race.course,
            "venue": race.venue,
            "gear": horse.gear,
            "weight": horse.weight,
            "class_type": race.class_type,
            "field_size": len(all_horses),
            "going": race.going,
            "days_since_last_run": horse.days_since_last_run,
            "race_date": race.date,
            # F-06 / DS-03 (2026-04-23): pass field_weights so extract_features()
            # uses field-relative median formula (matching training). Without this,
            # inference fell back to absolute-weight formula → 82% parity failure.
            "field_weights": field_weights,
        }

        # Graceful degradation: if the model is missing/expired/corrupt,
        # fall back to rule-only blending instead of crashing the dashboard.
        # IntegrityError fires for >90-day-old models (per ml/integrity.py);
        # RuntimeError for untrained models (F-17). Both are recoverable.
        try:
            ml_prediction = self.ml_analyzer.predict(horse_data, all_results or [], "")
            ml_available_for_horse = ML_AVAILABLE
        except Exception as e:
            from ml.model import MLPrediction
            logger.warning(
                f"ML predict failed for {horse.name} ({type(e).__name__}: {e}); "
                "degrading to rule-only blend"
            )
            ml_prediction = MLPrediction(
                horse_number=horse.number, horse_name=horse.name,
                win_probability=1.0 / max(len(all_horses or []), 2),
                place_probability=3.0 / max(len(all_horses or []), 2),
                ml_confidence='UNAVAILABLE', feature_importance={},
                model_version='',
            )
            ml_available_for_horse = False

        combined_prob, combined_confidence, breakdown = create_combined_prediction(
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            ml_prob=ml_prediction.win_probability,
            ml_confidence=ml_prediction.ml_confidence,
            ml_available=ml_available_for_horse,
            field_size=len(all_horses) if all_horses else 12,
        )

        # value_score blends rule + ML (on 0-100 scale) for ranking
        # combined_prob is 0-1 scale; convert back to 0-100 for value_score
        value_score = combined_prob * 100

        # Store raw combined score — bet recommendation assigned at race level
        return HorseAnalysis(
            horse=horse,
            rank=0,
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            ml_prediction=ml_prediction,
            combined_probability=combined_prob,
            combined_confidence=combined_confidence,
            value_score=value_score,
            expected_return=0.0,  # calculated at race level
            bet_recommendation="",  # assigned at race level
            reasoning=reasoning,
            prediction_breakdown=breakdown,
        )

    def analyze_race(
        self, race: Race, all_results: List = None
    ) -> List[HorseAnalysis]:
        # Build pace scenario for the field
        dist = int(race.distance) if race.distance and race.distance.isdigit() else 1200
        styles = []
        for horse in race.horses:
            style = horse.running_style or self.pace_analyzer.estimate_style_from_form(
                horse.last_6, horse.draw, dist
            )
            styles.append(style)

        pace_scenario = self.pace_analyzer.build_pace_scenario(styles, dist)

        analyses = [
            self.analyze_horse(horse, race, race.horses, all_results, pace_scenario)
            for horse in race.horses
        ]
        analyses.sort(key=lambda x: x.combined_probability, reverse=True)
        for rank, analysis in enumerate(analyses, 1):
            analysis.rank = rank

        # --- Calibrate probabilities & assign bet types at race level ---
        dist = int(race.distance) if race.distance and str(race.distance).isdigit() else 1200
        self._assign_bet_recommendations(analyses, field_size=len(race.horses), distance=dist)

        return analyses

    @staticmethod
    def _assign_bet_recommendations(analyses: List[HorseAnalysis],
                                    field_size: int = 12, distance: int = 1200):
        """Delegate to core.recommender."""
        assign_bet_recommendations(analyses, field_size=field_size, distance=distance)


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    @staticmethod
    def print_header():
        # Count real training data
        n_results = "synthetic"
        try:
            import sqlite3
            from pathlib import Path
            db_path = Path(__file__).parent / 'tracking_data' / 'horse_history.db'
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                n = conn.execute("SELECT COUNT(*) FROM race_results").fetchone()[0]
                conn.close()
                if n > 0:
                    n_results = f"{n:,}"
        except Exception:
            pass

        print("\n" + "=" * 80)
        print("  HKJC HORSE RACING ANALYZER v5.1 - PLATT CALIBRATED ML")
        print("=" * 80)
        print(f"""
  PREDICTION METHODOLOGY (v5.2 - 24-Factor + 23-Feature ML)
  ----------------------------------------------------------
  RULE-BASED (60%):  24 scoring factors with dynamic weighting
    Core:    Rating | Form | Tips | Jockey (Bayesian-smoothed) | Trainer | Draw
    Expert:  Going Preference | Rolling Form | Beaten Margins | Jockey Change
             Jockey/Trainer Hot Streaks | Field Size | Career Place Rate | Age
    Tech:    Pace Map | C&D Form | Class Movement | Gear | Weight | Sectionals
             Freshness | Rating Trend | Season Stakes
    Scoring: base-rate-anchored sigmoid (avg horse → 1/field_size, not 50%)
  ML-BASED  (40%):  23-feature XGBoost (win + place) on {n_results} results
    Calibration:     Platt scaling (recency-weighted) | Softmax temp 1.5 | dynamic market blend
  DATA:              Real jockey/trainer stats from DB | Live odds from HKJC
        """)

    @staticmethod
    def print_data_status(race: Race):
        """Show data freshness."""
        total = len(race.horses)
        with_odds = sum(1 for h in race.horses if h.win_odds > 0)
        with_rating = sum(1 for h in race.horses if h.rating > 0)
        with_form = sum(1 for h in race.horses if h.last_6)

        with_tips = sum(1 for h in race.horses if h.initial_tips_index > 0)
        top_picks = [h for h in race.horses if h.is_top_pick]

        status = "LIVE" if with_odds > 0 else "NO ODDS"
        tips_status = f"Tips: {with_tips}/{total}" if with_tips > 0 else "No Tips"
        print(f"\n  Data Status: {status} | {tips_status}")
        print(f"  Horses: {total} | Odds: {with_odds}/{total} | "
              f"Ratings: {with_rating}/{total} | Form: {with_form}/{total}")
        if top_picks:
            picks_str = ", ".join(f"#{h.number} {h.name}" for h in top_picks)
            print(f"  HKJC Top Picks: {picks_str}")
        if race.odds_updated_at:
            print(f"  Odds last updated: {race.odds_updated_at}")
        print()

    @staticmethod
    def print_race_analysis(race: Race, analyses: List[HorseAnalysis]):
        print(f"\n{'=' * 80}")
        print(f"  RACE {race.race_number} - {race.venue}")
        print(f"{'=' * 80}")
        print(f"  Date: {race.date} | Time: {race.time} | Distance: {race.distance}m")
        course_str = f" | Rail: {race.course}" if race.course else ""
        print(f"  Class: {race.class_type} | Going: {race.going} | Track: {race.track}{course_str}")

        ReportGenerator.print_data_status(race)

        print(f"  {'Rank':<5}{'No':<4}{'Horse':<20}{'Jockey':<16}"
              f"{'Odds':<8}{'Rule%':<8}{'ML%':<8}{'Comb%':<8}{'Bet':<7}{'Conf'}")
        print(f"  {'-' * 78}")

        for a in analyses:
            h = a.horse
            ml_prob = a.ml_prediction.win_probability
            rule_prob = a.rule_score / 100
            odds_str = f"${h.win_odds:.1f}" if h.win_odds > 0 else "N/A"

            print(
                f"  {a.rank:<5}{h.number:<4}{h.name:<20}{h.jockey:<16}"
                f"{odds_str:<8}{rule_prob * 100:<8.1f}{ml_prob * 100:<8.1f}"
                f"{a.combined_probability * 100:<8.1f}{a.bet_recommendation:<7}"
                f"{a.combined_confidence}"
            )

        # Top 3 detailed analysis
        for analysis in analyses[:3]:
            h = analysis.horse
            medal = {1: "1st", 2: "2nd", 3: "3rd"}[analysis.rank]

            print(f"\n  {'=' * 76}")
            print(f"  {medal} PICK: #{h.number} {h.name}")
            print(f"  {'=' * 76}")

            odds_str = f"${h.win_odds:.1f}" if h.win_odds > 0 else "N/A"
            place_str = f"${h.place_odds:.1f}" if h.place_odds > 0 else "N/A"
            print(f"  Win Odds: {odds_str} | Place Odds: {place_str} | "
                  f"Expected Return: {analysis.expected_return * 100:.1f}%")

            print(f"\n  Prediction Breakdown:")
            print(f"    Rule-based: {analysis.rule_score:.1f}/100 ({analysis.rule_confidence})")
            ml_data = analysis.prediction_breakdown["ml_based"]
            if ML_AVAILABLE:
                print(f"    ML-based:   {ml_data['probability'] * 100:.1f}% ({ml_data['confidence']})")
            else:
                print(f"    ML-based:   N/A (model not loaded)")
            print(f"    Combined:   {analysis.combined_probability * 100:.1f}% ({analysis.combined_confidence})")
            print(f"    Recommendation: {analysis.bet_recommendation}")

            print(f"\n  Reasoning:")
            for i, reason in enumerate(analysis.reasoning, 1):
                print(f"    {i}. {reason}")

        # Summary
        print(f"\n  {'=' * 76}")
        print(f"  BETTING SUMMARY")
        print(f"  {'=' * 76}")

        value_bets = [a for a in analyses if a.expected_return > 0.05]
        print(f"\n  Value Bets (EV > 5%): {len(value_bets)}")
        for a in value_bets[:5]:
            odds_str = f"${a.horse.win_odds:.1f}" if a.horse.win_odds > 0 else "N/A"
            print(f"    #{a.rank} {a.horse.name} @ {odds_str} "
                  f"(EV: {a.expected_return * 100:.1f}%, {a.combined_confidence})")


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

async def main():
    ReportGenerator.print_header()

    # Get race date
    race_date = input("  Enter race date (YYYY/MM/DD) or press Enter for auto: ").strip()

    fetcher = HKJCDataFetcher(use_playwright=True)

    try:
        if not race_date:
            # Auto-detect next meeting
            print("\n  Checking upcoming meetings...")
            dates = await fetcher.get_meeting_dates()
            if dates:
                print(f"  Available dates: {', '.join(dates[:5])}")
                race_date = dates[0]
                print(f"  Using: {race_date}")
            else:
                race_date = datetime.now().strftime("%Y/%m/%d")
                print(f"  No dates found, using today: {race_date}")

        race_input = input(
            "  Enter race number (1-11), 'all' for all races, or press Enter for 1: "
        ).strip()

        if race_input.lower() == "all":
            await _run_all_races(fetcher, race_date)
        else:
            race_no = int(race_input) if race_input else 1
            await _run_single_race(fetcher, race_date, race_no)

    finally:
        await fetcher.close()

    print("\n" + "=" * 80)
    print("  DISCLAIMER: For entertainment only. Not financial advice. Gamble responsibly.")
    print("=" * 80)


async def _run_single_race(fetcher: HKJCDataFetcher, race_date: str, race_no: int):
    """Analyze a single race."""
    print(f"\n  Fetching Race {race_no} data from HKJC...")
    rc = await fetcher.get_race_card(race_date, race_no)

    if not rc or not rc.horses:
        print(f"\n  No data found for Race {race_no} on {race_date}.")
        print("  This could mean:")
        print("    - No meeting on this date")
        print("    - Race card not yet published")
        print("    - HKJC website is blocking the request")
        return

    race = convert_race(rc)

    # If no odds, offer manual entry
    if not any(h.win_odds > 0 for h in race.horses):
        print(f"\n  Got {len(race.horses)} horses but no live odds.")
        manual = input("  Enter odds manually? (y/n): ").strip().lower()
        if manual == "y":
            _manual_odds_entry(race)

    analyzer = CombinedRaceAnalyzer()
    analyses = analyzer.analyze_race(race)
    ReportGenerator.print_race_analysis(race, analyses)

    # Betting optimizer
    _print_betting_optimization(race, analyses)

    # Offer odds refresh
    if any(h.win_odds > 0 for h in race.horses):
        refresh = input("\n  Refresh odds? (y/n): ").strip().lower()
        if refresh == "y":
            print("  Refreshing odds...")
            new_odds = await fetcher.get_live_odds(race_date, race_no)
            if new_odds:
                for h in race.horses:
                    if h.number in new_odds:
                        h.win_odds = new_odds[h.number]
                race.odds_updated_at = datetime.now().strftime("%H:%M:%S")
                analyses = analyzer.analyze_race(race)
                ReportGenerator.print_race_analysis(race, analyses)
            else:
                print("  Could not refresh odds.")


async def _run_all_races(fetcher: HKJCDataFetcher, race_date: str):
    """Analyze all races for a meeting."""
    print(f"\n  Fetching all races for {race_date}...")
    day = await fetcher.get_all_races(race_date)

    if not day.races:
        print(f"  No races found for {race_date}.")
        return

    print(f"\n  Found {len(day.races)} races at {day.venue}")

    analyzer = CombinedRaceAnalyzer()

    for rc in day.races:
        race = convert_race(rc)
        analyses = analyzer.analyze_race(race)
        ReportGenerator.print_race_analysis(race, analyses)


def _manual_odds_entry(race: Race):
    """Let user enter odds manually."""
    print(f"\n  Enter WIN odds for each horse (press Enter to skip):\n")
    for h in race.horses:
        while True:
            try:
                val = input(f"    #{h.number} {h.name}: ").strip()
                if not val:
                    break
                odds = float(val)
                if 1.0 <= odds <= 999:
                    h.win_odds = odds
                    if h.place_odds <= 0:
                        h.place_odds = max(1.1, odds / 3)
                    break
                else:
                    print("      Odds should be between 1.0 and 999")
            except ValueError:
                if val.lower() == "q":
                    return
                print("      Enter a valid number")
    race.odds_updated_at = "manual"


def _print_betting_optimization(race: Race, analyses: List[HorseAnalysis]):
    """Run betting optimizer."""
    budget = 100
    race_dict = {
        "race_number": race.race_number,
        "venue": race.venue,
        "distance": race.distance,
        "class_type": race.class_type,
    }

    race_analyses = []
    for a in analyses:
        race_analyses.append({
            "race_number": race.race_number,
            "horse_data": {
                "number": a.horse.number,
                "name": a.horse.name,
                "win_odds": a.horse.win_odds,
                "place_odds": a.horse.place_odds,
                "rating": a.horse.rating,
                "jockey": a.horse.jockey,
                "trainer": a.horse.trainer,
            },
            "combined_probability": a.combined_probability,
            "expected_return": a.expected_return,
            "confidence": a.combined_confidence,
        })

    BettingReportGenerator.print_betting_analysis(race_dict, race_analyses, budget)


if __name__ == "__main__":
    asyncio.run(main())
