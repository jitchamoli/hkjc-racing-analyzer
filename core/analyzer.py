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
from data.odds_tracker import OddsMovementAnalyzer

logging.basicConfig(level=logging.INFO)
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
    def analyze_jockey(jockey: str, db=None) -> tuple:
        """Score jockey using real DB stats when available, else fallback."""
        if db and jockey:
            # Try season stats first (most complete)
            stats = db.get_jockey_stats(jockey)
            if not stats:
                # Try partial name match across DB
                try:
                    c = db.conn.cursor()
                    c.execute("SELECT * FROM jockey_stats WHERE name LIKE ?",
                              (f'%{jockey.split()[-1]}%',))
                    row = c.fetchone()
                    if row:
                        stats = dict(row)
                except Exception:
                    pass

            if stats and stats.get('rides', 0) >= 5:
                win_rate = stats.get('win_rate', 0)
                place_rate = stats.get('place_rate', 0)
                rides = stats['rides']
                wins = stats.get('wins', 0)

                # Score based on actual win rate (HK avg ~8-10%)
                if win_rate >= 0.20:
                    score = min(95, 80 + win_rate * 75)
                    label = "Elite"
                elif win_rate >= 0.14:
                    score = 78 + win_rate * 50
                    label = "Top"
                elif win_rate >= 0.08:
                    score = 65 + win_rate * 80
                    label = "Solid"
                elif win_rate >= 0.04:
                    score = 55 + win_rate * 100
                    label = "Average"
                else:
                    score = max(35, 40 + win_rate * 200)
                    label = "Struggling"

                # Place rate bonus (consistent top-3 finisher)
                if place_rate >= 0.35:
                    score = min(97, score + 5)
                elif place_rate >= 0.25:
                    score = min(95, score + 3)

                score = min(97, max(30, score))
                reason = f"{label} jockey - {jockey} ({wins}W/{rides}R, {win_rate:.0%}W)"
                return score, reason

        # Fallback: tier-based defaults for known jockeys when DB unavailable
        TIER_1 = {"moreira", "purton", "bowman", "moore", "mcdonald"}
        TIER_2 = {"teetan", "badel", "hewitson", "shinn", "atzeni", "avdulla", "bentley"}
        TIER_3 = {"schofield", "chau", "ho", "kingscote", "ferraris", "borges", "leung"}

        jockey_lower = jockey.lower() if jockey else ""
        surname = jockey_lower.split()[-1] if jockey_lower.split() else ""

        if surname in TIER_1:
            return 90.0, f"Elite jockey - {jockey}"
        elif surname in TIER_2:
            return 78.0, f"Top jockey - {jockey}"
        elif surname in TIER_3:
            return 70.0, f"Experienced jockey - {jockey}"
        return 60.0, f"Jockey: {jockey}"

    @staticmethod
    def analyze_trainer(trainer: str, db=None) -> tuple:
        """Score trainer using real DB stats when available, else fallback."""
        if db and trainer:
            stats = db.get_trainer_stats(trainer)
            if not stats:
                try:
                    c = db.conn.cursor()
                    c.execute("SELECT * FROM trainer_stats WHERE name LIKE ?",
                              (f'%{trainer.split()[-1]}%',))
                    row = c.fetchone()
                    if row:
                        stats = dict(row)
                except Exception:
                    pass

            if stats and stats.get('runners', 0) >= 5:
                win_rate = stats.get('win_rate', 0)
                place_rate = stats.get('place_rate', 0)
                runners = stats['runners']
                wins = stats.get('wins', 0)

                # Score based on actual win rate (HK avg ~6-8%)
                if win_rate >= 0.18:
                    score = min(93, 78 + win_rate * 80)
                    label = "Elite"
                elif win_rate >= 0.12:
                    score = 72 + win_rate * 60
                    label = "Top"
                elif win_rate >= 0.06:
                    score = 58 + win_rate * 100
                    label = "Solid"
                elif win_rate >= 0.02:
                    score = 45 + win_rate * 200
                    label = "Average"
                else:
                    score = max(30, 35 + win_rate * 300)
                    label = "Struggling"

                if place_rate >= 0.30:
                    score = min(95, score + 4)
                elif place_rate >= 0.20:
                    score = min(93, score + 2)

                score = min(95, max(25, score))
                reason = f"{label} trainer - {trainer} ({wins}W/{runners}R, {win_rate:.0%}W)"
                return score, reason

        # Fallback: tier-based defaults for known trainers
        TIER_1 = {"size", "cruz", "lor"}
        TIER_2 = {"hall", "fownes", "lui", "whyte", "hayes", "richards", "newnham", "shum"}
        TIER_3 = {"ting", "yiu", "eustace", "crawford", "yip", "man", "tsui"}

        trainer_lower = trainer.lower() if trainer else ""
        surname = trainer_lower.split()[-1] if trainer_lower.split() else ""

        if surname in TIER_1:
            return 88.0, f"Elite trainer - {trainer}"
        elif surname in TIER_2:
            return 75.0, f"Top trainer - {trainer}"
        elif surname in TIER_3:
            return 68.0, f"Experienced trainer - {trainer}"
        return 58.0, f"Trainer: {trainer}"

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
        # Try to connect to horse database for historical lookups
        self._horse_db = None
        try:
            from data.database import HorseDatabase
            self._horse_db = HorseDatabase()
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

        # When form data is missing/empty, use market odds as proxy
        # A horse at $5 with no form is still well-regarded by the market
        if form_score == 0 and horse.win_odds > 0:
            odds_implied = 1.0 / horse.win_odds
            # Short-priced no-form horse = market sees something we don't
            if odds_implied >= 0.15:  # $6.67 or shorter
                form_score = 55.0
                form_reason = f"No form — market backed (${horse.win_odds:.0f})"
            elif odds_implied >= 0.08:  # $12.5 or shorter
                form_score = 45.0
                form_reason = f"No form — moderate market support (${horse.win_odds:.0f})"
            else:
                form_score = 30.0
                form_reason = f"No form — outsider (${horse.win_odds:.0f})"

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

        if horse.win_odds > 0:
            odds_prob = 1 / horse.win_odds
            factors["odds"] = min(100, odds_prob * 100)
            reasoning.append(
                f"Market: {odds_prob*100:.1f}% implied @ ${horse.win_odds:.1f} (LIVE)"
            )
        else:
            factors["odds"] = 0
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

        # --- NEW: Course & Distance Form ---
        if horse.cd_runs > 0:
            # Score directly from wins/runs to avoid reconstructing inaccurate history
            cd_wins = horse.cd_wins
            cd_runs = horse.cd_runs
            if cd_wins >= 3:
                cd_score, cd_reason = 95.0, f"Proven C&D star ({cd_wins}W from {cd_runs})"
            elif cd_wins == 2:
                cd_score, cd_reason = 85.0, f"Strong C&D form ({cd_wins}W from {cd_runs})"
            elif cd_wins == 1:
                cd_score, cd_reason = 70.0, f"C&D winner ({cd_wins}W from {cd_runs})"
            elif cd_runs >= 3:
                cd_score, cd_reason = 48.0, f"C&D: 0W from {cd_runs} runs"
            else:
                cd_score, cd_reason = 55.0, f"C&D: {cd_runs} runs, no wins"
        else:
            cd_score, cd_reason = 0.0, "No C&D data"
        factors["cd_form"] = cd_score
        if cd_score > 0:
            reasoning.append(f"C&D: {cd_reason}")

        # --- NEW: Odds Movement ---
        if horse.opening_odds > 0 and horse.win_odds > 0:
            move_score, move_reason = self.odds_movement.analyze_movement(
                horse.opening_odds, horse.win_odds
            )
        elif horse.tips_odds > 0 and horse.win_odds > 0:
            move_score, move_reason = self.odds_movement.analyze_from_tips(
                horse.tips_odds, horse.win_odds
            )
        else:
            move_score, move_reason = 0.0, "No odds movement data"
        factors["odds_movement"] = move_score
        if move_score > 0:
            reasoning.append(f"Odds: {move_reason}")

        # --- NEW: Class Movement ---
        # Without actual horse history with class data, we default to neutral
        # The class_movement factor only activates when horse history includes class info
        class_score = 50.0
        class_reason = "No previous class data"
        factors["class_movement"] = class_score
        # Class movement is always included in weight map (50=neutral is valid data)

        # --- NEW: Gear Changes ---
        gear_score, gear_reason = self.gear_analyzer.analyze_gear(
            horse.gear if hasattr(horse, 'gear') else ""
        )
        factors["gear"] = gear_score
        if gear_reason != "No gear changes":
            reasoning.append(f"Gear: {gear_reason}")

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
            horse_record = self._horse_db.get_horse_record(horse.name)
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
        # #1 predictor in gmalbert research (17.41% importance)
        # Small fields = favourites dominate. Large fields = more upsets.
        # Score: higher for horses with good odds in small fields,
        #        higher for outsiders in large fields (more chance of chaos)
        field_n = len(all_horses) if all_horses else 0
        if field_n > 0 and horse.win_odds > 0:
            implied_prob = 1.0 / horse.win_odds
            if field_n <= 8:
                # Small field: favourites dominate, penalize outsiders
                if implied_prob >= 0.15:
                    fs_score = 75.0
                    fs_reason = f"Small field ({field_n}) favours market leaders"
                else:
                    fs_score = 35.0
                    fs_reason = f"Small field ({field_n}) — hard for outsiders"
            elif field_n <= 11:
                # Medium field: balanced
                fs_score = 55.0
                fs_reason = f"Medium field ({field_n})"
            else:
                # Large field: more chaos, outsiders have more chance
                if implied_prob >= 0.12:
                    fs_score = 60.0
                    fs_reason = f"Large field ({field_n}) — still backed"
                elif implied_prob >= 0.06:
                    fs_score = 55.0
                    fs_reason = f"Large field ({field_n}) — live chance"
                else:
                    fs_score = 45.0
                    fs_reason = f"Large field ({field_n}) — longshot territory"
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
            hr = self._horse_db.get_horse_record(horse.name)
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
        if horse.age and horse.age.isdigit():
            horse_age = int(horse.age)
            field_ages = [int(h.age) for h in (all_horses or []) if h.age and h.age.isdigit()]
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

        # 9. Market confidence (short odds)
        if horse.win_odds > 0 and (1 / horse.win_odds) >= 0.10:
            banker_signals += 1
            banker_reasons.append("Market backed")

        # BANKER = 7+ signals out of 9 (must be truly exceptional)
        is_banker = banker_signals >= 7
        if is_banker:
            reasoning.insert(0, f"BANKER ({banker_signals}/9 signals: {', '.join(banker_reasons)})")

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
            "gear": horse.gear if hasattr(horse, 'gear') else "",
            "weight": horse.weight,
            "class_type": race.class_type,
            "field_size": len(all_horses),
            "going": race.going if hasattr(race, 'going') else "",
            "days_since_last_run": horse.days_since_last_run if hasattr(horse, 'days_since_last_run') else 0,
        }

        ml_prediction = self.ml_analyzer.predict(horse_data, all_results or [], "")

        combined_prob, combined_confidence, breakdown = create_combined_prediction(
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            ml_prob=ml_prediction.win_probability,
            ml_confidence=ml_prediction.ml_confidence,
            ml_available=ML_AVAILABLE,
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
        self._assign_bet_recommendations(analyses)

        return analyses

    @staticmethod
    def _assign_bet_recommendations(analyses: List[HorseAnalysis]):
        """Delegate to core.recommender."""
        assign_bet_recommendations(analyses)


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
        print("  HKJC HORSE RACING ANALYZER v5.0 - REAL DATA EDITION")
        print("=" * 80)
        print(f"""
  PREDICTION METHODOLOGY (v5.0 - 26-Factor + Real Data ML)
  --------------------------------------------------------
  RULE-BASED (75%):  26 scoring factors with dynamic weighting
    Core:    Rating | Form | Tips | Odds | Jockey (DB stats) | Trainer (DB stats) | Draw
    Expert:  Pace Map | C&D Form | Odds Movement | Going Preference | Rolling Form
             Beaten Margins | Jockey Change | Jockey/Trainer Hot Streaks
             Field Size | Career Place Rate | Age vs Field | Freshness | Rating Trend
    Tech:    Class Movement | Gear Changes | Weight | Sectional Speed | Season Stakes
  ML-BASED  (25%):  XGBoost trained on {n_results} real race results
  DATA:             Real jockey/trainer stats from DB | Live odds from HKJC
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
