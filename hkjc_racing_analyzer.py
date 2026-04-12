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

from hkjc_data_fetcher import HKJCDataFetcher, HorseEntry, RaceCard

from ml_predictor import (
    HKJCRacingMLModel,
    FeatureEngineering,
    CombinedRacingAnalyzer,
    MLPrediction,
    create_combined_prediction,
    ML_AVAILABLE,
)

from betting_optimizer import BettingOptimizer, BettingReportGenerator

from advanced_analytics import (
    DrawBiasAnalyzer,
    PaceAnalyzer,
    SectionalAnalyzer,
    BarrierTrialAnalyzer,
    CourseDistanceAnalyzer,
    ClassMovementAnalyzer,
    GearChangeAnalyzer,
    WeightAnalyzer,
    GoingPreferenceAnalyzer,
    RollingFormAnalyzer,
    BeatenMarginAnalyzer,
    JockeyChangeAnalyzer,
)
from odds_tracker import OddsMovementAnalyzer

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
    # Full 17-factor weights (when all data available)
    # Tips Index is the strongest signal — aggregates expert tipster consensus
    RATING_WEIGHT = 0.07
    FORM_WEIGHT = 0.09       # Recency-weighted form (recent wins boosted)
    ODDS_WEIGHT = 0.08       # Market implied probability
    JOCKEY_WEIGHT = 0.04
    TRAINER_WEIGHT = 0.02
    DRAW_WEIGHT = 0.04       # Enhanced with rail bias
    TIPS_WEIGHT = 0.12       # HKJC expert tips index (strongest signal)
    PACE_WEIGHT = 0.03       # Running style + pace scenario
    SECTIONAL_WEIGHT = 0.03  # Finishing speed
    CD_FORM_WEIGHT = 0.04    # Course & Distance record
    ODDS_MOVE_WEIGHT = 0.04  # Odds movement / steam detection
    CLASS_MOVE_WEIGHT = 0.03 # Class drop/rise advantage
    GEAR_WEIGHT = 0.02       # Equipment changes
    WEIGHT_WEIGHT = 0.04     # Weight carried vs rating
    FRESHNESS_WEIGHT = 0.04  # Days since last run
    RATING_TREND_WEIGHT = 0.03  # Rating going up or down
    STAKES_WEIGHT = 0.02     # Season prize money earned
    GOING_WEIGHT = 0.05      # Going (track condition) preference — high impact
    # New high-impact features (from CodeWorks Paris research)
    ROLLING_FORM_WEIGHT = 0.06  # Rolling avg position, win rate, trend
    MARGIN_WEIGHT = 0.05     # Beaten margins — <2.75L = strong next-out signal
    JOCKEY_CHANGE_WEIGHT = 0.04  # Jockey upgrade/downgrade detection
    JOCKEY_FORM_WEIGHT = 0.04   # Jockey recent win% (hot streak)
    TRAINER_FORM_WEIGHT = 0.03  # Trainer recent win% (hot streak)

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
    def analyze_jockey(jockey: str) -> tuple:
        top_jockeys = {
            "J Moreira": (95, "Elite jockey - J Moreira"),
            "Z Purton": (92, "Champion jockey - Z Purton"),
            "K Teetan": (85, "Top jockey - K Teetan"),
            "H Bowman": (85, "International star - H Bowman"),
            "C L Chau": (80, "Rising star - C L Chau"),
            "A Badel": (82, "Top jockey - A Badel"),
            "H Bentley": (80, "Skilled jockey - H Bentley"),
            "L Hewitson": (82, "Top jockey - L Hewitson"),
            "C Schofield": (78, "Skilled jockey - C Schofield"),
            "K C Leung": (75, "Experienced jockey"),
            "M Chadwick": (72, "Improving jockey"),
            "V Borges": (78, "Skilled jockey"),
            "B Shinn": (83, "Top jockey - B Shinn"),
            "A Atzeni": (84, "Top international - A Atzeni"),
            "B Avdulla": (80, "Skilled jockey - B Avdulla"),
            "C Y Ho": (78, "Experienced jockey - C Y Ho"),
            "R Kingscote": (80, "Skilled jockey - R Kingscote"),
            "L Ferraris": (78, "Rising star - L Ferraris"),
            "J Orman": (76, "Solid jockey - J Orman"),
            "M L Yeung": (72, "Local jockey - M L Yeung"),
            "H Y Yuen": (70, "Apprentice - H Y Yuen"),
            "H T Mo": (68, "Apprentice - H T Mo"),
            "E C W Wong": (68, "Apprentice - E C W Wong"),
            "P N Wong": (66, "Apprentice - P N Wong"),
            "Y L Chung": (68, "Local jockey - Y L Chung"),
        }
        for name, (score, reason) in top_jockeys.items():
            if name.lower() in jockey.lower():
                return score, reason
        return 65.0, f"Jockey: {jockey}"

    @staticmethod
    def analyze_trainer(trainer: str) -> tuple:
        top_trainers = {
            "J Size": (92, "Legendary trainer - J Size"),
            "A Cruz": (90, "Top trainer - A Cruz"),
            "A S Cruz": (90, "Top trainer - A S Cruz"),
            "F C Lor": (88, "Elite trainer - F C Lor"),
            "D J Hall": (82, "Top trainer - D J Hall"),
            "C Fownes": (82, "Top trainer - C Fownes"),
            "J Richards": (80, "Skilled trainer - J Richards"),
            "D Whyte": (80, "Top trainer - D Whyte"),
            "D J Whyte": (80, "Top trainer - D J Whyte"),
            "K H Ting": (78, "Consistent trainer - K H Ting"),
            "P F Yiu": (78, "Experienced trainer - P F Yiu"),
            "W Y So": (72, "Trainer - W Y So"),
            "C H Yip": (75, "Trainer - C H Yip"),
            "K W Lui": (82, "Top trainer - K W Lui"),
            "D A Hayes": (80, "Skilled trainer - D A Hayes"),
            "D Eustace": (76, "Trainer - D Eustace"),
            "B Crawford": (75, "Trainer - B Crawford"),
            "M Newnham": (78, "Skilled trainer - M Newnham"),
            "C S Shum": (78, "Experienced trainer - C S Shum"),
            "K L Man": (74, "Trainer - K L Man"),
            "P C Ng": (72, "Trainer - P C Ng"),
            "C W Chang": (72, "Trainer - C W Chang"),
            "W K Mo": (70, "Trainer - W K Mo"),
            "Y S Tsui": (74, "Trainer - Y S Tsui"),
        }
        for name, (score, reason) in top_trainers.items():
            if name.lower() in trainer.lower():
                return score, reason
        return 65.0, f"Trainer: {trainer}"

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
        self.ml_analyzer = CombinedRacingAnalyzer()
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
            from horse_database import HorseDatabase
            self._horse_db = HorseDatabase()
        except Exception:
            pass

    def analyze_horse(
        self, horse: Horse, race: Race, all_horses: List[Horse] = None,
        all_results: List = None, pace_scenario: Dict = None,
    ) -> HorseAnalysis:
        factors = {}
        reasoning = []
        distance = race.distance

        # --- Core factors (same as before) ---

        factors["rating"] = min(100, horse.rating)
        reasoning.append(
            f"Rating {horse.rating:.0f}: "
            f"{'Top class' if factors['rating'] > 90 else 'Good' if factors['rating'] > 80 else 'Average'}"
        )

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

        jockey_score, jockey_reason = self.strategy.analyze_jockey(horse.jockey)
        factors["jockey"] = jockey_score
        reasoning.append(jockey_reason)

        trainer_score, trainer_reason = self.strategy.analyze_trainer(horse.trainer)
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

        # --- Weighted score calculation (dynamic) ---
        # Build weight map only for available factors (score > 0)
        weight_map = {
            "rating": BettingStrategy.RATING_WEIGHT,
            "form": BettingStrategy.FORM_WEIGHT,
            "odds": BettingStrategy.ODDS_WEIGHT,
            "jockey": BettingStrategy.JOCKEY_WEIGHT,
            "trainer": BettingStrategy.TRAINER_WEIGHT,
            "draw": BettingStrategy.DRAW_WEIGHT,
        }

        # Conditionally add new factors
        if tips_score > 0:
            weight_map["tips"] = BettingStrategy.TIPS_WEIGHT
        if pace_score > 0:
            weight_map["pace"] = BettingStrategy.PACE_WEIGHT
        if sect_score > 0:
            weight_map["sectional"] = BettingStrategy.SECTIONAL_WEIGHT
        if cd_score > 0:
            weight_map["cd_form"] = BettingStrategy.CD_FORM_WEIGHT
        if move_score > 0:
            weight_map["odds_movement"] = BettingStrategy.ODDS_MOVE_WEIGHT
        # Class, gear, weight always included (50.0 = neutral, not missing)
        weight_map["class_movement"] = BettingStrategy.CLASS_MOVE_WEIGHT
        weight_map["gear"] = BettingStrategy.GEAR_WEIGHT
        weight_map["weight_carried"] = BettingStrategy.WEIGHT_WEIGHT
        # New high-impact factors
        if fresh_score > 0:
            weight_map["freshness"] = BettingStrategy.FRESHNESS_WEIGHT
        if trend_score > 0:
            weight_map["rating_trend"] = BettingStrategy.RATING_TREND_WEIGHT
        if stakes_score > 0:
            weight_map["season_stakes"] = BettingStrategy.STAKES_WEIGHT
        if going_score > 0:
            weight_map["going_pref"] = BettingStrategy.GOING_WEIGHT
        if rolling_score > 0:
            weight_map["rolling_form"] = BettingStrategy.ROLLING_FORM_WEIGHT
        if margin_score > 0:
            weight_map["margin"] = BettingStrategy.MARGIN_WEIGHT
        if jc_score > 0:
            weight_map["jockey_change"] = BettingStrategy.JOCKEY_CHANGE_WEIGHT
        if jf_score > 0:
            weight_map["jockey_form"] = BettingStrategy.JOCKEY_FORM_WEIGHT
        if tf_score > 0:
            weight_map["trainer_form"] = BettingStrategy.TRAINER_FORM_WEIGHT

        # Normalize weights to sum to 1.0
        total_weight = sum(weight_map.values())
        if total_weight > 0:
            rule_score = sum(
                factors[k] * (w / total_weight)
                for k, w in weight_map.items()
            )
        else:
            rule_score = 50.0

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
        }

        ml_prediction = self.ml_analyzer.predict(horse_data, all_results or [], "")

        combined_prob, combined_confidence, breakdown = create_combined_prediction(
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            ml_prob=ml_prediction.win_probability,
            ml_confidence=ml_prediction.ml_confidence,
            ml_available=ML_AVAILABLE,
        )

        # Store raw combined score — bet recommendation assigned at race level
        return HorseAnalysis(
            horse=horse,
            rank=0,
            rule_score=rule_score,
            rule_confidence=rule_confidence,
            ml_prediction=ml_prediction,
            combined_probability=combined_prob,
            combined_confidence=combined_confidence,
            value_score=rule_score,
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
        """Assign realistic bet recommendations based on field shape.

        The combined_probability is a relative score (0-1), NOT a calibrated
        win probability. We convert to realistic probabilities using softmax
        with temperature scaling to preserve score separation.
        """
        if not analyses:
            return

        n = len(analyses)

        # Softmax normalization: exp(score * temp) / sum(exp(score * temp))
        # Temperature controls separation: higher = more spread
        import math
        temp = 8.0  # amplifies score differences
        raw_scores = [a.combined_probability for a in analyses]
        max_score = max(raw_scores) if raw_scores else 0.5
        exp_scores = [math.exp((s - max_score) * temp) for s in raw_scores]
        exp_total = sum(exp_scores) or 1.0
        calibrated_probs = [e / exp_total for e in exp_scores]

        # Blend with market-implied probabilities (30% market, 70% model)
        # This prevents our model from severely underrating strong favourites
        market_probs = []
        for a in analyses:
            if a.horse.win_odds > 0:
                market_probs.append(1.0 / a.horse.win_odds)
            else:
                market_probs.append(0.0)
        market_total = sum(market_probs) or 1.0
        market_probs = [p / market_total for p in market_probs]  # normalize to fair probs

        blended_probs = []
        for i in range(len(calibrated_probs)):
            if market_probs[i] > 0:
                blended = 0.70 * calibrated_probs[i] + 0.30 * market_probs[i]
            else:
                blended = calibrated_probs[i]
            blended_probs.append(blended)
        # Re-normalize
        blend_total = sum(blended_probs) or 1.0
        calibrated_probs = [p / blend_total for p in blended_probs]

        # Store calibrated probabilities and calculate EV
        for i, a in enumerate(analyses):
            cal_prob = calibrated_probs[i]
            a.combined_probability = cal_prob  # overwrite raw score with calibrated prob
            if a.horse.win_odds > 0:
                a.expected_return = (a.horse.win_odds * cal_prob) - 1
            else:
                a.expected_return = -1.0

        # Field shape analysis for bet type assignment
        top = analyses[0] if n >= 1 else None
        second = analyses[1] if n >= 2 else None
        third = analyses[2] if n >= 3 else None

        top_prob = calibrated_probs[0] if n >= 1 else 0
        second_prob = calibrated_probs[1] if n >= 2 else 0
        third_prob = calibrated_probs[2] if n >= 3 else 0

        gap_1_2 = top_prob - second_prob  # gap between 1st and 2nd
        gap_2_3 = second_prob - third_prob  # gap between 2nd and 3rd
        top_3_share = top_prob + second_prob + third_prob  # top 3 share of total

        # Market-implied probability for comparison
        market_prob_top = (1 / top.horse.win_odds) if top and top.horse.win_odds > 0 else 0

        # Check if our #1 is a BANKER — if so, don't let market override it
        top_is_banker = any("BANKER" in r for r in analyses[0].reasoning) if analyses else False

        # Check if market favourite differs from our top pick — if so, re-sort
        # to respect the market when it strongly backs a different horse
        # BUT: never override a BANKER pick — our model has high conviction
        market_fav = min(analyses, key=lambda a: a.horse.win_odds if a.horse.win_odds > 0 else 999)
        market_fav_prob = (1 / market_fav.horse.win_odds) if market_fav.horse.win_odds > 0 else 0
        if (market_fav_prob >= 0.30 and market_fav.rank > 1
                and market_fav.horse.win_odds > 0
                and not top_is_banker):
            # Market strongly backs a horse we don't rank #1 — promote it
            # Re-sort: swap market fav to position 0
            old_idx = next(i for i, a in enumerate(analyses) if a is market_fav)
            analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
            analyses[0].rank = 1
            for i, a in enumerate(analyses):
                a.rank = i + 1
            # Recalculate field shape
            top = analyses[0]
            second = analyses[1] if n >= 2 else None
            top_prob = calibrated_probs[old_idx]
            calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
            second_prob = calibrated_probs[1] if n >= 2 else 0
            third_prob = calibrated_probs[2] if n >= 3 else 0
            gap_1_2 = top_prob - second_prob
            gap_2_3 = second_prob - third_prob
            top_3_share = top_prob + second_prob + third_prob
            market_prob_top = market_fav_prob

        # Check if any horse is a BANKER — if so, promote it to #1 and force WIN
        banker_horse = None
        for a in analyses:
            if any("BANKER" in r for r in a.reasoning):
                if banker_horse is None or a.rule_score > banker_horse.rule_score:
                    banker_horse = a

        if banker_horse and banker_horse.rank > 1:
            # Promote BANKER to rank 1
            old_idx = next(i for i, a in enumerate(analyses) if a is banker_horse)
            analyses[0], analyses[old_idx] = analyses[old_idx], analyses[0]
            for i, a in enumerate(analyses):
                a.rank = i + 1
            calibrated_probs[0], calibrated_probs[old_idx] = calibrated_probs[old_idx], calibrated_probs[0]
            top = analyses[0]
            top_prob = calibrated_probs[0]
            second_prob = calibrated_probs[1] if n >= 2 else 0
            gap_1_2 = top_prob - second_prob

        # Determine race-level bet type first, then assign to horses
        # QIN #1+#2 is the DEFAULT — our rank 2 wins often enough that
        # QIN on ranks 1+2 hits far more often than WIN on rank 1 alone.
        # WIN only for BANKER (7+ signals) or extreme market standouts.
        race_bet_type = "QIN"  # default: QIN on rank 1 + rank 2

        # BANKER overrides everything — always WIN
        if banker_horse and banker_horse.rank == 1:
            race_bet_type = "WIN"
        elif market_prob_top >= 0.35 and gap_1_2 > 0.08:
            race_bet_type = "WIN"  # very clear market standout (tighter threshold)
        elif gap_1_2 > 0.06 and top and top.expected_return > 0.15:
            race_bet_type = "WIN"  # very clear model standout (tighter threshold)
        elif gap_1_2 < 0.02 and top_3_share > 0.35:
            # Top 3 extremely bunched — wider exotic
            if gap_2_3 < 0.02 and n >= 3:
                race_bet_type = "TRIO"  # top 3 all bunched
            else:
                race_bet_type = "QNL PL"  # very tight top 2
        # else: stays QIN (default)

        for i, a in enumerate(analyses):
            rank = a.rank
            # Use place odds if available, otherwise estimate from win odds
            has_place_value = (a.horse.place_odds > 1.2) if a.horse.place_odds > 0 else (a.horse.win_odds >= 3.0)

            if race_bet_type == "WIN":
                if rank == 1:
                    # Check if this horse has BANKER flag in reasoning
                    is_banker = any("BANKER" in r for r in a.reasoning)
                    a.bet_recommendation = "BANKER WIN" if is_banker else "WIN"
                elif rank == 2 and (a.expected_return > 0.0 or has_place_value):
                    a.bet_recommendation = "PLACE"
                elif rank == 3 and has_place_value:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            elif race_bet_type == "QIN":
                if rank <= 2:
                    a.bet_recommendation = "QIN"
                elif rank == 3 and has_place_value:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            elif race_bet_type == "QNL PL":
                if rank <= 2:
                    a.bet_recommendation = "QNL PL"
                elif rank == 3 and has_place_value:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            elif race_bet_type == "TRIO":
                if rank <= 3:
                    a.bet_recommendation = "TRIO"
                elif rank == 4 and has_place_value:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            elif race_bet_type == "PLACE":
                if rank <= 2:
                    a.bet_recommendation = "PLACE"
                elif rank == 3 and (a.expected_return > 0.0 or has_place_value):
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            else:
                a.bet_recommendation = "SKIP"


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    @staticmethod
    def print_header():
        print("\n" + "=" * 80)
        print("  HKJC HORSE RACING ANALYZER v1.0.0 - REAL-TIME DATA EDITION")
        print("=" * 80)
        print("""
  PREDICTION METHODOLOGY (v3.0 - Full 14-Factor Analytics)
  --------------------------------------------------------
  RULE-BASED (50%):  Rating 12% | Form 12% | Tips 12% | Odds 9% | Jockey 7% | Trainer 5%
                     Draw+Rail 6% | Pace 6% | C&D 6% | Odds Move 6% | Sectional 5%
                     Class Move 5% | Gear 4% | Weight 5%
  ML-BASED   (50%):  XGBoost classifier (14 features)
  ADVANCED:          Rail bias | Pace map | Sectionals | Barrier trials | C&D form
                     Odds movement | Class drop/rise | Gear changes | Weight analysis
  DATA SOURCE:       Live from racing.hkjc.com
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
