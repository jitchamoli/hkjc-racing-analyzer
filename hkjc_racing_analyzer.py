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
    # Full 14-factor weights (when all data available)
    RATING_WEIGHT = 0.12
    FORM_WEIGHT = 0.12
    ODDS_WEIGHT = 0.09
    JOCKEY_WEIGHT = 0.07
    TRAINER_WEIGHT = 0.05
    DRAW_WEIGHT = 0.06       # Enhanced with rail bias
    TIPS_WEIGHT = 0.12       # HKJC expert tips index
    PACE_WEIGHT = 0.06       # Running style + pace scenario
    SECTIONAL_WEIGHT = 0.05  # Finishing speed
    CD_FORM_WEIGHT = 0.06    # Course & Distance record
    ODDS_MOVE_WEIGHT = 0.06  # Odds movement / steam detection
    CLASS_MOVE_WEIGHT = 0.05 # Class drop/rise advantage
    GEAR_WEIGHT = 0.04       # Equipment changes
    WEIGHT_WEIGHT = 0.05     # Weight carried vs rating

    @staticmethod
    def analyze_form(last_6: str) -> tuple:
        if not last_6 or len(last_6) < 1:
            return 0.0, "No recent form data"

        total_score = 0.0
        wins = places = seconds = 0

        for pos in str(last_6):
            if pos == "1":
                total_score += 100
                wins += 1
            elif pos == "2":
                total_score += 60
                seconds += 1
            elif pos == "3":
                total_score += 40
                places += 1
            elif pos == "4":
                total_score += 25
            elif pos == "5":
                total_score += 15
            elif pos == "6":
                total_score += 10

        avg_score = total_score / max(1, len(str(last_6)))

        if wins >= 3:
            reason = f"Strong form: {wins} wins in last 6 races"
        elif wins >= 1 and avg_score > 50:
            reason = f"Good recent form: {wins}W/{places}P"
        elif places >= 2:
            reason = f"Consistent placer: {places} places"
        else:
            reason = f"Moderate form: avg score {avg_score:.0f}"

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
            "F C Lor": (88, "Elite trainer - F C Lor"),
            "D J Hall": (82, "Top trainer - D J Hall"),
            "C Fownes": (82, "Top trainer - C Fownes"),
            "J Richards": (80, "Skilled trainer"),
            "D Whyte": (80, "Trainer - D Whyte"),
            "K H Ting": (78, "Consistent trainer"),
            "P F Yiu": (75, "Experienced trainer"),
            "W Y So": (72, "Trainer - W Y So"),
            "C H Yip": (75, "Trainer - C H Yip"),
        }
        for name, (score, reason) in top_trainers.items():
            if name.lower() in trainer.lower():
                return score, reason
        return 65.0, f"Trainer: {trainer}"

    @staticmethod
    def analyze_draw(draw: int, distance: str) -> tuple:
        dist = int(distance) if distance and distance.isdigit() else 1200

        if dist <= 1000:
            ideal = 3
        elif dist <= 1400:
            ideal = 5
        else:
            ideal = 8

        if draw == 0:
            return 50.0, "No draw data"

        if draw == ideal:
            return 95.0, f"Ideal barrier ({draw})"
        elif abs(draw - ideal) <= 2:
            return 80.0, f"Good barrier ({draw})"
        elif draw > 12:
            return 55.0, f"Wide draw ({draw})"
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

        # Calculate EV using calibrated probabilities
        for i, a in enumerate(analyses):
            cal_prob = calibrated_probs[i]
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

        for i, a in enumerate(analyses):
            cal_prob = calibrated_probs[i]
            rank = a.rank

            if rank == 1:
                # Strong standout: clear gap to 2nd AND positive EV
                if gap_1_2 > 0.04 and a.expected_return > 0.10:
                    a.bet_recommendation = "WIN"
                elif gap_1_2 > 0.02 and a.expected_return > 0.0:
                    a.bet_recommendation = "WIN"
                elif a.expected_return > 0.0:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "PLACE"

            elif rank == 2:
                # 2nd pick: close to 1st = QIN candidate, otherwise PLACE
                if gap_1_2 < 0.03 and a.expected_return > 0.0:
                    a.bet_recommendation = "QIN"
                elif a.expected_return > 0.05:
                    a.bet_recommendation = "PLACE"
                elif a.horse.place_odds > 1.2:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            elif rank == 3:
                # 3rd pick: tight top 3 = TRIO candidate
                if top_3_share > 0.38 and gap_2_3 < 0.03:
                    a.bet_recommendation = "TRIO"
                elif a.expected_return > 0.05 and a.horse.place_odds > 1.3:
                    a.bet_recommendation = "QNL PL"
                elif a.horse.place_odds > 1.5:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            elif rank == 4:
                # 4th pick: only if strong outsider value
                if a.expected_return > 0.15 and a.horse.win_odds >= 10:
                    a.bet_recommendation = "QNL PL"
                elif a.horse.place_odds > 2.0:
                    a.bet_recommendation = "PLACE"
                else:
                    a.bet_recommendation = "SKIP"

            else:
                # 5th+ : almost always skip
                if a.expected_return > 0.25 and a.horse.win_odds >= 20:
                    a.bet_recommendation = "PLACE"
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
