#!/usr/bin/env python3
"""
ML-Based Horse Racing Prediction Module
Uses historical HKJC data to train XGBoost model for win probability prediction
"""

import asyncio
import re
import json
import pickle
import os
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import logging
import numpy as np

import requests
from playwright.async_api import async_playwright

try:
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RaceResult:
    date: str
    venue: str
    race_number: int
    distance: int
    going: str
    class_type: str
    horse_number: int
    horse_name: str
    draw: int
    jockey: str
    trainer: str
    weight: int
    rating: float
    position: int
    win_odds: float
    place_odds: float
    last_6: str
    is_win: bool
    is_place: bool

@dataclass
class HorseFeatures:
    horse_number: int
    horse_name: str
    rating: float
    draw: int
    weight: int
    recent_form_score: float
    win_rate: float
    place_rate: float
    avg_position: float
    jockey_win_rate: float
    trainer_win_rate: float
    distance_suitability: float
    draw_advantage: float
    going_suitability: float

@dataclass
class MLPrediction:
    horse_number: int
    horse_name: str
    win_probability: float
    place_probability: float
    ml_confidence: str
    feature_importance: Dict[str, float]
    model_version: str

class HKJCHistoricalScraper:
    RESULTS_URL = "https://racing.hkjc.com/racing/information/English/Racing/LocalResults.aspx"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        self.results_cache = []
    
    async def scrape_historical_results(self, start_date: str, end_date: str) -> List[RaceResult]:
        results = []
        
        start = datetime.strptime(start_date, '%Y/%m/%d')
        end = datetime.strptime(end_date, '%Y/%m/%d')
        
        current = start
        while current <= end:
            date_str = current.strftime('%Y/%m/%d')
            logger.info(f"Scraping results for {date_str}...")
            
            day_results = await self._scrape_day_results(date_str)
            results.extend(day_results)
            
            current += timedelta(days=1)
        
        self.results_cache = results
        return results
    
    async def _scrape_day_results(self, race_date: str) -> List[RaceResult]:
        results = []
        
        pw = await async_playwright().start()
        try:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page()
            
            url = f"{self.RESULTS_URL}?RaceDate={race_date.replace('/', '%2F')}"
            await page.goto(url, wait_until='domcontentloaded', timeout=60000)
            await page.wait_for_timeout(3000)
            
            html = await page.content()
            day_results = self._parse_results_html(html, race_date)
            results.extend(day_results)
            
            await browser.close()
        except Exception as e:
            logger.error(f"Error scraping {race_date}: {e}")
        finally:
            await pw.stop()
        
        return results
    
    def _parse_results_html(self, html: str, race_date: str) -> List[RaceResult]:
        results = []
        
        venue = 'Sha Tin' if 'Sha Tin' in html else 'Happy Valley'
        
        tables = re.findall(r'<table[^>]*class=\"([^\"]*)\"[^>]*>(.*?)</table>', html, re.DOTALL)
        
        for cls, content in tables:
            if 'result' in cls.lower() or 'tableBorderBlue' in cls:
                rows = re.findall(r'<tr[^>]*>(.*?)</tr>', content, re.DOTALL)
                
                for row_html in rows:
                    cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
                    cells = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
                    
                    if len(cells) < 10:
                        continue
                    
                    try:
                        position_text = cells[0]
                        if not position_text.isdigit():
                            continue
                        
                        position = int(position_text)
                        horse_num = int(re.sub(r'[^0-9]', '', cells[1])) if cells[1] else 0
                        if horse_num == 0:
                            continue
                        
                        result = RaceResult(
                            date=race_date,
                            venue=venue,
                            race_number=1,
                            distance=1200,
                            going='Good',
                            class_type='Class',
                            horse_number=horse_num,
                            horse_name=cells[2] if len(cells) > 2 else '',
                            draw=0,
                            jockey=cells[3] if len(cells) > 3 else '',
                            trainer=cells[4] if len(cells) > 4 else '',
                            weight=0,
                            rating=0,
                            position=position,
                            win_odds=0,
                            place_odds=0,
                            last_6='',
                            is_win=(position == 1),
                            is_place=(position <= 3)
                        )
                        results.append(result)
                    except Exception as e:
                        continue
        
        return results

class FeatureEngineering:
    @staticmethod
    def parse_last_6(last_6: str) -> List[int]:
        positions = []
        for char in str(last_6):
            if char.isdigit():
                positions.append(int(char))
        return positions
    
    @staticmethod
    def calculate_form_score(last_6: str) -> float:
        positions = FeatureEngineering.parse_last_6(last_6)
        if not positions:
            return 50.0
        
        total = 0
        for pos in positions:
            if pos == 1:
                total += 100
            elif pos == 2:
                total += 70
            elif pos == 3:
                total += 50
            elif pos == 4:
                total += 30
            elif pos == 5:
                total += 15
            else:
                total += 5
        
        return total / len(positions)
    
    @staticmethod
    def calculate_win_rate(history: List[RaceResult]) -> float:
        if not history:
            return 0.15
        
        wins = sum(1 for r in history if r.is_win)
        return min(0.8, wins / len(history))
    
    @staticmethod
    def calculate_place_rate(history: List[RaceResult]) -> float:
        if not history:
            return 0.35
        
        places = sum(1 for r in history if r.is_place)
        return min(0.9, places / len(history))
    
    @staticmethod
    def calculate_avg_position(history: List[RaceResult]) -> float:
        if not history:
            return 6.0
        
        return sum(r.position for r in history) / len(history)
    
    @staticmethod
    def calculate_jockey_win_rate(jockey: str, all_results: List[RaceResult]) -> float:
        jockey_rides = [r for r in all_results if jockey.lower() in r.jockey.lower()]
        if not jockey_rides:
            return 0.12
        
        wins = sum(1 for r in jockey_rides if r.is_win)
        return min(0.6, wins / len(jockey_rides))
    
    @staticmethod
    def calculate_trainer_win_rate(trainer: str, all_results: List[RaceResult]) -> float:
        trainer_horses = [r for r in all_results if trainer.lower() in r.trainer.lower()]
        if not trainer_horses:
            return 0.12
        
        wins = sum(1 for r in trainer_horses if r.is_win)
        return min(0.5, wins / len(trainer_horses))
    
    @staticmethod
    def calculate_draw_advantage(draw: int, distance) -> float:
        try:
            distance = int(distance)
        except (ValueError, TypeError):
            distance = 1200
        
        if distance <= 1000:
            ideal = 4
        elif distance <= 1400:
            ideal = 6
        else:
            ideal = 10
        
        diff = abs(draw - ideal)
        if diff == 0:
            return 1.0
        elif diff <= 2:
            return 0.8
        elif diff <= 4:
            return 0.6
        else:
            return max(0.3, 1 - (diff * 0.1))
    
    @staticmethod
    def _style_to_float(style: str) -> float:
        """Convert running style to numeric feature."""
        return {"Leader": 0.9, "Stalker": 0.7, "Midfield": 0.5, "Closer": 0.3}.get(style, 0.5)

    @staticmethod
    def extract_features(horse_data: dict, all_results: List[RaceResult], venue: str) -> np.ndarray:
        features = []

        # Original 6 features
        features.append(horse_data.get('rating', 70) / 100)
        features.append(FeatureEngineering.calculate_form_score(horse_data.get('last_6', '')) / 100)
        features.append(FeatureEngineering.calculate_draw_advantage(
            horse_data.get('draw', 8),
            horse_data.get('distance', 1200)
        ))

        jockey = horse_data.get('jockey', '')
        trainer = horse_data.get('trainer', '')

        jockey_rate = FeatureEngineering.calculate_jockey_win_rate(jockey, all_results)
        trainer_rate = FeatureEngineering.calculate_trainer_win_rate(trainer, all_results)

        features.append(jockey_rate)
        features.append(trainer_rate)

        features.append(1.0 if 'Sha Tin' in venue else 0.7)

        # 5 new features
        # 7: rail_draw_bias (0-1 scale from draw analyzer)
        features.append(features[2])  # Same as draw_advantage as baseline; enhanced externally

        # 8: pace_advantage
        features.append(FeatureEngineering._style_to_float(horse_data.get('running_style', 'Midfield')))

        # 9: finishing_speed (last 400m normalized)
        l4 = horse_data.get('last_400m_avg', 0.0)
        if l4 > 0:
            features.append(max(0.0, min(1.0, (25.0 - l4) / 5.0)))  # 20s=1.0, 25s=0.0
        else:
            features.append(0.5)

        # 10: cd_form_score
        cd_wins = horse_data.get('cd_wins', 0)
        cd_runs = horse_data.get('cd_runs', 0)
        if cd_runs > 0:
            features.append(min(1.0, cd_wins / max(1, cd_runs) + 0.2))
        else:
            features.append(0.5)

        # 11: odds_movement_score
        opening = horse_data.get('opening_odds', 0) or horse_data.get('tips_odds', 0)
        current = horse_data.get('win_odds', 0)
        if opening > 0 and current > 0:
            pct = (opening - current) / opening
            features.append(max(0.0, min(1.0, 0.5 + pct)))
        else:
            features.append(0.5)

        # 12: class_movement (0=rise, 0.5=same, 1.0=drop)
        class_type = horse_data.get('class_type', '')
        import re as _re
        cm = _re.search(r'(\d)', class_type)
        class_rank = int(cm.group(1)) if cm else 3
        features.append(max(0.0, min(1.0, class_rank / 5.0)))

        # 13: gear_impact (0.5 = none, higher = positive gear)
        gear = horse_data.get('gear', '')
        if gear and gear.strip() not in ('--', '-', ''):
            gear_upper = gear.strip().upper()
            if 'B' in gear_upper:
                features.append(0.72)
            elif 'TT' in gear_upper:
                features.append(0.65)
            elif 'V' in gear_upper:
                features.append(0.62)
            else:
                features.append(0.55)
        else:
            features.append(0.5)

        # 14: weight_ratio (rating / weight, normalized)
        weight_str = horse_data.get('weight', '')
        rating = horse_data.get('rating', 0)
        wm = _re.match(r'(\d+)', str(weight_str))
        if wm and rating > 0:
            w = float(wm.group(1))
            features.append(max(0.0, min(1.0, rating / max(1, w))))
        else:
            features.append(0.5)

        return np.array(features, dtype=np.float32)

class HKJCRacingMLModel:
    MODEL_DIR = Path(__file__).parent / 'models'
    MODEL_FILE = MODEL_DIR / 'hkjc_racing_model.pkl'
    
    FEATURE_NAMES = [
        'rating_norm', 'form_score', 'draw_advantage',
        'jockey_win_rate', 'trainer_win_rate', 'venue_advantage',
        'rail_draw_bias', 'pace_advantage', 'finishing_speed',
        'cd_form_score', 'odds_movement_score',
        'class_movement', 'gear_impact', 'weight_ratio',
    ]
    
    def __init__(self):
        self.model = None
        self.is_trained = False
        self.model_version = "1.0"
        self.feature_importance = {}
        
        if ML_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        if self.MODEL_FILE.exists():
            try:
                with open(self.MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.model_version = data.get('version', '1.0')
                    self.feature_importance = data.get('feature_importance', {})
                    self.is_trained = self.model is not None
                    logger.info(f"Loaded model version {self.model_version}")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def _save_model(self):
        self.MODEL_DIR.mkdir(exist_ok=True)
        
        data = {
            'model': self.model,
            'version': self.model_version,
            'feature_importance': self.feature_importance,
            'trained_date': datetime.now().isoformat()
        }
        
        with open(self.MODEL_FILE, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved model version {self.model_version}")
    
    def train(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        if not ML_AVAILABLE:
            return {'error': 'ML libraries not available'}
        
        logger.info(f"Training with {len(features)} samples...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)
        
        self.feature_importance = dict(zip(
            self.FEATURE_NAMES,
            self.model.feature_importances_.tolist()
        ))
        
        self.is_trained = True
        self.model_version = datetime.now().strftime('%Y%m%d-%H%M%S')
        self._save_model()
        
        return {
            'accuracy': accuracy,
            'logloss': logloss,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version
        }
    
    def predict(self, features: np.ndarray) -> Tuple[float, str]:
        if not self.is_trained or self.model is None:
            base_prob = 1.0 / len(features)
            return base_prob, "MODEL_NOT_TRAINED"
        
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        
        win_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        
        if win_prob > 0.30:
            confidence = "HIGH ⭐⭐⭐"
        elif win_prob > 0.15:
            confidence = "MEDIUM ⭐⭐"
        elif win_prob > 0.08:
            confidence = "LOW ⭐"
        else:
            confidence = "VERY_LOW"
        
        return win_prob, confidence

def generate_synthetic_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)

    X = []
    y = []

    for _ in range(n_samples):
        rating = np.random.uniform(50, 100)
        form_score = np.random.uniform(20, 100)
        draw_adv = np.random.uniform(0.3, 1.0)
        jockey_rate = np.random.uniform(0.05, 0.40)
        trainer_rate = np.random.uniform(0.05, 0.35)
        venue_adv = np.random.choice([0.7, 1.0])
        # New features (11 total → 14)
        rail_draw = np.random.uniform(0.3, 1.0)
        pace_adv = np.random.uniform(0.3, 0.9)
        finish_speed = np.random.uniform(0.2, 0.9)
        cd_form = np.random.uniform(0.2, 1.0)
        odds_move = np.random.uniform(0.2, 0.8)
        class_move = np.random.uniform(0.2, 0.9)
        gear_impact = np.random.choice([0.5, 0.55, 0.62, 0.65, 0.72], p=[0.5, 0.1, 0.15, 0.1, 0.15])
        weight_ratio = np.random.uniform(0.35, 0.75)

        features = [
            rating / 100, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, venue_adv,
            rail_draw, pace_adv, finish_speed, cd_form, odds_move,
            class_move, gear_impact, weight_ratio,
        ]
        X.append(features)

        win_prob = (
            0.17 * (rating / 100)
            + 0.15 * (form_score / 100)
            + 0.07 * draw_adv
            + 0.12 * jockey_rate
            + 0.06 * trainer_rate
            + 0.05 * rail_draw
            + 0.06 * pace_adv
            + 0.05 * finish_speed
            + 0.06 * cd_form
            + 0.06 * odds_move
            + 0.05 * class_move
            + 0.04 * gear_impact
            + 0.06 * weight_ratio
        ) * venue_adv

        win_prob = min(0.7, max(0.03, win_prob))

        is_win = np.random.random() < win_prob
        y.append(1 if is_win else 0)

    return np.array(X), np.array(y)

def create_combined_prediction(
    rule_score: float,
    rule_confidence: str,
    ml_prob: float,
    ml_confidence: str,
    ml_available: bool
) -> Tuple[float, str, Dict]:
    weights = {'rule': 0.5, 'ml': 0.5}
    
    if not ml_available:
        weights = {'rule': 1.0, 'ml': 0.0}
    
    ml_prob_adjusted = ml_prob * weights['ml']
    rule_prob = (rule_score / 100) * weights['rule']
    
    combined_prob = rule_prob + ml_prob_adjusted
    
    if 'HIGH' in rule_confidence and 'HIGH' in ml_confidence:
        combined_confidence = "VERY_HIGH ⭐⭐⭐⭐"
    elif 'HIGH' in rule_confidence or 'HIGH' in ml_confidence:
        combined_confidence = "HIGH ⭐⭐⭐"
    elif 'MEDIUM' in rule_confidence or 'MEDIUM' in ml_confidence:
        combined_confidence = "MEDIUM ⭐⭐"
    else:
        combined_confidence = "LOW ⭐"
    
    breakdown = {
        'rule_based': {
            'score': rule_score,
            'contribution': rule_prob,
            'confidence': rule_confidence
        },
        'ml_based': {
            'probability': ml_prob,
            'contribution': ml_prob_adjusted,
            'confidence': ml_confidence if ml_available else 'NOT_AVAILABLE'
        },
        'final_probability': combined_prob,
        'weights': weights
    }
    
    return combined_prob, combined_confidence, breakdown

class CombinedRacingAnalyzer:
    def __init__(self):
        self.ml_model = HKJCRacingMLModel()
        self.feature_eng = FeatureEngineering()
        
        if ML_AVAILABLE and not self.ml_model.is_trained:
            logger.info("Training ML model with synthetic data...")
            X, y = generate_synthetic_training_data()
            result = self.ml_model.train(X, y)
            logger.info(f"Model trained: accuracy={result['accuracy']:.2%}")
    
    def predict(self, horse_data: dict, all_results: List[RaceResult], venue: str) -> MLPrediction:
        features = self.feature_eng.extract_features(horse_data, all_results, venue)
        
        ml_prob, ml_confidence = self.ml_model.predict(features)
        
        return MLPrediction(
            horse_number=horse_data.get('number', 0),
            horse_name=horse_data.get('name', ''),
            win_probability=ml_prob,
            place_probability=min(1.0, ml_prob * 2.5),
            ml_confidence=ml_confidence,
            feature_importance=dict(zip(
                self.ml_model.FEATURE_NAMES,
                (features * self.ml_model.feature_importance.get(f, 0.1) for f in self.ml_model.FEATURE_NAMES)
            )),
            model_version=self.ml_model.model_version
        )
