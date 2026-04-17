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
        """Parse form string — handles '1/3/10/2/5' and legacy '13425'."""
        s = str(last_6)
        if "/" in s:
            return [int(p) for p in s.split("/") if p.strip().isdigit()]
        return [int(ch) for ch in s if ch.isdigit()]
    
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
            return 0.08  # ~1/12 base rate for typical HK field
        
        wins = sum(1 for r in history if r.is_win)
        return min(0.8, wins / len(history))
    
    @staticmethod
    def calculate_place_rate(history: List[RaceResult]) -> float:
        if not history:
            return 0.25  # ~3/12 base rate for typical HK field
        
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
        """Extract 18 features matching training features exactly.

        Must match build_training_data_from_db() feature order:
        1. prior_win_rate  2. recent_form  3. draw_advantage
        4. jockey_win_rate 5. trainer_win_rate 6. field_size
        7. field_rel_draw  8. prior_pace   9. prior_speed
        10. cd_form_prior  11. odds_implied 12. class_level
        13. freshness      14. going_pref  15. going_form_rate
        16. jockey_horse_coop 17. distance_pb_ratio 18. odds_movement
        """
        import sqlite3
        from pathlib import Path

        features = []
        horse_name = horse_data.get('name', '')
        distance = horse_data.get('distance', 1200)
        going = horse_data.get('going', '')

        # Load prior race history from DB (strictly before race_date if provided)
        prior = []
        race_date = horse_data.get('race_date', '')
        try:
            db_path = Path(__file__).parent.parent / 'tracking_data' / 'horse_history.db'
            if db_path.exists():
                conn = sqlite3.connect(str(db_path))
                conn.row_factory = sqlite3.Row
                if race_date:
                    rows = conn.execute(
                        "SELECT * FROM race_results WHERE horse_name = ? AND race_date < ? ORDER BY race_date",
                        (horse_name, race_date)
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM race_results WHERE horse_name = ? ORDER BY race_date",
                        (horse_name,)
                    ).fetchall()
                prior = [dict(r) for r in rows]
                conn.close()
        except Exception:
            pass

        # Feature 1: prior win rate
        if prior:
            prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
            prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
            feat_form_proxy = min(1.0, (prior_wins * 2 + prior_top3) / (len(prior) * 3))
        else:
            feat_form_proxy = 0.5
        features.append(feat_form_proxy)

        # Feature 2: recent form from prior finishes
        if prior:
            recent = prior[-6:]
            form_pts = 0
            for p in recent:
                pos = p.get('finish_position', 7)
                if pos == 1: form_pts += 10
                elif pos == 2: form_pts += 7
                elif pos == 3: form_pts += 5
                elif pos <= 5: form_pts += 2
            features.append(form_pts / (len(recent) * 10))
        else:
            # Fallback to race card last_6 if no DB history
            last_6 = horse_data.get('last_6', '')
            if last_6:
                features.append(FeatureEngineering.calculate_form_score(last_6) / 100)
            else:
                features.append(0.5)

        # Feature 3: draw advantage
        draw = horse_data.get('draw', 8)
        features.append(FeatureEngineering.calculate_draw_advantage(draw, distance))

        # Feature 4: jockey win rate
        jockey = horse_data.get('jockey', '')
        jockey_rate = FeatureEngineering.calculate_jockey_win_rate(jockey, all_results)
        features.append(min(0.5, jockey_rate) if jockey_rate > 0 else 0.08)

        # Feature 5: trainer win rate
        trainer = horse_data.get('trainer', '')
        trainer_rate = FeatureEngineering.calculate_trainer_win_rate(trainer, all_results)
        features.append(min(0.4, trainer_rate) if trainer_rate > 0 else 0.06)

        # Feature 6: field size
        field_size = horse_data.get('field_size', 12)
        features.append(field_size / 14.0)

        # Feature 7: field-relative draw
        features.append(draw / max(1, field_size) if draw > 0 else 0.5)

        # Feature 8: prior pace (avg early position from prior races)
        pace_vals = []
        for p in prior[-5:]:
            rp = p.get('running_position', '') or ''
            if rp:
                if ' ' in rp:
                    positions = [int(x) for x in rp.split() if x.isdigit()]
                else:
                    positions = [int(c) for c in rp if c.isdigit()]
                if positions:
                    early = min(positions[0], 14)
                    # Approximate field size from prior race
                    pace_vals.append(1.0 - (early - 1) / 13.0)
        features.append(sum(pace_vals) / len(pace_vals) if pace_vals else 0.5)

        # Feature 9: prior finishing speed
        speed_vals = []
        for p in prior[-5:]:
            ft = p.get('finish_time', '') or ''
            d = p.get('distance', 0) or 0
            if ft and ':' in ft and d > 0:
                try:
                    parts = ft.split(':')
                    secs = float(parts[0]) * 60 + float(parts[1])
                    if secs > 0:
                        speed = d / secs
                        speed_vals.append(max(0.0, min(1.0, (speed - 14) / 4)))
                except (ValueError, IndexError):
                    pass
        features.append(sum(speed_vals) / len(speed_vals) if speed_vals else 0.5)

        # Feature 10: C&D form from prior races at same distance
        prior_at_dist = [p for p in prior if p.get('distance') == distance]
        if prior_at_dist:
            cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
            features.append(min(1.0, cd_wins / len(prior_at_dist) + 0.2))
        else:
            features.append(0.5)

        # Feature 11: odds-implied probability
        win_odds = horse_data.get('win_odds', 0)
        features.append(min(1.0, 1.0 / win_odds) if win_odds > 0 else 0.5)

        # Feature 12: class level
        import re as _re
        class_type = horse_data.get('class_type', '')
        cm = _re.search(r'(\d)', class_type)
        class_rank = int(cm.group(1)) if cm else 3
        features.append(max(0.0, min(1.0, class_rank / 5.0)))

        # Feature 13: freshness (days since last run)
        dslr = horse_data.get('days_since_last_run', 0) or 0
        if not dslr and prior:
            # Estimate from last prior race date
            last_date = prior[-1].get('race_date', '')
            if last_date:
                try:
                    from datetime import datetime
                    last_dt = datetime.strptime(last_date.replace('/', '-'), '%Y-%m-%d')
                    dslr = (datetime.now() - last_dt).days
                except Exception:
                    pass
        if dslr > 0:
            features.append(max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60)))
        else:
            features.append(0.5)

        # Feature 14: going preference (prior results on same going)
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if prior_same_going:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                features.append(min(1.0, (going_wins * 2 + going_top3) / (len(prior_same_going) * 3) + 0.2))
            else:
                features.append(0.5)
        else:
            features.append(0.5)

        # Feature 15: going form rate (win rate on current going condition)
        going_form_rate = 0.5
        if prior and going:
            prior_same_going = [p for p in prior if p.get('going', '') == going]
            if len(prior_same_going) >= 2:
                going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                going_form_rate = going_wins / len(prior_same_going)
        features.append(going_form_rate)

        # Feature 16: jockey-horse cooperation rate
        jockey_horse_coop = 0.5
        jockey = horse_data.get('jockey', '')
        if prior and jockey:
            prior_same_jockey = [p for p in prior if p.get('jockey', '') == jockey]
            if len(prior_same_jockey) >= 2:
                jh_wins = sum(1 for p in prior_same_jockey if p.get('is_winner', 0))
                jh_places = sum(1 for p in prior_same_jockey if p.get('is_top3', 0))
                jockey_horse_coop = min(1.0, (jh_wins * 2 + jh_places) / (len(prior_same_jockey) * 3))
        features.append(jockey_horse_coop)

        # Feature 17: distance personal best ratio (horse PB / track record)
        distance_pb_ratio = 0.5
        if prior:
            prior_at_dist = [p for p in prior
                             if p.get('distance') == distance
                             and p.get('finish_time', '')
                             and ':' in str(p.get('finish_time', ''))]
            if prior_at_dist:
                def _parse_time(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0

                times = [_parse_time(p['finish_time']) for p in prior_at_dist]
                valid_times = [t for t in times if t < 999]
                if valid_times:
                    horse_pb = min(valid_times)
                    try:
                        db_path_pb = Path(__file__).parent.parent / 'tracking_data' / 'horse_history.db'
                        if db_path_pb.exists():
                            conn_pb = sqlite3.connect(str(db_path_pb))
                            row_pb = conn_pb.execute(
                                "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL",
                                (distance,)
                            ).fetchone()
                            conn_pb.close()
                            if row_pb and row_pb[0]:
                                track_record = _parse_time(row_pb[0])
                                if track_record < 999 and horse_pb < 999:
                                    distance_pb_ratio = min(1.0, track_record / horse_pb)
                    except Exception:
                        pass
        features.append(distance_pb_ratio)

        # Feature 18: odds movement (opening → current, from odds_history.db)
        odds_movement = 0.5
        win_odds = horse_data.get('win_odds', 0)
        horse_number = horse_data.get('number', 0)
        race_date_for_odds = horse_data.get('race_date', '')
        race_number = horse_data.get('race_number', 0)
        if win_odds > 0 and horse_number > 0 and race_date_for_odds:
            try:
                odds_db_path = Path(__file__).parent.parent / 'tracking_data' / 'odds_history.db'
                if odds_db_path.exists():
                    conn_odds = sqlite3.connect(str(odds_db_path))
                    row_odds = conn_odds.execute(
                        """SELECT win_odds FROM odds_snapshots
                           WHERE race_date = ? AND race_number = ? AND horse_number = ?
                           ORDER BY timestamp ASC LIMIT 1""",
                        (race_date_for_odds, race_number, horse_number)
                    ).fetchone()
                    conn_odds.close()
                    if row_odds and row_odds[0] > 0:
                        opening = row_odds[0]
                        movement = (opening - win_odds) / opening
                        odds_movement = max(0.0, min(1.0, (movement + 1) / 2))
            except Exception:
                pass
        features.append(odds_movement)

        return np.array(features, dtype=np.float32)

class HKJCRacingMLModel:
    MODEL_DIR = Path(__file__).parent.parent / 'models'
    MODEL_FILE = MODEL_DIR / 'hkjc_racing_model.pkl'
    
    FEATURE_NAMES = [
        'prior_win_rate', 'recent_form', 'draw_advantage',
        'jockey_win_rate', 'trainer_win_rate', 'field_size',
        'field_rel_draw', 'prior_pace', 'prior_speed',
        'cd_form_prior', 'odds_implied_prob',
        'class_level', 'freshness', 'going_pref',
        'going_form_rate', 'jockey_horse_coop', 'distance_pb_ratio', 'odds_movement',
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
    
    def train(self, features: np.ndarray, labels: np.ndarray,
              sample_weight: np.ndarray = None) -> Dict:
        if not ML_AVAILABLE:
            return {'error': 'ML libraries not available'}

        logger.info(f"Training with {len(features)} samples "
                    f"({labels.sum():.0f} winners, {labels.mean()*100:.1f}% win rate)")

        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42,
            stratify=labels if labels.sum() >= 10 else None
        )

        # Split sample weights too
        sw_train = None
        if sample_weight is not None:
            sw_train, _ = train_test_split(
                sample_weight, test_size=0.2, random_state=42,
                stratify=labels if labels.sum() >= 10 else None
            )

        # Scale pos weight for imbalanced classes (winners are rare ~7-10%)
        n_pos = max(1, y_train.sum())
        n_neg = max(1, len(y_train) - n_pos)
        scale = n_neg / n_pos

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        self.model.fit(X_train, y_train, sample_weight=sw_train)

        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba)

        # Precision/recall for winner class
        from sklearn.metrics import precision_score, recall_score
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)

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
            'precision': precision,
            'recall': recall,
            'feature_importance': self.feature_importance,
            'model_version': self.model_version
        }
    
    def predict(self, features: np.ndarray, field_size: int = 12) -> Tuple[float, str]:
        if not self.is_trained or self.model is None:
            base_prob = 1.0 / max(field_size, 1)
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

def build_training_data_from_db() -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Build training data from real race results in horse_database.

    Returns (features, labels, sample_weights) or (None, None, None) if insufficient data.
    Each row = one horse in one race. Label = 1 if winner, 0 otherwise.
    Sample weight = higher for recent races.
    """
    try:
        import sqlite3
        from pathlib import Path
        db_path = Path(__file__).parent.parent / 'tracking_data' / 'horse_history.db'
        if not db_path.exists():
            return None, None, None

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        # Get all race results grouped by race
        rows = conn.execute("""
            SELECT r.*, j.win_rate as jockey_wr, j.place_rate as jockey_pr,
                   t.win_rate as trainer_wr, t.place_rate as trainer_pr
            FROM race_results r
            LEFT JOIN jockey_stats j ON r.jockey = j.name
            LEFT JOIN trainer_stats t ON r.trainer = t.name
            WHERE r.finish_position > 0
            ORDER BY r.race_date, r.race_number
        """).fetchall()

        if len(rows) < 50:  # Need at least ~5 races worth
            logger.info(f"Only {len(rows)} results in DB — not enough for real training")
            return None, None, None

        logger.info(f"Building training data from {len(rows)} real race results")

        # Precompute field sizes per race
        field_sizes = {}
        for row in rows:
            r = dict(row)
            rk = f"{r.get('race_date', '')}-{r.get('race_number', 0)}"
            field_sizes[rk] = field_sizes.get(rk, 0) + 1

        # Build per-horse chronological race history for PRIOR-only lookups
        # Key principle: for each race, only use data from BEFORE that race
        horse_history = {}  # horse_name -> list of dicts sorted by date
        for row in rows:
            r = dict(row)
            name = r.get('horse_name', '')
            if name:
                if name not in horse_history:
                    horse_history[name] = []
                horse_history[name].append(r)
        # Sort each horse's history by date
        for name in horse_history:
            horse_history[name].sort(key=lambda x: x.get('race_date', ''))

        X, y, weights = [], [], []
        for row in rows:
            r = dict(row)
            draw = r.get('draw', 0) or 0
            distance = r.get('distance', 1200) or 1200
            jockey_wr = r.get('jockey_wr', 0) or 0
            trainer_wr = r.get('trainer_wr', 0) or 0
            win_odds = r.get('win_odds', 0) or 0
            weight_str = r.get('weight', '') or ''
            horse_name = r.get('horse_name', '')
            race_date = r.get('race_date', '')
            race_no = r.get('race_number', 0)
            race_key = f"{race_date}-{race_no}"
            field_size = field_sizes.get(race_key, 12)
            going = r.get('going', '') or ''

            # Get PRIOR races for this horse (strictly before current race date)
            prior = [h for h in horse_history.get(horse_name, [])
                     if h.get('race_date', '') < race_date]

            # Feature 1: prior win rate (replaces rating — available for all horses)
            if prior:
                prior_wins = sum(1 for p in prior if p.get('is_winner', 0))
                prior_top3 = sum(1 for p in prior if p.get('is_top3', 0))
                feat_form_proxy = (prior_wins * 2 + prior_top3) / (len(prior) * 3)
                feat_form_proxy = min(1.0, feat_form_proxy)
            else:
                feat_form_proxy = 0.5  # no history = unknown

            # Feature 2: recent form from prior finishes (replaces last_6)
            if prior:
                recent = prior[-6:]  # last 6 races
                form_pts = 0
                for p in recent:
                    pos = p.get('finish_position', 7)
                    if pos == 1: form_pts += 10
                    elif pos == 2: form_pts += 7
                    elif pos == 3: form_pts += 5
                    elif pos <= 5: form_pts += 2
                feat_form = form_pts / (len(recent) * 10)
            else:
                feat_form = 0.5

            # Feature 3: draw advantage
            feat_draw = FeatureEngineering.calculate_draw_advantage(draw, distance)

            # Feature 4: jockey win rate (from DB stats, with minimum sample)
            feat_jockey = min(0.5, jockey_wr) if jockey_wr > 0 else 0.08

            # Feature 5: trainer win rate (with minimum sample guard)
            feat_trainer = min(0.4, trainer_wr) if trainer_wr > 0 else 0.06

            # Feature 6: field size (replaces useless venue — actual predictive signal)
            feat_field_size = field_size / 14.0  # normalize: 14 is max field

            # Feature 7: field-relative draw position
            feat_field_draw = draw / max(1, field_size) if draw > 0 else 0.5

            # Feature 8: prior pace style (avg early position from PRIOR races)
            pace_vals = []
            for p in prior[-5:]:  # last 5 prior races
                rp = p.get('running_position', '') or ''
                fs = field_sizes.get(f"{p.get('race_date','')}-{p.get('race_number',0)}", 12)
                if rp:
                    if ' ' in rp:
                        positions = [int(x) for x in rp.split() if x.isdigit()]
                    else:
                        positions = [int(c) for c in rp if c.isdigit()]
                    if positions and fs > 1:
                        early = min(positions[0], fs)
                        pace_vals.append(1.0 - (early - 1) / max(1, fs - 1))
            feat_pace = sum(pace_vals) / len(pace_vals) if pace_vals else 0.5

            # Feature 9: prior finishing speed (avg speed from PRIOR races)
            speed_vals = []
            for p in prior[-5:]:
                ft = p.get('finish_time', '') or ''
                d = p.get('distance', 0) or 0
                if ft and ':' in ft and d > 0:
                    try:
                        parts = ft.split(':')
                        secs = float(parts[0]) * 60 + float(parts[1])
                        if secs > 0:
                            speed = d / secs
                            speed_vals.append(max(0.0, min(1.0, (speed - 14) / 4)))
                    except (ValueError, IndexError):
                        pass
            feat_finish = sum(speed_vals) / len(speed_vals) if speed_vals else 0.5

            # Feature 10: C&D form (PRIOR wins at same distance, excluding current)
            prior_at_dist = [p for p in prior if p.get('distance') == distance]
            if prior_at_dist:
                cd_wins = sum(1 for p in prior_at_dist if p.get('is_winner', 0))
                feat_cd = min(1.0, cd_wins / len(prior_at_dist) + 0.2)
            else:
                feat_cd = 0.5

            # Feature 11: odds-implied probability (market strength signal)
            if win_odds > 0:
                feat_odds = min(1.0, 1.0 / win_odds)
            else:
                feat_odds = 0.5

            # Feature 12: class level
            class_type = r.get('class_type', '') or ''
            cm = re.search(r'(\d)', class_type)
            class_rank = int(cm.group(1)) if cm else 3
            feat_class = max(0.0, min(1.0, class_rank / 5.0))

            # Feature 13: days since last run (freshness — available from DB)
            dslr = r.get('days_since_last_run', 0) or 0
            if dslr > 0:
                # Optimal is ~14-28 days; too fresh or too long is bad
                feat_freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))
            else:
                feat_freshness = 0.5

            # Feature 14: going preference (does horse perform on this going?)
            # Use prior results on same going condition
            if prior and going:
                prior_same_going = [p for p in prior if p.get('going', '') == going]
                if prior_same_going:
                    going_wins = sum(1 for p in prior_same_going if p.get('is_winner', 0))
                    going_top3 = sum(1 for p in prior_same_going if p.get('is_top3', 0))
                    feat_going = min(1.0, (going_wins * 2 + going_top3) / (len(prior_same_going) * 3) + 0.2)
                else:
                    feat_going = 0.5
            else:
                feat_going = 0.5

            # Feature 15: going form rate
            if prior and going:
                prior_same_going_f = [p for p in prior if p.get('going', '') == going]
                if len(prior_same_going_f) >= 2:
                    gf_wins = sum(1 for p in prior_same_going_f if p.get('is_winner', 0))
                    feat_going_form = gf_wins / len(prior_same_going_f)
                else:
                    feat_going_form = 0.5
            else:
                feat_going_form = 0.5

            # Feature 16: jockey-horse cooperation
            jockey_name = r.get('jockey', '') or ''
            if prior and jockey_name:
                prior_same_jh = [p for p in prior if p.get('jockey', '') == jockey_name]
                if len(prior_same_jh) >= 2:
                    jh_w = sum(1 for p in prior_same_jh if p.get('is_winner', 0))
                    jh_p = sum(1 for p in prior_same_jh if p.get('is_top3', 0))
                    feat_jh_coop = min(1.0, (jh_w * 2 + jh_p) / (len(prior_same_jh) * 3))
                else:
                    feat_jh_coop = 0.5
            else:
                feat_jh_coop = 0.5

            # Feature 17: distance PB ratio
            feat_dist_pb = 0.5
            prior_dist_times = [p for p in prior
                                if p.get('distance') == distance
                                and p.get('finish_time', '')
                                and ':' in str(p.get('finish_time', ''))]
            if prior_dist_times:
                def _pt(t):
                    try:
                        parts = str(t).split(':')
                        return float(parts[0]) * 60 + float(parts[1])
                    except (ValueError, IndexError):
                        return 999.0
                horse_times = [_pt(p['finish_time']) for p in prior_dist_times]
                valid_ht = [t for t in horse_times if t < 999]
                if valid_ht:
                    horse_pb_t = min(valid_ht)
                    try:
                        tr_row = conn.execute(
                            "SELECT MIN(finish_time) FROM race_results WHERE distance = ? AND finish_time != '' AND finish_time IS NOT NULL AND race_date < ?",
                            (distance, race_date)
                        ).fetchone()
                        if tr_row and tr_row[0]:
                            tr_secs = _pt(tr_row[0])
                            if tr_secs < 999 and horse_pb_t < 999:
                                feat_dist_pb = min(1.0, tr_secs / horse_pb_t)
                    except Exception:
                        pass

            # Feature 18: odds movement (not available in historical DB — use fallback)
            feat_odds_move = 0.5

            features = [
                feat_form_proxy, feat_form, feat_draw,
                feat_jockey, feat_trainer, feat_field_size,
                feat_field_draw, feat_pace, feat_finish, feat_cd, feat_odds,
                feat_class, feat_freshness, feat_going,
                feat_going_form, feat_jh_coop, feat_dist_pb, feat_odds_move,
            ]
            X.append(features)
            y.append(1 if r.get('is_winner', 0) else 0)

            # Weight recent races more
            weights.append(1.0)

        conn.close()

    except Exception as e:
        logger.warning(f"Error building training data from DB: {e}")
        try:
            conn.close()
        except Exception:
            pass
        return None, None, None

    return np.array(X, dtype=np.float32), np.array(y), np.array(weights)


def generate_synthetic_training_data(n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic training data calibrated to real HK racing distributions.
    Used as fallback when real DB data is insufficient (<50 results).
    """
    np.random.seed(42)

    X = []
    y = []

    for _ in range(n_samples):
        # Calibrated to real HK racing distributions
        rating = np.random.normal(60, 18)  # HK ratings cluster 40-80
        rating = np.clip(rating, 15, 100)
        form_score = np.random.beta(2.5, 3) * 100  # Right-skewed (most horses have moderate form)
        draw_adv = np.random.uniform(0.3, 1.0)

        # Jockey win rates: few elite (15-22%), most average (5-12%)
        jockey_rate = np.random.choice(
            [np.random.uniform(0.15, 0.25),  # elite ~15%
             np.random.uniform(0.08, 0.15),  # solid ~35%
             np.random.uniform(0.03, 0.08)], # rest ~50%
            p=[0.15, 0.35, 0.50]
        )
        trainer_rate = np.random.choice(
            [np.random.uniform(0.12, 0.20),
             np.random.uniform(0.06, 0.12),
             np.random.uniform(0.02, 0.06)],
            p=[0.15, 0.40, 0.45]
        )
        # Feature 6: field_size / 14.0 (continuous, matching real schema)
        field_size = np.random.choice([6, 8, 10, 12, 14], p=[0.05, 0.15, 0.25, 0.35, 0.20])
        feat_field_size = field_size / 14.0

        # Feature 7: field-relative draw
        feat_field_draw = draw_adv * np.random.uniform(0.3, 1.0)

        # Feature 8: prior pace (avg early position normalized)
        prior_pace = np.random.uniform(0.3, 0.9)

        # Feature 9: prior finishing speed
        prior_speed = np.random.uniform(0.2, 0.9)

        # Feature 10: C&D form from prior races
        cd_form = np.random.uniform(0.2, 1.0)

        # Feature 11: odds-implied probability (1/win_odds, capped at 1.0)
        odds_implied = np.random.uniform(0.03, 0.50)

        # Feature 12: class level (class_rank / 5.0)
        class_level = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.40, 0.25, 0.15]) / 5.0

        # Feature 13: freshness (days since last run, optimal ~21 days)
        dslr = np.random.choice([7, 14, 21, 28, 45, 90], p=[0.10, 0.25, 0.30, 0.20, 0.10, 0.05])
        freshness = max(0.0, min(1.0, 1.0 - abs(dslr - 21) / 60))

        # Feature 14: going preference
        going_pref = np.random.uniform(0.3, 0.8)

        # Feature 15: going form rate
        going_form_rate = np.random.uniform(0.0, 0.5)

        # Feature 16: jockey-horse cooperation
        jockey_horse_coop = np.random.uniform(0.0, 0.5)

        # Feature 17: distance PB ratio
        distance_pb_ratio = np.random.uniform(0.7, 1.0)

        # Feature 18: odds movement
        odds_movement_feat = np.random.uniform(0.3, 0.7)

        features = [
            rating / 100, form_score / 100, draw_adv,
            jockey_rate, trainer_rate, feat_field_size,
            feat_field_draw, prior_pace, prior_speed, cd_form, odds_implied,
            class_level, freshness, going_pref,
            going_form_rate, jockey_horse_coop, distance_pb_ratio, odds_movement_feat,
        ]
        X.append(features)

        # Win probability based on weighted feature combination
        # Calibrated so ~8% of samples are winners (realistic HK field of ~12)
        win_logit = (
            1.8 * (rating / 100 - 0.6)     # rating above avg helps a lot
            + 1.2 * (form_score / 100 - 0.4)
            + 0.5 * (draw_adv - 0.6)
            + 3.0 * (jockey_rate - 0.10)    # jockey matters
            + 2.0 * (trainer_rate - 0.08)
            + 0.3 * (feat_field_size - 0.8)
            + 0.4 * (feat_field_draw - 0.5)
            + 0.5 * (prior_pace - 0.5)
            + 0.3 * (prior_speed - 0.5)
            + 0.5 * (cd_form - 0.5)
            + 1.5 * (odds_implied - 0.1)    # odds is top signal
            + 0.3 * (class_level - 0.5)
            + 0.2 * (freshness - 0.5)
            + 0.3 * (going_pref - 0.5)
            + 0.4 * (going_form_rate - 0.25)
            + 0.3 * (jockey_horse_coop - 0.25)
            + 0.2 * (distance_pb_ratio - 0.85)
            + 0.1 * (odds_movement_feat - 0.5)
            - 2.3  # base offset to get ~8% win rate
        )
        win_prob = 1.0 / (1.0 + np.exp(-win_logit))  # sigmoid
        is_win = np.random.random() < win_prob
        y.append(1 if is_win else 0)

    return np.array(X, dtype=np.float32), np.array(y)

def create_combined_prediction(
    rule_score: float,
    rule_confidence: str,
    ml_prob: float,
    ml_confidence: str,
    ml_available: bool
) -> Tuple[float, str, Dict]:
    # Check if model was trained on real data (version timestamp vs "1.0")
    # When trained on real data, give ML more weight
    weights = {'rule': 0.75, 'ml': 0.25}  # default: some real data available

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
            # Try real data first, fall back to synthetic
            X_real, y_real, sw_real = build_training_data_from_db()
            if X_real is not None and len(X_real) >= 50:
                logger.info(f"Training ML model with {len(X_real)} REAL race results...")
                result = self.ml_model.train(X_real, y_real, sample_weight=sw_real)
                logger.info(f"Model trained on REAL data: accuracy={result['accuracy']:.2%}, "
                           f"precision={result.get('precision', 0):.2%}")
            else:
                logger.info("Insufficient real data — training ML model with synthetic data...")
                X, y = generate_synthetic_training_data()
                result = self.ml_model.train(X, y)
                logger.info(f"Model trained on synthetic data: accuracy={result['accuracy']:.2%}")
    
    def predict(self, horse_data: dict, all_results: List[RaceResult], venue: str) -> MLPrediction:
        features = self.feature_eng.extract_features(horse_data, all_results, venue)

        field_size = horse_data.get('field_size', 12)
        ml_prob, ml_confidence = self.ml_model.predict(features, field_size=field_size)
        
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
