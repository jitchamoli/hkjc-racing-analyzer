#!/usr/bin/env python3
"""
HKJC Results Tracker - Records predictions vs actual results for learning
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Optional
from pathlib import Path

@dataclass
class PredictionRecord:
    date: str
    race_number: int
    horse_name: str
    predicted_probability: float
    predicted_rank: int
    actual_position: Optional[int]
    odds_used: float
    actual_odds: float
    bet_type: str
    bet_amount: float
    won: bool
    profit_loss: float

class ResultsTracker:
    def __init__(self):
        self.data_dir = Path(__file__).parent / 'tracking_data'
        self.data_dir.mkdir(exist_ok=True)
        self.records_file = self.data_dir / 'predictions.json'
        self.records: List[PredictionRecord] = []
        self._load_records()
    
    def _load_records(self):
        if self.records_file.exists():
            with open(self.records_file, 'r') as f:
                data = json.load(f)
                self.records = [PredictionRecord(**r) for r in data]
    
    def _save_records(self):
        with open(self.records_file, 'w') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
    
    def add_prediction(
        self,
        date: str,
        race_number: int,
        horse_name: str,
        predicted_probability: float,
        predicted_rank: int,
        odds_used: float,
        bet_type: str = 'WIN',
        bet_amount: float = 10
    ):
        record = PredictionRecord(
            date=date,
            race_number=race_number,
            horse_name=horse_name,
            predicted_probability=predicted_probability,
            predicted_rank=predicted_rank,
            actual_position=None,
            odds_used=odds_used,
            actual_odds=0,
            bet_type=bet_type,
            bet_amount=bet_amount,
            won=False,
            profit_loss=-bet_amount
        )
        self.records.append(record)
        self._save_records()
        return len(self.records) - 1
    
    def update_result(
        self,
        record_index: int,
        actual_position: int,
        actual_odds: float
    ):
        if 0 <= record_index < len(self.records):
            record = self.records[record_index]
            record.actual_position = actual_position
            record.actual_odds = actual_odds
            
            if record.bet_type == 'WIN':
                record.won = (actual_position == 1)
                if record.won:
                    record.profit_loss = record.bet_amount * actual_odds - record.bet_amount
            elif record.bet_type == 'PLACE':
                record.won = (actual_position <= 3)
                if record.won:
                    record.profit_loss = record.bet_amount * actual_odds - record.bet_amount
            
            self._save_records()
    
    def get_performance_stats(self) -> dict:
        if not self.records:
            return {
                'total_bets': 0,
                'wins': 0,
                'win_rate': 0,
                'total_invested': 0,
                'total_return': 0,
                'roi': 0,
                'avg_odds': 0
            }
        
        completed = [r for r in self.records if r.actual_position is not None]
        if not completed:
            return {
                'total_bets': len(self.records),
                'pending_results': len(self.records),
                'wins': 0,
                'win_rate': 0
            }
        
        wins = sum(1 for r in completed if r.won)
        total_invested = sum(r.bet_amount for r in completed)
        total_return = sum(r.profit_loss + r.bet_amount for r in completed if r.won)
        
        return {
            'total_bets': len(completed),
            'wins': wins,
            'win_rate': wins / len(completed) * 100 if completed else 0,
            'total_invested': total_invested,
            'total_return': total_return,
            'roi': (total_return - total_invested) / total_invested * 100 if total_invested > 0 else 0,
            'avg_odds': sum(r.actual_odds for r in completed) / len(completed) if completed else 0,
            'pending': len([r for r in self.records if r.actual_position is None])
        }
    
    def get_accuracy_by_rank(self) -> dict:
        completed = [r for r in self.records if r.actual_position is not None]
        
        accuracy = {}
        for rank in range(1, 6):
            predicted_rank = [r for r in completed if r.predicted_rank == rank]
            if predicted_rank:
                correct = sum(1 for r in predicted_rank if r.actual_position == 1)
                accuracy[f'rank_{rank}'] = {
                    'count': len(predicted_rank),
                    'won': correct,
                    'accuracy': correct / len(predicted_rank) * 100
                }
        
        return accuracy
    
    def get_best_bet_type(self) -> dict:
        by_type = {}
        for r in self.records:
            if r.actual_position is None:
                continue
            if r.bet_type not in by_type:
                by_type[r.bet_type] = {'wins': 0, 'total': 0, 'profit': 0}
            by_type[r.bet_type]['total'] += 1
            if r.won:
                by_type[r.bet_type]['wins'] += 1
            by_type[r.bet_type]['profit'] += r.profit_loss
        
        for bet_type, stats in by_type.items():
            stats['win_rate'] = stats['wins'] / stats['total'] * 100 if stats['total'] > 0 else 0
            stats['roi'] = stats['profit'] / (stats['total'] * 10) * 100 if stats['total'] > 0 else 0
        
        return by_type
    
    def print_performance_report(self):
        stats = self.get_performance_stats()
        accuracy = self.get_accuracy_by_rank()
        bet_types = self.get_best_bet_type()
        
        print("\n" + "="*60)
        print("📊 PREDICTION PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n📈 OVERALL STATS:")
        print(f"   Total Bets: {stats['total_bets']}")
        print(f"   Wins: {stats['wins']}")
        print(f"   Win Rate: {stats['win_rate']:.1f}%")
        print(f"   Total Invested: ${stats['total_invested']:.0f}")
        print(f"   Total Return: ${stats['total_return']:.0f}")
        print(f"   ROI: {stats['roi']:.1f}%")
        
        if accuracy:
            print(f"\n🎯 ACCURACY BY PREDICTED RANK:")
            for rank, data in accuracy.items():
                print(f"   #{rank.split('_')[1]}: {data['accuracy']:.1f}% ({data['count']} bets)")
        
        if bet_types:
            print(f"\n💰 PERFORMANCE BY BET TYPE:")
            for bet_type, data in bet_types.items():
                print(f"   {bet_type}: {data['win_rate']:.1f}% win rate, {data['roi']:.1f}% ROI")
        
        print("\n" + "="*60)
    
    def export_csv(self, filename: str = None):
        if not filename:
            filename = f"predictions_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = self.data_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("Date,Race,Horse,PredictedProb,PredictedRank,ActualPosition,Odds,ActualOdds,BetType,Won,ProfitLoss\n")
            for r in self.records:
                f.write(f"{r.date},{r.race_number},{r.horse_name},{r.predicted_probability:.2f},{r.predicted_rank},")
                f.write(f"{r.actual_position or ''},{r.odds_used:.1f},{r.actual_odds:.1f},")
                f.write(f"{r.bet_type},{r.won},{r.profit_loss:.2f}\n")
        
        return str(filepath)

if __name__ == "__main__":
    tracker = ResultsTracker()
    tracker.print_performance_report()
