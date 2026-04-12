#!/usr/bin/env python3
"""
HKJC Racing - Interactive Mode with Real Odds Input
Allows manual entry of real odds for accurate predictions
"""

import asyncio
import sys
from datetime import datetime
import logging

sys.path.insert(0, sys.path[0])

from hkjc_racing_analyzer import (
    HKJCRacingScraper,
    CombinedRaceAnalyzer,
    ReportGenerator,
    Race,
    Horse
)
from betting_optimizer import BettingOptimizer, BettingReportGenerator
from results_tracker import ResultsTracker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InteractiveRacingAnalyzer:
    def __init__(self):
        self.scraper = None
        self.analyzer = CombinedRaceAnalyzer()
        self.tracker = ResultsTracker()
        self.current_race = None
        self.current_analyses = None
    
    async def run(self):
        print("\n" + "="*70)
        print("🏇 HKJC RACING - INTERACTIVE MODE")
        print("="*70)
        print("\nEnter race date or press Enter for today")
        
        race_date = input("Date (YYYY/MM/DD): ").strip()
        if not race_date:
            race_date = datetime.now().strftime('%Y/%m/%d')
        
        race_number = input("Race number (1-10): ").strip() or "1"
        race_number = int(race_number)
        
        self.scraper = HKJCRacingScraper()
        await self.scraper.init_browser()
        
        try:
            page = await self.scraper.context.new_page()
            
            print(f"\n📡 Fetching race data...")
            self.current_race = await self.scraper.scrape_race_card(page, race_date, race_number)
            
            if not self.current_race or not self.current_race.horses:
                print("❌ Could not fetch race data")
                return
            
            await page.close()
            
            self.current_analyses = self.analyzer.analyze_race(self.current_race)
            
            self._print_race_with_odds_input()
            
            self._offer_betting_analysis()
            
            self._offer_tracking()
            
        finally:
            await self.scraper.close_browser()
    
    def _print_race_with_odds_input(self):
        print(f"\n{'='*70}")
        print(f"🏇 RACE {self.current_race.race_number} - {self.current_race.venue}")
        print(f"{'='*70}")
        print(f"Date: {self.current_race.date} | Distance: {self.current_race.distance}m")
        print(f"Class: {self.current_race.class_type}")
        print("-"*70)
        
        print("\n📋 HORSE LIST (Enter REAL odds from HKJC platform)")
        print("-"*70)
        
        self.real_odds = {}
        
        for analysis in self.current_analyses[:14]:
            h = analysis.horse
            print(f"  {h.number:>2}. {h.name:<18} | Rating: {h.rating:.0f} | Jockey: {h.jockey}")
        
        print("\n" + "-"*70)
        print("Enter REAL WIN odds for each horse (or press Enter to skip):")
        print("-"*70)
        
        for analysis in self.current_analyses[:14]:
            h = analysis.horse
            while True:
                try:
                    user_input = input(f"  {h.name} (#{h.number}): ").strip()
                    if not user_input:
                        self.real_odds[h.number] = h.win_odds
                        break
                    odd = float(user_input)
                    if 1.0 <= odd <= 100.0:
                        self.real_odds[h.number] = odd
                        break
                    else:
                        print("    ⚠️  Odds should be between 1.0 and 100.0")
                except ValueError:
                    if user_input.lower() == 'q':
                        break
                    print("    ⚠️  Please enter a valid number")
        
        for analysis in self.current_analyses:
            analysis.horse.win_odds = self.real_odds.get(analysis.horse.number, analysis.horse.win_odds)
            analysis.horse.place_odds = analysis.horse.win_odds / 3
        
        print("\n✅ Real odds updated!")
    
    def _offer_betting_analysis(self):
        print("\n" + "="*70)
        print("💰 BETTING ANALYSIS (with real odds)")
        print("="*70)
        
        budget = input("\nEnter budget (HK$) [default: 100]: ").strip()
        try:
            budget = float(budget) if budget else 100
        except:
            budget = 100
        
        race_dict = {
            'race_number': self.current_race.race_number,
            'venue': self.current_race.venue,
            'distance': self.current_race.distance,
            'class_type': self.current_race.class_type
        }
        
        race_analyses = []
        for a in self.current_analyses:
            race_analyses.append({
                'race_number': self.current_race.race_number,
                'horse_data': {
                    'number': a.horse.number,
                    'name': a.horse.name,
                    'win_odds': a.horse.win_odds,
                    'place_odds': a.horse.place_odds,
                    'rating': a.horse.rating,
                    'jockey': a.horse.jockey,
                    'trainer': a.horse.trainer
                },
                'combined_probability': a.combined_probability,
                'expected_return': (a.horse.win_odds * a.combined_probability) - 1,
                'confidence': a.combined_confidence
            })
        
        BettingReportGenerator.print_betting_analysis(race_dict, race_analyses, budget)
        
        self.race_analyses = race_analyses
    
    def _offer_tracking(self):
        print("\n" + "="*70)
        print("📊 TRACKING")
        print("="*70)
        
        save = input("\nSave predictions for tracking? (y/n): ").strip().lower()
        
        if save == 'y':
            for analysis in self.race_analyses[:5]:
                self.tracker.add_prediction(
                    date=self.current_race.date,
                    race_number=self.current_race.race_number,
                    horse_name=analysis['horse_data']['name'],
                    predicted_probability=analysis['combined_probability'],
                    predicted_rank=self._get_rank(analysis),
                    odds_used=analysis['horse_data']['win_odds'],
                    bet_type='WIN',
                    bet_amount=10
                )
            
            print(f"✅ Saved {len(self.race_analyses[:5])} predictions")
            print("   After the race, update results to track performance!")
        
        show_stats = input("\nShow performance stats? (y/n): ").strip().lower()
        if show_stats == 'y':
            self.tracker.print_performance_report()
    
    def _get_rank(self, analysis):
        for i, a in enumerate(self.current_analyses, 1):
            if a['horse_data']['number'] == analysis['horse_data']['number']:
                return i
        return 99

async def main():
    analyzer = InteractiveRacingAnalyzer()
    await analyzer.run()

if __name__ == "__main__":
    asyncio.run(main())
