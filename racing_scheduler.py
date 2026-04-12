#!/usr/bin/env python3
"""
HKJC Racing Scheduler - Automated Daily Race Analysis
Runs before each HKJC race meeting to fetch and analyze all races
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hkjc_racing_analyzer import (
    HKJCRacingScraper,
    CombinedRaceAnalyzer,
    ReportGenerator
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RacingScheduler:
    HKJC_RACE_DAYS = {
        'Wednesday': 'Happy Valley (evening)',
        'Saturday': 'Sha Tin (day/afternoon)',
        'Sunday': 'Sha Tin (day/afternoon)',
    }
    
    def __init__(self):
        self.scraper = None
        self.analyzer = CombinedRaceAnalyzer()
        self.output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def run_full_analysis(self, race_date: str = None):
        if race_date is None:
            race_date = datetime.now().strftime('%Y/%m/%d')
        
        logger.info(f"Starting daily analysis for {race_date}")
        
        self.scraper = HKJCRacingScraper()
        await self.scraper.init_browser()
        
        output_file = os.path.join(
            self.output_dir,
            f"race_analysis_{race_date.replace('/', '')}.txt"
        )
        
        results = []
        
        try:
            page = await self.scraper.context.new_page()
            
            for race_num in range(1, 11):
                logger.info(f"Analyzing Race {race_num}...")
                
                race = await self.scraper.scrape_race_card(page, race_date, race_num)
                
                if not race or not race.horses:
                    logger.warning(f"No data for Race {race_num}, skipping...")
                    continue
                
                analyses = self.analyzer.analyze_race(race)
                
                race_results = self._format_race_results(race, analyses)
                results.append(race_results)
                
                await asyncio.sleep(3)
            
            await page.close()
            
            self._save_results(output_file, race_date, results)
            
            logger.info(f"Analysis complete! Results saved to: {output_file}")
            
            self._print_summary(results)
            
        finally:
            await self.scraper.close_browser()
        
        return results
    
    def _format_race_results(self, race, analyses) -> dict:
        return {
            'race_number': race.race_number,
            'venue': race.venue,
            'distance': race.distance,
            'class_type': race.class_type,
            'top_picks': [
                {
                    'rank': a.rank,
                    'horse': a.horse.name,
                    'odds': a.horse.win_odds,
                    'probability': f"{a.combined_probability*100:.1f}%",
                    'confidence': a.combined_confidence,
                    'bet': a.bet_recommendation
                }
                for a in analyses[:3]
            ]
        }
    
    def _save_results(self, filepath: str, race_date: str, results: list):
        with open(filepath, 'w') as f:
            f.write("="*80 + "\n")
            f.write(f"HKJC RACING ANALYSIS - {race_date}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            for r in results:
                f.write(f"\n{'='*60}\n")
                f.write(f"RACE {r['race_number']} - {r['venue']}\n")
                f.write(f"Distance: {r['distance']}m | Class: {r['class_type']}\n")
                f.write(f"{'='*60}\n\n")
                f.write("TOP PICKS:\n")
                f.write("-"*60 + "\n")
                
                for pick in r['top_picks']:
                    medal = "🥇" if pick['rank'] == 1 else "🥈" if pick['rank'] == 2 else "🥉"
                    f.write(f"{medal} #{pick['rank']} {pick['horse']:<20} | ")
                    f.write(f"Odds: {pick['odds']:<6} | ")
                    f.write(f"Prob: {pick['probability']:<8} | ")
                    f.write(f"Conf: {pick['confidence']:<15} | ")
                    f.write(f"Bet: {pick['bet']}\n")
                
                f.write("\n")
    
    def _print_summary(self, results: list):
        print("\n" + "="*60)
        print("📊 DAILY SUMMARY - ALL RACES")
        print("="*60)
        
        for r in results:
            print(f"\nRace {r['race_number']} ({r['venue']}, {r['distance']}m):")
            for pick in r['top_picks']:
                medal = "🥇" if pick['rank'] == 1 else "🥈" if pick['rank'] == 2 else "🥉"
                print(f"  {medal} {pick['horse']} | {pick['probability']} | {pick['confidence']}")
        
        print("\n" + "="*60)

async def main():
    scheduler = RacingScheduler()
    
    if len(sys.argv) > 1:
        race_date = sys.argv[1]
    else:
        race_date = datetime.now().strftime('%Y/%m/%d')
    
    print(f"\n🏇 HKJC Racing Scheduler - Daily Analysis")
    print(f"Date: {race_date}")
    print("-"*40)
    
    await scheduler.run_full_analysis(race_date)

if __name__ == "__main__":
    asyncio.run(main())
