#!/usr/bin/env python3
"""
HKJC Results Fetcher - Fetch actual race results from HKJC website
After races are finished, fetches results and updates dashboard
"""

import asyncio
import sys
import os
import json
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hkjc_data_fetcher import HKJCDataFetcher

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ResultsFetcher:
    def __init__(self):
        self.fetcher = HKJCDataFetcher()
        self.results_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(self.results_dir, exist_ok=True)
    
    async def fetch_race_results(self, race_date: str, race_no: int) -> List[Dict]:
        """Fetch actual results for a specific race."""
        logger.info(f"Fetching results for Race {race_no} on {race_date}")
        results = await self.fetcher.get_results(race_date, race_no)
        return results
    
    async def fetch_day_results(self, race_date: str) -> Dict[int, List[Dict]]:
        """Fetch results for all races on a given day."""
        logger.info(f"Fetching all results for {race_date}")
        
        day_results = {}
        
        for race_no in range(1, 11):
            results = await self.fetch_race_results(race_date, race_no)
            if results:
                day_results[race_no] = results
                logger.info(f"  Race {race_no}: {len(results)} finished")
            else:
                logger.info(f"  Race {race_no}: No results yet (may not be finished)")
        
        return day_results
    
    def load_prediction_data(self, race_date: str) -> Optional[Dict]:
        """Load the prediction JSON data."""
        date_str = race_date.replace('/', '')
        filepath = os.path.join(self.results_dir, f"race_data_{date_str}.json")
        
        if not os.path.exists(filepath):
            logger.warning(f"No prediction data found: {filepath}")
            return None
        
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def merge_results_with_predictions(self, predictions: Dict, actual_results: Dict) -> Dict:
        """Merge actual results with predictions."""
        for race in predictions:
            race_no = race.get('race_number')
            if race_no in actual_results:
                race['actual_results'] = actual_results[race_no]
                
                actual_positions = {
                    r['number']: r['position'] 
                    for r in actual_results[race_no]
                }
                
                for horse in race.get('horses', []):
                    horse_no = horse.get('number')
                    if horse_no in actual_positions:
                        horse['actual_position'] = actual_positions[horse_no]
                    else:
                        horse['actual_position'] = None
        
        return predictions
    
    def save_merged_data(self, data: Dict, race_date: str):
        """Save merged prediction + actual results data."""
        date_str = race_date.replace('/', '')
        filepath = os.path.join(self.results_dir, f"race_data_{date_str}_with_results.json")
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved merged data to {filepath}")
        return filepath
    
    async def update_dashboard_with_results(self, race_date: str):
        """Fetch results and update dashboard."""
        predictions = self.load_prediction_data(race_date)
        if not predictions:
            logger.error("No prediction data to update")
            return None
        
        actual_results = await self.fetch_day_results(race_date)
        
        if not actual_results:
            logger.warning("No actual results found - races may not be finished yet")
            return None
        
        merged = self.merge_results_with_predictions(predictions, actual_results)
        
        output_file = self.save_merged_data(merged, race_date)
        
        logger.info("Results fetched successfully!")
        logger.info(f"  Date: {race_date}")
        logger.info(f"  Races with results: {len(actual_results)}")
        
        for race_no, results in actual_results.items():
            winners = [r for r in results if r.get('position') == 1]
            if winners:
                logger.info(f"  Race {race_no} winner: #{winners[0]['number']} {winners[0]['name']}")
        
        return output_file


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fetch HKJC race results')
    parser.add_argument('date', nargs='?', help='Race date (YYYY/MM/DD), defaults to today')
    args = parser.parse_args()
    
    race_date = args.date or datetime.now().strftime('%Y/%m/%d')
    
    fetcher = ResultsFetcher()
    await fetcher.update_dashboard_with_results(race_date)


if __name__ == '__main__':
    asyncio.run(main())