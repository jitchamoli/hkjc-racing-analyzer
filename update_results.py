#!/usr/bin/env python3
"""
HKJC Results Update - Enter actual race results for tracking
"""

import sys
from datetime import datetime
import os

sys.path.insert(0, sys.path[0])

from results_tracker import ResultsTracker

def update_results():
    tracker = ResultsTracker()
    
    print("\n" + "="*60)
    print("📊 UPDATE RACE RESULTS")
    print("="*60)
    
    pending = [r for r in tracker.records if r.actual_position is None]
    
    if not pending:
        print("\nNo pending predictions to update.")
        tracker.print_performance_report()
        return
    
    print(f"\n{len(pending)} pending predictions")
    print("-"*60)
    
    for i, record in enumerate(pending, 1):
        print(f"\n{i}. Race {record.race_number} | {record.horse_name}")
        print(f"   Predicted rank: #{record.predicted_rank}")
        print(f"   Odds used: {record.odds_used:.1f}")
        
        while True:
            try:
                position = input("   Actual position (1-14, or 0 if scratched): ").strip()
                position = int(position) if position else 0
                if 0 <= position <= 14:
                    break
                print("   ⚠️  Enter a number between 0 and 14")
            except ValueError:
                print("   ⚠️  Enter a valid number")
        
        if position == 0:
            print(f"   ⚠️  Horse scratched - no bet placed")
            tracker.records.remove(record)
            continue
        
        while True:
            try:
                actual_odds = input(f"   Actual winning odds (e.g. 2.5): ").strip()
                actual_odds = float(actual_odds) if actual_odds else record.odds_used
                if 0 < actual_odds <= 100:
                    break
                print("   ⚠️  Enter odds between 0 and 100")
            except ValueError:
                print("   ⚠️  Enter a valid number")
        
        idx = tracker.records.index(record)
        tracker.update_result(idx, position, actual_odds)
        
        if position == 1:
            print(f"   ✅ WIN! Profit: ${record.profit_loss:.0f}")
        elif position <= 3:
            print(f"   ✅ Placed (top 3)")
        else:
            print(f"   ❌ Lost ${abs(record.profit_loss):.0f}")
    
    tracker._save_records()
    
    print("\n" + "="*60)
    tracker.print_performance_report()
    
    export = input("\nExport to CSV? (y/n): ").strip().lower()
    if export == 'y':
        filepath = tracker.export_csv()
        print(f"✅ Exported to: {filepath}")

if __name__ == "__main__":
    update_results()
