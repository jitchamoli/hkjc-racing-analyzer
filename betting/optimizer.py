#!/usr/bin/env python3
"""
HKJC Betting Optimizer v2.0 - LONG SHOT VALUE STRATEGY
Maximizes returns by finding horses where our probability > market implied probability
"""

import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from itertools import combinations

@dataclass
class BetOption:
    bet_type: str
    selection: str
    cost: float
    odds: float
    probability: float
    expected_return: float
    roi_percent: float
    risk_level: str
    value_ratio: float = 0.0
    value_label: str = ""

@dataclass
class BettingRecommendation:
    horse: str
    horse_number: int
    best_bet: str
    suggested_amount: float
    potential_return: float
    roi: float
    confidence: str
    value_ratio: float
    value_label: str
    reasoning: List[str]
    alternatives: List[BetOption]

class BettingOptimizer:
    def __init__(self, min_investment: float = 10, max_investment: float = 200):
        self.min_investment = min_investment
        self.max_investment = max_investment
    
    def calculate_value_score(self, odds: float, probability: float) -> Tuple[float, str]:
        """
        NEW STRATEGY: Find VALUE BETS
        Value = Our Probability > Market Implied Probability
        """
        market_implied = 1 / odds if odds > 0 else 0
        
        value_ratio = probability / market_implied if market_implied > 0 else 0
        
        if value_ratio > 5.0:
            return value_ratio, "⭐⭐⭐⭐⭐ JUMBO VALUE"
        elif value_ratio > 3.0:
            return value_ratio, "⭐⭐⭐⭐ MASSIVE VALUE"
        elif value_ratio > 2.0:
            return value_ratio, "⭐⭐⭐ GREAT VALUE"
        elif value_ratio > 1.3:
            return value_ratio, "⭐⭐ GOOD VALUE"
        elif value_ratio > 1.0:
            return value_ratio, "⭐ BREAK EVEN"
        else:
            return value_ratio, "❌ NO VALUE"
    
    def calculate_win_bet(self, horse_data: dict, probability: float) -> BetOption:
        odds = horse_data.get('win_odds', 5.0)
        
        market_implied = 1 / odds if odds > 0 else 0
        
        expected_return = (odds * probability) - 1
        roi_percent = expected_return * 100
        
        value_ratio, value_label = self.calculate_value_score(odds, probability)
        
        risk = self._get_risk_level(probability, expected_return)
        
        return BetOption(
            bet_type='WIN',
            selection=str(horse_data.get('number', '')),
            cost=10,
            odds=odds,
            probability=probability,
            expected_return=expected_return,
            roi_percent=roi_percent,
            risk_level=risk,
            value_ratio=value_ratio,
            value_label=value_label
        )
    
    def calculate_place_bet(self, horse_data: dict, probability: float) -> BetOption:
        place_odds = horse_data.get('place_odds', 1.3)
        
        place_prob = min(0.85, probability * 1.8)  # empirical HK top-3 ratio
        expected_return = (place_odds * place_prob) - 1
        roi_percent = expected_return * 100
        
        value_ratio, value_label = self.calculate_value_score(place_odds, place_prob)
        
        risk = self._get_risk_level(place_prob, expected_return)
        
        return BetOption(
            bet_type='PLACE',
            selection=str(horse_data.get('number', '')),
            cost=10,
            odds=place_odds,
            probability=place_prob,
            expected_return=expected_return,
            roi_percent=roi_percent,
            risk_level=risk,
            value_ratio=value_ratio,
            value_label=value_label
        )
    
    def optimize_single_bet(
        self, 
        horse_data: dict, 
        combined_probability: float
    ) -> BettingRecommendation:
        win_bet = self.calculate_win_bet(horse_data, combined_probability)
        place_bet = self.calculate_place_bet(horse_data, combined_probability)
        
        # Normalize both to similar scale: value_ratio is ~0-3, roi as fraction ~0-2
        value_score = win_bet.value_ratio * 0.6 + (win_bet.roi_percent / 100) * 0.4
        
        all_bets = [
            (win_bet, value_score),
            (place_bet, value_score * 0.5)
        ]
        
        all_bets.sort(key=lambda x: x[1], reverse=True)
        best, _ = all_bets[0]
        alternatives = [b[0] for b in all_bets[1:] if b[1] > 0]
        
        suggested_amount = self._calculate_optimal_amount(win_bet, combined_probability)
        potential_return = suggested_amount * best.odds if best.expected_return > 0 else suggested_amount
        
        reasoning = [
            f"Our probability: {combined_probability*100:.1f}%",
            f"Market implied: {1/best.odds*100:.1f}%",
            f"Odds: {best.odds:.1f}",
            f"Value ratio: {best.value_ratio:.2f}x",
            f"{best.value_label}"
        ]
        
        return BettingRecommendation(
            horse=horse_data.get('name', ''),
            horse_number=horse_data.get('number', 0),
            best_bet=best.bet_type,
            suggested_amount=suggested_amount,
            potential_return=potential_return,
            roi=(potential_return - suggested_amount) / suggested_amount * 100 if suggested_amount > 0 else 0,
            confidence=self._get_confidence_level(combined_probability),
            value_ratio=best.value_ratio,
            value_label=best.value_label,
            reasoning=reasoning,
            alternatives=alternatives
        )
    
    def optimize_long_shot_strategy(
        self,
        horses: List[dict],
        probabilities: List[float]
    ) -> List[BettingRecommendation]:
        """
        NEW: Prioritize LONG SHOTS with high value
        """
        recommendations = []
        
        scored_horses = []
        for horse, prob in zip(horses, probabilities):
            odds = horse.get('win_odds', 5.0)
            value_ratio, value_label = self.calculate_value_score(odds, prob)
            
            long_shot_score = value_ratio * (odds / 5.0)
            
            scored_horses.append({
                'horse': horse,
                'probability': prob,
                'odds': odds,
                'value_ratio': value_ratio,
                'value_label': value_label,
                'long_shot_score': long_shot_score
            })
        
        scored_horses.sort(key=lambda x: x['long_shot_score'], reverse=True)
        
        for scored in scored_horses[:5]:
            rec = self.optimize_single_bet(scored['horse'], scored['probability'])
            recommendations.append(rec)
        
        return recommendations
    
    def generate_value_ticket(
        self,
        race_analyses: List[dict],
        budget: float = 100
    ) -> Dict:
        """
        NEW: Generate ticket prioritizing LONG SHOT VALUE
        """
        recommendations = self.optimize_long_shot_strategy(
            [a.get('horse_data', {}) for a in race_analyses],
            [a.get('combined_probability', 0.2) for a in race_analyses]
        )
        
        tickets = []
        total_cost = 0
        
        for rec in recommendations:
            if total_cost + rec.suggested_amount > budget:
                continue
            
            if rec.value_ratio > 1.5:
                tickets.append({
                    'horse': rec.horse,
                    'bet_type': 'WIN',
                    'amount': rec.suggested_amount,
                    'odds': rec.potential_return / rec.suggested_amount if rec.suggested_amount > 0 else 0,
                    'potential_return': rec.potential_return,
                    'value_ratio': rec.value_ratio,
                    'value_label': rec.value_label
                })
                total_cost += rec.suggested_amount
        
        max_return = sum(t.get('potential_return', 0) for t in tickets)
        # Best single ticket return (realistic — only one bet wins per race)
        best_single = max((t.get('potential_return', 0) for t in tickets), default=0)
        return {
            'total_investment': total_cost,
            'max_return': max_return,
            'tickets': tickets,
            'best_single_roi': ((best_single - total_cost) / total_cost * 100) if total_cost > 0 else 0,
        }
    
    def _get_risk_level(self, probability: float, expected_return: float) -> str:
        if probability > 0.4 and expected_return > 0.2:
            return "LOW 🟢"
        elif probability > 0.25 or expected_return > 0.1:
            return "MEDIUM 🟡"
        elif probability > 0.15:
            return "HIGH 🟠"
        else:
            return "VERY HIGH 🔴"
    
    def _get_confidence_level(self, probability: float) -> str:
        if probability > 0.4:
            return "VERY HIGH ⭐⭐⭐⭐"
        elif probability > 0.3:
            return "HIGH ⭐⭐⭐"
        elif probability > 0.2:
            return "MEDIUM ⭐⭐"
        else:
            return "LOW ⭐"
    
    def _calculate_optimal_amount(self, bet: BetOption, probability: float) -> float:
        if bet.expected_return <= 0:
            return 10
        
        if bet.odds <= 1.0:
            return self.min_investment
        b = bet.odds - 1
        kelly_fraction = (b * bet.probability - (1 - bet.probability)) / b
        kelly_fraction = max(0.0, min(0.25, kelly_fraction))
        if kelly_fraction == 0:
            return self.min_investment

        optimal = self.min_investment + (self.max_investment - self.min_investment) * kelly_fraction
        
        return round(optimal / 10) * 10


class BettingReportGenerator:
    @staticmethod
    def print_betting_analysis(
        race: dict,
        analyses: List[dict],
        budget: float = 100
    ):
        optimizer = BettingOptimizer(min_investment=10, max_investment=200)
        
        print(f"\n{'='*80}")
        print(f"💰 LONG SHOT VALUE STRATEGY - Race {race.get('race_number', '')}")
        print(f"{'='*80}")
        print(f"Available Budget: ${budget:.2f}")
        print("-"*80)
        
        print("\n📊 VALUE ANALYSIS (All Horses)")
        print("-"*80)
        print(f"{'Horse':<18}{'Odds':<8}{'Our Prob':<10}{'Market':<10}{'Value':<10}{'Status'}")
        print("-"*80)
        
        all_recs = optimizer.optimize_long_shot_strategy(
            [a.get('horse_data', {}) for a in analyses],
            [a.get('combined_probability', 0.2) for a in analyses]
        )
        
        for rec in all_recs:
            horse_data = rec.alternatives[0] if rec.alternatives else None
            if horse_data:
                odds = horse_data.odds
            else:
                odds = 5.0
            
            market = 1/odds*100 if odds > 0 else 0
            our_prob = rec.confidence
            
            print(f"{rec.horse:<18}{odds:<8.1f}{rec.confidence:<10}{market:<10.1f}{rec.value_ratio:<10.2f}{rec.value_label}")
        
        print("\n" + "="*80)
        print("🎯 LONG SHOT VALUE PICKS")
        print("="*80)
        
        value_picks = [r for r in all_recs if r.value_ratio > 1.5][:5]
        
        if value_picks:
            print("\n🏆 TOP VALUE BETS (Our prob > Market implied):")
            for i, rec in enumerate(value_picks, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
                print(f"\n{medal} {rec.horse} (#{rec.horse_number})")
                print(f"   Odds: {rec.alternatives[0].odds if rec.alternatives else 0:.1f}")
                print(f"   Our Probability: {rec.confidence}")
                print(f"   Value Ratio: {rec.value_ratio:.2f}x {rec.value_label}")
                print(f"   Suggested Bet: ${rec.suggested_amount:.0f} WIN")
                print(f"   Potential Return: ${rec.potential_return:.0f}")
                print(f"   Reason: {rec.reasoning[4]}")
        
        long_shots = [r for r in all_recs if (r.alternatives[0].odds if r.alternatives else 0) > 10][:3]
        
        if long_shots:
            print("\n🎲 LONG SHOT SPECIALS (15+ odds):")
            for rec in long_shots:
                odds = rec.alternatives[0].odds if rec.alternatives else 0
                if odds > 10:
                    print(f"\n   🐎 {rec.horse}")
                    print(f"      Odds: {odds:.1f} | Value: {rec.value_ratio:.2f}x")
                    print(f"      Bet: ${rec.suggested_amount:.0f} → Return: ${rec.potential_return:.0f}")
        
        ticket = optimizer.generate_value_ticket(analyses, budget)
        
        print("\n" + "="*80)
        print("🎫 SUGGESTED VALUE TICKET")
        print("="*80)
        print(f"\nTotal Investment: ${ticket['total_investment']:.0f}")
        print(f"Maximum Return: ${ticket['max_return']:.0f}")
        print(f"Best Single Bet ROI: {ticket['best_single_roi']:.0f}%")
        
        if ticket['tickets']:
            print("\nBets:")
            for t in ticket['tickets']:
                print(f"  {t['horse']} - {t['bet_type']} (${t['amount']:.0f}) @ {t['odds']:.1f} → ${t['potential_return']:.0f}")
                print(f"    {t['value_label']}")
        
        print("\n" + "="*80)
        print("⚠️  NEW STRATEGY: LONG SHOT VALUE")
        print("="*80)
        print("""
• Find horses where OUR probability > MARKET implied probability
• Example: Horse at 20.0 odds implies 5% chance
• If OUR model says 15% chance = 3x VALUE!
• Lower hit rate but MASSIVE returns when they hit
• MEGA MASTERMIND at 30.0 was a classic value bet!
        """)
