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
    
    def calculate_place_bet(self, horse_data: dict, probability: float,
                           place_probability: float = None) -> BetOption:
        place_odds = horse_data.get('place_odds', 1.3)

        # Use ML place model probability when available, else crude heuristic
        if place_probability is not None and place_probability > 0:
            place_prob = min(0.85, place_probability)
        else:
            place_prob = min(0.85, probability * 1.8)
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
        combined_probability: float,
        place_probability: float = None
    ) -> BettingRecommendation:
        win_bet = self.calculate_win_bet(horse_data, combined_probability)
        place_bet = self.calculate_place_bet(horse_data, combined_probability,
                                             place_probability=place_probability)
        
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



# ─────────────────────────────────────────────────────────────────────
# Bankroll-fraction stake suggestions for the dashboard.
# Independent of the legacy BettingOptimizer class above.
# ─────────────────────────────────────────────────────────────────────

# Per-tier base stake fractions. Conservative — tuned to keep the
# *typical* card's total exposure under 8-10% of bankroll across all
# bets when the average HK meeting has 2 HIGH-tier + 4 MED-tier + the
# rest LOW/SKIP. Half-Kelly equivalent for the WIN case.
_TIER_STAKE_FRACTION = {
    'EDGE': 0.030,   # 3.0% — rare; gets the largest single-bet allocation
    'HIGH': 0.020,   # 2.0%
    'MED':  0.010,   # 1.0%
    'LOW':  0.005,   # 0.5% — exposure only, no conviction
    'SKIP': 0.000,
}


def _confidence_scale(tier: str, top_2_share: float) -> float:
    """Within-tier confidence scaling on HIGH races (2026-05-17 learning).

    HIGH tier covers top_2_share 0.40-1.00. The bottom of that band
    (0.40-0.50) is meaningfully less concentrated than the top (>0.55).
    R8 (0.47) and R9 (0.55) both lost on full HIGH stake today, while
    R10 (0.45) hit. Variance dominates at any individual race, but
    long-run we should risk less when the model is just-above-threshold.

    Linear ramp: scale=0.75 at top_2_share=0.40, scale=1.0 at top_2_share=0.50+.
    EDGE / MED / LOW tiers unchanged — they have their own threshold logic.
    Bands chosen to keep current-meeting stakes within ±25% so we do not
    suddenly chop bets in half without forward validation.
    """
    if tier != 'HIGH' or top_2_share is None or top_2_share <= 0:
        return 1.0
    if top_2_share >= 0.50:
        return 1.0
    # Linear from (0.40, 0.75) to (0.50, 1.0).
    # Floor at 0.75 — HIGH tier sometimes fires below top_2=0.40 due to
    # tier-escalation rules; do not let stake collapse below 75%.
    scale = 0.75 + (top_2_share - 0.40) / 0.10 * 0.25
    return max(0.75, scale)


def compute_suggested_stake(*, conviction_tier: str, bet_type: str,
                            top_prob: float, top_odds: float,
                            bankroll: float,
                            top_2_share: float = None) -> dict:
    """Suggested per-card stake in HKD plus the % of bankroll it represents.

    Two paths:
    - WIN / BANKER WIN: half-Kelly on (top_prob, top_odds). Capped at the
      tier-fraction so a single hot bet can't blow the bankroll.
    - Exotic boxes (QIN/TRIO/F4/BOX Q4/etc): tier-based fraction. Kelly
      on multi-combo exotics requires the joint distribution of all
      combos and the dividend, which we don't have a clean handle on —
      so we use the tier as a proxy for confidence and let conviction
      gates do the size-discrimination work.

    top_2_share (optional): within-tier scaling on HIGH races — see
    _confidence_scale. Defaults to None which means no scaling (back-compat
    with callers that haven't passed the new param yet).

    Returns: {'amount': float HKD, 'pct': float of bankroll, 'method': str}
    """
    tier = (conviction_tier or 'SKIP').upper()
    frac_cap = _TIER_STAKE_FRACTION.get(tier, 0.0)
    # Within-tier confidence scaling (HIGH only — see _confidence_scale)
    conf_scale = _confidence_scale(tier, top_2_share)
    frac_cap = frac_cap * conf_scale
    bt = (bet_type or '').upper()

    # Top horse's bet is SKIP (recommender decided race-shape doesn't
    # support a PRIMARY pick). Return zero so we never suggest money on
    # a SKIP'd primary pick even when the tier is HIGH/MED — happens
    # rarely when conviction gates fail downstream of the tier label.
    if not bt or bt == 'SKIP':
        return {'amount': 0.0, 'pct': 0.0, 'method': 'skip (bet=SKIP)'}

    if frac_cap == 0 or bankroll <= 0:
        return {'amount': 0.0, 'pct': 0.0, 'method': 'skip'}

    if bet_type in ('WIN', 'BANKER WIN') and top_odds and top_odds > 1 and top_prob > 0:
        # Half-Kelly: f* = (b·p - q) / b, then × 0.5 for variance control.
        b = top_odds - 1
        p = max(0.0, min(1.0, top_prob))
        q = 1.0 - p
        kelly_full = (b * p - q) / b if b > 0 else 0.0
        kelly_half = max(0.0, kelly_full * 0.5)
        # Don't exceed the tier cap — Kelly on a high-odds longshot can
        # suggest 8-10% which violates the per-bet exposure budget.
        frac = min(kelly_half, frac_cap)
        amount = round(bankroll * frac, -1)  # round to nearest $10
        return {
            'amount': float(amount),
            'pct': round(frac * 100, 2),
            'method': f'half-Kelly capped at {tier} {frac_cap*100:.1f}%',
        }

    # Exotic boxes — use tier fraction directly. This is the "exposure"
    # for ALL combos in the box (not per-combo).
    amount = round(bankroll * frac_cap, -1)
    method = f'tier-fraction ({tier})'
    if conf_scale < 1.0 and top_2_share is not None:
        method += f' ×{conf_scale:.2f} (top_2={top_2_share:.2f})'
    return {
        'amount': float(amount),
        'pct': round(frac_cap * 100, 2),
        'method': method,
    }
