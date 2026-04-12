#!/usr/bin/env python3
"""
HKJC Mark Six Lottery Analyzer
Fetches draw results and statistics via HKJC GraphQL API, then suggests
numbers based on frequency analysis, hot/cold patterns, and overdue tracking.
"""

import sys
import os
from datetime import datetime
from collections import Counter
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


GRAPHQL_ENDPOINT = "https://info.cld.hkjc.com/graphql/base/"

# Exact query used by bet.hkjc.com Mark Six Results page
MARKSIX_QUERY = """fragment lotteryDrawsFragment on LotteryDraw {
  id
  year
  no
  openDate
  closeDate
  drawDate
  status
  snowballCode
  snowballName_en
  snowballName_ch
  lotteryPool {
    sell
    status
    totalInvestment
    jackpot
    unitBet
    estimatedPrize
    derivedFirstPrizeDiv
    lotteryPrizes {
      type
      winningUnit
      dividend
    }
  }
  drawResult {
    drawnNo
    xDrawnNo
  }
}

fragment lotteryStatFragment on LotteryStat {
  year
  no
  drawDate
  drawnNumbers {
    lastDrawnIn
    totalNumber
    drawnNo
  }
}

query marksix {
  lotteryDraws {
    ...lotteryDrawsFragment
  }
  lotteryStats {
    ...lotteryStatFragment
  }
}"""


@dataclass
class DrawResult:
    draw_no: str
    date: str
    drawn_numbers: list   # 6 main numbers (sorted)
    extra_number: int     # bonus/extra number
    investment: float = 0
    jackpot: float = 0
    snowball: str = ""


@dataclass
class NumberStat:
    number: int
    total_drawn: int       # all-time times drawn
    last_drawn_in: int     # draws since last appearance (0 = latest draw)


@dataclass
class Mark6Analysis:
    total_draws_analyzed: int   # from lotteryStats (all-time draw count)
    number_stats: list          # list of NumberStat
    hot_numbers: list           # most frequent all-time
    cold_numbers: list          # least frequent all-time
    overdue_numbers: list       # longest gap since last drawn
    recent_draws: list          # DrawResult list
    suggested_numbers: list     # final 6 suggestions
    suggested_extra: int        # suggested extra number
    current_draw: dict          # info about tonight's draw


class Mark6DataFetcher:
    """Fetches Mark Six data from HKJC GraphQL API."""

    HEADERS = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "Origin": "https://bet.hkjc.com",
        "Referer": "https://bet.hkjc.com/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Not:A-Brand";v="99", "HeadlessChrome";v="145", "Chromium";v="145"',
        "sec-ch-ua-platform": '"macOS"',
        "sec-ch-ua-mobile": "?0",
    }

    def fetch_all(self) -> dict:
        """Fetch Mark Six draws and statistics in one call."""
        payload = {
            "operationName": "marksix",
            "variables": {},
            "query": MARKSIX_QUERY
        }
        return self._post_graphql(payload)

    def _post_graphql(self, payload: dict) -> dict:
        if HAS_REQUESTS:
            return self._post_requests(payload)
        elif HAS_HTTPX:
            return self._post_httpx(payload)
        else:
            print("  ERROR: Install requests or httpx: pip install requests")
            return {}

    def _post_requests(self, payload: dict) -> dict:
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            resp = requests.post(
                GRAPHQL_ENDPOINT, json=payload, headers=self.HEADERS,
                timeout=15, verify=False
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  GraphQL request failed: {e}")
            return {}

    def _post_httpx(self, payload: dict) -> dict:
        try:
            with httpx.Client(timeout=15, verify=False) as client:
                resp = client.post(GRAPHQL_ENDPOINT, json=payload, headers=self.HEADERS)
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            print(f"  GraphQL request failed: {e}")
            return {}


class Mark6Analyzer:
    """Analyzes Mark Six data to suggest numbers."""

    ALL_NUMBERS = list(range(1, 50))  # 1-49

    def analyze(self, api_data: dict) -> Mark6Analysis:
        data = api_data.get("data", {})
        lottery_draws = data.get("lotteryDraws") or []
        lottery_stats = data.get("lotteryStats") or []

        # Parse draws
        recent_draws = []
        current_draw = {}
        for d in lottery_draws:
            dr = d.get("drawResult", {})
            pool = d.get("lotteryPool", {})
            nums = dr.get("drawnNo", [])

            if d.get("status") == "Result" and nums:
                recent_draws.append(DrawResult(
                    draw_no=d.get("id", ""),
                    date=d.get("drawDate", ""),
                    drawn_numbers=sorted(nums),
                    extra_number=dr.get("xDrawnNo", 0),
                    investment=float(pool.get("totalInvestment", 0) or 0),
                    jackpot=float(pool.get("jackpot", 0) or 0),
                    snowball=d.get("snowballName_en", "")
                ))
            elif d.get("status") in ("Defined", "StartSell", "Selling"):
                current_draw = {
                    "id": d.get("id", ""),
                    "no": d.get("no", 0),
                    "close_time": d.get("closeDate", ""),
                    "jackpot": float(pool.get("jackpot", 0) or 0),
                    "investment": float(pool.get("totalInvestment", 0) or 0),
                    "estimated_prize": pool.get("derivedFirstPrizeDiv", ""),
                    "snowball": d.get("snowballName_en", ""),
                    "selling": pool.get("sell", False),
                }

        # Parse all-time statistics
        number_stats = []
        if lottery_stats:
            stat = lottery_stats[0]  # latest stats snapshot
            for ns in stat.get("drawnNumbers", []):
                number_stats.append(NumberStat(
                    number=ns["drawnNo"],
                    total_drawn=ns["totalNumber"],
                    last_drawn_in=ns["lastDrawnIn"]
                ))

        if not number_stats:
            return self._empty_analysis()

        # Sort by total frequency (descending)
        by_freq = sorted(number_stats, key=lambda x: -x.total_drawn)
        hot_numbers = [ns.number for ns in by_freq[:10]]
        cold_numbers = [ns.number for ns in by_freq[-10:]]

        # Sort by overdue (descending lastDrawnIn)
        by_overdue = sorted(number_stats, key=lambda x: -x.last_drawn_in)
        overdue_numbers = [ns.number for ns in by_overdue[:10]]

        # Estimate total draws from average frequency
        avg_total = sum(ns.total_drawn for ns in number_stats) / len(number_stats)
        # Each draw has 6 numbers out of 49, so total_draws ~ avg_total * 49 / 6
        estimated_total_draws = int(avg_total * 49 / 6)

        # Generate suggestions
        suggested = self._generate_suggestions(number_stats)

        return Mark6Analysis(
            total_draws_analyzed=estimated_total_draws,
            number_stats=number_stats,
            hot_numbers=hot_numbers,
            cold_numbers=cold_numbers,
            overdue_numbers=overdue_numbers,
            recent_draws=recent_draws,
            suggested_numbers=sorted(suggested[:6]),
            suggested_extra=suggested[6] if len(suggested) > 6 else suggested[-1],
            current_draw=current_draw
        )

    def _generate_suggestions(self, stats: list) -> list:
        """
        Score each number 1-49 using:
        - All-time frequency (30%) - historically hot numbers tend to stay hot
        - Overdue factor (35%) - due for a draw (mean reversion)
        - Balance bonus (15%) - spread across number ranges
        - Avoid just-drawn penalty (20%) - numbers drawn in last draw are less likely to repeat
        """
        max_total = max(ns.total_drawn for ns in stats)
        min_total = min(ns.total_drawn for ns in stats)
        max_gap = max(ns.last_drawn_in for ns in stats)

        scores = {}
        for ns in stats:
            # Frequency score (normalized)
            if max_total > min_total:
                freq_score = (ns.total_drawn - min_total) / (max_total - min_total)
            else:
                freq_score = 0.5

            # Overdue score (higher gap = more overdue = higher score)
            overdue_score = ns.last_drawn_in / max_gap if max_gap > 0 else 0

            # Just-drawn penalty (drawn in latest draw gets penalty)
            recency_penalty = 0.3 if ns.last_drawn_in == 0 else 0

            scores[ns.number] = (
                freq_score * 0.30 +
                overdue_score * 0.35 -
                recency_penalty * 0.20
            )

        # Rank by score
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        candidates = [n for n, _ in ranked[:25]]

        # Balance across ranges: 1-9, 10-19, 20-29, 30-39, 40-49
        selected = []
        ranges = [(1, 10), (10, 20), (20, 30), (30, 40), (40, 50)]

        for lo, hi in ranges:
            range_picks = [n for n in candidates if lo <= n < hi]
            if range_picks and len(selected) < 6:
                selected.append(range_picks[0])

        # Fill remaining from top candidates
        for n in candidates:
            if n not in selected and len(selected) < 7:
                selected.append(n)

        # 7th number becomes the extra number suggestion
        if len(selected) >= 7:
            return sorted(selected[:6]) + [selected[6]]

        return sorted(selected[:6]) + [candidates[6] if len(candidates) > 6 else 1]

    def _empty_analysis(self):
        return Mark6Analysis(
            total_draws_analyzed=0, number_stats=[], hot_numbers=[],
            cold_numbers=[], overdue_numbers=[], recent_draws=[],
            suggested_numbers=[], suggested_extra=0, current_draw={}
        )


class Mark6ReportGenerator:
    """Generates formatted Mark Six analysis reports."""

    @staticmethod
    def print_report(analysis: Mark6Analysis):
        print("\n" + "=" * 70)
        print("  HKJC MARK SIX LOTTERY ANALYZER")
        print("=" * 70)

        cd = analysis.current_draw
        if cd:
            print(f"\n  Tonight's Draw: #{cd.get('id', '?')}")
            if cd.get("snowball"):
                print(f"  Type: {cd['snowball']}")
            jp = cd.get("jackpot", 0)
            print(f"  Jackpot: HK${jp:,.0f}" if jp else "  Jackpot: TBD")
            est = cd.get("estimated_prize", "")
            if est:
                print(f"  Est. 1st Prize: HK${float(est):,.0f}")
            inv = cd.get("investment", 0)
            if inv:
                print(f"  Total Investment: HK${inv:,.0f}")
            ct = cd.get("close_time", "")
            if ct:
                try:
                    close_dt = datetime.fromisoformat(ct.replace("+08:00", "+08:00"))
                    print(f"  Closes: {close_dt.strftime('%Y-%m-%d %H:%M')} HKT")
                except Exception:
                    print(f"  Closes: {ct}")
            if cd.get("selling"):
                print("  Status: SELLING NOW")

        print(f"\n  All-time data: ~{analysis.total_draws_analyzed:,} draws analyzed")
        print("-" * 70)

        # Recent results
        if analysis.recent_draws:
            print("\n  LAST DRAW RESULT")
            print("  " + "-" * 55)
            for d in analysis.recent_draws:
                nums = "  ".join(f"{n:>2}" for n in d.drawn_numbers)
                date_str = d.date[:10] if d.date else "?"
                print(f"  #{d.draw_no} | {date_str}")
                print(f"  Numbers:  {nums}  +  {d.extra_number}")
                if d.jackpot:
                    print(f"  Jackpot: HK${d.jackpot:,.0f} | Investment: HK${d.investment:,.0f}")

        # Hot numbers (all-time most frequent)
        print(f"\n  HOT NUMBERS (most frequent all-time)")
        print("  " + "-" * 55)
        for ns in sorted(analysis.number_stats, key=lambda x: -x.total_drawn)[:10]:
            bar = "#" * (ns.total_drawn // 20)
            print(f"  {ns.number:>2}: {ns.total_drawn:>3} times  {bar}")

        # Cold numbers
        print(f"\n  COLD NUMBERS (least frequent all-time)")
        print("  " + "-" * 55)
        for ns in sorted(analysis.number_stats, key=lambda x: x.total_drawn)[:10]:
            bar = "#" * (ns.total_drawn // 20)
            print(f"  {ns.number:>2}: {ns.total_drawn:>3} times  {bar}")

        # Overdue
        print(f"\n  OVERDUE NUMBERS (most draws since last appearance)")
        print("  " + "-" * 55)
        for ns in sorted(analysis.number_stats, key=lambda x: -x.last_drawn_in)[:10]:
            status = "JUST DRAWN" if ns.last_drawn_in == 0 else f"{ns.last_drawn_in} draws ago"
            print(f"  {ns.number:>2}: {status}")

        # Numbers drawn in last draw (to show pattern)
        just_drawn = [ns.number for ns in analysis.number_stats if ns.last_drawn_in == 0]
        if just_drawn:
            print(f"\n  JUST DRAWN (in latest draw #{analysis.recent_draws[0].draw_no if analysis.recent_draws else '?'})")
            print("  " + "-" * 55)
            print(f"  {', '.join(str(n) for n in sorted(just_drawn))}")

        # Suggestions
        print("\n" + "=" * 70)
        print("  SUGGESTED NUMBERS FOR TONIGHT")
        print("=" * 70)
        if analysis.suggested_numbers:
            nums = "  ".join(f"[{n:>2}]" for n in analysis.suggested_numbers)
            print(f"\n  Main:  {nums}  +  [{analysis.suggested_extra:>2}]")
            print()
            print("  Method: Weighted scoring combining:")
            print("    - All-time frequency (30%) - historically lucky numbers")
            print("    - Overdue factor (35%) - numbers due for appearance")
            print("    - Just-drawn avoidance (20%) - less likely to repeat")
            print("    - Range balance (15%) - spread across 1-49")
        else:
            print("  No data available for suggestions")

        print("\n" + "-" * 70)
        print("  Disclaimer: Lottery is purely random. No method guarantees wins.")
        print("  These suggestions are for entertainment only. Play responsibly.")
        print("=" * 70 + "\n")

    @staticmethod
    def save_report(analysis: Mark6Analysis, filepath: str):
        """Save analysis to file."""
        with open(filepath, 'w') as f:
            f.write("HKJC MARK SIX ANALYSIS\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Based on: ~{analysis.total_draws_analyzed:,} all-time draws\n")
            f.write("=" * 60 + "\n\n")

            cd = analysis.current_draw
            if cd:
                f.write(f"Draw: #{cd.get('id', '?')}\n")
                f.write(f"Jackpot: HK${cd.get('jackpot', 0):,.0f}\n\n")

            f.write("SUGGESTED NUMBERS:\n")
            nums = " ".join(str(n) for n in analysis.suggested_numbers)
            f.write(f"  {nums} + {analysis.suggested_extra}\n\n")

            f.write("HOT (all-time): " + ", ".join(str(n) for n in analysis.hot_numbers) + "\n")
            f.write("COLD (all-time): " + ", ".join(str(n) for n in analysis.cold_numbers) + "\n")
            f.write("OVERDUE: " + ", ".join(str(n) for n in analysis.overdue_numbers) + "\n\n")

            if analysis.recent_draws:
                f.write("LAST DRAW:\n")
                for d in analysis.recent_draws:
                    nums = " ".join(f"{n:>2}" for n in d.drawn_numbers)
                    f.write(f"  {d.draw_no} | {d.date[:10]} | {nums} + {d.extra_number}\n")


def main():
    print("\n  Fetching Mark Six data from HKJC...")

    fetcher = Mark6DataFetcher()
    analyzer = Mark6Analyzer()

    # Single API call gets everything
    api_data = fetcher.fetch_all()

    if not api_data or not api_data.get("data"):
        print("  Failed to fetch Mark Six data. Check your internet connection.")
        return

    # Analyze
    analysis = analyzer.analyze(api_data)

    if not analysis.number_stats:
        print("  No statistics data available.")
        return

    # Print report
    Mark6ReportGenerator.print_report(analysis)

    # Save to file
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, f"mark6_{datetime.now().strftime('%Y%m%d')}.txt")
    Mark6ReportGenerator.save_report(analysis, filepath)
    print(f"  Results saved to: {filepath}")


if __name__ == "__main__":
    main()
