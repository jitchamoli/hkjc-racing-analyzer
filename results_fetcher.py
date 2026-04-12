#!/usr/bin/env python3
"""
HKJC Results Fetcher - Fetch actual race results from HKJC website
After races are finished, fetches results, scores prediction accuracy,
and regenerates the interactive dashboard with results overlaid.
"""

import asyncio
import sys
import os
import json
import math
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

    async def fetch_day_results(self, race_date: str, num_races: int = 11) -> Dict[int, List[Dict]]:
        """Fetch results for all races on a given day."""
        logger.info(f"Fetching all results for {race_date}")

        day_results = {}

        for race_no in range(1, num_races + 1):
            results = await self.fetch_race_results(race_date, race_no)
            if results:
                day_results[race_no] = results
                logger.info(f"  Race {race_no}: {len(results)} finished")
            else:
                logger.info(f"  Race {race_no}: No results yet")

        return day_results

    def load_prediction_data(self, race_date: str) -> Optional[List]:
        """Load the prediction JSON data."""
        date_str = race_date.replace('/', '')
        filepath = os.path.join(self.results_dir, f"race_data_{date_str}.json")

        if not os.path.exists(filepath):
            logger.warning(f"No prediction data found: {filepath}")
            return None

        with open(filepath, 'r') as f:
            return json.load(f)

    def merge_results_with_predictions(self, predictions: List, actual_results: Dict) -> List:
        """Merge actual results into prediction data and score accuracy."""
        for race in predictions:
            race_no = race.get('race_number')
            if race_no in actual_results:
                race['actual_results'] = actual_results[race_no]
                race['results_available'] = True

                actual_positions = {
                    r['number']: r['position']
                    for r in actual_results[race_no]
                }
                actual_winner = next(
                    (r['number'] for r in actual_results[race_no] if r.get('position') == 1), None
                )
                actual_top3 = set(
                    r['number'] for r in actual_results[race_no] if r.get('position', 99) <= 3
                )
                actual_win_odds = next(
                    (r.get('win_odds', 0) for r in actual_results[race_no] if r.get('position') == 1), 0
                )

                race['actual_winner'] = actual_winner
                race['actual_top3'] = list(actual_top3)
                race['actual_win_odds'] = actual_win_odds

                for horse in race.get('horses', []):
                    horse_no = horse.get('number')
                    horse['actual_position'] = actual_positions.get(horse_no)

                # Score accuracy
                race['accuracy'] = self._score_race_accuracy(race)
            else:
                race['results_available'] = False

        return predictions

    @staticmethod
    def _score_race_accuracy(race: Dict) -> Dict:
        """Score how well our predictions matched actual results."""
        horses = race.get('horses', [])
        actual_winner = race.get('actual_winner')
        actual_top3 = set(race.get('actual_top3', []))

        if not horses or not actual_winner:
            return {}

        # Our top picks
        our_rank1 = horses[0]['number'] if horses else None
        our_rank2 = horses[1]['number'] if len(horses) > 1 else None
        our_rank3 = horses[2]['number'] if len(horses) > 2 else None
        our_top3 = {our_rank1, our_rank2, our_rank3} - {None}

        # Win prediction correct?
        win_correct = our_rank1 == actual_winner

        # Top 3 overlap
        top3_overlap = len(our_top3 & actual_top3)

        # What rank did the actual winner have in our predictions?
        winner_our_rank = None
        for h in horses:
            if h['number'] == actual_winner:
                winner_our_rank = h['rank']
                break

        # Did our bet recommendation hit?
        bet_hit = False
        profit = 0.0
        for h in horses:
            if h['number'] == actual_winner:
                if h['bet_recommendation'] in ('WIN',):
                    bet_hit = True
                    profit = h.get('win_odds', 0) - 1  # net profit per $1
                break
            elif h.get('actual_position') and h['actual_position'] <= 3:
                if h['bet_recommendation'] in ('PLACE', 'QIN', 'QNL PL', 'TRIO'):
                    bet_hit = True

        return {
            'win_correct': win_correct,
            'top3_overlap': top3_overlap,
            'winner_our_rank': winner_our_rank,
            'bet_hit': bet_hit,
            'profit': round(profit, 1),
        }

    def compute_day_summary(self, predictions: List) -> Dict:
        """Compute overall day accuracy summary."""
        races_with_results = [r for r in predictions if r.get('results_available')]
        if not races_with_results:
            return {}

        total = len(races_with_results)
        wins_correct = sum(1 for r in races_with_results if r.get('accuracy', {}).get('win_correct'))
        top3_total = sum(r.get('accuracy', {}).get('top3_overlap', 0) for r in races_with_results)
        bets_hit = sum(1 for r in races_with_results if r.get('accuracy', {}).get('bet_hit'))

        winner_ranks = [
            r['accuracy']['winner_our_rank']
            for r in races_with_results
            if r.get('accuracy', {}).get('winner_our_rank')
        ]
        avg_winner_rank = sum(winner_ranks) / len(winner_ranks) if winner_ranks else 0

        return {
            'total_races': total,
            'wins_correct': wins_correct,
            'win_accuracy': round(wins_correct / total * 100, 1) if total else 0,
            'top3_overlap_total': top3_total,
            'top3_accuracy': round(top3_total / (total * 3) * 100, 1) if total else 0,
            'bets_hit': bets_hit,
            'bet_hit_rate': round(bets_hit / total * 100, 1) if total else 0,
            'avg_winner_rank': round(avg_winner_rank, 1),
        }

    def save_merged_data(self, data: List, race_date: str, summary: Dict) -> str:
        """Save merged prediction + actual results data."""
        date_str = race_date.replace('/', '')
        output = {
            'race_date': race_date,
            'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'day_summary': summary,
            'races': data,
        }
        filepath = os.path.join(self.results_dir, f"race_data_{date_str}_with_results.json")

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved merged data to {filepath}")
        return filepath

    async def update_dashboard_with_results(self, race_date: str):
        """Fetch results, merge with predictions, score accuracy, regenerate dashboard."""
        predictions = self.load_prediction_data(race_date)
        if not predictions:
            logger.error("No prediction data to update")
            return None

        actual_results = await self.fetch_day_results(race_date)

        if not actual_results:
            logger.warning("No actual results found - races may not be finished yet")
            return None

        merged = self.merge_results_with_predictions(predictions, actual_results)
        summary = self.compute_day_summary(merged)

        output_file = self.save_merged_data(merged, race_date, summary)

        logger.info("Results fetched successfully!")
        logger.info(f"  Date: {race_date}")
        logger.info(f"  Races with results: {len(actual_results)}")
        if summary:
            logger.info(f"  Win accuracy: {summary['wins_correct']}/{summary['total_races']} ({summary['win_accuracy']}%)")
            logger.info(f"  Top 3 accuracy: {summary['top3_accuracy']}%")
            logger.info(f"  Avg winner rank in our predictions: {summary['avg_winner_rank']}")

        for race_no, results in sorted(actual_results.items()):
            winners = [r for r in results if r.get('position') == 1]
            if winners:
                logger.info(f"  Race {race_no} winner: #{winners[0]['number']} {winners[0]['name']}")

        # Regenerate dashboard with results
        self._regenerate_dashboard(merged, summary, race_date)

        return output_file

    def _regenerate_dashboard(self, data: List, summary: Dict, race_date: str):
        """Regenerate the interactive HTML dashboard with results overlaid."""
        date_str = race_date.replace('/', '')
        dashboard_path = os.path.join(self.results_dir, f"racing_dashboard_{date_str}.html")

        if not os.path.exists(dashboard_path):
            logger.warning(f"Dashboard not found: {dashboard_path}")
            return

        with open(dashboard_path, 'r') as f:
            html = f.read()

        # Inject results data into the existing dashboard
        results_json = json.dumps({
            'summary': summary,
            'races': [
                {
                    'race_number': r['race_number'],
                    'results_available': r.get('results_available', False),
                    'actual_results': r.get('actual_results', []),
                    'actual_winner': r.get('actual_winner'),
                    'actual_top3': r.get('actual_top3', []),
                    'actual_win_odds': r.get('actual_win_odds', 0),
                    'accuracy': r.get('accuracy', {}),
                    'horses': [
                        {
                            'number': h['number'],
                            'actual_position': h.get('actual_position'),
                        }
                        for h in r.get('horses', [])
                    ],
                }
                for r in data
            ],
        })

        # Inject before </script>
        inject_js = f'''

// ===== RESULTS DATA (injected by results_fetcher) =====
const RESULTS = {results_json};

function applyResults() {{
  if (!RESULTS || !RESULTS.races) return;

  // Merge actual positions into RACES data
  RESULTS.races.forEach(rr => {{
    const race = RACES.find(r => r.race_number === rr.race_number);
    if (!race) return;
    race.results_available = rr.results_available;
    race.actual_winner = rr.actual_winner;
    race.actual_top3 = rr.actual_top3 || [];
    race.actual_win_odds = rr.actual_win_odds;
    race.accuracy = rr.accuracy || {{}};

    rr.horses.forEach(rh => {{
      const h = race.horses.find(hh => hh.number === rh.number);
      if (h) h.actual_position = rh.actual_position;
    }});
  }});

  // Re-render current view
  const detail = document.getElementById('race-detail-view');
  if (detail && detail.style.display !== 'none') {{
    const activeTab = document.querySelector('.race-tab.active');
    if (activeTab && activeTab.id !== 'tab-all') {{
      const rn = parseInt(activeTab.id.replace('tab-', ''));
      showRace(rn);
    }}
  }} else {{
    showDashboard();
  }}
}}

// Patch renderDashboard to show results
const _origRenderDashboard = renderDashboard;
renderDashboard = function() {{
  _origRenderDashboard();

  // Overlay results badges
  if (!RESULTS || !RESULTS.summary) return;
  const s = RESULTS.summary;

  // Add summary banner
  const dashboard = document.getElementById('dashboard-view');
  const statsDiv = dashboard.querySelector('.stats');
  if (statsDiv && s.total_races > 0) {{
    const banner = document.createElement('div');
    banner.style.cssText = 'background:linear-gradient(90deg,rgba(16,185,129,.1),rgba(59,130,246,.1));border:1px solid var(--green);border-radius:12px;padding:14px 20px;margin-bottom:14px;display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:10px;text-align:center';
    banner.innerHTML = `
      <div><div style="font-size:.68rem;color:var(--green);font-weight:700;text-transform:uppercase">RESULTS IN</div></div>
      <div><div style="font-size:1.3rem;font-weight:800;color:var(--green)">${{s.wins_correct}}/${{s.total_races}}</div><div style="font-size:.65rem;color:var(--dim)">Win Picks Correct</div></div>
      <div><div style="font-size:1.3rem;font-weight:800;color:var(--cyan)">${{s.top3_accuracy}}%</div><div style="font-size:.65rem;color:var(--dim)">Top 3 Accuracy</div></div>
      <div><div style="font-size:1.3rem;font-weight:800;color:var(--accent)">${{s.avg_winner_rank}}</div><div style="font-size:.65rem;color:var(--dim)">Avg Winner Rank</div></div>
      <div><div style="font-size:1.3rem;font-weight:800;color:var(--purple)">${{s.bets_hit}}/${{s.total_races}}</div><div style="font-size:.65rem;color:var(--dim)">Bets Hit</div></div>
    `;
    statsDiv.parentNode.insertBefore(banner, statsDiv);
  }}

  // Add result badges to race cards
  RACES.forEach(race => {{
    if (!race.results_available) return;
    const card = document.querySelector(`.race-card[onclick="showRace(${{race.race_number}})"]`);
    if (!card) return;

    // Mark winner in horse rows
    const rows = card.querySelectorAll('.rc-horse');
    rows.forEach(row => {{
      const nameEl = row.querySelector('.rc-name');
      if (!nameEl) return;
      const numMatch = nameEl.textContent.match(/#(\\d+)/);
      if (!numMatch) return;
      const num = parseInt(numMatch[1]);
      const horse = race.horses.find(h => h.number === num);
      if (!horse || !horse.actual_position) return;

      const posLabel = horse.actual_position <= 3
        ? `<span style="background:${{horse.actual_position===1?'var(--gold)':horse.actual_position===2?'var(--dim)':'var(--orange)'}};color:#000;font-size:.6rem;font-weight:800;padding:1px 5px;border-radius:8px;margin-left:4px">${{horse.actual_position}}${{['st','nd','rd'][horse.actual_position-1]||'th'}}</span>`
        : `<span style="color:var(--dim);font-size:.65rem;margin-left:4px">${{horse.actual_position}}th</span>`;
      nameEl.innerHTML += posLabel;
    }});

    // Add result badge to card header
    const header = card.querySelector('.rc-header');
    if (header && race.accuracy) {{
      const a = race.accuracy;
      const badge = document.createElement('span');
      badge.style.cssText = 'font-size:.65rem;font-weight:700;padding:2px 8px;border-radius:10px;margin-right:6px';
      if (a.win_correct) {{
        badge.style.background = 'rgba(16,185,129,.2)';
        badge.style.color = 'var(--green)';
        badge.textContent = 'WIN HIT';
      }} else if (a.top3_overlap >= 2) {{
        badge.style.background = 'rgba(59,130,246,.2)';
        badge.style.color = 'var(--blue)';
        badge.textContent = a.top3_overlap + '/3 TOP3';
      }} else if (a.winner_our_rank && a.winner_our_rank <= 5) {{
        badge.style.background = 'rgba(245,158,11,.2)';
        badge.style.color = 'var(--accent)';
        badge.textContent = 'Winner R' + a.winner_our_rank;
      }} else {{
        badge.style.background = 'rgba(239,68,68,.2)';
        badge.style.color = 'var(--red)';
        badge.textContent = 'MISS';
      }}
      const gradeEl = header.querySelector('.grade');
      if (gradeEl) header.insertBefore(badge, gradeEl);
    }}
  }});
}};

// Patch renderRaceDetail to show results
const _origRenderRaceDetail = renderRaceDetail;
renderRaceDetail = function(race) {{
  _origRenderRaceDetail(race);

  if (!race.results_available) return;

  const el = document.getElementById('race-detail-view');

  // Add results banner at top
  const a = race.accuracy || {{}};
  const winner = race.actual_results?.find(r => r.position === 1);
  const winnerName = winner ? `#${{winner.number}} ${{winner.name}}` : 'Unknown';

  const banner = document.createElement('div');
  banner.style.cssText = 'background:linear-gradient(90deg,rgba(16,185,129,.08),rgba(59,130,246,.08));border:1px solid var(--green);border-radius:10px;padding:14px 18px;margin-bottom:14px';
  banner.innerHTML = `
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px">
      <div>
        <div style="font-size:.7rem;color:var(--green);font-weight:700;text-transform:uppercase">Official Result</div>
        <div style="font-size:1.1rem;font-weight:800;margin-top:4px">Winner: ${{winnerName}} @ $${{race.actual_win_odds || '?'}}</div>
      </div>
      <div style="display:flex;gap:16px;text-align:center">
        <div>
          <div style="font-size:1.1rem;font-weight:800;color:${{a.win_correct ? 'var(--green)' : 'var(--red)'}}">${{a.win_correct ? 'YES' : 'NO'}}</div>
          <div style="font-size:.65rem;color:var(--dim)">Win Pick</div>
        </div>
        <div>
          <div style="font-size:1.1rem;font-weight:800;color:var(--cyan)">${{a.top3_overlap || 0}}/3</div>
          <div style="font-size:.65rem;color:var(--dim)">Top 3</div>
        </div>
        <div>
          <div style="font-size:1.1rem;font-weight:800;color:var(--accent)">${{a.winner_our_rank ? 'Rank ' + a.winner_our_rank : '?'}}</div>
          <div style="font-size:.65rem;color:var(--dim)">Our Rank</div>
        </div>
      </div>
    </div>
  `;

  const firstChild = el.querySelector('.detail-header');
  if (firstChild) firstChild.after(banner);

  // Add actual position column to horse table
  const table = el.querySelector('.horse-table');
  if (table) {{
    // Add header
    const headerRow = table.querySelector('tr');
    if (headerRow) {{
      const th = document.createElement('th');
      th.textContent = 'Finish';
      th.style.cssText = 'text-align:center;width:50px;background:rgba(16,185,129,.1);color:var(--green)';
      headerRow.insertBefore(th, headerRow.children[1]);
    }}

    // Add finish position to each row
    const rows = table.querySelectorAll('tr');
    rows.forEach((row, idx) => {{
      if (idx === 0) return; // skip header
      const cells = row.querySelectorAll('td');
      if (!cells.length) return;
      const rankCell = cells[0];
      const rank = parseInt(rankCell.textContent);

      const horse = race.horses.find(h => h.rank === rank);
      const pos = horse?.actual_position;

      const td = document.createElement('td');
      td.style.cssText = 'text-align:center;font-weight:800';
      if (pos === 1) {{
        td.innerHTML = '<span style="color:var(--gold)">1st</span>';
        row.style.background = 'rgba(251,191,36,.08)';
      }} else if (pos === 2) {{
        td.innerHTML = '<span style="color:var(--dim)">2nd</span>';
        row.style.background = 'rgba(148,163,184,.05)';
      }} else if (pos === 3) {{
        td.innerHTML = '<span style="color:var(--orange)">3rd</span>';
        row.style.background = 'rgba(249,115,22,.05)';
      }} else if (pos) {{
        td.innerHTML = `<span style="color:var(--dim)">${{pos}}th</span>`;
      }} else {{
        td.textContent = '-';
        td.style.color = 'var(--dim)';
      }}
      row.insertBefore(td, row.children[1]);
    }});
  }}
}};

// Apply results on load
applyResults();
'''

        html = html.replace('</script>', inject_js + '\n</script>')

        with open(dashboard_path, 'w') as f:
            f.write(html)

        logger.info(f"Dashboard updated with results: {dashboard_path}")


async def main():
    import argparse
    parser = argparse.ArgumentParser(description='Fetch HKJC race results')
    parser.add_argument('date', nargs='?', help='Race date (YYYY/MM/DD), defaults to today')
    args = parser.parse_args()

    race_date = args.date or datetime.now().strftime('%Y/%m/%d')

    fetcher = ResultsFetcher()
    await fetcher.update_dashboard_with_results(race_date)
    await fetcher.fetcher.close()


if __name__ == '__main__':
    asyncio.run(main())