"""Scrape HKJC overseas Simulcast (World Pool) race cards into structured JSON.

HKJC only simulcasts marquee international FEATURE DAYS (Irish Derby, Eclipse,
July Cup ...). The card site is a Next.js SPA, so we render with Playwright and
read the runner table from the DOM. Betting-grade fields only — no OR/RPR/IDs.

Usage:
    python3 -m scripts.scrape_simulcast 20260628 S2      # Irish Derby Day
    python3 -m scripts.scrape_simulcast 20260628 S2 --races 1-9

Output: results/simulcast_{date}_{course}.json  (one entry per race).

This is isolated infrastructure — it touches NO HK code and writes only to
results/. See memory: project_overseas_db_location_and_ireland_dashboard.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

CARD_URL = ("https://racing.hkjc.com/en-us/overseas/race-card"
            "?RaceDate={date}&Racecourse={course}&RaceNo={no}")

# JS run inside the page to pull the runner table + race meta. Column order on
# the HKJC simulcast card: No | (silk) | Horse (CTY) | Wt | Jockey | Draw |
# Trainer | Runs (W-P-x) | Last5Finishes | Win% | Plc% | Gear.
# NOTE: the last column is GEAR codes (CP/V/TT/B/H), NOT win odds — this card
# view carries no prices. HK win odds live on the betting feed, fetched
# separately when needed.
_EXTRACT_JS = r"""
() => {
  const clean = s => (s||'').replace(/\s+/g,' ').trim();
  // race name + meta from the active race tab line
  const lines = document.body.innerText.split('\n').map(clean).filter(Boolean);
  // largest table = the runner table
  const tables=[...document.querySelectorAll('table')];
  let best=null,n=0; for(const t of tables){const r=t.querySelectorAll('tr').length; if(r>n){n=r;best=t;}}
  let runners=[];
  if(best){
    runners = [...best.querySelectorAll('tr')].map(tr=>{
      const c=[...tr.querySelectorAll('td')].map(x=>clean(x.innerText)).filter(x=>x!=='');
      return c;
    }).filter(c=>c.length>=6 && /^\d+$/.test(c[0]));
  }
  return {url: location.href, runners};
}
"""


def _parse_runner(cells):
    """Map a cell list to a runner dict. Tolerant of SCR (scratched) rows."""
    out = {"number": cells[0], "raw": cells}
    rest = cells[1:]
    if any("SCR" in c.upper() for c in rest):
        out["horse"] = rest[0] if rest else ""
        out["scratched"] = True
        return out
    out["scratched"] = False
    # rest = [horse, wt, jockey, draw, trainer, runsWPx, last5, win%, plc%, gear]
    keys = ["horse", "weight", "jockey", "draw", "trainer",
            "runs", "last_form", "win_pct", "place_pct", "gear"]
    for i, k in enumerate(keys):
        out[k] = rest[i] if i < len(rest) else ""
    return out


def scrape(date: str, course: str, races):
    from playwright.sync_api import sync_playwright

    out = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"))
        for rno in races:
            url = CARD_URL.format(date=date, course=course, no=rno)
            page.goto(url, wait_until="networkidle", timeout=45000)
            # let the SPA render the table
            try:
                page.wait_for_selector("table tr td", timeout=15000)
            except Exception:
                pass
            time.sleep(1.0)
            data = page.evaluate(_EXTRACT_JS)
            runners = [_parse_runner(c) for c in data.get("runners", [])]
            live = [r for r in runners if not r["scratched"]]
            print(f"  R{rno}: {len(live)} runners "
                  f"({sum(r['scratched'] for r in runners)} SCR)", flush=True)
            out.append({"race_no": rno, "url": data.get("url", url),
                        "runners": runners})
        browser.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", help="YYYYMMDD")
    ap.add_argument("course", help="Simulcast course code, e.g. S2")
    ap.add_argument("--races", default="1-9", help="e.g. 1-9 or 1,3,7")
    args = ap.parse_args()

    spec = args.races
    if "-" in spec:
        a, b = spec.split("-"); races = list(range(int(a), int(b) + 1))
    else:
        races = [int(x) for x in spec.split(",")]

    print(f"Scraping HKJC simulcast {args.date} {args.course} races {races}")
    cards = scrape(args.date, args.course, races)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"simulcast_{args.date}_{args.course}.json"
    out_path.write_text(json.dumps(
        {"date": args.date, "course": course_or(args), "races": cards}, indent=2))
    nlive = sum(len([r for r in c["runners"] if not r["scratched"]]) for c in cards)
    print(f"Saved {len(cards)} races / {nlive} runners → {out_path}")
    return 0


def course_or(args):
    return args.course


if __name__ == "__main__":
    sys.exit(main())
