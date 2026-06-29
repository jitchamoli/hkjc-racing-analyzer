"""Scrape HKJC overseas Simulcast RESULTS + dividends into an isolated DB.

Companion to scrape_simulcast.py (which grabs the pre-race cards). This pulls
the post-race finishing order, win odds and HK dividend payouts and upserts
them into a DEDICATED sqlite db.

IMPORTANT — this writes to tracking_data/simulcast_history.db, NOT uk_history.db.
The UK DB is GB-only and feeds the UK model's training set; HKJC-scraped Irish
results use a different name format and lack OR/RPR, so mixing them in would
corrupt training and leak. Kept fully separate. See memory:
project_overseas_db_location_and_ireland_dashboard.

Usage:
    python3 -m scripts.save_simulcast_result 20260628 S2 --races 5,6
    python3 -m scripts.save_simulcast_result 20260628 S2 --races 7
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "tracking_data" / "simulcast_history.db"

RESULT_URL = ("https://racing.hkjc.com/en-us/overseas/results"
              "?RaceDate={date}&Racecourse={course}&RaceNo={no}")

# Pull the finishing-order table, the dividend table and a meta blob from the
# rendered SPA. Result tables vary in column count between races, so we map by
# header text rather than fixed indices.
_EXTRACT_JS = r"""
() => {
  const clean = s => (s||'').replace(/\s+/g,' ').trim();
  const tables=[...document.querySelectorAll('table')];
  let results=[], dividends=[];
  for(const t of tables){
    const rows=[...t.querySelectorAll('tr')]
      .map(tr=>[...tr.querySelectorAll('th,td')].map(x=>clean(x.innerText)));
    if(!rows.length) continue;
    const h=(rows[0]||[]).map(x=>x.toLowerCase());
    const hj=h.join('|');
    // results table: has a 'Pla.' column and a horse column
    if(hj.includes('pla') && hj.includes('horse') && !results.length){
      const idx=n=>h.findIndex(x=>x.includes(n));
      const ci={
        pla: idx('pla'),
        no: h.findIndex(x=>x.includes('horse no')),
        horse: h.findIndex(x=>x==='horse' || (x.includes('horse') && !x.includes('no'))),
        jockey: idx('jockey'),
        trainer: idx('trainer'),
        odds: idx('win odds'),
        margin: idx('margin'),
      };
      for(const r of rows.slice(1)){
        const pos=r[ci.pla];
        if(!pos || !/^\d+$/.test(pos)) continue;
        results.push({
          pla: pos,
          no: ci.no>=0 ? r[ci.no] : '',
          horse: ci.horse>=0 ? r[ci.horse] : '',
          jockey: ci.jockey>=0 ? r[ci.jockey] : '',
          trainer: ci.trainer>=0 ? r[ci.trainer] : '',
          win_odds: ci.odds>=0 ? r[ci.odds] : '',
          margin: ci.margin>=0 ? r[ci.margin] : '',
        });
      }
    }
    // dividend table: HAS NO HEADER ROW — its first cell is a pool name
    // (Win/Place/Quinella/...). Rows are [pool, combination, dividend] but pool
    // is blank on continuation rows (e.g. the three Place dividends), so carry
    // it forward. Detect by the first row's first cell being a known pool.
    const POOLS=['win','place','quinella','forecast','tierce','trio','first 4',
                 'quartet','double','treble','six up'];
    const firstCell=((rows[0]||[])[0]||'').toLowerCase();
    if(POOLS.some(p=>firstCell.startsWith(p)) && !dividends.length){
      let pool='';
      for(const r of rows){
        let c=r.filter(x=>x!=='');
        if(c.length>=3){ pool=c[0]; dividends.push({pool, combo:c[1], div:c[2]}); }
        else if(c.length===2 && pool){ dividends.push({pool, combo:c[0], div:c[1]}); }
      }
    }
  }
  // meta: race title line + going + winning time from the text
  const lines=document.body.innerText.split('\n').map(clean).filter(Boolean);
  let meta={name:'', cls:'', dist:'', going:'', time:''};
  for(const ln of lines){
    if(!meta.name && /(Group|Listed|Handicap|Maiden|Stakes).*\d+M/i.test(ln)){ meta.name=ln; }
    if(/^Going:/i.test(ln)) meta.going=ln.replace(/^Going:\s*/i,'');
    if(/^Winning Time:/i.test(ln)) meta.time=ln.replace(/^Winning Time:\s*/i,'');
  }
  return {results, dividends, meta};
}
"""


def _ensure_schema(con: sqlite3.Connection) -> None:
    con.executescript("""
    CREATE TABLE IF NOT EXISTS simulcast_meta (
        date TEXT, course TEXT, race_no INTEGER,
        race_name TEXT, going TEXT, win_time TEXT,
        PRIMARY KEY (date, course, race_no));
    CREATE TABLE IF NOT EXISTS simulcast_results (
        date TEXT, course TEXT, race_no INTEGER,
        finish_pos INTEGER, horse_no TEXT, horse TEXT,
        jockey TEXT, trainer TEXT, win_odds REAL, margin TEXT,
        PRIMARY KEY (date, course, race_no, horse_no));
    CREATE TABLE IF NOT EXISTS simulcast_dividends (
        date TEXT, course TEXT, race_no INTEGER,
        pool TEXT, combination TEXT, dividend REAL,
        PRIMARY KEY (date, course, race_no, pool, combination));
    """)


def _to_float(s):
    try:
        return float(str(s).replace(",", ""))
    except (ValueError, TypeError):
        return None


def save_race(con, date, course, rno, data) -> int:
    res = data.get("results", [])
    divs = data.get("dividends", [])
    meta = data.get("meta", {})
    con.execute("INSERT OR REPLACE INTO simulcast_meta VALUES (?,?,?,?,?,?)",
                (date, course, rno, meta.get("name", ""),
                 meta.get("going", ""), meta.get("time", "")))
    for r in res:
        con.execute("INSERT OR REPLACE INTO simulcast_results VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (date, course, rno, int(r["pla"]), r["no"], r["horse"],
                     r["jockey"], r["trainer"], _to_float(r["win_odds"]), r["margin"]))
    for d in divs:
        con.execute("INSERT OR REPLACE INTO simulcast_dividends VALUES (?,?,?,?,?,?)",
                    (date, course, rno, d["pool"], d["combo"], _to_float(d["div"])))
    con.commit()
    return len(res)


def scrape_results(date, course, races):
    from playwright.sync_api import sync_playwright

    out = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"))
        for rno in races:
            url = RESULT_URL.format(date=date, course=course, no=rno)
            page.goto(url, wait_until="networkidle", timeout=45000)
            try:
                page.wait_for_selector("table tr td", timeout=15000)
            except Exception:
                pass
            time.sleep(1.0)
            out[rno] = page.evaluate(_EXTRACT_JS)
        browser.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date", help="YYYYMMDD")
    ap.add_argument("course", help="Simulcast course code, e.g. S2")
    ap.add_argument("--races", default="1-9", help="e.g. 5,6 or 1-9")
    args = ap.parse_args()

    spec = args.races
    if "-" in spec:
        a, b = spec.split("-"); races = list(range(int(a), int(b) + 1))
    else:
        races = [int(x) for x in spec.split(",")]

    print(f"Scraping HKJC simulcast RESULTS {args.date} {args.course} races {races}")
    scraped = scrape_results(args.date, args.course, races)

    DB_PATH.parent.mkdir(exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    _ensure_schema(con)
    total = 0
    for rno in races:
        data = scraped.get(rno, {})
        res = data.get("results", [])
        if not res:
            print(f"  R{rno}: no finishers parsed (race may not be official yet) — skipped")
            continue
        n = save_race(con, args.date, args.course, rno, data)
        win = next((r for r in res if r["pla"] == "1"), None)
        wtxt = f"  1st: #{win['no']} {win['horse']} @ {win['win_odds']}" if win else ""
        print(f"  R{rno}: saved {n} finishers, {len(data.get('dividends', []))} dividends.{wtxt}")
        total += n
    con.close()
    print(f"Saved → {DB_PATH}  ({total} finisher rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
