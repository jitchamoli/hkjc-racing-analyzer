"""One-off: re-score 2026/06/07 with the current (retrained) model and diff the
top-3 ranking vs the stored old-model predictions, scored against real results.

Uses the production path (scripts.backtest.predict_race_production +
assign_bet_recommendations) on DB rows (cleaned data, current model+calibrator).
Odds basis = DB final SP (the 06/07 live predictions were made near post, so
odds are ~SP); this isolates model+data effects at the decision level.
"""
import json, os, sqlite3, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scripts.backtest import predict_race_production, group_by_race
from core.recommender import assign_bet_recommendations

DATE = "2026/06/07"
JSON = "results/race_data_20260607_with_results.json"


def main():
    old = {}
    data = json.load(open(JSON))
    races = data["races"] if isinstance(data, dict) and "races" in data else data
    for r in races:
        rn = r.get("race_number")
        hs = sorted(r["horses"], key=lambda h: h.get("rank", 99))
        old[rn] = {
            "top3": [h.get("number") for h in hs[:3]],
            "rank1": (hs[0].get("number"), hs[0].get("name")) if hs else None,
            "actual": [(h.get("number")) for h in sorted(
                (x for x in r["horses"] if (x.get("actual_position") or 0) in (1, 2)),
                key=lambda x: x["actual_position"])],
        }

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = [dict(x) for x in conn.execute(
        "SELECT * FROM race_results WHERE race_date=? AND finish_position>0 "
        "ORDER BY race_number", (DATE,)).fetchall()]
    by_race = group_by_race(rows)

    print(f"{'R':>3} | {'old top-3':14} {'new top-3':14} | {'1-2':7} | old new")
    print("-" * 60)
    flips = 0
    old_hits = new_hits = 0
    for (d, rn), hrows in sorted(by_race.items(), key=lambda kv: kv[0][1]):
        preds = predict_race_production(hrows, conn)
        if not preds:
            continue
        analyses = [p[3] for p in preds]
        dist = int(hrows[0].get("distance") or 1200)
        assign_bet_recommendations(analyses, field_size=len(analyses), distance=dist)
        ranked = sorted(analyses, key=lambda a: getattr(a, "rank", 99))
        new_top3 = [a.horse.number for a in ranked[:3]]
        o = old.get(rn, {})
        actual12 = o.get("actual", [])
        ohit = len(actual12) == 2 and all(n in o["top3"] for n in actual12)
        nhit = len(actual12) == 2 and all(n in new_top3 for n in actual12)
        old_hits += ohit; new_hits += nhit
        changed = new_top3 != o.get("top3")
        flips += changed
        print(f"R{rn:>2} | {str(o.get('top3','')):14} {str(new_top3):14} | "
              f"{str(actual12):7} | {'HIT' if ohit else '—':3} {'HIT' if nhit else '—':3}"
              f"{'   <-- changed' if changed else ''}")
    conn.close()
    print("-" * 60)
    print(f"top-3 changed in {flips}/{len(by_race)} races | "
          f"QIN-3 hits  OLD={old_hits}  NEW={new_hits}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
