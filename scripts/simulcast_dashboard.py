"""Build an HONEST Ireland/overseas simulcast dashboard from a scraped card.

Reads results/simulcast_{date}_{course}.json (from scripts.scrape_simulcast),
scores each runner with the GB-trained UK model WHERE the runner has GB form
history, and renders an HTML card. Runners/races without GB coverage are shown
display-only — NEVER given a fabricated model probability.

Key honesty rules (see memory project_overseas_db_location_and_ireland_dashboard):
  * UK DB is GB-only. A runner with no GB history gets no model prob.
  * sp_dec is the model's top feature. Pre-odds (HK odds not posted), the race
    is flagged PRE-ODDS and model output is labelled low-confidence.
  * No claim of validated edge — this is decision support, not a proven recommender.

Usage:
    python3 -m scripts.simulcast_dashboard 20260628 S2
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config
from ml.uk_features import FEATURE_COLS
from ml.uk_model import UKRacingModel

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"
TODAY = date.today()


# ---------------------------------------------------------------------------
# Per-runner feature construction from the GB DB
# ---------------------------------------------------------------------------
def _name_variants(name: str):
    name = (name or "").strip()
    base = re.sub(r"\s*\([A-Z]{2,3}\)\s*$", "", name).strip()
    out = []
    for v in (name, base):
        if v and v.upper() not in [x.upper() for x in out]:
            out.append(v)
    return out


def _horse_history(conn, name):
    """Return dict of prior-form stats from the GB DB, or None if no history."""
    for v in _name_variants(name):
        rows = conn.execute(
            """SELECT rr.finish_position fp, rr.official_rating orr, rr.age,
                      r.date AS d
               FROM uk_race_results rr JOIN uk_races r ON r.race_id = rr.race_id
               WHERE UPPER(rr.horse) = ? AND rr.finish_position IS NOT NULL
               ORDER BY r.date""", (v.upper(),)).fetchall()
        if rows:
            fps = [x[0] for x in rows]
            ors = [x[1] for x in rows if x[1] is not None]
            ages = [x[2] for x in rows if x[2] is not None]
            last_d = rows[-1][3]
            try:
                dsl = (TODAY - datetime.strptime(last_d[:10], "%Y-%m-%d").date()).days
            except Exception:
                dsl = -1
            # estimate current age: last known age + whole years elapsed
            age = None
            if ages:
                yrs = max(0, (TODAY.year - datetime.strptime(last_d[:10], "%Y-%m-%d").date().year))
                age = ages[-1] + yrs
            return {
                "runs_prior": len(fps),
                "wins_prior": sum(1 for f in fps if f == 1),
                "places_prior": sum(1 for f in fps if f <= 3),
                "avg_pos_last5": float(np.mean(fps[-5:])),
                "days_since_last_run": dsl,
                "or_rating": ors[-1] if ors else None,
                "age": age,
                "n_gb": len(fps),
            }
    return None


def _person_wr(conn, col, name, n=50):
    """Win rate over last `n` GB runs for a jockey/trainer name."""
    if not name:
        return 0.0, 0
    rows = conn.execute(
        f"""SELECT rr.finish_position fp FROM uk_race_results rr
            JOIN uk_races r ON r.race_id = rr.race_id
            WHERE UPPER(rr.{col}) = ? AND rr.finish_position IS NOT NULL
            ORDER BY r.date DESC LIMIT ?""", (name.upper(), n)).fetchall()
    if not rows:
        return 0.0, 0
    wins = sum(1 for x in rows if x[0] == 1)
    return wins / len(rows), len(rows)


def _parse_odds(s):
    try:
        v = float(str(s).strip())
        return v if v > 1.0 else None
    except Exception:
        return None


def build_race_features(conn, runners):
    """Return (DataFrame[FEATURE_COLS], meta list) aligned to live runners."""
    recs, meta = [], []
    for r in runners:
        if r.get("scratched"):
            continue
        hist = _horse_history(conn, r.get("horse", ""))
        jwr, jn = _person_wr(conn, "jockey", r.get("jockey", ""))
        twr, tn = _person_wr(conn, "trainer", r.get("trainer", ""))
        odds = _parse_odds(r.get("odds"))  # win odds (not on card view → None; live feed fills later)
        try:
            wt = float(r.get("weight") or 0) or np.nan
        except Exception:
            wt = np.nan
        try:
            draw = float(r.get("draw") or 0)
        except Exception:
            draw = 0.0
        h = hist or {}
        rec = {
            "sp_dec": odds if odds else np.nan,
            "or_rating": h.get("or_rating") if h.get("or_rating") is not None else np.nan,
            "weight_lbs": wt,
            "draw": draw,
            "age": h.get("age") if h.get("age") is not None else 4,
            "runs_prior": h.get("runs_prior", 0),
            "wins_prior": h.get("wins_prior", 0),
            "places_prior": h.get("places_prior", 0),
            "avg_pos_last5": h.get("avg_pos_last5", 99),
            "days_since_last_run": h.get("days_since_last_run", -1),
            "jockey_50r_wr": jwr,
            "trainer_50r_wr": twr,
        }
        recs.append(rec)
        meta.append({
            "number": r.get("number"), "horse": r.get("horse"),
            "jockey": r.get("jockey"), "trainer": r.get("trainer"),
            "draw": r.get("draw"), "weight": r.get("weight"),
            "card_form": r.get("last_form"), "card_win": r.get("win_pct"),
            "card_place": r.get("place_pct"),
            "odds": r.get("odds"),          # win odds from live feed (absent on card view)
            "gear": r.get("gear", ""),      # CP/V/TT/B/H equipment codes
            "has_gb": bool(hist), "n_gb": h.get("n_gb", 0),
            "or": h.get("or_rating"), "jwr": jwr, "jn": jn, "twr": twr, "tn": tn,
        })
    if not recs:
        return pd.DataFrame(columns=FEATURE_COLS), []
    df = pd.DataFrame(recs)
    # race-context features
    df["field_size"] = len(df)
    df["sp_rank"] = df["sp_dec"].rank(method="min")
    df["or_field_avg"] = df["or_rating"].mean()
    df["or_vs_field_avg"] = df["or_rating"] - df["or_field_avg"]
    # fillna to training defaults (mirror ml/uk_features.compute_features)
    df["avg_pos_last5"] = df["avg_pos_last5"].fillna(99)
    df["or_rating"] = df["or_rating"].fillna(df["or_field_avg"]).fillna(70)
    df["or_vs_field_avg"] = df["or_vs_field_avg"].fillna(0)
    df["draw"] = df["draw"].fillna(0)
    df["weight_lbs"] = df["weight_lbs"].fillna(df["weight_lbs"].median())
    return df[FEATURE_COLS], meta


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
def _esc(s):
    return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(date_str, course, race_blocks):
    css = """
    body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;background:#0f1115;color:#e6e6e6;margin:0;padding:16px}
    h1{font-size:20px} h2{font-size:16px;margin:18px 0 6px;border-bottom:1px solid #333;padding-bottom:4px}
    .note{background:#1b2330;border:1px solid #2c3a4f;border-radius:8px;padding:10px 12px;font-size:13px;color:#c9d4e3;margin:8px 0 16px}
    table{border-collapse:collapse;width:100%;font-size:13px;margin-bottom:8px}
    th,td{padding:5px 8px;text-align:left;border-bottom:1px solid #222}
    th{color:#8aa;font-weight:600;font-size:11px;text-transform:uppercase}
    .prob{font-weight:700;color:#7ee787}
    .badge{font-size:10px;padding:1px 6px;border-radius:6px;margin-left:4px}
    .gb{background:#13361f;color:#7ee787} .nogb{background:#3a2630;color:#ff9eae}
    .top{background:#161d27} .dim{color:#7a8699}
    .flag{color:#ffd479;font-size:12px}
    .preodds{color:#ff9eae;font-size:12px;font-weight:600}
    """
    parts = [f"<!doctype html><meta charset=utf-8><title>Simulcast {date_str} {course}</title>",
             f"<style>{css}</style>",
             f"<h1>🇮🇪 Simulcast Dashboard — {course} · {date_str}</h1>",
             "<div class=note><b>Honest-mode dashboard.</b> Probabilities come from the "
             "GB-trained UK model and appear <b>only for runners with GB form history</b> "
             "(badge <span class='badge gb'>GB</span>). Runners with "
             "<span class='badge nogb'>no GB</span> form are display-only — no model "
             "opinion. <b>SP odds are the model's top input</b>; until HK odds post, races "
             "are flagged <span class=preodds>PRE-ODDS</span> and model ranking is "
             "form-only / low-confidence. This is decision support, <b>not a validated-edge "
             "recommender</b>.</div>"]
    for b in race_blocks:
        parts.append(f"<h2>{_esc(b['title'])}" +
                     (" <span class=preodds>· PRE-ODDS (form-only)</span>" if b["pre_odds"] else "") +
                     f" <span class=dim>· GB coverage {b['cov']}/{b['n']}</span></h2>")
        if b["suggestion"]:
            parts.append(f"<div class=flag>💡 {_esc(b['suggestion'])}</div>")
        modcol = "Form rank" if b["pre_odds"] else "Model share"
        oddscol = "Gear" if b["pre_odds"] else "Odds"
        parts.append(f"<table><tr><th>#</th><th>Horse</th><th>{modcol}</th><th>Jockey</th>"
                     f"<th>Trainer</th><th>Dr</th><th>OR</th><th>GB form</th><th>{oddscol}</th></tr>")
        for row in b["rows"]:
            cls = " class=top" if row["top"] else ""
            if row["prob"] is None:
                prob = "<span class=dim>—</span>"
            elif b["pre_odds"]:
                prob = f"<span class=prob>#{row.get('form_rank','?')}</span>"
            else:
                prob = f"<span class=prob>{row['prob']*100:.0f}%</span>"
            badge = "<span class='badge gb'>GB</span>" if row["has_gb"] else "<span class='badge nogb'>no GB</span>"
            gbform = f"{row['n_gb']} runs" + (f", OR {row['or']}" if row['or'] else "") if row["has_gb"] else "<span class=dim>first GB run</span>"
            parts.append(
                f"<tr{cls}><td>{_esc(row['number'])}</td>"
                f"<td>{_esc(row['horse'])} {badge}</td>"
                f"<td>{prob}</td><td>{_esc(row['jockey'])}</td><td>{_esc(row['trainer'])}</td>"
                f"<td>{_esc(row['draw'])}</td><td>{_esc(row['or'] or '—')}</td>"
                f"<td class=dim>{gbform}</td>"
                f"<td>{_esc(row.get('gear') or '—') if b['pre_odds'] else _esc(row.get('odds') or '—')}</td></tr>")
        parts.append("</table>")
    parts.append(f"<p class=dim>Generated {datetime.now():%Y-%m-%d %H:%M} · "
                 "model: uk_racing_model.pkl (GB-trained) · ⚠️ gamble responsibly.</p>")
    return "\n".join(parts)


RACE_NAMES = {1:"R1 2yo Fillies Maiden",2:"R2 Rockingham Handicap",3:"R3 Dash Stakes",
              4:"R4 Derby Festival Handicap",5:"R5 International Stakes (Gp)",
              6:"R6 Celebration Stakes",7:"R7 IRISH DERBY ⭐",8:"R8 Maddenstown Handicap",
              9:"R9 Ragusa Handicap"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("date"); ap.add_argument("course")
    args = ap.parse_args()

    src = RESULTS_DIR / f"simulcast_{args.date}_{args.course}.json"
    if not src.exists():
        print(f"ERROR: {src} not found. Run scripts.scrape_simulcast first.", file=sys.stderr)
        return 2
    cards = json.loads(src.read_text())
    model = UKRacingModel.load()
    conn = sqlite3.connect(str(config.UK_DB_PATH)); conn.row_factory = sqlite3.Row

    blocks = []
    for race in cards["races"]:
        rno = race["race_no"]
        X, meta = build_race_features(conn, race["runners"])
        if X.empty:
            continue
        probs = model.predict_proba(X)
        n = len(meta); cov = sum(m["has_gb"] for m in meta)
        any_odds = any(_parse_odds(m["odds"]) for m in meta)
        # Raw per-runner P(win) is uncalibrated and (pre-odds, with sp_dec
        # missing) wildly inflated — normalise across GB-covered runners so the
        # numbers are a RELATIVE share within the field that sums to 100%.
        # The ORDER is a form-based signal; pre-odds the magnitudes are not
        # trustworthy (flagged in the UI).
        raw_sum = sum(float(probs[i]) for i, m in enumerate(meta) if m["has_gb"]) or 1.0
        rows = []
        for i, m in enumerate(meta):
            share = float(probs[i]) / raw_sum if m["has_gb"] else None
            rows.append({**m, "prob": share, "top": False})
        # rank covered runners by model share
        covered = sorted([r for r in rows if r["prob"] is not None],
                         key=lambda r: r["prob"], reverse=True)
        for rank, r in enumerate(covered, 1):
            r["form_rank"] = rank
        for r in covered[:1]:
            r["top"] = True
        rows = covered + [r for r in rows if r["prob"] is None]
        # honest suggestion only for OK-coverage races
        suggestion = ""
        if cov >= max(3, n * 0.5) and covered:
            t = covered[0]
            if any_odds:
                suggestion = (f"Model top GB-form pick: #{t['number']} {t['horse']} "
                              f"({t['prob']*100:.0f}% share). HK odds live — overlay vs price.")
            else:
                suggestion = (f"Form-only lean (PRE-ODDS, low confidence): #{t['number']} "
                              f"{t['horse']}. Re-run once HK odds post — the model leans "
                              f"heavily on price, so treat this as form-shortlist only.")
        elif cov < 3:
            suggestion = "Too few GB-form runners to offer a model view — display only."
        blocks.append({"title": RACE_NAMES.get(rno, f"R{rno}"), "rows": rows,
                       "n": n, "cov": cov, "pre_odds": not any_odds,
                       "suggestion": suggestion})

    html = render(args.date, args.course, blocks)
    out = RESULTS_DIR / f"simulcast_dashboard_{args.date}_{args.course}.html"
    out.write_text(html)
    print(f"Dashboard → {out}  ({len(blocks)} races)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
