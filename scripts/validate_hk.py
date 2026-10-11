"""HK forward-validation log — the ONLY un-burned evidence we can generate.

WHY THIS EXISTS
---------------
The 668-race backtest holdout is BURNED: dozens of ship/kill decisions have
been made against it, so its ROI figures are upper bounds shaped by selection,
not honest out-of-sample edge. And a severe rule-engine date-leak (fixed
2026-05-22) inflated those figures further. Leak-free, the exotic boxes show
NEGATIVE ROI on that holdout — but we can't fully trust the negative either,
because the holdout is burned and the dividends are heavy-tailed.

The only way to know the truth is to FREEZE a cutoff and score every FUTURE
meeting exactly once, never tuning on it. This script does that.

HOW IT WORKS
------------
- It scores the ACTUAL LIVE predictions stored in prediction_log (predicted_rank
  per horse at race time) — NOT a re-run of the current model. That's what you
  would really have bet.
- A frozen cutoff date lives in tracking_data/hk_forward_cutoff.txt. Meetings
  STRICTLY AFTER the cutoff are the clean forward sample. On first run the cutoff
  defaults to the latest meeting already in prediction_log (so all prior, burned
  data is excluded).
- Per-meeting results are appended to tracking_data/hk_forward_validation.csv
  (idempotent — a date already logged is skipped). The CSV is the permanent record.

STRATEGIES SCORED (real HKJC dividends, per-$1 basis):
  straight QIN  — our top-2 == actual top-2          (stake 1)
  BOX-Q4        — actual top-2 ⊂ our top-4            (stake 6)
  TRIO-5        — actual top-3 ⊂ our top-5            (stake 10)
  TRIO-6        — actual top-3 ⊂ our top-6            (stake 20)
  F4 BOX-5      — actual top-4 ⊂ our top-5            (stake 5)

Usage:
  python3 -m scripts.validate_hk                 # log new clean meetings + report
  python3 -m scripts.validate_hk --report        # cumulative report only (no scan)
  python3 -m scripts.validate_hk --include-burned # ALSO show the pre-cutoff meetings,
                                                  #   clearly labelled (NOT written to CSV)
  python3 -m scripts.validate_hk --set-cutoff 2026/05/20
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import config
from betting.dividends import hkjc_declared_to_unit
from betting.dead_heat import winning_combo_entries

CUTOFF_RE = re.compile(r'^\d{4}/\d{2}/\d{2}$')

CUTOFF_FILE = Path(config.DB_PATH).parent / "hk_forward_cutoff.txt"
LOG_FILE = Path(config.DB_PATH).parent / "hk_forward_validation.csv"
REPLAY_LOG_FILE = Path(config.DB_PATH).parent / "hk_forward_validation_replay.csv"

# (label, leg_n, box_n, n_combos, pool)  — leg/box define capture; pool the dividend
STRATS = [
    ("straight_QIN", 2, 2, 1, "QUINELLA"),
    ("BOX_Q4",       2, 4, 6, "QUINELLA"),
    ("TRIO_5",       3, 5, 10, "TRIO"),
    ("TRIO_6",       3, 6, 20, "TRIO"),
    ("F4_BOX_5",     4, 5, 5,  "FIRST 4"),
]
CSV_COLS = ["race_date", "races"] + [f"{s[0]}_{k}" for s in STRATS for k in ("stake", "ret", "hits")]


# The dead-heat enumeration now lives in betting/dead_heat.py so stake_ab and
# the tracker scripts share ONE implementation (code review 2026-08-03 found
# five re-implementations, of which stake_ab's had silently diverged and was
# scoring genuine dead-heat winners as misses). Alias kept: this name is used
# throughout this module and by tests.
#
# Historical note preserved: both scorers here used to key finishers by
# position (dict) and bail out on a tie — _score_meeting `continue`d past the
# whole race (survivorship on a POST-race property, silently deleting a real
# HK$37 winner from the forward log), while _score_meeting_replay let the tie
# overwrite a finisher and then booked the stake as a guaranteed loss.
_winning_combo_entries = winning_combo_entries


def _combo_div(conn, d, rno, pool, entries: list[dict]) -> float | None:
    """Dividend per HK$1 for one winning combination (sorted-number key)."""
    nums = sorted(int(e["number"]) for e in entries)
    key = ",".join(str(n) for n in nums)
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool=? AND combination=?", (d, rno, pool, key)).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _score_box(conn, d, rno, pool, our_ids: set, combos: list[list[dict]],
               key_fn) -> tuple[bool, float | None]:
    """(hit, total return per HK$1 staked per combo held).

    Sums EVERY winning combination our box contains — on a dead heat a box can
    legitimately hold more than one, and HKJC pays each. Returns (True, None)
    when a winning combo has no dividend row (can't price → caller skips).
    """
    won = [c for c in combos if {key_fn(e) for e in c} <= our_ids]
    if not won:
        return False, 0.0
    total = 0.0
    for c in won:
        div = _combo_div(conn, d, rno, pool, c)
        if div is None:
            return True, None
        total += div
    return True, total


def _quin_div(conn, d, rno, a, b):
    lo, hi = sorted([int(a), int(b)])
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool='QUINELLA' AND combination=?", (d, rno, f"{lo},{hi}")).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _exotic_div(conn, d, rno, pool, combination):
    # combination is the winning permutation/set as stored in race_dividends:
    # TRIO + FIRST 4 = sorted horse-numbers comma-joined ("1,3,7", "1,3,7,8");
    # TIERCE + QUARTET = finish-order comma-joined ("7,8,1", "7,8,1,3").
    # Without this filter, fetchone() picks whichever row SQLite returns first
    # — fine today (one row per race/pool) but the May-22 dividend-basis bug
    # was the same family of silent-fetch mistake. Filter explicitly.
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool=? AND combination=?", (d, rno, pool, combination)).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _win_div(conn, d, rno, horse_number):
    """Per-$1 WIN dividend for a specific winning horse number.

    Added 2026-05-25 (GPT-5.5 Finding #4): prediction_log.actual_odds is
    populated 0 for all rows, so the prior WIN-from-actual_odds path
    silently dropped every WIN replay. race_dividends.pool='WIN' has
    4,611 rows — query directly.
    """
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool='WIN' AND combination=?",
        (d, rno, str(int(horse_number))),
    ).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _place_div(conn, d, rno, horse_number):
    """Per-$1 PLACE dividend for a specific placed horse number.

    pool='PLACE' has one row per placed horse per race (3 rows when 8+
    runners, 2 rows when 4-7 runners, 1 when ≤4). Returns None if the
    horse did not place.
    """
    row = conn.execute(
        "SELECT dividend FROM race_dividends WHERE race_date=? AND race_number=? "
        "AND pool='PLACE' AND combination=?",
        (d, rno, str(int(horse_number))),
    ).fetchone()
    return hkjc_declared_to_unit(row[0]) if row else None


def _load_cutoff(conn) -> str:
    if CUTOFF_FILE.exists():
        cutoff = CUTOFF_FILE.read_text().strip()
        # An empty/whitespace/corrupted cutoff would silently make every
        # row pass `race_date > ''`, contaminating un-burned evidence with
        # all burned data. Refuse to proceed.
        if not CUTOFF_RE.match(cutoff):
            raise SystemExit(
                f"refusing to run: {CUTOFF_FILE} contains {cutoff!r}, "
                f"not a YYYY/MM/DD date. Fix the file or re-init with "
                f"`python3 -m scripts.validate_hk --set-cutoff YYYY/MM/DD`."
            )
        return cutoff
    row = conn.execute("SELECT MAX(race_date) FROM prediction_log WHERE results_filled=1").fetchone()
    cutoff = row[0] or "2026/05/20"
    _atomic_write_text(CUTOFF_FILE, cutoff)
    print(f"[init] froze forward-validation cutoff at {cutoff} "
          f"(meetings AFTER this date are the clean sample)")
    return cutoff


def _meeting_dates(conn, after: str | None, only_burned_through: str | None = None):
    q = "SELECT DISTINCT race_date FROM prediction_log WHERE results_filled=1"
    params = []
    if after:
        q += " AND race_date > ?"; params.append(after)
    if only_burned_through:
        q += " AND race_date <= ?"; params.append(only_burned_through)
    q += " ORDER BY race_date"
    return [r[0] for r in conn.execute(q, params).fetchall()]


def _score_meeting(conn, date) -> dict:
    """Return {strat: [stake, ret, hits]} + race count for one meeting."""
    rows = conn.execute(
        "SELECT race_number, horse_number, horse_name, predicted_rank, actual_position "
        "FROM prediction_log WHERE race_date=? AND results_filled=1", (date,)).fetchall()
    races = defaultdict(list)
    for r in rows:
        races[r["race_number"]].append(dict(r))
    acc = {s[0]: [0.0, 0.0, 0] for s in STRATS}
    n_races = 0
    for rno, horses in races.items():
        ranked = sorted([h for h in horses if h["predicted_rank"]], key=lambda h: h["predicted_rank"])
        if len(ranked) < 2:
            continue  # need a pair to price even the smallest (straight QIN) bet
        # Key picks/finishers by horse NUMBER (saddle), not name — robust to the
        # name casing/brand-suffix drift in the HKJC parser gotchas, and
        # consistent with _score_meeting_replay (audit 2026-06-09 #1).
        max_box = min(6, len(ranked))
        our = {n: set(str(h["horse_number"]).strip() for h in ranked[:n])
               for n in range(2, max_box + 1)}
        # position -> [entries]; a dead heat legitimately puts >1 horse on a
        # position, so this must be a LIST, never a dict keyed by position.
        pos_to_entries: dict[int, list[dict]] = {}
        for h in horses:
            p = h["actual_position"]
            if p in (1, 2, 3, 4) and h["horse_number"] is not None:
                pos_to_entries.setdefault(p, []).append({"number": h["horse_number"]})
        if sum(len(v) for v in pos_to_entries.values()) < 2:
            continue  # need two finishers to price even a straight QIN
        n_races += 1

        def _key(e): return str(e["number"]).strip()
        combos_by_leg = {k: _winning_combo_entries(pos_to_entries, k) for k in (2, 3, 4)}

        for label, leg, box, ncombo, pool in STRATS:
            if box > len(ranked):
                continue  # not enough ranked horses to fill this box this race
            combos = combos_by_leg[leg]
            if combos:
                hit, div = _score_box(conn, date, rno, pool, our[box], combos, _key)
            else:
                # Fewer known finishers than the bet needs — it cannot have won,
                # but it WAS staked. Book the loss; don't drop the bet.
                hit, div = False, 0.0
            if hit and div is None:
                continue  # can't price a win — skip (rare; ~99.6% coverage)
            acc[label][0] += ncombo
            acc[label][1] += div if hit else 0.0
            acc[label][2] += 1 if hit else 0
    acc["_races"] = n_races
    return acc


def _score_meeting_replay(conn, date) -> dict | None:
    """Score the ACTUAL frozen bet_ticket for each race on `date`.

    Returns None when no race_bet_ticket rows exist (forward-only — meetings
    before 2026-05-24 weren't logged). Returns {label: [stake_combos, ret,
    hits]} keyed by the user-visible ticket label (BOX QIN-4, BOX TRIO-5,
    F4 BOX-5, ...) — so the report names match what was actually staked.

    Solves Finding #2 2026-05-24: the synthetic _score_meeting reported
    F4 BOX-5 hits on R8/R11 that were never in the ticket (gated out).
    """
    ticket_rows = conn.execute(
        "SELECT race_number, ticket_json FROM race_bet_ticket WHERE race_date=?",
        (date,),
    ).fetchall()
    if not ticket_rows:
        return None

    horse_rows = conn.execute(
        "SELECT race_number, horse_number, horse_name, actual_position "
        "FROM prediction_log WHERE race_date=? AND results_filled=1",
        (date,),
    ).fetchall()
    by_race: dict[int, dict[str, dict]] = defaultdict(dict)
    for r in horse_rows:
        by_race[r["race_number"]][r["horse_name"]] = {
            "number": r["horse_number"],
            "actual_position": r["actual_position"],
        }

    acc: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0, 0])
    metrics: dict[str, dict] = {}
    n_races = 0
    skipped_unverified: list[int] = []
    for row in ticket_rows:
        rno = row["race_number"]
        try:
            ticket = json.loads(row["ticket_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        # Quarantined by scripts/repair_post_off_tickets: the ticket was rebuilt
        # AFTER the race ran and no pre-off freeze survives, so we cannot say what
        # was really staked. Scoring it would credit or charge invented bets — the
        # exact contamination this ledger exists to avoid. Drop it, loudly.
        if ticket.get("post_off_unverified"):
            skipped_unverified.append(rno)
            continue

        horses = by_race.get(rno, {})
        # position -> {"name", "number"} so hit logic can key by NUMBER
        # (robust to name casing/spelling drift between ticket legs and
        # prediction_log rows — audit 2026-06-08 #3) and the dividend lookups
        # below can read the saddle number directly.
        # position -> [entries]. MUST be a list: keying by position let a dead
        # heat silently overwrite a finisher, which left the position below it
        # vacant and auto-missed every FIRST 4 ticket bet — while still booking
        # the stake (audit 2026-07-12).
        pos_to_entries: dict[int, list[dict]] = {}
        for name, h in horses.items():
            p = h["actual_position"]
            if p in (1, 2, 3, 4):
                pos_to_entries.setdefault(p, []).append(
                    {"name": name, "number": h["number"]})
        finishers = {p: v[0] for p, v in pos_to_entries.items()}  # legacy single-finisher view
        if sum(len(v) for v in pos_to_entries.values()) < 2:
            continue
        n_races += 1
        # Normalized-name -> number map for the legacy (no-leg-number) fallback.
        name_to_num = {(n or "").strip().upper(): h["number"]
                       for n, h in horses.items()}

        for bet in ticket.get("bets", []) or []:
            label = bet.get("label", "?")
            # Paper bets are reference/tracking only — never staked, so they
            # must not appear in the actual-staked replay ROI. They are booked
            # under a prefixed label instead of dropped, so they stay VISIBLE
            # in the report without polluting the real-money rows.
            if bet.get("paper"):
                label = PAPER_LABEL_PREFIX + label
            pool = bet.get("pool", "")
            combos = int(bet.get("combos", 0) or 0)
            unit_stake = float(bet.get("unit_stake", 0) or 0)
            legs = bet.get("legs") or []
            if not legs or combos <= 0:
                continue
            # Fall back to $1 unit if the ticket didn't size the stake — keeps
            # ROI% meaningful even on legacy tickets without unit_stake.
            stake_per_combo = unit_stake if unit_stake > 0 else 1.0

            # Hit key: prefer horse_number; fall back to a normalized name for
            # legacy tickets/rows missing numbers. Kept in sync with
            # stake_ab._score_key (not imported — bankroll_guard imports this
            # module, so a back-import would risk a cycle).
            legs_have = all(l.get("number") is not None for l in legs)
            fin_have = all(e["number"] is not None
                           for v in pos_to_entries.values() for e in v)
            if legs_have and fin_have:
                def _key_leg(l): return str(l.get("number")).strip()
                def _key_ent(e): return str(e["number"]).strip()
            else:
                def _key_leg(l): return (l.get("name") or "").strip().upper()
                def _key_ent(e): return (e["name"] or "").strip().upper()

            leg_ids = {_key_leg(l) for l in legs if _key_leg(l)}
            if not leg_ids:
                continue
            # Every horse that placed top-3 — dead heats included.
            placed_top3 = {_key_ent(e) for p in (1, 2, 3)
                           for e in pos_to_entries.get(p, [])}

            hit = False
            # QPL ticket structure: legs[0] is the pivot, legs[1:] are bankers
            # (server-side build_bet_ticket). Each (pivot, banker_i) pair is
            # one combo and pays independently when both horses place top-3.
            qpl_hit_pairs: list[tuple[dict, dict]] = []   # (pivot_leg, banker_leg)

            # `div_per_unit` is the payout PER HK$1 of stake on a winning combo.
            # Multiply by stake_per_combo to get actual dollar return per combo.
            div_per_unit: float | None = None

            _BOX_LEGS = {"QUINELLA": 2, "TRIO": 3, "FIRST 4": 4}
            if pool in _BOX_LEGS:
                win_combos = _winning_combo_entries(pos_to_entries, _BOX_LEGS[pool])
                if win_combos:
                    hit, div_per_unit = _score_box(
                        conn, date, rno, pool, leg_ids, win_combos, _key_ent)
                else:
                    # Fewer known finishers than the bet needs — it cannot have
                    # won, but it WAS staked. Book the loss, don't drop the bet.
                    hit, div_per_unit = False, 0.0
            elif pool == "WIN":
                hit = any(_key_ent(e) in leg_ids
                          for e in pos_to_entries.get(1, []))
            elif pool == "PLACE":
                hit = any(_key_ent(e) in leg_ids
                          for p in (1, 2, 3) for e in pos_to_entries.get(p, []))
            elif pool == "QIN PLACE":
                if _key_leg(legs[0]) in placed_top3:
                    for bk_leg in legs[1:]:
                        if _key_leg(bk_leg) in placed_top3:
                            qpl_hit_pairs.append((legs[0], bk_leg))
                hit = len(qpl_hit_pairs) > 0

            if hit and pool == "WIN":
                # 2026-05-25 (Finding #4): price from race_dividends pool='WIN'
                # for the winning horse number (prediction_log.actual_odds is
                # populated 0 for all rows, so the old path silently dropped
                # every WIN replay). A dead heat for 1st declares a WIN dividend
                # per winner; sum the ones our legs actually held.
                win_div_sum = 0.0
                for e in pos_to_entries.get(1, []):
                    if _key_ent(e) in leg_ids and e["number"]:
                        d1 = _win_div(conn, date, rno, e["number"])
                        if d1:
                            win_div_sum += d1
                if win_div_sum:
                    div_per_unit = win_div_sum
            elif hit and pool == "PLACE":
                # PLACE pays once per placed leg horse (1-3 placed horses per
                # race depending on field size). Sum dividends across all leg
                # horses that actually placed — most PLACE bets have 1 leg
                # but the data structure supports multiple.
                placed_legs_div = 0.0
                placed_any = False
                for leg in legs:
                    if _key_leg(leg) in placed_top3:
                        ln_num = leg.get("number")
                        if ln_num is None:
                            ln_num = name_to_num.get(_key_leg(leg))
                        if ln_num:
                            d2 = _place_div(conn, date, rno, ln_num)
                            if d2:
                                placed_legs_div += d2
                                placed_any = True
                if placed_any:
                    div_per_unit = placed_legs_div
            elif hit and pool == "QIN PLACE":
                # 2026-05-26 (GPT-5.5 R7 Finding #1): pool was missing from
                # replay, so QPL BANKER bets were unscored. race_dividends
                # pool='QUINELLA PLACE' has the per-pair dividends. Sum
                # across hit pairs — multi-pair hits compound the payout.
                qpl_div_sum = 0.0
                priced_any = False
                for piv_leg, bk_leg in qpl_hit_pairs:
                    piv_num = piv_leg.get("number") or name_to_num.get(_key_leg(piv_leg))
                    bk_num = bk_leg.get("number") or name_to_num.get(_key_leg(bk_leg))
                    if piv_num and bk_num:
                        lo, hi = sorted([int(piv_num), int(bk_num)])
                        row = conn.execute(
                            "SELECT dividend FROM race_dividends WHERE race_date=? "
                            "AND race_number=? AND pool='QUINELLA PLACE' "
                            "AND combination=?",
                            (date, rno, f"{lo},{hi}"),
                        ).fetchone()
                        if row:
                            qpl_div_sum += hkjc_declared_to_unit(row[0])
                            priced_any = True
                if priced_any:
                    div_per_unit = qpl_div_sum

            _book_bet(acc, label, combos, stake_per_combo, hit, div_per_unit, metrics)

    if skipped_unverified:
        # Never drop races silently — a hidden exclusion reads as "we covered
        # everything" when we did not.
        print(f"  ⚠ {date}: skipped race(s) {sorted(skipped_unverified)} — ticket was "
              f"rebuilt post-off with no pre-off freeze on record (unverifiable)")

    return {"_races": n_races, "labels": dict(acc), "metrics": metrics}


def _read_csv() -> list[dict]:
    if not LOG_FILE.exists():
        return []
    with open(LOG_FILE) as f:
        return list(csv.DictReader(f))


# Paper bets are booked under a prefixed label so they appear in the replay
# report WITHOUT being mixed into real-money ROI. Before 2026-09-06 paper bets
# were simply `continue`d, which was right while EDGE PLACE (tracked separately
# via edge_log.csv) was the only one — but demoting BOX TRIO-6 to paper would
# then have deleted the card's largest line from the only report that measures
# it. The prefix is non-alphanumeric so a paper row can never be misread as
# staked money.
PAPER_LABEL_PREFIX = "~paper "


def _book_bet(acc, label, combos, stake_per_combo, hit, div_per_unit,
              metrics=None) -> None:
    """Book one staked bet into `acc[label]` = [stake, return, hits, unpriced].

    The unpriced counter exists because this used to `continue` — dropping the
    stake AND the return — whenever a HIT could not be priced. A miss always
    prices (return 0.0), so only winners could take that path: the ledger lost
    winners and kept losers, biasing every replay ROI downward. Re-scoring
    2026/07/12 alone moved BOX TRIO-5 from -62.0% to -37.6%.

    An unpriced win is a known win of unknown size, so its stake and its hit
    are real and must be booked; only the return is unknown, and 0.0 makes the
    reported ROI an explicit LOWER BOUND rather than a silent deletion. The
    count is surfaced in the report so the floor is visible.

    Box bets: only the one winning combo pays, not every combo in the box.
    Stake covers all `combos`; return is sized to the single winning combo.
    """
    row = acc.setdefault(label, [0.0, 0.0, 0, 0])   # works for dict + defaultdict
    if len(row) == 3:                               # legacy 3-field row
        row.append(0)
    row[0] += combos * stake_per_combo
    if metrics is not None:
        m = metrics.setdefault(label, {"tickets": 0, "profitable": 0})
        m["tickets"] += 1
        if hit and div_per_unit and div_per_unit > combos:
            m["profitable"] += 1
    if hit:
        row[2] += 1
        if div_per_unit:
            row[1] += stake_per_combo * div_per_unit
        else:
            row[3] += 1


def _read_replay_csv() -> list[dict]:
    if not REPLAY_LOG_FILE.exists():
        return []
    with open(REPLAY_LOG_FILE) as f:
        return list(csv.DictReader(f))


def _write_replay_row(date, replay):
    """Upsert one meeting's replay row, keeping the file in date order.

    UPSERT, not append: dividends arrive via a separate backfill
    (`scripts.backfill_dividends`), so a meeting scored before its dividends
    landed was permanently frozen as a total loss. 2026/07/12 sat in this log
    as 0 hits / $0 across all seven pools while the DB held 175 dividend rows
    for that date. Re-scoring every run is what makes the ledger self-healing.
    """
    rows = [r for r in _read_replay_csv() if r.get("race_date") != date]
    labels = replay["labels"]
    newrow = {"race_date": date, "races": replay["_races"],
              "ticket_json": json.dumps(labels),
              "metrics_json": json.dumps(replay.get("metrics", {}))}
    rows.append(newrow)
    rows.sort(key=lambda r: r.get("race_date", ""))
    tmp = REPLAY_LOG_FILE.with_suffix(".csv.tmp")
    cols = ["race_date", "races", "ticket_json", "metrics_json"]
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    tmp.replace(REPLAY_LOG_FILE)


def _report_replay(rows: list[dict], title: str):
    print(f"\n{'='*68}\n  {title}\n{'='*68}")
    if not rows:
        print("  (no meetings with frozen tickets yet — replay-mode is forward-only)")
        return
    dates = sorted(r["race_date"] for r in rows)
    print(f"  meetings: {len(dates)}  ({dates[0]} .. {dates[-1]})")
    gate = "" if len(dates) >= 6 else "   [WARN] N<6 — DIRECTIONAL ONLY"
    print(f"  {'bet label':<23}{'hits/tickets':>13}{'Hit%':>8} {'95% CI':>13}"
          f"{'Profit%':>9}{'stake':>9}{'P&L':>11}{'ROI':>9}{gate}")
    print("  " + "-" * 106)
    totals: dict[str, list] = defaultdict(lambda: [0.0, 0.0, 0, 0])
    counts = defaultdict(lambda: [0, 0])
    unknown = set()
    for r in rows:
        try:
            labels = json.loads(r.get("ticket_json", "{}"))
        except json.JSONDecodeError:
            continue
        metrics = json.loads(r.get("metrics_json") or "{}")
        for lbl, vals in labels.items():
            # Rows written before 2026-08-03 have no `unpriced` field.
            stake, ret, hits = vals[0], vals[1], vals[2]
            unpriced = vals[3] if len(vals) > 3 else 0
            totals[lbl][0] += float(stake)
            totals[lbl][1] += float(ret)
            totals[lbl][2] += int(hits)
            totals[lbl][3] += int(unpriced)
            if lbl in metrics:
                counts[lbl][0] += int(metrics[lbl]["tickets"])
                counts[lbl][1] += int(metrics[lbl]["profitable"])
            else:
                unknown.add(lbl)
    total_unpriced = 0
    for lbl in sorted(totals):
        stake, ret, hits, unpriced = totals[lbl]
        total_unpriced += unpriced
        if stake <= 0:
            continue
        roi = 100 * (ret - stake) / stake
        flag = f"  [{unpriced} unpriced win(s)]" if unpriced else ""
        tickets, profitable = counts[lbl]
        if lbl in unknown or not tickets:
            hit_count, hit_pct, interval, profit_pct = str(hits) + '/unknown', '-', '-', '-'
        else:
            from betting.tier_stats import wilson_ci_95
            low, high = wilson_ci_95(hits, tickets)
            hit_count = f'{hits}/{tickets}'
            hit_pct = f'{100 * hits / tickets:.1f}%'
            interval = f'{100 * low:.1f}-{100 * high:.1f}%'
            profit_pct = f'{100 * profitable / tickets:.1f}%'
        print(f"  {lbl:<23}{hit_count:>13}{hit_pct:>8} {interval:>13}"
              f"{profit_pct:>9}{stake:>9.0f}{ret-stake:>+11.1f}{roi:>+9.1f}%{flag}")
    print("\n  REPLAY: scored from the actual frozen bet_ticket (not synthetic).")
    print("  This is what was really recommended/staked, post-gate filtering.")
    print("  Hit% = winning selections, including unpriced wins; Profit% = return > entire ticket cost.")
    print("  95% CI is a ticket-level Wilson interval; correlated races/meetings may widen it.")
    if unknown:
        print("  Legacy denominators unknown: re-run --replay to rebuild ticket metrics.")
    if total_unpriced:
        # Never let a floor read as a point estimate.
        print(f"  ⚠ {total_unpriced} winning bet(s) had NO dividend row and were "
              f"booked at $0 return —\n    the ROI above is a LOWER BOUND. Run "
              f"`python3 -m scripts.backfill_dividends --force` then re-run.")


def _atomic_write_text(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _append_csv(date, acc):
    # Atomic read-modify-rewrite (NOT append): a torn append would corrupt the
    # only un-burned evidence file, and a non-atomic write races backups/concurrent
    # runs. Rebuild the full table and os.replace it in.
    rows = _read_csv()
    newrow = {"race_date": date, "races": acc["_races"]}
    for s in STRATS:
        newrow[f"{s[0]}_stake"] = acc[s[0]][0]
        newrow[f"{s[0]}_ret"] = round(acc[s[0]][1], 2)
        newrow[f"{s[0]}_hits"] = acc[s[0]][2]
    rows.append(newrow)
    tmp = LOG_FILE.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_COLS})
    tmp.replace(LOG_FILE)


def _report(logged: list[dict], title: str):
    print(f"\n{'='*68}\n  {title}\n{'='*68}")
    if not logged:
        print("  (no meetings logged yet — run after the next HK meeting completes)")
        return
    dates = sorted(r["race_date"] for r in logged)
    n_m = len(dates)
    print(f"  meetings: {n_m}  ({dates[0]} .. {dates[-1]})")
    gate = "" if n_m >= 6 else "   [WARN] N<6 — DIRECTIONAL ONLY (not a ship/kill signal yet)"
    print(f"  {'strategy':<14}{'hits':>6}{'stake':>9}{'P&L':>11}{'ROI':>9}{gate}")
    print("  " + "-" * 50)
    for label, *_ in [(s[0],) for s in STRATS]:
        stake = sum(float(r[f"{label}_stake"]) for r in logged)
        ret = sum(float(r[f"{label}_ret"]) for r in logged)
        hits = sum(int(r[f"{label}_hits"]) for r in logged)
        if stake <= 0:
            continue
        roi = 100 * (ret - stake) / stake
        print(f"  {label:<14}{hits:>6}{stake:>9.0f}{ret-stake:>+11.1f}{roi:>+9.1f}%")
    print("\n  NOTE: real HKJC dividends, per-$1 basis. This is the un-burned forward")
    print("  sample — never used for tuning. Trust it over the burned backtest.")
    if "SYNTHETIC" in title:
        print("\n  ⚠ SYNTHETIC = pre-gate UPPER BOUND. Scores each strategy on every")
        print("    race with enough ranked horses to fill its box (>=2);")
        print("    production skips LOW/SKIP tier and field<8. Real-money ROI is what the")
        print("    REPLAY report shows (frozen race_bet_ticket). Pass --replay to see it.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", action="store_true", help="cumulative report only")
    p.add_argument("--include-burned", action="store_true",
                   help="also show pre-cutoff meetings (labelled, NOT written to CSV)")
    p.add_argument("--set-cutoff", metavar="YYYY/MM/DD", help="set the frozen cutoff date")
    p.add_argument("--replay", action="store_true",
                   help="ALSO score the actual frozen bet_ticket (Finding #2 fix). "
                        "Forward-only — meetings before 2026-05-24 have no ticket.")
    args = p.parse_args()

    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    if args.set_cutoff:
        new_cutoff = args.set_cutoff.strip()
        if not CUTOFF_RE.match(new_cutoff):
            raise SystemExit(
                f"--set-cutoff expects YYYY/MM/DD, got {new_cutoff!r}"
            )
        _atomic_write_text(CUTOFF_FILE, new_cutoff)
        print(f"cutoff set to {new_cutoff}")
    cutoff = _load_cutoff(conn)
    print(f"forward cutoff: {cutoff}  (clean sample = meetings AFTER this)")

    logged = _read_csv()
    already = {r["race_date"] for r in logged}

    if not args.report:
        new_dates = [d for d in _meeting_dates(conn, after=cutoff) if d not in already]
        for d in new_dates:
            acc = _score_meeting(conn, d)
            _append_csv(d, acc)
            print(f"  logged {d}: {acc['_races']} races")
        if not new_dates:
            print("  no new clean meetings to log.")
        logged = _read_csv()

    if args.replay:
        # Re-score EVERY meeting, not just unseen ones. Dividends land via
        # a separate backfill, so skipping already-logged dates froze any
        # meeting scored before its dividends arrived as a permanent total
        # loss (2026/07/12: 0 hits / $0 logged, 175 dividend rows in DB).
        # _write_replay_row upserts, so this is idempotent.
        #
        # Runs for --report too (2026-10-04). It used to sit inside the
        # `not args.report` branch, so `--replay --report` printed the CSV
        # as-is — the very staleness this loop exists to prevent, on the
        # flag people reach for when they want the cumulative number.
        # Caught restoring the emptied 2026/09/16 R7 ticket: the totals
        # came back byte-identical until --report was dropped.
        for d in _meeting_dates(conn, after=cutoff):
            replay = _score_meeting_replay(conn, d)
            if replay is None:
                continue  # no frozen ticket — pre-fix meeting
            _write_replay_row(d, replay)
            print(f"  [replay] scored {d}: {replay['_races']} races")

    _report(logged, "HK FORWARD VALIDATION (clean, un-burned) — SYNTHETIC strategies")

    if args.replay:
        _report_replay(_read_replay_csv(),
                       "HK FORWARD VALIDATION — REPLAY (actual staked bet_tickets)")

    if args.include_burned:
        burned = []
        for d in _meeting_dates(conn, after=None, only_burned_through=cutoff):
            acc = _score_meeting(conn, d)
            row = {"race_date": d}
            for s in STRATS:
                row[f"{s[0]}_stake"], row[f"{s[0]}_ret"], row[f"{s[0]}_hits"] = acc[s[0]]
            burned.append(row)
        _report(burned, "PROVISIONAL — pre-cutoff meetings (BURNED window, directional only)")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
