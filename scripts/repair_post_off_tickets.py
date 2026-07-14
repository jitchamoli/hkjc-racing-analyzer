#!/usr/bin/env python3
"""One-shot repair: restore the true PRE-OFF freeze of every corrupted bet ticket.

Until the 2026-07-12 fix, `finalize_tickets_and_stakes` decided "has this race
run?" from results alone (`results_available` / `actual_position`). But
`raceday live` generates BEFORE it fetches results, so for the whole post-off /
pre-results window a race that had ALREADY RUN still looked live — and its frozen
ticket was rebuilt on post-race odds. The mechanism is visible in the data:
2026/06/24 R4's ticket carries frozen_at 20:20:44, the exact timestamp of R5's
pre-off freeze — one `live` run before R5 rebuilt the already-run R4.

`race_bet_ticket` is the single source of truth for replay ROI, stake_ab and the
dashboard STRIKE badges, so this corruption flows straight into the only honest
measure the project has. It drifts BOTH ways: bets ADDED post-off get credited on
money never staked, bets LOST post-off erase real wagers.

Every rebuild archived the ticket it replaced into `superseded_tickets[]`, so the
true pre-off ticket is still on disk. For each race we take the LAST candidate
(archived or active) frozen at or before the scheduled post time (plus a small
grace — the scheduled time is nominal and a freeze seconds later is still pre-off)
and make it active again.

Two things this deliberately does NOT do:
  * It does not compare bets by LABEL alone. Post-race odds re-rank the field, so
    the commonest corruption keeps every label and swaps a LEG. Comparison is on
    the full (label, legs) signature.
  * It does not GUESS a post time. Races whose scheduled time survives nowhere
    (2026/06/24 R4-R9 — the results JSON was truncated by the drop-race bug, and
    HKJC no longer serves times for a finished meeting) are reported and marked
    `post_off_unverified`, NOT silently skipped and NOT repaired from an invented
    time.

Idempotent. Usage:
    python3 -m scripts.repair_post_off_tickets [--apply]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# The scheduled post time is nominal; a freeze a few seconds after it is still the
# last pre-off refresh (2026/06/27 R1 froze at 16:00:50 against a 16:00 post, and
# a strict cutoff rejected the only clean ticket on record). Corrupting rebuilds
# land many minutes later, so this window does not admit them.
POST_GRACE = timedelta(minutes=2)


def _post_times() -> dict:
    """(race_date, race_number) -> scheduled post time 'HH:MM' from results JSON."""
    times: dict = {}
    for path in (glob.glob("results/race_data_*_with_results.json")
                 + glob.glob("results/race_data_2*.json")):
        m = re.search(r"race_data_(\d{8})", path)
        if not m:
            continue
        d = f"{m.group(1)[:4]}/{m.group(1)[4:6]}/{m.group(1)[6:]}"
        try:
            with open(path) as f:
                j = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        races = j.get("races", j) if isinstance(j, dict) else j
        if not isinstance(races, list):
            continue
        for r in races:
            if isinstance(r, dict) and r.get("time"):
                times.setdefault((d, r.get("race_number")), r["time"])
    return times


def _parse(ts):
    try:
        return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        return None


def _signature(t: dict):
    """(label, sorted legs) per bet — the identity of a ticket.

    Comparing labels alone misses the commonest drift: post-race odds re-rank the
    field, so BOX TRIO-5 keeps its label and swaps a horse.
    """
    return sorted(
        (b.get("label"), tuple(sorted(l.get("number") for l in (b.get("legs") or []))))
        for b in (t.get("bets") or [])
    )


def _fmt(t: dict) -> str:
    return ", ".join(
        f"{lab}{list(legs)}" for lab, legs in _signature(t)
    ) or "(no bets)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="write the repair (default: dry run)")
    args = ap.parse_args()

    times = _post_times()
    conn = sqlite3.connect(str(config.DB_PATH))
    conn.row_factory = sqlite3.Row

    repaired = unverified = clean = 0
    for row in conn.execute(
            "SELECT race_date, race_number, ticket_json FROM race_bet_ticket "
            "ORDER BY race_date, race_number"):
        d, rno = row["race_date"], row["race_number"]
        try:
            ticket = json.loads(row["ticket_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        tstr = times.get((d, rno))
        if not tstr:
            # No scheduled time survives anywhere. Do NOT invent one — flag it so
            # the replay ledger can exclude the race instead of trusting a ticket
            # we cannot verify.
            unverified += 1
            print(f"  {d} R{rno}  UNVERIFIED — no post time on record; not repaired")
            if args.apply and not ticket.get("post_off_unverified"):
                ticket["post_off_unverified"] = True
                conn.execute(
                    "UPDATE race_bet_ticket SET ticket_json=? "
                    "WHERE race_date=? AND race_number=?",
                    (json.dumps(ticket), d, rno))
            continue

        try:
            post = datetime.strptime(f"{d} {tstr}", "%Y/%m/%d %H:%M")
        except ValueError:
            continue
        cutoff = post + POST_GRACE

        # The signal is `superseded_at` (when the CONTENT changed), NOT `frozen_at`.
        # _merge_ticket_pre_result archives the old ticket ONLY when the signature
        # changes, but refreshes frozen_at on EVERY regen — so a post-off frozen_at
        # with no post-off archive just means the rebuild produced an identical
        # ticket, and the active content is still exactly what was staked. Keying
        # on frozen_at would "repair" those by reverting them to a stale morning
        # ticket, destroying a legitimate pre-off odds refresh.
        #
        # The FIRST archive event after the off is the one that displaced the
        # ticket which was live when the race ran — so the entry archived by it is
        # what we actually staked.
        post_off_archives = sorted(
            (s for s in (ticket.get("superseded_tickets") or [])
             if (_parse(s.get("superseded_at")) or datetime.min) > cutoff),
            key=lambda s: s.get("superseded_at") or "",
        )
        if not post_off_archives:
            clean += 1
            continue  # content never changed after the off

        pre = post_off_archives[0]
        if _signature(pre) == _signature(ticket):
            clean += 1
            continue

        frozen = _parse(ticket.get("frozen_at"))
        print(f"  {d} R{rno}  off {post:%H:%M}  content changed post-off at "
              f"{pre.get('superseded_at')}")
        print(f"      staked (pre-off) : {_fmt(pre)}")
        print(f"      stored (post-off): {_fmt(ticket)}")

        if args.apply:
            fixed = dict(pre)
            fixed["superseded_tickets"] = ticket.get("superseded_tickets") or []
            fixed["repaired_from_post_off"] = {
                "at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "discarded": _fmt(ticket),
                "discarded_frozen_at": ticket.get("frozen_at"),
            }
            fixed.pop("post_off_unverified", None)
            conn.execute(
                "UPDATE race_bet_ticket SET ticket_json=? "
                "WHERE race_date=? AND race_number=?",
                (json.dumps(fixed), d, rno))
        repaired += 1

    if args.apply:
        conn.commit()
    conn.close()

    print()
    print(f"  already correct           : {clean}")
    print(f"  {'REPAIRED' if args.apply else 'to repair':<26}: {repaired}")
    print(f"  UNVERIFIED (quarantined)  : {unverified}")
    if unverified:
        print("    -> excluded from validate_hk --replay; ticket cannot be trusted")
    if not args.apply:
        print("\n  dry run — pass --apply to write.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
