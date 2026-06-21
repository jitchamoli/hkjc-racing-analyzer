"""Stop-loss / daily-cap bankroll guard — a stateless discipline layer.

The single biggest gap in the recommendation system (vs. standard bankroll
discipline) was session state: the optimizer sizes each bet from a bankroll
number but never knows how much you've already lost today, so there is no
daily-loss limit and no chase-prevention.

This module fills that gap. It recomputes the day's realised P&L from
`race_bet_ticket` (what was actually staked) + results EVERY call — no
persistent state file, so it can't drift. Realised P&L is taken from the
canonical replay scorer (`scripts.validate_hk._score_meeting_replay`), the same
one `validate_hk --replay` and `stake_ab` use, so the numbers match everywhere
(parity invariant).

No edge is assumed. With the project's no-demonstrated-edge posture the only
sound staking is small/flat; this guard simply enforces the loss limits and
caps that tip #5 calls for and stops betting once they're hit.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from typing import Optional, Tuple

import config


@dataclass
class DailyStatus:
    date: str
    staked: float             # total staked today (resolved + pending)
    staked_resolved: float    # staked on races that have results
    returned: float           # returns on resolved tickets
    realized_pnl: float       # returned - staked_resolved
    realized_pnl_pct: float   # realized_pnl / bankroll
    exposure_pct: float       # staked / bankroll
    n_bets: int               # number of bet lines today
    n_resolved_races: int
    bankroll: float
    halted: bool
    halt_reason: str          # "" when not halted

    def banner(self) -> str:
        flag = f"STOP - {self.halt_reason}" if self.halted else "OK"
        return (f"DAY {self.date}: staked ${self.staked:.0f} "
                f"({self.exposure_pct * 100:.1f}%) | "
                f"P&L ${self.realized_pnl:+.0f} ({self.realized_pnl_pct * 100:+.1f}%) | "
                f"{self.n_resolved_races} races resolved | {flag}")


def _open(conn: Optional[sqlite3.Connection]) -> Tuple[sqlite3.Connection, bool]:
    if conn is not None:
        return conn, False
    c = sqlite3.connect(str(config.DB_PATH))
    c.row_factory = sqlite3.Row
    return c, True


def _is_counted_label(label: str) -> bool:
    """Whether a bet line counts toward the guard's stake / P&L.

    The user plays QIN only (STRAIGHT QIN + the BOX QIN-4 PROTECTION leg) at a
    flat HK$10/combo; the TRIO/F4 boxes on the system ticket are never staked. A
    realized stop-loss scored on the FULL ticket therefore fired on losses the
    user never took and blanked every later HIGH race as NO SYSTEM BET. Scope the
    guard to the QIN pool the user actually plays. User decision 2026-06-21.

    Matches 'STRAIGHT QIN' and 'BOX QIN-*' (incl. 'BOX QIN-4 PROTECTION'); excludes
    TRIO, FIRST 4, WIN, PLACE, and QIN PLACE (QPL) — a different pool.
    """
    lab = (label or "").strip().upper()
    return lab == "STRAIGHT QIN" or lab.startswith("BOX QIN-")


def _ticket_total_stake(ticket_json: str, counted_only: bool = False) -> Tuple[float, int]:
    """Total $ stake + bet-line count across one ticket.

    Mirrors the stake formula in validate_hk._score_meeting_replay exactly
    (combos x unit_stake, $1 fallback, skip combos<=0 or no usable legs) so the
    pending+resolved total stays consistent with the replay scorer. The replay
    scorer keys legs by number-or-name, so a number-only leg (no name) must still
    count toward exposure — accept a leg with a name OR a number, only drop legs
    that carry neither identifier (audit 2026-06-09 #9). The parity is pinned by
    test_resolved_stake_matches_replay_scorer.

    `counted_only=True` restricts the total to the QIN pool the user actually
    plays (see _is_counted_label) — used by daily_status so the stop-loss /
    exposure reflect real exposure, not the unplayed TRIO/F4 boxes.
    """
    try:
        ticket = json.loads(ticket_json)
    except (json.JSONDecodeError, TypeError):
        return 0.0, 0
    total = 0.0
    n = 0
    for bet in ticket.get("bets", []) or []:
        if counted_only and not _is_counted_label(bet.get("label")):
            continue
        combos = int(bet.get("combos", 0) or 0)
        legs = [l for l in (bet.get("legs") or [])
                if l.get("name") or l.get("number") is not None]
        if combos <= 0 or not legs:
            continue
        unit = float(bet.get("unit_stake", 0) or 0) or 1.0
        total += combos * unit
        n += 1
    return total, n


def daily_status(date: str, bankroll: Optional[float] = None,
                 conn: Optional[sqlite3.Connection] = None) -> DailyStatus:
    """Compute the bankroll-guard status for `date` (YYYY/MM/DD)."""
    # Imported lazily — validate_hk pulls in a lot, and this avoids a cycle.
    from scripts.validate_hk import _score_meeting_replay

    bankroll = config.BANKROLL if bankroll is None else bankroll
    c, owned = _open(conn)
    try:
        rows = c.execute(
            "SELECT ticket_json FROM race_bet_ticket WHERE race_date=?", (date,)
        ).fetchall()
        total_staked = 0.0
        n_bets = 0
        for r in rows:
            tj = r["ticket_json"] if isinstance(r, sqlite3.Row) else r[0]
            s, n = _ticket_total_stake(tj, counted_only=True)
            total_staked += s
            n_bets += n

        replay = _score_meeting_replay(c, date)
        if replay is None:
            staked_resolved = returned = 0.0
            n_resolved = 0
        else:
            # Scope realized P&L to the QIN pool the user actually plays so a
            # stop-loss reflects real losses, not the unplayed TRIO/F4 boxes.
            labels = {k: v for k, v in replay["labels"].items()
                      if _is_counted_label(k)}
            staked_resolved = sum(v[0] for v in labels.values())
            returned = sum(v[1] for v in labels.values())
            n_resolved = replay["_races"]
    finally:
        if owned:
            c.close()

    realized_pnl = returned - staked_resolved
    loss_cap = bankroll * config.DAILY_LOSS_CAP_PCT
    expo_cap = bankroll * config.DAILY_EXPOSURE_CAP_PCT

    halted, reason = False, ""
    if realized_pnl <= -loss_cap:
        halted, reason = True, f"DAILY LOSS CAP (-{config.DAILY_LOSS_CAP_PCT * 100:.0f}%)"
    elif total_staked >= expo_cap:
        halted, reason = True, f"DAILY EXPOSURE CAP ({config.DAILY_EXPOSURE_CAP_PCT * 100:.0f}%)"

    return DailyStatus(
        date=date,
        staked=total_staked,
        staked_resolved=staked_resolved,
        returned=returned,
        realized_pnl=realized_pnl,
        realized_pnl_pct=(realized_pnl / bankroll) if bankroll else 0.0,
        exposure_pct=(total_staked / bankroll) if bankroll else 0.0,
        n_bets=n_bets,
        n_resolved_races=n_resolved,
        bankroll=bankroll,
        halted=halted,
        halt_reason=reason,
    )


def check_new_bet(proposed_stake: float, date: str,
                  bankroll: Optional[float] = None,
                  conn: Optional[sqlite3.Connection] = None) -> Tuple[bool, float, str]:
    """Decide whether a NEW bet of `proposed_stake` may be placed today.

    Returns (allowed, capped_stake, reason):
      - halted day            -> (False, 0.0, "STOP: ...")
      - over per-race cap      -> (True, capped, "capped to N%/race")
      - over remaining expo    -> (True, remaining, "capped to remaining exposure")
                                  or (False, 0.0, "STOP: ...") if nothing left
    """
    bankroll = config.BANKROLL if bankroll is None else bankroll
    st = daily_status(date, bankroll, conn)
    if st.halted:
        return False, 0.0, f"STOP: {st.halt_reason} reached"

    per_race_cap = bankroll * config.MAX_STAKE_PER_RACE_PCT
    expo_cap = bankroll * config.DAILY_EXPOSURE_CAP_PCT
    remaining = expo_cap - st.staked
    if remaining <= 0:
        return False, 0.0, f"STOP: daily exposure cap (${expo_cap:.0f}) reached"

    capped = min(proposed_stake, per_race_cap)
    reason = ""
    if capped < proposed_stake:
        reason = f"capped to {config.MAX_STAKE_PER_RACE_PCT * 100:.0f}%/race (${per_race_cap:.0f})"
    if capped > remaining:
        capped = remaining
        reason = f"capped to remaining daily exposure (${remaining:.0f})"
    return True, round(capped, 1), reason
