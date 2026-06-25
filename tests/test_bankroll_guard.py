"""Bankroll guard (stop-loss / daily cap) — betting/bankroll_guard.py.

Verifies the stateful stop-loss logic + that the stake formula stays in parity
with the canonical replay scorer. _score_meeting_replay is monkeypatched for
the P&L half so these tests don't need the full prediction_log/dividends
fixtures (the scorer has its own tests).
"""
import json
import sqlite3

import scripts.validate_hk as vh
from betting import bankroll_guard as bg


def _bet(combos, unit, names, pool="QUINELLA", label="STRAIGHT QIN"):
    # Default to a QIN-pool bet — daily_status only counts QIN labels the user
    # actually plays (STRAIGHT QIN / BOX QIN-*); see _is_counted_label.
    return {"combos": combos, "unit_stake": unit, "pool": pool,
            "label": label, "legs": [{"name": n} for n in names]}


def _ticket(bets):
    return {"bets": bets}


def _db(tickets):
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.execute("CREATE TABLE race_bet_ticket "
              "(race_date TEXT, race_number INTEGER, ticket_json TEXT, logged_at TEXT)")
    for rno, tj in tickets:
        c.execute("INSERT INTO race_bet_ticket VALUES (?,?,?,?)",
                  ("2026/06/03", rno, json.dumps(tj), ""))
    c.commit()
    return c


# --- pure stake helper ---

def test_ticket_total_stake():
    s, n = bg._ticket_total_stake(json.dumps(_ticket([
        _bet(10, 1.0, ["A", "B", "C"]),   # 10
        _bet(5, 2.0, ["A", "B"]),         # 10
    ])))
    assert s == 20.0 and n == 2


def test_ticket_total_stake_skips_invalid():
    s, n = bg._ticket_total_stake(json.dumps(_ticket([
        {"combos": 0, "legs": [{"name": "A"}]},      # combos<=0 -> skip
        {"combos": 3, "legs": [{"name": None}]},     # no named legs -> skip
    ])))
    assert (s, n) == (0.0, 0)
    assert bg._ticket_total_stake("not json") == (0.0, 0)


def test_ticket_total_stake_counts_number_only_legs():
    # audit 2026-06-09 #9: a leg with a number but no name still counts toward
    # exposure (the replay scorer keys by number-or-name); only legs with neither
    # identifier are dropped.
    s, n = bg._ticket_total_stake(json.dumps(_ticket([
        {"combos": 4, "unit_stake": 1.0, "legs": [{"number": 5}, {"number": 7}]},
    ])))
    assert (s, n) == (4.0, 1)
    # legs with neither name nor number are still skipped
    s2, n2 = bg._ticket_total_stake(json.dumps(_ticket([
        {"combos": 3, "unit_stake": 1.0, "legs": [{"foo": 1}]},
    ])))
    assert (s2, n2) == (0.0, 0)


def test_ticket_total_stake_unit_fallback():
    # missing unit_stake falls back to $1/combo
    s, n = bg._ticket_total_stake(json.dumps(_ticket([
        {"combos": 6, "legs": [{"name": "A"}, {"name": "B"}]},
    ])))
    assert s == 6.0 and n == 1


# --- daily_status ---

def test_not_halted_under_caps(monkeypatch):
    monkeypatch.setattr(vh, "_score_meeting_replay",
                        lambda conn, date: {"_races": 2, "labels": {"STRAIGHT QIN": [20.0, 15.0, 1]}})
    conn = _db([(1, _ticket([_bet(10, 1.0, ["A", "B", "C"])])),
                (2, _ticket([_bet(10, 1.0, ["A", "B", "C"])]))])
    st = bg.daily_status("2026/06/03", bankroll=1000, conn=conn)
    assert st.staked == 20.0
    assert st.realized_pnl == -5.0      # 15 returned - 20 staked
    assert not st.halted


def test_loss_cap_halts(monkeypatch):
    # realised loss 100 on bankroll 1000; loss cap 8% = 80 -> HALT
    monkeypatch.setattr(vh, "_score_meeting_replay",
                        lambda conn, date: {"_races": 3, "labels": {"STRAIGHT QIN": [100.0, 0.0, 0]}})
    conn = _db([(1, _ticket([_bet(100, 1.0, ["A", "B", "C"])]))])
    st = bg.daily_status("2026/06/03", bankroll=1000, conn=conn)
    assert st.halted and "LOSS" in st.halt_reason


def test_exposure_cap_halts(monkeypatch):
    # staked 150 on bankroll 1000; exposure cap 10% = 100 -> HALT (nothing resolved)
    monkeypatch.setattr(vh, "_score_meeting_replay", lambda conn, date: None)
    conn = _db([(1, _ticket([_bet(150, 1.0, ["A", "B", "C"])]))])
    st = bg.daily_status("2026/06/03", bankroll=1000, conn=conn)
    assert st.halted and "EXPOSURE" in st.halt_reason


def test_empty_day(monkeypatch):
    monkeypatch.setattr(vh, "_score_meeting_replay", lambda conn, date: None)
    st = bg.daily_status("2026/06/03", bankroll=1000, conn=_db([]))
    assert st.staked == 0.0 and st.realized_pnl == 0.0 and not st.halted


# --- check_new_bet ---

def test_check_new_bet_caps_per_race(monkeypatch):
    monkeypatch.setattr(vh, "_score_meeting_replay", lambda conn, date: None)
    allowed, capped, reason = bg.check_new_bet(500, "2026/06/03", bankroll=1000, conn=_db([]))
    assert allowed and capped == 30.0   # 3% of 1000, capped down from 500


def test_check_new_bet_blocked_when_halted(monkeypatch):
    monkeypatch.setattr(vh, "_score_meeting_replay",
                        lambda conn, date: {"_races": 1, "labels": {"STRAIGHT QIN": [200.0, 0.0, 0]}})
    conn = _db([(1, _ticket([_bet(200, 1.0, ["A", "B"])]))])
    allowed, capped, reason = bg.check_new_bet(10, "2026/06/03", bankroll=1000, conn=conn)
    assert not allowed and capped == 0.0


def test_resolved_stake_matches_replay_scorer(monkeypatch):
    # Parity: _ticket_total_stake on a fully-resolved meeting must equal the
    # staked sum the replay scorer reports (both use combos x unit_stake).
    labels = {"STRAIGHT QIN": [40.0, 0.0, 0]}   # scorer-reported staked = 40
    monkeypatch.setattr(vh, "_score_meeting_replay",
                        lambda conn, date: {"_races": 1, "labels": labels})
    conn = _db([(1, _ticket([_bet(10, 2.0, ["A", "B", "C"]),    # 20
                             _bet(20, 1.0, ["A", "B"])]))])      # 20  -> total 40
    st = bg.daily_status("2026/06/03", bankroll=1000, conn=conn)
    scorer_staked = sum(v[0] for v in labels.values())
    assert st.staked == scorer_staked == 40.0


# --- QIN-only scoping (user decision 2026-06-21) ---

def test_is_counted_label_qin_only():
    assert bg._is_counted_label("STRAIGHT QIN")
    assert bg._is_counted_label("BOX QIN-3")
    assert bg._is_counted_label("BOX QIN-4 PROTECTION")
    for excl in ("BOX TRIO-5", "F4 BOX-5", "BOX TRIO-6", "F4 SAVER",
                 "WIN", "PLACE", "EDGE PLACE", "QIN PLACE BANKER", "", None):
        assert not bg._is_counted_label(excl), excl


def test_daily_status_excludes_non_qin_losses(monkeypatch):
    # A big TRIO/F4 loss must NOT halt the day — the user only plays QIN.
    # Realized: QIN +5 (profit), TRIO -900 (ignored). Loss cap 8% of 1000 = 80.
    monkeypatch.setattr(vh, "_score_meeting_replay", lambda conn, date: {
        "_races": 3,
        "labels": {"STRAIGHT QIN": [30.0, 35.0, 1],   # +5
                   "BOX TRIO-5": [900.0, 0.0, 0]},     # -900, must be excluded
    })
    conn = _db([(1, _ticket([
        _bet(3, 10.0, ["A", "B"], label="STRAIGHT QIN"),         # 30, counted
        _bet(10, 90.0, ["A", "B", "C"], pool="TRIO", label="BOX TRIO-5"),  # not counted
    ]))])
    st = bg.daily_status("2026/06/03", bankroll=1000, conn=conn)
    assert st.realized_pnl == 5.0     # QIN only, TRIO loss ignored
    assert st.staked == 30.0          # QIN stake only
    assert not st.halted
