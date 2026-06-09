#!/usr/bin/env python3
"""Race day odds monitor.

Wakes up ~6 minutes before each race, fetches odds, detects market moves,
and regenerates the dashboard with auto-refresh enabled.
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.fetcher import HKJCDataFetcher
from core.analyzer import CombinedRaceAnalyzer
from data.database import HorseDatabase
from data.odds_tracker import OddsDatabase
from dashboard.template import generate_html
from dashboard.generator import (
    build_race_data, _merge_with_existing, finalize_tickets_and_stakes,
    preflight_dashboard_ml,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [MONITOR] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)

import config
POLL_INTERVAL = config.POLL_INTERVAL
PRE_RACE_WINDOW = config.PRE_RACE_WINDOW


def _look_up_latest_odds(odds_db, race_date, race_num, horse_num):
    """Peek the most recent win_odds for a horse from odds history.

    Returns 0 if no row is found or the query fails — callers must
    guard against a 0 return (no comparison if prev or curr is 0).
    """
    try:
        conn = odds_db._get_conn()
        try:
            row = conn.execute(
                "SELECT win_odds FROM odds_snapshots "
                "WHERE race_date = ? AND race_number = ? AND horse_number = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (race_date, race_num, horse_num),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
    except Exception:
        return 0


def parse_race_time(time_str: str, race_date: str) -> datetime:
    """Parse race time string like '13:00' into datetime."""
    try:
        date_part = race_date.replace('/', '-')
        return datetime.strptime(f"{date_part} {time_str}", "%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return None


async def fetch_race_times(fetcher: HKJCDataFetcher, race_date: str) -> list:
    """Get race times for all races on a given date.

    Returns list of (race_number, race_time_datetime) sorted by time.
    """
    race_times = []
    # Fetch all race cards concurrently, then process in order.
    # Detect the venue once and pass it to every card fetch. Detecting per-race
    # inside the gather fires 11 redundant HTTP calls and risks an inconsistent
    # venue mid-gather (→ wrong URL → empty HTML → truncated card).
    venue = fetcher.api.detect_venue(race_date)
    fetched = await asyncio.gather(
        *(fetcher.get_race_card(race_date, rn, venue=venue) for rn in range(1, 12)),
        return_exceptions=True,
    )
    # All 11 cards are already fetched concurrently above, so there is no cost
    # to inspecting every one. Do NOT break on an empty/error race — a single
    # transient miss on an early race must not drop the schedule for every later
    # race (which would blind the monitor for the rest of the meeting). Just skip
    # empties; non-existent races (meetings with <11) simply have no time.
    for race_no, race in enumerate(fetched, start=1):
        if isinstance(race, Exception) or not race or not race.horses:
            logger.warning(f"Race {race_no}: empty/error — skipping (later races still scheduled)")
            continue
        race_time = parse_race_time(race.time, race_date)
        if race_time:
            race_times.append((race_no, race_time))
            logger.info(f"Race {race_no}: {race.time} ({len(race.horses)} runners)")
        else:
            logger.warning(f"Race {race_no}: no time available")
    return race_times


async def snapshot_odds(fetcher: HKJCDataFetcher, race_date: str, race_no: int,
                        odds_db: OddsDatabase = None, venue: str = ""):
    """Fetch and store current odds for a race."""
    owns_db = odds_db is None
    odds_db = odds_db or OddsDatabase()
    try:
        # Pass the pre-detected venue so get_race_card skips its own detect_venue
        # HTTP call on every pre-race snapshot.
        race = await fetcher.get_race_card(race_date, race_no, venue=venue)
        if race and race.horses:
            horses = [
                {
                    'number': h.number,
                    'name': h.name,
                    'win_odds': h.win_odds,
                    'place_odds': h.place_odds,
                }
                for h in race.horses
                if h.number > 0 and h.win_odds > 0
            ]
            if horses:
                odds_db.record_snapshot(race_date, race_no, horses, source='monitor')
                logger.info(f"R{race_no}: Snapshotted odds for {len(horses)} horses")
                return len(horses)
    except Exception as e:
        logger.warning(f"R{race_no}: Failed to snapshot odds: {e}")
    return 0


async def regenerate_dashboard(race_date: str, fetcher: HKJCDataFetcher = None,
                               analyzer: CombinedRaceAnalyzer = None,
                               db: HorseDatabase = None, odds_db: OddsDatabase = None):
    """Regenerate dashboard with auto-refresh enabled."""
    owns_db = db is None
    try:
        fetcher = fetcher or HKJCDataFetcher()
        # fail_closed_ml=True matches the generator's contract: if the ML
        # pickle is missing/corrupt or fails the integrity check, refuse to
        # emit recommendations instead of falling through to rule-only.
        # Monitor used to default fail-open and could silently degrade
        # live recs to rule-only (GPT-5.5 R12 #3 2026-05-25).
        analyzer = analyzer or CombinedRaceAnalyzer(fail_closed_ml=True)
        db = db or HorseDatabase()
        odds_db = odds_db or OddsDatabase()

        results_dir = Path(__file__).parent.parent / 'results'
        results_dir.mkdir(exist_ok=True)

        all_races = []

        # Detect venue once (avoid 11 redundant detect_venue HTTP calls + the
        # mid-gather inconsistency risk), then fetch all cards concurrently;
        # analyze/build stay sequential (DB).
        detected_venue = fetcher.api.detect_venue(race_date)
        fetched = await asyncio.gather(
            *(fetcher.get_race_card(race_date, rn, venue=detected_venue) for rn in range(1, 12)),
            return_exceptions=True,
        )
        for race_no, race in enumerate(fetched, start=1):
            if isinstance(race, Exception) or not race or not race.horses:
                # Skip, do NOT break: all 11 are already fetched, so a transient
                # empty/error on an early race must not drop every later race from
                # the live dashboard (same rationale as fetch_race_times).
                logger.warning(f"R{race_no}: empty/error — skipping (later races still rendered)")
                continue
            analyses = analyzer.analyze_race(race)
            race_data = build_race_data(race, analyses, race_date, db, odds_db, analyzer=analyzer)
            all_races.append(race_data)

        # Restore frozen race-level + bet_ticket state from prior JSON
        # BEFORE recomputing tickets. Without this _merge_with_existing
        # step the monitor would re-derive the ticket from scratch every
        # 6-minute refresh — gate flips driven by transient odds drift
        # would silently replace the morning ticket with no archive in
        # superseded_tickets[]. The main generator (generate_predictions.py
        # line ~1227) calls this same merge step before finalize_tickets;
        # without it the monitor violated the "ticket is truth" invariant
        # documented in CLAUDE.md. Locked by
        # tests/test_audit_2026_05_29.py::test_monitor_merges_existing_before_finalize.
        date_str_local = race_date.replace('/', '')
        json_file = str(results_dir / f"race_data_{date_str_local}.json")
        all_races = _merge_with_existing(json_file, all_races, race_date=race_date)

        # Inject tier_stats so the monitor-rendered dashboard's tier
        # track-record trust panel populates, matching the main generator
        # (generator.py _run_generation does this before finalize). Pure
        # read-only display data — load_cached_stats() never mutates
        # (audit-2026-05-29 follow-up D). NOT replicated: log_predictions
        # (would corrupt predicted_rank if re-logged mid-race — the freeze
        # logic exists precisely to prevent that) and compute_alerts (the
        # sharp-move path already spawns a full generate_predictions
        # subprocess that fires alerts; recomputing here would double-fire).
        from betting.tier_stats import load_cached_stats
        tier_stats = load_cached_stats()
        for r in all_races:
            r['tier_stats'] = tier_stats

        # GPT-5.5 R13 #2 2026-05-25 — monitor uses the same stake +
        # bet_ticket pipeline as the main generator. Without this the
        # live monitor dashboard rendered horses with bet_recommendation
        # but no frozen ticket and no suggested stake → WA/detail panel
        # showed SKIP everywhere. Shared finalize fn keeps the contract
        # in one place; both surfaces emit identical ticket structure.
        # finalize_tickets_and_stakes' Layer 2 (_merge_ticket_pre_result)
        # now sees the prior ticket and archives it correctly on any
        # gate flip.
        finalize_tickets_and_stakes(all_races, race_date, analyzer)

        # Snapshot the live trust panels BEFORE closing the DB. Without these,
        # generate_html() defaults both to None and the monitor re-render wipes
        # the EDGE banner + 60-day accuracy footer on every pre-race refresh
        # (same class as the results_fetcher panel-wipe bug, commit 84603e1).
        try:
            accuracy_summary = db.get_accuracy_summary(days=60)
            if isinstance(accuracy_summary, dict) and 'message' in accuracy_summary:
                accuracy_summary = None
        except Exception:
            accuracy_summary = None

        if owns_db:
            db.conn.close()

        edge_stats = None
        try:
            es_path = results_dir.parent / 'tracking_data' / 'edge_stats.json'
            if es_path.exists():
                with open(es_path) as f:
                    edge_stats = json.load(f)
        except (json.JSONDecodeError, IOError, OSError):
            pass

        date_str = race_date.replace('/', '')
        json_data = json.dumps(all_races)
        json_data = json_data.replace('</script>', r'<\/script>').replace('</Script>', r'<\/Script>')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        venue = "Happy Valley" if any(r.get("venue", "").startswith("Happy") for r in all_races) else "Sha Tin"

        html = generate_html(json_data, now, race_date, venue, auto_refresh=True,
                             accuracy_summary=accuracy_summary, edge_stats=edge_stats)

        html_file = results_dir / f"racing_dashboard_{date_str}.html"
        latest_file = results_dir / "racing_dashboard.html"
        # Atomic write so a browser polling the file never sees a half-written
        # HTML during regeneration. os.replace is atomic on POSIX/Windows.
        # try/finally cleans up tmp on write failure so stale .tmp.{pid} files
        # don't accumulate in the results directory.
        for path in (html_file, latest_file):
            tmp = path.with_suffix(path.suffix + f'.tmp.{os.getpid()}')
            try:
                tmp.write_text(html)
                os.replace(tmp, path)
            except Exception:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise

        logger.info(f"Dashboard regenerated with {len(all_races)} races")
        return True

    except Exception as e:
        # Surface the failure to the caller instead of returning a silent
        # "success". The atomic writes above mean a crash leaves the prior good
        # dashboard in place; run_monitor logs prominently and continues to the
        # next race rather than aborting the whole meeting on one transient error.
        logger.error(f"Dashboard regeneration failed: {e}", exc_info=True)
        return False


async def run_monitor(race_date: str):
    """Main monitor loop."""
    logger.info(f"Starting race day monitor for {race_date}")
    logger.info(f"Pre-race window: {PRE_RACE_WINDOW}s, Poll interval: {POLL_INTERVAL}s")

    # ML integrity preflight ONCE at startup, mirroring the generator's main():
    # refuse to run a whole meeting against a missing/corrupt model or blend
    # calibrator. The model doesn't change mid-meeting, so one check suffices.
    try:
        preflight_dashboard_ml()
    except Exception as e:
        logger.error(
            "ML integrity preflight failed; refusing to start the monitor: "
            f"{type(e).__name__}: {e}"
        )
        raise SystemExit(2) from e

    fetcher = HKJCDataFetcher()
    # fail_closed_ml=True — see comment in regenerate_dashboard above.
    analyzer = CombinedRaceAnalyzer(fail_closed_ml=True)
    db = HorseDatabase()
    odds_db = OddsDatabase()

    try:
        # Step 1: Get race times
        race_times = await fetch_race_times(fetcher, race_date)
        if not race_times:
            logger.error("No race times found. Waiting 30 minutes then retrying...")
            await asyncio.sleep(1800)
            race_times = await fetch_race_times(fetcher, race_date)
            if not race_times:
                logger.error("Still no race times. Exiting.")
                return

        logger.info(f"Found {len(race_times)} races")

        # Detect the meeting venue once and reuse it for every snapshot (avoids a
        # redundant detect_venue HTTP call per pre-race window).
        meeting_venue = fetcher.api.detect_venue(race_date)

        # Step 2: Process each race
        for race_no, race_time in race_times:
            wake_time = race_time - timedelta(seconds=PRE_RACE_WINDOW)
            now = datetime.now()

            if now < wake_time:
                sleep_secs = (wake_time - now).total_seconds()
                logger.info(f"Sleeping {sleep_secs:.0f}s until R{race_no} window ({wake_time.strftime('%H:%M')})")
                await asyncio.sleep(sleep_secs)
            elif now > race_time + timedelta(minutes=5):
                logger.info(f"R{race_no} already started 5+ min ago, skipping")
                continue

            # Snapshot and regenerate
            logger.info(f"R{race_no}: Pre-race window active")
            await snapshot_odds(fetcher, race_date, race_no, odds_db=odds_db, venue=meeting_venue)

            # Sharp-move auto-rerun trigger (spec §7): if any top-2 pick moved
            # ≥ 25% (steam) or ≥ 40% (cold), force a full re-run.
            date_str = race_date.replace('/', '')
            race_data_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'results', f'race_data_{date_str}.json',
            )
            sharp_move = False
            if os.path.exists(race_data_path):
                try:
                    with open(race_data_path) as f:
                        prev = json.load(f)
                    prev_races = prev if isinstance(prev, list) else prev.get('races', [])
                    for race_prev in prev_races:
                        for h in race_prev.get('horses', [])[:2]:  # top-2 only
                            prev_odds = h.get('win_odds', 0)
                            curr_odds = _look_up_latest_odds(
                                odds_db, race_date,
                                race_prev['race_number'], h['number'],
                            )
                            if prev_odds and curr_odds:
                                pct = (curr_odds - prev_odds) / prev_odds
                                if pct <= -0.25 or pct >= 0.40:
                                    sharp_move = True
                                    logger.info(
                                        f"Sharp move on R{race_prev['race_number']} "
                                        f"#{h['number']}: {prev_odds} → {curr_odds} "
                                        f"({pct:+.1%}) — forcing re-run"
                                    )
                                    break
                        if sharp_move:
                            break
                except Exception as e:
                    logger.warning(f"Sharp-move detection failed: {e}")

            if sharp_move:
                # Full re-run in subprocess — re-fetches cards AND odds.
                # Use sys.executable + absolute path so this works regardless
                # of CWD or which python is on PATH.
                from pathlib import Path
                gen_script = Path(__file__).resolve().parent.parent / 'generate_predictions.py'
                subprocess.run(
                    [sys.executable, str(gen_script), race_date],
                    check=False, timeout=300,
                )
            else:
                ok = await regenerate_dashboard(race_date, fetcher=fetcher, analyzer=analyzer,
                                                db=db, odds_db=odds_db)
                if not ok:
                    logger.warning(
                        f"R{race_no}: dashboard regeneration FAILED — prior dashboard "
                        "left in place; continuing to next race window."
                    )

            # After every pre-race window, pull results for any finished
            # races so STRIKE/MISS badges on past races stay current. The
            # results_fetcher is idempotent — only writes data for races
            # that actually have results published.
            from pathlib import Path
            rf_script = Path(__file__).resolve().parent.parent / 'results_fetcher.py'
            try:
                subprocess.run(
                    [sys.executable, str(rf_script), race_date],
                    check=False, timeout=120,
                )
                logger.info(f"Results merge done after R{race_no} window")
            except subprocess.TimeoutExpired:
                logger.warning(f"Results merge timed out after R{race_no}")

        logger.info("All races processed. Monitor complete.")
    finally:
        # Close the Playwright browser — without this the chromium subprocess
        # lingers for the whole meeting and after the monitor exits (the
        # generator's main() already does this; the monitor never did).
        try:
            await fetcher.close()
        except Exception as e:
            logger.warning(f"fetcher.close() failed: {e}")
        db.conn.close()


def main():
    import re
    race_date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime('%Y/%m/%d')
    if not re.fullmatch(r'\d{4}/\d{2}/\d{2}', race_date):
        print(f"Invalid date format: {race_date}. Expected YYYY/MM/DD")
        sys.exit(1)
    asyncio.run(run_monitor(race_date))


if __name__ == '__main__':
    main()
