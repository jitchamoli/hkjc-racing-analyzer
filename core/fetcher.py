#!/usr/bin/env python3
"""
HKJC Real-Time Data Fetcher v1.0
Fetches live race cards, odds, and results from HKJC using multiple methods:
  1. HKJC JSON API endpoints (fastest, most reliable)
  2. Playwright network interception (catches XHR data)
  3. BeautifulSoup HTML parsing (fallback)
"""

import asyncio
import json
import re
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HorseEntry:
    """A single horse entry with real data from HKJC."""
    number: int
    name: str
    draw: int = 0
    jockey: str = ""
    trainer: str = ""
    weight: str = ""
    rating: float = 0.0
    win_odds: float = 0.0
    place_odds: float = 0.0
    last_6: str = ""
    gear: str = ""
    best_time: str = ""
    age: str = ""
    colour: str = ""
    season_stakes: str = ""
    # Tips Index data from HKJC experts
    tips_odds: float = 0.0          # Odds from tips page (can differ from main odds)
    initial_tips_index: float = 0.0  # Initial expert composite (lower = more favoured)
    raceday_tips_index: float = 0.0  # Race-day updated tips index
    is_top_pick: bool = False        # Highlighted as top pick by HKJC tipsters
    # Advanced analytics fields
    running_style: str = ""          # Leader/Stalker/Midfield/Closer
    last_400m_avg: float = 0.0      # Average last-400m sectional time
    cd_wins: int = 0                # Course & Distance wins
    cd_runs: int = 0                # Course & Distance total runs
    trial_position: int = 0         # Best barrier trial finishing position
    trial_margin: str = ""          # Margin in best trial
    opening_odds: float = 0.0       # Opening odds for movement tracking
    # New high-impact fields
    days_since_last_run: int = 0     # Days since last race (freshness)
    rating_change: int = 0           # Rating +/- from last race
    horse_weight: int = 0            # Declared horse body weight (lbs)
    weight_change: int = 0           # Weight change from last declaration
    import_cat: str = ""             # Import category (PP, PPG, ISG)
    priority: str = ""               # Priority ranking
    sex: str = ""                    # g=gelding, c=colt, f=filly, h=horse


@dataclass
class RaceCard:
    """Full race card with all entries."""
    race_number: int
    venue: str = ""
    date: str = ""
    time: str = ""
    distance: str = ""
    class_type: str = ""
    going: str = ""
    track: str = ""
    course: str = ""
    prize: str = ""
    horses: List[HorseEntry] = field(default_factory=list)
    odds_updated_at: str = ""


@dataclass
class RaceDay:
    """All races for a given meeting date."""
    date: str
    venue: str = ""
    total_races: int = 0
    races: List[RaceCard] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HKJC API Client (Method 1 - JSON endpoints)
# ---------------------------------------------------------------------------

class HKJCApiClient:
    """Fetches data from HKJC's internal JSON/API endpoints via requests."""

    BASE = "https://racing.hkjc.com/racing"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://racing.hkjc.com/racing/information/English/racing/RaceCard.aspx",
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    # -- Race card (HTML) ---------------------------------------------------

    def fetch_race_card_html(self, race_date: str, race_no: int,
                              venue: str = "ST") -> Optional[str]:
        """Fetch raw HTML of the race card page.
        venue: ST (Sha Tin) or HV (Happy Valley)."""
        url = (
            f"{self.BASE}/information/English/racing/"
            f"RaceCard.aspx?racedate={race_date}&Racecourse={venue}&RaceNo={race_no}"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Race card HTML fetch failed: {e}")
            return None

    # -- Odds page (HTML) ---------------------------------------------------

    def fetch_odds_html(self, race_date: str, race_no: int,
                        odds_type: str = "WIN") -> Optional[str]:
        """Fetch odds page HTML.  odds_type: WIN, PLA, QIN, QPL, FCT, TCE, ..."""
        # Primary URL format
        url = (
            f"{self.BASE}/information/English/Odds/"
            f"DisplayOdds.aspx?RaceNo={race_no}&OddsType={odds_type}"
        )
        try:
            resp = self.session.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 5000:
                return resp.text
        except Exception as e:
            logger.warning(f"Odds primary URL failed: {e}")

        # Fallback URL with date
        url2 = (
            f"{self.BASE}/information/English/Racing/"
            f"DisplayOdds.aspx?RaceDate={race_date}&RaceNo={race_no}&OddsType={odds_type}"
        )
        try:
            resp = self.session.get(url2, timeout=30)
            if resp.status_code == 200 and len(resp.text) > 5000:
                return resp.text
        except Exception as e:
            logger.warning(f"Odds fallback URL failed: {e}")

        return None

    # -- Results page (HTML) ------------------------------------------------

    def fetch_results_html(self, race_date: str, race_no: int) -> Optional[str]:
        """Fetch results page for a completed race."""
        url = (
            f"{self.BASE}/information/English/Racing/"
            f"LocalResults.aspx?RaceDate={race_date}&RaceNo={race_no}"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            logger.warning(f"Results HTML fetch failed: {e}")
            return None

    # -- Meeting schedule ---------------------------------------------------

    def fetch_meeting_dates(self) -> List[str]:
        """Return upcoming meeting dates (YYYY/MM/DD) by extracting from page links."""
        url = f"{self.BASE}/information/English/racing/RaceCard.aspx"
        try:
            resp = self.session.get(url, timeout=30)
            # Extract dates from racedate= links in the page
            dates = set()
            for m in re.finditer(r"racedate=(\d{4}/\d{2}/\d{2})", resp.text, re.I):
                dates.add(m.group(1))
            return sorted(dates)
        except Exception as e:
            logger.warning(f"Meeting dates fetch failed: {e}")
            return []

    # -- Tips Index page ---------------------------------------------------

    def fetch_tips_html(self, race_no: int) -> Optional[str]:
        """Fetch HKJC Tips Index page for a race.
        Contains: odds, initial tips index, race-day tips index, top picks."""
        url = (
            f"https://racing.hkjc.com/racing/english/tipsindex/"
            f"tips_index.asp?RaceNo={race_no}"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            if len(resp.text) > 5000:
                return resp.text
        except Exception as e:
            logger.warning(f"Tips page fetch failed: {e}")
        return None

    # -- Horse profile page --------------------------------------------------

    def fetch_horse_profile_html(self, horse_name: str) -> Optional[str]:
        """Fetch a horse's profile/history page from HKJC."""
        from urllib.parse import quote
        encoded = quote(horse_name.strip())
        url = (
            f"{self.BASE}/information/English/Horse/"
            f"Horse.aspx?HorseName={encoded}"
        )
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            if len(resp.text) > 3000:
                return resp.text
        except Exception as e:
            logger.warning(f"Horse profile fetch failed for {horse_name}: {e}")
        return None

    # -- Barrier trial results ------------------------------------------------

    def fetch_barrier_trial_html(self, date: str = "") -> Optional[str]:
        """Fetch barrier trial results page."""
        url = (
            f"{self.BASE}/information/English/Trackwork/"
            f"TrackworkResult.aspx"
        )
        if date:
            url += f"?Date={date}"
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
            if len(resp.text) > 3000:
                return resp.text
        except Exception as e:
            logger.warning(f"Barrier trial fetch failed: {e}")
        return None

    def detect_venue(self, race_date: str) -> str:
        """Detect venue (ST or HV) for a given date."""
        url = f"{self.BASE}/information/English/racing/RaceCard.aspx"
        try:
            resp = self.session.get(url, timeout=30)
            # Look for Racecourse= parameter linked to this date
            pattern = rf"racedate={re.escape(race_date)}&Racecourse=(\w+)"
            m = re.search(pattern, resp.text, re.I)
            if m:
                return m.group(1)
        except Exception:
            pass
        return "ST"  # Default to Sha Tin


# ---------------------------------------------------------------------------
# HTML Parsers (Method 3 - BeautifulSoup)
# ---------------------------------------------------------------------------

class HKJCParser:
    """Parse HKJC HTML pages into structured data using BeautifulSoup.

    Confirmed HKJC race card table column layout (27 cells per row):
      [0]  Horse No.
      [1]  Last 6 Runs (e.g. "7/2/1/4/10" or "-")
      [2]  Colour
      [3]  Horse Name (contains <a> link)
      [4]  Brand No.
      [5]  Wt. (declared weight)
      [6]  Jockey (contains <a> link, may include claim e.g. "(-2)")
      [7]  Over Wt.
      [8]  Draw
      [9]  Trainer (contains <a> link)
      [10] Int'l Rtg.
      [11] Rtg. (local rating, may be "-" for new horses)
      [12] Rtg.+/-
      [13] Horse Wt. (Declaration)
      [14] Wt.+/- (vs Declaration)
      [15] Best Time
      [16-26] Extra info (age, sire, dam, owner, gear, etc.)
    """

    @staticmethod
    def parse_race_card(html: str, race_no: int, race_date: str) -> Optional[RaceCard]:
        """Parse race card HTML into RaceCard dataclass."""
        if not html:
            return None

        soup = BeautifulSoup(html, "html.parser")
        race = RaceCard(race_number=race_no, date=race_date)

        # -- Extract race info from page text --
        page_text = soup.get_text(" ", strip=True)

        # Venue
        venue_match = re.search(r"(Sha Tin|Happy Valley)", page_text)
        if venue_match:
            race.venue = venue_match.group(1)

        # Distance (e.g. "1200M" or "1600 M")
        dist_match = re.search(r"(\d{3,4})\s*M", page_text)
        if dist_match:
            race.distance = dist_match.group(1)

        # Class - look for "Class N" near "Rating:" which is the race info area
        class_match = re.search(r"(?:Rating[^,]*,\s*)(Class\s*\d)", page_text)
        if class_match:
            race.class_type = class_match.group(1)
        else:
            # Broader search
            class_match2 = re.search(r"(Class\s*\d)", page_text)
            if class_match2:
                race.class_type = class_match2.group(1)
            elif re.search(r"Griffin", page_text, re.I):
                race.class_type = "Griffin"
            elif re.search(r"Group\s*\d", page_text):
                gm = re.search(r"Group\s*\d", page_text)
                race.class_type = gm.group(0)

        # Going (e.g. "Good", "Good to Firm", "Yielding")
        going_match = re.search(
            r"(Good to Firm|Good to Yielding|Good|Firm|Yielding|Soft|Heavy)",
            page_text, re.I,
        )
        if going_match:
            race.going = going_match.group(1)

        # Track / Course (e.g. "Turf - \"A\" Course", "All Weather Track")
        track_match = re.search(
            r'(Turf|All Weather Track|AWT)\s*[-\s]*"?([A-C]+)?"?',
            page_text, re.I,
        )
        if track_match:
            race.track = track_match.group(0).strip()

        # Race time
        time_match = re.search(r"(\d{1,2}:\d{2})\s*(pm|am)?", page_text, re.I)
        if time_match:
            race.time = time_match.group(0).strip()

        # Prize
        prize_match = re.search(r"HK\$\s*([\d,]+)", page_text)
        if prize_match:
            race.prize = prize_match.group(1)

        # -- Extract horses from the race card table --
        race.horses = HKJCParser._parse_horse_table(soup)

        if race.horses:
            from data.validation import validate_race_card
            warnings = validate_race_card(race)
            if warnings:
                logger.warning(f"Race {race_no}: {len(warnings)} validation warnings")

        return race if race.horses else None

    @staticmethod
    def _parse_horse_table(soup: BeautifulSoup) -> List[HorseEntry]:
        """Extract horse entries from the race card table."""
        horses = []

        # HKJC race card table: class contains "tableBorderBlue" and "tdAlignC"
        table = soup.find(
            "table",
            class_=lambda c: c and "tableBorderBlue" in c and "tdAlignC" in c,
        )

        if not table:
            # Fallback: any tableBorderBlue
            tables = soup.find_all(
                "table", class_=lambda c: c and "tableBorderBlue" in c
            )
            for t in tables:
                rows = t.find_all("tr")
                if any(len(r.find_all("td")) >= 10 for r in rows):
                    table = t
                    break

        if not table:
            return horses

        for row in table.find_all("tr"):
            cells = row.find_all("td")
            if len(cells) < 10:
                continue

            texts = [c.get_text(strip=True) for c in cells]

            # Cell [0] must be a horse number (1-24)
            if not re.match(r"^\d{1,2}$", texts[0]):
                continue
            horse_num = int(texts[0])
            if horse_num < 1 or horse_num > 24:
                continue

            # Horse name from cell [3] (prefer <a> link text)
            name = ""
            if len(cells) > 3:
                links = cells[3].find_all("a")
                name = links[0].get_text(strip=True) if links else texts[3]
            if not name or len(name) < 2:
                continue

            # Last 6 runs from cell [1] (e.g. "7/2/1/4/10" or "-")
            last_6_raw = texts[1] if len(texts) > 1 else ""
            # Normalize: keep digits and slashes, replace "/" with nothing for
            # compatibility with the analysis code that expects "121213" format
            last_6_clean = re.sub(r"[^\d/\-]", "", last_6_raw)
            # Keep slash-separated format to preserve multi-digit positions
            # "7/2/1/4/10" → "7/2/1/4/10" (not "721410")
            last_6_positions = []
            for part in last_6_clean.split("/"):
                part = part.strip("-")
                if part.isdigit():
                    last_6_positions.append(part)
            last_6 = "/".join(last_6_positions)

            # Jockey from cell [6] (may include claim like "(-2)")
            jockey = ""
            if len(cells) > 6:
                j_links = cells[6].find_all("a")
                jockey = j_links[0].get_text(strip=True) if j_links else texts[6]
            # Strip claim notation
            jockey = re.sub(r"\s*\(-?\d+\)", "", jockey).strip()

            # Trainer from cell [9]
            trainer = ""
            if len(cells) > 9:
                t_links = cells[9].find_all("a")
                trainer = t_links[0].get_text(strip=True) if t_links else texts[9]

            # Draw from cell [8]
            draw = HKJCParser._safe_int(texts[8]) if len(texts) > 8 else 0

            # Weight from cell [5]
            weight = texts[5] if len(texts) > 5 else ""

            # Rating from cell [11] (may be "-" for unrated horses)
            rating = 0.0
            if len(texts) > 11 and texts[11] != "-":
                rating = HKJCParser._safe_float(texts[11])

            # Int'l rating from cell [10] as fallback
            if rating == 0 and len(texts) > 10 and texts[10] != "-":
                rating = HKJCParser._safe_float(texts[10])

            # Rating +/- from cell [12]
            rating_change = 0
            if len(texts) > 12 and texts[12] not in ("-", ""):
                rating_change = HKJCParser._safe_int(texts[12])

            # Horse weight (declaration) from cell [13]
            horse_weight = 0
            if len(texts) > 13 and texts[13] not in ("-", ""):
                horse_weight = HKJCParser._safe_int(texts[13])

            # Weight change from cell [14]
            weight_change = 0
            if len(texts) > 14 and texts[14] not in ("-", ""):
                weight_change = HKJCParser._safe_int(texts[14])

            # Best time from cell [15]
            best_time = texts[15] if len(texts) > 15 else ""

            # Age from cell [16]
            age = texts[16] if len(texts) > 16 else ""

            # Sex from cell [18]
            sex = texts[18] if len(texts) > 18 else ""

            # Season stakes from cell [19]
            season_stakes = texts[19] if len(texts) > 19 else ""

            # Priority from cell [20]
            priority = texts[20] if len(texts) > 20 else ""

            # Days since last run from cell [21]
            days_since_last_run = 0
            if len(texts) > 21 and texts[21] not in ("-", ""):
                days_since_last_run = HKJCParser._safe_int(texts[21])

            # Gear from cell [22]
            gear = ""
            if len(texts) > 22:
                gear = texts[22]

            # Cell 23=Owner, 24=Sire, 25=Dam, 26=Import Cat
            import_cat = ""
            if len(texts) > 26 and re.match(r'^(PP|PPG|ISG|ISA)$', texts[26]):
                import_cat = texts[26]
            elif len(texts) > 25:
                for ci in range(23, min(len(texts), 28)):
                    if re.match(r'^(PP|PPG|ISG|ISA)$', texts[ci]):
                        import_cat = texts[ci]
                        break

            if not jockey:
                continue

            horses.append(HorseEntry(
                number=horse_num,
                name=name,
                draw=draw,
                jockey=jockey,
                trainer=trainer,
                weight=weight,
                rating=rating,
                last_6=last_6,
                gear=gear,
                best_time=best_time,
                age=age,
                season_stakes=season_stakes,
                days_since_last_run=days_since_last_run,
                rating_change=rating_change,
                horse_weight=horse_weight,
                weight_change=weight_change,
                import_cat=import_cat,
                priority=priority,
                sex=sex,
            ))

        return horses

    @staticmethod
    def parse_odds(html: str) -> Dict[int, float]:
        """Parse WIN odds page HTML → {horse_number: decimal_odds}."""
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        odds: Dict[int, float] = {}

        # --- Method A: structured table parsing ---
        tables = soup.select("table.tableBorderBlue, table.table_bd, table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 2:
                    continue
                texts = [c.get_text(strip=True) for c in cells]
                # Look for patterns: horse_number ... odds_value
                horse_no = HKJCParser._safe_int(texts[0])
                if horse_no < 1 or horse_no > 24:
                    continue
                # Odds is usually the last numeric cell
                for t in reversed(texts[1:]):
                    odds_val = HKJCParser._safe_float(t)
                    if 1.0 < odds_val < 999:
                        odds[horse_no] = odds_val
                        break

        if odds:
            return odds

        # --- Method B: regex on full page text ---
        page_text = soup.get_text(" ")
        # Pattern: horse_no followed by odds somewhere nearby
        for m in re.finditer(r"(\d{1,2})\s+[\w\s']+?\s+(\d+\.?\d*)\b", page_text):
            hno = int(m.group(1))
            oval = float(m.group(2))
            if 1 <= hno <= 24 and 1.0 < oval < 999:
                odds.setdefault(hno, oval)

        return odds

    @staticmethod
    def parse_place_odds(html: str) -> Dict[int, float]:
        """Parse PLACE odds page HTML → {horse_number: decimal_odds}."""
        return HKJCParser.parse_odds(html)  # Same structure, different page

    @staticmethod
    def parse_results(html: str) -> List[Dict]:
        """Parse results page → list of finishing positions.

        Dedupes by (position, horse_number). The table selector
        "table.tableBorderBlue, table.table_bd, table" overlaps — bare `table`
        matches everything including the first two. Post-meeting HTML also
        contains summary tables that repeat the same race data. Without
        dedup, re-running results_fetcher appends duplicate actual_results
        rows to race_data_*_with_results.json (some with win_odds=2000
        sentinel from "no longer accepting bets" rows).
        """
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        results = []
        seen = set()

        tables = soup.select("table.tableBorderBlue, table.table_bd, table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                texts = [c.get_text(strip=True) for c in cells]
                pos = HKJCParser._safe_int(texts[0])
                if pos < 1 or pos > 24:
                    continue
                horse_no = HKJCParser._safe_int(texts[1])
                raw_name = texts[2] if len(texts) > 2 else ""
                # HKJC results cell collapses to e.g. 'PODIUM(H173)' — strip
                # the brand parenthetical so the name matches prediction_log
                # (which stores 'PODIUM'). Without this, _fill_prediction_log
                # silently UPDATEs 0 rows on every results fetch.
                name = re.sub(r"\s*\([^)]+\)\s*$", "", raw_name).strip()
                if horse_no <= 0 or not name:
                    continue
                key = (pos, horse_no)
                if key in seen:
                    continue
                seen.add(key)
                results.append({
                    "position": pos,
                    "number": horse_no,
                    "name": name,
                    "finish_time": texts[4] if len(texts) > 4 else "",
                    "win_odds": HKJCParser._safe_float(texts[-1]) if texts[-1] else 0,
                })

        return results

    @staticmethod
    def parse_tips(html: str) -> Dict[int, Dict]:
        """Parse HKJC Tips Index page.

        Returns {horse_number: {odds, initial_tips_index, raceday_tips_index, is_top_pick}}.

        The tips page contains:
          - Top 2 highlighted picks with TipsIndex and Odds
          - A full table:  No | Name | Weight | Trainer | Jockey | Draw | Odds | Initial Tips Index | Race Day Tips Index
        """
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        tips: Dict[int, Dict] = {}

        # --- Extract the main tips table ---
        # Find the table whose header row contains "Odds" and "Initial"
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            header_text = rows[0].get_text(" ", strip=True)
            if "Odds" not in header_text:
                continue
            if "Initial" not in header_text and "Tips Index" not in header_text:
                continue

            # This looks like the tips table; parse data rows
            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) < 8:
                    continue
                # Columns: No | Name | Weight | Trainer | Jockey | Draw | Odds | Initial Tips Index | Race Day Tips Index
                try:
                    horse_no = int(cells[0])
                except (ValueError, IndexError):
                    continue
                if horse_no < 1 or horse_no > 24:
                    continue

                odds = HKJCParser._safe_float(cells[6])
                initial_ti = HKJCParser._safe_float(cells[7])
                raceday_ti = HKJCParser._safe_float(cells[8]) if len(cells) > 8 else 0.0

                tips[horse_no] = {
                    "odds": odds,
                    "initial_tips_index": initial_ti,
                    "raceday_tips_index": raceday_ti,
                    "is_top_pick": False,
                }

            if tips:
                break

        # --- Extract top picks (the 2 highlighted horses) ---
        page_text = soup.get_text(" ", strip=True)
        # Pattern: "number:N...Name:HORSE_NAME...TipsIndex:X.X"
        top_picks = re.findall(
            r"number:\s*(\d+).*?Name:\s*([A-Z\s']+?)(?:Trainer|$).*?TipsIndex:\s*([\d.]+)",
            page_text,
        )
        for no_str, name, ti in top_picks:
            no = int(no_str)
            if no in tips:
                tips[no]["is_top_pick"] = True
            else:
                tips[no] = {
                    "odds": 0,
                    "initial_tips_index": HKJCParser._safe_float(ti),
                    "raceday_tips_index": 0,
                    "is_top_pick": True,
                }

        return tips

    # -- Horse history parsing -------------------------------------------------

    @staticmethod
    def parse_horse_history(html: str) -> List[Dict]:
        """Parse horse profile page → list of past race records.

        Returns list of dicts with keys:
            date, venue, distance (int), course, position (int),
            class_type, running_positions (list[int]), last_400m (float)
        """
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        records: List[Dict] = []

        # The horse history table typically has class "tableBorderBlue" or "bigborder"
        tables = soup.select("table.tableBorderBlue, table.bigborder, table.table_bd, table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            # Check if this looks like a race history table
            header = rows[0].get_text(" ", strip=True).lower()
            if "place" not in header and "pos" not in header:
                continue

            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) < 6:
                    continue

                # Try to extract position from first few cells
                pos = 0
                date_str = ""
                venue_str = ""
                dist_val = 0
                course_str = ""

                for i, cell in enumerate(cells):
                    if not pos and re.match(r"^\d{1,2}$", cell):
                        pos = int(cell)
                    if not date_str and re.match(r"\d{2}/\d{2}/\d{2,4}", cell):
                        date_str = cell
                    if not venue_str and ("Sha Tin" in cell or "Happy Valley" in cell or "ST" == cell or "HV" == cell):
                        venue_str = cell
                    if not dist_val:
                        dm = re.match(r"^(\d{3,4})$", cell)
                        if dm:
                            dist_val = int(dm.group(1))
                    if not course_str and re.search(r'"?[A-C](?:\+\d)?"?', cell):
                        course_str = cell

                if pos > 0 and pos <= 24:
                    # Try to extract running positions (columns showing position at each call)
                    running_pos = []
                    for cell in cells:
                        if re.match(r"^\d{1,2}$", cell):
                            rp = int(cell)
                            if 1 <= rp <= 24:
                                running_pos.append(rp)

                    # Try to extract sectional / last 400m time
                    last_400m = 0.0
                    for cell in cells:
                        tm = re.match(r"^(\d{2}\.\d{1,2})$", cell)
                        if tm:
                            t = float(tm.group(1))
                            if 20.0 < t < 30.0:  # reasonable last-400m time
                                last_400m = t

                    records.append({
                        "date": date_str,
                        "venue": venue_str,
                        "distance": dist_val,
                        "course": course_str,
                        "position": pos,
                        "class_type": "",
                        "running_positions": running_pos[1:] if len(running_pos) > 1 else [],
                        "last_400m": last_400m,
                    })

            if records:
                break  # Found the right table

        return records

    # -- Barrier trial parsing -------------------------------------------------

    @staticmethod
    def parse_barrier_trials(html: str) -> List[Dict]:
        """Parse barrier trial results page → list of trial results.

        Returns list of dicts: {horse_name, position, time, margin, date}
        """
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        trials: List[Dict] = []

        tables = soup.select("table.tableBorderBlue, table.table_bd, table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 3:
                continue

            header = rows[0].get_text(" ", strip=True).lower()
            if "horse" not in header and "name" not in header:
                continue

            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all("td")]
                if len(cells) < 3:
                    continue

                pos = HKJCParser._safe_int(cells[0])
                if pos < 1 or pos > 24:
                    continue

                name = ""
                time_str = ""
                margin = ""
                date_str = ""

                for cell in cells[1:]:
                    if not name and re.match(r"^[A-Z\s']+$", cell) and len(cell) > 3:
                        name = cell.strip()
                    if not time_str and re.match(r"\d{1,2}:\d{2}\.\d", cell):
                        time_str = cell
                    if not margin and re.match(r"^[\d.]+L?$|^HD$|^NK$|^SHD?$|^N$", cell):
                        margin = cell
                    if not date_str and re.match(r"\d{2}/\d{2}/\d{2,4}", cell):
                        date_str = cell

                if name:
                    trials.append({
                        "horse_name": name,
                        "position": pos,
                        "time": time_str,
                        "margin": margin,
                        "date": date_str,
                    })

            if trials:
                break

        return trials

    # -- Enhanced results parsing (running positions + sectionals) -------------

    @staticmethod
    def parse_results_enhanced(html: str) -> List[Dict]:
        """Parse results page with running positions and sectional times.

        Returns list of dicts with keys:
            position, number, name, finish_time, win_odds,
            running_positions (list[int]), last_400m (float)
        """
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        results = []

        tables = soup.select("table.tableBorderBlue, table.table_bd, table")
        for table in tables:
            rows = table.find_all("tr")
            for row in rows:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue
                texts = [c.get_text(strip=True) for c in cells]
                pos = HKJCParser._safe_int(texts[0])
                if pos < 1 or pos > 24:
                    continue
                horse_no = HKJCParser._safe_int(texts[1])
                # Parity with parse_results — strip trailing brand parenthetical
                # like 'PODIUM(H173)' so downstream consumers see clean names.
                raw_name = texts[2] if len(texts) > 2 else ""
                name = re.sub(r"\s*\([^)]+\)\s*$", "", raw_name).strip()
                if horse_no <= 0 or not name:
                    continue

                # Extract running positions (numeric cells that represent call positions)
                running_pos = []
                for t in texts[3:]:
                    if re.match(r"^\d{1,2}$", t):
                        rp = int(t)
                        if 1 <= rp <= 24:
                            running_pos.append(rp)

                # Extract sectional / last-400m time
                last_400m = 0.0
                for t in texts:
                    tm = re.match(r"^(\d{2}\.\d{1,2})$", t)
                    if tm:
                        val = float(tm.group(1))
                        if 20.0 < val < 30.0:
                            last_400m = val

                results.append({
                    "position": pos,
                    "number": horse_no,
                    "name": name,
                    "finish_time": texts[4] if len(texts) > 4 else "",
                    "win_odds": HKJCParser._safe_float(texts[-1]) if texts[-1] else 0,
                    "running_positions": running_pos,
                    "last_400m": last_400m,
                })

        return results

    @staticmethod
    def _safe_int(text: str) -> int:
        try:
            cleaned = re.sub(r"[^\d\-]", "", str(text))
            # Handle edge cases like empty string or lone minus
            return int(cleaned) if cleaned and cleaned != '-' else 0
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def _safe_float(text: str) -> float:
        try:
            m = re.search(r"-?[\d]+\.?\d*", str(text))
            return float(m.group()) if m else 0.0
        except (ValueError, TypeError):
            return 0.0


# ---------------------------------------------------------------------------
# Playwright-based fetcher with network interception (Method 2)
# ---------------------------------------------------------------------------

class HKJCPlaywrightFetcher:
    """
    Uses Playwright to load HKJC pages and intercept XHR responses
    that contain JSON race data / odds updates.
    """

    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        # NB: intercepted XHR data is held in a per-call local list (not
        # instance state) so concurrent fetches don't clobber each other.

    async def start(self):
        from playwright.async_api import async_playwright
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        ua = (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
        self.context = await self.browser.new_context(
            user_agent=ua,
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="Asia/Hong_Kong",
        )
        # Stealth
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
            window.chrome = { runtime: {} };
        """)

    async def stop(self):
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()

    async def _await_content(self, page, timeout: int = 4000):
        """Wait for the main HKJC content table instead of a blind sleep.

        page.goto(wait_until="networkidle") already settled the network; this
        just confirms the table we parse is actually in the DOM, returning as
        soon as it appears (usually near-instant) rather than always sleeping
        a fixed 1-2s. Falls through on timeout — the parser handles a missing
        table gracefully.
        """
        try:
            await page.wait_for_selector("table.tableBorderBlue", timeout=timeout)
        except Exception:
            pass

    async def fetch_race_card_with_odds(
        self, race_date: str, race_no: int
    ) -> Optional[RaceCard]:
        """
        Load race card page, intercept any JSON XHR responses,
        and parse both HTML + intercepted data.
        """
        # Per-call local — concurrent fetches must not share this list.
        intercepted: List[dict] = []
        page = await self.context.new_page()

        # Intercept network responses that might contain JSON race data
        async def on_response(response):
            url = response.url.lower()
            if any(kw in url for kw in ["odds", "racecard", "json", "getodds", "displayodds"]):
                try:
                    body = await response.text()
                    if body.strip().startswith(("{", "[")):
                        data = json.loads(body)
                        intercepted.append({
                            "url": response.url,
                            "data": data,
                        })
                        logger.info(f"Intercepted JSON from: {response.url}")
                except Exception:
                    pass

        page.on("response", on_response)

        try:
            url = (
                f"https://racing.hkjc.com/racing/information/English/Racing/"
                f"RaceCard.aspx?RaceDate={race_date}&RaceNo={race_no}"
            )
            await page.goto(url, wait_until="networkidle", timeout=45000)
            await self._await_content(page)

            html = await page.content()
            race = HKJCParser.parse_race_card(html, race_no, race_date)

            if race:
                # Try to get odds from the same page or intercepted data
                odds = self._extract_odds_from_intercepted(intercepted)
                if not odds:
                    odds = HKJCParser.parse_odds(html)

                if odds:
                    for horse in race.horses:
                        if horse.number in odds:
                            horse.win_odds = odds[horse.number]
                    race.odds_updated_at = datetime.now().strftime("%H:%M:%S")

            # If no odds yet, try the dedicated odds page
            if race and not any(h.win_odds > 0 for h in race.horses):
                await self._fetch_and_apply_odds(page, race, race_date, race_no, intercepted)

            await page.close()
            return race

        except Exception as e:
            logger.error(f"Playwright fetch failed: {e}")
            await page.close()
            return None

    async def fetch_live_odds(self, race_date: str, race_no: int) -> Dict[int, float]:
        """Fetch just the odds for a race (quick refresh)."""
        # Per-call local — concurrent fetches must not share this list.
        intercepted: List[dict] = []
        page = await self.context.new_page()

        async def on_response(response):
            url = response.url.lower()
            if "odds" in url or "json" in url:
                try:
                    body = await response.text()
                    if body.strip().startswith(("{", "[")):
                        intercepted.append({
                            "url": response.url,
                            "data": json.loads(body),
                        })
                except Exception:
                    pass

        page.on("response", on_response)

        try:
            url = (
                f"https://racing.hkjc.com/racing/information/English/Racing/"
                f"DisplayOdds.aspx?RaceDate={race_date}&RaceNo={race_no}&OddsType=WIN"
            )
            await page.goto(url, wait_until="networkidle", timeout=30000)
            await self._await_content(page)

            odds = self._extract_odds_from_intercepted(intercepted)
            if not odds:
                html = await page.content()
                odds = HKJCParser.parse_odds(html)

            await page.close()
            return odds

        except Exception as e:
            logger.error(f"Live odds fetch failed: {e}")
            await page.close()
            return {}

    async def _fetch_and_apply_odds(
        self, page, race: RaceCard, race_date: str, race_no: int,
        intercepted: List[dict],
    ):
        """Navigate to odds page and apply odds to race card.

        `intercepted` is the caller's per-call XHR list — this page's
        on_response handler (registered by the caller) appends to it.
        """
        try:
            odds_url = (
                f"https://racing.hkjc.com/racing/information/English/Racing/"
                f"DisplayOdds.aspx?RaceDate={race_date}&RaceNo={race_no}&OddsType=WIN"
            )
            await page.goto(odds_url, wait_until="networkidle", timeout=30000)
            await self._await_content(page)

            odds = self._extract_odds_from_intercepted(intercepted)
            if not odds:
                html = await page.content()
                odds = HKJCParser.parse_odds(html)

            if odds:
                for horse in race.horses:
                    if horse.number in odds:
                        horse.win_odds = odds[horse.number]
                race.odds_updated_at = datetime.now().strftime("%H:%M:%S")
                logger.info(f"Applied {len(odds)} win odds from odds page")

            # Also try place odds
            place_url = odds_url.replace("OddsType=WIN", "OddsType=PLA")
            await page.goto(place_url, wait_until="networkidle", timeout=30000)
            await self._await_content(page)

            place_odds = self._extract_odds_from_intercepted(intercepted)
            if not place_odds:
                html = await page.content()
                place_odds = HKJCParser.parse_place_odds(html)

            if place_odds:
                for horse in race.horses:
                    if horse.number in place_odds:
                        horse.place_odds = place_odds[horse.number]

        except Exception as e:
            logger.warning(f"Odds page fetch failed: {e}")

    def _extract_odds_from_intercepted(self, intercepted: List[dict]) -> Dict[int, float]:
        """Extract horse odds from intercepted XHR JSON data."""
        odds: Dict[int, float] = {}

        for item in intercepted:
            data = item["data"]
            self._walk_json_for_odds(data, odds)

        return odds

    def _walk_json_for_odds(self, data, odds: Dict[int, float], depth: int = 0):
        """Recursively walk JSON to find horse number + odds pairs."""
        if depth > 30:
            return
        if isinstance(data, dict):
            # Common HKJC JSON patterns
            no_keys = ["no", "horseNo", "horse_no", "number", "No"]
            odds_keys = ["odds", "winOdds", "win_odds", "currentOdds", "Odds", "WinOdds"]

            horse_no = None
            odds_val = None

            for nk in no_keys:
                if nk in data:
                    try:
                        horse_no = int(data[nk])
                    except (ValueError, TypeError):
                        pass
                    break

            for ok in odds_keys:
                if ok in data:
                    try:
                        odds_val = float(data[ok])
                    except (ValueError, TypeError):
                        pass
                    break

            if horse_no and odds_val and odds_val > 1.0:
                odds[horse_no] = odds_val

            for v in data.values():
                if isinstance(v, (dict, list)):
                    self._walk_json_for_odds(v, odds, depth + 1)

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    self._walk_json_for_odds(item, odds, depth + 1)


# ---------------------------------------------------------------------------
# Unified Data Fetcher (orchestrates all methods)
# ---------------------------------------------------------------------------

class HKJCDataFetcher:
    """
    High-level interface that tries multiple methods to get real-time data:
      1. requests + BeautifulSoup (fast, no browser needed)
      2. Playwright with network interception (handles JS-rendered content)

    Usage:
        fetcher = HKJCDataFetcher()
        race = await fetcher.get_race_card("2026/04/12", 1)
        odds = await fetcher.get_live_odds("2026/04/12", 1)
    """

    def __init__(self, use_playwright: bool = True):
        self.api = HKJCApiClient()
        self.use_playwright = use_playwright
        self._pw_fetcher: Optional[HKJCPlaywrightFetcher] = None
        self._pw_lock = asyncio.Lock()
        self._horse_history_cache: Dict[str, List[Dict]] = {}
        self._horse_trial_cache: Dict[str, List[Dict]] = {}

    async def _get_pw(self) -> HKJCPlaywrightFetcher:
        # Lock guards the lazy-singleton init — without it, concurrent
        # get_race_card() calls could each launch a browser and leak.
        async with self._pw_lock:
            if self._pw_fetcher is None:
                self._pw_fetcher = HKJCPlaywrightFetcher()
                await self._pw_fetcher.start()
            return self._pw_fetcher

    async def close(self):
        if self._pw_fetcher:
            await self._pw_fetcher.stop()
            self._pw_fetcher = None

    async def get_race_card(self, race_date: str, race_no: int,
                            venue: str = "") -> Optional[RaceCard]:
        """
        Get full race card with horse entries.
        Tries requests first, then Playwright if needed.
        """
        if not venue:
            venue = self.api.detect_venue(race_date)

        # Method 1: requests + BS4 (fast)
        logger.info(f"[Method 1] Fetching race card via requests ({venue})...")
        html = self.api.fetch_race_card_html(race_date, race_no, venue)
        race = HKJCParser.parse_race_card(html, race_no, race_date) if html else None

        if race and race.horses:
            logger.info(f"[Method 1] Got {len(race.horses)} horses from HTML")
            # Try to get odds via requests too
            await self._apply_odds_via_requests(race, race_date, race_no)

            # Always apply tips data (has odds + expert ratings)
            await self._apply_tips_data(race, race_no)

            if any(h.win_odds > 0 for h in race.horses):
                return race

        # Method 2: Playwright with network interception
        if self.use_playwright:
            logger.info(f"[Method 2] Fetching via Playwright...")
            try:
                pw = await self._get_pw()
                pw_race = await pw.fetch_race_card_with_odds(race_date, race_no)
                if pw_race and pw_race.horses:
                    logger.info(
                        f"[Method 2] Got {len(pw_race.horses)} horses, "
                        f"odds for {sum(1 for h in pw_race.horses if h.win_odds > 0)}"
                    )
                    # Merge: prefer Playwright data if it has more info
                    if not race or len(pw_race.horses) >= len(race.horses):
                        return pw_race
                    # Or just merge odds into existing race card
                    if race:
                        pw_odds = {h.number: h.win_odds for h in pw_race.horses if h.win_odds > 0}
                        pw_place = {h.number: h.place_odds for h in pw_race.horses if h.place_odds > 0}
                        for h in race.horses:
                            if h.number in pw_odds:
                                h.win_odds = pw_odds[h.number]
                            if h.number in pw_place:
                                h.place_odds = pw_place[h.number]
                        race.odds_updated_at = pw_race.odds_updated_at
                        return race
            except Exception as e:
                logger.warning(f"Playwright method failed: {e}")

        return race

    async def get_live_odds(self, race_date: str, race_no: int) -> Dict[int, float]:
        """Get latest win odds (for refreshing during a meeting)."""
        # Try requests first
        html = self.api.fetch_odds_html(race_date, race_no, "WIN")
        odds = HKJCParser.parse_odds(html) if html else {}

        if odds:
            return odds

        # Playwright fallback
        if self.use_playwright:
            try:
                pw = await self._get_pw()
                return await pw.fetch_live_odds(race_date, race_no)
            except Exception as e:
                logger.warning(f"Playwright odds fetch failed: {e}")

        return {}

    async def get_place_odds(self, race_date: str, race_no: int) -> Dict[int, float]:
        """Get latest place odds."""
        html = self.api.fetch_odds_html(race_date, race_no, "PLA")
        odds = HKJCParser.parse_place_odds(html) if html else {}
        return odds

    async def get_results(self, race_date: str, race_no: int) -> List[Dict]:
        """Get race results (after the race is finished)."""
        html = self.api.fetch_results_html(race_date, race_no)
        return HKJCParser.parse_results(html) if html else []

    async def get_meeting_dates(self) -> List[str]:
        """Get upcoming/recent meeting dates."""
        return self.api.fetch_meeting_dates()

    async def get_all_races(self, race_date: str) -> RaceDay:
        """Fetch all races for a given meeting date."""
        day = RaceDay(date=race_date)

        # Detect venue before fetching to avoid wrong-venue empty HTML
        venue = self.api.detect_venue(race_date)

        # First, figure out how many races there are
        html = self.api.fetch_race_card_html(race_date, 1, venue=venue)
        if not html:
            return day

        soup = BeautifulSoup(html, "html.parser")
        race_nums = set()
        for a in soup.find_all("a", href=True):
            m = re.search(r"RaceNo=(\d+)", a["href"])
            if m:
                race_nums.add(int(m.group(1)))

        if not race_nums:
            race_nums = {1}  # At least try race 1

        day.total_races = max(race_nums) if race_nums else 0

        # Fetch each race
        for rno in sorted(race_nums):
            race = await self.get_race_card(race_date, rno, venue=venue)
            if race:
                day.races.append(race)
                if not day.venue and race.venue:
                    day.venue = race.venue

        return day

    async def get_horse_history(
        self, horse_name: str, num_races: int = 6
    ) -> List[Dict]:
        """Get a horse's past race records (with caching).

        Returns list of dicts: date, venue, distance, course, position,
        running_positions, last_400m.
        """
        cache_key = horse_name.strip().upper()
        if cache_key in self._horse_history_cache:
            return self._horse_history_cache[cache_key][:num_races]

        await asyncio.sleep(1.5)  # Rate limit (non-blocking)

        html = self.api.fetch_horse_profile_html(horse_name)
        history = HKJCParser.parse_horse_history(html) if html else []
        self._horse_history_cache[cache_key] = history
        return history[:num_races]

    async def get_barrier_trials(self, horse_name: str) -> List[Dict]:
        """Get barrier trial results for a horse (with caching).

        Returns list of dicts: horse_name, position, time, margin, date.
        """
        cache_key = horse_name.strip().upper()
        if cache_key in self._horse_trial_cache:
            return self._horse_trial_cache[cache_key]

        # Fetch recent trial results page
        html = self.api.fetch_barrier_trial_html()
        all_trials = HKJCParser.parse_barrier_trials(html) if html else []

        # Filter for this horse
        horse_trials = [
            t for t in all_trials
            if horse_name.upper() in t.get("horse_name", "").upper()
        ]
        self._horse_trial_cache[cache_key] = horse_trials
        return horse_trials

    async def snapshot_odds(self, race_date: str, race_no: int):
        """Fetch current odds and record to SQLite for movement tracking."""
        from odds_tracker import OddsDatabase

        odds = await self.get_live_odds(race_date, race_no)
        if odds:
            db = OddsDatabase()
            horses = [
                {"number": num, "name": "", "win_odds": ov, "place_odds": 0}
                for num, ov in odds.items()
            ]
            db.record_snapshot(race_date, race_no, horses, source="fetcher")

    async def _apply_odds_via_requests(
        self, race: RaceCard, race_date: str, race_no: int
    ):
        """Try to get odds via plain requests and apply to race card."""
        win_html = self.api.fetch_odds_html(race_date, race_no, "WIN")
        win_odds = HKJCParser.parse_odds(win_html) if win_html else {}

        if win_odds:
            for h in race.horses:
                if h.number in win_odds:
                    h.win_odds = win_odds[h.number]
            race.odds_updated_at = datetime.now().strftime("%H:%M:%S")

        pla_html = self.api.fetch_odds_html(race_date, race_no, "PLA")
        pla_odds = HKJCParser.parse_place_odds(pla_html) if pla_html else {}

        if pla_odds:
            for h in race.horses:
                if h.number in pla_odds:
                    h.place_odds = pla_odds[h.number]

    async def _apply_tips_data(self, race: RaceCard, race_no: int):
        """Fetch HKJC Tips Index data and merge into race card."""
        tips_html = self.api.fetch_tips_html(race_no)
        tips = HKJCParser.parse_tips(tips_html) if tips_html else {}

        if not tips:
            return

        for h in race.horses:
            if h.number not in tips:
                continue
            td = tips[h.number]

            # Use tips odds as win_odds if we don't have odds yet
            if td["odds"] > 0:
                if h.win_odds <= 0:
                    h.win_odds = td["odds"]
                h.tips_odds = td["odds"]

            h.initial_tips_index = td["initial_tips_index"]
            h.raceday_tips_index = td["raceday_tips_index"]
            h.is_top_pick = td["is_top_pick"]

        tips_count = sum(1 for h in race.horses if h.initial_tips_index > 0)
        odds_count = sum(1 for h in race.horses if h.win_odds > 0)
        logger.info(f"Tips data applied: {tips_count} with tips index, {odds_count} with odds")

        if not race.odds_updated_at and odds_count > 0:
            race.odds_updated_at = datetime.now().strftime("%H:%M:%S") + " (tips)"


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------

async def test_fetcher():
    """Quick test of the data fetcher."""
    logging.basicConfig(level=logging.INFO)

    fetcher = HKJCDataFetcher(use_playwright=False)

    print("=" * 70)
    print("HKJC Data Fetcher - Test")
    print("=" * 70)

    # Check meeting dates
    dates = await fetcher.get_meeting_dates()
    if dates:
        print(f"\nUpcoming/recent meetings: {dates[:5]}")
        test_date = dates[0]
    else:
        test_date = datetime.now().strftime("%Y/%m/%d")
        print(f"\nNo dates found, using today: {test_date}")

    # Fetch race 1
    print(f"\nFetching Race 1 for {test_date}...")
    race = await fetcher.get_race_card(test_date, 1)

    if race:
        print(f"\n  Venue: {race.venue}")
        print(f"  Distance: {race.distance}m | Class: {race.class_type}")
        print(f"  Going: {race.going}")
        print(f"  Horses: {len(race.horses)}")
        print(f"  Odds updated: {race.odds_updated_at or 'N/A'}")
        print()

        has_odds = sum(1 for h in race.horses if h.win_odds > 0)
        print(f"  {'No':<4}{'Name':<20}{'Jockey':<18}{'Draw':<6}{'Rtg':<6}{'WinOdds':<10}")
        print(f"  {'-'*64}")
        for h in race.horses:
            print(
                f"  {h.number:<4}{h.name:<20}{h.jockey:<18}"
                f"{h.draw:<6}{h.rating:<6.0f}"
                f"{h.win_odds if h.win_odds > 0 else 'N/A':<10}"
            )

        print(f"\n  Horses with real odds: {has_odds}/{len(race.horses)}")
    else:
        print("  No race data found (no meeting on this date?)")

    await fetcher.close()


if __name__ == "__main__":
    asyncio.run(test_fetcher())
