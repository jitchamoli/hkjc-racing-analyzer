"""HKJC sectional-time scraper + parser (Path B, Step 1).

The displaysectionaltime page gives per-horse, per-section running data:
position, margin behind leader, section time, and 200m sub-splits for the
final sections. This is the richest "how the race was run" data HKJC
publishes — it feeds the Path B sectional-speed features.

URL: https://racing.hkjc.com/en-us/local/information/displaysectionaltime
     ?racedate=DD/MM/YYYY&RaceNo=N
     (note: DD/MM/YYYY in the URL, not the project-standard YYYY/MM/DD)

Plain `requests` is enough — no Playwright (confirmed in Phase 0 recon).

Per-horse sectional cells are concatenated, e.g.:
    "5227.84"               -> pos 5, margin "2",     time 27.84
    "31-1/424.2212.0012.22" -> pos 3, margin "1-1/4", time 24.22, splits [12.00, 12.22]
    "1SH24.1011.8112.29"    -> pos 1, margin "SH",    time 24.10, splits [11.81, 12.29]

Parsing strategy: the section time is the FIRST NN.NN token in the cell —
always unambiguous. Everything before it is "{position}{margin}", everything
after is 200m sub-splits. The position/margin split is genuinely ambiguous
in isolation ("11" = pos-1-margin-1 OR pos-11-margin-empty), so it is
resolved per-section using the constraint that the N horses' positions form
a permutation of 1..N.

Section count varies by race distance (3-6); unused trailing cells are empty.
Sub-splits are only present on the final ~2 sections (the 200m-split run-in).
"""
import logging
import re
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup

from analytics.speed_figures import parse_finish_time

logger = logging.getLogger(__name__)

SECTIONAL_URL = (
    "https://racing.hkjc.com/en-us/local/information/displaysectionaltime"
)

# A valid margin token: empty (leader), a standard HK margin abbreviation, a
# whole number of lengths, or a fraction like "3/4", "1/2", "1-1/4", "5-3/4".
_MARGIN_RE = re.compile(r"^(?:|SH|HD|NK|NS|N|DH|SHD|\d+(?:-\d+/\d+|/\d+)?)$")
# A section / sub-split time token: exactly NN.NN seconds.
_TIME_RE = re.compile(r"\d{2}\.\d{2}")
# Horse name + brand number: "PODIUM (H173)" (the gap is a non-breaking space).
_NAME_BRAND_RE = re.compile(r"^(.*?)\s*\(([^)]+)\)\s*$")


def _is_valid_margin(s: str) -> bool:
    return bool(_MARGIN_RE.match(s))


def parse_sectional_cell(cell: str) -> Optional[Dict]:
    """Parse one concatenated sectional cell.

    Returns {"prefix": str, "time": float, "sub_splits": [float, ...]} or
    None for an empty/unparseable cell. The "{position}{margin}" prefix is
    left raw — splitting it needs the per-section permutation constraint
    (see _resolve_section_positions).
    """
    s = (cell or "").strip()
    if not s:
        return None
    times = _TIME_RE.findall(s)
    if not times:
        return None
    section_time = float(times[0])
    prefix = s[: s.index(times[0])]
    sub_splits = [float(t) for t in times[1:]]
    return {"prefix": prefix, "time": section_time, "sub_splits": sub_splits}


def _pos_margin_candidates(prefix: str) -> List[Tuple[int, str]]:
    """All valid (position, margin) interpretations of a raw prefix string."""
    cands: List[Tuple[int, str]] = []
    # 2-digit position (10-20 covers HK field sizes with headroom).
    if len(prefix) >= 2 and prefix[:2].isdigit():
        p2, m2 = int(prefix[:2]), prefix[2:]
        if 10 <= p2 <= 20 and _is_valid_margin(m2):
            cands.append((p2, m2))
    # 1-digit position.
    if prefix and prefix[0].isdigit():
        p1, m1 = int(prefix[0]), prefix[1:]
        if 1 <= p1 <= 9 and _is_valid_margin(m1):
            cands.append((p1, m1))
    return cands


def _resolve_section_positions(
    prefixes: List[str],
) -> List[Tuple[Optional[int], str]]:
    """Disambiguate (position, margin) for one section's cells.

    Uses the constraint that the N positions form a permutation of 1..N:
    candidates with a position outside 1..N are discarded (a position can't
    exceed the field size), unambiguous cells are assigned first, then
    ambiguous cells take whichever candidate position is still free. Cells
    that can't be resolved keep position=None and the raw prefix as margin.
    """
    n = len(prefixes)
    # Only positions valid for this field size survive — this alone resolves
    # most ambiguity (e.g. "11" in a 2-horse section can only be position 1).
    cand_lists = [
        [(pos, m) for pos, m in _pos_margin_candidates(p) if pos <= n]
        for p in prefixes
    ]
    resolved: List[Optional[Tuple[Optional[int], str]]] = [None] * len(prefixes)
    taken = set()
    # Pass 1: cells with exactly one interpretation.
    for i, cands in enumerate(cand_lists):
        if len(cands) == 1:
            resolved[i] = cands[0]
            taken.add(cands[0][0])
    # Pass 2: ambiguous cells — prefer the candidate whose position is free.
    for i, cands in enumerate(cand_lists):
        if resolved[i] is not None:
            continue
        pick = next((c for c in cands if c[0] not in taken), None)
        if pick is None:
            pick = cands[0] if cands else (None, prefixes[i])
        resolved[i] = pick
        if pick[0] is not None:
            taken.add(pick[0])
    return resolved  # type: ignore[return-value]


def _split_name_brand(raw: str) -> Tuple[str, str]:
    """'PODIUM\\xa0(H173)' -> ('PODIUM', 'H173')."""
    cleaned = (raw or "").replace("\xa0", " ").strip()
    m = _NAME_BRAND_RE.match(cleaned)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return cleaned, ""


def parse_sectional_html(
    html: str, race_date: str = "", race_no: int = 0
) -> Optional[Dict]:
    """Parse a displaysectionaltime page into structured per-horse data.

    Returns a dict (see module docstring) or None if the page has no
    sectional table (e.g. a no-meeting date or a race that hasn't run).
    """
    if not html:
        return None
    soup = BeautifulSoup(html, "html.parser")

    race_section_times: List[float] = []
    horse_rows: List[List[str]] = []

    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        if len(rows) < 2:
            continue
        first_cells = [c.get_text(strip=True) for c in rows[0].find_all(["td", "th"])]
        if not first_cells:
            continue
        head = first_cells[0]

        # Race-level table: "Sectional Time:" row holds per-section leader times.
        if head.startswith("Race Time"):
            for r in rows[1:]:
                cells = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
                if cells and cells[0].startswith("Sectional Time"):
                    for c in cells[1:]:
                        t = _TIME_RE.search(c)
                        if t:
                            race_section_times.append(float(t.group()))

        # Per-horse table: header starts with "Finishing Order".
        elif head.startswith("Finishing Order"):
            for r in rows[1:]:
                cells = [c.get_text(strip=True) for c in r.find_all(["td", "th"])]
                # Data rows lead with two integers (order, horse no).
                if len(cells) >= 4 and cells[0].isdigit() and cells[1].isdigit():
                    horse_rows.append(cells)

    if not horse_rows:
        return None

    # Each data row: [order, horse_no, name(brand), sec1..secK, final_time].
    # The middle slice (index 3 .. -1) is the up-to-6 section cells.
    parsed_horses = []
    raw_sections: List[List[str]] = []  # raw section cells, per horse
    for cells in horse_rows:
        section_cells = cells[3:-1]
        raw_sections.append(section_cells)
        name, brand = _split_name_brand(cells[2])
        parsed_horses.append({
            "finishing_order": int(cells[0]),
            "horse_no": int(cells[1]),
            "horse_name": name,
            "horse_brand": brand,
            "final_time": parse_finish_time(cells[-1]),
            "sections": [],
            "last_400m": None,
        })

    # Number of sections actually run = max non-empty section index across horses.
    num_sections = 0
    for sc in raw_sections:
        for idx, cell in enumerate(sc):
            if parse_sectional_cell(cell) is not None:
                num_sections = max(num_sections, idx + 1)

    # Parse + position-resolve one section at a time (the permutation
    # constraint is per-section, so all horses' cells for a section together).
    for sec_idx in range(num_sections):
        parsed_cells = []
        for horse_i in range(len(parsed_horses)):
            cell = raw_sections[horse_i][sec_idx] if sec_idx < len(raw_sections[horse_i]) else ""
            parsed_cells.append(parse_sectional_cell(cell))

        prefixes = [pc["prefix"] if pc else "" for pc in parsed_cells]
        pos_margins = _resolve_section_positions(prefixes)

        for horse_i, pc in enumerate(parsed_cells):
            if pc is None:
                continue
            position, margin = pos_margins[horse_i]
            parsed_horses[horse_i]["sections"].append({
                "section": sec_idx + 1,
                "position": position,
                "margin": margin,
                "time": pc["time"],
                "sub_splits": pc["sub_splits"],
            })

    # Derived: last_400m = the final run section's time (each section is 400m;
    # only the first section's length varies, so the last is always 400m).
    for h in parsed_horses:
        if h["sections"]:
            h["last_400m"] = h["sections"][-1]["time"]

    return {
        "race_date": race_date,
        "race_no": race_no,
        "num_sections": num_sections,
        "race_section_times": race_section_times,
        "horses": parsed_horses,
    }


def fetch_sectional_data(
    race_date: str, race_no: int, session: Optional[requests.Session] = None
) -> Optional[Dict]:
    """Fetch + parse one race's sectional data.

    `race_date` is the project-standard YYYY/MM/DD; it is converted to the
    DD/MM/YYYY the sectional URL expects. Returns the parsed dict, or None
    if the page is missing / has no sectional table.
    """
    try:
        yyyy, mm, dd = race_date.split("/")
    except ValueError:
        logger.warning(f"Bad race_date format (expected YYYY/MM/DD): {race_date!r}")
        return None

    url = f"{SECTIONAL_URL}?racedate={dd}/{mm}/{yyyy}&RaceNo={race_no}"
    sess = session or requests.Session()
    try:
        resp = sess.get(url, timeout=30, headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
        })
        if resp.status_code != 200:
            logger.info(f"Sectional page {race_date} R{race_no}: HTTP {resp.status_code}")
            return None
        return parse_sectional_html(resp.text, race_date, race_no)
    except Exception as e:
        logger.warning(f"Sectional fetch failed for {race_date} R{race_no}: {e}")
        return None
