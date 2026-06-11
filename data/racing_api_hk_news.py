"""Normalises theracingapi.com HK racecards into HKJC-native format.

Our HK pipeline scrapes HKJC directly — this module is purely additive,
giving us expert prose (comment/spotlight), gear/medical flags, and
trainer 14-day form. Every field is converted to match the HKJC schema
emitted by core/fetcher.py (sample from results/race_data_*.json), so
the result can be merged into an HKJC race dict without further work.

API (UK convention)              → HKJC native
  off_dt BST datetime              → time '13:30' (HHMM HKT) + venue 'Sha Tin'/'Happy Valley'
  distance '1m4f' / '6f'           → distance '2400' / '1200' (string meters)
  course 'Sha Tin' / 'Happy Valley'→ venue 'Sha Tin' / 'Happy Valley' (passthrough)
  going 'Good'                     → going 'Good' (passthrough; HKJC titlecase matches)
  horse 'Grouper'                  → name 'GROUPER' (UPPERCASE)
  jockey 'James McDonald'          → jockey 'J McDonald' (first-initial + last)
  trainer 'D J Hall'               → trainer 'D J Hall' (passthrough)
  headgear 'tb' (UK letters)       → gear 'TT/B' (HKJC uppercase codes)
  ofr '40'                         → rating 40 (int)
  lbs '135'                        → weight '135' (string)
  number '1' / draw '4'            → number 1 / draw 4 (int)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from data.uk_racing_api import UKRacingAPI

_HKT = timezone(timedelta(hours=8))

# theracingapi distance strings → meters. Covers all HK distances; extend
# if a new one shows up (the constructor will fall back to 0 and log).
_DISTANCE_M = {
    '5f':   1000, '5.5f': 1100, '6f':   1200, '6.5f': 1300,
    '7f':   1400, '7.5f': 1500, '1m':   1600, '1m1f': 1800,
    '1m1.5f': 1900, '1m2f': 2000, '1m3f': 2200, '1m4f': 2400,
    '1m6f': 2800, '2m':   3200,
}

# HKJC stores the full venue name ('Sha Tin'/'Happy Valley') in race_data
# JSON, NOT the ST/HV code. Pass through after canonicalising whitespace.
_VENUE_CANONICAL = {
    'sha tin':      'Sha Tin',
    'happy valley': 'Happy Valley',
}

# UK theracingapi codes letters individually; HKJC uses multi-letter codes
# joined with '/'. Map each lowercase letter to its HKJC equivalent.
# Source for HKJC codes: HKJC race-card 'Gear' column legend.
_GEAR_LETTER_TO_HKJC = {
    'b':  'B',     # Blinkers
    'v':  'V',     # Visor
    'h':  'H',     # Hood
    't':  'TT',    # Tongue Tie
    'p':  'CP',    # Cheek Pieces (sheepskin)
    'e':  'E',     # Ear muff/plugs
    'x':  'XB',    # Cross-over noseband (rare)
}


def _norm_name(s: Optional[str]) -> str:
    """Upcase + strip. Match HKJC convention so callers can lookup by
    `horse_data['name']` directly."""
    return (s or '').upper().strip()


def _to_int(v, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _distance_to_meters(s: Optional[str]) -> int:
    if not s:
        return 0
    return _DISTANCE_M.get(s.strip().lower(), 0)


def _venue_canonical(s: Optional[str]) -> str:
    """HKJC native: full name 'Sha Tin' / 'Happy Valley' (NOT ST/HV code)."""
    return _VENUE_CANONICAL.get((s or '').strip().lower(), (s or '').strip())


def _shorten_jockey(name: Optional[str]) -> str:
    """Convert API 'James McDonald' → HKJC 'J McDonald'.

    HKJC convention is first-initial + last-name with native casing
    preserved ('J McDonald', 'Z Purton', 'A Atzeni'). Only collapse the
    first word when there are exactly 2 space-separated words AND the
    first word is a plausible first name (no dots, length >1). Otherwise
    pass through — handles 'K H Chan', 'A Hamelin', and single names.
    """
    s = (name or '').strip()
    if not s:
        return ''
    parts = s.split()
    if len(parts) == 2 and len(parts[0]) > 1 and '.' not in parts[0]:
        return f"{parts[0][0]} {parts[1]}"
    return s


def _gear_to_hkjc(api_code: Optional[str]) -> str:
    """Convert UK lowercase combo 'tb' → HKJC 'TT/B'. Unknown letters drop."""
    s = (api_code or '').strip().lower()
    if not s:
        return ''
    pieces = [_GEAR_LETTER_TO_HKJC[c] for c in s if c in _GEAR_LETTER_TO_HKJC]
    return '/'.join(pieces)


@dataclass
class HKHorseNews:
    # HKJC-native fields (match keys in results/race_data_*.json)
    name: str                       # 'GROUPER'  (UPPERCASE)
    number: Optional[int]           # 4
    draw: Optional[int]             # 4
    jockey: str                     # 'J McDonald'  (HKJC short form)
    trainer: str                    # 'D J Hall'
    gear: str                       # 'TT/B'  (HKJC codes joined by '/')
    weight: str                     # '135'  (carried weight, lbs, string)
    rating: Optional[float]         # 40.0  (Official Rating; HKJC stores as float)
    age: str                        # '4'

    # News-only additive fields (no HKJC equivalent)
    headgear_first_time: bool
    wind_surgery_first_time: bool
    medical_notes: list = field(default_factory=list)
    comment: str = ''               # short expert one-liner
    spotlight: str = ''             # longer analysis
    trainer_14d_runs: int = 0
    trainer_14d_wins: int = 0
    trainer_14d_pct: float = 0.0

    @property
    def flags(self) -> list:
        out = []
        if self.gear and self.headgear_first_time:
            out.append(f"1ST {self.gear}")
        if self.wind_surgery_first_time:
            out.append("POST-WIND-SURGERY 1ST")
        if self.medical_notes:
            out.append(f"MEDICAL: {self.medical_notes}")
        return out

    def as_hkjc_dict(self) -> dict:
        """Emit dict in the same shape as HKJC race_data_*.json horse rows.
        News-only fields are nested under `news` so they don't collide."""
        return {
            'name':     self.name,
            'number':   self.number,
            'draw':     self.draw,
            'jockey':   self.jockey,
            'trainer':  self.trainer,
            'gear':     self.gear,
            'weight':   self.weight,
            'rating':   self.rating,
            'age':      self.age,
            'news': {
                'comment':              self.comment,
                'spotlight':            self.spotlight,
                'flags':                self.flags,
                'trainer_14d_runs':     self.trainer_14d_runs,
                'trainer_14d_wins':     self.trainer_14d_wins,
                'trainer_14d_pct':      self.trainer_14d_pct,
                'headgear_first_time':  self.headgear_first_time,
                'wind_surgery_first_time': self.wind_surgery_first_time,
                'medical_notes':        self.medical_notes,
            },
        }


@dataclass
class HKRaceNews:
    # HKJC-native fields
    race_number: int                # 1..N (HKJC card order)
    time: str                       # '13:30'  (HHMM HKT, string)
    venue: str                      # 'Sha Tin' / 'Happy Valley'
    distance: str                   # '1200'  (string meters, HKJC convention)
    going: str                      # 'Good'
    race_name: str
    race_class: str
    field_size: int

    # Original BST datetime kept for ordering / debugging only
    off_hkt: datetime
    horses: List[HKHorseNews]

    # Editorial overlay (theracingapi PRO). Optional — consumed by the
    # confluence snapshot (tip_match + morning line parsed from forecast).
    betting_forecast: str = ''
    tip: str = ''
    verdict: str = ''

    def find(self, horse_name: str) -> Optional[HKHorseNews]:
        target = _norm_name(horse_name)
        return next((h for h in self.horses if h.name == target), None)

    def as_hkjc_dict(self) -> dict:
        """Emit dict matching HKJC race_data_*.json race shape."""
        return {
            'race_number':  self.race_number,
            'time':         self.time,
            'venue':        self.venue,
            'distance':     self.distance,
            'going':        self.going,
            'race_name':    self.race_name,
            'race_class':   self.race_class,
            'field_size':   self.field_size,
            'horses':       [h.as_hkjc_dict() for h in self.horses],
        }

    def merge_into_race(self, race: dict) -> dict:
        """Mutate an HKJC race dict by attaching news to each horse (matched
        by name). HKJC fields untouched — pure additive. Returns the same
        dict for chaining. Horses with no matching news get news=None."""
        for h in race.get('horses', []):
            news = self.find(h.get('name', ''))
            h['news'] = news.as_hkjc_dict()['news'] if news else None
        return race


def _parse_runner(r: dict) -> HKHorseNews:
    t14 = r.get('trainer_14_days') or {}
    return HKHorseNews(
        name=_norm_name(r.get('horse')),
        number=_to_int(r.get('number')),
        draw=_to_int(r.get('draw')),
        jockey=_shorten_jockey(r.get('jockey')),
        trainer=(r.get('trainer') or '').strip(),
        gear=_gear_to_hkjc(r.get('headgear')),
        weight=str(r.get('lbs') or '').strip(),
        rating=_to_float(r.get('ofr'), None) if r.get('ofr') not in (None, '', '-') else None,
        age=str(r.get('age') or '').strip(),
        headgear_first_time=(r.get('headgear_run') == '0'
                             and bool(r.get('headgear'))),
        wind_surgery_first_time=(r.get('wind_surgery_run') == '0'
                                 and bool(r.get('wind_surgery'))),
        medical_notes=list(r.get('medical') or []),
        comment=(r.get('comment') or '').strip(),
        spotlight=(r.get('spotlight') or '').strip(),
        trainer_14d_runs=_to_int(t14.get('runs'), 0) or 0,
        trainer_14d_wins=_to_int(t14.get('wins'), 0) or 0,
        trainer_14d_pct=_to_float(t14.get('percent')),
    )


def _parse_card(c: dict) -> HKRaceNews:
    off_dt_str = c.get('off_dt') or ''
    try:
        off_hkt = datetime.fromisoformat(off_dt_str).astimezone(_HKT)
    except ValueError:
        off_hkt = datetime.min.replace(tzinfo=_HKT)
    runners = [_parse_runner(r) for r in (c.get('runners') or [])]
    distance_m = _distance_to_meters(c.get('distance'))
    return HKRaceNews(
        race_number=0,              # filled in by caller after sort
        time=off_hkt.strftime('%H:%M'),
        venue=_venue_canonical(c.get('course')),
        distance=str(distance_m) if distance_m else '',
        going=(c.get('going') or '').strip(),
        race_name=(c.get('race_name') or '').strip(),
        race_class=(c.get('race_class') or '').strip(),
        field_size=_to_int(c.get('field_size'), len(runners)) or len(runners),
        off_hkt=off_hkt,
        horses=runners,
        betting_forecast=(c.get('betting_forecast') or '').strip(),
        tip=(c.get('tip') or '').strip(),
        verdict=(c.get('verdict') or '').strip(),
    )


def fetch_hk_news(
    date_yyyy_mm_dd: Optional[str] = None,
    api: Optional[UKRacingAPI] = None,
    tier: str = 'pro',
) -> List[HKRaceNews]:
    """Fetch + normalise today's HK racecards.

    Returns races in HKJC race-order (1..N) by HKT off-time. The API
    `/racecards` endpoint only returns today + tomorrow; if a specific
    date is passed, races outside that date are filtered out.
    """
    api = api or UKRacingAPI()
    cards = api.get_racecards(tier=tier)
    hk = [c for c in cards if (c.get('region') or '').upper() == 'HK']
    races = [_parse_card(c) for c in hk]
    if date_yyyy_mm_dd:
        # Callers pass HKJC-convention slash dates ('2026/06/10'); off_hkt
        # formats with dashes. Normalise so the comparison can ever match.
        want = date_yyyy_mm_dd.replace('/', '-')
        races = [r for r in races if r.off_hkt.strftime('%Y-%m-%d') == want]
    races.sort(key=lambda r: r.off_hkt)
    for i, r in enumerate(races, 1):
        r.race_number = i
    return races
