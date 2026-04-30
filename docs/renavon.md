# Renavon Bulk Datasets — Detailed Reference

Companion to `CLAUDE.md`. Holds the 11-table inventory, indexes, backfill impact, rebuild sequence, and untapped feature candidates. CLAUDE.md keeps the one-line "frozen snapshot" warning + a pointer here.

## Background (2026-04-25 import)

After purchasing Renavon's **one-time** $49 HKJC History Export, 11 gzipped CSVs were ingested into dedicated `renavon_*` tables alongside our scraped data. The DB grew from ~4MB to ~614MB; **`tracking_data/*.db` is now gitignored**, so the database must be regenerated locally from the source CSVs in `renavon_Hongkong_Jockey_Club_races_and_horses/` (also gitignored).

## Frozen snapshot, not a subscription

**IMPORTANT**: All `renavon_*` tables end at 2026/04/22 and will never grow. For new race data, dividends, and rankings, use the HKJC scraper (`scripts/scrape.py`, `scripts/scrape.py --rankings`, `scripts/backfill_dividends.py`). Don't design features that assume continuous renavon_* updates — they won't arrive. Renavon's role is one-shot historical backfill; ongoing freshness comes from the scraper.

## Rebuild from scratch

```bash
python3 -m scripts.scrape --days 240            # results-page scrape (basic results)
python3 -m scripts.load_renavon_csvs            # 11 renavon_* tables
python3 -m scripts.backfill_card_fields         # backfill from saved race_data JSONs
python3 -m scripts.backfill_from_renavon        # backfill rating/last_6/gear/age/rating_change
python3 -m scripts.backfill_dividends           # real HKJC dividends
python3 -m scripts.train                        # retrain on now-feature-complete data
```

## 11-table inventory

| Table | Rows | Cols | What it has |
|-------|------|------|-------------|
| `renavon_race_results` | 368,330 | 84 | Per-horse-per-race: `entry_rating`, `gear`, `horse_age`, sectional times (s1-s6), splits 800m/600m/400m/final 200m, lbw per section, breeding (sire/dam/dam_sire), career stats, owner, declared horse weight |
| `renavon_horses` | 17,789 | 32 | Per horse: `current_rating`, `last_rating`, `start_of_season_rating`, `season_rating_change`, sex, colour, age, breeding, career win/place rates, total prize money, days_since_last_race, stable_location |
| `renavon_sectional_times` | 126,140 | 44 | Extended pace data: led_early, closed_strongly, consistent_position, wire_to_wire flags + per-section positions and lbw |
| `renavon_dividends` | 306,657 | 16 | Pool payouts 2001-present (50× more than our scraped `race_dividends`) |
| `renavon_race_catalog` | 30,959 | 27 | Race-level catalog with track conditions (penetrometer/hammer, track_going, distance_category, race_class, video_replay_url) |
| `renavon_pool_investments` | 1,646,369 | 19 | Pool investment snapshots over time (per pool per race per minute-to-post) |
| `renavon_jackpots` | 49,223 | 12 | Multi-race pool jackpots (Six Up, Triple Trio, etc.) with estimated dividends |
| `renavon_jockey_rankings` | 1,015 | 20 | Per-snapshot jockey rankings (career + season stats) |
| `renavon_trainer_rankings` | 1,091 | 20 | Per-snapshot trainer rankings |
| `renavon_racing_changes` | 443 | 18 | Late changes: scratches, jockey/gear/barrier swaps |
| `renavon_track_conditions` | 26 | 10 | Pure penetrometer/hammer readings per meeting day |

**Indexes**: `renavon_race_results` indexed on `(race_date, race_number)`, `horse_name`, `horse_id`. `renavon_horses` indexed on `horse_name`, `horse_id`. `renavon_sectional_times` indexed on `race_date`, `horse_name`.

## Backfill impact

`scripts/backfill_from_renavon.py` lifted card-field coverage in our existing `race_results` table:

| Field | Before | After |
|-------|--------|-------|
| rating | 5.0% | **98.5%** |
| last_6 | 4.8% | **94.8%** |
| gear | 4.4% | **52.1%** |
| age | 3.3% | **99.8%** |
| rating_change | 2.4% | **62.6%** |

## Future feature engineering candidates (untapped)

- `renavon_race_results.section_*_time` + per-section `position_*` for richer pace features (closer/leader/midfield over multiple sectionals)
- `renavon_horses.season_rating_change` for "in form vs out of form" signal independent of recent finishes
- `renavon_racing_changes` for late-breaking change alerts (gear changes, jockey swaps within 24h)
- `renavon_horses.career_stage` (Experienced / New / Returning) as a categorical feature
- `renavon_track_conditions.penetrometer_reading` + `hammer_reading` for objective ground hardness over the going string
