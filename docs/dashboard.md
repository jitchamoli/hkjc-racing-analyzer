# Dashboard — Detailed Reference

Companion to `CLAUDE.md`. Holds the v6.5 card-stack layout history, full BET TICKET prose, Top Bets sub-toggle behaviour, WhatsApp Copy spec, escape rules, results-merging mechanism, enhanced JSON fields, and Race Day Monitor. CLAUDE.md keeps the BET TICKET 5-layer summary, race-day workflow, the inline-handler `window.X` rule, and the atomic-write rule.

## Card-stack layout (v6.5)

`generate_predictions.py` generates a self-contained HTML dashboard with embedded race data as JSON. The venue is detected dynamically from race data. `results/racing_dashboard.html` is always overwritten with the latest run (date-stamped archive also saved).

Topbar with brand + **📋 Race Order / ⭐ Top Bets** view tabs → hero stack of full bet cards (one per race in Race Order, curated subset in Top Bets) → sticky alerts rail on ≥1024px.

Each card is **click-to-expand**: clicking anywhere on the card toggles a per-race detail panel inserted directly below it (per-horse table, **same-format history** of each horse's prior runs at this exact venue+distance, speed map, reasoning). Top-right chevron flips ▼ → ▲ and the card gets a gold ring when expanded. **Copy WA** button uses `event.stopPropagation` so it does not toggle the card. Each expanded panel also has a × close button in the top-LEFT (top-RIGHT was hidden behind the tier badge in earlier versions).

The legacy 11-tile strip + per-tile detail panel are still in the DOM (hidden by default) for backwards compatibility; the unified card stack supersedes them. Legacy A+/B+ grade system, `is_best_ev`/`is_worst_ev` ring highlights, and `strategy_confidence` string have all been removed. Empty-state honest-day message renders in Top Bets view when no race clears HIGH/MED tier.

## 5-layer BET TICKET (full prose)

Each card (and expanded panel) shows a 5-layer **BET TICKET**:

1. **PRIMARY** (gold pill) — the recommender's choice, each pick tagged with jockey + 30-day strike rate (gold) + all-time strike rate (dim).
2. **+ PROTECTION · BOX QIN-4** (green pill) — always-on top-4 model picks when there are 4+ ranked horses, even when the recommender chose a tighter primary. Catches placing-pair upsets where the winner sits in model ranks 3-4.
3. **+ F4 SAVER** (purple pill) — top-4 model picks placed in 1-2-3-4 exact order as a single-unit First-4 longshot saver.
4. **+ QIN PLACE BANKER** (blue pill) — only renders when top pick odds ≤ 1.8. For short-favorite races a straight WIN bet pays barely above stake; QPL on (favorite + 3 next picks) gives 3 combos that hit when the favorite places (very likely at sub-2.0) AND any banker also places. Validated on Apr 26 R9 ROMANTIC WARRIOR 1.4 — 2 of 3 QPL combos hit alongside the WIN.
5. **+ BOX TRIO-5** (pink pill) — top-5 model picks in a 10-combo TRIO box. Hits when any 3 of the 5 finish top-3 in any order. Catches the rank-5/6 spoilers that QIN-4 misses (validated by user's Apr 26 R10 manual play paying 12x; subsequent R11 hit cleanly on first auto-rendered iteration).

## Top Bets sub-toggle (Next / Past)

Inside the Top Bets tab, two more buttons (`Next` / `Past`) filter to upcoming or finished races. `Next` is default. Past races render with an outcome badge in the card header:
- 🟡 `★ STRIKE · WIN` — top model pick won
- 🟢 `★ STRIKE · QIN` — winning quinella pair was inside top-2 picks
- 🟢 `★ STRIKE · BOX-4` — winning quinella pair was inside top-4 picks
- 🔴 `✗ MISS (winner #X was rank Y)` — winning pair fell outside top-4

Badges render in BOTH views (Race Order shows them on finished races inline; Top Bets · Past filters to ONLY finished races for focused review). Determined client-side via `_didBetHit(race)` from each horse's `actual_position` field.

Surfaces upcoming HIGH/MED/EDGE picks (Next, default) or finished-race results with STRIKE/MISS badges (Past). Auto-flips to Past at end-of-meeting (when all races have `results_available=True`) so a click on Top Bets after the card is over lands on the day's record instead of an empty pane. Empty-state messages are case-aware: "All races finished — no next race today" with inline jump-links when post-meeting Next is hit, "No past races yet" pre-meeting Past, original "no bets meet threshold" copy mid-day.

## WhatsApp Copy

Mirrors the BET TICKET in plain text — first line is a scannable summary like `BET: BOX QIN-4 → 1, 9, 2, 13` followed by `F4 (exact): 1-9-2-13 · 1 unit` and `BOX TRIO-5: 1, 9, 2, 13, 8 · 10 combos`, then per-horse breakdown with jockey strike rates, then "Why strong" bullets.

## XSS / escape rules

**Attribute-context escape (`escAttr`)** is mandatory for any value rendered into an HTML attribute via string concatenation (`data-payload`, etc.). The text-context `esc()` does NOT escape `"` or `'` so values containing those would close the attribute prematurely. `escAttr` is implemented with a charcode-built regex (avoids quote-literal pairs that the JS-syntax test scanner misreads).

All scraped data (horse names, jockeys, trainers, reasoning) is HTML-escaped via a JS `esc()` helper before `innerHTML` injection (XSS protection). The JSON embedding sanitizes `</script>` tags to prevent script block injection. The WA Copy buttons use a `data-payload` attribute + `addEventListener` pattern (not inline `onclick`) to prevent attribute-injection on horse names containing apostrophes or backslashes.

## Inline-handler IIFE pattern

All click-handler functions (`showRaceDetail`, `closeRaceDetail`, `toggleCardDetail`, `setStripView`, `hardReload`, `refreshDashboard`, `copyWA`, `showInfoPop`, `_closeInfoPop`, `toggleExpandAll`) are explicitly assigned to `window.X = X` at the bottom of the IIFE. Without this, the IIFE wraps everything and inline `onclick=""` attributes throw "X is not defined". Add the same `window.X` line whenever introducing a new function called from inline HTML.

## Results merging into the dashboard

`dashboard/results_fetcher.py:_regenerate_dashboard` uses `generate_html()` to re-render from the merged-results JSON, then writes BOTH `results/racing_dashboard_{date}.html` (archive) AND `results/racing_dashboard.html` (always-latest, what the user views via file://). The legacy runtime-injection mechanism that patched `renderDashboard`/`renderRaceDetail` was REMOVED in v6.5 because the card-stack rewrite removed those helpers. Atomic write (tmp + `os.replace`) so a polling browser never sees a half-written file.

Plain-English Pure / Market / Edge labels with click-to-toggle ⓘ popovers (replaces native `title=""` tooltips so it works on touch). Topbar **⟳ Hard reload** button bypasses browser cache.

## Race-day live workflow

Recommended setup uses two persistent monitors:
- **Pre-race refresh + results merge** at **T-6 min** before each race (user explicitly bumped from T-5 to T-6 mid-day on Apr 26 so PLACE odds have time to publish). Each fire chains `generate_predictions.py` then `results_fetcher.py` so the dashboard has fresh win/place odds AND the past STRIKE/MISS badges persist after the regen.
- **Results poll + auto-learn** every 15 min via `python3 results_fetcher.py 'YYYY/MM/DD' --auto-learn` so each new race result ingests into `tracking_data/horse_history.db` (race_results table) and triggers an attempted retrain. The auto-learn scraper drops the LAST race of a card (end-of-card detection bug); on race day 2026-04-26 R11 was direct-ingested from `race_data_*_with_results.json` to recover the missing 4 horse-rows.

## Enhanced Dashboard JSON fields

JSON output now includes per-horse: `going_form`, `jockey_horse_stats`, `distance_pb`, `distance_record`, `odds_movement`, `steam_move`, `cold_horse`. Per-race: `speed_map`. Dashboard renders these as enhanced stats grid, market move badges (steam/drift/cold), and a visual speed map.

## Race Day Monitor

`python3 -m scripts.monitor '2026/04/16'` — polls odds 6 minutes before each race (every 5 min), snapshots to odds_history.db, detects steam/drift/cold horses, regenerates dashboard with auto-refresh (60s). Dashboard at `results/racing_dashboard.html` reloads automatically in browser.
