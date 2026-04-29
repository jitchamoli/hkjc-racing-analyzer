# CLAUDE.md Trim — Design

**Date**: 2026-04-29
**Status**: Approved (in implementation)

## Problem

`CLAUDE.md` has grown to **459 lines / ~41 KB**. It mixes three kinds of content:

1. **Operational rules** Claude must follow on every session (commands, conventions, ML critical rules, dashboard inline-handler rule, etc.)
2. **Architecture summaries** of current state (database tables, probability pipeline, BET TICKET layers)
3. **Historical/audit narrative** — version-by-version changelog (v6.0 → v6.5), feature drop history, backtest ROI tables across releases, Renavon table inventory, training pipeline tuning history

The third kind dilutes the first two. Future Claude sessions read all 459 lines on every turn; rules get buried.

## Goal

Trim `CLAUDE.md` to **~140 lines** of operational essentials + pointers. Move historical/detail content to topic-organised files under `docs/`. Lose no information — every fact in the current file lands somewhere accessible.

## What stays in `CLAUDE.md`

| Section | Keep what |
|---------|-----------|
| Commands | All commands, compacted formatting |
| Architecture / Package structure | The 8-line code block |
| Architecture / Data Pipeline | The diagram |
| Architecture / Scoring System | One short paragraph; pointer to `docs/ml-model.md` |
| Architecture / Database | Table list + integrity rules + auto-computed fields |
| Architecture / Data Fetching | Three-method fallback + venue detection + parser brittleness |
| Architecture / Dashboard | BET TICKET 5-layer summary, race-day workflow, inline-handler `window.X` rule, atomic-write rule, XSS escape rules |
| Architecture / Probability Pipeline | The diagram block + the "edge math uses pure_prob" rule |
| Codebase Conventions | All 7 bullets verbatim |
| ML Critical Rules | All 8 numbered rules verbatim |
| DB Data Gaps | The bullet list (current state) |
| Renavon (one-line warning) | "Frozen snapshot, not a subscription" + pointer |
| Pointers | Links to the 4 `docs/*.md` files |

## What moves to `docs/`

### `docs/ml-model.md`
- Current 19 feature list (groups 1–8, 9–16, 17–19)
- Feature importance table (v6.5 50K-row training)
- Removed-features changelog (v6.4, Wave 2c, earlier audits)
- `extract_class_rank()` mapping table (Group 1/2 → 0, Class 1–5 → 1–5, etc.)
- Training pipeline detail: 5-year window rationale, 60/20/20 split, exponential recency weighting math, field clipping fix, freshness fallback fix, isotonic calibration, scale_pos_weight, experiment_log integrity fix
- Speed Figures DEPRECATED note

### `docs/dashboard.md`
- Card-stack layout v6.5 detail (click-to-expand, chevron, close-button history)
- 5-layer BET TICKET full prose (PRIMARY, BOX QIN-4, F4 SAVER, QIN PLACE BANKER, BOX TRIO-5) — short summary stays in CLAUDE.md
- Top Bets sub-toggle (Next / Past) behavior + auto-flip + empty-state copy
- WhatsApp Copy format spec
- `escAttr` rationale and the `window.X` IIFE pattern
- Results merging mechanism (`_regenerate_dashboard`)
- Enhanced Dashboard JSON fields list
- Race Day Monitor

### `docs/betting-strategy.md`
- Edge Math + Conviction Tiers v6.0 detail (per-horse fields, per-race fields)
- 6-gate EDGE filter table + sample-size note (Corr -0.21)
- Recommender logic (BOX Q4 / BOX Q / QIN / TRIO / WIN gates) — full rules
- Dynamic market/model blend weights + grid-sweep selection (`base_model_w=0.45`)
- Backtest history (v6.1 / v6.2 / v6.3 / v6.4 / v6.5 ROI table)
- Edge correlation note (still negative on disagreements)
- Exotic bet calculator pricing (exacta / quinella / trifecta / first4 conditional elimination)
- Backtest reproduction commands

### `docs/renavon.md`
- "Frozen snapshot, not a subscription" full warning
- 11-table inventory (rows × cols × content)
- Indexes
- Backfill impact table (rating 5%→98.5% etc.)
- Rebuild-from-scratch command sequence
- Future feature engineering candidates

## Pointer convention

For each section that gets trimmed, leave a one-line link in CLAUDE.md:

> Full feature importance & removed-features changelog: `docs/ml-model.md`
> Dashboard layout history & BET TICKET rationale: `docs/dashboard.md`
> Backtest history & exotic pricing math: `docs/betting-strategy.md`
> Renavon table inventory & rebuild instructions: `docs/renavon.md`

## Implementation order

1. Create the 4 `docs/*.md` files with moved content (verbatim — no rewriting; preserve the truth as currently stated).
2. Rewrite `CLAUDE.md` with kept-content + pointers.
3. Verify nothing trimmed is referenced from code or other docs (grep for distinctive phrases).
4. Stage but do **not** commit — leave the diff for user review.

## Non-goals

- No fact-checking the historical content (e.g., not re-validating "v6.4 Brier 0.108" — copy as-is).
- No code changes.
- No re-ordering or re-phrasing of kept content beyond compaction.
