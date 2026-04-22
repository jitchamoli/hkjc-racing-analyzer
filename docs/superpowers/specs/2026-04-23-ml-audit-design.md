# ML + Probability Audit — Design Spec

**Date:** 2026-04-23
**Trigger:** Underperformance on Apr 19 (Sun, longshot-heavy) and Apr 22 (Wed, Happy Valley) race meetings. User requested a probability-expert + data-scientist review of the full pipeline.
**Status:** Design approved, pending user review before plan-writing.

---

## Goal

Produce a prioritized fix list derived from a rigorous audit of the HKJC racing analyzer's ML, probability, and betting-recommendation pipeline. The audit must operate under two independent expert lenses (probability theory and data science) and produce findings grounded in code citations and race-level evidence. The audit **does not** apply fixes — fixes are scoped into a subsequent plan that the user approves separately.

## Non-Goals

- No code edits during audit. Investigation is strictly read-only.
- Dashboard visual/UX redesign is a separate workstream to be brainstormed after this audit completes.
- Not a full ML re-architecture — scope is "audit existing pipeline for correctness and statistical soundness," not "design v2."

## Stages

### Stage 1 — Forensic (single-threaded, main agent)

**Evidence set:** Full 71-race holdout from `scripts/backtest.py`, with particular attention on Apr 22 races if present (or adjacent races if held-out window excludes Apr 22). Apr 19 is referenced only as already-addressed baseline and is revisited only if Stage 2 surfaces a systematic issue to cross-check against it.

**Probes:**
- For every holdout race where rank-1 finished 4th or worse, pull full feature vector, ML probability, rule score, blend weights used, and final calibrated rank
- For every holdout race where both rank-1 and rank-2 finished outside top-3, same trace
- For every holdout race where the winning pair was 5+ combinations deep in our ranking, same trace
- Identify whether misses cluster by layer: feature-driven (bad feature value), calibration-driven (Platt/softmax), or blend-driven (market vs ML weight)
- Cross-reference Apr 22 specifically: was it a regime the model has seen before (Wed HV Class 4-5), or was it out-of-distribution?

**Output:** `docs/audits/2026-04-23/stage1-forensic.md` — race-level failure modes tagged to suspected pipeline layer, each with file:line citations and numeric evidence.

### Stage 2 — Systematic (two parallel subagents, main agent merges)

Both subagents receive Stage 1 findings, full codebase read access, and an explicit guardrail: **read-only, no edits, produce findings with file:line citations and severity ratings.**

**Probability-expert subagent scope:**
- Platt scaling: validation-set size, recency-weight consistency with XGBoost, test-metric unbiasedness
- Base-rate sigmoid (`rule_score → prob` anchored at `1/field_size`): slope of 0.06 — justified or arbitrary? Does it produce sensible spreads across realistic score ranges?
- Softmax temperature (1.5): provenance, tuning, over/under-spreading risk
- Dynamic market blend (35-65% by field size/distance): empirical basis for race classification
- Harville independence assumption in quinella/trifecta pricing: correlation failure for pace pairings; is the placer boost a patch over a model limitation?
- Calibration curve: 5-bin report — does it mask tail miscalibration (favourites vs longshots)?
- Kelly or bet-sizing logic (if present)
- `qin_bet_ev`: internal consistency of using model probs for hit-rate and market probs for dividend

**Data-scientist subagent scope:**
- **First task:** Verify what code path `scripts/backtest.py` actually exercises — which model pickle, which recommender logic, whether reported ROI reflects today's box-4 strategy or a prior version. If stale, flag as top-priority Stage 2 finding and caveat all downstream conclusions.
- Feature leakage: every feature in `extract_features()` and `build_training_data_from_db()` — any sneaks in post-race data?
- Train/predict consistency: does every feature compute identically in training vs inference? (CLAUDE.md flags as known risk.)
- Holdout integrity: 60/20/20 temporal split truly temporal? Validation-set Platt fit contaminating test metrics?
- Sample weighting (recency decay 1.0 → 0.3): justified or arbitrary? Last-month overfit risk?
- Feature correlation / near-constants: any of 23 features still near-constant or >0.7-correlated?
- Par-time lookups for adjusted speed: date-filtering correct (CLAUDE.md says yes, verify)?
- `rebuild_stats()` overwriting HKJC rankings: is this silently happening during ingestion?
- ML fallback (`predict()` returning `1/field_size` when untrained): ever firing silently in production?
- Backtest methodology integrity: dividend date-format matching, holdout overlap, metric definitions

**Disagreement protocol:** If personas cover the same area with different severity, preserve both opinions in merge notes and flag for user arbitration. If personas return 80%+ overlapping findings, flag that as signal.

**Output:** `docs/audits/2026-04-23/stage2-systematic.md` with three sections: `## Probability-Expert Findings`, `## Data-Scientist Findings`, `## Merge Notes` (disagreements, overlaps, cross-cutting observations).

### Stage 3 — Strategic (main agent, synthesizes 1+2)

**Critique targets:**
- Quinella-first + box-4 default: is it defensible given 71-race sample size (12 straight-Q hits, 24 box-3 hits)? What are the 95% CIs on the reported ROI? Is the pivot overfit to 1-2 race days' evidence?
- "Relative EV" ranking across races: actionable signal, or noise given we're always in negative absolute EV territory?
- Rank-2 calibration audit: was the conclusion (box-4 captures longshot-2nds) statistically sound, or was it reading a pattern into 1-2 days of data?
- Sample-size reality check: what's the minimum N needed to narrow current CIs to ±5 percentage-point ROI? (rough estimate)

**Acknowledged outcome:** A possible strategic conclusion is "pivot to box-4 was made on statistically insufficient evidence; pause and collect more data before further changes." User has confirmed this is an acceptable conclusion and should still produce a fix list — where "collect more data, don't change strategy" becomes an explicit fix-list entry.

**Output:** Folded into `docs/audits/2026-04-23/final-report.md` along with the consolidated fix list.

## Final Deliverable

`docs/audits/2026-04-23/final-report.md` containing:
1. Executive summary (<1 page)
2. Consolidated findings, prioritized by severity × tractability
3. Strategic-layer critique of current betting approach
4. **Fix list** — each entry with:
   - Finding ID (e.g., `P-03` for Probability finding 3, `DS-07` for Data-Science 7, `ST-02` for Strategic 2)
   - Severity: Critical / High / Medium / Low / Informational
   - Tractability: `<1h` / `<1d` / `redesign` / `needs-data`
   - One-line description
   - Pointer to source finding in stage reports
   - "What success looks like" criterion (how we know the fix worked)

## Guardrails

- Subagents are **read-only**: no edits, no Writes except their findings file
- All findings must cite file:line and, where possible, a specific race or training example
- If Stage 1 forensic pass finds a clear smoking gun that invalidates Stage 2 before it runs (e.g., model is effectively broken), main agent pauses and brings to user rather than continuing
- Severity ratings defined explicitly at top of Stage 2 doc
- Audit itself is time-boxed loosely; if a subagent is clearly thrashing, main agent cuts it short and notes the limitation

## Process After Audit

1. User reviews `final-report.md`
2. Separate brainstorm cycle scopes the fix-implementation (some fixes may be deferred, combined, or rejected)
3. That brainstorm terminates in `writing-plans` invocation producing a detailed fix-implementation plan
4. Fix plan executes in a fresh session
5. Dashboard visual/UX redesign is a third, separate workstream brainstormed after fixes are complete

## Open Questions (resolved during brainstorm)

- Q: Sequencing of ML audit vs dashboard redesign? → **A:** ML audit first (determines data/signals dashboard should surface)
- Q: Audit motivation — forensic, systematic, strategic, or all? → **A:** All three, in order
- Q: Any preloaded hypotheses? → **A:** No — treat as "below expectations, find why"
- Q: Deliverable form — report only, report+fixes, report+fix plan, fix as we go? → **A:** Report + separate fix plan (implemented as writing-plans cycle after audit)
- Q: Evidence set — recent days, recent weeks, full holdout, or combo? → **A:** Full 71-race holdout
- Q: Scope of pipeline — features, probability only, full pipeline, full+backtest? → **A:** Full pipeline + backtest methodology itself
- Q: Approach — single-threaded, parallel subagents, or sequential handoff? → **A:** Parallel subagents (Approach 2)

## Known Limitations

- 71-race holdout is statistically thin. Any ROI conclusion has wide CIs. This is itself a finding, not a flaw in the audit.
- The two personas will have some unavoidable overlap (e.g., calibration quality is both a stats and a probability question). Merge Notes will handle.
- Stage 1 forensic may not find dramatic failures — the rough days may be within statistical norms, in which case Stage 2 systematic review is the more productive phase.
- Apr 22 may fall outside the holdout window if the split is chronological-pre-Apr-22. If so, forensic will probe it against the current production model separately from the holdout.
