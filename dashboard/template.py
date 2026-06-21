#!/usr/bin/env python3
"""HKJC dashboard HTML generator — edge-first layout.

Renders: top bar, hero (0-5 bet cards with trust cards), 11-race strip,
race detail (click to expand), sticky alerts rail on desktop/iPad landscape.

Legacy grade system (A+/A/B+/B/B-), strategy_confidence, is_best_ev,
and combined_score are NOT rendered. See spec §3 for IA rules and §5
for field hygiene verdicts.
"""

TIER_COLORS = {
    'EDGE': '#fbbf24', 'HIGH': '#10b981', 'MED': '#f59e0b',
    'LOW': '#64748b', 'SKIP': '#475569', 'CHAOS': '#dc2626',
}

ALERT_COLORS = {
    'EDGE_OPENED': '#fbbf24', 'EDGE_WIDENED': '#fbbf24',
    'EDGE_CLOSED': '#94a3b8', 'STEAM': '#ef4444',
    'COLD': '#06b6d4', 'RACE_UPCOMING': '#10b981',
}


def generate_html(races_json: str, now: str, race_date: str,
                  venue: str = 'Sha Tin', auto_refresh: bool = False,
                  accuracy_summary: dict = None,
                  edge_stats: dict = None) -> str:
    import json as _json
    import html as _html

    # XSS defense — strip </script> from JSON payload
    races_json_safe = races_json.replace('</script>', r'<\/script>') \
                                .replace('</Script>', r'<\/Script>')

    auto_refresh_meta = '<meta http-equiv="refresh" content="60">' if auto_refresh else ''

    # Server-side render strip tiles so R1..R11 appear in static HTML
    # (required for test_strip_shows_all_11_races and fast first paint)
    try:
        races_list = _json.loads(races_json)
    except (ValueError, TypeError):
        races_list = []

    strip_tiles_html = _build_strip_tiles(races_list)
    accuracy_panel_html = _build_accuracy_panel(accuracy_summary or {})
    # JSON-encoded for the JS layer (WhatsApp copy footer reads from it).
    # Match the same defense-in-depth pattern as races_json_safe — strip both
    # casings of </script> to avoid breaking out of the <script> block.
    accuracy_summary_json = _json.dumps(accuracy_summary or {}) \
        .replace('</script>', r'<\/script>') \
        .replace('</Script>', r'<\/Script>')

    # EDGE stats (n_fires / n_placed / n_won) — fed by scripts/track_edge.py.
    # Banner text in WA + dashboard reads from this so the count stays live
    # instead of hardcoded. Defaults to empty when stats file missing.
    edge_stats_json = _json.dumps(edge_stats or {}) \
        .replace('</script>', r'<\/script>') \
        .replace('</Script>', r'<\/Script>')

    return _HTML_TEMPLATE.format(
        races_json=races_json_safe,
        now=now,
        race_date=race_date,
        venue=venue,
        auto_refresh_meta=auto_refresh_meta,
        strip_tiles_html=strip_tiles_html,
        accuracy_panel_html=accuracy_panel_html,
        accuracy_summary_json=accuracy_summary_json,
        edge_stats_json=edge_stats_json,
    )


def _wilson_ci_pct(hits: int, n: int, z: float = 1.96):
    """Wilson 95% CI as (lo%, hi%); None if n=0. Mirrors scripts/accuracy.py."""
    if n <= 0:
        return None
    import math as _math
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * _math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - half) * 100, min(1.0, center + half) * 100)


def _build_accuracy_panel(summary: dict) -> str:
    """Render two-dimension accuracy panel (ML Confidence + Conviction Tier).

    Pulls from db.get_accuracy_summary(60). Hidden cleanly when summary is
    empty or in 'no results' state.
    """
    if not summary or 'message' in summary:
        return ''  # silent when no results yet

    tier_order = {'EDGE': 0, 'VERY_HIGH': 1, 'HIGH': 2, 'MEDIUM': 3, 'MED': 3,
                  'LOW': 4, 'UNKNOWN': 5}

    def _render_table(title: str, by_tier: dict) -> str:
        if not by_tier:
            return ''
        rows_html = []
        for conf, data in sorted(by_tier.items(),
                                  key=lambda kv: tier_order.get(kv[0], 99)):
            if data.get('total', 0) == 0:
                continue
            bets = data.get('bets', 0)
            hits = data.get('hits', 0)
            profit = data.get('profit', 0.0)
            stake = data.get('stake', 0.0)
            if bets > 0 and stake > 0:
                hit_pct = hits / bets * 100
                roi = profit / stake * 100
                ci_str = ''
                if bets < 30:
                    ci = _wilson_ci_pct(hits, bets)
                    if ci:
                        ci_str = ' <span style="color:var(--dim);font-size:.7rem">[CI {:.0f}-{:.0f}%]</span>'.format(ci[0], ci[1])
                roi_color = 'var(--green)' if roi >= 0 else 'var(--red)'
                rows_html.append(
                    '<tr><td>{conf}</td><td>{bets}</td><td>{hits}</td>'
                    '<td>{hit_pct:.1f}%{ci}</td>'
                    '<td>${profit:+.0f}</td>'
                    '<td style="color:{roi_color}">{roi:+.1f}%</td></tr>'.format(
                        conf=_esc(conf), bets=bets, hits=hits,
                        hit_pct=hit_pct, ci=ci_str,
                        profit=profit, roi=roi, roi_color=roi_color,
                    )
                )
            else:
                rows_html.append(
                    '<tr><td>{conf}</td><td>{bets}</td><td>—</td>'
                    '<td>—</td><td>—</td><td>—</td></tr>'.format(conf=_esc(conf), bets=bets)
                )
        if not rows_html:
            return ''
        return (
            '<div style="margin-top:8px">'
            '<div style="font-size:.75rem;color:var(--dim);margin-bottom:4px">{title}</div>'
            '<table class="acc-table"><thead><tr>'
            '<th>tier</th><th>bets</th><th>hits</th><th>hit%</th><th>P&amp;L (1u)</th><th>ROI</th>'
            '</tr></thead><tbody>{rows}</tbody></table>'
            '</div>'
        ).format(title=_esc(title), rows=''.join(rows_html))

    total_races = summary.get('total_races', 0)
    total_profit = summary.get('total_profit', 0.0)
    roi = summary.get('roi_percent', 0.0)
    bets_placed = summary.get('bets_placed', 0)
    bets_won = summary.get('bets_won', 0)
    period_days = summary.get('period_days', 60)
    roi_color = 'var(--green)' if roi >= 0 else 'var(--red)'

    ml_table = _render_table('By ML Confidence', summary.get('by_confidence', {}))
    conv_table = _render_table('By Conviction Tier (strategy edge)',
                                summary.get('by_conviction', {}))

    return (
        '<details class="accuracy-panel" style="margin:10px 0;background:var(--card);'
        'border:1px solid var(--border);border-radius:8px;padding:10px">'
        '<summary style="cursor:pointer;font-weight:600;font-size:.85rem">'
        'Recent accuracy — last {days}d · {races} races · '
        '<span style="color:{roi_color}">${profit:+.0f} ({roi:+.1f}% ROI)</span> · '
        '{hits}/{bets} bets hit</summary>'
        '<div style="font-size:.75rem;color:var(--dim);margin-top:6px">'
        '1u-stake terms (WIN 1u, BOX Q3 3u, BOX Q4 6u). Wilson 95% CI shown when N&lt;30.'
        '</div>'
        '{ml_table}{conv_table}'
        '</details>'
    ).format(
        days=period_days, races=total_races, hits=bets_won, bets=bets_placed,
        profit=total_profit, roi=roi, roi_color=roi_color,
        ml_table=ml_table, conv_table=conv_table,
    )


def _esc(s) -> str:
    """HTML-escape a value for server-side rendering."""
    import html as _html
    return _html.escape(str(s) if s is not None else '')


def _build_strip_tiles(races_list: list) -> str:
    """Pre-render strip tiles so race numbers appear in static HTML."""
    tiles = []
    for r in races_list:
        num = r.get('race_number', '?')
        tier = r.get('conviction_tier') or 'SKIP'
        if tier not in ('EDGE', 'HIGH', 'MED', 'LOW', 'SKIP', 'CHAOS'):
            tier = 'SKIP'
        label = 'R' + str(num) + (' ★' if tier == 'EDGE' else
                                   ' ⚠' if tier == 'CHAOS' else '')
        horses = r.get('horses') or []
        sub = ''
        if tier == 'CHAOS':
            sub = 'no play'
        elif horses:
            bet = horses[0].get('bet_recommendation', '') or ''
            sub = 'skip' if tier == 'SKIP' else bet
        tiles.append(
            '<div class="strip-tile tier-{tier}" id="tile-{num}"'
            ' onclick="showRaceDetail({num})" title="Click to expand · {tier}">'
            '<div class="strip-num">{label}</div>'
            '<div class="strip-sub">{sub}</div>'
            '<div class="strip-chevron">▼</div>'
            '</div>'.format(
                tier=_esc(tier),
                num=_esc(num),
                label=_esc(label),
                sub=_esc(sub),
            )
        )
    return '\n'.join(tiles)


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
{auto_refresh_meta}
<title>HKJC Racing Dashboard — {race_date}</title>
<style>
:root {{
  --bg:#0f172a; --panel:#1e293b; --panel2:#162032; --border:#334155;
  --fg:#e2e8f0; --dim:#94a3b8;
  --gold:#fbbf24; --green:#10b981; --amber:#f59e0b; --red:#ef4444;
  --cyan:#06b6d4; --violet:#8b5cf6;
}}
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:ui-sans-serif,system-ui,sans-serif; background:var(--bg); color:var(--fg); min-height:100vh; }}
button {{ font-family:inherit; cursor:pointer; }}

/* ── Top bar ── */
.topbar {{
  display:flex; justify-content:space-between; align-items:center;
  padding:12px 20px; border-bottom:2px solid var(--gold);
  background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 40%,#312e81 100%);
  position:sticky; top:0; z-index:100; flex-wrap:wrap; gap:8px;
}}
.topbar-brand {{ font-size:1.4rem; font-weight:900;
  background:linear-gradient(90deg,#fbbf24,#f59e0b,#d97706);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }}
.topbar-meta {{ font-size:.75rem; color:var(--dim); }}
.live-dot {{ width:7px; height:7px; border-radius:50%; background:var(--green);
  animation:pulse 2s infinite; display:inline-block; margin-right:5px; }}
@keyframes pulse {{ 0%,100% {{ opacity:1; }} 50% {{ opacity:.4; }} }}

/* ── Shell layout ── */
.shell {{
  display:grid;
  grid-template-columns:1fr 280px;
  grid-template-rows:auto;
  gap:16px;
  padding:16px;
  max-width:1440px;
  margin:0 auto;
}}
.main-col {{ min-width:0; }}
.alerts-rail {{
  position:sticky; top:62px; align-self:start;
  background:rgba(15,23,42,.5); border:1px solid var(--border); border-radius:10px;
  padding:12px; max-height:calc(100vh - 80px); overflow-y:auto;
}}
@media (max-width:1024px) {{
  .shell {{ grid-template-columns:1fr; }}
  .alerts-rail {{ position:static !important; top:auto; max-height:none; }}
}}

/* ── Hero ── */
.hero {{ margin-bottom:12px; }}
.hero-label {{
  font-size:.6rem; letter-spacing:.12em; color:var(--dim);
  text-transform:uppercase; margin-bottom:8px;
}}
.bet-card {{
  border-radius:10px; padding:14px; margin-bottom:8px;
  border:1px solid var(--border);
}}
.card-clickable {{ cursor:pointer; transition:transform .1s, box-shadow .1s; }}
.card-clickable:hover {{ transform:translateY(-1px); box-shadow:0 4px 12px rgba(0,0,0,.4); }}
.card-clickable.card-expanded {{ box-shadow:0 0 0 2px var(--gold); }}
.card-detail-inline {{ margin:-4px 0 12px 0; }}
.card-expand-chevron {{ display:inline-block; transition:transform .15s; }}
.bet-card.tier-EDGE {{
  background:linear-gradient(135deg,rgba(251,191,36,.18),rgba(251,191,36,.06));
  border-color:rgba(251,191,36,.4);
}}
.bet-card.tier-HIGH {{
  background:rgba(16,185,129,.08);
  border-color:rgba(16,185,129,.3);
}}
.bet-card.tier-MED {{
  background:rgba(245,158,11,.06);
  border-color:rgba(245,158,11,.25);
}}
.tier-badge {{
  display:inline-block; padding:2px 8px; border-radius:4px;
  font-size:.6rem; font-weight:800; color:#000;
}}
.bet-card-top {{ display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:6px; flex-wrap:wrap; gap:6px; }}
.bet-card-pick {{ margin:4px 0 8px; font-size:1rem; font-weight:700; }}
.bet-card-meta {{ font-size:.72rem; color:var(--dim); margin-top:2px; }}

/* ── Trust card ── */
.trust-card {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px; margin-top:8px; }}
@media (max-width:640px) {{ .trust-card {{ grid-template-columns:1fr; }} }}
.trust-block {{
  background:rgba(15,23,42,.5); padding:8px; border-radius:6px;
  border:1px solid var(--border);
}}
.trust-label {{
  font-size:.58rem; letter-spacing:.1em; color:var(--dim);
  text-transform:uppercase; margin-bottom:4px;
}}
.trust-body {{ font-size:.72rem; line-height:1.5; }}
.why-list {{ margin:4px 0 0 14px; padding:0; font-size:.72rem; line-height:1.6; }}

/* ── Copy WA button ── */
.wa-btn {{
  background:linear-gradient(135deg,#25d366,#128c7e); color:#fff;
  border:none; padding:5px 12px; border-radius:6px;
  font-size:.72rem; font-weight:700; cursor:pointer;
  display:inline-flex; align-items:center; gap:5px; transition:opacity .2s;
}}
.wa-btn:hover {{ opacity:.85; }}
.wa-btn + .wa-btn {{ margin-left:6px; }}

/* ── Empty state ── */
.empty-hero {{
  background:rgba(15,23,42,.3); border:1px dashed var(--border);
  border-radius:10px; padding:28px 24px; text-align:center;
}}
.empty-horse {{ font-size:2.4rem; margin-bottom:8px; }}
.empty-title {{ font-size:1.05rem; font-weight:700; margin-bottom:8px; }}
.empty-body {{ color:var(--dim); max-width:440px; margin:0 auto; line-height:1.7; font-size:.88rem; }}
.closest-miss {{
  margin-top:14px; font-size:.78rem; color:var(--amber);
  background:rgba(245,158,11,.08); border:1px solid rgba(245,158,11,.2);
  border-radius:6px; padding:6px 12px; display:inline-block;
}}
.edge-note {{
  font-size:.7rem; color:var(--dim); font-style:italic;
  margin:-4px 0 10px 0; padding:4px 0;
  border-bottom:1px dashed var(--border);
}}

/* ── Race strip ── */
.strip-section {{ margin-bottom:12px; }}
.strip-label {{ font-size:.6rem; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; margin-bottom:6px; display:flex; align-items:center; justify-content:space-between; }}
.expand-all-btn {{ background:transparent; border:1px solid var(--border); color:var(--fg); padding:3px 10px; border-radius:4px; font-size:.66rem; font-weight:600; letter-spacing:.05em; cursor:pointer; text-transform:uppercase; }}
.expand-all-btn:hover {{ background:rgba(15,23,42,.5); border-color:var(--gold); }}
.expand-all-btn.active {{ background:var(--gold); color:#000; border-color:var(--gold); }}
.strip {{
  display:grid; grid-template-columns:repeat(11,1fr); gap:4px;
  background:rgba(15,23,42,.3); border:1px solid var(--border);
  border-radius:10px; padding:10px;
}}
@media (max-width:600px) {{ .strip {{ grid-template-columns:repeat(6,1fr); }} }}
.strip-tile {{
  padding:7px 3px; border-radius:5px; text-align:center;
  cursor:pointer; transition:transform .12s, box-shadow .12s;
  border:1px solid transparent;
}}
.strip-tile:hover {{ transform:translateY(-2px); box-shadow:0 4px 12px rgba(0,0,0,.3); }}
.strip-tile.active {{ box-shadow:0 0 0 2px var(--gold); }}
.tier-EDGE {{ background:rgba(251,191,36,.2); border-color:rgba(251,191,36,.5); }}
.tier-HIGH {{ background:rgba(16,185,129,.15); border-color:rgba(16,185,129,.35); }}
.tier-MED  {{ background:rgba(245,158,11,.15); border-color:rgba(245,158,11,.35); }}
.tier-LOW  {{ background:rgba(100,116,139,.18); }}
.tier-SKIP {{ background:rgba(71,85,105,.18); opacity:.6; }}
.tier-CHAOS {{ background:rgba(220,38,38,.22); border:1px solid rgba(220,38,38,.4); }}
.strip-num {{ font-size:.72rem; font-weight:700; }}
.strip-sub {{ font-size:.58rem; color:var(--dim); margin-top:1px; }}
.strip-chevron {{ font-size:.6rem; color:var(--dim); margin-top:2px; transition:transform .15s; }}
.strip-tile.active .strip-chevron {{ transform:rotate(180deg); color:var(--gold); }}
.strip-tab {{ background:transparent; border:1px solid var(--border); color:var(--dim); padding:3px 10px; border-radius:4px; font-size:.66rem; font-weight:600; letter-spacing:.05em; cursor:pointer; text-transform:uppercase; transition:all .12s; }}
.strip-tab:hover {{ background:rgba(15,23,42,.5); color:var(--fg); }}
.strip-tab.active {{ background:var(--gold); color:#000; border-color:var(--gold); }}
.topbets-sub-tab {{ background:transparent; border:1px solid var(--border); color:var(--dim); padding:2px 10px; border-radius:4px; font-size:.65rem; font-weight:700; letter-spacing:.04em; cursor:pointer; text-transform:uppercase; margin-left:6px; }}
.topbets-sub-tab:hover {{ background:rgba(15,23,42,.5); color:var(--fg); }}
.topbets-sub-tab.active {{ background:var(--gold); color:#000; border-color:var(--gold); }}
.strike-badge {{ display:inline-block; padding:2px 8px; border-radius:4px; font-size:.62rem; font-weight:800; letter-spacing:.06em; margin-left:8px; vertical-align:middle; }}
.strike-badge.win {{ background:var(--gold); color:#000; }}
.strike-badge.qin {{ background:var(--green); color:#000; }}
.strike-badge.box4 {{ background:rgba(16,185,129,.7); color:#000; }}
.strike-badge.trio {{ background:rgba(236,72,153,.85); color:#000; }}
.strike-badge.f4 {{ background:linear-gradient(135deg,rgba(168,85,247,.95),rgba(236,72,153,.85)); color:#000; box-shadow:0 0 6px rgba(168,85,247,.45); }}
.strike-badge.trio6 {{ background:rgba(244,63,94,.85); color:#fff; }}
.strike-badge.qpl {{ background:rgba(34,197,94,.85); color:#000; }}
.strike-badge.place {{ background:rgba(99,102,241,.85); color:#fff; }}
.strike-badge.multi {{ background:linear-gradient(135deg,var(--gold),rgba(236,72,153,.95)); color:#000; box-shadow:0 0 8px rgba(251,191,36,.5); }}
.strike-badge.miss {{ background:rgba(239,68,68,.6); color:#fff; }}

/* View tabs — by default boxed (legacy below-strip placement).
   In the .view-tabs-topbar variant they live inline in the topbar
   so they need no own background or border. */
.view-tabs {{
  display:flex; align-items:center; gap:8px; margin:8px 0 12px 0;
  padding:8px; background:rgba(30,41,59,.5); border:1px solid var(--border);
  border-radius:8px; flex-wrap:wrap;
}}
.view-tabs-topbar {{
  margin:0; padding:0; background:transparent; border:none; border-radius:0;
}}
.view-tab {{
  background:transparent; border:1.5px solid var(--border);
  color:var(--dim); padding:9px 18px; border-radius:6px;
  font-size:.85rem; font-weight:700; letter-spacing:.04em;
  cursor:pointer; transition:all .15s;
  display:inline-flex; align-items:center; gap:6px;
}}
.view-tab-icon {{ font-size:.95rem; }}
.view-tab:hover {{ background:rgba(15,23,42,.6); color:var(--fg); border-color:var(--fg); }}
.view-tab.active {{ background:var(--gold); color:#000; border-color:var(--gold); box-shadow:0 2px 8px rgba(251,191,36,.3); }}
.info-tip {{ display:inline-block; margin-left:6px; font-size:.7rem; color:var(--dim); cursor:pointer; border:1px solid var(--border); border-radius:50%; width:16px; height:16px; line-height:14px; text-align:center; vertical-align:middle; user-select:none; }}
.info-tip:hover {{ color:var(--gold); border-color:var(--gold); background:rgba(251,191,36,.08); }}
.info-pop {{ position:fixed; max-width:320px; background:#1e293b; border:1px solid var(--gold); color:var(--fg); padding:10px 12px; border-radius:6px; font-size:.75rem; line-height:1.45; z-index:9999; box-shadow:0 8px 24px rgba(0,0,0,.5); }}
.info-pop-title {{ font-weight:700; color:var(--gold); margin-bottom:6px; font-size:.78rem; }}
.info-pop-close {{ float:right; cursor:pointer; color:var(--dim); margin-left:8px; font-weight:700; }}
.info-pop-close:hover {{ color:var(--fg); }}

/* ── Race detail ── */
#race-detail {{ margin-top:4px; }}
.detail-panel {{
  background:rgba(15,23,42,.4); border:1px solid var(--border);
  border-radius:10px; padding:16px;
}}
.detail-header {{
  display:flex; justify-content:space-between; align-items:flex-start;
  margin-bottom:12px; flex-wrap:wrap; gap:10px;
}}
.detail-title {{ font-size:1.1rem; font-weight:800; }}
.detail-meta {{ font-size:.78rem; color:var(--dim); margin-top:3px; }}
.horse-table {{ width:100%; border-collapse:collapse; font-size:.82rem; }}
.horse-table th {{
  text-align:left; padding:8px 10px; font-size:.65rem;
  text-transform:uppercase; letter-spacing:.06em; color:var(--dim);
  border-bottom:2px solid var(--border); background:rgba(15,23,42,.5);
}}
.horse-table td {{ padding:8px 10px; border-bottom:1px solid var(--border); }}
.horse-table tr:hover td {{ background:rgba(255,255,255,.03); }}
.horse-table tr.top-pick td {{ background:rgba(251,191,36,.04); }}
.edge-positive {{ color:var(--green); font-weight:700; }}
.edge-neutral {{ color:var(--dim); }}

/* ── Detail sections (speed map + reasoning) ── */
.detail-section {{
  margin-top:14px; background:rgba(15,23,42,.3);
  border:1px solid var(--border); border-radius:8px; padding:12px;
}}
.section-label {{
  font-size:.6rem; letter-spacing:.12em; text-transform:uppercase;
  color:var(--dim); margin-bottom:8px;
}}
.sm-row {{ font-size:.78rem; margin-bottom:4px; }}
.rsn-row {{ margin-bottom:10px; }}
.rsn-row ul {{ margin:4px 0 0 16px; padding:0; font-size:.78rem; line-height:1.6; color:var(--dim); }}

/* ── Copy all bets button ── */
.wa-btn-all {{
  background:linear-gradient(135deg,#25d366,#128c7e); color:#fff;
  border:none; padding:5px 12px; border-radius:6px;
  font-size:.72rem; font-weight:700; cursor:pointer;
  display:inline-flex; align-items:center; gap:5px; transition:opacity .2s;
}}
.wa-btn-all:hover {{ opacity:.85; }}

/* ── Alerts rail ── */
.rail-label {{ font-size:.6rem; letter-spacing:.12em; color:var(--dim); text-transform:uppercase; margin-bottom:10px; }}
.alert-item {{
  padding:8px 10px; border-radius:5px; margin-bottom:6px;
  border-left:3px solid; background:rgba(15,23,42,.5);
}}
.alert-type {{ font-size:.7rem; font-weight:700; }}
.alert-detail {{ font-size:.64rem; color:var(--dim); margin-top:2px; }}

/* ── Refresh banner ── */
#refresh-banner {{
  position:fixed; top:0; left:0; right:0;
  background:linear-gradient(90deg,#1e40af,#3b82f6,#1e40af);
  background-size:200% 100%; animation:shimmer 3s linear infinite;
  color:#fff; padding:10px 20px; z-index:1000; display:none;
  box-shadow:0 2px 12px rgba(0,0,0,.4);
}}
@keyframes shimmer {{ 0% {{ background-position:-200% 0; }} 100% {{ background-position:200% 0; }} }}
@keyframes spin {{ from {{ transform:rotate(0deg); }} to {{ transform:rotate(360deg); }} }}
#refresh-banner .rb-row {{ display:flex; align-items:center; gap:12px; max-width:1200px; margin:0 auto; flex-wrap:wrap; }}
#refresh-banner .rb-icon {{ font-size:1.2rem; animation:spin 1.5s linear infinite; }}
#refresh-banner .rb-bar {{ height:3px; background:rgba(255,255,255,.2); border-radius:2px; width:100%; margin-top:6px; overflow:hidden; }}
#refresh-banner .rb-bar-fill {{ height:100%; background:rgba(255,255,255,.85); border-radius:2px; transition:width .5s ease; }}

/* ── Footer ── */
.footer {{ text-align:center; padding:20px; font-size:.7rem; color:var(--dim); border-top:1px solid var(--border); margin-top:16px; }}
.disclaimer {{ font-size:.72rem; color:#fca5a5; background:rgba(239,68,68,.07); border:1px solid rgba(239,68,68,.18); border-radius:6px; padding:8px 14px; margin-top:12px; }}
.accuracy-panel summary {{ list-style:none; }}
.accuracy-panel summary::-webkit-details-marker {{ display:none; }}
.accuracy-panel summary::before {{ content:"▸ "; color:var(--dim); }}
.accuracy-panel[open] summary::before {{ content:"▾ "; }}
.acc-table {{ width:100%; border-collapse:collapse; font-size:.75rem; margin-top:4px; }}
.acc-table th, .acc-table td {{ padding:3px 8px; text-align:right; border-bottom:1px solid var(--border); }}
.acc-table th:first-child, .acc-table td:first-child {{ text-align:left; }}
.acc-table th {{ font-weight:500; color:var(--dim); text-transform:uppercase; font-size:.65rem; }}
</style>
</head>
<body>

<div id="refresh-banner">
  <div class="rb-row">
    <span class="rb-icon">↻</span>
    <div style="flex:1;min-width:200px">
      <div style="font-weight:700;font-size:.9rem">Refreshing race data</div>
      <div style="opacity:.85;font-size:.78rem" id="rb-sub">Starting…</div>
    </div>
    <span style="opacity:.7;font-size:.72rem;font-variant-numeric:tabular-nums" id="rb-elapsed">0s</span>
  </div>
  <div class="rb-bar"><div class="rb-bar-fill" id="rb-bar-fill" style="width:0%"></div></div>
</div>

<div class="topbar">
  <div style="display:flex;align-items:center;gap:14px;flex-wrap:wrap">
    <div>
      <span class="topbar-brand">HKJC</span>
      <span class="topbar-meta" style="margin-left:10px">{race_date} · {venue}</span>
    </div>
    <div class="view-tabs view-tabs-topbar">
      <button class="view-tab active" id="tab-race-order" onclick="setStripView('race')">
        <span class="view-tab-icon">📋</span> Race Order
      </button>
      <button class="view-tab" id="tab-top-bets" onclick="setStripView('bets')">
        <span class="view-tab-icon">⭐</span> Top Bets
      </button>
    </div>
  </div>
  <div class="topbar-meta">
    <span class="live-dot"></span>
    <span id="clock">{now}</span>
    <span style="margin-left:8px">|</span>
    <span style="margin-left:8px">Last refresh {now}</span>
    <button id="refresh-btn" onclick="refreshDashboard()" style="display:none;margin-left:12px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);color:#fff;border:none;padding:5px 12px;border-radius:6px;font-size:.75rem;font-weight:700;cursor:pointer;">
      ↻ Refresh
    </button>
    <button id="hard-reload-btn" onclick="hardReload()" title="Bypass browser cache and reload this page" style="margin-left:8px;background:transparent;color:var(--fg);border:1px solid var(--border);padding:5px 10px;border-radius:6px;font-size:.72rem;font-weight:600;cursor:pointer;">
      ⟳ Hard reload
    </button>
    <button id="copy-all-btn" class="wa-btn-all" style="margin-left:8px">Copy all bets</button>
  </div>
</div>

<div class="shell">
  <div class="main-col">
    <div class="hero" id="hero"></div>
    <div class="strip-section" id="strip-section">
      <div class="strip" id="strip">{strip_tiles_html}</div>
    </div>
    <div id="top-bets-view" style="display:none"></div>
    <div id="race-detail"></div>
    {accuracy_panel_html}
    <div class="disclaimer"><strong>DISCLAIMER:</strong> For entertainment only. Not financial advice. Gamble responsibly.</div>
    <div class="footer">HKJC Racing Analyzer v6.11 — Generated: {now}</div>
  </div>
  <div class="alerts-rail" id="alerts-rail"></div>
</div>

<script>
(function() {{
// XSS-safe escape helper
function esc(s) {{ var d=document.createElement('div'); d.textContent=String(s!=null?s:''); return d.innerHTML; }}
// Attribute-context escape — implemented with character-code lookup
// so no quote literals appear next to other quote literals (the
// JS-syntax test scanner is naive about regex char classes).
// Use for values rendered inside an HTML attribute; esc() handles text.
var _ATTR_MAP = {{ 38:'&amp;', 60:'&lt;', 62:'&gt;', 34:'&quot;', 39:'&#39;' }};
var _ATTR_RE = new RegExp(
  '[' + String.fromCharCode(38, 60, 62, 34, 39) + ']', 'g');
function escAttr(s) {{
  return String(s != null ? s : '').replace(_ATTR_RE, function(c) {{
    return _ATTR_MAP[c.charCodeAt(0)] || c;
  }});
}}

const RACES = {races_json};
const ACCURACY_SUMMARY = {accuracy_summary_json};
const EDGE_STATS = {edge_stats_json};
const RACE_DATE = "{race_date}";
const GENERATED = "{now}";

// ─────────────────────────────────────────────────────────────
// Tier colours (mirrors Python TIER_COLORS)
// ─────────────────────────────────────────────────────────────
const TIER_COLORS = {{
  EDGE:'#fbbf24', HIGH:'#10b981', MED:'#f59e0b', LOW:'#64748b', SKIP:'#475569', CHAOS:'#dc2626'
}};
const ALERT_COLORS = {{
  EDGE_OPENED:'#fbbf24', EDGE_WIDENED:'#fbbf24', EDGE_CLOSED:'#94a3b8',
  STEAM:'#ef4444', COLD:'#06b6d4', RACE_UPCOMING:'#10b981'
}};

// ─────────────────────────────────────────────────────────────
// Live clock
// ─────────────────────────────────────────────────────────────
setInterval(function() {{
  var el = document.getElementById('clock');
  if (el) {{
    var now = new Date();
    el.textContent = now.toLocaleTimeString('en-HK', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
  }}
}}, 1000);

// ─────────────────────────────────────────────────────────────
// Hero: 0-5 bet cards from EDGE/HIGH/MED races
// ─────────────────────────────────────────────────────────────
// Top Bets sub-filter: 'next' (default — unfinished races) or 'past'
// (races with results, with STRIKE/MISS badges).
var _topBetsFilter = 'next';

// _placedFromRace — extract [p1, p2, p3, p4] horse numbers from a finished
// race. Returns null when fewer than 2 horses have actual_position (race
// not finished or partial results). Shared by _betHitsTicket and
// _allStrikeLabels so they read placements identically.
function _placedFromRace(race) {{
  if (!race.results_available) return null;
  var horsesByPos = race.horses
    .filter(function(h) {{ return h.actual_position; }})
    .sort(function(a, b) {{ return a.actual_position - b.actual_position; }});
  if (horsesByPos.length < 2) return null;
  var p1 = horsesByPos[0].number;
  var p2 = horsesByPos[1].number;
  var p3 = horsesByPos.length >= 3 ? horsesByPos[2].number : null;
  var p4 = horsesByPos.length >= 4 ? horsesByPos[3].number : null;
  return {{
    p1: p1, p2: p2, p3: p3, p4: p4,
    top2: [p1, p2],
    top3: p3 != null ? [p1, p2, p3] : null,
    top4: p4 != null ? [p1, p2, p3, p4] : null,
  }};
}}

// _betHitsTicket — per-bet hit/miss. Returns true/false/null per single
// ticket entry. null = race not yet finished / partial results.
// Mirrors validate_hk._score_meeting_replay pool branches exactly so
// dashboard display agrees with replay scoring.
function _betHitsTicket(bet, race) {{
  var placed = _placedFromRace(race);
  if (placed === null) return null;
  var pool = bet.pool || '';
  var legs = bet.legs || [];
  var nLegs = legs.length;
  var nums = legs.map(function(l) {{ return l.number; }});
  function _containsAll(needed) {{
    if (!needed) return false;
    for (var i = 0; i < needed.length; i++) {{
      if (nums.indexOf(needed[i]) < 0) return false;
    }}
    return true;
  }}
  if (pool === 'WIN' || pool === 'BANKER WIN') {{
    return !!(legs[0] && legs[0].number === placed.p1);
  }}
  if (pool === 'PLACE') {{
    if (!placed.top3) return false;
    for (var k = 0; k < nLegs; k++) {{
      if (placed.top3.indexOf(legs[k].number) >= 0) return true;
    }}
    return false;
  }}
  if (pool === 'QUINELLA') {{
    return _containsAll(placed.top2);
  }}
  if (pool === 'QIN PLACE') {{
    // QPL BANKER: legs[0] pivot + legs[1:] bankers. Need pivot top-3 AND
    // at least one banker top-3.
    if (!placed.top3 || nLegs < 2) return false;
    if (placed.top3.indexOf(legs[0].number) < 0) return false;
    for (var k = 1; k < nLegs; k++) {{
      if (placed.top3.indexOf(legs[k].number) >= 0) return true;
    }}
    return false;
  }}
  if (pool === 'TRIO') {{
    return placed.top3 != null && _containsAll(placed.top3);
  }}
  if (pool === 'FIRST 4') {{
    return placed.top4 != null && _containsAll(placed.top4);
  }}
  return false;
}}

// _allStrikeLabels — return short labels for EVERY hit bet, priority-
// ordered. Used by both _didBetHit (which collapses to a single highest-
// priority badge) and the multi-hit STRIKE badge renderer (which lists
// all of them when ≥2 hit).
function _allStrikeLabels(race) {{
  var placed = _placedFromRace(race);
  if (placed === null) return null;
  var ticket = race.bet_ticket;
  var bets = (ticket && ticket.bets) ? ticket.bets : [];
  if (bets.length === 0) return [];
  // Dedupe across multiple bets of the same pool (e.g., two TRIO bets
  // both hitting → one TRIO5 entry).
  var seen = {{}};
  var ordered = [];
  function _push(short) {{ if (!seen[short]) {{ seen[short] = true; ordered.push(short); }} }}
  for (var i = 0; i < bets.length; i++) {{
    var bet = bets[i];
    if (!_betHitsTicket(bet, race)) continue;
    var pool = bet.pool || '';
    var nLegs = (bet.legs || []).length;
    if (pool === 'WIN' || pool === 'BANKER WIN') _push('WIN');
    else if (pool === 'PLACE') _push('PLACE');
    else if (pool === 'QUINELLA') _push(nLegs === 4 ? 'BOX4' : 'QIN');
    else if (pool === 'QIN PLACE') _push('QPL');
    else if (pool === 'TRIO') _push(nLegs >= 6 ? 'TRIO6' : 'TRIO5');
    else if (pool === 'FIRST 4') _push('F4-5');
  }}
  // Re-sort by priority so the highest-priority hit leads the multi-hit chip.
  var priority = ['WIN', 'QIN', 'QPL', 'F4-5', 'BOX4', 'TRIO5', 'TRIO6', 'PLACE'];
  ordered.sort(function(a, b) {{
    var ai = priority.indexOf(a);
    var bi = priority.indexOf(b);
    return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
  }});
  return ordered;
}}

function _didBetHit(race) {{
  // Returns the strongest STRIKE badge for any bet ACTUALLY in the frozen
  // bet_ticket that hit. Returns MISS only when a ticket existed but no
  // leg landed. Returns null for SKIP races (no ticket) so the badge is
  // suppressed entirely — a SKIP is not a miss, we never staked. Returns
  // null while the race is unfinished.
  //
  // Delegates per-bet hit detection to _betHitsTicket (shared with the
  // detail-panel per-bet markers + multi-hit badge renderer) so all three
  // surfaces compute hits identically. Priority order:
  // WIN > QIN > QPL > F4-5 > BOX4 > TRIO5 > TRIO6 > PLACE.
  var labels = _allStrikeLabels(race);
  if (labels === null) return null;
  if (labels.length === 0) {{
    var ticket = race.bet_ticket;
    var bets = (ticket && ticket.bets) ? ticket.bets : [];
    // SKIP race → no badge. Real miss → MISS.
    return bets.length === 0 ? null : 'MISS';
  }}
  return 'STRIKE-' + labels[0];
}}

function setTopBetsFilter(f) {{
  _topBetsFilter = f;
  var nb = document.getElementById('tb-next');
  var pb = document.getElementById('tb-past');
  if (nb) nb.classList.toggle('active', f === 'next');
  if (pb) pb.classList.toggle('active', f === 'past');
  renderHero();
}}

function renderHero() {{
  // Behaviour driven by _stripView:
  //   'race' (default) → all 11 races as cards in race order (R1, R2, ...)
  //   'bets'           → curated stack of HIGH/MED/EDGE picks, tier-sorted,
  //                      filtered by _topBetsFilter (next vs past).
  var hero = document.getElementById('hero');
  var label, races;
  if (_stripView === 'bets') {{
    var tierOrder = {{EDGE:0, HIGH:1, MED:2}};
    var pool = RACES
      .filter(function(r) {{ return ['EDGE','HIGH','MED'].indexOf(r.conviction_tier) >= 0; }});
    if (_topBetsFilter === 'past') {{
      pool = pool.filter(function(r) {{ return r.results_available; }});
    }} else {{
      pool = pool.filter(function(r) {{ return !r.results_available; }});
    }}
    races = pool.sort(function(a, b) {{
      if (tierOrder[a.conviction_tier] !== tierOrder[b.conviction_tier])
        return tierOrder[a.conviction_tier] - tierOrder[b.conviction_tier];
      var aPp = Math.max.apply(null, a.horses.map(function(h) {{ return h.edge_pp || 0; }}));
      var bPp = Math.max.apply(null, b.horses.map(function(h) {{ return h.edge_pp || 0; }}));
      if (aPp !== bPp) return bPp - aPp;
      return (b.top_3_share || 0) - (a.top_3_share || 0);
    }});
    var lbl = _topBetsFilter === 'past' ? 'PAST RESULTS' : 'NEXT BETS';
    label = 'TOP BETS · ' + lbl + ' — ' + races.length + ' RACES' +
      // Inline Next/Past toggle right after the label
      ' &nbsp;<button class="topbets-sub-tab' + (_topBetsFilter === 'next' ? ' active' : '') + '" id="tb-next" data-testid="top-bets-next" onclick="setTopBetsFilter(&quot;next&quot;)">Next</button>' +
      '<button class="topbets-sub-tab' + (_topBetsFilter === 'past' ? ' active' : '') + '" id="tb-past" data-testid="top-bets-past" onclick="setTopBetsFilter(&quot;past&quot;)">Past</button>';
  }} else {{
    races = RACES.slice().sort(function(a, b) {{ return a.race_number - b.race_number; }});
    label = 'ALL RACES — ' + races.length + ' RACES IN CHRONOLOGICAL ORDER';
  }}

  if (races.length === 0) {{
    // Three distinct empty cases:
    //   (a) Top Bets · Next AND all races finished  → "no next race today"
    //   (b) Top Bets · Past AND no past races yet   → "no past races yet"
    //   (c) Top Bets · Next AND no race clears HIGH → "no bets meet threshold"
    var allFinished = RACES.length > 0 && RACES.every(function(r) {{ return r.results_available; }});
    var anyFinished = RACES.some(function(r) {{ return r.results_available; }});
    var emptyTitle, emptyBody, emptyIcon = '🏇';
    if (_stripView === 'bets' && _topBetsFilter === 'next' && allFinished) {{
      emptyIcon = '🏁';
      emptyTitle = 'All races finished — no next race today';
      emptyBody = 'Card complete. Switch to <strong onclick="setTopBetsFilter(&quot;past&quot;);" style="cursor:pointer;color:var(--gold);text-decoration:underline">Past</strong> to see today&#39;s record, or <strong onclick="setStripView(&quot;race&quot;);" style="cursor:pointer;color:var(--gold);text-decoration:underline">Race Order</strong> to review every race with results.';
    }} else if (_stripView === 'bets' && _topBetsFilter === 'past' && !anyFinished) {{
      emptyTitle = 'No past races yet';
      emptyBody = 'No race has finished. Switch to <strong onclick="setTopBetsFilter(&quot;next&quot;);" style="cursor:pointer;color:var(--gold);text-decoration:underline">Next</strong> to see upcoming bets.';
    }} else {{
      var closest = RACES
        .filter(function(r) {{ return r.top_3_share != null; }})
        .sort(function(a, b) {{ return (b.top_3_share||0) - (a.top_3_share||0); }})[0];
      var closestMsg = closest
        ? 'Closest to HIGH: R' + closest.race_number + ' (top_3_share ' + Math.round((closest.top_3_share||0)*100) + '%)'
        : '';
      emptyTitle = 'No bets meet the threshold today';
      emptyBody = 'Model and market agree too closely for an edge play, and no race clears HIGH/MED conviction on pair identification. Switch to <strong onclick="setStripView(&quot;race&quot;);" style="cursor:pointer;color:var(--gold);text-decoration:underline">Race Order</strong> to browse all races anyway.' +
        (closestMsg ? '<div class="closest-miss" style="margin-top:8px">' + esc(closestMsg) + '</div>' : '');
    }}
    hero.innerHTML =
      '<div class="empty-hero">' +
        '<div class="empty-horse">' + emptyIcon + '</div>' +
        '<div class="empty-title">' + emptyTitle + '</div>' +
        '<div class="empty-body">' + emptyBody + '</div>' +
      '</div>';
    return;
  }}

  var hasEdgeRace = races.some(function(r) {{ return r.has_edge; }});
  var html = '<div class="hero-label">' + label + '</div>';
  if (!hasEdgeRace && _stripView === 'bets') {{
    html += '<div class="edge-note">★ EDGE not seen in current sample — HIGH / MED tiers are the operational signal.</div>';
  }}
  races.forEach(function(race) {{ html += renderBetCard(race); }});
  hero.innerHTML = html;
  hero.querySelectorAll('.wa-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      // data-payload is a plain string (escAttr-encoded for the attribute);
      // browser already decoded it, so no JSON.parse needed.
      copyWA(btn.dataset.payload, btn);
    }});
  }});
}}

function renderBetCard(race) {{
  if (!race.horses || !race.horses.length) return '';
  var topH = race.horses[0];
  var tier = race.conviction_tier;
  var isEdge = tier === 'EDGE';
  var tierColor = TIER_COLORS[tier] || '#64748b';

  // Why Strong bullets
  var whyItems = (topH.reasoning || []).slice(0, 3);
  var whyHtml = whyItems.length
    ? '<ul class="why-list">' + whyItems.map(function(r) {{ return '<li>' + esc(r) + '</li>'; }}).join('') + '</ul>'
    : '<span style="color:var(--dim)">No reasoning captured</span>';

  // Our vs Market — show BOTH pre-blend (drives edge) and post-blend (drives
  // recommender). Was previously only "Our model" = pure_prob, which made
  // "+6pp edge / SKIP" cards look contradictory. Both stages now visible.
  var purePct = topH.pure_prob != null ? (topH.pure_prob * 100).toFixed(0) : '?';
  var calPct  = topH.calibrated_prob != null ? (topH.calibrated_prob * 100).toFixed(0) : '?';
  var mktPct  = topH.market_prob != null ? (topH.market_prob * 100).toFixed(0) : '?';
  var edgePp  = topH.edge_pp != null ? topH.edge_pp.toFixed(1) : '0.0';

  // Tier Track Record
  var stats = (race.tier_stats || {{}})[tier];
  var trackHtml = stats
    ? (tier + ' hit ' + stats.hit + '/' + stats.total + ' (' + Math.round((stats.rate||0)*100) + '%)' +
       '<br><span style="color:var(--dim);font-size:.65rem">Wilson [' + Math.round((stats.ci_low||0)*100) + '–' + Math.round((stats.ci_high||0)*100) + '%]</span>')
    : '<span style="color:var(--dim)">Insufficient sample</span>';

  // Suggested stake — bankroll-fraction sizing, see betting/optimizer.
  // Hidden when amount=0 so SKIP cards do not show a wasted line.
  // Distributes the total across firing MAIN-tier bets (MAIN + BOX TRIO-5
  // + F4 BOX-5) by combo-count, so the user sees how the $X splits.
  var ss = race.suggested_stake || {{}};
  var stakeHtml = '';
  // LONGSHOT-TRIO 🧪 + BOLTER TRIO 🍀 removed from the live dashboard
  // 2026-05-25 per user direction. Both still compete with real system
  // bets in the ticket area despite EXPERIMENT/NOT STAKED labels:
  //   - LONGSHOT-TRIO variant 1: N=7 meetings, KILL verdict
  //     (memory: project_bolter_calc_hot_killed_2026-05-24).
  //   - BOLTER TRIO mix_hot_cold: 3 meetings, 0/22 hits, INSUFFICIENT_DATA.
  //     User wants tracker running silently (scripts.track_longshot_trio
  //     still logs to tracking_data/longshot_trio_log.csv) but UI hidden
  //     until N≥6 forward, then ROI verdict.
  // Helper functions _longshotPick + _bolterTrio also removed (dead).
  if (ss.amount && ss.amount > 0) {{
    var br = race.bankroll || 0;
    var betLines = [];
    var totalCombos = 0;
    var unitRounded = 0;
    var allocated = 0;
    if (race.bet_ticket && race.bet_ticket.bets && race.bet_ticket.bets.length) {{
      // Ticket-first: read unit_stake and per-bet amount from the frozen
      // ticket (GPT-5.5 R12 #1 2026-05-25). Server computed them once at
      // ticket-build time; visual + WA + replay must all read the SAME
      // numbers so the user sees the actual stake everywhere.
      unitRounded = race.bet_ticket.unit_stake || 0;
      allocated = race.bet_ticket.allocated_amount || 0;
      race.bet_ticket.bets.forEach(function(b) {{
        var combos = b.combos || 0;
        if (combos > 0) {{
          betLines.push({{
            lbl: b.label || b.pool || 'BET',
            combos: combos,
            amount: b.amount || (combos * unitRounded),
          }});
          totalCombos += combos;
        }}
      }});
    }} else {{
      // Legacy fallback for older JSON files without server-side bet_ticket.
      var mainBets = {{}};
      race.horses.forEach(function(h) {{
        var b = (h.bet_recommendation || '').trim();
        if (b && b !== 'SKIP') {{ (mainBets[b] = mainBets[b] || []).push(h); }}
      }});
      Object.keys(mainBets).forEach(function(bt) {{
        var n = mainBets[bt].length;
        var combos = 0, lbl = bt;
        if (bt === 'WIN' || bt === 'BANKER WIN') {{ combos = 1; lbl = bt; }}
        else if (bt === 'PLACE') {{ combos = 1; lbl = 'PLACE'; }}
        else if ((bt === 'QIN' || bt === 'BOX QIN')) {{
          combos = n === 2 ? 1 : (n*(n-1)/2);
          lbl = n === 2 ? 'STRAIGHT QIN' : 'BOX QIN-' + n;
        }} else if (bt === 'TRIO') {{
          combos = n === 3 ? 1 : (n*(n-1)*(n-2)/6);
          lbl = n === 3 ? 'STRAIGHT TRIO' : 'BOX TRIO-' + n;
        }}
        if (combos > 0) {{
          betLines.push({{lbl: lbl, combos: combos, amount: 0}});
          totalCombos += combos;
        }}
      }});
      // Legacy path: recompute unit + per-bet amount the old way.
      var unit = totalCombos > 0 ? (ss.amount / totalCombos) : 0;
      unitRounded = Math.max(10, Math.floor(unit / 10) * 10);
      betLines.forEach(function(b) {{
        b.amount = b.combos * unitRounded;
        allocated += b.amount;
      }});
    }}
    var breakdownRows = '';
    betLines.forEach(function(b) {{
      breakdownRows +=
        '<tr><td style="padding:1px 6px 1px 0;color:var(--dim)">' + esc(b.lbl) + '</td>' +
        '<td style="padding:1px 0;text-align:right">' + b.combos + ' × $' + unitRounded + '</td>' +
        '<td style="padding:1px 0 1px 8px;text-align:right;color:var(--gold);font-weight:600">$' + Math.round(b.amount) + '</td></tr>';
    }});
    // Over-budget transparency: when ticket allocated > suggested, surface
    // a row so the user sees the actual spend (GPT-5.5 R12 #2 2026-05-25).
    var _overBudget = !!(race.bet_ticket && race.bet_ticket.over_budget);
    var _overAmount = (race.bet_ticket && race.bet_ticket.over_amount) || 0;
    var overRow = (_overBudget && _overAmount >= 1)
      ? '<tr><td colspan="3" style="padding:3px 0 1px 0;color:#f59e0b;font-size:.65rem;font-style:italic">⚠ $' +
          Math.round(_overAmount) + ' over budget (HK$10 unit floor)</td></tr>'
      : '';
    var breakdownHtml = breakdownRows
      ? '<table style="font-size:.7rem;border-collapse:collapse;margin-top:6px;width:100%">' +
          breakdownRows +
          '<tr style="border-top:1px solid var(--border)"><td style="padding:3px 0 1px 0;color:var(--dim)">Total</td>' +
          '<td></td><td style="padding:3px 0 1px 8px;text-align:right;color:var(--gold);font-weight:600">$' + Math.round(allocated) + '</td></tr>' +
          overRow +
        '</table>'
      : '';
    stakeHtml =
      '<div class="trust-block">' +
        '<div class="trust-label">Suggested Stake ' +
          '<span class="info-tip" onclick="showInfoPop(event, &quot;Suggested Stake&quot;, &quot;Per-card stake sized off your bankroll (config.BANKROLL = $' + br + ' HKD). EDGE 3%, HIGH 2%, MED 1%, LOW 0.5%. WIN bets use half-Kelly capped at the tier %. The budget is split across firing MAIN bets by combo count — unit stake is floored at the HK$10 HKJC minimum and rounded to $10. Adjust BANKROLL in config.py to match your rolling bankroll.&quot;)">ⓘ</span>' +
        '</div>' +
        '<div class="trust-body" style="margin-top:4px;font-size:.85rem">' +
          '<strong style="color:var(--gold)">$' + Math.round(ss.amount) + '</strong> HKD budget' +
          '<br><span style="color:var(--dim);font-size:.65rem">' + (ss.pct||0).toFixed(2) + '% of $' + br + ' · ' + esc(ss.method||'') + '</span>' +
          breakdownHtml +
        '</div>' +
      '</div>';
  }}
  // Bankroll guard STOP — daily loss/exposure cap hit (server zeroed the
  // suggested stake; audit 2026-06-08 #5). Show NO SYSTEM BET + the reason.
  if (race.bankroll_halt) {{
    stakeHtml =
      '<div class="trust-block" style="border-color:rgba(239,68,68,.4)">' +
        '<div class="trust-label" style="color:#fca5a5">⛔ BANKROLL STOP</div>' +
        '<div class="trust-body" style="margin-top:4px;font-size:.78rem;color:#fca5a5">' +
          'NO SYSTEM BET — ' + esc(race.bankroll_halt) +
        '</div>' +
      '</div>';
  }}
  // Card-exposure over-cap disclosure (audit 2026-06-08 #1). Bets are NOT
  // dropped; the real allocated total simply exceeds the day cap, so flag it
  // honestly for a manual trim. Hidden on a halted day (already NO SYSTEM BET).
  if (race.card_over_cap && !race.bankroll_halt) {{
    stakeHtml +=
      '<div style="margin-top:6px;font-size:.7rem;color:#f59e0b;background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:6px;padding:6px 10px">' +
        '⚠ Card exposure $' + Math.round(race.card_allocated_total||0) +
        ' exceeds ' + (race.card_exposure_cap_pct||10) + '% cap ($' +
        Math.round(race.card_exposure_cap||0) + ') by $' +
        Math.round(race.card_over_cap_amount||0) + ' — trim manually.' +
      '</div>';
  }}

  // Bet type sub-label
  var betRec = topH.bet_recommendation || 'QIN';
  var raceLabel = 'R' + race.race_number + ' · ' + esc(race.time || '') + ' · ' + esc(race.class_type || '') + ' ' + race.distance + 'm';

  // STRIKE / MISS badge for past races (resolved on the fly from
  // _didBetHit). Surfaces the bet outcome at the card header so the
  // user can scan past races for what hit without expanding each card.
  //
  // Multi-hit handling (added 2026-05-25): when ≥2 bet types hit on the
  // same race (e.g., WIN + QIN + TRIO-5 all fire on a model-nailed race),
  // the badge shows `★ STRIKE × N · A/B/C` instead of masking the lower
  // hits under the single highest-priority badge. Replay/stake_ab already
  // count all hits; this just surfaces them on the card scan.
  var strikeStatus = _didBetHit(race);
  var allStrikes = _allStrikeLabels(race) || [];
  var strikeBadge = '';
  if (allStrikes.length >= 2) {{
    var displayMap = {{
      'WIN': 'WIN', 'QIN': 'QIN', 'QPL': 'QPL', 'BOX4': 'BOX-4',
      'F4-5': 'F4-5', 'TRIO5': 'TRIO-5', 'TRIO6': 'TRIO-6', 'PLACE': 'PLACE',
    }};
    var pretty = allStrikes.map(function(s) {{ return displayMap[s] || s; }}).join('/');
    strikeBadge = '<span class="strike-badge multi" title="Multiple bet types hit on this race — see BET TICKET for per-bet result">★ STRIKE × ' + allStrikes.length + ' · ' + pretty + '</span>';
  }} else if (strikeStatus === 'STRIKE-WIN') {{
    strikeBadge = '<span class="strike-badge win" title="Top pick won the race">★ STRIKE · WIN</span>';
  }} else if (strikeStatus === 'STRIKE-QIN') {{
    strikeBadge = '<span class="strike-badge qin" title="Winning quinella pair was inside top-2 model picks">★ STRIKE · QIN</span>';
  }} else if (strikeStatus === 'STRIKE-QPL') {{
    strikeBadge = '<span class="strike-badge qpl" title="Quinella PLACE banker pair both finished top-3 (QPL hit)">★ STRIKE · QPL</span>';
  }} else if (strikeStatus === 'STRIKE-BOX4') {{
    strikeBadge = '<span class="strike-badge box4" title="Winning quinella pair was inside top-4 model picks (Box-4 hit)">★ STRIKE · BOX-4</span>';
  }} else if (strikeStatus === 'STRIKE-F4-5') {{
    strikeBadge = '<span class="strike-badge f4" title="Top-4 finishers all inside top-5 model picks (F4 BOX-5 hit — bigger dividend than TRIO-5)">★ STRIKE · F4-5</span>';
  }} else if (strikeStatus === 'STRIKE-TRIO5') {{
    strikeBadge = '<span class="strike-badge trio" title="Full podium inside top-5 model picks (BOX TRIO-5 hit)">★ STRIKE · TRIO-5</span>';
  }} else if (strikeStatus === 'STRIKE-TRIO6') {{
    strikeBadge = '<span class="strike-badge trio6" title="Full podium inside top-6 model picks (BOX TRIO-6 hit — TRIO-5 missed)">★ STRIKE · TRIO-6</span>';
  }} else if (strikeStatus === 'STRIKE-PLACE') {{
    strikeBadge = '<span class="strike-badge place" title="Top pick finished top-3 (PLACE hit)">★ STRIKE · PLACE</span>';
  }} else if (strikeStatus === 'MISS') {{
    var horsesByPos = race.horses.filter(function(h) {{ return h.actual_position; }})
      .sort(function(a, b) {{ return a.actual_position - b.actual_position; }});
    var winnerInfo = horsesByPos.length
      ? ' (winner #' + horsesByPos[0].number + ' was rank ' + horsesByPos[0].rank + ')'
      : '';
    strikeBadge = '<span class="strike-badge miss" title="Winning pair fell outside our top-4 picks">✗ MISS' + winnerInfo + '</span>';
  }}
  var edgeDetail = isEdge
    ? ('+' + edgePp + 'pp gap · ev +' + (topH.ev_win != null ? topH.ev_win.toFixed(2) : '0.00'))
    : ('top_3 ' + Math.round((race.top_3_share||0)*100) + '%');

  // WA copy payload — mirrors the BET TICKET so the copied message has
  // the same primary + protection bets the user sees on screen, plus
  // jockey strike rates and the top "why strong" bullets. Uses
  // WhatsApp markdown (*bold*, _italic_) for visual hierarchy on mobile.
  function _waPick(h) {{
    var p30 = h.jockey_form_30d != null ? Math.round(h.jockey_form_30d * 100) : null;
    var jk = h.jockey
      ? ' · ' + h.jockey + (p30 != null && p30 > 0 ? ' ' + p30 + '%' : '')
      : '';
    return '#' + h.number + ' ' + (h.name||'') + ' @$' + (h.win_odds||'?') + jk;
  }}
  var waLines = [];
  // Server-side bet_ticket is the single source of truth for WA bet
  // labels too. Added 2026-05-24 (GPT-5.5 Playwright audit): each WA
  // emitter gates on `_waTicketHas(label)` so client-side fabrications
  // cannot leak into the copied message.
  var _waTicketLabels = (race.bet_ticket && race.bet_ticket.bets)
    ? race.bet_ticket.bets.map(function(b) {{ return b.label; }})
    : [];
  function _waTicketHas(label) {{ return _waTicketLabels.indexOf(label) >= 0; }}
  // Resolve frozen ticket legs for a given bet label, hydrating each leg
  // into a full horse object (by number first, rank fallback) so WA picks
  // the same legs replay scores. Mirrors visual `_ticketLegsFor`. Added
  // 2026-05-25 (GPT-5.5 R14): without this, WA reconstructed QPL/BOX QIN-4/
  // F4 SAVER/BOX TRIO-5/F4 BOX-5 legs from current rank order and could
  // diverge from frozen ticket if ranks shifted after build.
  function _waTicketLegsFor(label) {{
    var _bets = (race.bet_ticket && race.bet_ticket.bets) || [];
    var bet = _bets.filter(function(b) {{ return b.label === label; }})[0];
    if (!bet) return [];
    var byNum = {{}};
    var byRank = {{}};
    race.horses.forEach(function(h) {{
      if (h.number != null) byNum[h.number] = h;
      if (h.rank != null) byRank[h.rank] = h;
    }});
    return (bet.legs || []).map(function(l) {{
      return byNum[l.number] || byRank[l.rank] || {{
        number: l.number, name: l.name, rank: l.rank,
      }};
    }});
  }}
  // Plain-text per-bet hit/miss mark for WA copy. Mirrors _ticketHitMark
  // (visual) and validate_hk replay scoring. Empty string while race
  // unresolved so live-prep messages do not carry false ✗ MISS marks.
  function _waHitMark(label) {{
    var _bets = (race.bet_ticket && race.bet_ticket.bets) || [];
    var bet = _bets.filter(function(b) {{ return b.label === label; }})[0];
    if (!bet) return '';
    var hit = _betHitsTicket(bet, race);
    if (hit === null) return '';
    return hit ? ' ✓ HIT' : ' ✗ MISS';
  }}
  // Race header — bold race id + meta on a second line for scannability
  var venueWa = race.venue || '';
  waLines.push('*R' + race.race_number + ' · ' + (race.time||'') + (venueWa ? ' · ' + venueWa : '') + '*');
  waLines.push((race.distance||'') + 'm · ' + (race.class_type||'') + ' · ' +
               race.horses.length + ' horses · Tier: ' + tier);
  // Suggested-stake banner — mirrors the dashboard per-card budget so
  // the user knows the total $ before scanning individual MAIN blocks.
  // Hidden on SKIP races (amount=0 there). Index is remembered so the
  // banner can be updated later with the unallocated residual once
  // waUnit + combo counts are known (budget floor often leaves $5-60 free).
  var _ssBannerIdx = -1;
  if (ss && ss.amount && ss.amount > 0) {{
    var ssPctTxt = (ss.pct != null) ? (' (' + (ss.pct||0).toFixed(2) + '% of $' + (race.bankroll||0) + ')') : '';
    _ssBannerIdx = waLines.length;
    waLines.push('💰 *Suggested stake: $' + Math.round(ss.amount) + ' HKD*' + ssPctTxt);
  }}
  // Only show betting copy when a real system ticket exists. A halted
  // day (bankroll STOP) or an empty/SKIP ticket has bet_ticket.bets=[], and
  // the WA-copy + visual ticket render from bet_ticket.bets — so the EDGE
  // "anchor TRIO-5/F4 with this horse" banner must not fire when there is no
  // TRIO-5/F4 in the ticket (audit 2026-06-08 #2).
  var hasSystemTicket = !!(race.bet_ticket && race.bet_ticket.bets && race.bet_ticket.bets.length);
  if (tier === 'EDGE' && hasSystemTicket) {{
    // Banner text reads live stats from EDGE_STATS (track_edge.py). Falls
    // back to generic "place > win" framing when no stats file present.
    var _es = (typeof EDGE_STATS !== 'undefined' && EDGE_STATS) ? EDGE_STATS : {{}};
    var _esText = (_es.n_fires && _es.n_fires > 0)
      ? ' (N=' + _es.n_fires + ' historical: ' + (_es.n_placed||0) + ' placed, ' + (_es.n_won||0) + ' won — do NOT WIN-bet)'
      : ' (historically places > wins — anchor TRIO, do NOT WIN-bet)';
    waLines.push('*★ RARE EDGE — anchor TRIO-5/F4 with this horse*' + _esText);
  }}

  // Group horses by primary bet type
  var waBets = {{}};
  race.horses.forEach(function(h) {{
    var b = (h.bet_recommendation || '').trim();
    if (b && b !== 'SKIP') {{ (waBets[b] = waBets[b] || []).push(h); }}
  }});

  // Recommender-driven bets from the FROZEN bet_ticket — single source of
  // truth for the WA PRIMARY block (closes the WA-fabrication gap from
  // GPT-5.5 audit 2026-05-24). Fallback to bet_recommendation when no
  // ticket exists (legacy / non-HK paths).
  var _waRecBets = (race.bet_ticket && race.bet_ticket.bets)
    ? race.bet_ticket.bets.filter(function(b) {{ return b.reason === 'recommender'; }})
    : [];

  // WA reads frozen bet_ticket.unit_stake / per-bet amounts (GPT-5.5 R12
  // #1 2026-05-25). Prior code re-derived `waUnit` from ss.amount and
  // counted combos via inline gate-clone logic — both drifted whenever
  // production gates changed. The ticket is the single source of truth
  // for staking; WA just renders it.
  var _allBets = (race.bet_ticket && race.bet_ticket.bets) || [];
  var waUnit = (race.bet_ticket && race.bet_ticket.unit_stake) || 0;
  var waTotalCombos = _allBets.reduce(function(s, b) {{ return s + (b.combos || 0); }}, 0);
  var _waAllocated = (race.bet_ticket && race.bet_ticket.allocated_amount) || 0;
  var _waOverBudget = !!(race.bet_ticket && race.bet_ticket.over_budget);
  var _waOverAmount = (race.bet_ticket && race.bet_ticket.over_amount) || 0;
  // Per-bet amount lookup — `bet.amount` is what the server allocated.
  function _waAmountFor(label) {{
    for (var i = 0; i < _allBets.length; i++) {{
      if (_allBets[i].label === label) return _allBets[i].amount || 0;
    }}
    return 0;
  }}

  // Stake banner annotation: when frozen ticket exceeds budget (HK$5 unit
  // floor pushed allocated > suggested), surface the over-spend honestly
  // instead of silently showing the lower budget. Otherwise, show the
  // residual when allocated < budget (existing behaviour).
  if (_ssBannerIdx >= 0 && ss && ss.amount > 0) {{
    if (_waOverBudget && _waOverAmount >= 1) {{
      waLines[_ssBannerIdx] = waLines[_ssBannerIdx].replace(
        ' HKD*',
        ' HKD — actual $' + Math.round(_waAllocated) +
        ' (+$' + Math.round(_waOverAmount) + ' over, $10 unit floor)*'
      );
    }} else if (_waAllocated > 0) {{
      var _unalloc = Math.round(ss.amount) - Math.round(_waAllocated);
      if (_unalloc >= 5) {{
        waLines[_ssBannerIdx] = waLines[_ssBannerIdx].replace(
          ' HKD*', ' HKD — $' + _unalloc + ' unallocated*'
        );
      }}
    }}
  }}

  // MAIN BET — one focused block (not a duplicate summary + detail).
  // Iterates the frozen ticket recommender bets; legacy fallback to
  // waBets dict when no ticket present.
  var _waPrimary = _waRecBets;
  var isSkipRace = _waPrimary.length === 0;
  function _waPickFromLeg(leg) {{
    var byNum = {{}};
    race.horses.forEach(function(h) {{ byNum[h.number] = h; }});
    var h = byNum[leg.number] || leg;
    return _waPick(h);
  }}
  if (!isSkipRace) {{
    _waPrimary.forEach(function(bet) {{
      var betStake = (bet.amount && bet.amount > 0)
        ? (' · $' + Math.round(bet.amount))
        : '';
      waLines.push('');
      waLines.push('*MAIN: ' + bet.label + betStake + '*' + _waHitMark(bet.label));
      (bet.legs || []).forEach(function(leg) {{ waLines.push(_waPickFromLeg(leg)); }});
    }});
  }} else {{
    waLines.push('');
    if (tier === 'CHAOS') {{
      var reason = race.chaos_reason || 'chaos race';
      waLines.push('*CHAOS — NO PLAY* (' + reason + ')');
    }} else {{
      // GPT-5.5 R15 2026-05-25 — say NO SYSTEM BET explicitly before any
      // horse suggestions so the user cannot mistake a "manual force pick"
      // section below for a model-recommended stake.
      waLines.push('*NO SYSTEM BET* — no frozen ticket for this race');
    }}
  }}

  // Optional add-on layers (BOX QIN-4, F4, TRIO-5, TRIO-6, QPL BANKER)
  // collected as bullets so the reader sees them as discretionary.
  // All leg numbers read from the FROZEN bet_ticket so WA = visual = replay
  // even when rank order shifts after the ticket is built (GPT-5.5 R14).
  var addOns = [];
  var top6wa = race.horses.slice(0, 6);

  // QIN PLACE BANKER for short-favorite races
  if (_waTicketHas('QIN PLACE BANKER')) {{
    var qplLegsWa = _waTicketLegsFor('QIN PLACE BANKER');
    if (qplLegsWa.length >= 4) {{
      var bnkNums = qplLegsWa.slice(1, 4).map(function(h) {{ return '#' + h.number; }}).join(' + ');
      addOns.push('QPL BANKER (3 combos): #' + qplLegsWa[0].number + ' with ' + bnkNums + _waHitMark('QIN PLACE BANKER'));
    }}
  }}

  // BOX QIN-4 PROTECTION — only when server included it in bet_ticket
  if (_waTicketHas('BOX QIN-4 PROTECTION')) {{
    var box4LegsWa = _waTicketLegsFor('BOX QIN-4 PROTECTION');
    if (box4LegsWa.length === 4) {{
      var box4Nums = box4LegsWa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
      addOns.push('BOX QIN-4 (6 combos): ' + box4Nums + _waHitMark('BOX QIN-4 PROTECTION'));
    }}
  }}

  // F4 SAVER — only when server included it in bet_ticket
  if (_waTicketHas('F4 SAVER')) {{
    var f4SaverLegsWa = _waTicketLegsFor('F4 SAVER');
    if (f4SaverLegsWa.length === 4) {{
      var f4Order = f4SaverLegsWa.map(function(h) {{ return '#' + h.number; }}).join(' → ');
      addOns.push('F4 SAVER (1 unit · top-4 any order): ' + f4Order + _waHitMark('F4 SAVER'));
    }}
  }}

  // Rank-5 swap REVERTED 2026-05-17 after R5 forward test: shipped on N=1
  // retrospective (R4) per user [[n6_validation_rule]] should have been ≥6
  // meetings. First live test (R5) swap turned a default BOX TRIO-5 HIT into
  // a MISS (removed #2 ISLAND BUDDY which finished 3rd at $28). Keep nulls
  // so downstream annotations no-op. Re-evaluate via backtest at N≥6 meetings.
  var top5waSwap = null;
  var top5waSwapOut = null;

  // BOX TRIO-5 — promoted to MAIN bet block (between PRIMARY and Optional
  // adds) per user feedback after R3 2026-05-03: TRIO-5 caught the podium
  // when the model's top-1/top-2 ordering was unreliable. It's the floor
  // for trio coverage and should never be buried as an "optional add".
  // Inserted into the lines array directly above the addOns block below.
  var t5MainBlock = null;
  if (_waTicketHas('BOX TRIO-5')) {{
    var t5LegsWa = _waTicketLegsFor('BOX TRIO-5');
    if (t5LegsWa.length === 5) {{
      var t5Nums = t5LegsWa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
      var _t5Amt = _waAmountFor('BOX TRIO-5');
      var t5Stake = _t5Amt > 0 ? (' · $' + Math.round(_t5Amt)) : '';
      var t5SwapAnno = top5waSwap
        ? ' · 🔁 swapped #' + top5waSwapOut.number + '→#' + top5waSwap.number
        : '';
      t5MainBlock = ['', '*MAIN: BOX TRIO-5' + t5Stake + '* (10 combos)' + t5SwapAnno + _waHitMark('BOX TRIO-5'), t5Nums];
    }}
  }}

  // F4 BOX-5 — second MAIN bet block (shipped 2026-05-06, +54% ROI on 60-race
  // backtest, strictly better economics than TRIO-5). Same horses as TRIO-5
  // but only 5 combos x $10 = $50 vs the TRIO-5 box at 10 combos x $10 = $100.
  // Same gates as F4 SAVER apply. Kept TRIO-5 as parallel MAIN since they
  // cover different misses (TRIO-5 hits when 4th finisher is outside top-5).
  var f4bMainBlock = null;
  if (_waTicketHas('F4 BOX-5')) {{
    var f4bLegsWa = _waTicketLegsFor('F4 BOX-5');
    if (f4bLegsWa.length === 5) {{
      var f4bNums = f4bLegsWa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
      var _f4bAmt = _waAmountFor('F4 BOX-5');
      var f4bStake = _f4bAmt > 0 ? (' · $' + Math.round(_f4bAmt)) : '';
      var f4bSwapAnno = top5waSwap
        ? ' · 🔁 swapped #' + top5waSwapOut.number + '→#' + top5waSwap.number
        : '';
      f4bMainBlock = ['', '*MAIN: F4 BOX-5' + f4bStake + '* (5 combos · top-4 any order)' + f4bSwapAnno + _waHitMark('F4 BOX-5'), f4bNums];
    }}
  }}

  // BOX TRIO-6 — annotate with TAKE/SKIP/YOUR CALL + chaos so user sees
  // the gate decision in the WA message (matches dashboard visual).
  // Mirror the spoiler-swap logic from the visual ticket so WA picks
  // match what the dashboard recommends. t6MainBlock is set when verdict
  // says TAKE so the line is rendered as MAIN instead of "Optional".
  var t6MainBlock = null;
  // TRIO-6 legs come from the FROZEN bet_ticket (server-side
  // _pick_trio6_top6 already chose any swap). WA just reads — same
  // source of truth as the detail-panel above (GPT-5.5 Finding #2).
  var _t6BetWa = (race.bet_ticket && race.bet_ticket.bets || []).find(
    function(b) {{ return b.label === 'BOX TRIO-6'; }});
  if (_t6BetWa) {{
    var _waByNum = {{}};
    race.horses.forEach(function(h) {{ _waByNum[h.number] = h; }});
    top6wa = (_t6BetWa.legs || []).map(function(l) {{
      return _waByNum[l.number] || l;
    }});
  }}
  var _swapWa = (_t6BetWa && _t6BetWa.swap) || null;
  var _waByNum2 = {{}};
  race.horses.forEach(function(h) {{ _waByNum2[h.number] = h; }});
  var top6waSwap = (_swapWa && _swapWa.kind === 'auto') ? (_waByNum2[_swapWa.in_number] || null) : null;
  var top6waSwapOut = (_swapWa && _swapWa.kind === 'auto') ? (_waByNum2[_swapWa.out_number] || null) : null;
  var top6waManual = (_swapWa && _swapWa.kind === 'manual') ? (_waByNum2[_swapWa.in_number] || null) : null;
  var top6waManualOut = (_swapWa && _swapWa.kind === 'manual') ? (_waByNum2[_swapWa.out_number] || null) : null;
  if (top6wa.length === 6) {{
    var t6Nums = top6wa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
    // Single source of truth: server-side _compute_trio6_decision in
    // dashboard/generator.py (race.trio6_decision). Before consolidation
    // (2026-05-29) this site recomputed independently and had drifted from
    // both the server gate and the detail panel.
    var t6dec = race.trio6_decision || {{}};
    var chaosScore = t6dec.chaos_score || 0;
    var t6Verdict = t6dec.verdict_label || 'YOUR CALL';
    var swapAnno = (top6waManual && top6waManualOut)
      ? ' · 🛠 MANUAL #' + top6waManualOut.number + '→#' + top6waManual.number + (race.manual_trio6_reason ? ' (' + race.manual_trio6_reason + ')' : '')
      : (top6waSwap && top6waSwapOut)
      ? ' · 🔁 swapped #' + top6waSwapOut.number + '→#' + top6waSwap.number + ' (gear ' + (top6waSwap.gear||'') + ')'
      : '';
    // When verdict says TAKE, hoist out of "Optional adds" into a MAIN block
    // so the section header matches the recommendation (was: TAKE shown under
    // "Optional", contradicting the verdict). Skip races stay in addOns
    // alongside the other demoted MAIN bets.
    var isT6Take = t6Verdict.indexOf('★ TAKE') === 0;
    if (_waTicketHas('BOX TRIO-6')) {{
      if (isT6Take && !isSkipRace) {{
        var _t6Amt = _waAmountFor('BOX TRIO-6');
        var t6Stake = _t6Amt > 0 ? (' · $' + Math.round(_t6Amt)) : '';
        t6MainBlock = ['', '*MAIN: BOX TRIO-6' + t6Stake + '* (20 combos · ' + t6Verdict + ' · CHAOS ' + chaosScore + '/4)' + swapAnno + _waHitMark('BOX TRIO-6'), t6Nums];
      }} else {{
        addOns.push('BOX TRIO-6 (20 combos · ' + t6Verdict + ' · CHAOS ' + chaosScore + '/4): ' + t6Nums + swapAnno + _waHitMark('BOX TRIO-6'));
      }}
    }}
  }}

  // Insert the TRIO-5 MAIN block before the optional adds so it reads
  // as a primary bet alongside PRIMARY, not buried under "optional".
  // On SKIP races we demote both MAIN blocks (TRIO-5, F4 BOX-5) to the
  // optional-adds bucket so the pasted text does not contradict itself
  // (was: "SKIP — no bet recommended ... MAIN: BOX TRIO-5"). Audit fix
  // 2026-05-17 for WA-copy consistency with dashboard.
  if (t5MainBlock) {{
    if (isSkipRace) {{
      // Demote on SKIP — render as a single optional-adds bullet so the
      // pasted message does not say SKIP + MAIN in the same race.
      addOns.push('BOX TRIO-5 (10 combos): ' + t5MainBlock[2]);
    }} else {{
      t5MainBlock.forEach(function(l) {{ waLines.push(l); }});
    }}
  }}
  if (f4bMainBlock) {{
    if (isSkipRace) {{
      addOns.push('F4 BOX-5 (5 combos · top-4 any order): ' + f4bMainBlock[2]);
    }} else {{
      f4bMainBlock.forEach(function(l) {{ waLines.push(l); }});
    }}
  }}
  if (t6MainBlock) {{
    t6MainBlock.forEach(function(l) {{ waLines.push(l); }});
  }}

  if (addOns.length) {{
    waLines.push('');
    waLines.push('*Optional adds:*');
    addOns.forEach(function(a) {{ waLines.push('• ' + a); }});
  }}

  // PAPER EXPERIMENT Longshot-TRIO block removed from WA 2026-05-25 per
  // user direction (same reason as the visual paperBlock removal above):
  // even with "NOT A BET" framing, the line competed with real system
  // bets in the WA copy. Tracker scripts still log to
  // tracking_data/longshot_trio_log.csv; surface results only in
  // post-race report, not live WA.

  // Why-strong block, italicised
  if (whyItems.length) {{
    waLines.push('');
    waLines.push('_Why: ' + whyItems.slice(0, 3).join(' · ') + '_');
  }}

  // Model trust footer — recent accuracy for the conviction tier on this race.
  // Reads from ACCURACY_SUMMARY (60d window from db.get_accuracy_summary).
  // Silent when no data yet, or when the tier has 0 bets in the window.
  try {{
    var convBucket = ACCURACY_SUMMARY && ACCURACY_SUMMARY.by_conviction;
    var tierStats = convBucket && convBucket[tier];
    if (tierStats && tierStats.bets > 0 && tierStats.stake > 0) {{
      var tBets = tierStats.bets, tHits = tierStats.hits;
      var tHitPct = (tHits / tBets * 100).toFixed(0);
      // Compare BEFORE .toFixed() so tiny negatives do not render as "+-0%".
      var tRoiNum = tierStats.profit / tierStats.stake * 100;
      var tRoiStr = (tRoiNum >= 0 ? '+' : '') + tRoiNum.toFixed(0) + '%';
      var noteN = tBets < 30 ? ' (small N)' : '';
      waLines.push('');
      waLines.push('_Model 60d · ' + tier + ' tier: ' + tHits + '/' + tBets +
                   ' (' + tHitPct + '%) · ' + tRoiStr + ' ROI' + noteN + '_');
    }}
  }} catch (e) {{ /* silent — never break WA copy */ }}

  var waPayload = waLines.join('\\n');

  // ─────────────────────────────────────────────────────────────
  // Compact ticket-first WA payload (GPT-5.5 R14 2026-05-25, refined
  // R15 2026-05-25 per user spec).
  //
  // RACE-DAY rules:
  //   - If frozen bet_ticket has bets: render STAKE + BET TICKET only.
  //     No TOP PICK, no NOTES — those bias the user away from the
  //     exact entry instructions at the terminal.
  //   - If frozen bet_ticket is empty: NO STAKE / NO BET TICKET.
  //     Show NO SYSTEM BET + OPTIONAL FORCE PICK (top-3 horses) with
  //     a clear "manual only — not counted as model ROI" footer so
  //     the user has manual guidance without polluting model ROI.
  //
  // Pool labels are DERIVED from bet.pool + combos + legs where
  // possible, not hardcoded. Generator-emitted protection/saver
  // variants keep their intent suffix (PROTECTION/SAVER/BANKER).
  // ─────────────────────────────────────────────────────────────
  function _hkjcPoolLabel(bet) {{
    var p = (bet.pool || '').toUpperCase();
    var L = bet.label || '';
    var n = (bet.legs || []).length;
    // Special generator-emitted intent variants — preserve intent
    // suffix so the user knows why the bet exists.
    if (L === 'QIN PLACE BANKER') return 'QPL BANKER';
    if (L === 'BOX QIN-4 PROTECTION') return 'QIN BOX-4 (PROTECTION)';
    if (L === 'F4 SAVER') return 'FIRST 4 SAVER';
    // Generic derivation: HKJC pool name + STRAIGHT/BOX-N from leg count.
    if (p === 'QUINELLA') return n === 2 ? 'STRAIGHT QIN' : ('QIN BOX-' + n);
    if (p === 'TRIO') return n === 3 ? 'STRAIGHT TRIO' : ('TRIO BOX-' + n);
    if (p === 'FIRST 4') return 'FIRST 4 BOX-' + n;
    if (p === 'QIN PLACE') return 'QPL';
    if (p === 'WIN' || p === 'PLACE') return p;
    return L || p;
  }}

  var ticketLinesCompact = [];
  var hdrVenue = race.venue ? ' · ' + race.venue : '';
  var hdrTime = race.time ? ' · ' + race.time : '';
  var hdrTier = isEdge ? ' · ★ EDGE' : (tier ? ' · ' + tier : '');
  ticketLinesCompact.push('*HKJC R' + race.race_number + hdrVenue + hdrTime + hdrTier + '*');

  var oa = (race.bet_ticket && race.bet_ticket.odds_age_minutes != null)
    ? race.bet_ticket.odds_age_minutes
    : (race.odds_age_minutes != null ? race.odds_age_minutes : null);
  var stale = (oa != null && oa >= 15);
  var statusWord = race.results_available ? 'LOCKED' : 'LIVE';
  if (stale && !race.results_available) statusWord = 'LIVE · STALE ODDS';
  var statusBits = ['STATUS: ' + statusWord];
  // Only surface odds age when it carries information — a freshly-fetched
  // 0m reading is noise. Stale odds (≥15m) are flagged in statusWord anyway.
  if (oa != null && Math.round(oa) > 0) statusBits.push('odds age ' + Math.round(oa) + 'm');
  var modelV = (race.bet_ticket && race.bet_ticket.model_version) || null;
  if (modelV) {{
    // model_version is stamped YYYYMMDD-HHMMSS at training time — surface it
    // as a readable trained-date rather than the raw timestamp string.
    var _isDate = modelV.length >= 8 && /^\\d+$/.test(modelV.slice(0, 8));
    statusBits.push(_isDate
      ? ('model trained ' + modelV.slice(0, 4) + '-' + modelV.slice(4, 6) + '-' + modelV.slice(6, 8))
      : ('model ' + modelV));
  }}
  ticketLinesCompact.push(statusBits.join(' · '));

  var _ticketBetsAll = (race.bet_ticket && race.bet_ticket.bets) || [];
  if (_ticketBetsAll.length) {{
    // ── Real frozen system ticket: STAKE + BET TICKET only ──
    var ssAmt = (ss && ss.amount) ? ss.amount : 0;
    var ticketTotal = (race.bet_ticket && race.bet_ticket.allocated_amount) || 0;
    var unitC = (race.bet_ticket && race.bet_ticket.unit_stake) || 0;
    var overBudC = !!(race.bet_ticket && race.bet_ticket.over_budget);
    var overAmtC = (race.bet_ticket && race.bet_ticket.over_amount) || 0;
    ticketLinesCompact.push('');
    ticketLinesCompact.push('*STAKE*');
    var stakeLine = 'Budget $' + Math.round(ssAmt) +
                    ' · Ticket $' + Math.round(ticketTotal) +
                    ' · Unit $' + Math.round(unitC);
    if (overBudC && overAmtC > 0) {{
      stakeLine += ' · +$' + Math.round(overAmtC) + ' over (HK$' +
                   Math.round(unitC) + ' unit floor)';
    }}
    ticketLinesCompact.push(stakeLine);

    ticketLinesCompact.push('');
    ticketLinesCompact.push('*BET TICKET*');
    _ticketBetsAll.forEach(function(bet, idx) {{
      var poolLabel = _hkjcPoolLabel(bet);
      var amtBit = (bet.amount && bet.amount > 0)
        ? ' · $' + Math.round(bet.amount) : '';
      var comboBit = (bet.combos && bet.combos > 1)
        ? ' · ' + bet.combos + ' combos' : '';
      ticketLinesCompact.push((idx + 1) + '. ' + poolLabel + amtBit +
                              comboBit + _waHitMark(bet.label));
      var nums = (bet.legs || []).map(function(l) {{ return '#' + l.number; }});
      if (nums.length) {{
        if (bet.label === 'QIN PLACE BANKER' && nums.length >= 2) {{
          ticketLinesCompact.push('   ' + nums[0] + ' with ' +
                                  nums.slice(1).join(' + '));
        }} else {{
          ticketLinesCompact.push('   ' + nums.join(' + '));
        }}
      }}
    }});
  }} else {{
    // ── No system bet: drop STAKE/BET TICKET entirely, give the
    // user a clearly-labelled MANUAL guidance block instead. The
    // "Manual only" footer is load-bearing — without it the user
    // might paste this into a ticket and expect it to count toward
    // model ROI accounting.
    ticketLinesCompact.push('');
    ticketLinesCompact.push('*NO SYSTEM BET*');
    var skipReason = 'Tier ' + (race.conviction_tier || 'SKIP') +
                     ' / no frozen ticket';
    ticketLinesCompact.push('Reason: ' + skipReason);
    if (race.horses && race.horses.length) {{
      ticketLinesCompact.push('');
      ticketLinesCompact.push('*OPTIONAL FORCE PICK*');
      race.horses.slice(0, 3).forEach(function(h, i) {{
        var fp = race.horses[i];
        if (!fp) return;
        var prob = (fp.calibrated_prob != null)
          ? Math.round(fp.calibrated_prob * 100) : null;
        var bits = ['#' + fp.number + ' ' + (fp.name || '')];
        if (fp.win_odds) bits.push('$' + fp.win_odds);
        bits.push('rank ' + (fp.rank || (i + 1)));
        if (prob != null) bits.push('prob ' + prob + '%');
        ticketLinesCompact.push((i + 1) + '. ' + bits.join(' · '));
      }});
      ticketLinesCompact.push('');
      ticketLinesCompact.push('Manual only — not counted as model ROI ticket.');
    }}
  }}

  var waPayloadCompact = ticketLinesCompact.join('\\n');

  // Whole card is clickable → toggles the per-race detail panel below it.
  // The WA button uses stopPropagation so clicking Copy WA does not also
  // toggle the expansion. The expanded indicator (▼/▲) lives in the
  // top-right next to Copy WA so the affordance is unmissable.
  // EDGE alert banner — fires <1× per meeting on average (0 in 124-race
  // backtest holdout, +5pp threshold). When it does fire, treat it as a
  // genuine value bet, not a routine HIGH-tier pick. Render a prominent
  // notice above the bet ticket so the user cannot miss it.
  // Name the horse that actually fired EDGE — G4 allows max_model_rank=2 so
  // rank-2 can fire while rank-1 does not. Was hardcoded "top pick" → wrong
  // when rank-2 is the edge horse.
  var edgeRank = race.edge_horse_rank || 1;
  var edgeWho = (edgeRank === 1)
    ? 'top pick'
    : 'rank-' + edgeRank + ' pick #' + esc(race.edge_horse_number) + ' ' + esc(race.edge_horse_name);
  // Suppress the EDGE staking banner when there is no system ticket to
  // anchor (halted day / empty ticket — audit 2026-06-08 #2).
  var _edgeHasTicket = !!(race.bet_ticket && race.bet_ticket.bets && race.bet_ticket.bets.length);
  var edgeAlertHTML = (isEdge && _edgeHasTicket)
    ? '<div style="background:linear-gradient(135deg,rgba(251,191,36,.25),rgba(251,191,36,.08));border:2px solid var(--gold);border-radius:6px;padding:8px 12px;margin:6px 0 10px 0;font-size:.78rem">' +
        '<span style="color:var(--gold);font-weight:800;letter-spacing:.06em">★ RARE EDGE — anchor TRIO-5 with this horse</span>' +
        '<div style="color:var(--fg);margin-top:3px;font-size:.72rem;line-height:1.4">' +
          'Pre-blend model probability ≥ +5pp above market — fires <1× per meeting. ' +
          'The model has found a real mispricing on the ' + edgeWho + '. ' +
          (function() {{
            var es = (typeof EDGE_STATS !== 'undefined' && EDGE_STATS) ? EDGE_STATS : {{}};
            return (es.n_fires && es.n_fires > 0)
              ? '<strong>Historical pattern (N=' + es.n_fires + '): ' + (es.n_placed||0) + ' placed top-3, ' + (es.n_won||0) + ' won.</strong> '
              : '<strong>Historically places more than wins.</strong> ';
          }})() +
          'Use as TRIO-5/F4 anchor — do NOT bet WIN on the EDGE horse.' +
        '</div>' +
      '</div>'
    : '';

  return '<div class="bet-card tier-' + tier + ' card-clickable" id="card-' + race.race_number +
    '" data-testid="race-card-R' + race.race_number + '"' +
    ' data-race-num="' + race.race_number + '" onclick="toggleCardDetail(' + race.race_number + ', event)" ' +
    (isEdge ? 'style="box-shadow:0 0 0 2px var(--gold), 0 4px 14px rgba(251,191,36,.35)" ' : '') +
    'title="Click anywhere on the card to see full per-horse detail, format history, speed map and reasoning">' +
    '<div class="bet-card-top">' +
      '<div>' +
        '<span class="tier-badge" style="background:' + tierColor + (isEdge ? ';box-shadow:0 0 8px rgba(251,191,36,.7)' : '') + '">' + (isEdge ? '★ EDGE · RARE' : esc(tier)) + '</span>' +
        '<strong style="margin-left:8px">' + raceLabel + '</strong>' +
        (strikeBadge ? strikeBadge.replace('<span class="strike-badge', '<span data-testid="strike-badge-R' + race.race_number + '" class="strike-badge') : '') +
      '</div>' +
      '<div style="display:flex;align-items:center;gap:8px">' +
        '<span style="font-size:.7rem;color:var(--dim)">' + esc(edgeDetail) + '</span>' +
        '<button class="wa-btn" data-testid="copy-wa-ticket-R' + race.race_number + '" data-payload="' + escAttr(waPayloadCompact) +
          '" onclick="event.stopPropagation()" title="Compact ticket-first WA copy — exact bet entry, no rationale">Copy WA Ticket</button>' +
        '<button class="wa-btn" data-testid="copy-wa-R' + race.race_number + '" data-payload="' + escAttr(waPayload) +
          '" onclick="event.stopPropagation()" title="Full WA copy — ticket plus rationale, edge, odds age, model info">Copy WA Full</button>' +
        '<span class="card-expand-chevron" id="chev-' + race.race_number + '" style="font-size:.85rem;color:var(--gold);font-weight:700">▼</span>' +
      '</div>' +
    '</div>' +
    edgeAlertHTML +
    '<div class="bet-card-pick">#' + esc(topH.number) + ' ' + esc(topH.name) +
      '<span style="font-weight:400;margin-left:8px;font-size:.82rem">odds ' + esc(topH.win_odds) + ' · ' + esc(betRec) + '</span>' +
    '</div>' +
    '<div class="trust-card">' +
      '<div class="trust-block"><div class="trust-label">Why Strong</div><div class="trust-body">' + whyHtml + '</div></div>' +
      '<div class="trust-block"><div class="trust-label">Our vs Market ' +
        '<span class="info-tip" onclick="showInfoPop(event, &quot;Our vs Market&quot;, &quot;Pre-blend = pure model (form, draw, jockey, etc) — drives Edge. Post-blend = pre-blend mixed with market odds + isotonic calibrator — drives the bet recommendation (QIN/WIN/SKIP). Market = bookmakers implied probability (1 / odds). Edge = pre-blend minus market in percentage points. Positive edge with SKIP recommendation can happen when post-blend agrees with market.&quot;)">ⓘ</span>' +
      '</div>' +
        '<div class="trust-body" style="margin-top:4px;font-size:.72rem;line-height:1.5">' +
          'Pre-blend <strong style="color:var(--green)">' + purePct + '%</strong> ' +
            '<span style="color:var(--dim);font-size:.65rem">(drives Edge)</span><br>' +
          'Post-blend <strong style="color:#60a5fa">' + calPct + '%</strong> ' +
            '<span style="color:var(--dim);font-size:.65rem">(drives bet)</span><br>' +
          'Market <strong style="color:var(--dim)">' + mktPct + '%</strong><br>' +
          'Edge <strong style="color:var(--gold)">' + edgePp + 'pp</strong>' +
        '</div>' +
      '</div>' +
      '<div class="trust-block"><div class="trust-label">Tier Track Record</div><div class="trust-body" style="margin-top:4px">' + trackHtml + '</div></div>' +
      stakeHtml +
    '</div>' +
  '</div>';
}}

// ─────────────────────────────────────────────────────────────
// WhatsApp copy
// ─────────────────────────────────────────────────────────────
function copyWA(text, btn) {{
  // Per-card WA copy. Two failure paths must both work:
  //   1. navigator.clipboard undefined on file:// + non-Chromium
  //   2. clipboard promise rejects under Safari focus rules / sandboxed iframes
  // The bulk Copy-All-bets handler already had the full execCommand +
  // window.prompt fallback ladder; the per-card path silently failed
  // until 2026-05-29 due to a missing .catch. Locked by
  // tests/test_template_js_syntax.py.
  var fullText = text + '\\n_⚠️ For entertainment only. Gamble responsibly._';
  var orig = btn.textContent;
  var ok = function() {{
    btn.textContent = 'Copied!';
    btn.style.background = 'var(--green)';
    btn.style.color = '#000';
    setTimeout(function() {{
      btn.textContent = orig;
      btn.style.background = '';
      btn.style.color = '';
    }}, 2000);
  }};
  var fallback = function() {{
    try {{
      var ta = document.createElement('textarea');
      ta.value = fullText;
      ta.style.position = 'fixed'; ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.focus(); ta.select();
      var success = document.execCommand('copy');
      document.body.removeChild(ta);
      if (success) {{ ok(); return; }}
    }} catch(e) {{}}
    btn.textContent = 'Copy failed';
    setTimeout(function() {{ btn.textContent = orig; }}, 1800);
    try {{ window.prompt('Copy manually (Cmd+C / Ctrl+C):', fullText); }} catch(e) {{}}
  }};
  if (navigator.clipboard && navigator.clipboard.writeText) {{
    navigator.clipboard.writeText(fullText).then(ok).catch(fallback);
  }} else {{
    fallback();
  }}
}}

// ─────────────────────────────────────────────────────────────
// Race strip
// ─────────────────────────────────────────────────────────────
// Set of race_numbers currently expanded in #race-detail.
// Multiple races can be expanded simultaneously (accordion behaviour).
var _activeRaces = new Set();

var _stripView = 'race';  // 'race' (race-order) or 'bets' (top-bets)

function _stripTierWeight(tier) {{
  // Higher = more conviction; used to sort under "Top bets" view.
  if (tier === 'EDGE') return 5;
  if (tier === 'HIGH') return 4;
  if (tier === 'MED')  return 3;
  if (tier === 'LOW')  return 2;
  return 1;  // SKIP
}}

function _topPickEdge(r) {{
  var h = (r.horses && r.horses[0]) ? r.horses[0] : {{}};
  return h.edge_pp != null ? h.edge_pp : -100;
}}

function toggleCardDetail(raceNum, event) {{
  // Click on a bet-card → toggle a detail panel inserted right after it.
  // The detail panel reuses _buildRaceDetailPanel so it has the same
  // per-horse table, format history, speed map and reasoning as the
  // legacy strip-tile expansion did.
  if (event) event.stopPropagation();

  var card = document.getElementById('card-' + raceNum);
  if (!card) return;
  var existingDetail = document.getElementById('card-detail-' + raceNum);
  var chev = document.getElementById('chev-' + raceNum);

  if (existingDetail) {{
    existingDetail.remove();
    if (chev) chev.textContent = '▼';
    card.classList.remove('card-expanded');
    return;
  }}

  var race = null;
  for (var i=0; i<RACES.length; i++) {{ if (RACES[i].race_number === raceNum) {{ race = RACES[i]; break; }} }}
  if (!race) return;

  var wrapper = document.createElement('div');
  wrapper.id = 'card-detail-' + raceNum;
  wrapper.className = 'card-detail-inline';
  wrapper.innerHTML = _buildRaceDetailPanel(race);
  card.parentNode.insertBefore(wrapper, card.nextSibling);
  if (chev) chev.textContent = '▲';
  card.classList.add('card-expanded');

  wrapper.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}

function setStripView(view) {{
  _stripView = view;
  document.getElementById('tab-race-order').classList.toggle('active', view === 'race');
  document.getElementById('tab-top-bets').classList.toggle('active', view === 'bets');

  // End-of-meeting auto-switch: if all races are finished, default the
  // Top Bets sub-toggle to past mode so the user sees the day record
  // instead of an empty Next pane.
  if (view === 'bets') {{
    var allFinished = RACES.length > 0 && RACES.every(function(r) {{ return r.results_available; }});
    if (allFinished) {{
      _topBetsFilter = 'past';
    }}
  }}

  // Both views render their content into #hero via renderHero() (cards
  // for all 11 races in race-order, or curated cards for bets).
  // The strip + race-detail panels are hidden in both — cards in the
  // hero now carry all the actionable info inline. The Top Bets
  // dedicated container is also hidden because the hero now handles
  // both modes (kept in DOM for backwards-compat with prior snapshots).
  var stripSection = document.getElementById('strip-section');
  var topBetsView = document.getElementById('top-bets-view');
  var raceDetail = document.getElementById('race-detail');
  var expandBtn = document.getElementById('expand-all-btn');
  if (stripSection) stripSection.style.display = 'none';
  if (raceDetail) raceDetail.style.display = 'none';
  if (expandBtn) expandBtn.style.display = 'none';
  if (topBetsView) topBetsView.style.display = 'none';

  renderHero();
}}

// LONGSHOT-TRIO + BOLTER TRIO removed from live UI 2026-05-25 per user
// direction. Tracker scripts (scripts/track_longshot_trio.py) keep
// logging silently; surface results only in post-race reports. See:
//   feedback_hide_paper_experiments_from_live_ui_2026-05-25
//   project_bolter_calc_hot_killed_2026-05-24 (variant 1 KILL @ N=7)

function renderTopBetsView() {{
  // Stacked betting-opportunity cards: every race that has a non-SKIP bet,
  // sorted by tier strength → top-pick edge_pp → race_number. Each card
  // shows the BET TICKET, top picks, and "View full race" button.
  var container = document.getElementById('top-bets-view');
  if (!container) return;

  var actionable = RACES.filter(function(r) {{
    return r.horses && r.horses.some(function(h) {{
      var b = (h.bet_recommendation || '').trim();
      return b && b !== 'SKIP';
    }});
  }});
  actionable.sort(function(a, b) {{
    var ta = _stripTierWeight(a.conviction_tier || 'SKIP');
    var tb = _stripTierWeight(b.conviction_tier || 'SKIP');
    if (ta !== tb) return tb - ta;
    var ea = _topPickEdge(a), eb = _topPickEdge(b);
    if (ea !== eb) return eb - ea;
    return a.race_number - b.race_number;
  }});

  if (actionable.length === 0) {{
    container.innerHTML =
      '<div style="background:rgba(71,85,105,.18);border:1px solid var(--border);' +
      'padding:24px;border-radius:8px;text-align:center;color:var(--dim);font-size:.95rem">' +
        '<div style="font-size:1.6rem;margin-bottom:8px">🤷</div>' +
        '<strong>No bets meet the conviction threshold today.</strong><br>' +
        '<span style="font-size:.8rem">Switch to Race Order to browse all races anyway.</span>' +
      '</div>';
    return;
  }}

  container.innerHTML = actionable.map(function(r) {{
    var tier = r.conviction_tier || 'SKIP';
    var tierColor = TIER_COLORS[tier] || '#64748b';

    // Group horses by bet type for a clean ticket synthesis
    var bets = {{}};
    r.horses.forEach(function(h) {{
      var b = (h.bet_recommendation || '').trim();
      if (b && b !== 'SKIP') {{ (bets[b] = bets[b] || []).push(h); }}
    }});
    var ticketLines = Object.keys(bets).map(function(betType) {{
      var picks = bets[betType];
      var label = betType;
      if ((betType === 'QIN' || betType === 'BOX QIN') && picks.length >= 2) {{
        label = picks.length === 2 ? 'STRAIGHT QIN' : 'BOX QIN-' + picks.length;
      }} else if (betType === 'TRIO' && picks.length >= 3) {{
        label = picks.length === 3 ? 'STRAIGHT TRIO' : 'BOX TRIO-' + picks.length;
      }}
      var ids = picks.map(function(h) {{
        return '<span style="display:inline-block;background:rgba(15,23,42,.7);padding:2px 8px;border-radius:4px;margin:2px 4px 2px 0;font-size:.82rem">' +
          '#' + esc(h.number) + ' ' + esc(h.name) +
          ' <span style="color:var(--dim);font-size:.72rem">@' + esc(h.win_odds) + '</span>' +
        '</span>';
      }}).join('');
      return '<div style="margin:6px 0">' +
        '<span style="display:inline-block;background:var(--gold);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' + esc(label) + '</span>' +
        ids +
      '</div>';
    }}).join('');

    return '<div class="top-bet-card" style="background:rgba(30,41,59,.6);border:1px solid var(--border);border-left:4px solid ' + tierColor + ';border-radius:8px;padding:14px 18px;margin-bottom:10px">' +
      '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">' +
        '<div>' +
          '<span style="font-size:1.05rem;font-weight:800">R' + r.race_number + '</span>' +
          '<span style="margin-left:10px;color:var(--dim);font-size:.78rem">' +
            esc(r.time||'') + ' · ' + esc(r.distance) + 'm · ' + esc(r.class_type||'') + ' · ' + esc(r.going||'') +
          '</span>' +
        '</div>' +
        '<span style="font-size:.7rem;padding:3px 10px;border-radius:4px;background:' + tierColor + ';color:#000;font-weight:800">' + esc(tier) + '</span>' +
      '</div>' +
      ticketLines +
      '<button onclick="setStripView(&quot;race&quot;); showRaceDetail(' + r.race_number + ')" ' +
        'style="margin-top:8px;background:transparent;border:1px solid var(--border);color:var(--fg);padding:5px 12px;border-radius:4px;font-size:.74rem;cursor:pointer">' +
        'View full race →</button>' +
    '</div>';
  }}).join('');
}}

function renderStrip() {{
  var strip = document.getElementById('strip');
  var ordered = RACES.slice();
  if (_stripView === 'bets') {{
    // Sort: tier weight DESC, then edge_pp DESC, then race_number ASC
    ordered.sort(function(a, b) {{
      var ta = _stripTierWeight(a.conviction_tier || 'SKIP');
      var tb = _stripTierWeight(b.conviction_tier || 'SKIP');
      if (ta !== tb) return tb - ta;
      var ea = _topPickEdge(a), eb = _topPickEdge(b);
      if (ea !== eb) return eb - ea;
      return a.race_number - b.race_number;
    }});
  }}
  strip.innerHTML = ordered.map(function(r) {{
    var tier = r.conviction_tier || 'SKIP';
    var label = tier === 'EDGE' ? 'R' + r.race_number + ' ★' : 'R' + r.race_number;
    var sub = tier === 'SKIP' ? 'skip' : (r.horses[0] ? (r.horses[0].bet_recommendation || '') : '');
    return '<div class="strip-tile tier-' + tier + '" id="tile-' + r.race_number + '" onclick="showRaceDetail(' + r.race_number + ')" title="Click to expand · ' + esc(tier) + '">' +
      '<div class="strip-num">' + esc(label) + '</div>' +
      '<div class="strip-sub">' + esc(sub) + '</div>' +
      '<div class="strip-chevron">▼</div>' +
    '</div>';
  }}).join('');
}}

// ─────────────────────────────────────────────────────────────
// Race detail (inline below strip)
// ─────────────────────────────────────────────────────────────
function _buildRaceDetailPanel(race) {{
  if (!race.horses || !race.horses.length) return '<div class="muted">No runner data.</div>';
  var rowsHtml = race.horses.map(function(h, idx) {{
    var edgeClass = (h.edge_pp > 0) ? 'edge-positive' : 'edge-neutral';
    var edgeStr = (h.edge_pp != null) ? (h.edge_pp > 0 ? '+' : '') + h.edge_pp.toFixed(1) : '—';
    var purePct = h.pure_prob != null ? (h.pure_prob*100).toFixed(1)+'%' : '—';
    var mktPct  = h.market_prob != null ? (h.market_prob*100).toFixed(1)+'%' : '—';
    var betRec  = h.bet_recommendation || 'SKIP';
    var isTop   = idx === 0;
    return '<tr' + (isTop ? ' class="top-pick"' : '') + '>' +
      '<td style="color:var(--dim)">' + esc(h.rank || idx+1) + '</td>' +
      '<td><strong>#' + esc(h.number) + '</strong> ' + esc(h.name) +
        '<div style="font-size:.66rem;color:var(--dim)">' + esc(h.jockey||'') + ' / ' + esc(h.trainer||'') + '</div>' +
      '</td>' +
      '<td style="text-align:right">' + esc(purePct) + '</td>' +
      '<td style="text-align:right">' + esc(mktPct) + '</td>' +
      '<td style="text-align:right" class="' + edgeClass + '">' + esc(edgeStr) + '</td>' +
      '<td style="text-align:right;color:var(--cyan)">$' + esc(h.win_odds||'—') + '</td>' +
      '<td>' + esc(betRec) + '</td>' +
    '</tr>';
  }}).join('');

  // Per-horse detail table (spec §5 KEEP: last_6, horse_weight, days_since_last_run,
  // rating_change, going_form, jockey_horse_stats, distance_pb)
  var detailFields = [
    {{key: 'last_6', label: 'L6', render: function(h){{ return esc(h.last_6 || ''); }}}},
    {{key: 'horse_weight', label: 'WT', render: function(h){{ return h.horse_weight != null ? h.horse_weight : '—'; }}}},
    {{key: 'days_since_last_run', label: 'DSLR', render: function(h){{ return h.days_since_last_run != null ? h.days_since_last_run : '—'; }}}},
    {{key: 'rating_change', label: 'ΔR', render: function(h){{ return h.rating_change != null ? h.rating_change : '—'; }}}},
    {{key: 'going_form', label: 'GOING', render: function(h){{
      if (!h.going_form) return '—';
      if (typeof h.going_form === 'object' && h.going_form.wins != null && h.going_form.runs != null) {{
        return h.going_form.wins + '/' + h.going_form.runs;
      }}
      return esc(String(h.going_form));
    }}}},
    {{key: 'jockey_horse_stats', label: 'J×H', render: function(h){{
      if (!h.jockey_horse_stats) return '—';
      var j = h.jockey_horse_stats;
      if (typeof j === 'object' && j.starts != null) {{
        return (j.wins || 0) + '/' + j.starts;
      }}
      return esc(String(j));
    }}}},
    {{key: 'distance_pb', label: 'PB', render: function(h){{
      if (!h.distance_pb) return '—';
      if (typeof h.distance_pb === 'object' && h.distance_pb.time != null) {{
        return esc(String(h.distance_pb.time));
      }}
      return esc(String(h.distance_pb));
    }}}},
  ];
  var activeCols = detailFields.filter(function(col) {{
    return race.horses.some(function(h) {{
      var v = h[col.key];
      return v != null && v !== '' && !(typeof v === 'object' && Object.keys(v).length === 0);
    }});
  }});
  var detailTableHTML = '';
  if (activeCols.length) {{
    detailTableHTML = '<div class="detail-section">' +
      '<div class="section-label">PER-HORSE DETAIL</div>' +
      '<table style="width:100%;border-collapse:collapse;font-size:.76rem">' +
      '<thead><tr><th style="text-align:left;padding:4px">#</th><th style="text-align:left;padding:4px">NAME</th>' +
        activeCols.map(function(col) {{
          return '<th style="text-align:left;padding:4px">' + esc(col.label) + '</th>';
        }}).join('') +
      '</tr></thead><tbody>' +
      race.horses.map(function(h) {{
        return '<tr>' +
          '<td style="padding:4px">' + esc(h.number) + '</td>' +
          '<td style="padding:4px">' + esc(h.name) + '</td>' +
          activeCols.map(function(col) {{
            return '<td style="padding:4px">' + col.render(h) + '</td>';
          }}).join('') +
        '</tr>';
      }}).join('') +
      '</tbody></table></div>';
  }}

  // Speed map block
  var speedMapHTML = '';
  var sm = race.speed_map;
  if (sm) {{
    if (Array.isArray(sm) && sm.length) {{
      // Production shape — List[Dict] from analytics/pace.generate_speed_map()
      speedMapHTML = '<div class="detail-section">' +
        '<div class="section-label">SPEED MAP</div>' +
        '<table style="width:100%;border-collapse:collapse;font-size:.78rem"><thead><tr>' +
          '<th style="text-align:left;padding:4px">#</th>' +
          '<th style="text-align:left;padding:4px">NAME</th>' +
          '<th style="text-align:left;padding:4px">STYLE</th>' +
          '<th style="text-align:right;padding:4px">POS</th>' +
          '<th style="text-align:right;padding:4px">DRAW</th>' +
        '</tr></thead><tbody>' +
        sm.map(function(entry) {{
          return '<tr>' +
            '<td style="padding:4px">' + esc(entry.number) + '</td>' +
            '<td style="padding:4px">' + esc(entry.name) + '</td>' +
            '<td style="padding:4px">' + esc(entry.style || '') + '</td>' +
            '<td style="padding:4px;text-align:right">' + esc(entry.predicted_pos != null ? entry.predicted_pos : '') + '</td>' +
            '<td style="padding:4px;text-align:right">' + esc(entry.draw != null ? entry.draw : '') + '</td>' +
          '</tr>';
        }}).join('') +
        '</tbody></table></div>';
    }} else if (typeof sm === 'object') {{
      // Legacy/test shape — Dict[str, Any]
      var speedKeys = Object.keys(sm);
      if (speedKeys.length) {{
        speedMapHTML = '<div class="detail-section">' +
          '<div class="section-label">SPEED MAP</div>' +
          speedKeys.map(function(k) {{
            var v = sm[k];
            var inner;
            if (Array.isArray(v)) {{
              inner = v.map(function(x) {{ return esc(String(x)); }}).join(', ');
            }} else {{
              inner = esc(String(v));
            }}
            return '<div class="sm-row"><strong>' + esc(k) + ':</strong> ' + inner + '</div>';
          }}).join('') +
        '</div>';
      }}
    }}
  }}

  // Per-horse reasoning block
  var reasoningHTML = race.horses.filter(function(h) {{
    return (h.reasoning || []).length > 0;
  }}).map(function(h) {{
    return '<div class="rsn-row">' +
      '<strong>#' + esc(h.number) + ' ' + esc(h.name) + '</strong>' +
      '<ul>' + (h.reasoning || []).map(function(r) {{
        return '<li>' + esc(r) + '</li>';
      }}).join('') + '</ul>' +
    '</div>';
  }}).join('');
  var reasoningBlock = reasoningHTML
    ? '<div class="detail-section"><div class="section-label">REASONING</div>' + reasoningHTML + '</div>'
    : '';

  // Same-format history block. For every horse with prior runs at the
  // current venue and distance, render their last 5 attempts so the user
  // can see what happened in identical conditions, inline.
  var formatHistRows = race.horses.filter(function(h) {{
    return h.format_history && h.format_history.length > 0;
  }}).map(function(h) {{
    var rows = h.format_history.map(function(p) {{
      var pos = p.finish_position;
      var posStyle = pos === 1 ? 'color:var(--gold);font-weight:700'
                   : (pos > 0 && pos <= 3) ? 'color:var(--green);font-weight:600'
                   : 'color:var(--dim)';
      return '<tr>' +
        '<td style="padding:3px 6px">' + esc(p.race_date||'') + '</td>' +
        '<td style="padding:3px 6px">' + esc(p.going||'') + '</td>' +
        '<td style="padding:3px 6px">' + esc(p.class_type||'') + '</td>' +
        '<td style="padding:3px 6px;text-align:right;' + posStyle + '">' +
          (pos > 0 ? esc(pos) : '–') + '</td>' +
        '<td style="padding:3px 6px;text-align:right">' +
          (p.win_odds ? esc(p.win_odds) : '–') + '</td>' +
        '<td style="padding:3px 6px;text-align:right;color:var(--dim)">' +
          (p.margin > 0 ? esc(p.margin) + 'L' : (pos === 1 ? 'won' : '–')) + '</td>' +
        '<td style="padding:3px 6px;color:var(--dim)">' + esc(p.jockey||'') + '</td>' +
      '</tr>';
    }}).join('');
    return '<tr><td colspan="7" style="padding:6px 4px 2px 4px;font-weight:700;color:var(--gold);font-size:.78rem">' +
        '#' + esc(h.number) + ' ' + esc(h.name) +
        ' <span style="color:var(--dim);font-weight:400;font-size:.72rem">' +
          '(' + h.format_history.length + ' prior run' + (h.format_history.length === 1 ? '' : 's') +
          ' at ' + esc(race.venue||'') + ' ' + esc(race.distance) + 'm)' +
        '</span></td></tr>' + rows;
  }}).join('');

  var formatHistoryHTML = '';
  if (formatHistRows) {{
    formatHistoryHTML = '<div class="detail-section">' +
      '<div class="section-label">SAME-FORMAT HISTORY ' +
        '<span class="info-tip" onclick="showInfoPop(event, &quot;Same-format history&quot;, &quot;Past runs by these horses at the SAME venue and SAME distance. Position colours: gold = win, green = top-3, grey = other. Use this to spot horses that consistently perform (or struggle) under todays exact conditions.&quot;)">ⓘ</span>' +
      '</div>' +
      '<table style="width:100%;border-collapse:collapse;font-size:.74rem">' +
        '<thead><tr style="border-bottom:1px solid var(--border);color:var(--dim)">' +
          '<th style="text-align:left;padding:3px 6px">DATE</th>' +
          '<th style="text-align:left;padding:3px 6px">GOING</th>' +
          '<th style="text-align:left;padding:3px 6px">CLASS</th>' +
          '<th style="text-align:right;padding:3px 6px">POS</th>' +
          '<th style="text-align:right;padding:3px 6px">ODDS</th>' +
          '<th style="text-align:right;padding:3px 6px">MARGIN</th>' +
          '<th style="text-align:left;padding:3px 6px">JOCKEY</th>' +
        '</tr></thead><tbody>' + formatHistRows + '</tbody>' +
      '</table>' +
    '</div>';
  }}

  // BET TICKET block — shows the recommender primary bet plus an
  // always-on BOX Q4 protection line whenever the race has 4+ horses
  // ranked. Each pick surfaces its jockey + 30-day strike rate so the
  // user can see at a glance who is riding hot.
  function _renderPick(h) {{
    var pct30 = h.jockey_form_30d != null ? Math.round(h.jockey_form_30d * 100) : null;
    var pctAll = h.jockey_strike_rate != null ? Math.round(h.jockey_strike_rate * 100) : null;
    var jockeyTag = h.jockey
      ? '<span style="color:var(--dim);font-size:.7rem;margin-left:4px">' +
          'J: ' + esc(h.jockey) +
          (pct30 != null && pct30 > 0 ? ' <span style="color:var(--gold)">' + pct30 + '%</span> 30d' : '') +
          (pctAll != null && pctAll > 0 ? ' · ' + pctAll + '% all-time' : '') +
        '</span>'
      : '';
    return '<span style="display:inline-block;background:rgba(15,23,42,.6);padding:3px 8px;border-radius:4px;margin:2px 4px 2px 0;font-size:.82rem">' +
      '#' + esc(h.number) + ' ' + esc(h.name) +
      ' <span style="color:var(--dim);font-size:.72rem">@' + esc(h.win_odds || '?') + '</span>' +
      jockeyTag +
    '</span>';
  }}

  // Server-side bet_ticket is the single source of truth for which bet
  // blocks render. Added 2026-05-24 (GPT-5.5 Playwright audit): the
  // PRIMARY block + the gated extras below all read from race.bet_ticket
  // so the server-side build_bet_ticket is the only place that decides
  // what gets staked, displayed, and WA-copied (closes R8/R11 phantom-hit class).
  var _ticketBets = (race.bet_ticket && race.bet_ticket.bets) || [];
  var _ticketLabels = _ticketBets.map(function(b) {{ return b.label; }});
  function _ticketHas(label) {{ return _ticketLabels.indexOf(label) >= 0; }}
  // Per-bet ✓/✗ chip — surfaces hit/miss on each ticket line so the user
  // sees which of N bets hit (the card-header STRIKE badge can only show
  // the highest-priority single hit; this gives the full picture inside
  // the BET TICKET block). Added 2026-05-25.
  function _ticketHitMark(label) {{
    var bet = _ticketBets.filter(function(b) {{ return b.label === label; }})[0];
    if (!bet) return '';
    var hit = _betHitsTicket(bet, race);
    if (hit === null) return '';
    return hit
      ? '<span style="background:rgba(34,197,94,.95);color:#000;font-weight:800;padding:1px 6px;border-radius:3px;font-size:.62rem;margin-left:8px;vertical-align:middle">✓ HIT</span>'
      : '<span style="background:rgba(239,68,68,.7);color:#fff;font-weight:800;padding:1px 6px;border-radius:3px;font-size:.62rem;margin-left:8px;vertical-align:middle">✗ MISS</span>';
  }}
  function _legsToHorses(legs) {{
    // Re-hydrate full horse objects from the frozen ticket leg numbers
    // so _renderPick has odds, jockey, edge, etc. Resolve by NUMBER
    // first (stable across rank shifts) then fall back to rank
    // (GPT-5.5 R13 #6 2026-05-25 — visual must render from ticket legs,
    // not current rank order, so display matches what replay scores).
    var byNum = {{}};
    var byRank = {{}};
    race.horses.forEach(function(h) {{
      if (h.number != null) byNum[h.number] = h;
      if (h.rank != null) byRank[h.rank] = h;
    }});
    return legs.map(function(l) {{
      return byNum[l.number] || byRank[l.rank] || {{
        number: l.number, name: l.name, rank: l.rank,
      }};
    }});
  }}
  // _ticketLegsFor — return [{{number, name, rank, win_odds, ...}}] for the
  // first bet matching `label` in the frozen ticket. Empty array when the
  // label is absent (caller should also gate on _ticketHas).
  function _ticketLegsFor(label) {{
    var bet = _ticketBets.filter(function(b) {{ return b.label === label; }})[0];
    return bet ? _legsToHorses(bet.legs || []) : [];
  }}
  var ticketLines = [];
  // PRIMARY block — render every recommender-driven bet from the frozen
  // ticket. Each `reason='recommender'` entry is one WIN/PLACE/QIN/TRIO
  // bet that the server-side recommender chose.
  _ticketBets.forEach(function(bet) {{
    if (bet.reason !== 'recommender') return;
    ticketLines.push(
      '<div style="margin:6px 0">' +
        '<span style="display:inline-block;background:var(--gold);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' +
          'PRIMARY · ' + esc(bet.label) +
        '</span>' +
        _ticketHitMark(bet.label) +
        _legsToHorses(bet.legs || []).map(_renderPick).join('') +
      '</div>'
    );
  }});

  // QIN PLACE BANKER — for short-favorite races. Read legs from the
  // frozen ticket so display matches what replay scores (GPT-5.5 R13 #6).
  if (_ticketHas('QIN PLACE BANKER')) {{
    var qplLegs = _ticketLegsFor('QIN PLACE BANKER');
    if (qplLegs.length >= 2) {{
      var qplPivot = qplLegs[0];
      var qplBankers = qplLegs.slice(1);
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25)">' +
          '<span style="display:inline-block;background:rgba(59,130,246,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' +
            '+ QIN PLACE BANKER · 3 combos' +
          '</span>' +
          _ticketHitMark('QIN PLACE BANKER') +
          '<span style="font-weight:600">#' + esc(qplPivot.number) + ' ' + esc(qplPivot.name) + ' </span>' +
          '<span style="color:var(--dim);margin:0 4px">with</span>' +
          qplBankers.map(function(h) {{ return '#' + esc(h.number) + ' ' + esc(h.name); }}).join(' / ') +
          '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">Hits when fav places (very likely at $' + esc(qplPivot.win_odds || '?') + ') and any banker places. Way better ROI than straight WIN.</div>' +
        '</div>'
      );
    }}
  }}

  // BOX QIN-4 PROTECTION — legs from frozen ticket.
  if (_ticketHas('BOX QIN-4 PROTECTION')) {{
    var box4Legs = _ticketLegsFor('BOX QIN-4 PROTECTION');
    if (box4Legs.length === 4) {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25)">' +
          '<span style="display:inline-block;background:rgba(16,185,129,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' +
            '+ PROTECTION · BOX QIN-4 (6 combos)' +
          '</span>' +
          _ticketHitMark('BOX QIN-4 PROTECTION') +
          box4Legs.map(_renderPick).join('') +
        '</div>'
      );
    }}
  }}

  // F4 SAVER — FIRST 4 ANY-ORDER 1-unit. Legs from frozen ticket.
  if (_ticketHas('F4 SAVER')) {{
    var f4SaverLegs = _ticketLegsFor('F4 SAVER');
    if (f4SaverLegs.length === 4) {{
      var f4Nums = f4SaverLegs.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ');
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem">' +
          '<span style="display:inline-block;background:rgba(168,85,247,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
            'F4 SAVER · 1 unit' +
          '</span>' +
          _ticketHitMark('F4 SAVER') +
          '<span style="color:var(--fg);font-weight:600">' + f4Nums + '</span>' +
          '<span style="color:var(--dim);font-size:.7rem;margin-left:8px">(top-4 any order — FIRST 4 pool)</span>' +
        '</div>'
      );
    }}
  }}

  // Rank-5 swap REVERTED 2026-05-17 — see WA-path comment above for rationale.
  // Keep null-shaped vars so downstream blocks (F4 BOX-5, TRIO-5) compile.
  var top5_pre = race.horses.slice(0, 5);
  var top5_eff = top5_pre;
  var top5_swap = null;
  var top5_swap_out = null;
  var swapBadge = '';

  // F4 BOX-5 — legs from frozen ticket (GPT-5.5 R13 #6).
  if (_ticketHas('F4 BOX-5')) {{
    var f4BoxLegs = _ticketLegsFor('F4 BOX-5');
    if (f4BoxLegs.length === 5) {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem">' +
          '<span style="display:inline-block;background:linear-gradient(135deg,rgba(168,85,247,.95),rgba(236,72,153,.85));color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
            '★ MAIN · F4 BOX-5 · 5 combos' +
          '</span>' +
          _ticketHitMark('F4 BOX-5') +
          swapBadge +
          f4BoxLegs.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ') +
          '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">Hits when the top-4 finishers are all inside these 5. Backtest: 17% hit, +54% ROI on 60 races (median dividend ~$461 per hit, 9× the $50 stake).</div>' +
        '</div>'
      );
    }}
  }}

  // BOX TRIO-5 — legs from frozen ticket (GPT-5.5 R13 #6).
  if (_ticketHas('BOX TRIO-5')) {{
    var t5Legs = _ticketLegsFor('BOX TRIO-5');
    if (t5Legs.length === 5) {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem">' +
          '<span style="display:inline-block;background:rgba(236,72,153,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
            '+ BOX TRIO-5 · 10 combos' +
          '</span>' +
          _ticketHitMark('BOX TRIO-5') +
          swapBadge +
          t5Legs.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ') +
          '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">Hits when any 3 of these 5 finish top-3 in any order. Bigger field coverage than BOX QIN-4 — catches the rank-5/6 spoilers.</div>' +
        '</div>'
      );
    }}
  }}

  // BOX TRIO-6 — OPTIONAL high-cost protection layer.
  // Empirical evidence across 6 meetings (Apr 12-29, 60 races):
  //   ST (42 races): TRIO-5 48% → TRIO-6 57% (+10pp, +4 strikes)
  //   HV (18 races): TRIO-5 22% → TRIO-6 33% (+11pp, +2 strikes)
  // ~10pp lift at both venues, but 2× stake. Break-even vs TRIO-5
  // requires avg TRIO dividend ≥ $111. Odds clustering does NOT predict
  // which races will be chaos (tested across 6 meetings — no signal).
  // Real chaos signals: field size ≥11, no clear favorite (rank-1 odds
  // > $3), multiple morning-steam horses, going changed mid-day.
  // Spoiler-swap detection: scan horses ranked 7-12 for a candidate with
  // BOTH (a) market odds materially shorter than rank-6 (≥20% shorter) AND
  // (b) gear visible (B/TT/CP/etc — gear-change signal). If found, swap
  // it into the rank-6 slot in the displayed TRIO-6 box. Validated by
  // R5 2026-05-03 #8 FLYING KNIGHT (rank 9, $7.4 vs rank-6 $11, gear
  // B/TT) which won at $8 — would have hit TRIO-6 with the swap.
  // TRIO-6 legs come from the FROZEN bet_ticket (2026-05-25, GPT-5.5
  // Finding #2). Server-side _pick_trio6_top6 chose the swap candidate
  // (manual override or auto longshot) — JS just reads it so display
  // matches what validate_hk --replay scores. swapMeta drives the badge.
  var _t6Bet = _ticketBets.find(function(b) {{ return b.label === 'BOX TRIO-6'; }});
  var top6 = _t6Bet
    ? _legsToHorses(_t6Bet.legs || [])
    : race.horses.slice(0, 6);
  var _swap = (_t6Bet && _t6Bet.swap) || null;
  var byNumLookup = {{}};
  race.horses.forEach(function(h) {{ byNumLookup[h.number] = h; }});
  var swapCandidate = (_swap && _swap.kind === 'auto') ? (byNumLookup[_swap.in_number] || null) : null;
  var swappedOut = (_swap && _swap.kind === 'auto') ? (byNumLookup[_swap.out_number] || null) : null;
  var manualCandidate = (_swap && _swap.kind === 'manual') ? (byNumLookup[_swap.in_number] || null) : null;
  var manualSwappedOut = (_swap && _swap.kind === 'manual') ? (byNumLookup[_swap.out_number] || null) : null;
  if (top6.length === 6) {{
    // Single source of truth: server-side _compute_trio6_decision in
    // dashboard/generator.py (race.trio6_decision). Before consolidation
    // (2026-05-29) this site recomputed independently and had drifted —
    // the staked ticket and the displayed verdict could disagree.
    var t6dec = race.trio6_decision || {{}};
    var top1Odds = t6dec.top1_odds || ((race.horses[0] && race.horses[0].win_odds) || 0);
    var fieldSize = t6dec.field_size || race.horses.length;
    var tier = race.conviction_tier || '';
    var steamCount = t6dec.steam_count || 0;
    var chaosScore = t6dec.chaos_score || 0;
    var chaosFactors = t6dec.chaos_factors || [];
    var chaosColor = chaosScore >= 3 ? 'rgba(244,63,94,.85)' :
                     chaosScore >= 2 ? 'rgba(251,191,36,.7)' :
                     'rgba(148,163,184,.3)';
    var chaosTextColor = chaosScore >= 3 ? '#fff' : (chaosScore >= 2 ? '#000' : 'var(--dim)');
    var chaosBadge = '<span style="background:' + chaosColor + ';color:' + chaosTextColor + ';padding:1px 6px;border-radius:3px;font-size:.62rem;font-weight:700;margin-left:6px" title="' + escAttr(chaosFactors.join(' · ')) + '">CHAOS ' + chaosScore + '/4</span>';

    // Derive the badge + reason from the verdict label produced by
    // _compute_trio6_decision (server-side). Keep visual styling here —
    // the server only computes the label, never markup.
    var verdict = t6dec.verdict_label || 'YOUR CALL';
    var rec, recBadge, recReason;
    if (verdict.indexOf('SKIP') === 0) {{
      rec = 'SKIP';
      recBadge = '<span style="background:rgba(148,163,184,.3);color:var(--dim);padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:700">SKIP — heavy fav ($' + top1Odds.toFixed(1) + ')</span>';
      recReason = 'Top pick is heavy favorite — podium concentrates in TOP-3, TRIO-6 wastes 2× cost.';
    }} else if (verdict.indexOf('★ TAKE') === 0 && steamCount >= 2) {{
      rec = 'TAKE';
      recBadge = '<span style="background:rgba(34,197,94,.85);color:#000;padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:800">★ TAKE — ' + steamCount + ' steam horses</span>';
      recReason = 'Multiple morning-steam horses — smart money disagrees with model, rank-6 spoiler likely.';
    }} else if (verdict.indexOf('★ TAKE') === 0) {{
      rec = 'TAKE';
      recBadge = '<span style="background:rgba(34,197,94,.85);color:#000;padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:800">★ TAKE — HIGH+big field+$' + top1Odds.toFixed(1) + ' fav</span>';
      recReason = 'HIGH conviction + field ≥11 + competitive favorite — chaos likely, hedge worth it.';
    }} else {{
      rec = 'USER';
      recBadge = '<span style="background:rgba(148,163,184,.4);color:#fff;padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:700">YOUR CALL</span>';
      recReason = 'Coin-flip economically. +10pp strike lift but 2× cost. Take if you sense chaos; skip for ROI.';
    }}

    var swapBadge = '';
    var swapNote = '';
    if (manualCandidate && manualSwappedOut) {{
      swapBadge = '<span style="background:rgba(245,158,11,.95);color:#000;font-weight:800;padding:1px 6px;border-radius:3px;font-size:.62rem;margin-left:6px" title="Manual override via tracking_data/manual_overrides.json">🛠 MANUAL #' + manualSwappedOut.number + '→#' + manualCandidate.number + '</span>';
      swapNote = ' · <span style="color:#f59e0b">MANUAL OVERRIDE: #' + manualSwappedOut.number + ' (rank 6, $' + (manualSwappedOut.win_odds||0).toFixed(1) + ') → #' + manualCandidate.number + ' (rank ' + manualCandidate.rank + ', $' + (manualCandidate.win_odds||0).toFixed(1) + ')' + (race.manual_trio6_reason ? ' — ' + esc(race.manual_trio6_reason) : '') + '</span>';
    }} else if (swapCandidate && swappedOut) {{
      swapBadge = '<span style="background:rgba(168,85,247,.85);color:#fff;font-weight:800;padding:1px 6px;border-radius:3px;font-size:.62rem;margin-left:6px" title="Spoiler swap: rank-' + (swapCandidate.rank) + ' #' + swapCandidate.number + ' has shorter market odds + gear change vs default rank-6 #' + swappedOut.number + '">🔁 SWAP #' + swappedOut.number + '→#' + swapCandidate.number + '</span>';
      swapNote = ' · <span style="color:#a855f7">Swapped #' + swappedOut.number + ' (rank 6, $' + (swappedOut.win_odds||0).toFixed(1) + ') for #' + swapCandidate.number + ' (rank ' + swapCandidate.rank + ', $' + (swapCandidate.win_odds||0).toFixed(1) + ', gear ' + esc(swapCandidate.gear||'') + ') — market support + gear change</span>';
    }}
    if (_ticketHas('BOX TRIO-6')) {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem;opacity:.92">' +
          '<span style="display:inline-block;background:rgba(244,63,94,.6);color:#fff;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
            '+ BOX TRIO-6 · 20 combos · OPTIONAL' +
          '</span>' +
          _ticketHitMark('BOX TRIO-6') +
          top6.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ') +
          ' · ' + recBadge + chaosBadge + swapBadge +
          '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">' + recReason +
            ' Signals: ' + (chaosFactors.length ? chaosFactors.join(', ') : 'none') + '.' + swapNote +
          '</div>' +
        '</div>'
      );
    }}
  }}

  var betTicketHTML = ticketLines.length
    ? '<div class="bet-ticket" data-testid="bet-ticket-R' + race.race_number + '" style="background:rgba(251,191,36,.08);border-left:3px solid var(--gold);padding:8px 10px;margin:10px 0;border-radius:4px">' +
        '<div style="font-size:.62rem;color:var(--gold);letter-spacing:.12em;font-weight:700;margin-bottom:4px">BET TICKET</div>' +
        ticketLines.join('') +
      '</div>'
    : '<div class="bet-ticket" data-testid="bet-ticket-R' + race.race_number + '" style="background:rgba(71,85,105,.18);border-left:3px solid var(--dim);padding:8px 10px;margin:10px 0;border-radius:4px;color:var(--dim);font-size:.78rem">' +
        '<strong>SKIP</strong> · No bet recommended for this race' +
      '</div>';

  return '<div class="detail-panel">' +
      '<div class="detail-header">' +
        '<div>' +
          '<div class="detail-title">R' + race.race_number + '</div>' +
          '<div class="detail-meta">' + esc(race.time||'') + ' · ' + esc(race.distance) + 'm · ' + esc(race.class_type||'') + ' · ' + esc(race.going||'') + '</div>' +
        '</div>' +
        '<span style="font-size:.72rem;padding:2px 8px;border-radius:4px;background:' + (TIER_COLORS[race.conviction_tier]||'#64748b') + ';color:#000;font-weight:800">' + esc(race.conviction_tier||'SKIP') + '</span>' +
      '</div>' +
      betTicketHTML +
      '<table class="horse-table">' +
        '<thead><tr>' +
          '<th>Rank</th>' +
          '<th>Horse</th>' +
          '<th style="text-align:right">Pure%</th>' +
          '<th style="text-align:right">Market%</th>' +
          '<th style="text-align:right">Edge pp</th>' +
          '<th style="text-align:right">Odds</th>' +
          '<th>Bet</th>' +
        '</tr></thead>' +
        '<tbody>' + rowsHtml + '</tbody>' +
      '</table>' +
      detailTableHTML +
      formatHistoryHTML +
      speedMapHTML +
      reasoningBlock +
    '</div>';
}}

function _renderActiveRaces(scrollTarget) {{
  // Re-render #race-detail with every currently-active race stacked in
  // strip order (R1 → R11), regardless of click order. Updates tile
  // active states. scrollTarget (raceNum) optionally scrolls the
  // detail container into view.
  var det = document.getElementById('race-detail');
  if (!det) return;

  document.querySelectorAll('.strip-tile').forEach(function(t) {{ t.classList.remove('active'); }});
  if (_activeRaces.size === 0) {{
    det.innerHTML = '';
    _syncExpandAllButton();
    return;
  }}

  var sorted = RACES
    .filter(function(r) {{ return _activeRaces.has(r.race_number); }})
    .sort(function(a,b) {{ return a.race_number - b.race_number; }});

  det.innerHTML = sorted.map(function(r) {{
    // Position × in the top-LEFT (the tier badge sits in the top-RIGHT of
    // the detail panel and was previously stealing the click). z-index 5
    // keeps it above any sibling content even if layout shifts.
    var closeBtn =
      '<button class="race-close-btn" onclick="closeRaceDetail(' + r.race_number + ')" ' +
      'title="Collapse this race" ' +
      'style="position:absolute;top:10px;left:10px;z-index:5;background:rgba(15,23,42,.85);' +
      'border:1px solid var(--border);color:var(--fg);width:26px;height:26px;border-radius:4px;' +
      'cursor:pointer;font-size:1rem;line-height:1;font-weight:700">×</button>';
    // Pad the panel left so the × does not overlap the R1 title text.
    return '<div style="position:relative;margin-bottom:12px;padding-left:36px">' + closeBtn + _buildRaceDetailPanel(r) + '</div>';
  }}).join('');

  sorted.forEach(function(r) {{
    var tile = document.getElementById('tile-' + r.race_number);
    if (tile) tile.classList.add('active');
  }});

  _syncExpandAllButton();

  if (scrollTarget) {{
    var anchor = document.getElementById('tile-' + scrollTarget);
    var det2 = document.getElementById('race-detail');
    if (det2) det2.scrollIntoView({{behavior:'smooth', block:'nearest'}});
  }}
}}

function _syncExpandAllButton() {{
  var btn = document.getElementById('expand-all-btn');
  if (!btn) return;
  if (_activeRaces.size === RACES.length && RACES.length > 0) {{
    btn.classList.add('active');
    btn.textContent = 'Collapse all';
  }} else {{
    btn.classList.remove('active');
    btn.textContent = 'Expand all';
  }}
}}

function showRaceDetail(raceNum) {{
  // Toggle this race independently — multiple races can stay expanded.
  if (_activeRaces.has(raceNum)) {{
    _activeRaces.delete(raceNum);
    _renderActiveRaces(null);
  }} else {{
    _activeRaces.add(raceNum);
    _renderActiveRaces(raceNum);
  }}
}}

function closeRaceDetail(raceNum) {{
  // Per-panel × button — collapses a single race without re-clicking the tile.
  _activeRaces.delete(raceNum);
  _renderActiveRaces(null);
}}

function toggleExpandAll() {{
  if (_activeRaces.size === RACES.length) {{
    _activeRaces.clear();
  }} else {{
    _activeRaces = new Set(RACES.map(function(r) {{ return r.race_number; }}));
  }}
  _renderActiveRaces(null);
}}

// ─────────────────────────────────────────────────────────────
// Alerts rail
// ─────────────────────────────────────────────────────────────
function renderAlertsRail() {{
  var rail = document.getElementById('alerts-rail');
  var allAlerts = [];
  RACES.forEach(function(r) {{
    (r.alerts || []).forEach(function(a) {{ allAlerts.push(a); }});
  }});
  allAlerts.sort(function(a,b) {{ return (b.ts||0) - (a.ts||0); }});

  if (allAlerts.length === 0) {{
    rail.innerHTML =
      '<div class="rail-label">Live Alerts</div>' +
      '<div style="color:var(--dim);font-size:.78rem;margin-top:8px">No live events yet.</div>';
    return;
  }}

  var html = '<div class="rail-label">Live Alerts (' + allAlerts.length + ')</div>';
  allAlerts.forEach(function(a) {{
    var color = ALERT_COLORS[a.type] || '#64748b';
    var label = (a.type||'').replace(/_/g, ' ');
    html += '<div class="alert-item" style="border-left-color:' + color + ';background:' + color + '14">' +
      '<div class="alert-type" style="color:' + color + '">' + esc(label) + '</div>' +
      '<div class="alert-detail">R' + esc(a.race||'?') +
        (a.horse ? ' #' + esc(a.horse) : '') +
        (a.detail ? ' · ' + esc(a.detail) : '') +
      '</div>' +
    '</div>';
  }});
  rail.innerHTML = html;
}}

// ─────────────────────────────────────────────────────────────
// Refresh banner + button (only active when served via scripts.serve)
// ─────────────────────────────────────────────────────────────
var EXPECTED_REFRESH_SECONDS = 60;

function _closeInfoPop() {{
  var existing = document.getElementById('info-pop');
  if (existing) existing.remove();
  document.removeEventListener('click', _outsideClickHandler, true);
}}

function _outsideClickHandler(e) {{
  var pop = document.getElementById('info-pop');
  if (!pop) return;
  if (pop.contains(e.target) || (e.target.classList && e.target.classList.contains('info-tip'))) return;
  _closeInfoPop();
}}

function showInfoPop(event, title, body) {{
  event.stopPropagation();
  _closeInfoPop();
  var pop = document.createElement('div');
  pop.id = 'info-pop';
  pop.className = 'info-pop';
  pop.innerHTML =
    '<span class="info-pop-close" onclick="_closeInfoPop()">×</span>' +
    '<div class="info-pop-title">' + esc(title) + '</div>' +
    '<div>' + esc(body) + '</div>';
  document.body.appendChild(pop);
  // Position the popover near the icon, clamped to viewport
  var rect = event.target.getBoundingClientRect();
  var popRect = pop.getBoundingClientRect();
  var left = Math.min(rect.left, window.innerWidth - popRect.width - 12);
  var top = rect.bottom + 6;
  if (top + popRect.height > window.innerHeight - 12) {{
    top = Math.max(12, rect.top - popRect.height - 6);
  }}
  pop.style.left = Math.max(8, left) + 'px';
  pop.style.top = top + 'px';
  // Close on outside click (capture phase so it fires before re-renders)
  setTimeout(function() {{
    document.addEventListener('click', _outsideClickHandler, true);
  }}, 0);
}}

function hardReload() {{
  // Force a fresh fetch by appending a timestamp query string. Works
  // for both file:// and http:// — the new URL guarantees the browser
  // cannot serve a cached copy. Strips any prior ?v= cache-buster first.
  var cleanPath = location.pathname + location.hash.replace(/[?&]v=\\d+/, '');
  location.href = cleanPath + '?v=' + Date.now();
}}

function showBanner() {{ var b=document.getElementById('refresh-banner'); if(b) b.style.display='block'; }}
function hideBanner() {{ var b=document.getElementById('refresh-banner'); if(b) b.style.display='none'; }}
function updateBanner(elapsed, subText) {{
  var sub=document.getElementById('rb-sub'); var el=document.getElementById('rb-elapsed'); var bar=document.getElementById('rb-bar-fill');
  if(sub) sub.textContent=subText; if(el) el.textContent=elapsed+'s';
  if(bar) bar.style.width=Math.min(95,(elapsed/EXPECTED_REFRESH_SECONDS)*100)+'%';
}}

function parseProgress(logText) {{
  if(!logText) return 'Starting…';
  var lines=logText.split('\\n'); var lastRace=0; var phase='starting';
  for(var i=0;i<lines.length;i++) {{
    var line=lines[i];
    var done=line.match(/Race (\\d+): \\d+ horses, bet: (\\S+)/);
    if(done) {{ var n=parseInt(done[1]); if(n>=lastRace) {{ lastRace=n; phase='done:'+done[2]; }} continue; }}
    var fetching=line.match(/Fetching Race (\\d+)/);
    if(fetching) {{ var n=parseInt(fetching[1]); if(n>=lastRace) {{ lastRace=n; phase='fetching'; }} continue; }}
    if(line.indexOf('Tips data applied')>=0) phase='applying tips';
    if(line.indexOf('Saved dashboard')>=0) return 'Finalizing…';
    if(line.indexOf('Logged predictions')>=0) return 'Saving predictions…';
  }}
  if(lastRace===0) return 'Starting…';
  if(phase.indexOf('fetching')===0) return 'Fetching Race '+lastRace+'…';
  if(phase==='applying tips') return 'Applying tips for Race '+lastRace+'…';
  if(phase.indexOf('done:')===0) return 'Race '+lastRace+' complete ('+phase.slice(5)+')';
  return 'Race '+lastRace;
}}

async function pollUntilDone() {{
  var elapsed=0;
  while(elapsed<600) {{
    await new Promise(function(r) {{ setTimeout(r,2000); }});
    elapsed+=2;
    try {{
      var [sResp,lResp]=await Promise.all([fetch('/api/status',{{cache:'no-store'}}),fetch('/api/refresh/log',{{cache:'no-store'}})]);
      var status=await sResp.json();
      var logData=await lResp.json().catch(function() {{ return {{log:''}}; }});
      var progress=parseProgress(logData.log||'');
      updateBanner(elapsed,progress);
      if(!status.refreshing) {{ if(status.error) return {{ok:false,error:status.error}}; return {{ok:true}}; }}
    }} catch(e) {{ updateBanner(elapsed,'Connection lost — retrying…'); }}
  }}
  return {{ok:false,error:'timed out after 10 min'}};
}}

async function initRefreshButton() {{
  try {{
    var resp=await fetch('/api/status',{{cache:'no-store'}});
    if(!resp.ok) return;
    var data=await resp.json();
    var btn=document.getElementById('refresh-btn');
    if(btn) btn.style.display='inline-block';
    if(data.refreshing) {{
      showBanner();
      var result=await pollUntilDone();
      if(result.ok) {{ updateBanner(0,'Done — reloading…'); setTimeout(function() {{ location.reload(); }},500); }}
      else {{ hideBanner(); alert('Refresh failed: '+result.error); }}
    }}
  }} catch(e) {{ /* static-file mode */ }}
}}

async function refreshDashboard() {{
  var btn=document.getElementById('refresh-btn');
  if(btn) {{ btn.disabled=true; btn.style.opacity='0.6'; }}
  showBanner(); updateBanner(0,'Sending request…');
  try {{
    var resp=await fetch('/api/refresh',{{method:'POST',headers:{{'X-Race-Date':RACE_DATE}}}});
    if(!resp.ok && resp.status!==409) throw new Error('HTTP '+resp.status);
    var result=await pollUntilDone();
    if(result.ok) {{
      updateBanner(0,'Done — reloading…');
      document.getElementById('rb-bar-fill').style.width='100%';
      setTimeout(function() {{ location.reload(); }},400);
    }} else throw new Error(result.error);
  }} catch(e) {{
    hideBanner();
    if(btn) {{ btn.disabled=false; btn.style.opacity='1'; }}
    alert('Refresh failed: '+e.message);
  }}
}}

window.addEventListener('DOMContentLoaded', function() {{
  initRefreshButton();
  var expandAllBtn = document.getElementById('expand-all-btn');
  if (expandAllBtn) {{
    expandAllBtn.addEventListener('click', toggleExpandAll);
  }}
  var copyAllBtn = document.getElementById('copy-all-btn');
  if (copyAllBtn) {{
    copyAllBtn.addEventListener('click', function() {{
      // Only the COMPACT (ticket-first) buttons, one per race. The
      // Copy-WA-Full buttons would duplicate everything with rationale —
      // useless for a bulk dump. Filter out NO-BET cards so the paste is
      // actionable.
      var heroBtns = document.querySelectorAll('#hero .wa-btn[data-testid^="copy-wa-ticket-"]');
      var payloads = [];
      heroBtns.forEach(function(btn) {{
        var p = btn.dataset.payload || '';
        if (p && p.indexOf('NO SYSTEM BET') === -1) payloads.push(p);
      }});
      if (payloads.length === 0) {{
        copyAllBtn.textContent = 'No bets today';
        setTimeout(function() {{ copyAllBtn.textContent = 'Copy all bets'; }}, 1500);
        return;
      }}
      // Append disclaimer once (the compact per-card payloads do NOT already
      // carry the entertainment-only footer — only copyWA() appends it for
      // per-card buttons). Without this the bulk dump goes out unguarded.
      var combined = payloads.join('\\n\\n---\\n\\n') +
        '\\n\\n_⚠️ For entertainment only. Gamble responsibly._';
      var ok = function() {{
        copyAllBtn.textContent = 'Copied ' + payloads.length + '!';
        setTimeout(function() {{ copyAllBtn.textContent = 'Copy all bets'; }}, 1500);
      }};
      var fallback = function() {{
        // execCommand fallback for insecure contexts (file://, http://) and
        // browsers that reject clipboard.writeText silently.
        try {{
          var ta = document.createElement('textarea');
          ta.value = combined;
          ta.style.position = 'fixed'; ta.style.opacity = '0';
          document.body.appendChild(ta);
          ta.focus(); ta.select();
          var success = document.execCommand('copy');
          document.body.removeChild(ta);
          if (success) {{ ok(); return; }}
        }} catch(e) {{}}
        copyAllBtn.textContent = 'Copy failed';
        setTimeout(function() {{ copyAllBtn.textContent = 'Copy all bets'; }}, 1800);
        // Last-resort: open prompt so user can manually copy.
        try {{ window.prompt('Copy bets manually (Cmd+C / Ctrl+C):', combined); }} catch(e) {{}}
      }};
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(combined).then(ok).catch(fallback);
      }} else {{
        fallback();
      }}
    }});
  }}
}});

// ─────────────────────────────────────────────────────────────
// Expose IIFE-local functions to global scope so inline onclick=""
// attributes (e.g. onclick="showRaceDetail(1)") can call them.
// Without this, the IIFE wraps everything and inline handlers throw
// "X is not defined" — the bug that broke individual-tile expansion.
// ─────────────────────────────────────────────────────────────
window.showRaceDetail = showRaceDetail;
window.closeRaceDetail = closeRaceDetail;
window.toggleExpandAll = toggleExpandAll;
window.hardReload = hardReload;
window.refreshDashboard = refreshDashboard;
window.copyWA = copyWA;
window.showInfoPop = showInfoPop;
window._closeInfoPop = _closeInfoPop;
window.setStripView = setStripView;
window.toggleCardDetail = toggleCardDetail;
window.setTopBetsFilter = setTopBetsFilter;

// ─────────────────────────────────────────────────────────────
// Boot — Race Order is the default view (all 11 races as cards).
// renderStrip is still called so the hidden strip stays in sync if
// the user toggles to a layout that needs it.
// ─────────────────────────────────────────────────────────────
setStripView('race');
renderStrip();
renderAlertsRail();

}})();
</script>
</body>
</html>"""
