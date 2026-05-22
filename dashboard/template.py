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
    'LOW': '#64748b', 'SKIP': '#475569',
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
        if tier not in ('EDGE', 'HIGH', 'MED', 'LOW', 'SKIP'):
            tier = 'SKIP'
        label = 'R' + str(num) + (' ★' if tier == 'EDGE' else '')
        horses = r.get('horses') or []
        sub = ''
        if horses:
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
      <span class="topbar-brand">HKJC Racing</span>
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
    <div class="footer">HKJC Racing Analyzer v6.6 — Generated: {now}</div>
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
  EDGE:'#fbbf24', HIGH:'#10b981', MED:'#f59e0b', LOW:'#64748b', SKIP:'#475569'
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

function _didBetHit(race) {{
  // Determine whether ANY of our recommended bets hit for a finished race.
  // Returns one of: 'STRIKE-WIN' (rank-1 won), 'STRIKE-QIN' (top-2 quinella),
  // 'STRIKE-BOX4' (winning quinella pair was in top-4 picks),
  // 'STRIKE-TRIO5' (full podium in top-5), 'STRIKE-TRIO6' (full podium in top-6),
  // 'MISS' (none), or null if the race is not finished.
  if (!race.results_available) return null;
  var horsesByPos = race.horses
    .filter(function(h) {{ return h.actual_position; }})
    .sort(function(a, b) {{ return a.actual_position - b.actual_position; }});
  if (horsesByPos.length < 2) return null;
  var p1 = horsesByPos[0].number, p2 = horsesByPos[1].number;
  var p3 = horsesByPos.length >= 3 ? horsesByPos[2].number : null;
  var p4 = horsesByPos.length >= 4 ? horsesByPos[3].number : null;
  var top2Nums = race.horses.slice(0, 2).map(function(h) {{ return h.number; }});
  var top4Nums = race.horses.slice(0, 4).map(function(h) {{ return h.number; }});
  var top5Nums = race.horses.slice(0, 5).map(function(h) {{ return h.number; }});
  var top6Nums = race.horses.slice(0, 6).map(function(h) {{ return h.number; }});
  var rank1 = race.horses[0];
  if (rank1 && rank1.number === p1) return 'STRIKE-WIN';
  if (top2Nums.indexOf(p1) >= 0 && top2Nums.indexOf(p2) >= 0) return 'STRIKE-QIN';
  // F4 BOX-5 strike: all 4 podium horses inside top-5 picks (rarer than TRIO-5,
  // pays bigger — shipped 2026-05-06 after 60-race backtest showed +54% ROI).
  // Checked BEFORE BOX-4 because F4-5 is the stronger, more specific result —
  // when both are satisfied (common: QIN pair in top-4 AND all 4 in top-5),
  // we want the badge to reflect F4-5 not BOX-4 (was undercounting F4-5 hits).
  if (p4 != null && top5Nums.indexOf(p1)>=0 && top5Nums.indexOf(p2)>=0 &&
      top5Nums.indexOf(p3)>=0 && top5Nums.indexOf(p4)>=0) return 'STRIKE-F4-5';
  if (top4Nums.indexOf(p1) >= 0 && top4Nums.indexOf(p2) >= 0) return 'STRIKE-BOX4';
  // TRIO box checks require all 3 podium horses inside the box.
  if (p3 != null) {{
    if (top5Nums.indexOf(p1)>=0 && top5Nums.indexOf(p2)>=0 && top5Nums.indexOf(p3)>=0) return 'STRIKE-TRIO5';
    if (top6Nums.indexOf(p1)>=0 && top6Nums.indexOf(p2)>=0 && top6Nums.indexOf(p3)>=0) return 'STRIKE-TRIO6';
  }}
  return 'MISS';
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
      ' &nbsp;<button class="topbets-sub-tab' + (_topBetsFilter === 'next' ? ' active' : '') + '" id="tb-next" onclick="setTopBetsFilter(&quot;next&quot;)">Next</button>' +
      '<button class="topbets-sub-tab' + (_topBetsFilter === 'past' ? ' active' : '') + '" id="tb-past" onclick="setTopBetsFilter(&quot;past&quot;)">Past</button>';
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
  // Longshot paper bet — rendered in the ticket area on EVERY qualifying race
  // (steamed rank 6-8 present), including SKIP races that have no stake table.
  // Purple/dashed, explicitly NOT STAKED, never affects the real budget total.
  var _lp = _longshotPick(race);
  var paperBlock = _lp
    ? '<div class="trust-block" style="margin-top:6px;border:1px dashed #5b21b6;border-radius:6px;padding:6px 10px;background:rgba(91,33,182,.10)">' +
        '<span style="background:#5b21b6;color:#ede9fe;font-weight:800;padding:1px 7px;border-radius:3px;font-size:.62rem">🧪 LONGSHOT-TRIO · PAPER · NOT STAKED</span>' +
        (_lp.tag ? ' <span style="color:' + _lp.tagColor + ';font-weight:800;font-size:.64rem;margin-left:6px">' + _lp.tag + '</span>' : '') +
        '<div style="color:var(--dim);font-size:.66rem;margin-top:3px">20-combo box · top-5 + ' +
          '<span style="color:#c4b5fd;font-weight:700">#' + esc(_lp.best.number) + ' ' + esc(_lp.best.name) +
          ' (rank ' + esc(_lp.best.rank) + ', @' + esc(_lp.best.win_odds) + ', -' + _lp.pct + '%)</span></div>' +
      '</div>'
    : '';
  // EXPERIMENT (owner-only): top-2 + random 3rd straight TRIO. Amber/dashed,
  // NOT STAKED, isolated from ML. See _luckyTrio().
  var _lt = _luckyTrio(race);
  var luckyBlock = _lt
    ? '<div class="trust-block" style="margin-top:6px;border:1px dashed #b45309;border-radius:6px;padding:6px 10px;background:rgba(180,83,9,.10)">' +
        '<span style="background:#b45309;color:#fef3c7;font-weight:800;padding:1px 7px;border-radius:3px;font-size:.62rem">\\uD83C\\uDFB2 LUCKY TRIO \\u00b7 EXPERIMENT \\u00b7 NOT STAKED</span>' +
        (_lt.tag ? ' <span style="color:' + _lt.tagColor + ';font-weight:800;font-size:.64rem;margin-left:6px">' + _lt.tag + '</span>' : '') +
        '<div style="color:var(--dim);font-size:.66rem;margin-top:3px">straight TRIO \\u00b7 top-2 + random 3rd: ' +
          '<span style="color:#fcd34d;font-weight:700">' + _lt.nums.map(function(n) {{ return '#' + esc(n); }}).join('-') + '</span>' +
          ' <span style="opacity:.75">(3rd = #' + esc(_lt.third.number) + ' ' + esc(_lt.third.name) +
          ', rank ' + esc(_lt.third.rank) + ', @' + esc(_lt.third.win_odds) + ')</span></div>' +
      '</div>'
    : '';
  if (ss.amount && ss.amount > 0) {{
    var br = race.bankroll || 0;
    // Enumerate firing MAIN bets and their combo counts
    var mainBets = {{}};
    race.horses.forEach(function(h) {{
      var b = (h.bet_recommendation || '').trim();
      if (b && b !== 'SKIP') {{ (mainBets[b] = mainBets[b] || []).push(h); }}
    }});
    var betLines = [];
    var totalCombos = 0;
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
        betLines.push({{lbl: lbl, combos: combos}});
        totalCombos += combos;
      }}
    }});
    // BOX TRIO-5 and F4 BOX-5 also fire as MAIN tier when top5 length is 5
    var top5len = race.horses.slice(0, 5).length;
    if (top5len === 5) {{
      betLines.push({{lbl: 'BOX TRIO-5', combos: 10}});
      totalCombos += 10;
      // F4 BOX-5 has its own gate (mirrors logic at line ~915)
      var f4top1 = (race.horses[0] && race.horses[0].win_odds) || 0;
      var f4field = race.horses.length;
      var f4Skip = (f4top1 > 0 && f4top1 < 2.5) ||
                    (tier !== 'EDGE' && tier !== 'HIGH' && tier !== 'MED') ||
                    (f4field < 8);
      if (!f4Skip) {{
        betLines.push({{lbl: 'F4 BOX-5', combos: 5}});
        totalCombos += 5;
      }}
      // BOX TRIO-6 fires as MAIN only when verdict is TAKE (mirrors logic
      // at line ~1088). Adding 20 combos dilutes the per-bet unit stake
      // (e.g., $200 / 36 combos = $5/combo) which is the intended trade —
      // the user sees TRIO-6 sized in the breakdown when it matters.
      var t6t1 = f4top1;
      var t6fld = f4field;
      var t6stm = race.horses.slice(0, 12).filter(function(h) {{
        return h.steam_move && h.steam_move.is_steam;
      }}).length;
      var t6Take = (t6t1 > 0 && t6t1 < 2.5) ? false :
                    (t6stm >= 2) ? true :
                    (tier === 'HIGH' && t6fld >= 11 && t6t1 >= 2.5 && t6t1 <= 5.0) ? true :
                    false;
      if (t6Take) {{
        betLines.push({{lbl: 'BOX TRIO-6', combos: 20}});
        totalCombos += 20;
      }}
    }}
    // Unit stake — divide budget across all combos. Floor to nearest $5
    // so per-bet stakes never sum above the budget (was: round() inflated
    // unit when amount/combos > .5 boundary, e.g., $300/16 = $18.75 → $20
    // pushed the breakdown total to $320 over the $300 budget). Mirrors the
    // WA-payload unit calc above to keep the two views in sync.
    var unit = totalCombos > 0 ? (ss.amount / totalCombos) : 0;
    var unitRounded = Math.max(5, Math.floor(unit / 5) * 5);
    var breakdownRows = '';
    var allocated = 0;
    betLines.forEach(function(b) {{
      var amt = b.combos * unitRounded;
      allocated += amt;
      breakdownRows +=
        '<tr><td style="padding:1px 6px 1px 0;color:var(--dim)">' + esc(b.lbl) + '</td>' +
        '<td style="padding:1px 0;text-align:right">' + b.combos + ' × $' + unitRounded + '</td>' +
        '<td style="padding:1px 0 1px 8px;text-align:right;color:var(--gold);font-weight:600">$' + amt + '</td></tr>';
    }});
    var breakdownHtml = breakdownRows
      ? '<table style="font-size:.7rem;border-collapse:collapse;margin-top:6px;width:100%">' +
          breakdownRows +
          '<tr style="border-top:1px solid var(--border)"><td style="padding:3px 0 1px 0;color:var(--dim)">Total</td>' +
          '<td></td><td style="padding:3px 0 1px 8px;text-align:right;color:var(--gold);font-weight:600">$' + allocated + '</td></tr>' +
        '</table>'
      : '';
    stakeHtml =
      '<div class="trust-block">' +
        '<div class="trust-label">Suggested Stake ' +
          '<span class="info-tip" onclick="showInfoPop(event, &quot;Suggested Stake&quot;, &quot;Per-card stake sized off your bankroll (config.BANKROLL = $' + br + ' HKD). EDGE 3%, HIGH 2%, MED 1%, LOW 0.5%. WIN bets use half-Kelly capped at the tier %. The budget is split across firing MAIN bets by combo count — unit stake is rounded to the nearest $5 for clean bet entry. Adjust BANKROLL in config.py to match your rolling bankroll.&quot;)">ⓘ</span>' +
        '</div>' +
        '<div class="trust-body" style="margin-top:4px;font-size:.85rem">' +
          '<strong style="color:var(--gold)">$' + Math.round(ss.amount) + '</strong> HKD budget' +
          '<br><span style="color:var(--dim);font-size:.65rem">' + (ss.pct||0).toFixed(2) + '% of $' + br + ' · ' + esc(ss.method||'') + '</span>' +
          breakdownHtml +
        '</div>' +
      '</div>' + paperBlock + luckyBlock;
  }}
  // SKIP / no-stake races still surface the paper experiments if any qualify.
  if (!stakeHtml) stakeHtml = paperBlock + luckyBlock;

  // Bet type sub-label
  var betRec = topH.bet_recommendation || 'QIN';
  var raceLabel = 'R' + race.race_number + ' · ' + esc(race.time || '') + ' · ' + esc(race.class_type || '') + ' ' + race.distance + 'm';

  // STRIKE / MISS badge for past races (resolved on the fly from
  // _didBetHit). Surfaces the bet outcome at the card header so the
  // user can scan past races for what hit without expanding each card.
  var strikeStatus = _didBetHit(race);
  var strikeBadge = '';
  if (strikeStatus === 'STRIKE-WIN') {{
    strikeBadge = '<span class="strike-badge win" title="Top pick won the race">★ STRIKE · WIN</span>';
  }} else if (strikeStatus === 'STRIKE-QIN') {{
    strikeBadge = '<span class="strike-badge qin" title="Winning quinella pair was inside top-2 model picks">★ STRIKE · QIN</span>';
  }} else if (strikeStatus === 'STRIKE-BOX4') {{
    strikeBadge = '<span class="strike-badge box4" title="Winning quinella pair was inside top-4 model picks (Box-4 hit)">★ STRIKE · BOX-4</span>';
  }} else if (strikeStatus === 'STRIKE-F4-5') {{
    strikeBadge = '<span class="strike-badge f4" title="Top-4 finishers all inside top-5 model picks (F4 BOX-5 hit — bigger dividend than TRIO-5)">★ STRIKE · F4-5</span>';
  }} else if (strikeStatus === 'STRIKE-TRIO5') {{
    strikeBadge = '<span class="strike-badge trio" title="Full podium inside top-5 model picks (BOX TRIO-5 hit)">★ STRIKE · TRIO-5</span>';
  }} else if (strikeStatus === 'STRIKE-TRIO6') {{
    strikeBadge = '<span class="strike-badge trio6" title="Full podium inside top-6 model picks (BOX TRIO-6 hit — TRIO-5 missed)">★ STRIKE · TRIO-6</span>';
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
  if (tier === 'EDGE') {{
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

  // Compute per-bet stake amounts so each MAIN block shows its $ allocation
  // inline (matches the dashboard stake-breakdown table). Shared unit
  // stake = ss.amount / total_combos, rounded to nearest $5.
  var waTotalCombos = 0;
  var waMainCombos = {{}};
  var mainBetTypes = Object.keys(waBets);
  mainBetTypes.forEach(function(bt) {{
    var n = waBets[bt].length;
    var c = 0;
    if (bt === 'WIN' || bt === 'BANKER WIN' || bt === 'PLACE') c = 1;
    else if (bt === 'QIN' || bt === 'BOX QIN') c = n === 2 ? 1 : (n*(n-1)/2);
    else if (bt === 'TRIO') c = n === 3 ? 1 : (n*(n-1)*(n-2)/6);
    waMainCombos[bt] = c;
    waTotalCombos += c;
  }});
  // top5/F4/T6 gating mirrors the block below — compute inline to avoid
  // a forward reference to top5wa (declared later in the addOns section).
  var waTop5Len = race.horses.slice(0, 5).length;
  var waT5Combos = (waTop5Len === 5) ? 10 : 0;
  var waF4Combos = 0;
  var waT6Combos = 0;
  if (waTop5Len === 5) {{
    var f4t1 = (race.horses[0] && race.horses[0].win_odds) || 0;
    var f4f = race.horses.length;
    var f4Sk = (f4t1 > 0 && f4t1 < 2.5) ||
                (tier !== 'EDGE' && tier !== 'HIGH' && tier !== 'MED') ||
                (f4f < 8);
    if (!f4Sk) waF4Combos = 5;
    // TRIO-6 included only when verdict is TAKE (mirrors logic at line ~1088).
    var t6stm = race.horses.slice(0, 12).filter(function(h) {{
      return h.steam_move && h.steam_move.is_steam;
    }}).length;
    var t6Tk = (f4t1 > 0 && f4t1 < 2.5) ? false :
                (t6stm >= 2) ? true :
                (tier === 'HIGH' && f4f >= 11 && f4t1 >= 2.5 && f4t1 <= 5.0) ? true :
                false;
    if (t6Tk && mainBetTypes.length > 0) waT6Combos = 20;
  }}
  waTotalCombos += waT5Combos + waF4Combos + waT6Combos;
  // Floor (not round) to nearest $5 so the per-bet stakes never sum above
  // the budget. Was: Math.round() inflated unit when amount/combos > .5
  // boundary (e.g., $300/16 = $18.75 → $20 → total $320, over the $300
  // budget). Floor gives $15 → total $240, under budget — user can manually
  // add the residual to a preferred bet if desired.
  var waUnit = (ss && ss.amount > 0 && waTotalCombos > 0)
    ? Math.max(5, Math.floor((ss.amount / waTotalCombos) / 5) * 5)
    : 0;
  // Annotate the stake banner with unallocated residual when floor leaves
  // budget on the table. User can then manually deploy the residual to a
  // preferred bet (e.g., round TRIO-5 up to next $5). Skip hint when
  // residual < $5 (rounding noise, not actionable).
  if (_ssBannerIdx >= 0 && waUnit > 0) {{
    var _allocated = waTotalCombos * waUnit;
    var _unalloc = Math.round(ss.amount) - _allocated;
    if (_unalloc >= 5) {{
      waLines[_ssBannerIdx] = waLines[_ssBannerIdx].replace(' HKD*', ' HKD — $' + _unalloc + ' unallocated*');
    }}
  }}

  // MAIN BET — one focused block (not a duplicate summary + detail).
  var isSkipRace = mainBetTypes.length === 0;
  if (!isSkipRace) {{
    mainBetTypes.forEach(function(bt) {{
      var picks = waBets[bt];
      var lbl = bt;
      if (bt === 'BANKER WIN' || bt === 'WIN') {{
        lbl = bt === 'BANKER WIN' ? 'BANKER WIN' : 'WIN';
      }} else if ((bt === 'QIN' || bt === 'BOX QIN') && picks.length >= 2) {{
        lbl = picks.length === 2 ? 'STRAIGHT QIN' : 'BOX QIN-' + picks.length;
      }} else if (bt === 'TRIO' && picks.length >= 3) {{
        lbl = picks.length === 3 ? 'STRAIGHT TRIO' : 'BOX TRIO-' + picks.length;
      }}
      // Inline stake: combos × unit, shown beside the label
      var betStake = waUnit > 0 ? (' · $' + (waMainCombos[bt] * waUnit)) : '';
      waLines.push('');
      waLines.push('*MAIN: ' + lbl + betStake + '*');
      picks.forEach(function(h) {{ waLines.push(_waPick(h)); }});
    }});
  }} else {{
    waLines.push('');
    waLines.push('*SKIP* — no bet recommended');
  }}

  // Optional add-on layers (BOX QIN-4, F4, TRIO-5, TRIO-6, QPL BANKER)
  // collected as bullets so the reader sees them as discretionary.
  var addOns = [];
  var top4wa = race.horses.slice(0, 4);
  var top5wa = race.horses.slice(0, 5);
  var top6wa = race.horses.slice(0, 6);
  var top1Wa = race.horses[0];
  var primNumsWa = {{}};
  mainBetTypes.forEach(function(bt) {{ waBets[bt].forEach(function(h) {{ primNumsWa[h.number] = true; }}); }});

  // QIN PLACE BANKER for short-favorite races
  if (top1Wa && top1Wa.win_odds && top1Wa.win_odds <= 1.8 && race.horses.length >= 4) {{
    var bnkNums = race.horses.slice(1, 4).map(function(h) {{ return '#' + h.number; }}).join(' + ');
    addOns.push('QPL BANKER (3 combos): #' + top1Wa.number + ' with ' + bnkNums);
  }}

  // BOX QIN-4 protection
  if (top4wa.length === 4) {{
    var hasNonPrimWa = top4wa.some(function(h) {{ return !primNumsWa[h.number]; }});
    if (hasNonPrimWa || mainBetTypes.length === 0) {{
      var box4Nums = top4wa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
      addOns.push('BOX QIN-4 (6 combos): ' + box4Nums);
    }}
  }}

  // F4 SAVER — gated to skip low-EV setups (same rule as visual ticket)
  if (top4wa.length === 4) {{
    var f4Top1Odds = (race.horses[0] && race.horses[0].win_odds) || 0;
    var f4Field = race.horses.length;
    var f4Skip = (f4Top1Odds > 0 && f4Top1Odds < 2.5) ||
                 (tier !== 'EDGE' && tier !== 'HIGH' && tier !== 'MED') ||
                 (f4Field < 8);
    if (!f4Skip) {{
      var f4Order = top4wa.map(function(h) {{ return '#' + h.number; }}).join(' → ');
      addOns.push('F4 SAVER (1 unit · exact 1→4): ' + f4Order);
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
  if (top5wa.length === 5) {{
    var t5Nums = top5wa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
    var t5Stake = waUnit > 0 ? (' · $' + (10 * waUnit)) : '';
    var t5SwapAnno = top5waSwap
      ? ' · 🔁 swapped #' + top5waSwapOut.number + '→#' + top5waSwap.number
      : '';
    t5MainBlock = ['', '*MAIN: BOX TRIO-5' + t5Stake + '* (10 combos)' + t5SwapAnno, t5Nums];
  }}

  // F4 BOX-5 — second MAIN bet block (shipped 2026-05-06, +54% ROI on 60-race
  // backtest, strictly better economics than TRIO-5). Same horses as TRIO-5
  // but only 5 combos x $10 = $50 vs the TRIO-5 box at 10 combos x $10 = $100.
  // Same gates as F4 SAVER apply. Kept TRIO-5 as parallel MAIN since they
  // cover different misses (TRIO-5 hits when 4th finisher is outside top-5).
  var f4bMainBlock = null;
  if (top5wa.length === 5) {{
    var f4bTop1OddsWa = (race.horses[0] && race.horses[0].win_odds) || 0;
    var f4bFieldWa = race.horses.length;
    var f4bSkipWa = (f4bTop1OddsWa > 0 && f4bTop1OddsWa < 2.5) ||
                    (tier !== 'EDGE' && tier !== 'HIGH' && tier !== 'MED') ||
                    (f4bFieldWa < 8);
    if (!f4bSkipWa) {{
      var f4bNums = top5wa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
      var f4bStake = waUnit > 0 ? (' · $' + (5 * waUnit)) : '';
      var f4bSwapAnno = top5waSwap
        ? ' · 🔁 swapped #' + top5waSwapOut.number + '→#' + top5waSwap.number
        : '';
      f4bMainBlock = ['', '*MAIN: F4 BOX-5' + f4bStake + '* (5 combos · top-4 any order)' + f4bSwapAnno, f4bNums];
    }}
  }}

  // BOX TRIO-6 — annotate with TAKE/SKIP/YOUR CALL + chaos so user sees
  // the gate decision in the WA message (matches dashboard visual).
  // Mirror the spoiler-swap logic from the visual ticket so WA picks
  // match what the dashboard recommends. t6MainBlock is set when verdict
  // says TAKE so the line is rendered as MAIN instead of "Optional".
  var t6MainBlock = null;
  var top6waSwap = null;
  var top6waSwapOut = null;
  var top6waManual = null;
  var top6waManualOut = null;
  // Manual TRIO-6 override (race.manual_trio6_horse) takes precedence over
  // the auto rank-6 swap. Set via tracking_data/manual_overrides.json when
  // racing-API form/RPR signal beats the auto candidate. If the override
  // names a horse that is ALREADY rank-6, the override just suppresses the
  // auto-swap (no visible badge) — nothing to swap.
  if (top6wa.length === 6 && race.manual_trio6_horse) {{
    if (top6wa[5].number !== race.manual_trio6_horse) {{
      top6waManual = race.horses.find(function(h) {{ return h.number === race.manual_trio6_horse; }});
      if (top6waManual) {{
        top6waManualOut = top6wa[5];
        top6wa = race.horses.slice(0, 5).concat([top6waManual]);
      }}
    }}
    // else: override matches current rank-6 — no swap, no badge, but auto-swap is suppressed
  }} else if (top6wa.length === 6) {{
    var rank6wa = top6wa[5];
    var rank6OddsWa = (rank6wa && rank6wa.win_odds) || 0;
    if (rank6OddsWa > 0) {{
      for (var swiW = 6; swiW < Math.min(12, race.horses.length); swiW++) {{
        var candW = race.horses[swiW];
        var candOddsW = candW.win_odds || 0;
        var candGearW = (candW.gear || '').trim();
        if (candOddsW > 0 && candGearW && candOddsW <= rank6OddsWa * 0.8) {{
          if (!top6waSwap || candOddsW < top6waSwap.win_odds) {{
            top6waSwap = candW;
          }}
        }}
      }}
    }}
    if (top6waSwap) {{
      top6waSwapOut = top6wa[5];
      top6wa = race.horses.slice(0, 5).concat([top6waSwap]);
    }}
  }}
  if (top6wa.length === 6) {{
    var t6Nums = top6wa.map(function(h) {{ return '#' + h.number; }}).join(' + ');
    var t6Top1 = (race.horses[0] && race.horses[0].win_odds) || 0;
    var t6Field = race.horses.length;
    var t6Steam = race.horses.slice(0, 12).filter(function(h) {{
      return h.steam_move && h.steam_move.is_steam;
    }}).length;
    // CHAOS score 0-4 — only count truly differentiating signals.
    // 2026-05-17: dropped "tier HIGH/MED" (always-on for bettable races) and
    // raised field-size to ≥13 (was ≥11 — most ST races already 12-14). Added
    // top_2_share<0.40 (model itself uncertain — the strongest chaos signal).
    var t6Top2Share = race.top_2_share || 0;
    var chaosScore = 0;
    if (t6Field >= 13) chaosScore++;
    if (t6Top1 >= 2.5 && t6Top1 <= 5.0) chaosScore++;
    if (t6Steam >= 2) chaosScore++;
    if (t6Top2Share > 0 && t6Top2Share < 0.40) chaosScore++;
    var t6Verdict;
    if (t6Top1 > 0 && t6Top1 < 2.5) t6Verdict = 'SKIP heavy fav';
    else if (t6Steam >= 2) t6Verdict = '★ TAKE (' + t6Steam + ' steam)';
    else if (tier === 'HIGH' && t6Field >= 11 && t6Top1 >= 2.5 && t6Top1 <= 5.0) t6Verdict = '★ TAKE';
    else t6Verdict = 'YOUR CALL';
    var swapAnno = top6waManual
      ? ' · 🛠 MANUAL #' + top6waManualOut.number + '→#' + top6waManual.number + (race.manual_trio6_reason ? ' (' + race.manual_trio6_reason + ')' : '')
      : top6waSwap
      ? ' · 🔁 swapped #' + top6waSwapOut.number + '→#' + top6waSwap.number + ' (gear ' + (top6waSwap.gear||'') + ')'
      : '';
    // When verdict says TAKE, hoist out of "Optional adds" into a MAIN block
    // so the section header matches the recommendation (was: TAKE shown under
    // "Optional", contradicting the verdict). Skip races stay in addOns
    // alongside the other demoted MAIN bets.
    var isT6Take = t6Verdict.indexOf('★ TAKE') === 0;
    if (isT6Take && !isSkipRace) {{
      var t6Stake = waUnit > 0 ? (' · $' + (20 * waUnit)) : '';
      t6MainBlock = ['', '*MAIN: BOX TRIO-6' + t6Stake + '* (20 combos · ' + t6Verdict + ' · CHAOS ' + chaosScore + '/4)' + swapAnno, t6Nums];
    }} else {{
      addOns.push('BOX TRIO-6 (20 combos · ' + t6Verdict + ' · CHAOS ' + chaosScore + '/4): ' + t6Nums + swapAnno);
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

  // PAPER EXPERIMENT (2026-05-20): longshot-inclusive TRIO. Top-5 + most-steamed
  // rank 6-8 horse. NOT a recommended bet — flagged explicitly so it never reads
  // as a real add. Mirrors _select() in scripts/track_longshot_trio.py. Lives
  // outside the *Optional adds* block on purpose, below it, with its own header.
  if (race.horses && race.horses.length >= 8) {{
    var _ltSorted = race.horses.slice().sort(function(a, b) {{ return (a.rank || 99) - (b.rank || 99); }});
    var _ltBest = null, _ltPct = -1;
    for (var _li = 5; _li < 8; _li++) {{
      var _lsm = _ltSorted[_li].steam_move || {{}};
      if (!_lsm.is_steam) continue;
      var _lm = (_lsm.label || '').match(/-(\\d+)%/);
      var _lp = _lm ? parseInt(_lm[1], 10) : 0;
      if (_lp > _ltPct) {{ _ltPct = _lp; _ltBest = _ltSorted[_li]; }}
    }}
    if (_ltBest) {{
      var _ltNums = _ltSorted.slice(0, 5).map(function(h) {{ return '#' + h.number; }}).join(' ');
      waLines.push('');
      waLines.push('🧪 *PAPER EXPERIMENT — NOT A BET*');
      waLines.push('Longshot-TRIO: ' + _ltNums + ' + #' + _ltBest.number + ' ' +
                   (_ltBest.name || '') + ' (rank ' + _ltBest.rank + ', @$' +
                   (_ltBest.win_odds || '?') + ', -' + _ltPct + '%)');
      waLines.push('_Tracking only — do not stake until 6-meeting verdict._');
    }}
  }}

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
  var edgeAlertHTML = isEdge
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
    '" data-race-num="' + race.race_number + '" onclick="toggleCardDetail(' + race.race_number + ', event)" ' +
    (isEdge ? 'style="box-shadow:0 0 0 2px var(--gold), 0 4px 14px rgba(251,191,36,.35)" ' : '') +
    'title="Click anywhere on the card to see full per-horse detail, format history, speed map and reasoning">' +
    '<div class="bet-card-top">' +
      '<div>' +
        '<span class="tier-badge" style="background:' + tierColor + (isEdge ? ';box-shadow:0 0 8px rgba(251,191,36,.7)' : '') + '">' + (isEdge ? '★ EDGE · RARE' : esc(tier)) + '</span>' +
        '<strong style="margin-left:8px">' + raceLabel + '</strong>' + strikeBadge +
      '</div>' +
      '<div style="display:flex;align-items:center;gap:8px">' +
        '<span style="font-size:.7rem;color:var(--dim)">' + esc(edgeDetail) + '</span>' +
        '<button class="wa-btn" data-payload="' + escAttr(waPayload) +
          '" onclick="event.stopPropagation()">Copy WA</button>' +
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
  var fullText = text + '\\n⚠️ For entertainment only. Gamble responsibly.';
  navigator.clipboard.writeText(fullText).then(function() {{
    var orig = btn.textContent;
    btn.textContent = 'Copied!';
    btn.style.background = 'var(--green)';
    btn.style.color = '#000';
    setTimeout(function() {{
      btn.textContent = orig;
      btn.style.background = '';
      btn.style.color = '';
    }}, 2000);
  }});
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

// ── PAPER EXPERIMENT (2026-05-20): longshot-inclusive TRIO ──────────────
// Selection = model top-5 + most-steamed rank 6-8 horse. Rendered as a PAPER
// row inside the bet-ticket stake breakdown — clearly labelled, never staked,
// excluded from the budget total. Tracking only until the N>=6 ship gate.
// Mirrors _select() in scripts/track_longshot_trio.py so dashboard == log.
// Returns an object (best, pct, box6nums, tag, tagColor) or null when there is
// no qualifying horse.
function _longshotPick(race) {{
  if (!race.horses || race.horses.length < 8) return null;
  var sorted = race.horses.slice().sort(function(a, b) {{ return (a.rank || 99) - (b.rank || 99); }});
  var top5 = sorted.slice(0, 5);
  var best = null, bestPct = -1;
  for (var i = 5; i < 8; i++) {{
    var sm = sorted[i].steam_move || {{}};
    if (!sm.is_steam) continue;
    var m = (sm.label || '').match(/-(\\d+)%/);
    var pct = m ? parseInt(m[1], 10) : 0;
    if (pct > bestPct) {{ bestPct = pct; best = sorted[i]; }}
  }}
  if (!best) return null;  // no qualifying steamed rank 6-8 horse → no paper bet
  var top5nums = top5.map(function(h) {{ return h.number; }});
  var box6nums = top5nums.concat([best.number]);
  var tag = '', tagColor = '#94a3b8';
  if (race.results_available && Array.isArray(race.actual_top3) && race.actual_top3.length) {{
    var boxHit = race.actual_top3.every(function(n) {{ return box6nums.indexOf(n) !== -1; }});
    var t5Hit = race.actual_top3.every(function(n) {{ return top5nums.indexOf(n) !== -1; }});
    if (boxHit && !t5Hit) {{ tag = 'RESCUED ✓'; tagColor = '#22c55e'; }}
    else if (boxHit) {{ tag = 'hit'; tagColor = '#94a3b8'; }}
    else {{ tag = 'missed'; tagColor = '#ef4444'; }}
  }}
  return {{ best: best, pct: bestPct, box6nums: box6nums, tag: tag, tagColor: tagColor }};
}}

// EXPERIMENT (owner-requested, display-only): a straight 3-horse TRIO of our
// top-2 + one "lucky" RANDOM third drawn from the rest of the field. This is
// NOT model-driven for the 3rd leg, NOT staked, and is completely isolated
// from the ML pipeline (never a feature, never trained on, never fed to the
// recommender). The pick is DETERMINISTIC per race (FNV-1a hash of the race
// number + eligible horse numbers) so the "lucky" horse stays the same across
// dashboard refreshes — a bet you can actually place, not re-rolled each render.
function _luckyTrio(race) {{
  if (race.conviction_tier === 'EDGE') return null;  // skip high-conviction EDGE races
  if (!race.horses || race.horses.length < 4) return null;
  var sorted = race.horses.slice().sort(function(a, b) {{ return (a.rank || 99) - (b.rank || 99); }});
  var top2 = sorted.slice(0, 2);
  var rest = sorted.slice(2);
  if (rest.length === 0) return null;
  var seedStr = String(race.race_number || '') + ':' + rest.map(function(h) {{ return h.number; }}).join(',');
  var hsh = 2166136261;
  for (var i = 0; i < seedStr.length; i++) {{ hsh ^= seedStr.charCodeAt(i); hsh = Math.imul(hsh, 16777619); }}
  var third = rest[(hsh >>> 0) % rest.length];
  var nums = [top2[0].number, top2[1].number, third.number];
  var tag = '', tagColor = '#94a3b8';
  if (race.results_available && Array.isArray(race.actual_top3) && race.actual_top3.length === 3) {{
    var hit = race.actual_top3.every(function(n) {{ return nums.indexOf(n) !== -1; }});
    tag = hit ? 'HIT \\u2713' : 'missed';
    tagColor = hit ? '#22c55e' : '#ef4444';
  }}
  return {{ legs: [top2[0], top2[1], third], nums: nums, third: third, tag: tag, tagColor: tagColor }};
}}

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
          '<td style="padding:4px">' + h.number + '</td>' +
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

  var bets = {{}};
  race.horses.forEach(function(h) {{
    var b = (h.bet_recommendation || '').trim();
    if (b && b !== 'SKIP') {{ (bets[b] = bets[b] || []).push(h); }}
  }});
  var ticketLines = [];
  Object.keys(bets).forEach(function(betType) {{
    var picks = bets[betType];
    var label = betType;
    if ((betType === 'QIN' || betType === 'BOX QIN') && picks.length >= 2) {{
      label = picks.length === 2 ? 'STRAIGHT QIN' : 'BOX QIN-' + picks.length;
    }} else if (betType === 'TRIO' && picks.length >= 3) {{
      label = picks.length === 3 ? 'STRAIGHT TRIO' : 'BOX TRIO-' + picks.length;
    }}
    ticketLines.push(
      '<div style="margin:6px 0">' +
        '<span style="display:inline-block;background:var(--gold);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' +
          'PRIMARY · ' + esc(label) +
        '</span>' +
        picks.map(_renderPick).join('') +
      '</div>'
    );
  }});

  // QIN PLACE BANKER — for short-favorite races (top pick odds <= 1.8)
  // a straight WIN bet pays barely above stake. QIN PLACE on the
  // favorite with the next 3 model picks gives us 3 combos that hit
  // when the favorite finishes top-3 (very likely) AND any of the
  // bankers finishes top-3. Much better EV than WIN at sub-2.0 odds.
  var topH2 = race.horses[0];
  if (topH2 && topH2.win_odds && topH2.win_odds <= 1.8 && race.horses.length >= 4) {{
    var bankers = race.horses.slice(1, 4);
    ticketLines.push(
      '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25)">' +
        '<span style="display:inline-block;background:rgba(59,130,246,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' +
          '+ QIN PLACE BANKER · 3 combos' +
        '</span>' +
        '<span style="font-weight:600">#' + esc(topH2.number) + ' ' + esc(topH2.name) + ' </span>' +
        '<span style="color:var(--dim);margin:0 4px">with</span>' +
        bankers.map(function(h) {{ return '#' + esc(h.number) + ' ' + esc(h.name); }}).join(' / ') +
        '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">Hits when fav places (very likely at $' + esc(topH2.win_odds) + ') and any banker places. Way better ROI than straight WIN.</div>' +
      '</div>'
    );
  }}

  // BOX Q4 PROTECTION — always shown when the race has at least 4 ranked
  // horses, regardless of the recommender primary choice. Catches the
  // R1-PACKING-KING-style upsets where the placing pair sits in model
  // ranks 3-4 outside our straight QIN. 6 combinations.
  var top4 = race.horses.slice(0, 4);
  if (top4.length === 4) {{
    var primaryNums = {{}};
    Object.keys(bets).forEach(function(bt) {{ bets[bt].forEach(function(h) {{ primaryNums[h.number] = true; }}); }});
    var hasNonPrimary = top4.some(function(h) {{ return !primaryNums[h.number]; }});
    if (hasNonPrimary || ticketLines.length === 0) {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25)">' +
          '<span style="display:inline-block;background:rgba(16,185,129,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.74rem;margin-right:8px;vertical-align:middle">' +
            '+ PROTECTION · BOX QIN-4 (6 combos)' +
          '</span>' +
          top4.map(_renderPick).join('') +
        '</div>'
      );
    }}
    // FIRST 4 (exact order) — 1 unit, longshot saver. Top 4 model
    // picks placed in 1st-2nd-3rd-4th. Pays big when it hits.
    // Gated to skip low-EV setups: heavy favs collapse the F4 dividend
    // pool; LOW conviction means the rank-1→4 ordering is unreliable;
    // small fields shrink combos and dividends. Only fires when the
    // model has real conviction in the order AND the field is wide
    // enough for a meaningful payout.
    var f4Top1Odds = (race.horses[0] && race.horses[0].win_odds) || 0;
    var f4Tier = race.conviction_tier || '';
    var f4Field = race.horses.length;
    var f4SkipReason = '';
    if (f4Top1Odds > 0 && f4Top1Odds < 2.5) {{
      f4SkipReason = 'heavy fav $' + f4Top1Odds.toFixed(1) + ' — F4 dividends collapse';
    }} else if (f4Tier !== 'EDGE' && f4Tier !== 'HIGH' && f4Tier !== 'MED') {{
      f4SkipReason = f4Tier + ' conviction — top-4 ordering unreliable';
    }} else if (f4Field < 8) {{
      f4SkipReason = 'small field ' + f4Field + ' — F4 pool too thin';
    }}
    if (!f4SkipReason) {{
      var f4Nums = top4.map(function(h) {{ return '#' + esc(h.number); }}).join('  →  ');
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem">' +
          '<span style="display:inline-block;background:rgba(168,85,247,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
            'F4 SAVER · 1 unit' +
          '</span>' +
          '<span style="color:var(--fg);font-weight:600">' + f4Nums + '</span>' +
          '<span style="color:var(--dim);font-size:.7rem;margin-left:8px">(exact order — 1st → 4th)</span>' +
        '</div>'
      );
    }} else {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.72rem;color:var(--dim)">' +
          '<span style="display:inline-block;background:rgba(71,85,105,.4);color:var(--dim);font-weight:700;padding:2px 8px;border-radius:3px;font-size:.65rem;margin-right:8px">' +
            'F4 SAVER · SKIPPED' +
          '</span>' +
          esc(f4SkipReason) +
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

  // F4 BOX-5 — top 5 model picks in a 5-combo F4 box (any-order).
  // Shipped 2026-05-06 after 60-race backtest: 17% hit rate, +54% ROI
  // ($50 stake, median $461 dividend per hit) — strictly better economics
  // than BOX TRIO-5 (+8% ROI). Same gates as F4 SAVER apply: heavy fav,
  // LOW conv, or small field collapse the F4 dividend pool.
  var top5_f4 = top5_eff;
  if (top5_f4.length === 5) {{
    var f4bTop1Odds = (race.horses[0] && race.horses[0].win_odds) || 0;
    var f4bTier = race.conviction_tier || '';
    var f4bField = race.horses.length;
    var f4bSkipReason = '';
    if (f4bTop1Odds > 0 && f4bTop1Odds < 2.5) {{
      f4bSkipReason = 'heavy fav $' + f4bTop1Odds.toFixed(1) + ' — F4 dividends collapse';
    }} else if (f4bTier !== 'EDGE' && f4bTier !== 'HIGH' && f4bTier !== 'MED') {{
      f4bSkipReason = f4bTier + ' conviction — top-5 coverage unreliable';
    }} else if (f4bField < 8) {{
      f4bSkipReason = 'small field ' + f4bField + ' — F4 pool too thin';
    }}
    if (!f4bSkipReason) {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem">' +
          '<span style="display:inline-block;background:linear-gradient(135deg,rgba(168,85,247,.95),rgba(236,72,153,.85));color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
            '★ MAIN · F4 BOX-5 · 5 combos' +
          '</span>' +
          swapBadge +
          top5_f4.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ') +
          '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">Hits when the top-4 finishers are all inside these 5. Backtest: 17% hit, +54% ROI on 60 races (median dividend ~$461 per hit, 9× the $50 stake).</div>' +
        '</div>'
      );
    }} else {{
      ticketLines.push(
        '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.72rem;color:var(--dim)">' +
          '<span style="display:inline-block;background:rgba(71,85,105,.4);color:var(--dim);font-weight:700;padding:2px 8px;border-radius:3px;font-size:.65rem;margin-right:8px">' +
            'F4 BOX-5 · SKIPPED' +
          '</span>' +
          esc(f4bSkipReason) +
        '</div>'
      );
    }}
  }}

  // BOX TRIO-5 protection — top 5 model picks in a 10-combo box.
  // Hits when any 3 of the 5 finish top-3 in any order. Validated on
  // Apr 26 R10 by a user-driven 4-horse trio that paid 12x. Adds the
  // 5th model pick to expand placing-pool coverage beyond the QIN-4
  // box. 10 combos at 1 unit = 10-unit stake. Uses the swapped top-5
  // (same rank-5 swap as F4 BOX-5) for consistency.
  var top5 = top5_eff;
  if (top5.length === 5) {{
    ticketLines.push(
      '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem">' +
        '<span style="display:inline-block;background:rgba(236,72,153,.85);color:#000;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
          '+ BOX TRIO-5 · 10 combos' +
        '</span>' +
        swapBadge +
        top5.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ') +
        '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">Hits when any 3 of these 5 finish top-3 in any order. Bigger field coverage than BOX QIN-4 — catches the rank-5/6 spoilers.</div>' +
      '</div>'
    );
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
  var top6 = race.horses.slice(0, 6);
  var swapCandidate = null;
  var swappedOut = null;
  var manualCandidate = null;
  var manualSwappedOut = null;
  // Manual override (race.manual_trio6_horse) wins over auto rank-6 swap.
  // If override matches the current rank-6 horse, it is a no-op swap — just
  // suppress the auto-swap; do not render a misleading "MANUAL #X→#X" badge.
  if (top6.length === 6 && race.manual_trio6_horse) {{
    if (top6[5].number !== race.manual_trio6_horse) {{
      manualCandidate = race.horses.find(function(h) {{ return h.number === race.manual_trio6_horse; }});
      if (manualCandidate) {{
        manualSwappedOut = top6[5];
        top6 = race.horses.slice(0, 5).concat([manualCandidate]);
      }}
    }}
    // else: override matches current rank-6 — suppress auto-swap silently
  }} else if (top6.length === 6) {{
    var rank6 = top6[5];
    var rank6Odds = (rank6 && rank6.win_odds) || 0;
    if (rank6Odds > 0) {{
      for (var swi = 6; swi < Math.min(12, race.horses.length); swi++) {{
        var cand = race.horses[swi];
        var candOdds = cand.win_odds || 0;
        var candGear = (cand.gear || '').trim();
        if (candOdds > 0 && candGear && candOdds <= rank6Odds * 0.8) {{
          if (!swapCandidate || candOdds < swapCandidate.win_odds) {{
            swapCandidate = cand;
          }}
        }}
      }}
    }}
    if (swapCandidate) {{
      swappedOut = top6[5];
      top6 = race.horses.slice(0, 5).concat([swapCandidate]);
    }}
  }}
  if (top6.length === 6) {{
    // Practical decision rule (validated across 6 meetings, 60 races):
    //   SKIP if heavy favorite (top1 odds < $2.5) — model+market aligned
    //   TAKE if ≥2 morning-steam horses (smart money disagrees → spoilers)
    //   TAKE if HIGH tier + field ≥11 + top1 odds $2.5-$5 (competitive race)
    //   USER CALL otherwise (coin-flip economically; +10pp lift, 2× cost)
    var top1Odds = (race.horses[0] && race.horses[0].win_odds) || 0;
    var fieldSize = race.horses.length;
    var tier = race.conviction_tier || '';
    var steamCount = race.horses.slice(0, 12).filter(function(h) {{
      return h.steam_move && h.steam_move.is_steam;
    }}).length;

    // Chaos score 0-4: each signal that suggests rank-6 spoiler likely
    // CHAOS score 0-4 — same tightened conditions as WA path. Tier and
    // ≥11-field were always-on at ST; dropped/raised. Added top_2_share<0.40
    // (model uncertain) which is the strongest chaos signal.
    var top2ShareChaos = race.top_2_share || 0;
    var chaosScore = 0;
    var chaosFactors = [];
    if (fieldSize >= 13) {{ chaosScore++; chaosFactors.push('large field ' + fieldSize); }}
    if (top1Odds >= 2.5 && top1Odds <= 5.0) {{ chaosScore++; chaosFactors.push('competitive fav $' + top1Odds.toFixed(1)); }}
    if (steamCount >= 2) {{ chaosScore++; chaosFactors.push(steamCount + ' steam horses'); }}
    if (top2ShareChaos > 0 && top2ShareChaos < 0.40) {{ chaosScore++; chaosFactors.push('top-2 share ' + (top2ShareChaos*100).toFixed(0) + '% (model uncertain)'); }}
    var chaosColor = chaosScore >= 3 ? 'rgba(244,63,94,.85)' :
                     chaosScore >= 2 ? 'rgba(251,191,36,.7)' :
                     'rgba(148,163,184,.3)';
    var chaosTextColor = chaosScore >= 3 ? '#fff' : (chaosScore >= 2 ? '#000' : 'var(--dim)');
    var chaosBadge = '<span style="background:' + chaosColor + ';color:' + chaosTextColor + ';padding:1px 6px;border-radius:3px;font-size:.62rem;font-weight:700;margin-left:6px" title="' + chaosFactors.join(' · ') + '">CHAOS ' + chaosScore + '/4</span>';

    var rec = '';
    var recBadge = '';
    var recReason = '';
    if (top1Odds > 0 && top1Odds < 2.5) {{
      rec = 'SKIP';
      recBadge = '<span style="background:rgba(148,163,184,.3);color:var(--dim);padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:700">SKIP — heavy fav ($' + top1Odds.toFixed(1) + ')</span>';
      recReason = 'Top pick is heavy favorite — podium concentrates in TOP-3, TRIO-6 wastes 2× cost.';
    }} else if (steamCount >= 2) {{
      rec = 'TAKE';
      recBadge = '<span style="background:rgba(34,197,94,.85);color:#000;padding:2px 8px;border-radius:3px;font-size:.65rem;font-weight:800">★ TAKE — ' + steamCount + ' steam horses</span>';
      recReason = 'Multiple morning-steam horses — smart money disagrees with model, rank-6 spoiler likely.';
    }} else if (tier === 'HIGH' && fieldSize >= 11 && top1Odds >= 2.5 && top1Odds <= 5.0) {{
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
    ticketLines.push(
      '<div style="margin:6px 0;padding-top:6px;border-top:1px dashed rgba(251,191,36,.25);font-size:.78rem;opacity:.92">' +
        '<span style="display:inline-block;background:rgba(244,63,94,.6);color:#fff;font-weight:800;padding:3px 10px;border-radius:4px;font-size:.7rem;margin-right:8px;vertical-align:middle">' +
          '+ BOX TRIO-6 · 20 combos · OPTIONAL' +
        '</span>' +
        top6.map(function(h) {{ return '#' + esc(h.number); }}).join('  +  ') +
        ' · ' + recBadge + chaosBadge + swapBadge +
        '<div style="font-size:.7rem;color:var(--dim);margin-top:3px">' + recReason +
          ' Signals: ' + (chaosFactors.length ? chaosFactors.join(', ') : 'none') + '.' + swapNote +
        '</div>' +
      '</div>'
    );
  }}

  var betTicketHTML = ticketLines.length
    ? '<div class="bet-ticket" style="background:rgba(251,191,36,.08);border-left:3px solid var(--gold);padding:8px 10px;margin:10px 0;border-radius:4px">' +
        '<div style="font-size:.62rem;color:var(--gold);letter-spacing:.12em;font-weight:700;margin-bottom:4px">BET TICKET</div>' +
        ticketLines.join('') +
      '</div>'
    : '<div class="bet-ticket" style="background:rgba(71,85,105,.18);border-left:3px solid var(--dim);padding:8px 10px;margin:10px 0;border-radius:4px;color:var(--dim);font-size:.78rem">' +
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
  var cleanPath = location.pathname + location.hash.replace(/[?&]v=\d+/, '');
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
      var heroBtns = document.querySelectorAll('#hero .wa-btn');
      if (heroBtns.length === 0) {{
        copyAllBtn.textContent = 'No bets today';
        setTimeout(function() {{ copyAllBtn.textContent = 'Copy all bets'; }}, 1500);
        return;
      }}
      var payloads = [];
      heroBtns.forEach(function(btn) {{
        // data-payload is a plain string (escAttr-encoded); already decoded by browser.
        if (btn.dataset.payload) payloads.push(btn.dataset.payload);
      }});
      var combined = payloads.join('\\n\\n---\\n\\n');
      navigator.clipboard.writeText(combined).then(function() {{
        copyAllBtn.textContent = 'Copied!';
        setTimeout(function() {{ copyAllBtn.textContent = 'Copy all bets'; }}, 1500);
      }});
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
