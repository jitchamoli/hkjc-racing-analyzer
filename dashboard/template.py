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
                  venue: str = 'Sha Tin', auto_refresh: bool = False) -> str:
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

    return _HTML_TEMPLATE.format(
        races_json=races_json_safe,
        now=now,
        race_date=race_date,
        venue=venue,
        auto_refresh_meta=auto_refresh_meta,
        strip_tiles_html=strip_tiles_html,
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
            ' onclick="showRaceDetail({num})" title="{tier}">'
            '<div class="strip-num">{label}</div>'
            '<div class="strip-sub">{sub}</div>'
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
  <div>
    <span class="topbar-brand">HKJC Racing</span>
    <span class="topbar-meta" style="margin-left:10px">{race_date} · {venue}</span>
  </div>
  <div class="topbar-meta">
    <span class="live-dot"></span>
    <span id="clock">{now}</span>
    <span style="margin-left:8px">|</span>
    <span style="margin-left:8px">Last refresh {now}</span>
    <button id="refresh-btn" onclick="refreshDashboard()" style="display:none;margin-left:12px;background:linear-gradient(135deg,#3b82f6,#1d4ed8);color:#fff;border:none;padding:5px 12px;border-radius:6px;font-size:.75rem;font-weight:700;cursor:pointer;">
      ↻ Refresh
    </button>
    <button id="copy-all-btn" class="wa-btn-all" style="margin-left:8px">Copy all bets</button>
  </div>
</div>

<div class="shell">
  <div class="main-col">
    <div class="hero" id="hero"></div>
    <div class="strip-section">
      <div class="strip-label">
        All Races
        <button id="expand-all-btn" class="expand-all-btn">Expand all</button>
      </div>
      <div class="strip" id="strip">{strip_tiles_html}</div>
    </div>
    <div id="race-detail"></div>
    <div class="disclaimer"><strong>DISCLAIMER:</strong> For entertainment only. Not financial advice. Gamble responsibly.</div>
    <div class="footer">HKJC Racing Analyzer v6.4 — Generated: {now}</div>
  </div>
  <div class="alerts-rail" id="alerts-rail"></div>
</div>

<script>
(function() {{
// XSS-safe escape helper
function esc(s) {{ var d=document.createElement('div'); d.textContent=String(s!=null?s:''); return d.innerHTML; }}

const RACES = {races_json};
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
function renderHero() {{
  var hero = document.getElementById('hero');
  var tierOrder = {{EDGE:0, HIGH:1, MED:2}};

  var bets = RACES
    .filter(function(r) {{ return ['EDGE','HIGH','MED'].indexOf(r.conviction_tier) >= 0; }})
    .sort(function(a, b) {{
      if (tierOrder[a.conviction_tier] !== tierOrder[b.conviction_tier])
        return tierOrder[a.conviction_tier] - tierOrder[b.conviction_tier];
      var aPp = Math.max.apply(null, a.horses.map(function(h) {{ return h.edge_pp || 0; }}));
      var bPp = Math.max.apply(null, b.horses.map(function(h) {{ return h.edge_pp || 0; }}));
      if (aPp !== bPp) return bPp - aPp;
      return (b.top_3_share || 0) - (a.top_3_share || 0);
    }})
    .slice(0, 5);

  if (bets.length === 0) {{
    var closest = RACES
      .filter(function(r) {{ return r.top_3_share != null; }})
      .sort(function(a, b) {{ return (b.top_3_share||0) - (a.top_3_share||0); }})[0];
    var closestMsg = closest
      ? 'Closest to HIGH: R' + closest.race_number + ' (top_3_share ' + Math.round((closest.top_3_share||0)*100) + '%)'
      : '';
    hero.innerHTML =
      '<div class="empty-hero">' +
        '<div class="empty-horse">🏇</div>' +
        '<div class="empty-title">No bets meet the threshold today</div>' +
        '<div class="empty-body">Model and market agree too closely for an edge play, and no race clears HIGH/MED conviction on pair identification. The strip below shows all races if you want to drill in, but <strong>skipping the card is the dashboard&#39;s recommendation.</strong></div>' +
        (closestMsg ? '<div class="closest-miss">' + esc(closestMsg) + '</div>' : '') +
      '</div>';
    return;
  }}

  var hasEdgeRace = bets.some(function(r) {{ return r.has_edge; }});
  var html = '<div class="hero-label">TODAY&#39;S BETS — ' + bets.length + ' OF ' + RACES.length + ' RACES</div>';
  if (!hasEdgeRace) {{
    html += '<div class="edge-note">★ EDGE not seen in current sample — HIGH / MED tiers are the operational signal.</div>';
  }}
  bets.forEach(function(race) {{ html += renderBetCard(race); }});
  hero.innerHTML = html;
  hero.querySelectorAll('.wa-btn').forEach(function(btn) {{
    btn.addEventListener('click', function() {{
      copyWA(JSON.parse(btn.dataset.payload), btn);
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

  // Our vs Market
  var purePct = topH.pure_prob != null ? (topH.pure_prob * 100).toFixed(0) : '?';
  var mktPct  = topH.market_prob != null ? (topH.market_prob * 100).toFixed(0) : '?';
  var edgePp  = topH.edge_pp != null ? topH.edge_pp.toFixed(1) : '0.0';

  // Tier Track Record
  var stats = (race.tier_stats || {{}})[tier];
  var trackHtml = stats
    ? (tier + ' hit ' + stats.hit + '/' + stats.total + ' (' + Math.round((stats.rate||0)*100) + '%)' +
       '<br><span style="color:var(--dim);font-size:.65rem">Wilson [' + Math.round((stats.ci_low||0)*100) + '–' + Math.round((stats.ci_high||0)*100) + '%]</span>')
    : '<span style="color:var(--dim)">Insufficient sample</span>';

  // Bet type sub-label
  var betRec = topH.bet_recommendation || 'QIN';
  var raceLabel = 'R' + race.race_number + ' · ' + esc(race.time || '') + ' · ' + esc(race.class_type || '') + ' ' + race.distance + 'm';
  var edgeDetail = isEdge
    ? ('+' + edgePp + 'pp gap · ev +' + (topH.ev_win != null ? topH.ev_win.toFixed(2) : '0.00'))
    : ('top_3 ' + Math.round((race.top_3_share||0)*100) + '%');

  // WA copy payload
  var waPayload = tier + ' — R' + race.race_number + ' #' + topH.number + ' ' + (topH.name||'') +
    ' @ $' + (topH.win_odds||'?') + ' | ' + betRec +
    (whyItems.length ? '\\n' + whyItems.slice(0,2).join(' | ') : '');

  return '<div class="bet-card tier-' + tier + '">' +
    '<div class="bet-card-top">' +
      '<div>' +
        '<span class="tier-badge" style="background:' + tierColor + '">' + (isEdge ? '★ EDGE' : esc(tier)) + '</span>' +
        '<strong style="margin-left:8px">' + raceLabel + '</strong>' +
      '</div>' +
      '<div style="display:flex;align-items:center;gap:8px">' +
        '<span style="font-size:.7rem;color:var(--dim)">' + esc(edgeDetail) + '</span>' +
        '<button class="wa-btn" data-payload="' + esc(JSON.stringify(waPayload)) + '">Copy WA</button>' +
      '</div>' +
    '</div>' +
    '<div class="bet-card-pick">#' + esc(topH.number) + ' ' + esc(topH.name) +
      '<span style="font-weight:400;margin-left:8px;font-size:.82rem">odds ' + esc(topH.win_odds) + ' · ' + esc(betRec) + '</span>' +
    '</div>' +
    '<div class="trust-card">' +
      '<div class="trust-block"><div class="trust-label">Why Strong</div><div class="trust-body">' + whyHtml + '</div></div>' +
      '<div class="trust-block"><div class="trust-label">Our vs Market</div>' +
        '<div class="trust-body" style="margin-top:4px">' +
          'Pure <strong style="color:var(--green)">' + purePct + '%</strong><br>' +
          'Market <strong style="color:var(--dim)">' + mktPct + '%</strong><br>' +
          'Edge <strong style="color:var(--gold)">' + edgePp + 'pp</strong>' +
        '</div>' +
      '</div>' +
      '<div class="trust-block"><div class="trust-label">Tier Track Record</div><div class="trust-body" style="margin-top:4px">' + trackHtml + '</div></div>' +
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
var _activeStrip = null;

function renderStrip() {{
  var strip = document.getElementById('strip');
  strip.innerHTML = RACES.map(function(r) {{
    var tier = r.conviction_tier || 'SKIP';
    var label = tier === 'EDGE' ? 'R' + r.race_number + ' ★' : 'R' + r.race_number;
    var sub = tier === 'SKIP' ? 'skip' : (r.horses[0] ? (r.horses[0].bet_recommendation || '') : '');
    return '<div class="strip-tile tier-' + tier + '" id="tile-' + r.race_number + '" onclick="showRaceDetail(' + r.race_number + ')" title="' + esc(tier) + '">' +
      '<div class="strip-num">' + esc(label) + '</div>' +
      '<div class="strip-sub">' + esc(sub) + '</div>' +
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

  return '<div class="detail-panel">' +
      '<div class="detail-header">' +
        '<div>' +
          '<div class="detail-title">R' + race.race_number + '</div>' +
          '<div class="detail-meta">' + esc(race.time||'') + ' · ' + esc(race.distance) + 'm · ' + esc(race.class_type||'') + ' · ' + esc(race.going||'') + '</div>' +
        '</div>' +
        '<span style="font-size:.72rem;padding:2px 8px;border-radius:4px;background:' + (TIER_COLORS[race.conviction_tier]||'#64748b') + ';color:#000;font-weight:800">' + esc(race.conviction_tier||'SKIP') + '</span>' +
      '</div>' +
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
      speedMapHTML +
      reasoningBlock +
    '</div>';
}}

function showRaceDetail(raceNum) {{
  var race = null;
  for (var i=0; i<RACES.length; i++) {{ if (RACES[i].race_number === raceNum) {{ race = RACES[i]; break; }} }}
  if (!race) return;

  // If "Expand all" is active, clicking a tile collapses back to single-race view
  var expandBtn = document.getElementById('expand-all-btn');
  if (expandBtn && expandBtn.classList.contains('active')) {{
    expandBtn.classList.remove('active');
    expandBtn.textContent = 'Expand all';
  }}

  // Toggle: clicking the same tile again collapses it
  if (_activeStrip === raceNum) {{
    document.getElementById('race-detail').innerHTML = '';
    document.querySelectorAll('.strip-tile').forEach(function(t) {{ t.classList.remove('active'); }});
    _activeStrip = null;
    return;
  }}
  _activeStrip = raceNum;
  document.querySelectorAll('.strip-tile').forEach(function(t) {{ t.classList.remove('active'); }});
  var activeTile = document.getElementById('tile-' + raceNum);
  if (activeTile) activeTile.classList.add('active');

  var det = document.getElementById('race-detail');
  det.innerHTML = _buildRaceDetailPanel(race);
  det.scrollIntoView({{behavior:'smooth', block:'nearest'}});
}}

function toggleExpandAll() {{
  var btn = document.getElementById('expand-all-btn');
  var det = document.getElementById('race-detail');
  if (!btn || !det) return;

  if (btn.classList.contains('active')) {{
    // Collapse
    btn.classList.remove('active');
    btn.textContent = 'Expand all';
    det.innerHTML = '';
    document.querySelectorAll('.strip-tile').forEach(function(t) {{ t.classList.remove('active'); }});
    _activeStrip = null;
    return;
  }}

  // Expand: render every race detail panel stacked, separated by spacing
  btn.classList.add('active');
  btn.textContent = 'Collapse';
  document.querySelectorAll('.strip-tile').forEach(function(t) {{ t.classList.remove('active'); }});
  _activeStrip = null;
  det.innerHTML = RACES.map(function(r) {{
    return '<div style="margin-bottom:12px">' + _buildRaceDetailPanel(r) + '</div>';
  }}).join('');
  det.scrollIntoView({{behavior:'smooth', block:'nearest'}});
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
        try {{ payloads.push(JSON.parse(btn.dataset.payload)); }} catch(e) {{}}
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
// Boot
// ─────────────────────────────────────────────────────────────
renderHero();
renderStrip();
renderAlertsRail();

}})();
</script>
</body>
</html>"""
