#!/usr/bin/env python3
"""HTML template for the HKJC Racing Dashboard."""


def generate_html(races_json, now, race_date, venue="Sha Tin", auto_refresh=False):
    refresh_tag = '\n<meta http-equiv="refresh" content="60">' if auto_refresh else ''
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">{refresh_tag}
<title>HKJC Racing Dashboard - {race_date}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
:root{{--bg:#0a0e1a;--surface:#111827;--surface2:#1a2236;--surface3:#1e293b;--border:#2d3748;--text:#e2e8f0;--dim:#8892a4;--accent:#f59e0b;--gold:#fbbf24;--green:#10b981;--red:#ef4444;--blue:#3b82f6;--purple:#8b5cf6;--cyan:#06b6d4;--pink:#ec4899;--orange:#f97316}}
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh}}
button{{font-family:inherit;cursor:pointer}}
.header{{background:linear-gradient(135deg,#0f172a 0%,#1e1b4b 40%,#312e81 100%);border-bottom:2px solid var(--accent);padding:16px 24px;position:sticky;top:0;z-index:100}}
.header-row{{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px}}
.header h1{{font-size:1.6rem;font-weight:900;background:linear-gradient(90deg,#fbbf24,#f59e0b,#d97706);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.header-meta{{display:flex;gap:12px;align-items:center;font-size:.8rem;color:var(--dim)}}
.version{{background:linear-gradient(90deg,var(--accent),#d97706);color:#000;padding:3px 10px;border-radius:20px;font-size:.68rem;font-weight:800}}
.live-dot{{width:8px;height:8px;border-radius:50%;background:var(--green);animation:pulse 2s infinite;display:inline-block}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.race-tabs{{display:flex;gap:6px;padding:12px 24px;overflow-x:auto;background:var(--surface);border-bottom:1px solid var(--border);position:sticky;top:62px;z-index:99}}
.race-tab{{padding:8px 16px;border-radius:8px;border:1px solid var(--border);background:transparent;color:var(--dim);font-size:.78rem;font-weight:600;transition:all .2s;white-space:nowrap;display:flex;align-items:center;gap:6px}}
.race-tab:hover{{background:var(--surface2);color:var(--text);border-color:var(--accent)}}
.race-tab.active{{background:var(--accent);color:#000;border-color:var(--accent);font-weight:800}}
.race-tab .tab-grade{{font-size:.6rem;padding:1px 5px;border-radius:10px;font-weight:800}}
.race-tab.active .tab-grade{{background:rgba(0,0,0,.2);color:#000}}
.tab-grade.g-aplus{{background:rgba(251,191,36,.25);color:var(--gold)}}
.tab-grade.g-a{{background:rgba(16,185,129,.25);color:var(--green)}}
.tab-grade.g-bplus{{background:rgba(59,130,246,.25);color:var(--blue)}}
.tab-grade.g-b{{background:rgba(139,92,246,.25);color:var(--purple)}}
.tab-grade.g-c{{background:rgba(100,116,139,.2);color:var(--dim)}}
.next-badge{{background:var(--green);color:#000;font-size:.58rem;padding:1px 5px;border-radius:8px;font-weight:800}}
.content{{padding:20px 24px;max-width:1400px;margin:0 auto}}
#dashboard-view{{}}
#race-detail-view{{display:none}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:8px;margin-bottom:16px}}
.stat{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px;text-align:center}}
.stat .v{{font-size:1.3rem;font-weight:800}}
.stat .l{{font-size:.65rem;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;margin-top:2px}}
.legend{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;font-size:.74rem}}
.legend .item{{display:flex;align-items:center;gap:5px}}
.legend .dot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}
.race-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(380px,1fr));gap:14px;margin-bottom:20px}}
.race-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden;cursor:pointer;transition:all .25s}}
.race-card:hover{{transform:translateY(-2px);box-shadow:0 6px 25px rgba(0,0,0,.4);border-color:var(--accent)}}
.race-card.g-aplus{{border-top:3px solid var(--gold)}}
.race-card.g-a{{border-top:3px solid var(--green)}}
.race-card.g-bplus{{border-top:3px solid var(--blue)}}
.race-card.g-b{{border-top:3px solid var(--purple)}}
.race-card.g-c{{border-top:3px solid var(--border)}}
.rc-header{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border)}}
.rc-num{{font-weight:800;font-size:.95rem}}
.rc-meta{{font-size:.7rem;color:var(--dim)}}
.grade{{font-size:.65rem;font-weight:800;padding:3px 9px;border-radius:16px}}
.grade.aplus{{background:rgba(251,191,36,.2);color:var(--gold)}}
.grade.a{{background:rgba(16,185,129,.2);color:var(--green)}}
.grade.bplus{{background:rgba(59,130,246,.2);color:var(--blue)}}
.grade.b{{background:rgba(139,92,246,.2);color:var(--purple)}}
.grade.c{{background:rgba(100,116,139,.2);color:var(--dim)}}
.rc-preview{{padding:10px 16px}}
.rc-horse{{display:grid;grid-template-columns:24px 1fr 54px 48px 52px;align-items:center;gap:6px;padding:5px 0;font-size:.78rem;border-bottom:1px solid rgba(45,55,72,.3)}}
.rc-horse:last-child{{border:none}}
.rc-rank{{font-size:.68rem;font-weight:800;color:var(--dim);width:22px;height:22px;display:flex;align-items:center;justify-content:center;border-radius:50%;background:rgba(255,255,255,.05)}}
.rc-horse.top .rc-rank{{background:var(--accent);color:#000}}
.rc-name{{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.rc-odds{{text-align:right;color:var(--cyan);font-weight:600;font-size:.8rem}}
.rc-prob{{text-align:right;font-weight:700;font-size:.8rem}}
.bt{{display:inline-block;font-size:.62rem;font-weight:700;padding:2px 6px;border-radius:4px;text-align:center}}
.rc-footer{{padding:8px 16px;background:var(--surface2);font-size:.7rem;color:var(--dim);display:flex;justify-content:space-between;align-items:center}}
.rc-footer .click-hint{{color:var(--blue);font-weight:600}}
.detail-header{{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:16px;flex-wrap:wrap;gap:12px}}
.detail-title{{font-size:1.4rem;font-weight:800}}
.detail-meta{{font-size:.85rem;color:var(--dim);margin-top:4px}}
.back-btn{{background:var(--surface2);color:var(--text);border:1px solid var(--border);padding:8px 18px;border-radius:8px;font-size:.82rem;font-weight:600;transition:all .2s}}
.back-btn:hover{{background:var(--accent);color:#000;border-color:var(--accent)}}
.detail-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin-bottom:16px}}
.horse-table{{width:100%;border-collapse:collapse;font-size:.82rem;margin-bottom:16px}}
.horse-table th{{background:var(--surface2);padding:10px 12px;text-align:left;font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:var(--dim);border-bottom:2px solid var(--border)}}
.horse-table td{{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:top}}
.horse-table tr:hover{{background:var(--surface2)}}
.horse-table tr.top-row{{background:rgba(245,158,11,.05)}}
.horse-table .reasoning-cell{{font-size:.72rem;color:var(--dim);line-height:1.5;max-width:300px}}
.nav-arrows{{display:flex;gap:8px;margin-top:16px;justify-content:center}}
.nav-btn{{background:var(--surface);border:1px solid var(--border);color:var(--text);padding:10px 20px;border-radius:8px;font-size:.85rem;font-weight:600;transition:all .2s}}
.nav-btn:hover{{background:var(--accent);color:#000;border-color:var(--accent)}}
.nav-btn:disabled{{opacity:.3;cursor:not-allowed}}
.nav-btn:disabled:hover{{background:var(--surface);color:var(--text)}}
.strategy-panel{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px}}
.strategy-panel h3{{font-size:.9rem;font-weight:700;margin-bottom:10px}}
.strat-row{{display:flex;gap:12px;align-items:center;padding:6px 0;font-size:.82rem;border-bottom:1px solid rgba(45,55,72,.3)}}
.strat-row:last-child{{border:none}}
.disclaimer{{background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.2);border-radius:8px;padding:10px 16px;margin-top:16px;font-size:.72rem;color:#fca5a5}}
.footer{{text-align:center;padding:20px;font-size:.72rem;color:var(--dim);border-top:1px solid var(--border);margin-top:16px}}
@media(max-width:768px){{.race-grid{{grid-template-columns:1fr}}.content{{padding:12px 14px}}.race-tabs{{padding:10px 14px}}.header{{padding:12px 14px}}.detail-header{{flex-direction:column}}}}
.market-badge{{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:12px;font-size:.65rem;font-weight:700}}
.market-badge.steam{{background:rgba(16,185,129,.2);color:var(--green)}}
.market-badge.drift{{background:rgba(239,68,68,.2);color:var(--red)}}
.market-badge.cold{{background:rgba(251,191,36,.2);color:var(--gold)}}
.enhanced-stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:6px;margin-top:8px;padding:8px;background:var(--surface2);border-radius:8px;font-size:.72rem}}
.enhanced-stats .es-item{{display:flex;flex-direction:column;gap:2px}}
.enhanced-stats .es-label{{color:var(--dim);font-size:.62rem;text-transform:uppercase}}
.enhanced-stats .es-value{{font-weight:600}}
.speed-map{{background:var(--surface2);border-radius:10px;padding:14px;margin:12px 0}}
.speed-map-title{{font-size:.78rem;font-weight:700;margin-bottom:10px;color:var(--accent)}}
.speed-map-track{{position:relative;height:180px;background:linear-gradient(180deg,rgba(16,185,129,.05) 0%,rgba(16,185,129,.02) 100%);border:1px solid var(--border);border-radius:8px;overflow:hidden}}
.speed-map-zone{{position:absolute;left:0;right:0;display:flex;align-items:center;padding:0 10px;font-size:.6rem;color:var(--dim)}}
.speed-map-zone.leaders{{top:0;height:25%;border-bottom:1px dashed var(--border)}}
.speed-map-zone.stalkers{{top:25%;height:25%;border-bottom:1px dashed var(--border)}}
.speed-map-zone.midfield{{top:50%;height:25%;border-bottom:1px dashed var(--border)}}
.speed-map-zone.closers{{top:75%;height:25%}}
.speed-map-horse{{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;background:var(--accent);color:#000;font-size:.68rem;font-weight:800;margin:2px}}
.speed-map-horse.inside{{border:2px solid var(--green)}}
.speed-map-horse.outside{{border:2px solid var(--red)}}
</style>
</head>
<body>
<div class="header">
  <div class="header-row">
    <h1>HKJC Racing Dashboard</h1>
    <div class="header-meta">
      <span class="live-dot"></span>
      <span>{venue} | {race_date}</span>
      <span>|</span>
      <span id="clock">{now}</span>
      <span class="version">v5.3</span>
    </div>
  </div>
</div>
<div class="race-tabs" id="race-tabs">
  <button class="race-tab active" onclick="showDashboard()" id="tab-all">All Races</button>
</div>
<div class="content">
  <div id="dashboard-view"></div>
  <div id="race-detail-view"></div>
</div>
<script>
function esc(s){{var d=document.createElement('div');d.textContent=String(s!=null?s:'');return d.innerHTML;}}
const RACES = {races_json};
const RACE_DATE = "{race_date}";
const GENERATED = "{now}";

function getNextRace() {{
  var now = new Date();
  var parts = RACE_DATE.split('/');
  // Check if all races have results
  var allDone = RACES.every(function(r) {{ return r.results_available; }});
  if (allDone) return -1;  // Race day complete
  for (var i = 0; i < RACES.length; i++) {{
    var r = RACES[i];
    if (r.results_available) continue;
    var tp = (r.time || '').split(':');
    if (tp.length === 2) {{
      var raceTime = new Date(parseInt(parts[0]), parseInt(parts[1])-1, parseInt(parts[2]), parseInt(tp[0]), parseInt(tp[1]));
      if (raceTime > now) return r.race_number;
    }}
    // Race time passed but no result yet — it's likely running now
    return r.race_number;
  }}
  return -1;  // All done
}}

function formatCountdown(raceNum) {{
  var race = RACES.find(function(r) {{ return r.race_number === raceNum; }});
  if (!race || !race.time) return '';
  var parts = RACE_DATE.split('/');
  var tp = race.time.split(':');
  var raceTime = new Date(parseInt(parts[0]), parseInt(parts[1])-1, parseInt(parts[2]), parseInt(tp[0]), parseInt(tp[1]));
  var diff = raceTime - new Date();
  if (diff <= 0) return 'NOW';
  var h = Math.floor(diff/3600000);
  var m = Math.floor((diff%3600000)/60000);
  var s = Math.floor((diff%60000)/1000);
  if (h > 0) return h + 'h ' + m + 'm';
  if (m > 0) return m + 'm ' + s + 's';
  return s + 's';
}}

var NEXT_RACE = getNextRace();

// Live clock + next race update
setInterval(function() {{
  var el = document.getElementById('clock');
  if (el) {{
    var now = new Date();
    var timeStr = now.toLocaleTimeString('en-HK', {{hour:'2-digit',minute:'2-digit',second:'2-digit'}});
    var nextR = getNextRace();
    if (nextR === -1) {{
      el.textContent = timeStr + ' | Race Day Complete';
    }} else {{
      var cd = formatCountdown(nextR);
      el.textContent = timeStr + (cd ? ' | R' + nextR + ' in ' + cd : ' | R' + nextR + ' NOW');
    }}
    if (nextR !== NEXT_RACE) {{
      NEXT_RACE = nextR;
      updateNextBadge();
    }}
  }}
}}, 1000);

function updateNextBadge() {{
  document.querySelectorAll('.next-badge').forEach(function(b) {{ b.remove(); }});
  if (NEXT_RACE === -1) return;  // All races done, no badge
  var tab = document.getElementById('tab-' + NEXT_RACE);
  if (tab) {{
    var badge = document.createElement('span');
    badge.className = 'next-badge';
    badge.textContent = 'NEXT';
    tab.appendChild(badge);
  }}
}}

function betColor(bt) {{return {{'BANKER WIN':'#fbbf24','WIN':'#10b981','PLACE':'#8b5cf6','QIN':'#f59e0b','QNL PL':'#06b6d4','TRIO':'#ec4899','SKIP':'#64748b','E/W':'#3b82f6'}}[bt] || '#64748b'}}
function betBg(bt) {{return {{'BANKER WIN':'rgba(251,191,36,.25)','WIN':'rgba(16,185,129,.15)','PLACE':'rgba(139,92,246,.15)','QIN':'rgba(245,158,11,.15)','QNL PL':'rgba(6,182,212,.15)','TRIO':'rgba(236,72,153,.15)','SKIP':'rgba(100,116,139,.1)'}}[bt] || 'rgba(100,116,139,.1)'}}
function gradeRace(race) {{const p = (race.horses[0]?.calibrated_prob || 0) * 100;if(p>=18)return'A+';if(p>=14)return'A';if(p>=11)return'B+';if(p>=9)return'B';return'B-'}}
function gradeClass(g) {{return {{'A+':'aplus','A':'a','B+':'bplus','B':'b','B-':'c'}}[g]||'c'}}
function probColor(p) {{var pct=p*100;if(pct>=15)return'var(--green)';if(pct>=10)return'var(--accent)';return'var(--dim)'}}

function initTabs() {{
  const tabs = document.getElementById('race-tabs');
  RACES.forEach(r => {{
    const g = gradeRace(r);
    const gc = gradeClass(g);
    const isNext = NEXT_RACE !== -1 && r.race_number === NEXT_RACE;
    const btn = document.createElement('button');
    btn.className = 'race-tab';
    btn.id = 'tab-' + r.race_number;
    btn.onclick = () => showRace(r.race_number);
    btn.innerHTML = 'R' + r.race_number + ' <span class="tab-grade g-' + gc + '">' + g + '</span>' + (isNext ? ' <span class="next-badge">NEXT</span>' : '');
    tabs.appendChild(btn);
  }});
}}

function showDashboard() {{
  document.getElementById('dashboard-view').style.display = 'block';
  document.getElementById('race-detail-view').style.display = 'none';
  document.querySelectorAll('.race-tab').forEach(t => t.classList.remove('active'));
  document.getElementById('tab-all').classList.add('active');
  renderDashboard();
}}

function renderDashboard() {{
  const el = document.getElementById('dashboard-view');
  const counts = {{'BANKER WIN':0,WIN:0,PLACE:0,QIN:0,'QNL PL':0,TRIO:0,SKIP:0}};
  RACES.forEach(r => r.horses.forEach(h => {{if(counts[h.bet_recommendation]!==undefined)counts[h.bet_recommendation]++}}));
  const active = counts.WIN + counts.PLACE + counts.QIN + counts['QNL PL'] + counts.TRIO;
  
  let html = '<div class="stats">' +
    '<div class="stat"><div class="v" style="color:var(--gold)">' + counts.WIN + '</div><div class="l">WIN</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--purple)">' + counts.PLACE + '</div><div class="l">PLACE</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--accent)">' + counts.QIN + '</div><div class="l">QIN</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--cyan)">' + counts['QNL PL'] + '</div><div class="l">QNL PL</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--pink)">' + counts.TRIO + '</div><div class="l">TRIO</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--dim)">' + counts.SKIP + '</div><div class="l">SKIP</div></div>' +
    '<div class="stat"><div class="v">' + RACES.length + '</div><div class="l">Races</div></div>' +
    '<div class="stat"><div class="v">24</div><div class="l">Factors</div></div>' +
  '</div>' +
  '<div class="legend">' +
    '<div class="item"><div class="dot" style="background:var(--green)"></div>WIN</div>' +
    '<div class="item"><div class="dot" style="background:var(--purple)"></div>PLACE</div>' +
    '<div class="item"><div class="dot" style="background:var(--accent)"></div>QIN</div>' +
    '<div class="item"><div class="dot" style="background:var(--cyan)"></div>QNL PL</div>' +
    '<div class="item"><div class="dot" style="background:var(--pink)"></div>TRIO</div>' +
    '<div class="item"><div class="dot" style="background:var(--dim)"></div>SKIP</div>' +
  '</div>' +
  '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:14px;font-size:.72rem;color:var(--dim);align-items:center">' +
    '<span style="font-weight:700;color:var(--text)">Race Grades:</span>' +
    '<span><span class="grade aplus" style="font-size:.62rem;padding:2px 6px">A+</span> Strong favourite \\u2014 back to win</span>' +
    '<span><span class="grade a" style="font-size:.62rem;padding:2px 6px">A</span> Good edge \\u2014 win or quinella</span>' +
    '<span><span class="grade bplus" style="font-size:.62rem;padding:2px 6px">B+</span> Decent pick \\u2014 quinella or place</span>' +
    '<span><span class="grade b" style="font-size:.62rem;padding:2px 6px">B</span> Open race \\u2014 exotic bets (trio)</span>' +
    '<span><span class="grade c" style="font-size:.62rem;padding:2px 6px">B-</span> Wide open \\u2014 small bets or skip</span>' +
  '</div>' +
  '<div style="display:flex;justify-content:flex-end;margin-bottom:10px"><button id="wa-btn-all" onclick="copyAllToWhatsApp()" style="background:linear-gradient(135deg,#25d366,#128c7e);color:#fff;border:none;padding:8px 16px;border-radius:8px;font-size:.82rem;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:6px;transition:all .2s"><svg width=\\"16\\" height=\\"16\\" viewBox=\\"0 0 24 24\\" fill=\\"white\\"><path d=\\"M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347z\\"/><path d=\\"M12 0C5.373 0 0 5.373 0 12c0 2.625.846 5.059 2.284 7.034L.789 23.492l4.608-1.467A11.945 11.945 0 0012 24c6.627 0 12-5.373 12-12S18.627 0 12 0zm0 22c-2.188 0-4.214-.696-5.869-1.879l-.421-.314-2.732.87.907-2.677-.346-.449A9.95 9.95 0 012 12C2 6.486 6.486 2 12 2s10 4.486 10 10-4.486 10-10 10z\\"/></svg>Copy All Races</button></div>' +
  '<div class="race-grid">';
  
  RACES.forEach(r => {{
    const g = gradeRace(r);
    const gc = gradeClass(g);
    const hasResults = r.results_available || false;
    const winner = hasResults ? r.actual_winner : null;
    const winnerHorse = winner ? r.horses.find(h => h.number === winner) : null;
    const winnerName = winnerHorse ? winnerHorse.name : '';
    const winOdds = r.actual_win_odds || 0;
    const accuracy = r.accuracy || {{}};
    const winCorrect = accuracy.win_correct || false;
    const top3Overlap = accuracy.top3_overlap || 0;
    const winnerOurRank = accuracy.winner_our_rank || '?';

    html += '<div class="race-card g-' + gc + '" onclick="showRace(' + r.race_number + ')">' +
      '<div class="rc-header">' +
        '<div><div class="rc-num">Race ' + r.race_number + '</div><div class="rc-meta">' + r.time + ' | ' + r.distance + 'm | ' + r.class_type + ' | ' + r.going + '</div></div>' +
        '<span class="grade ' + gc + '">' + g + '</span>' +
      '</div>';

    // Show result banner if results available
    if (hasResults && winner) {{
      var resultBg = winCorrect ? 'rgba(16,185,129,.2)' : (winnerOurRank <= 3 ? 'rgba(245,158,11,.15)' : 'rgba(239,68,68,.12)');
      var resultBorder = winCorrect ? 'var(--green)' : (winnerOurRank <= 3 ? 'var(--accent)' : '#ef4444');
      var resultIcon = winCorrect ? '\\u2705' : (winnerOurRank <= 3 ? '\\ud83d\\udfe1' : '\\u274c');
      var resultLabel = winCorrect ? 'WIN CORRECT!' : (winnerOurRank <= 3 ? 'Top 3 hit (Rank ' + winnerOurRank + ')' : 'Winner was Rank ' + winnerOurRank);
      html += '<div style="background:' + resultBg + ';border-left:3px solid ' + resultBorder + ';padding:8px 12px;margin:0 -1px;display:flex;justify-content:space-between;align-items:center">' +
        '<div style="font-size:.78rem;font-weight:700">' + resultIcon + ' Winner: #' + esc(winner) + ' ' + esc(winnerName) + ' @ $' + esc(winOdds) + '</div>' +
        '<div style="font-size:.68rem;color:var(--dim)">' + resultLabel + ' | Top3: ' + top3Overlap + '/3</div>' +
      '</div>';
    }}

    html += '<div class="rc-preview">';
    r.horses.slice(0, 4).forEach((h, i) => {{
      const tc = i === 0 ? ' top' : '';
      const isActualWinner = hasResults && h.number === winner;
      const actualPos = h.actual_position || '';
      const posLabel = hasResults && actualPos ? '<span style="font-size:.6rem;padding:1px 4px;border-radius:3px;margin-left:4px;' + (actualPos === 1 ? 'background:var(--green);color:#000' : actualPos <= 3 ? 'background:var(--accent);color:#000' : 'background:var(--surface);color:var(--dim)') + '">Fin:' + actualPos + '</span>' : '';
      html += '<div class="rc-horse' + tc + '"' + (isActualWinner ? ' style="background:rgba(16,185,129,.12);border-radius:6px"' : '') + '>' +
        '<div class="rc-rank">' + h.rank + '</div>' +
        '<div class="rc-name">#' + esc(h.number) + ' ' + esc(h.name) + posLabel + '<div style="font-size:.65rem;color:var(--dim);font-weight:400;margin-top:1px">' + esc(h.jockey) + ' | ' + esc(h.trainer) + '</div></div>' +
        '<div class="rc-odds">$' + h.win_odds + '</div>' +
        '<div class="rc-prob" style="color:' + probColor(h.calibrated_prob) + '">' + (h.calibrated_prob * 100).toFixed(1) + '%</div>' +
        '<div class="bt" style="background:' + betBg(h.bet_recommendation) + ';color:' + betColor(h.bet_recommendation) + '">' + h.bet_recommendation + '</div>' +
      '</div>';
    }});
    // Bet badge + exotic count
    var topBet = r.horses[0].bet_recommendation || 'SKIP';
    var betBadge = '<span class="bt" style="background:' + betBg(topBet) + ';color:' + betColor(topBet) + ';margin-right:6px">' + topBet + '</span>';
    var exoticCount = 0;
    if (r.exotic_value) {{
      ['exacta','quinella','trifecta','first4'].forEach(function(t) {{
        exoticCount += (r.exotic_value[t] || []).length;
      }});
    }}
    var exoticBadge = exoticCount > 0 ? '<span style="background:rgba(251,191,36,.2);color:var(--gold);font-size:.62rem;font-weight:700;padding:2px 6px;border-radius:4px;margin-right:6px">' + exoticCount + ' exotics</span>' : '';

    html += '</div>' +
      '<div class="rc-footer"><span>' + betBadge + exoticBadge + r.horses.length + ' runners</span><span class="click-hint">Click to expand</span></div>' +
    '</div>';
  }});
  
  html += '</div>' +
    '<div class="disclaimer"><strong>DISCLAIMER:</strong> For entertainment only. Not financial advice. Gamble responsibly.</div>' +
    '<div class="footer">HKJC Racing Analyzer v5.3 | Generated: ' + GENERATED + '</div>';
  el.innerHTML = html;
}}

function showRace(raceNum) {{
  const race = RACES.find(r => r.race_number === raceNum);
  if(!race) return;
  document.getElementById('dashboard-view').style.display = 'none';
  document.getElementById('race-detail-view').style.display = 'block';
  document.querySelectorAll('.race-tab').forEach(t => t.classList.remove('active'));
  const tab = document.getElementById('tab-' + raceNum);
  if(tab) tab.classList.add('active');
  renderRaceDetail(race);
}}

function buildStrategyPanel(race) {{
  var horses = race.horses;
  var active = horses.filter(function(h) {{ return h.bet_recommendation !== 'SKIP'; }});
  if (!active.length) return '<div class="strategy-panel"><h3>Strategy: SKIP</h3><p style="color:var(--dim);font-size:.82rem">No strong picks in this race. Save your money for a better one.</p></div>';

  // Determine primary bet type
  var betTypes = {{}};
  active.forEach(function(h) {{ betTypes[h.bet_recommendation] = (betTypes[h.bet_recommendation] || 0) + 1; }});
  var primaryBet = Object.keys(betTypes).sort(function(a,b) {{ return betTypes[b] - betTypes[a]; }})[0];

  // Gather selections per type
  var bankerPicks = horses.filter(function(h) {{ return h.bet_recommendation === 'BANKER WIN'; }});
  var winPicks = horses.filter(function(h) {{ return h.bet_recommendation === 'WIN' || h.bet_recommendation === 'BANKER WIN'; }});
  var placePicks = horses.filter(function(h) {{ return h.bet_recommendation === 'PLACE'; }});
  var qinPicks = horses.filter(function(h) {{ return h.bet_recommendation === 'QIN'; }});
  var qnlPicks = horses.filter(function(h) {{ return h.bet_recommendation === 'QNL PL'; }});
  var trioPicks = horses.filter(function(h) {{ return h.bet_recommendation === 'TRIO'; }});

  // Field analysis
  var top3 = horses.slice(0, 3);
  var gap = top3.length >= 2 ? (top3[0].calibrated_prob - top3[1].calibrated_prob) : 0;
  var topShare = top3.reduce(function(s, h) {{ return s + h.calibrated_prob; }}, 0);
  var fieldType = gap > 0.05 ? 'One horse stands out' : gap > 0.02 ? 'Clear top pick' : topShare > 0.4 ? 'Top 3 are close' : 'Wide open race';

  // Best bets (odds in our favour)
  var valueBets = horses.filter(function(h) {{ return h.expected_return > 0.05 && h.bet_recommendation !== 'SKIP'; }});

  var s = '<div class="strategy-panel">';
  s += '<h3 style="display:flex;align-items:center;gap:8px;margin-bottom:12px"><span style="font-size:1.1rem">What To Bet</span><span style="font-size:.7rem;padding:3px 10px;border-radius:12px;background:' + betBg(primaryBet) + ';color:' + betColor(primaryBet) + ';font-weight:700">' + primaryBet + '</span><span style="font-size:.72rem;color:var(--dim)">' + fieldType + ' | ' + race.horses.length + ' runners</span></h3>';

  // Primary recommendation
  s += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:12px;margin-bottom:14px">';

  if (bankerPicks.length) {{
    bankerPicks.forEach(function(h) {{
      var bankerReason = (h.reasoning || []).find(function(r) {{ return r.indexOf('BANKER') === 0; }}) || '';
      s += '<div style="background:linear-gradient(135deg,rgba(251,191,36,.12),rgba(245,158,11,.08));border:2px solid rgba(251,191,36,.5);border-radius:10px;padding:14px 16px">'
        + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        + '<span style="font-size:.8rem;font-weight:800;color:var(--gold);text-transform:uppercase;letter-spacing:1px">BANKER WIN</span>'
        + '<span style="font-size:.68rem;background:rgba(251,191,36,.2);color:var(--gold);padding:2px 8px;border-radius:8px;font-weight:700">BET BIG</span></div>'
        + '<div style="font-size:1.1rem;font-weight:800">#' + esc(h.number) + ' ' + esc(h.name) + '</div>'
        + '<div style="font-size:.78rem;color:var(--dim);margin-top:2px">' + esc(h.jockey) + ' | $' + h.win_odds + ' | ' + (h.calibrated_prob * 100).toFixed(1) + '% chance</div>'
        + (bankerReason ? '<div style="font-size:.7rem;color:var(--gold);margin-top:6px;font-weight:600">' + esc(bankerReason) + '</div>' : '')
        + '</div>';
    }});
  }}

  if (winPicks.length && !bankerPicks.length) {{
    winPicks.forEach(function(h) {{
      var edgeLabel = h.expected_return > 0 ? 'Good odds' : 'Tight odds';
      var edgeColor = h.expected_return > 0 ? 'var(--green)' : 'var(--dim)';
      s += '<div style="background:rgba(16,185,129,.06);border:1px solid rgba(16,185,129,.25);border-radius:10px;padding:12px 16px">'
        + '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">'
        + '<span style="font-size:.72rem;font-weight:700;color:var(--green);text-transform:uppercase">WIN Bet</span>'
        + '<span style="font-size:.72rem;color:' + edgeColor + ';font-weight:700">' + edgeLabel + '</span></div>'
        + '<div style="font-size:1rem;font-weight:800">#' + esc(h.number) + ' ' + esc(h.name) + '</div>'
        + '<div style="font-size:.78rem;color:var(--dim);margin-top:2px">' + esc(h.jockey) + ' | $' + h.win_odds + ' | ' + (h.calibrated_prob * 100).toFixed(1) + '% win chance</div>'
        + '</div>';
    }});
  }}

  if (qinPicks.length >= 2) {{
    var nums = qinPicks.map(function(h) {{ return '#' + esc(h.number); }}).join(' + ');
    var names = qinPicks.map(function(h) {{ return esc(h.name); }}).join(' &amp; ');
    s += '<div style="background:rgba(245,158,11,.06);border:1px solid rgba(245,158,11,.25);border-radius:10px;padding:12px 16px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--accent);text-transform:uppercase;margin-bottom:6px">Quinella (QIN)</div>'
      + '<div style="font-size:1rem;font-weight:800">' + nums + '</div>'
      + '<div style="font-size:.78rem;color:var(--dim);margin-top:2px">' + names + '</div>'
      + '<div style="font-size:.72rem;color:var(--dim);margin-top:4px">These two to finish 1st &amp; 2nd in any order</div>'
      + '</div>';
  }}

  if (qnlPicks.length >= 2) {{
    var nums = qnlPicks.map(function(h) {{ return '#' + esc(h.number); }}).join(' + ');
    var names = qnlPicks.map(function(h) {{ return esc(h.name); }}).join(' &amp; ');
    s += '<div style="background:rgba(6,182,212,.06);border:1px solid rgba(6,182,212,.25);border-radius:10px;padding:12px 16px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--cyan);text-transform:uppercase;margin-bottom:6px">Quinella Place (QNL PL)</div>'
      + '<div style="font-size:1rem;font-weight:800">' + nums + '</div>'
      + '<div style="font-size:.78rem;color:var(--dim);margin-top:2px">' + names + '</div>'
      + '<div style="font-size:.72rem;color:var(--dim);margin-top:4px">These two to both finish in top 3 (any order)</div>'
      + '</div>';
  }}

  if (trioPicks.length >= 3) {{
    var nums = trioPicks.map(function(h) {{ return '#' + esc(h.number); }}).join(' + ');
    var names = trioPicks.map(function(h) {{ return esc(h.name); }}).join(', ');
    s += '<div style="background:rgba(236,72,153,.06);border:1px solid rgba(236,72,153,.25);border-radius:10px;padding:12px 16px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--pink);text-transform:uppercase;margin-bottom:6px">Trio</div>'
      + '<div style="font-size:1rem;font-weight:800">' + nums + '</div>'
      + '<div style="font-size:.78rem;color:var(--dim);margin-top:2px">' + names + '</div>'
      + '<div style="font-size:.72rem;color:var(--dim);margin-top:4px">These three to finish 1st, 2nd &amp; 3rd in any order</div>'
      + '</div>';
  }}

  if (placePicks.length) {{
    s += '<div style="background:rgba(139,92,246,.06);border:1px solid rgba(139,92,246,.25);border-radius:10px;padding:12px 16px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--purple);text-transform:uppercase;margin-bottom:6px">Place Bets</div>';
    placePicks.forEach(function(h) {{
      s += '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;border-bottom:1px solid rgba(139,92,246,.1)">'
        + '<span style="font-weight:700;font-size:.85rem">#' + esc(h.number) + ' ' + esc(h.name) + '</span>'
        + '<span style="font-size:.78rem;color:var(--dim)">' + esc(h.jockey) + ' | $' + esc(h.win_odds) + '</span></div>';
    }});
    s += '<div style="font-size:.72rem;color:var(--dim);margin-top:6px">Each to finish in top 3</div></div>';
  }}

  s += '</div>';

  // Best value picks — odds are better than they should be
  if (valueBets.length) {{
    s += '<div style="background:rgba(251,191,36,.06);border:1px solid rgba(251,191,36,.2);border-radius:8px;padding:10px 14px;margin-bottom:10px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--gold);margin-bottom:6px">BEST VALUE — Odds are in our favour</div>';
    valueBets.forEach(function(h) {{
      s += '<div style="display:flex;justify-content:space-between;align-items:center;padding:3px 0;font-size:.8rem">'
        + '<span><span style="font-weight:700">#' + esc(h.number) + '</span> ' + esc(h.name) + ' @ $' + esc(h.win_odds) + '</span>'
        + '<span style="color:var(--green);font-weight:700">Paying more than it should</span></div>';
    }});
    s += '</div>';
  }}

  // Ones to watch (high odds but decent rank)
  var dangers = horses.filter(function(h) {{ return h.rank >= 4 && h.rank <= 7 && h.win_odds >= 8 && h.calibrated_prob >= 0.06; }});
  if (dangers.length) {{
    s += '<div style="background:rgba(239,68,68,.05);border:1px solid rgba(239,68,68,.15);border-radius:8px;padding:10px 14px;margin-bottom:10px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--red);margin-bottom:6px">ONES TO WATCH (could upset)</div>';
    dangers.forEach(function(h) {{
      s += '<div style="font-size:.8rem;padding:2px 0"><span style="font-weight:700">#' + esc(h.number) + '</span> ' + esc(h.name) + ' @ $' + esc(h.win_odds) + ' <span style="color:var(--dim)">(' + (h.calibrated_prob * 100).toFixed(1) + '%)</span></div>';
    }});
    s += '</div>';
  }}

  // Wildcard bet — exotic combo including top danger horse
  if (dangers.length && horses.length >= 3) {{
    var topDanger = dangers[0];
    var top2 = horses.slice(0, 2);
    var wcNums = top2.map(function(h) {{ return '#' + esc(h.number); }}).concat(['#' + esc(topDanger.number)]);
    var wcNames = top2.map(function(h) {{ return esc(h.name); }}).concat([esc(topDanger.name)]);
    s += '<div style="background:rgba(251,191,36,.08);border:1px solid rgba(251,191,36,.3);border-radius:8px;padding:10px 14px">'
      + '<div style="font-size:.72rem;font-weight:700;color:var(--gold);margin-bottom:6px">WILDCARD BET (includes outsider)</div>'
      + '<div style="font-size:.9rem;font-weight:800">' + wcNums.join(' + ') + '</div>'
      + '<div style="font-size:.78rem;color:var(--dim);margin-top:2px">' + wcNames.join(', ') + '</div>'
      + '<div style="font-size:.72rem;color:var(--dim);margin-top:4px">TRIO or QIN box — covers upset scenario with #' + esc(topDanger.number) + ' ' + esc(topDanger.name) + ' @ $' + esc(topDanger.win_odds) + '</div>'
      + '</div>';
  }}

  s += '</div>';
  return s;
}}

function buildWhatsAppText(race) {{
  var h = race.horses;
  var active = h.filter(function(x) {{ return x.bet_recommendation !== 'SKIP'; }});
  var winPicks = h.filter(function(x) {{ return x.bet_recommendation === 'WIN'; }});
  var placePicks = h.filter(function(x) {{ return x.bet_recommendation === 'PLACE'; }});
  var qinPicks = h.filter(function(x) {{ return x.bet_recommendation === 'QIN'; }});
  var qnlPicks = h.filter(function(x) {{ return x.bet_recommendation === 'QNL PL'; }});
  var trioPicks = h.filter(function(x) {{ return x.bet_recommendation === 'TRIO'; }});
  var dangers = h.filter(function(x) {{ return x.rank >= 4 && x.rank <= 7 && x.win_odds >= 8 && x.calibrated_prob >= 0.06; }});
  var bankerPicks = h.filter(function(x) {{ return x.bet_recommendation === 'BANKER WIN'; }});
  var valueBets = h.filter(function(x) {{ return x.expected_return > 0.05 && x.bet_recommendation !== 'SKIP'; }});

  // Field analysis
  var top3 = h.slice(0, 3);
  var gap = top3.length >= 2 ? (top3[0].calibrated_prob - top3[1].calibrated_prob) : 0;
  var topShare = top3.reduce(function(s, x) {{ return s + x.calibrated_prob; }}, 0);
  var fieldType = gap > 0.05 ? 'One horse stands out' : gap > 0.02 ? 'Clear top pick' : topShare > 0.4 ? 'Top 3 are close' : 'Wide open race';

  var t = '';

  // ── HEADER ──
  t += '\\ud83c\\udfc7 *R' + race.race_number + ' — ' + race.venue + ' ' + race.time + '*\\n';
  t += race.distance + 'm | ' + (race.class_type || '') + ' | ' + (race.going || '') + ' | ' + h.length + ' runners\\n';
  t += fieldType + '\\n';

  // ── QUICK ACTION (what to bet) ──
  t += '\\n\\ud83c\\udfaf *WHAT TO BET:*\\n';

  if (!active.length) {{
    t += '\\u274c SKIP this race — no strong picks\\n';
  }} else {{
    if (bankerPicks.length) {{
      bankerPicks.forEach(function(x) {{
        t += '\\ud83d\\udca5 *BANKER* \\u2192 #' + x.number + ' ' + x.name + ' WIN @ $' + x.win_odds + '\\n';
        t += '     Confidence: ' + (x.calibrated_prob * 100).toFixed(0) + '% | Strong bet\\n';
      }});
    }}

    if (winPicks.length && !bankerPicks.length) {{
      winPicks.forEach(function(x) {{
        t += '\\u2705 *WIN* \\u2192 #' + x.number + ' ' + x.name + ' @ $' + x.win_odds;
        if (x.expected_return > 0) t += ' (good odds)';
        t += '\\n';
      }});
    }}

    if (qinPicks.length >= 2) {{
      t += '\\ud83d\\udd04 *QUINELLA* \\u2192 ';
      t += qinPicks.map(function(x) {{ return '#' + x.number; }}).join(' \\u00d7 ') + '\\n';
      t += '     ' + qinPicks.map(function(x) {{ return x.name; }}).join(' & ') + '\\n';
      t += '     (1st & 2nd in any order)\\n';
    }}

    if (qnlPicks.length >= 2) {{
      t += '\\ud83d\\udd35 *QNL PLACE* \\u2192 ';
      t += qnlPicks.map(function(x) {{ return '#' + x.number; }}).join(' \\u00d7 ') + '\\n';
      t += '     ' + qnlPicks.map(function(x) {{ return x.name; }}).join(' & ') + '\\n';
      t += '     (Both finish top 3 in any order)\\n';
    }}

    if (trioPicks.length >= 3) {{
      t += '\\ud83d\\udccd *TRIO* \\u2192 ';
      t += trioPicks.map(function(x) {{ return '#' + x.number; }}).join(' \\u00d7 ') + '\\n';
      t += '     ' + trioPicks.map(function(x) {{ return x.name; }}).join(' + ') + '\\n';
      t += '     (1st, 2nd & 3rd in any order)\\n';
    }}

    if (placePicks.length) {{
      placePicks.forEach(function(x) {{
        t += '\\ud83d\\udfe3 *PLACE* \\u2192 #' + x.number + ' ' + x.name + ' @ $' + x.win_odds + '\\n';
        t += '     (Finish top 3)\\n';
      }});
    }}

    if (valueBets.length) {{
      t += '\\n\\ud83d\\udcb0 *BEST VALUE:*\\n';
      valueBets.forEach(function(x) {{
        t += '     #' + x.number + ' ' + x.name + ' @ $' + x.win_odds + ' \\u2192 Odds better than expected\\n';
      }});
    }}
  }}

  // ── TOP PICKS (analysis) ──
  t += '\\n\\ud83d\\udcca *TOP PICKS:*\\n';
  h.slice(0, 4).forEach(function(x) {{
    var medal = x.rank === 1 ? '\\ud83e\\udd47' : x.rank === 2 ? '\\ud83e\\udd48' : x.rank === 3 ? '\\ud83e\\udd49' : '4\\ufe0f\\u20e3';
    var bet = x.bet_recommendation !== 'SKIP' ? ' \\u2192 ' + x.bet_recommendation : '';
    t += medal + ' #' + x.number + ' *' + x.name + '* $' + x.win_odds + ' (' + (x.calibrated_prob * 100).toFixed(1) + '%)' + bet + '\\n';

    // Top 3 reasons, one per line for readability
    var reasons = x.reasoning || [];
    reasons.slice(0, 3).forEach(function(r) {{
      t += '      \\u2022 ' + r + '\\n';
    }});
  }});

  // Wildcard longshot trio suggestion
  if (dangers.length && active.length) {{
    var wd = dangers[0];
    t += '\\n\\ud83c\\udfb2 *LONGSHOT TRIO:* #' + h[0].number + ' \\u00d7 #' + h[1].number + ' \\u00d7 #' + wd.number + ' (' + wd.name + ' @ $' + wd.win_odds + ')\\n';
  }}

  return t;
}}

function copyToWhatsApp(raceNum) {{
  var race = RACES.find(function(r) {{ return r.race_number === raceNum; }});
  if (!race) return;
  var text = buildWhatsAppText(race).replace(/\\n{{3,}}/g, '\\n\\n').trim();
  navigator.clipboard.writeText(text).then(function() {{
    var btn = document.getElementById('wa-btn-' + raceNum);
    if (btn) {{
      btn.textContent = 'Copied!';
      btn.style.background = 'var(--green)';
      btn.style.color = '#000';
      setTimeout(function() {{
        btn.textContent = 'Copy to WhatsApp';
        btn.style.background = '';
        btn.style.color = '';
      }}, 2000);
    }}
  }});
}}

function copyAllToWhatsApp() {{
  var text = '\\ud83c\\udfc7 *HKJC RACE DAY — ' + RACE_DATE + '*\\n';
  text += '\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\n\\n';
  RACES.forEach(function(race) {{
    text += buildWhatsAppText(race) + '\\n\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\u2500\\n\\n';
  }});
  text += '_\\u26a0\\ufe0f For entertainment only. Gamble responsibly._\\n';
  text += '_HKJC Analyzer v5.3 | 24 Factors + 23-Feature ML_';
  text = text.replace(/\\n{{3,}}/g, '\\n\\n').trim();
  navigator.clipboard.writeText(text).then(function() {{
    var btn = document.getElementById('wa-btn-all');
    if (btn) {{
      btn.textContent = 'Copied All!';
      btn.style.background = 'var(--green)';
      btn.style.color = '#000';
      setTimeout(function() {{
        btn.textContent = 'Copy All Races';
        btn.style.background = '';
        btn.style.color = '';
      }}, 2000);
    }}
  }});
}}

function renderSpeedMap(speedMap) {{
  if (!speedMap || speedMap.length === 0) return '';
  var zones = {{leaders: [], stalkers: [], midfield: [], closers: []}};
  speedMap.forEach(function(h) {{
    var pos = h.predicted_pos;
    var zone = pos <= Math.ceil(speedMap.length * 0.25) ? 'leaders' :
               pos <= Math.ceil(speedMap.length * 0.5) ? 'stalkers' :
               pos <= Math.ceil(speedMap.length * 0.75) ? 'midfield' : 'closers';
    zones[zone].push(h);
  }});
  var html = '<div class="speed-map"><div class="speed-map-title">Predicted Speed Map</div>';
  html += '<div class="speed-map-track">';
  ['leaders', 'stalkers', 'midfield', 'closers'].forEach(function(zone) {{
    html += '<div class="speed-map-zone ' + zone + '"><span>' + zone.charAt(0).toUpperCase() + zone.slice(1) + '</span>';
    zones[zone].forEach(function(h) {{
      html += '<span class="speed-map-horse ' + esc(h.lane || '') + '" title="' + esc(h.name) + ' (Draw ' + h.draw + ')">' + h.number + '</span>';
    }});
    html += '</div>';
  }});
  html += '</div></div>';
  return html;
}}

function renderExoticPanel(race) {{
  var ex = race.exotic_value;
  if (!ex) return '';
  var hasAny = (ex.exacta && ex.exacta.length) || (ex.trifecta && ex.trifecta.length) || (ex.quinella && ex.quinella.length) || (ex.first4 && ex.first4.length);
  if (!hasAny) return '';
  var nameMap = {{}};
  race.horses.forEach(function(h) {{ nameMap[h.number] = h.name; }});
  var html = '<div style="background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px;margin-bottom:16px">';
  html += '<div style="font-size:1rem;font-weight:800;margin-bottom:12px;color:var(--gold)">EXOTIC VALUE PICKS</div>';
  html += '<div style="font-size:.7rem;color:var(--dim);margin-bottom:10px">Combinations where our model finds 10%+ edge vs market pricing</div>';
  html += '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px">';
  ['exacta', 'quinella', 'trifecta', 'first4'].forEach(function(type) {{
    var bets = ex[type];
    if (!bets || !bets.length) return;
    var label = type.toUpperCase();
    html += '<div style="background:var(--surface2);border-radius:8px;padding:12px">';
    html += '<div style="font-size:.82rem;font-weight:700;color:var(--accent);margin-bottom:8px">' + label + '</div>';
    bets.forEach(function(b) {{
      var horses = b.horses.map(function(n) {{ return '#' + n + ' ' + esc(nameMap[n] || ''); }}).join(' \\u2192 ');
      var edgeColor = b.edge >= 0.25 ? 'var(--green)' : 'var(--accent)';
      html += '<div style="padding:4px 0;border-bottom:1px solid rgba(45,55,72,.3);font-size:.75rem">';
      html += '<div style="font-weight:600">' + horses + '</div>';
      html += '<div style="display:flex;gap:12px;color:var(--dim);margin-top:2px">';
      html += '<span>Prob: ' + (b.probability * 100).toFixed(2) + '%</span>';
      html += '<span>Min odds: $' + b.min_odds.toFixed(0) + '</span>';
      html += '<span style="color:' + edgeColor + ';font-weight:700">+' + (b.edge * 100).toFixed(0) + '% edge</span>';
      html += '</div></div>';
    }});
    html += '</div>';
  }});
  html += '</div></div>';
  return html;
}}

function renderRaceDetail(race) {{
  const el = document.getElementById('race-detail-view');
  const g = gradeRace(race);
  const gc = gradeClass(g);
  const top = race.horses[0];

  const topBet = top ? top.bet_recommendation : 'N/A';
  const topCal = top ? (top.calibrated_prob * 100).toFixed(1) : '0';
  const topPlace = top && top.place_prob ? (top.place_prob * 100).toFixed(1) : '';

  let html = '<div class="detail-header">' +
    '<div><div class="detail-title">Race ' + race.race_number + '</div><div class="detail-meta">' + esc(race.time) + ' | ' + esc(race.distance) + 'm | ' + esc(race.class_type) + ' | ' + esc(race.going) + '</div></div>' +
    '<div style="display:flex;gap:8px;align-items:center">' +
    '<button id="wa-btn-' + race.race_number + '" onclick="copyToWhatsApp(' + race.race_number + ')" style="background:linear-gradient(135deg,#25d366,#128c7e);color:#fff;border:none;padding:8px 16px;border-radius:8px;font-size:.82rem;font-weight:700;cursor:pointer;display:flex;align-items:center;gap:6px;transition:all .2s"><svg width=\"16\" height=\"16\" viewBox=\"0 0 24 24\" fill=\"white\"><path d=\"M17.472 14.382c-.297-.149-1.758-.867-2.03-.967-.273-.099-.471-.148-.67.15-.197.297-.767.966-.94 1.164-.173.199-.347.223-.644.075-.297-.15-1.255-.463-2.39-1.475-.883-.788-1.48-1.761-1.653-2.059-.173-.297-.018-.458.13-.606.134-.133.298-.347.446-.52.149-.174.198-.298.298-.497.099-.198.05-.371-.025-.52-.075-.149-.669-1.612-.916-2.207-.242-.579-.487-.5-.669-.51-.173-.008-.371-.01-.57-.01-.198 0-.52.074-.792.372-.272.297-1.04 1.016-1.04 2.479 0 1.462 1.065 2.875 1.213 3.074.149.198 2.096 3.2 5.077 4.487.709.306 1.262.489 1.694.625.712.227 1.36.195 1.871.118.571-.085 1.758-.719 2.006-1.413.248-.694.248-1.289.173-1.413-.074-.124-.272-.198-.57-.347z\"/><path d=\"M12 0C5.373 0 0 5.373 0 12c0 2.625.846 5.059 2.284 7.034L.789 23.492l4.608-1.467A11.945 11.945 0 0012 24c6.627 0 12-5.373 12-12S18.627 0 12 0zm0 22c-2.188 0-4.214-.696-5.869-1.879l-.421-.314-2.732.87.907-2.677-.346-.449A9.95 9.95 0 012 12C2 6.486 6.486 2 12 2s10 4.486 10 10-4.486 10-10 10z\"/></svg>Copy to WhatsApp</button>' +
    '<button class="back-btn" onclick="showDashboard()">Back to All Races</button>' +
    '</div>' +
  '</div>';
  
  html += '<div class="detail-stats">' +
    '<div class="stat"><div class="v">' + race.horses.length + '</div><div class="l">Runners</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--gold)">' + topBet + '</div><div class="l">Top Pick</div></div>' +
    '<div class="stat"><div class="v" style="color:var(--accent)">' + topCal + '%</div><div class="l">Win Prob</div></div>' +
    (topPlace ? '<div class="stat"><div class="v" style="color:var(--cyan)">' + topPlace + '%</div><div class="l">Place Prob</div></div>' : '') +
  '</div>';

  // Result banner in detail view
  var dHasResults = race.results_available || false;
  var dWinner = dHasResults ? race.actual_winner : null;
  var dWinnerHorse = dWinner ? race.horses.find(function(h) {{ return h.number === dWinner; }}) : null;
  var dWinOdds = race.actual_win_odds || 0;
  var dAccuracy = race.accuracy || {{}};
  var dWinCorrect = dAccuracy.win_correct || false;
  var dTop3 = dAccuracy.top3_overlap || 0;
  var dWinnerRank = dAccuracy.winner_our_rank || '?';

  if (dHasResults && dWinner) {{
    var dBg = dWinCorrect ? 'rgba(16,185,129,.2)' : (dWinnerRank <= 3 ? 'rgba(245,158,11,.15)' : 'rgba(239,68,68,.12)');
    var dBorder = dWinCorrect ? 'var(--green)' : (dWinnerRank <= 3 ? 'var(--accent)' : '#ef4444');
    var dIcon = dWinCorrect ? '\\u2705' : (dWinnerRank <= 3 ? '\\ud83d\\udfe1' : '\\u274c');
    var dLabel = dWinCorrect ? 'OUR #1 WON!' : (dWinnerRank <= 3 ? 'Winner was our Rank ' + dWinnerRank : 'Winner was Rank ' + dWinnerRank);
    // Show top 3 results
    var top3Results = (race.actual_top3 || []).map(function(num) {{
      var rh = race.horses.find(function(h) {{ return h.number === num; }});
      return rh ? '#' + esc(num) + ' ' + esc(rh.name) : '#' + esc(num);
    }});
    html += '<div style="background:' + dBg + ';border-left:4px solid ' + dBorder + ';padding:12px 16px;border-radius:8px;margin-bottom:16px">' +
      '<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">' +
        '<div>' +
          '<div style="font-size:.9rem;font-weight:700">' + dIcon + ' RESULT: #' + esc(dWinner) + ' ' + esc(dWinnerHorse ? dWinnerHorse.name : '') + ' WON @ $' + esc(dWinOdds) + '</div>' +
          '<div style="font-size:.75rem;color:var(--dim);margin-top:4px">Top 3: ' + top3Results.join(' | ') + '</div>' +
        '</div>' +
        '<div style="text-align:right">' +
          '<div style="font-size:.78rem;font-weight:600;color:' + dBorder + '">' + dLabel + '</div>' +
          '<div style="font-size:.7rem;color:var(--dim)">Top 3 overlap: ' + dTop3 + '/3</div>' +
        '</div>' +
      '</div>' +
    '</div>';
  }}

  html += buildStrategyPanel(race);

  // Exotic value panel
  html += renderExoticPanel(race);

  // Speed map
  html += renderSpeedMap(race.speed_map);

  var hasAnyResults = dHasResults && dWinner;
  html += '<table class="horse-table"><thead><tr><th>#</th><th>Horse</th><th>Jockey</th><th>Draw</th><th>Odds</th>' + (hasAnyResults ? '<th>Fin</th>' : '') + '<th>Win%</th><th>Place%</th><th>Bet</th><th>Reasoning</th></tr></thead><tbody>';

  race.horses.forEach((h, i) => {{
    const isTop = i === 0;
    const reasoning = h.reasoning ? h.reasoning.map(function(r){{return esc(r);}}).join('<br>') : '';
    const isWinner = hasAnyResults && h.number === dWinner;
    const actualPos = h.actual_position || '';
    const finBg = actualPos === 1 ? 'background:var(--green);color:#000' : actualPos <= 3 ? 'background:var(--accent);color:#000' : 'background:var(--surface);color:var(--dim)';
    const finCell = hasAnyResults ? '<td>' + (actualPos ? '<span style="display:inline-block;padding:2px 8px;border-radius:4px;font-weight:700;font-size:.75rem;' + finBg + '">' + actualPos + '</span>' : '-') + '</td>' : '';
    html += '<tr class="' + (isTop?'top-row':'') + '"' + (isWinner ? ' style="background:rgba(16,185,129,.1)"' : '') + '>' +
      '<td><div class="rc-rank">' + h.rank + '</div></td>' +
      '<td><div class="rc-name">#' + esc(h.number) + ' ' + esc(h.name) + (isWinner ? ' <span style="font-size:.6rem;background:var(--green);color:#000;padding:1px 5px;border-radius:3px;font-weight:700">WINNER</span>' : '') + '</div><div style="font-size:.7rem;color:var(--dim)">' + esc(h.trainer) + '</div></td>' +
      '<td>' + esc(h.jockey) + '</td>' +
      '<td>' + h.draw + '</td>' +
      '<td style="color:var(--cyan)">$' + h.win_odds + '</td>' +
      finCell +
      '<td style="color:' + probColor(h.calibrated_prob) + '">' + (h.calibrated_prob * 100).toFixed(1) + '%</td>' +
      '<td style="color:var(--cyan)">' + (h.place_prob ? (h.place_prob * 100).toFixed(1) + '%' : '-') + '</td>' +
      '<td><div class="bt" style="background:' + betBg(h.bet_recommendation) + ';color:' + betColor(h.bet_recommendation) + '">' + h.bet_recommendation + '</div></td>' +
      '<td class="reasoning-cell">' + reasoning +
      (function() {{
        var enh = '';
        // Market badges
        if (h.steam_move && h.steam_move.is_steam) {{
          enh += '<span class="market-badge steam">\\u25B2 ' + esc(h.steam_move.label) + '</span> ';
        }}
        if (h.cold_horse && h.cold_horse.is_cold) {{
          enh += '<span class="market-badge cold">\\u2605 COLD</span> ';
        }}
        if (h.odds_movement && h.odds_movement.change_pct && h.odds_movement.change_pct < -15) {{
          enh += '<span class="market-badge drift">\\u25BC DRIFT ' + Math.abs(h.odds_movement.change_pct).toFixed(0) + '%</span> ';
        }}
        // Enhanced stats
        var stats = '';
        var gf = h.going_form;
        if (gf && gf.runs > 0) {{
          stats += '<div class="es-item"><span class="es-label">Going Form</span><span class="es-value">' + gf.wins + 'W/' + gf.places + 'P/' + gf.runs + 'R</span></div>';
        }}
        var jh = h.jockey_horse_stats;
        if (jh && jh.rides > 0) {{
          stats += '<div class="es-item"><span class="es-label">Jockey-Horse</span><span class="es-value">' + jh.wins + 'W ' + jh.places + 'P / ' + jh.rides + 'R</span></div>';
        }}
        if (h.distance_pb) {{
          var pbText = 'PB ' + esc(h.distance_pb);
          if (h.distance_record) pbText += ' (rec ' + esc(h.distance_record) + ')';
          stats += '<div class="es-item"><span class="es-label">Distance Best</span><span class="es-value">' + pbText + '</span></div>';
        }}
        if (stats) enh += '<div class="enhanced-stats">' + stats + '</div>';
        return enh;
      }})() +
      '</td>' +
    '</tr>';
  }});
  
  html += '</tbody></table>';
  
  html += '<div class="nav-arrows">' +
    '<button class="nav-btn" onclick="showRace(' + (race.race_number-1) + ')">Previous</button>' +
    '<button class="nav-btn" onclick="showRace(' + (race.race_number+1) + ')">Next</button>' +
  '</div>';
  
  html += '<div class="disclaimer"><strong>DISCLAIMER:</strong> For entertainment only. Not financial advice. Gamble responsibly.</div>';
  el.innerHTML = html;
}}

initTabs();
renderDashboard();
</script>
</body>
</html>'''
    return html

