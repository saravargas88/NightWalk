"""review_tool.py
Generates an interactive HTML review tool for manually correcting machine labels.

Pre-filters to images where the machine label disagrees with what the sun says:
  - Uses approximate NYC sunrise/sunset times by month
  - Machine=night but sun was up  → likely mislabeled
  - Machine=day  but sun was down → likely mislabeled

Open the output HTML in a browser, click Night / Day / Skip on each card,
then hit "Download CSV" to save your corrections.
"""
import json
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import DATA_DIR

# ── Config ────────────────────────────────────────────────────────────────────
CSV_IN     = Path("all_brightness.csv")
VIEWER_OUT = Path("review_tool.html")

# Approximate NYC civil twilight hours by month (local time, conservative)
NYC_SUNSET  = {1:17, 2:17, 3:18, 4:20, 5:20, 6:21, 7:21, 8:20, 9:19, 10:18, 11:17, 12:16}
NYC_SUNRISE = {1:7,  2:7,  3:6,  4:6,  5:5,  6:5,  7:5,  8:6,  9:6,  10:7,  11:6,  12:7}

# ── Load & compute expected labels ────────────────────────────────────────────
df = pd.read_csv(CSV_IN, parse_dates=["taken_on_short"])
df["month"] = df["taken_on_short"].dt.month
df["date"]  = df["taken_on_short"].dt.strftime("%-m/%-d/%Y")

df["machine_night"]  = df["period"].isin(["evening", "night"])
df["expected_night"] = df.apply(
    lambda r: r["hour"] >= NYC_SUNSET[r["month"]] or r["hour"] < NYC_SUNRISE[r["month"]],
    axis=1
)
df["mismatch"] = df["expected_night"] != df["machine_night"]

suspect = df[df["mismatch"]].copy()
suspect["reason"] = suspect.apply(
    lambda r: "machine=night, sun=up" if r["machine_night"] else "machine=day, sun=down",
    axis=1
)
suspect = suspect.sort_values(["hour", "grey"]).reset_index(drop=True)

print(f"Total images:   {len(df):,}")
print(f"Suspect images: {len(suspect):,}")
print(f"\nBreakdown:")
print(f"  Machine=night, sun=up:   {(suspect['reason'] == 'machine=night, sun=up').sum():,}")
print(f"  Machine=day,  sun=down:  {(suspect['reason'] == 'machine=day, sun=down').sum():,}")
print(f"\nMonth distribution:")
print(suspect["month"].value_counts().sort_index().to_string())

# ── Build JSON for viewer ─────────────────────────────────────────────────────
MONTH_NAMES = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

rows = []
for _, r in suspect.iterrows():
    rows.append({
        "image":   f"../urban-mosaic/washington-square/{r['image']}",
        "fname":   r["image"],
        "grey":    round(r["grey"], 1),
        "hour":    int(r["hour"]),
        "month":   MONTH_NAMES[int(r["month"])],
        "date":    r["date"],
        "period":  r["period"],
        "reason":  r["reason"],
    })

rows_json = json.dumps(rows)

# ── Generate HTML ─────────────────────────────────────────────────────────────
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Manual Review Tool — {len(rows):,} suspect images</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h2 {{ margin: 0 0 4px; }}
  p.sub {{ color: #888; font-size: 13px; margin: 0 0 14px; }}

  /* ── controls ── */
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 16px; }}
  .toggle {{ display: flex; background: #1e1e1e; border-radius: 6px; overflow: hidden; border: 1px solid #333; }}
  .toggle button {{ background: none; border: none; color: #888; padding: 7px 13px; cursor: pointer; font-size: 12px; white-space: nowrap; }}
  .toggle button.active {{ background: #444; color: #fff; }}
  select, input[type=range] {{ background: #1e1e1e; border: 1px solid #333; color: #eee; padding: 6px 10px; border-radius: 6px; font-size: 13px; }}
  label {{ font-size: 13px; color: #888; }}
  .stat {{ font-size: 13px; color: #666; }}
  .stat span {{ color: #fff; font-weight: bold; }}

  /* ── grid ── */
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ width: 210px; background: #1e1e1e; border-radius: 8px; overflow: hidden; border: 2px solid transparent; transition: border-color 0.15s; }}
  .card.labeled-night {{ border-color: #3b6ea5; }}
  .card.labeled-day   {{ border-color: #5a9e5a; }}
  .card img {{ width: 100%; height: 140px; object-fit: cover; display: block; cursor: pointer; }}
  .card img:hover {{ opacity: 0.85; }}

  .meta {{ padding: 5px 8px 4px; font-size: 11px; display: flex; flex-wrap: wrap; gap: 3px 8px; color: #888; }}
  .meta .grey {{ color: #fff; font-weight: bold; font-size: 13px; }}
  .badge {{ font-size: 10px; padding: 2px 6px; border-radius: 99px; background: #2a2a2a; color: #aaa; border: 1px solid #333; }}
  .badge.reason-night {{ border-color: #7a3030; color: #c96060; }}
  .badge.reason-day   {{ border-color: #5a7a30; color: #8ab050; }}

  .btns {{ display: flex; border-top: 1px solid #2a2a2a; }}
  .btns button {{ flex: 1; border: none; background: none; color: #666; padding: 8px 4px; font-size: 12px; cursor: pointer; transition: background 0.1s, color 0.1s; }}
  .btns button:hover {{ background: #2a2a2a; }}
  .btns .btn-night {{ border-right: 1px solid #2a2a2a; }}
  .btns .btn-day   {{ border-right: 1px solid #2a2a2a; }}
  .card.labeled-night .btn-night {{ background: #1a3a5a; color: #7ab0e0; font-weight: bold; }}
  .card.labeled-day   .btn-day   {{ background: #1a3a1a; color: #7ac07a; font-weight: bold; }}
  .card.labeled-skip  .btns button.btn-skip {{ background: #2a2a1a; color: #a0a060; font-weight: bold; }}

  /* ── bulk + download ── */
  .bulk-wrap {{ display: flex; gap: 8px; align-items: center; margin-bottom: 12px; }}
  .bulk-btn {{ padding: 8px 18px; border-radius: 6px; font-size: 13px; cursor: pointer; border: 1px solid; }}
  .bulk-day   {{ background: #1a3a1a; border-color: #3a6a3a; color: #7ac07a; }}
  .bulk-day:hover {{ background: #2a4a2a; }}
  .bulk-night {{ background: #1a2a3a; border-color: #2a4a6a; color: #7ab0e0; }}
  .bulk-night:hover {{ background: #1a3a5a; }}
  .bulk-clear {{ background: #2a2a2a; border-color: #444; color: #888; }}
  .bulk-clear:hover {{ background: #333; }}
  .dl-btn {{ padding: 10px 24px; background: #2a4a2a; border: 1px solid #3a6a3a; border-radius: 6px; color: #7ac07a; font-size: 14px; cursor: pointer; }}
  .dl-btn:hover {{ background: #3a5a3a; }}

  #progress {{ font-size: 13px; color: #888; margin-top: 8px; }}
  #progress span {{ color: #fff; }}
</style>
</head>
<body>
<h2>Manual Review — {len(rows):,} suspect images</h2>
<p class="sub">Images flagged for twilight hours or brightness contradicting the machine label. Click Night / Day / Skip, then download your corrections.</p>

<div class="controls">
  <!-- filter by reason -->
  <div class="toggle">
    <button id="f-all"    class="active" onclick="setFilter('all')">All</button>
    <button id="f-night"               onclick="setFilter('machine=night, sun=up')">Machine=Night, Sun=Up</button>
    <button id="f-day"                 onclick="setFilter('machine=day, sun=down')">Machine=Day, Sun=Down</button>
  </div>
  <!-- filter by review status -->
  <div class="toggle">
    <button id="s-all"    class="active" onclick="setStatus('all')">All</button>
    <button id="s-unseen"               onclick="setStatus('unseen')">Unseen</button>
    <button id="s-done"                 onclick="setStatus('done')">Reviewed</button>
  </div>
  <!-- sort -->
  <label>Sort:
    <select onchange="setSort(this.value)">
      <option value="hour">Hour</option>
      <option value="grey-asc">Brightness ↑</option>
      <option value="grey-desc">Brightness ↓</option>
      <option value="month">Month</option>
    </select>
  </label>
  <!-- brightness cutoff slider -->
  <label>Night cutoff: <input type="range" min="0" max="255" value="106" step="1" id="thresh" oninput="document.getElementById('tv').textContent=this.value; render()"> <span id="tv">106</span></label>
</div>

<div class="stat">
  Reviewed <span id="ct-done">0</span> / <span id="ct-total">{len(rows)}</span> &nbsp;|&nbsp;
  Night: <span id="ct-night">0</span> &nbsp; Day: <span id="ct-day">0</span> &nbsp; Skip: <span id="ct-skip">0</span>
</div>

<div class="bulk-wrap">
  <button class="bulk-btn bulk-day"   onclick="markAllVisible('day')">☀️ Mark all visible as Day</button>
  <button class="bulk-btn bulk-night" onclick="markAllVisible('night')">🌙 Mark all visible as Night</button>
  <button class="bulk-btn bulk-clear" onclick="markAllVisible(null)">✕ Clear all visible</button>
</div>

<div class="grid" id="grid"></div>

<div>
  <button class="dl-btn" onclick="downloadCSV()">Download corrections CSV</button>
  <div id="progress"></div>
</div>

<script>
const ALL_DATA = {rows_json};

// label state per image: null | 'night' | 'day' | 'skip'
const labels = {{}};

let filterReason = 'all';
let filterStatus = 'all';
let sortMode     = 'hour';

function setFilter(r) {{
  filterReason = r;
  ['all','night','day'].forEach(k => document.getElementById('f-'+k).classList.remove('active'));
  const keyMap = {{'all':'all','machine=night, sun=up':'night','machine=day, sun=down':'day'}};
  document.getElementById('f-' + (keyMap[r] || r)).classList.add('active');
  render();
}}

function setStatus(s) {{
  filterStatus = s;
  ['all','unseen','done'].forEach(k => document.getElementById('s-'+k).classList.remove('active'));
  document.getElementById('s-'+s).classList.add('active');
  render();
}}

function setSort(v) {{ sortMode = v; render(); }}

function label(fname, val) {{
  labels[fname] = labels[fname] === val ? null : val;  // toggle
  updateStats();
  // re-render just that card's classes + buttons
  const card = document.getElementById('card-' + CSS.escape(fname));
  if (card) applyLabel(card, fname);
}}

function markAllVisible(val) {{
  // Apply label to everything currently shown in the grid
  const cards = document.querySelectorAll('#grid .card');
  cards.forEach(card => {{
    const fname = card.dataset.fname;
    labels[fname] = val;
    applyLabel(card, fname);
  }});
  updateStats();
}}

function applyLabel(card, fname) {{
  card.classList.remove('labeled-night','labeled-day','labeled-skip');
  const v = labels[fname];
  if (v === 'night') card.classList.add('labeled-night');
  if (v === 'day')   card.classList.add('labeled-day');
  if (v === 'skip')  card.classList.add('labeled-skip');
}}

function updateStats() {{
  const vals = Object.values(labels);
  document.getElementById('ct-done').textContent  = vals.filter(v => v).length;
  document.getElementById('ct-night').textContent = vals.filter(v => v === 'night').length;
  document.getElementById('ct-day').textContent   = vals.filter(v => v === 'day').length;
  document.getElementById('ct-skip').textContent  = vals.filter(v => v === 'skip').length;
}}

function reasonClass(r) {{
  if (r === 'machine=night, sun=up')   return 'reason-night';
  if (r === 'machine=day, sun=down')   return 'reason-day';
  return '';
}}

function render() {{
  const thresh = +document.getElementById('thresh').value;

  let data = [...ALL_DATA];

  // filter
  if (filterReason !== 'all') data = data.filter(r => r.reason === filterReason);
  if (filterStatus === 'unseen') data = data.filter(r => !labels[r.fname]);
  if (filterStatus === 'done')   data = data.filter(r =>  labels[r.fname]);

  // sort
  if (sortMode === 'hour')       data.sort((a,b) => a.hour - b.hour || a.grey - b.grey);
  if (sortMode === 'grey-asc')   data.sort((a,b) => a.grey - b.grey);
  if (sortMode === 'grey-desc')  data.sort((a,b) => b.grey - a.grey);
  if (sortMode === 'month')      data.sort((a,b) => a.date.localeCompare(b.date));

  document.getElementById('ct-total').textContent = data.length;

  document.getElementById('grid').innerHTML = data.map(r => {{
    const lbl = labels[r.fname] || '';
    const lcls = lbl === 'night' ? 'labeled-night' : lbl === 'day' ? 'labeled-day' : lbl === 'skip' ? 'labeled-skip' : '';
    const aboveThresh = r.grey > thresh;
    const id = CSS.escape(r.fname);
    return `
      <div class="card ${{lcls}}" id="card-${{id}}" data-fname="${{r.fname}}">
        <img src="${{r.image}}" loading="lazy" title="${{r.fname}}"
             onerror="this.style.background='#2a2a2a';this.removeAttribute('src')">
        <div class="meta">
          <span class="grey">${{r.grey}}</span>
          <span class="badge">${{r.hour}}:00</span>
          <span class="badge">${{r.month}}</span>
          <span class="badge ${{reasonClass(r.reason)}}">${{r.reason}}</span>
          <span class="badge" style="color:#666">${{r.period}}</span>
        </div>
        <div class="btns">
          <button class="btn-night" onclick="label('${{r.fname}}','night')">🌙 Night</button>
          <button class="btn-day"   onclick="label('${{r.fname}}','day')">☀️ Day</button>
          <button class="btn-skip"  onclick="label('${{r.fname}}','skip')">— Skip</button>
        </div>
      </div>`;
  }}).join('');

  updateStats();
}}

function downloadCSV() {{
  const labeled = ALL_DATA.filter(r => labels[r.fname] && labels[r.fname] !== 'skip');
  if (!labeled.length) {{
    document.getElementById('progress').textContent = 'No Night/Day labels to export yet.';
    return;
  }}
  const lines = ['image,grey,hour,month,date,machine_period,manual_label'];
  labeled.forEach(r => {{
    lines.push(`${{r.fname}},${{r.grey}},${{r.hour}},${{r.month}},${{r.date}},${{r.period}},${{labels[r.fname]}}`);
  }});
  const blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'manual_corrections.csv';
  a.click();
  document.getElementById('progress').textContent = `Exported ${{labeled.length}} corrections.`;
}}

render();
</script>
</body>
</html>"""

VIEWER_OUT.write_text(html)
print(f"\nSaved {VIEWER_OUT}")
print(f"Open it from the brightness_experiments/ folder in your browser.")
print(f"Click Night / Day / Skip on each image, then hit 'Download corrections CSV'.")
