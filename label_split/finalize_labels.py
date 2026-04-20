"""finalize_labels.py
Reconciles all_brightness.csv with manual_corrections.csv to produce
a single authoritative labels_final.csv, then generates a final viewer
for one last visual check of the night/day split.

Usage:
    python finalize_labels.py                        # uses defaults below
    python finalize_labels.py --threshold 106        # override brightness cutoff
    python finalize_labels.py --corrections path/to/manual_corrections.csv
"""
import json
import argparse
import pandas as pd
from pathlib import Path

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--brightness-csv",  default="all_brightness.csv")
parser.add_argument("--corrections",     default="manual_corrections.csv")
parser.add_argument("--threshold",       type=float, default=70,
                    help="Greyscale cutoff applied within the twilight buffer (default 70)")
parser.add_argument("--twilight-buffer", type=int, default=45,
                    help="Minutes around sunrise/sunset treated as ambiguous (default 45)")
parser.add_argument("--dark-override",   type=float, default=30,
                    help="Images this dark get called night even in core-day hours (default 30)")
parser.add_argument("--out-csv",         default="labels_final.csv")
parser.add_argument("--out-html",        default="final_viewer.html")
parser.add_argument("--sample-night",    type=int, default=500,
                    help="Max night images shown in the viewer (default 500)")
parser.add_argument("--sample-day",      type=int, default=2000,
                    help="Max day images shown in the viewer, darkest first (default 2000)")
args = parser.parse_args()

BRIGHTNESS_CSV  = Path(args.brightness_csv)
CORRECTIONS_CSV = Path(args.corrections)
THRESHOLD        = args.threshold
TWILIGHT_BUFFER  = args.twilight_buffer   # minutes
DARK_OVERRIDE    = args.dark_override     # brightness below which even core-day → night
OUT_CSV          = Path(args.out_csv)

# Exact per-date sunrise/sunset using astral (Washington Square, NYC)
from astral import LocationInfo
from astral.sun import sun as astral_sun
import pytz

_WS   = LocationInfo("Washington Square", "USA", "America/New_York", 40.7282, -73.9942)
_TZ   = pytz.timezone("America/New_York")
_cache = {}

def get_sun(d):
    """Return (sunrise_decimal_hour, sunset_decimal_hour) for a given date."""
    if d not in _cache:
        s = astral_sun(_WS.observer, date=d, tzinfo=_TZ)
        sr = s["sunrise"].hour + s["sunrise"].minute / 60
        ss = s["sunset"].hour  + s["sunset"].minute  / 60
        _cache[d] = (sr, ss)
    return _cache[d]
OUT_HTML        = Path(args.out_html)
SAMPLE_NIGHT    = args.sample_night
SAMPLE_DAY      = args.sample_day

# ── Load brightness data ──────────────────────────────────────────────────────
df = pd.read_csv(BRIGHTNESS_CSV, parse_dates=["taken_on_short"])
df["month"] = df["taken_on_short"].dt.month
df["date"]  = df["taken_on_short"].dt.strftime("%-m/%-d/%Y")

# ── Per-date three-zone machine label ────────────────────────────────────────
# Uses exact per-date sunrise/sunset (astral) with a configurable twilight buffer:
#   Core day:   time > sunrise+buffer AND < sunset-buffer
#               → day, UNLESS brightness < dark_override (e.g. tunnel/garage)
#   Core night: time < sunrise-buffer OR  > sunset+buffer → always night
#   Twilight:   within buffer of sunrise or sunset        → brightness threshold

buf = TWILIGHT_BUFFER / 60.0

print(f"Loaded {len(df):,} images from {BRIGHTNESS_CSV}")
print(f"Computing exact sunrise/sunset per date (astral)...")

df["decimal_hour"] = df["taken_on_short"].dt.hour + df["taken_on_short"].dt.minute / 60.0
df["date_only"]    = df["taken_on_short"].dt.date
df[["sunrise","sunset"]] = df["date_only"].apply(lambda d: pd.Series(get_sun(d)))

def get_zone(row):
    t, sr, ss = row["decimal_hour"], row["sunrise"], row["sunset"]
    if t > sr + buf and t < ss - buf:
        return "core_day"
    if t < sr - buf or t > ss + buf:
        return "core_night"
    return "twilight"

def machine_label(row):
    zone = row["zone"]
    if zone == "core_night":
        return "night"
    if zone == "twilight":
        return "night" if row["grey"] <= THRESHOLD else "day"
    # core_day: day unless implausibly dark (tunnel, underground, sensor error)
    if row["grey"] <= DARK_OVERRIDE:
        return "night"
    return "day"

df["zone"]          = df.apply(get_zone, axis=1)
df["machine_label"] = df.apply(machine_label, axis=1)

print(f"\nZones (exact per-date sunrise/sunset, ±{TWILIGHT_BUFFER} min buffer):")
print(f"  Core day   → day (override if grey ≤ {DARK_OVERRIDE}): {(df['zone']=='core_day').sum():,}")
print(f"  Core night → always night:                              {(df['zone']=='core_night').sum():,}")
print(f"  Twilight   → brightness threshold {THRESHOLD}:          {(df['zone']=='twilight').sum():,}")
print(f"\nMachine labels:")
print(f"  night={(df['machine_label']=='night').sum():,}  day={(df['machine_label']=='day').sum():,}")

# ── Load manual corrections ───────────────────────────────────────────────────
if CORRECTIONS_CSV.exists():
    corr = pd.read_csv(CORRECTIONS_CSV)[["image", "manual_label"]]
    corr = corr[corr["manual_label"].isin(["night", "day"])]   # drop skips
    print(f"Loaded {len(corr):,} manual corrections from {CORRECTIONS_CSV}")
else:
    corr = pd.DataFrame(columns=["image", "manual_label"])
    print(f"No corrections file found at {CORRECTIONS_CSV} — using machine labels only")

# ── Reconcile ─────────────────────────────────────────────────────────────────
df = df.merge(corr, on="image", how="left")
df["final_label"] = df["manual_label"].combine_first(df["machine_label"])
df["source"] = df["manual_label"].notna().map({True: "manual", False: "machine"})

print(f"\nFinal labels:")
print(f"  night:   {(df['final_label']=='night').sum():,}")
print(f"  day:     {(df['final_label']=='day').sum():,}")
print(f"  manual overrides applied: {(df['source']=='manual').sum():,}")

# ── Save CSV ──────────────────────────────────────────────────────────────────
out_cols = ["image", "grey", "hour", "taken_on_short", "period", "machine_label", "source", "final_label"]
df[out_cols].to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

# ── Build final viewer ────────────────────────────────────────────────────────
MONTH_NAMES = ["","Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def to_row(r, cls):
    return {
        "image":  f"../urban-mosaic/washington-square/{r['image']}",
        "fname":  r["image"],
        "grey":   round(r["grey"], 1),
        "hour":   int(r["hour"]),
        "month":  MONTH_NAMES[int(r["month"])],
        "date":   r["date"],
        "source": r["source"],
        "label":  cls,
    }

night_df = df[df["final_label"] == "night"].sort_values("grey")
day_df   = df[df["final_label"] == "day"].sort_values("grey")   # darkest first — most likely to be wrong

# Sample for viewer
night_rows = [to_row(r, "night") for _, r in night_df.head(SAMPLE_NIGHT).iterrows()]
day_rows   = [to_row(r, "day")   for _, r in day_df.head(SAMPLE_DAY).iterrows()]

night_json = json.dumps(night_rows)
day_json   = json.dumps(day_rows)

n_night_total = (df["final_label"] == "night").sum()
n_day_total   = (df["final_label"] == "day").sum()

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Final label check — Night vs Day</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: sans-serif; background: #111; color: #eee; margin: 0; padding: 16px; }}
  h2 {{ margin: 0 0 4px; }}
  p.sub {{ color: #888; font-size: 13px; margin: 0 0 14px; }}

  /* ── tabs ── */
  .tabs {{ display: flex; gap: 0; margin-bottom: 16px; border-bottom: 2px solid #222; }}
  .tab {{ padding: 10px 28px; cursor: pointer; font-size: 14px; color: #888; border-bottom: 3px solid transparent; margin-bottom: -2px; }}
  .tab.active {{ color: #fff; border-bottom-color: #5599dd; }}

  /* ── controls ── */
  .controls {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 12px; }}
  .toggle {{ display: flex; background: #1e1e1e; border-radius: 6px; overflow: hidden; border: 1px solid #333; }}
  .toggle button {{ background: none; border: none; color: #888; padding: 7px 13px; cursor: pointer; font-size: 12px; }}
  .toggle button.active {{ background: #444; color: #fff; }}
  label {{ font-size: 13px; color: #888; }}
  select {{ background: #1e1e1e; border: 1px solid #333; color: #eee; padding: 6px 10px; border-radius: 6px; font-size: 13px; }}

  /* ── hour picker ── */
  .hour-picker {{ display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 12px; }}
  .hr-btn {{ background: #1e1e1e; border: 1px solid #333; color: #666; border-radius: 5px;
             padding: 4px 0; width: 52px; font-size: 11px; text-align: center; cursor: pointer; line-height: 1.4; }}
  .hr-btn:hover {{ border-color: #555; color: #aaa; }}
  .hr-btn.has-images {{ color: #bbb; border-color: #444; }}
  .hr-btn.active {{ background: #2a4a6a; border-color: #5599dd; color: #fff; font-weight: bold; }}
  .hr-btn .hr-label {{ font-size: 12px; }}
  .hr-btn .hr-count {{ font-size: 10px; color: #555; }}
  .hr-btn.has-images .hr-count {{ color: #888; }}
  .hr-btn.active .hr-count {{ color: #aaa; }}

  /* ── stats bar ── */
  .statbar {{ font-size: 13px; color: #666; margin-bottom: 12px; }}
  .statbar span {{ color: #fff; font-weight: bold; }}
  .statbar .warn {{ color: #e09050; }}

  /* ── grid ── */
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ width: 200px; background: #1e1e1e; border-radius: 8px; overflow: hidden; border: 2px solid transparent; }}
  .card.flagged {{ border-color: #e05050; }}
  .card img {{ width: 100%; height: 133px; object-fit: cover; display: block; }}
  .meta {{ padding: 5px 8px; font-size: 11px; display: flex; flex-wrap: wrap; gap: 3px 8px; color: #888; }}
  .meta .grey {{ color: #fff; font-weight: bold; font-size: 12px; }}
  .badge {{ font-size: 10px; padding: 2px 5px; border-radius: 99px; background: #2a2a2a; color: #aaa; border: 1px solid #333; }}
  .badge.manual {{ border-color: #a07030; color: #c09050; }}
  .flag-btn {{ width: 100%; border: none; background: #1a1a1a; color: #666; padding: 7px; font-size: 12px; cursor: pointer; border-top: 1px solid #2a2a2a; }}
  .flag-btn:hover {{ background: #2a1a1a; color: #e08080; }}
  .card.flagged .flag-btn {{ background: #2a1a1a; color: #e05050; font-weight: bold; }}

  /* ── export ── */
  .dl-btn {{ margin-top: 20px; padding: 10px 24px; background: #2a4a2a; border: 1px solid #3a6a3a; border-radius: 6px; color: #7ac07a; font-size: 14px; cursor: pointer; }}
  .dl-btn:hover {{ background: #3a5a3a; }}
  #export-note {{ font-size: 13px; color: #888; margin-top: 8px; }}

  .panel {{ display: none; }}
  .panel.active {{ display: block; }}
</style>
</head>
<body>
<h2>Final label check</h2>
<p class="sub">
  Night: <strong>{n_night_total:,}</strong> total (showing darkest {min(SAMPLE_NIGHT, n_night_total):,}) &nbsp;|&nbsp;
  Day: <strong>{n_day_total:,}</strong> total (showing darkest {min(SAMPLE_DAY, n_day_total):,} — most likely to be wrong) &nbsp;|&nbsp;
  Manual overrides: <strong>{(df['source']=='manual').sum():,}</strong>
</p>

<div class="tabs">
  <div class="tab active" onclick="switchTab('night', this)">🌙 Night &nbsp;<small style="color:#555">{n_night_total:,}</small></div>
  <div class="tab"        onclick="switchTab('day',   this)">☀️ Day &nbsp;<small style="color:#555">{n_day_total:,}</small></div>
</div>

<!-- Night panel -->
<div class="panel active" id="panel-night">
  <div class="controls">
    <div class="toggle">
      <button class="active" onclick="nightSort('grey-asc')">Darkest first</button>
      <button               onclick="nightSort('grey-desc')">Brightest first</button>
      <button               onclick="nightSort('hour')">By hour</button>
      <button               onclick="nightSort('month')">By month</button>
    </div>
    <label>Show: <select onchange="nightFilter(this.value)">
      <option value="all">All</option>
      <option value="manual">Manual overrides only</option>
    </select></label>
  </div>
  <div class="hour-picker" id="hour-picker-night"></div>
  <div class="statbar">Showing <span id="n-shown">—</span> &nbsp;|&nbsp; Flagged: <span class="warn" id="n-flagged">0</span></div>
  <div class="grid" id="grid-night"></div>
</div>

<!-- Day panel -->
<div class="panel" id="panel-day">
  <div class="controls">
    <div class="toggle">
      <button class="active" onclick="daySort('grey-desc')">Brightest first</button>
      <button               onclick="daySort('grey-asc')">Darkest first</button>
      <button               onclick="daySort('hour')">By hour</button>
      <button               onclick="daySort('month')">By month</button>
    </div>
    <label>Show: <select onchange="dayFilter(this.value)">
      <option value="all">All</option>
      <option value="manual">Manual overrides only</option>
    </select></label>
  </div>
  <div class="hour-picker" id="hour-picker-day"></div>
  <div class="statbar">Showing <span id="d-shown">—</span> &nbsp;|&nbsp; Flagged: <span class="warn" id="d-flagged">0</span></div>
  <div class="grid" id="grid-day"></div>
</div>

<div>
  <button class="dl-btn" onclick="downloadFlags()">Download flagged images CSV</button>
  <div id="export-note"></div>
</div>

<script>
const NIGHT_DATA = {night_json};
const DAY_DATA   = {day_json};

const flagged = {{}};

let nightSortMode   = 'grey-asc';
let daySortMode     = 'grey-desc';
let nightFilterMode = 'all';
let dayFilterMode   = 'all';
let nightHour       = null;   // null = all hours
let dayHour         = null;

// ── Hour picker ───────────────────────────────────────────────────────────────
function buildHourPicker(pickerId, data, activeHour, onClickFn) {{
  // count images per hour across full dataset
  const counts = {{}};
  data.forEach(r => {{ counts[r.hour] = (counts[r.hour] || 0) + 1; }});

  const el = document.getElementById(pickerId);
  let html = `<button class="hr-btn has-images ${{activeHour === null ? 'active' : ''}}"
    onclick="${{onClickFn}}(null)">
    <div class="hr-label">All</div>
    <div class="hr-count">${{data.length}}</div>
  </button>`;

  for (let h = 0; h < 24; h++) {{
    const n = counts[h] || 0;
    const hasCls  = n > 0 ? 'has-images' : '';
    const activeCls = activeHour === h ? 'active' : '';
    html += `<button class="hr-btn ${{hasCls}} ${{activeCls}}" onclick="${{onClickFn}}(${{h}})">
      <div class="hr-label">${{String(h).padStart(2,'0')}}:00</div>
      <div class="hr-count">${{n > 0 ? n : '—'}}</div>
    </button>`;
  }}
  el.innerHTML = html;
}}

function setNightHour(h) {{ nightHour = (nightHour === h ? null : h); renderNight(); }}
function setDayHour(h)   {{ dayHour   = (dayHour   === h ? null : h); renderDay();   }}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(tab, el) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  el.classList.add('active');
  document.getElementById('panel-' + tab).classList.add('active');
}}

// ── Flagging ──────────────────────────────────────────────────────────────────
function flag(fname) {{
  flagged[fname] = !flagged[fname];
  if (!flagged[fname]) delete flagged[fname];
  const card = document.getElementById('c-' + CSS.escape(fname));
  if (card) card.classList.toggle('flagged', !!flagged[fname]);
  updateFlagCounts();
}}

function updateFlagCounts() {{
  const all = Object.keys(flagged);
  document.getElementById('n-flagged').textContent = all.filter(f => NIGHT_DATA.find(r => r.fname === f)).length;
  document.getElementById('d-flagged').textContent = all.filter(f => DAY_DATA.find(r => r.fname === f)).length;
}}

// ── Rendering ─────────────────────────────────────────────────────────────────
function sortData(data, mode) {{
  const d = [...data];
  if (mode === 'grey-asc')  d.sort((a,b) => a.grey - b.grey);
  if (mode === 'grey-desc') d.sort((a,b) => b.grey - a.grey);
  if (mode === 'hour')      d.sort((a,b) => a.hour - b.hour);
  if (mode === 'month')     d.sort((a,b) => a.date.localeCompare(b.date));
  return d;
}}

function renderGrid(containerId, data, sortMode, filterMode, hourFilter, shownId) {{
  let d = sortData(data, sortMode);
  if (filterMode === 'manual') d = d.filter(r => r.source === 'manual');
  if (hourFilter !== null)     d = d.filter(r => r.hour === hourFilter);
  document.getElementById(shownId).textContent = d.length;
  document.getElementById(containerId).innerHTML = d.map(r => `
    <div class="card ${{flagged[r.fname] ? 'flagged' : ''}}" id="c-${{CSS.escape(r.fname)}}">
      <img src="${{r.image}}" loading="lazy"
           onerror="this.style.background='#2a2a2a';this.style.height='133px';this.removeAttribute('src')">
      <div class="meta">
        <span class="grey">${{r.grey}}</span>
        <span class="badge">${{r.hour}}:00</span>
        <span class="badge">${{r.month}}</span>
        ${{r.source === 'manual' ? '<span class="badge manual">manual</span>' : ''}}
      </div>
      <button class="flag-btn" onclick="flag('${{r.fname}}')">${{flagged[r.fname] ? '⚑ Flagged — click to unflag' : '⚐ Flag as wrong'}}</button>
    </div>`).join('');
}}

function nightSort(m)   {{ nightSortMode = m;   renderNight(); }}
function daySort(m)     {{ daySortMode = m;     renderDay(); }}
function nightFilter(m) {{ nightFilterMode = m; renderNight(); }}
function dayFilter(m)   {{ dayFilterMode = m;   renderDay(); }}

function renderNight() {{
  buildHourPicker('hour-picker-night', NIGHT_DATA, nightHour, 'setNightHour');
  renderGrid('grid-night', NIGHT_DATA, nightSortMode, nightFilterMode, nightHour, 'n-shown');
}}
function renderDay() {{
  buildHourPicker('hour-picker-day', DAY_DATA, dayHour, 'setDayHour');
  renderGrid('grid-day', DAY_DATA, daySortMode, dayFilterMode, dayHour, 'd-shown');
}}

// ── Export ────────────────────────────────────────────────────────────────────
function downloadFlags() {{
  const keys = Object.keys(flagged);
  if (!keys.length) {{
    document.getElementById('export-note').textContent = 'Nothing flagged yet.';
    return;
  }}
  const all = [...NIGHT_DATA, ...DAY_DATA];
  const lines = ['image,grey,hour,month,date,label,source'];
  keys.forEach(fname => {{
    const r = all.find(x => x.fname === fname);
    if (r) lines.push(`${{r.fname}},${{r.grey}},${{r.hour}},${{r.month}},${{r.date}},${{r.label}},${{r.source}}`);
  }});
  const blob = new Blob([lines.join('\\n')], {{type:'text/csv'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'flagged_for_review.csv';
  a.click();
  document.getElementById('export-note').textContent = `Exported ${{keys.length}} flagged images.`;
}}

renderNight();
renderDay();
</script>
</body>
</html>"""

OUT_HTML.write_text(html)
print(f"Saved {OUT_HTML}")
print(f"\nWorkflow:")
print(f"  1. Open {OUT_HTML} in your browser")
print(f"  2. Scan Night tab (darkest first) and Day tab (brightest first)")
print(f"  3. Click 'Flag as wrong' on any misclassified images")
print(f"  4. Download flagged CSV — re-run review_tool.py with those if needed")
