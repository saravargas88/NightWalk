"""
Day-night image pairing script.
1. Loads the CSV and splits into day/night by hour from timestamp
2. Matches each night image to the nearest daytime image by lat/lon
3. Copies matched day and night images into local folders
4. Generates a side-by-side viewer HTML
"""
import csv
import json
import shutil
import numpy as np
from pathlib import Path
from scipy.spatial import cKDTree
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/mariasilva/data/urban-mosaic/washington-square")
CSV_PATH = Path("/Users/mariasilva/data/urban-mosaic/washington-square.csv")

NIGHT_DIR = Path("evening")
DAY_DIR   = Path("day_matched")
VIEWER    = Path("pairs_viewer.html")

NIGHT_HOUR_START = 19   # images from this hour onwards = night

# ── Load and split ────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df["hour"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.tz_convert("America/New_York").dt.hour

night = df[df["hour"] >= NIGHT_HOUR_START].copy().reset_index(drop=True)
day   = df[
    (df["hour"] < NIGHT_HOUR_START) &
    (~df["image"].isin(night["image"]))   # exclude any image that also appears in night
].copy().reset_index(drop=True)

print(f"Day pool:   {len(day)}")
print(f"Night pool: {len(night)}")
print(f"Overlap:    {len(set(day['image']) & set(night['image']))}")  # should be 0

# ── Match by lat/lon ──────────────────────────────────────────────────────────
day_coords   = day[["lat", "lon"]].values
night_coords = night[["lat", "lon"]].values

tree = cKDTree(day_coords)
distances, indices = tree.query(night_coords, k=1)

dist_m = distances * 111000
print(f"\nMedian match distance: {np.median(dist_m):.1f} m")
print(f"Max match distance:    {np.max(dist_m):.1f} m")
print(f"Matches within 10m:    {(dist_m < 10).sum()} / {len(dist_m)}")
print(f"Matches within 50m:    {(dist_m < 50).sum()} / {len(dist_m)}")

night["matched_day_image"] = day.iloc[indices]["image"].values
night["dist_m"] = dist_m

# ── Copy images ───────────────────────────────────────────────────────────────
NIGHT_DIR.mkdir(exist_ok=True)
DAY_DIR.mkdir(exist_ok=True)

night["night_local"] = ""
night["day_local"]   = ""

night_copied = 0
day_copied   = 0
missing      = 0

for idx, row in night.iterrows():
    # Night image
    night_src = DATA_DIR / row["image"]
    if not night_src.exists():
        missing += 1
        continue
    night_dst = NIGHT_DIR / night_src.name
    shutil.copy2(night_src, night_dst)
    night.at[idx, "night_local"] = night_src.name
    night_copied += 1

    # Day image — use night stem as prefix to avoid collisions
    day_src  = DATA_DIR / row["matched_day_image"]
    if not day_src.exists():
        missing += 1
        continue
    day_dst_name = f"{night_src.stem}__day{day_src.suffix}"
    shutil.copy2(day_src, DAY_DIR / day_dst_name)
    night.at[idx, "day_local"] = day_dst_name
    day_copied += 1

print(f"\nCopied {night_copied} night images to '{NIGHT_DIR}/'")
print(f"Copied {day_copied} day images to '{DAY_DIR}/'")
if missing:
    print(f"Missing files skipped: {missing}")

# ── Generate viewer ───────────────────────────────────────────────────────────
pairs = night[night["day_local"] != ""][["night_local", "day_local", "lat", "lon", "dist_m"]]

rows_json = json.dumps([
    {
        "night": f"evening/{row['night_local']}",
        "day":   f"day_matched/{row['day_local']}",
        "lat":   round(row["lat"], 6),
        "lon":   round(row["lon"], 6),
        "dist":  round(row["dist_m"], 1),
    }
    for _, row in pairs.iterrows()
])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Day-Night Pairs</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; padding: 16px; }}
  h2 {{ margin-bottom: 4px; }}
  p  {{ color: #aaa; font-size: 13px; margin-bottom: 16px; }}
  .controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }}
  .controls input {{ background: #222; border: 1px solid #444; color: #eee; padding: 6px 10px; border-radius: 6px; font-size: 13px; width: 200px; }}
  .count {{ font-size: 13px; color: #888; }}
  .grid {{ display: flex; flex-direction: column; gap: 12px; }}
  .pair {{ background: #1e1e1e; border-radius: 8px; overflow: hidden; border: 1px solid #333; }}
  .pair-imgs {{ display: grid; grid-template-columns: 1fr 1fr; }}
  .half {{ position: relative; }}
  .half img {{ width: 100%; display: block; height: 200px; object-fit: cover; }}
  .half-label {{ position: absolute; top: 6px; left: 6px; background: rgba(0,0,0,0.65);
                 font-size: 11px; padding: 3px 8px; border-radius: 4px; font-weight: bold; }}
  .night-label {{ color: #7ec8e3; }}
  .day-label   {{ color: #f0c060; }}
  .meta {{ font-size: 11px; color: #666; padding: 6px 10px; }}
</style>
</head>
<body>
<h2>Day–night pairs</h2>
<p>Each row shows the matched daytime and nighttime image from the same location.</p>
<div class="controls">
  <input type="text" id="search" placeholder="Filter by filename…" oninput="render()">
  <span class="count" id="count"></span>
</div>
<div class="grid" id="grid"></div>
<script>
const data = {rows_json};

function render() {{
  const q = document.getElementById('search').value.toLowerCase();
  const filtered = q ? data.filter(r => r.night.toLowerCase().includes(q) || r.day.toLowerCase().includes(q)) : data;
  document.getElementById('count').textContent = filtered.length + ' pairs';
  document.getElementById('grid').innerHTML = filtered.map(r => `
    <div class="pair">
      <div class="pair-imgs">
        <div class="half">
          <img src="${{r.night}}" loading="lazy" onerror="this.style.background='#333';this.style.height='200px'">
          <div class="half-label night-label">night</div>
        </div>
        <div class="half">
          <img src="${{r.day}}" loading="lazy" onerror="this.style.background='#333';this.style.height='200px'">
          <div class="half-label day-label">day</div>
        </div>
      </div>
      <div class="meta">lat ${{r.lat}} &nbsp; lon ${{r.lon}} &nbsp;·&nbsp; ${{r.dist.toFixed(1)}}m apart</div>
    </div>
  `).join('');
}}

render();
</script>
</body>
</html>"""

VIEWER.write_text(html)
print(f"\nSaved {VIEWER} — open in browser from this directory")