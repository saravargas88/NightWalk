"""ground_truth.py
Computing the brightness of each evening image using five methods:
  - grey:     Greyscale mean
  - hsv:      HSV Value channel mean
  - cropped:  Greyscale mean of center crop (excludes sky + road)
  - luminance: Perceptual luminance via YCbCr Y channel
  - dark_ratio: Fraction of pixels below brightness=50 (higher = darker)
"""
import csv
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import DATA_DIR as BASE, CSV_PATH

OUT_DIR  = Path("evening")
OUT_DIR.mkdir(exist_ok=True)

# ── Brightness functions ──────────────────────────────────────────────────────
def greyscale_brightness(img: Image.Image) -> float:
    """Mean pixel value of greyscale conversion (0–255)."""
    return np.array(img.convert("L")).mean()

def hsv_brightness(img: Image.Image) -> float:
    """Mean Value channel from HSV (0–255)."""
    return np.array(img.convert("HSV"))[:, :, 2].mean()

def cropped_brightness(img: Image.Image) -> float:
    """Greyscale mean of center band — excludes sky (top 20%) and road (bottom 30%)."""
    w, h = img.size
    cropped = img.crop((0, int(h * 0.2), w, int(h * 0.7)))
    return np.array(cropped.convert("L")).mean()

def luminance_brightness(img: Image.Image) -> float:
    """Perceptual luminance via Y channel of YCbCr (weights green most, blue least)."""
    return np.array(img.convert("YCbCr"))[:, :, 0].mean()

def dark_pixel_ratio(img: Image.Image) -> float:
    arr = np.array(img.convert("L"))
    ratio = (arr < 50).sum() / arr.size
    return round((1 - ratio) * 255, 2)  # invert so 0=dark, 255=bright

# ── Step 1: Compute all metrics ───────────────────────────────────────────────
results = []
for img_path in sorted(OUT_DIR.glob("*.jpg")):
    img = Image.open(img_path)
    results.append({
        "name":      img_path.name,
        "grey":      round(greyscale_brightness(img), 1),
        "hsv":       round(hsv_brightness(img), 1),
        "cropped":   round(cropped_brightness(img), 1),
        "luminance": round(luminance_brightness(img), 1),
        "dark":      dark_pixel_ratio(img),
    })

results.sort(key=lambda x: x["grey"])

print(f"\n{'Metric':<12} {'Min':>6} {'Max':>6}")
for key in ["grey", "hsv", "cropped", "luminance", "dark"]:
    vals = [r[key] for r in results]
    print(f"  {key:<10} {min(vals):>6.1f} {max(vals):>6.1f}")

print("\nDarkest 5 (greyscale):")
for r in results[:5]:
    print(f"  grey={r['grey']:5.1f}  hsv={r['hsv']:5.1f}  lum={r['luminance']:5.1f}  crop={r['cropped']:5.1f}  dark={r['dark']:5.1f}  {r['name']}")

print("\nBrightest 5 (greyscale):")
for r in results[-5:]:
    print(f"  grey={r['grey']:5.1f}  hsv={r['hsv']:5.1f}  lum={r['luminance']:5.1f}  crop={r['cropped']:5.1f}  dark={r['dark']:5.1f}  {r['name']}")

# ── Step 2: Save CSV ──────────────────────────────────────────────────────────
with open("evening_brightness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["name", "grey", "hsv", "cropped", "luminance", "dark"])
    w.writeheader()
    w.writerows(results)

print("\nSaved brightness scores to evening_brightness.csv")

# ── Step 3: Generate HTML viewer ──────────────────────────────────────────────
import json
rows_json = json.dumps(results)

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evening image brightness</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; padding: 16px; }}
  h2 {{ margin-bottom: 4px; }}
  p  {{ margin-top: 0; color: #aaa; font-size: 13px; margin-bottom: 12px; }}
  .controls {{ display: flex; align-items: center; gap: 12px; margin-bottom: 16px; flex-wrap: wrap; }}
  .toggle {{ display: flex; background: #2a2a2a; border-radius: 6px; overflow: hidden; border: 1px solid #444; }}
  .toggle button {{ background: none; border: none; color: #aaa; padding: 7px 14px; cursor: pointer; font-size: 12px; transition: background 0.15s; white-space: nowrap; }}
  .toggle button.active {{ background: #555; color: #fff; }}
  .threshold-wrap {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #aaa; }}
  .threshold-wrap input {{ width: 140px; accent-color: #888; }}
  #thresh-val {{ color: #fff; font-weight: bold; min-width: 36px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ width: 200px; background: #222; border-radius: 6px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; }}
  .card.dimmed {{ opacity: 0.2; }}
  .scores {{ padding: 6px 8px; font-size: 11px; display: flex; flex-direction: column; gap: 2px; }}
  .score-row {{ display: flex; justify-content: space-between; color: #666; }}
  .score-row.active {{ color: #fff; font-weight: bold; }}
  .score-row.active span.label {{ color: #7ec8e3; }}
</style>
</head>
<body>
<h2>Evening images — brightness viewer</h2>
<p>Toggle which metric to sort by. Images above the threshold are dimmed. All five scores shown per image.</p>

<div class="controls">
  <div class="toggle">
    <button id="btn-grey"      class="active" onclick="setSort('grey')">Greyscale</button>
    <button id="btn-hsv"                      onclick="setSort('hsv')">HSV</button>
    <button id="btn-cropped"                  onclick="setSort('cropped')">Cropped</button>
    <button id="btn-luminance"                onclick="setSort('luminance')">Luminance</button>
    <button id="btn-dark"                     onclick="setSort('dark')">Dark ratio</button>
  </div>
  <div class="threshold-wrap">
    Nighttime cutoff:
    <input type="range" min="0" max="255" value="80" step="1" id="thresh-slider" oninput="updateThreshold()">
    <span id="thresh-val">80</span>
  </div>
</div>

<div class="grid" id="grid"></div>

<script>
const data = {rows_json};
const METRICS = [
  {{ key: 'grey',      label: 'grey' }},
  {{ key: 'hsv',       label: 'hsv' }},
  {{ key: 'cropped',   label: 'crop' }},
  {{ key: 'luminance', label: 'lum' }},
  {{ key: 'dark',      label: 'dark' }},
];
let sortMode = 'grey';

function setSort(mode) {{
  sortMode = mode;
  METRICS.forEach(m => {{
    document.getElementById('btn-' + m.key).classList.toggle('active', m.key === mode);
  }});
  render();
}}

function updateThreshold() {{
  const val = document.getElementById('thresh-slider').value;
  document.getElementById('thresh-val').textContent = val;
  render();
}}

function render() {{
  const thresh = +document.getElementById('thresh-slider').value;
  const sorted = [...data].sort((a, b) => a[sortMode] - b[sortMode]);
  document.getElementById('grid').innerHTML = sorted.map(r => {{
    const isNight = r[sortMode] <= thresh;
    const scoreRows = METRICS.map(m => {{
      const active = m.key === sortMode ? 'active' : '';
      return `<div class="score-row ${{active}}"><span class="label">${{m.label}}</span><span>${{r[m.key].toFixed(1)}}</span></div>`;
    }}).join('');
    return `
      <div class="card ${{isNight ? '' : 'dimmed'}}">
        <img src="evening/${{r.name}}" loading="lazy">
        <div class="scores">${{scoreRows}}</div>
      </div>`;
  }}).join('');
}}

render();
</script>
</body>
</html>"""

with open("evening_viewer.html", "w") as f:
    f.write(html)

print("Saved evening_viewer.html — open in a browser to pick your threshold")