"""brightness_scorer.py
Compute greyscale brightness scores for all images in the dataset.
Does NOT copy files — just scores and saves to CSV.
Then generates a viewer for the darkest N images.
"""
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.paths import DATA_DIR, CSV_PATH

# ── Config ────────────────────────────────────────────────────────────────────
OUTPUT_CSV  = Path("all_brightness.csv")
VIEWER      = Path("darkest_viewer.html")

DARKEST_N   = 1000   # how many images to show in the viewer

# ── Load CSV ──────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
print(f"Total images in dataset: {len(df)}")

# ── Compute brightness ────────────────────────────────────────────────────────
def greyscale_brightness(path: Path) -> float | None:
    try:
        img = Image.open(path).convert("L")
        return round(np.array(img).mean(), 2)
    except Exception:
        return None

results = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring", unit="img"):
    img_path = DATA_DIR / row["image"]
    score = greyscale_brightness(img_path)
    if score is None:
        continue
    results.append({
        "image":    row["image"],
        "period":   row.get("period", ""),
        "hour":     row.get("hour", ""),
        "grey":     score,
    })

results_df = pd.DataFrame(results).sort_values("grey").reset_index(drop=True)

print(f"\nScored {len(results_df)} images")
print(f"Brightness range: {results_df['grey'].min():.1f} – {results_df['grey'].max():.1f}")
print(f"\nDarkest 10:")
print(results_df[["image", "period", "grey"]].head(10).to_string())

# ── Save CSV ──────────────────────────────────────────────────────────────────
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved scores to {OUTPUT_CSV}")

# ── Distribution summary ──────────────────────────────────────────────────────
print(f"\nBrightness distribution:")
for cutoff in [50, 80, 106, 120, 150, 180]:
    n = (results_df["grey"] <= cutoff).sum()
    print(f"  <= {cutoff:3d}:  {n:5d} images  ({n/len(results_df)*100:.1f}%)")

# ── Generate viewer for darkest N ─────────────────────────────────────────────
darkest = results_df.head(DARKEST_N)

rows_json = json.dumps([
    {
      "image": f"../urban-mosaic/washington-square/{row['image']}",
        "grey":   row["grey"],
        "period": row["period"],
    }
    for _, row in darkest.iterrows()
])

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Darkest {DARKEST_N} images</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; padding: 16px; }}
  h2 {{ margin-bottom: 4px; }}
  p  {{ color: #aaa; font-size: 13px; margin-bottom: 12px; }}
  .controls {{ display: flex; gap: 12px; align-items: center; margin-bottom: 16px; flex-wrap: wrap; }}
  .controls input {{ background: #222; border: 1px solid #444; color: #eee; padding: 6px 10px; border-radius: 6px; font-size: 13px; width: 160px; }}
  .threshold-wrap {{ display: flex; align-items: center; gap: 8px; font-size: 13px; color: #aaa; }}
  .threshold-wrap input[type=range] {{ width: 140px; accent-color: #888; }}
  #thresh-val {{ color: #fff; font-weight: bold; min-width: 36px; }}
  .count {{ font-size: 13px; color: #888; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ width: 200px; background: #222; border-radius: 6px; overflow: hidden; }}
  .card.dimmed {{ opacity: 0.2; }}
  .card img {{ width: 100%; height: 140px; object-fit: cover; display: block; }}
  .meta {{ font-size: 11px; padding: 5px 8px; display: flex; justify-content: space-between; color: #888; }}
  .meta .grey {{ color: #fff; font-weight: bold; }}
  .badge {{ font-size: 10px; padding: 2px 6px; border-radius: 99px; background: #333; color: #aaa; }}
</style>
</head>
<body>
<h2>Darkest {DARKEST_N} images — greyscale brightness</h2>
<p>Sorted darkest to brightest. Use threshold slider to mark cutoff.</p>

<div class="controls">
  <div class="threshold-wrap">
    Cutoff:
    <input type="range" min="0" max="255" value="106" step="1" id="thresh-slider" oninput="updateThreshold()">
    <span id="thresh-val">106</span>
  </div>
  <span class="count" id="count"></span>
</div>

<div class="grid" id="grid"></div>

<script>
const data = {rows_json};

function updateThreshold() {{
  const val = document.getElementById('thresh-slider').value;
  document.getElementById('thresh-val').textContent = val;
  render();
}}

function render() {{
  const thresh = +document.getElementById('thresh-slider').value;
  const below = data.filter(r => r.grey <= thresh).length;
  document.getElementById('count').textContent = below + ' images below cutoff';
  document.getElementById('grid').innerHTML = data.map(r => `
    <div class="card ${{r.grey > thresh ? 'dimmed' : ''}}">
      <img src="${{r.image}}" loading="lazy" onerror="this.style.background='#333';this.style.height='140px'">
      <div class="meta">
        <span class="grey">${{r.grey.toFixed(1)}}</span>
        <span class="badge">${{r.period}}</span>
      </div>
    </div>
  `).join('');
}}

render();
</script>
</body>
</html>"""

VIEWER.write_text(html)
print(f"Saved {VIEWER} — serve from /Users/mariasilva/Documents/NightWalk and open in browser")