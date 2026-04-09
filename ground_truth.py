

'''
Computing the brightness of each evening imge
Figure out pixel threshold to select the nighttime images
Then find pair for daytime (either urban-mosaic or google street view)

use those as the ground truth for training the model to predict brightness from daytime proxies.

'''
import csv
import shutil
from pathlib import Path

BASE     = Path("urban-mosaic/washington-square")
CSV_PATH = Path("urban-mosaic/washington-square.csv")
OUT_DIR  = Path("evening")

OUT_DIR.mkdir(exist_ok=True)

copied = 0
missing = 0

with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["period"] != "evening":
            continue
        src = BASE / row["image"]
        if not src.exists():
            missing += 1
            continue
        dst = OUT_DIR / src.name
        shutil.copy2(src, dst)
        copied += 1

print(f"Copied {copied} evening images to '{OUT_DIR}/'  ({missing} missing)")

# ── Step 1: Compute mean pixel brightness for each evening image ─────────────
import numpy as np
from PIL import Image

results = []

for img_path in sorted(OUT_DIR.glob("*.jpg")):
    img = Image.open(img_path).convert("L")   # grayscale
    brightness = np.array(img).mean()
    results.append((img_path.name, brightness))

results.sort(key=lambda x: x[1])

print(f"\nBrightness range: {results[0][1]:.1f} – {results[-1][1]:.1f}  (0=black, 255=white)")
print("\nDarkest 5:")
for name, b in results[:5]:
    print(f"  {b:5.1f}  {name}")
print("\nBrightest 5:")
for name, b in results[-5:]:
    print(f"  {b:5.1f}  {name}")

# save full brightness list to CSV for inspection
import csv as _csv
with open("evening_brightness.csv", "w", newline="") as f:
    w = _csv.writer(f)
    w.writerow(["filename", "mean_brightness"])
    w.writerows(results)
print("\nSaved brightness scores to evening_brightness.csv")

# ── Step 2: Generate HTML viewer to visually pick brightness threshold ────────
rows_html = ""
for name, b in results:
    img_src = f"evening/{name}"
    rows_html += f"""
    <div class="card">
      <img src="{img_src}" loading="lazy">
      <div class="score">{b:.1f}</div>
    </div>"""

html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Evening image brightness</title>
<style>
  body {{ font-family: sans-serif; background: #111; color: #eee; padding: 16px; }}
  h2 {{ margin-bottom: 4px; }}
  p  {{ margin-top: 0; color: #aaa; font-size: 13px; }}
  .grid {{ display: flex; flex-wrap: wrap; gap: 10px; }}
  .card {{ width: 220px; background: #222; border-radius: 6px; overflow: hidden; }}
  .card img {{ width: 100%; display: block; }}
  .score {{ text-align: center; padding: 6px; font-size: 14px; font-weight: bold; }}
</style>
</head>
<body>
<h2>Evening images — sorted darkest to brightest (mean pixel brightness 0–255)</h2>
<p>Scroll through and decide where the cutoff between nighttime and still-daytime should be.</p>
<div class="grid">{rows_html}
</div>
</body>
</html>"""

with open("evening_viewer.html", "w") as f:
    f.write(html)
print("Saved evening_viewer.html — open in a browser to pick your threshold")
