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
        "image":     str(OUT_DIR / img_path.name),  # e.g. "evening/filename.jpg"
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
    print(f"  grey={r['grey']:5.1f}  hsv={r['hsv']:5.1f}  lum={r['luminance']:5.1f}  crop={r['cropped']:5.1f}  dark={r['dark']:5.1f}  {r['image']}")

print("\nBrightest 5 (greyscale):")
for r in results[-5:]:
    print(f"  grey={r['grey']:5.1f}  hsv={r['hsv']:5.1f}  lum={r['luminance']:5.1f}  crop={r['cropped']:5.1f}  dark={r['dark']:5.1f}  {r['image']}")

# ── Step 2: Save CSV ──────────────────────────────────────────────────────────
with open("evening_brightness.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["image", "grey", "hsv", "cropped", "luminance", "dark"])
    w.writeheader()
    w.writerows(results)

print("\nSaved brightness scores to evening_brightness.csv")