"""make_pairs.py
For each image in daytime/, finds the matching image in urban-mosaic/washington-square/
using the ID from washington-square.csv, then stacks them vertically and saves 10 pairs.
"""
import sys
import pandas as pd
from pathlib import Path
from PIL import Image

ROOT       = Path(__file__).parent
DATA_DIR   = ROOT / "urban-mosaic" / "washington-square"
CSV_PATH   = ROOT / "urban-mosaic" / "washington-square.csv"
DAY_DIR    = ROOT / "daytime"
OUT_DIR    = ROOT / "day_night_pairs"
OUT_DIR.mkdir(exist_ok=True)

N = 10

# Load CSV
df = pd.read_csv(CSV_PATH)
# Build a lookup: id -> image path in urban-mosaic
id_to_night = {str(row["id"]): DATA_DIR / row["image"] for _, row in df.iterrows()}

saved = 0
for day_img_path in sorted(DAY_DIR.iterdir()):
    if saved >= N:
        break
    if day_img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue

    # filename is "id-lat-long.jpg" — extract the id (first part before first dash)
    stem = day_img_path.stem  # e.g. "abc123-40.123--73.456"
    img_id = stem.split("_")[0]

    night_path = id_to_night.get(img_id)
    if night_path is None or not night_path.exists():
        print(f"No match for id={img_id} ({day_img_path.name}), skipping")
        continue

    day   = Image.open(day_img_path).convert("RGB")
    night = Image.open(night_path).convert("RGB")

    # Resize night to match day width, maintaining aspect ratio
    if day.width != night.width:
        scale  = day.width / night.width
        night  = night.resize((day.width, int(night.height * scale)), Image.LANCZOS)

    # Stack horizontally
    combined = Image.new("RGB", (day.width + night.width, max(day.height, night.height)))
    combined.paste(day,   (0, 0))
    combined.paste(night, (day.width, 0))

    out_path = OUT_DIR / f"pair_{saved+1:02d}_{img_id}.jpg"
    combined.save(out_path, quality=90)
    print(f"[{saved+1}/{N}] Saved {out_path.name}  (day: {day_img_path.name}, night: {night_path.name})")
    saved += 1

if saved < N:
    print(f"\nWarning: only found {saved} valid pairs (wanted {N})")
else:
    print(f"\nDone. {saved} pairs saved to {OUT_DIR}/")