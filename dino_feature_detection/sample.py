import shutil
from pathlib import Path
import json

IMAGE_STEM = "dr5rsp4zyhx4-dr5rsp4zvwpv-cds-79ddade2485edbde-20160717-1131-40"

FOLDERS = [
    "original__low",
    "original__high",
    "rephrased__medium",
    "specific__medium",
]

SRC_BASE = Path("dino_grid")  # adjust if your grid output is elsewhere
OUT_DIR  = Path("figure_samples")
OUT_DIR.mkdir(exist_ok=True)
# load the box data json

for folder in FOLDERS:
    matches = list((SRC_BASE / folder).glob(f"{IMAGE_STEM}.*"))
    if not matches:
        print(f"WARNING: no match found in {folder}")
        continue
    src = matches[0]
    dst = OUT_DIR / f"{folder}{src.suffix}"
    shutil.copy(src, dst)
    print(f"Copied {src} -> {dst}")

print(f"\nDone. Files saved to {OUT_DIR}/")