"""
Using Grounding DINO, identify proxies for lightness in daytime images.
Counts: trees, storefronts, and lampposts.

Edit the CONFIG block below to experiment with different runs.
Each run saves to a uniquely named CSV so you can compare results.
"""
import csv
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ============================================================
# CONFIG — edit this block to experiment between runs
# ============================================================

MODEL_ID = "IDEA-Research/grounding-dino-base"  # or "grounding-dino-tiny"

# Text prompt fed to the model — period-separated labels
TEXT_LABELS = "a street tree . a shop front . a street lamp pole . a streetlight"

# Detection thresholds — lower = more detections, more noise
THRESHOLD      = 0.32
TEXT_THRESHOLD = 0.25

# How many daytime images to process
SAMPLES = 50

# Paths
DATA_DIR = "/Users/mariasilva/data/urban-mosaic/washington-square"
CSV_PATH = "/Users/mariasilva/data/urban-mosaic/washington-square.csv"

# Output CSV name — edit this to label your run, e.g. "run_base_low_thresh"
RUN_NAME   = "run_base_rephrased_low_thresh"
OUTPUT_CSV = f"proxy_counts_{RUN_NAME}.csv"

# ============================================================
# LABEL MATCHING — update if you change TEXT_LABELS above
# ============================================================
COUNT_PATTERNS = {
    "tree":       re.compile(r"\b(tree|street tree)\b"),
    "storefront": re.compile(r"\b(storefront|shop front|shopfront|awning|shop sign)\b"),
    "lamppost":   re.compile(r"\b(lamppost|streetlight|street light|street lamp|lamp pole)\b"),
}

# ============================================================
# RUN
# ============================================================
print(f"Model:          {MODEL_ID}")
print(f"Prompt:         {TEXT_LABELS}")
print(f"Thresholds:     box={THRESHOLD}  text={TEXT_THRESHOLD}")
print(f"Samples:        {SAMPLES}")
print(f"Output:         {OUTPUT_CSV}")
print("-" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)

output_rows = []
processed   = 0

with open(CSV_PATH, newline="") as f:
    reader = csv.DictReader(f)
    rows = [row for row in reader if row["period"] != "evening"]

for row in tqdm(rows[:SAMPLES * 2], total=SAMPLES, desc="Processing", unit="img"):
    if processed >= SAMPLES:
        break

    img_path = DATA_DIR + "/" + row["image"]
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        tqdm.write(f"Skipping {img_path}: {e}")
        continue

    inputs = processor(
        images=image,
        text=TEXT_LABELS,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    detections = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )[0]

    counts = {key: 0 for key in COUNT_PATTERNS}
    for label in detections["labels"]:
        label_lower = label.lower()
        for key, pattern in COUNT_PATTERNS.items():
            if pattern.search(label_lower):
                counts[key] += 1

    output_rows.append({
        "image":       row["image"],
        "taken_on":    row["taken_on"],
        "period":      row["period"],
        "trees":       counts["tree"],
        "storefronts": counts["storefront"],
        "lampposts":   counts["lamppost"],
        "run":         RUN_NAME,
    })

    processed += 1
    tqdm.write(
        f"[{processed}/{SAMPLES}] {row['image']}  "
        f"trees={counts['tree']}  "
        f"storefronts={counts['storefront']}  "
        f"lampposts={counts['lamppost']}"
    )

fieldnames = ["image", "taken_on", "period", "trees", "storefronts", "lampposts", "run"]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nSaved {len(output_rows)} rows to {OUTPUT_CSV}")