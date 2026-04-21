"""dino_testset.py
Run Grounding DINO on a flat folder of images (no CSV / metadata needed).
Saves counts CSV and bounding box JSON to dino_counts/, compatible with
proxy_viewer.html.
"""
import csv
import re
import json
import yaml
import sys
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ── Config ────────────────────────────────────────────────────────────────────
TEST_DIR         = Path("test-set")   # folder of .jpg images
PROMPT_NAME      = "informed_prompt_3"                 # must match prompts.yaml
COUNTS_DIR       = Path("dino_counts")
MODEL_ID         = "IDEA-Research/grounding-dino-base"
BOX_THRESHOLD    = 0.30
TEXT_THRESHOLD   = 0.25

COUNTS_DIR.mkdir(exist_ok=True)

# ── Load prompts ──────────────────────────────────────────────────────────────
with open("prompts.yaml") as f:
    _config = yaml.safe_load(f)

PROMPT_MAP = {
    p["name"]: {
        "name":     p["name"],
        "text":     p["text"],
        "patterns": {k: re.compile(v) for k, v in p["patterns"].items()}
    }
    for p in _config["prompts"]
}

if PROMPT_NAME not in PROMPT_MAP:
    raise ValueError(f"Prompt '{PROMPT_NAME}' not found in prompts.yaml")

prompt = PROMPT_MAP[PROMPT_NAME]

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device)
print(f"Model loaded: {MODEL_ID} on {device}")

# ── Find images ───────────────────────────────────────────────────────────────
image_paths = sorted(TEST_DIR.glob("*.jpg"))
print(f"Found {len(image_paths)} images in {TEST_DIR}")
print(f"Prompt: {PROMPT_NAME}")
print(f"Text:   {prompt['text']}")
print("-" * 60)

# ── Helpers ───────────────────────────────────────────────────────────────────
def detect(image):
    inputs = processor(images=image, text=prompt["text"], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        target_sizes=[image.size[::-1]]
    )[0]

def count_detections(detections):
    counts = {key: 0 for key in prompt["patterns"]}
    label_map = {}
    for label in detections["labels"]:
        label_lower = label.lower()
        matched = "other"
        for key, pattern in prompt["patterns"].items():
            if pattern.search(label_lower):
                counts[key] += 1
                matched = key
                break
        label_map[label] = matched
    return counts, label_map

# ── Run ───────────────────────────────────────────────────────────────────────
output_rows = []
box_data    = {}

for img_path in tqdm(image_paths, desc="Processing", unit="img"):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        tqdm.write(f"Skipping {img_path.name}: {e}")
        continue

    dets = detect(image)
    counts, label_map = count_detections(dets)

    box_data[img_path.name] = [
        {
            "box":      box,
            "label":    label,
            "score":    round(float(score), 3),
            "category": label_map.get(label, "other"),
        }
        for box, label, score in zip(
            dets["boxes"].cpu().numpy().tolist(),
            dets["labels"],
            dets["scores"].cpu().numpy().tolist()
        )
    ]

    output_rows.append({
        "image":  img_path.name,
        "period": "test",
        **counts,
        "run":    PROMPT_NAME,
    })

    tqdm.write(
        f"{img_path.name[-50:]}  " +
        "  ".join(f"{k}={v}" for k, v in counts.items())
    )

# ── Save ──────────────────────────────────────────────────────────────────────
run_name    = f"{PROMPT_NAME}_testset"
output_csv  = COUNTS_DIR / f"dino_counts_{run_name}.csv"
output_json = COUNTS_DIR / f"dino_counts_{run_name}.json"

fieldnames = ["image", "period"] + list(prompt["patterns"].keys()) + ["run"]
with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(output_rows)

with open(output_json, "w") as f:
    json.dump(box_data, f)

print(f"\nSaved {len(output_rows)} rows to {output_csv}")
print(f"Saved box data to {output_json}")