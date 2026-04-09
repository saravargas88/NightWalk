

'''
Using Grounding DINO identify the proxies for lightness in daytime images 
For now trees storefronts and lampposts


'''


import csv

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

model_id = "IDEA-Research/grounding-dino-tiny"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

BASE     = "urban-mosaic/washington-square"
CSV_PATH = "urban-mosaic/washington-square.csv"
OUT_CSV  = "proxy_counts.csv"
LIMIT    = 50

text_labels = [["a tree . a storefront . a lamppost"]]

output_rows = []
processed = 0

for row in csv.DictReader(open(CSV_PATH)):
    if row["period"] == "evening":
        continue
    if processed >= LIMIT:
        break

    img_path = BASE + "/" + row["image"]
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_path}: {e}")
        continue

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    detections = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]

    counts = {"tree": 0, "storefront": 0, "lamppost": 0}
    for label in detections["labels"]:
        for key in counts:
            if key in label.lower():
                counts[key] += 1

    output_rows.append({
        "image": row["image"],
        "taken_on": row["taken_on"],
        "period": row["period"],
        "trees": counts["tree"],
        "storefronts": counts["storefront"],
        "lampposts": counts["lamppost"],
    })

    processed += 1
    print(f"[{processed}/{LIMIT}] {row['image']}  trees={counts['tree']} storefronts={counts['storefront']} lampposts={counts['lamppost']}")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "taken_on", "period", "trees", "storefronts", "lampposts"])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nSaved {len(output_rows)} rows to {OUT_CSV}")
