

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


#loop through 50 images and get the proxy counts 
data_dir     = "urban-mosaic/washington-square"
csv_path = "urban-mosaic/washington-square.csv"
outut_csv  = "proxy_counts.csv"
samples = 50

text_labels = [["a tree . a storefront . a lamppost"]]

output_rows = []
processed = 0

for row in csv.DictReader(open(csv_path)):
    #skip evening img
    if row["period"] == "evening":
        continue

    img_path = data_dir + "/" + row["image"]
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Skipping {img_path}: {e}")
        continue

    inputs = processor(images=image, text=text_labels, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    detections = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )[0]

    #so the model will return a list of detected objects w labels
    # we cound how many times a label appears and increment count 
    counts = {"tree": 0, "storefront": 0, "lamppost": 0}
    for label in detections["labels"]:
        for key in counts:
            if key in label.lower():
                counts[key] += 1

    #data of image in csv  
    output_rows.append({
        "image": row["image"],
        "taken_on": row["taken_on"],
        "period": row["period"],
        "trees": counts["tree"],
        "storefronts": counts["storefront"],
        "lampposts": counts["lamppost"],
    })

    processed += 1
    print(f"[{processed}/{samples}] {row['image']}  trees={counts['tree']} storefronts={counts['storefront']} lampposts={counts['lamppost']}")
    
    if processed >= samples:
        break

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["image", "taken_on", "period", "trees", "storefronts", "lampposts"])
    writer.writeheader()
    writer.writerows(output_rows)

print(f"\nSaved {len(output_rows)} rows to {output_csv}")
