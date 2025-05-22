import os
import json
import csv
import numpy as np
import cv2
import requests
import traceback
from PIL import Image
from io import BytesIO
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession

# === Settings ===
DOMAIN = "https://blobpics.tech"
LIST_URL = f"{DOMAIN}/api/list"
IMAGE_URL = f"{DOMAIN}/images/{{id}}"
TAGS_URL = f"{DOMAIN}/images/tags/{{id}}.json"
UPLOAD_COMPLETE_URL = f"{DOMAIN}/api/upload-tagged-json"
TAG_THRESHOLD = 0.5
TAG_BLACKLIST = ["blue_skin"]
CATEGORY_MAP = {
    0: "general", 1: "character", 2: "copyright",
    3: "artist", 4: "meta", 5: "rating", 9: "rating"
}

# === Download model and tag list ===
print("🔽 Downloading model and tag list...")
model_path = hf_hub_download("SmilingWolf/wd-vit-tagger-v3", "model.onnx")
tags_path = hf_hub_download("SmilingWolf/wd-vit-tagger-v3", "selected_tags.csv")
model = InferenceSession(model_path)
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
_, height, _, _ = model.get_inputs()[0].shape

# === Load tag metadata ===
id2label, id2cat = {}, {}
with open(tags_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        id2label[i] = row["name"]
        id2cat[i] = int(row["category"])
print(f"📄 Loaded {len(id2label)} tags.")

# === Get unprocessed image IDs ===
print("🌐 Fetching image list from server...")
try:
    r = requests.get(LIST_URL)
    r.raise_for_status()
    gallery = r.json()
    ids = [img["imageId"] for img in gallery if "imageId" in img]
    print(f"🖼️ Found {len(ids)} images to process.")
except Exception as e:
    print(f"❌ Failed to fetch list: {e}")
    exit(1)

# === Process each image
for image_id in ids:
    try:
        image_url = IMAGE_URL.format(id=image_id)
        print(f"🔍 Tagging {image_url}...")

        resp = requests.get(image_url)
        if resp.status_code != 200:
            print(f"❌ Failed to fetch image {image_id} (HTTP {resp.status_code})")
            continue

        image = Image.open(BytesIO(resp.content)).convert("RGB")
        image = np.array(image)[:, :, ::-1]

        def make_square(img, target_size):
            h, w = img.shape[:2]
            desired = max(h, w, target_size)
            delta_w = desired - w
            delta_h = desired - h
            return cv2.copyMakeBorder(
                img,
                delta_h // 2, delta_h - delta_h // 2,
                delta_w // 2, delta_w - delta_w // 2,
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

        def smart_resize(img, size):
            return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC)

        image = smart_resize(make_square(image, height), height).astype(np.float32)
        tensor = np.expand_dims(image, axis=0)

        probs = model.run([output_name], {input_name: tensor})[0][0]
        grouped = {k: [] for k in CATEGORY_MAP.values()}
        grouped["unknown"] = []

        for i, score in enumerate(probs):
            if i not in id2label: continue
            tag, conf = id2label[i], round(float(score), 4)
            if conf > TAG_THRESHOLD and tag not in TAG_BLACKLIST:
                category = CATEGORY_MAP.get(id2cat.get(i, -1), "unknown")
                grouped[category].append((tag, conf))

        for tags in grouped.values():
            tags.sort(key=lambda x: -x[1])

        all_tags = [tag for group in grouped.values() for tag, _ in group]
        booru = " ".join(f"({tag}:{score:.3f})" for group in grouped.values() for tag, score in group)

        full_data = {
            "imageId": image_id,
            "fileName": image_id,
            "tags": all_tags,
            "booru": booru,
            "groupedTags": grouped
        }

        # === Upload JSON to worker
        resp = requests.post(
            UPLOAD_COMPLETE_URL,
            headers={"Content-Type": "application/json"},
            data=json.dumps(full_data)
        )
        if resp.status_code >= 400:
            print(f"❌ Upload failed: {resp.status_code} → {resp.text}")
        else:
            print(f"✅ Tagged and uploaded {image_id}")

    except Exception as e:
        print(f"❌ Error on {image_id}: {e}")
        traceback.print_exc()
