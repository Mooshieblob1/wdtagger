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
DOMAIN = "blobpics.tech"
LIST_URL = f"https://{DOMAIN}/list"
UPLOAD_JSON_URL = f"https://{DOMAIN}/upload-json"
ORIGINAL_URL = f"https://{DOMAIN}/original/{{id}}.png"
TAG_THRESHOLD = 0.5
TAG_BLACKLIST = ["blue_skin"]
CATEGORY_MAP = {
    0: "general", 1: "character", 2: "copyright",
    3: "artist", 4: "meta", 5: "rating", 9: "rating"
}
TAG_PATH = "tagged_images.json"

# === Download model and tags ===
print("🔽 Downloading model and tag list from HuggingFace...")
model_path = hf_hub_download("SmilingWolf/wd-vit-tagger-v3", "model.onnx")
tags_path = hf_hub_download("SmilingWolf/wd-vit-tagger-v3", "selected_tags.csv")
print(f"✅ Model: {model_path}")
print(f"✅ Tags: {tags_path}")

# === Load ONNX model ===
model = InferenceSession(model_path)
input_name = model.get_inputs()[0].name
output_name = model.get_outputs()[0].name
input_shape = model.get_inputs()[0].shape
_, height, _, _ = input_shape
print(f"📐 Model input shape: {input_shape}")

# === Load tag metadata ===
id2label, id2cat = {}, {}
with open(tags_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        id2label[i] = row["name"]
        id2cat[i] = int(row["category"])
print(f"📄 Loaded {len(id2label)} tags.")

# === Fetch all image IDs from BlobPics ===
print("🌐 Fetching image IDs from BlobPics /list...")
ids = []
try:
    r = requests.get(LIST_URL)
    r.raise_for_status()
    ids = r.json()["images"]
    print(f"🖼️ Found {len(ids)} images.")
except Exception as e:
    print(f"❌ Failed to fetch image list: {e}")
    exit(1)

tagged_by_id = {}
for image_id in ids:
    image_url = ORIGINAL_URL.format(id=image_id)
    print(f"🔍 Tagging {image_url}...")

    try:
        resp = requests.get(image_url)
        if resp.status_code != 200:
            print(f"❌ Could not download: {image_url} (HTTP {resp.status_code})")
            continue

        image = Image.open(BytesIO(resp.content)).convert("RGB")
        image = np.array(image)
        image = image[:, :, ::-1]

        def make_square(img, target_size):
            old_size = img.shape[:2]
            desired_size = max(old_size)
            desired_size = max(desired_size, target_size)
            delta_w = desired_size - old_size[1]
            delta_h = desired_size - old_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = [255, 255, 255]
            return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        def smart_resize(img, size):
            if img.shape[0] > size:
                return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            elif img.shape[0] < size:
                return cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
            return img

        image = make_square(image, height)
        image = smart_resize(image, height)
        image = image.astype(np.float32)
        tensor = np.expand_dims(image, axis=0)

        # Predict tags
        probs = model.run([output_name], {input_name: tensor})[0][0]
        grouped = {k: [] for k in CATEGORY_MAP.values()}
        grouped["unknown"] = []
        for i, score in enumerate(probs):
            if i not in id2label:
                continue
            tag, conf = id2label[i], round(float(score), 4)
            if conf > TAG_THRESHOLD and tag not in TAG_BLACKLIST:
                category = CATEGORY_MAP.get(id2cat.get(i, -1), "unknown")
                grouped[category].append((tag, conf))
        for tags in grouped.values():
            tags.sort(key=lambda x: -x[1])

        all_tags = [tag for g in grouped.values() for tag, _ in g]
        booru = " ".join(f"({tag}:{score:.3f})" for g in grouped.values() for tag, score in g)

        # Save full_data
        full_data = {
            "imageId": image_id,
            "fileName": f"{image_id}.png",
            "tags": all_tags,
            "booru": booru,
            "groupedTags": grouped
        }
        tagged_by_id[image_id] = full_data

        # Save per-image .json file
        json_filename = f"{image_id}.json"
        with open(json_filename, "w") as f:
            json.dump(full_data, f, indent=2)
        print(f"✅ Tags saved: {json_filename}")

        # Upload .json to your server
        with open(json_filename, "rb") as f:
            resp = requests.post(
                UPLOAD_JSON_URL,
                files={"json": (json_filename, f, "application/json")}
            )
            print(f"⬆️ Uploaded {json_filename}: HTTP {resp.status_code}")

    except Exception as e:
        print(f"❌ Error processing {image_id}: {e}")
        traceback.print_exc()

# Save combined cache if needed
with open(TAG_PATH, "w") as f:
    json.dump(list(tagged_by_id.values()), f, indent=2)
print("📝 All tags cached locally.")
