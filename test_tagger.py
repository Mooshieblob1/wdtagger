import os
import json
import csv
import numpy as np
import cv2
import torch
import traceback
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from onnxruntime import InferenceSession
from torchvision import transforms
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases
from appwrite.query import Query

# === Load environment variables ===
load_dotenv()
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")

# === Appwrite Configuration ===
APPWRITE_ENDPOINT = "https://syd.cloud.appwrite.io/v1"
APPWRITE_PROJECT_ID = "682b826b003d9cba9018"
BUCKET_ID = "682cfa1a0016991596f5"
DATABASE_ID = "682b89cc0016319fcf30"
COLLECTION_ID = "682d7b240022ba63cd02"
TAG_PATH = "tagged_images.json"
TAG_THRESHOLD = 0.5
TAG_BLACKLIST = ["blue_skin"]

# === Category Mapping ===
CATEGORY_MAP = {
    0: "general", 1: "character", 2: "copyright",
    3: "artist", 4: "meta", 5: "rating", 9: "rating"
}

# === Initialize Appwrite ===
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT).set_project(APPWRITE_PROJECT_ID).set_key(APPWRITE_API_KEY)
storage = Storage(client)
databases = Databases(client)

# === Download model + tag CSV from HuggingFace ===
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
print(f"📐 Model input shape: {input_shape}")
_, height, _, _ = input_shape  # May raise error if shape is not [1, 3, H, W]
print("✅ Model loaded.")

# === Load tag metadata ===
id2label, id2cat = {}, {}
with open(tags_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        id2label[i] = row["name"]
        id2cat[i] = int(row["category"])
print(f"📄 Loaded {len(id2label)} tags.")

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((height, height)),
    transforms.CenterCrop(height),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# === Local tag cache ===
tagged_by_id = {}
if os.path.exists(TAG_PATH):
    with open(TAG_PATH, "r") as f:
        for entry in json.load(f):
            tagged_by_id[entry["imageId"]] = entry

# === Fetch files from Appwrite ===
print("📦 Fetching images...")
file_list = storage.list_files(BUCKET_ID)
print(f"📁 Found {len(file_list['files'])} files.")

# === Tag each image ===
for file in file_list["files"]:
    file_id = file["$id"]
    file_name = file["name"]
    print(f"🔍 Tagging {file_name} (ID: {file_id})...")

    try:
                # Download and prepare image (preprocessing for NHWC ONNX layout)
        image_bytes = storage.get_file_download(BUCKET_ID, file_id)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        # Convert RGB → BGR (some ONNX models trained with OpenCV expect this)
        image = image[:, :, ::-1]

        # Pad to square and resize using OpenCV
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

        # Expand dims for ONNX model: [1, H, W, C]
        tensor = np.expand_dims(image, axis=0)
        print(f"📐 Input tensor shape: {tensor.shape}")



        # Predict
        probs = model.run([output_name], {input_name: tensor})[0][0]

        # Group tags
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

        # Save locally
        full_data = {
            "imageId": file_id,
            "fileName": file_name,
            "tags": all_tags,
            "booru": booru,
            "groupedTags": grouped
        }
        tagged_by_id[file_id] = full_data
        with open(TAG_PATH, "w") as f:
            json.dump(list(tagged_by_id.values()), f, indent=2)
        print("✅ Tags cached.")

        # Sync to Appwrite (basic fields only)
        try:
            minimal = {k: full_data[k] for k in ("imageId", "fileName", "tags")}
            existing = databases.list_documents(DATABASE_ID, COLLECTION_ID, [Query.equal("imageId", file_id)])
            if existing["total"] > 0:
                doc_id = existing["documents"][0]["$id"]
                databases.update_document(DATABASE_ID, COLLECTION_ID, doc_id, data=minimal)
                print("🔁 Updated document.")
            else:
                databases.create_document(DATABASE_ID, COLLECTION_ID, "unique()", data=minimal)
                print("📤 Created new document.")
        except Exception as e:
            print(f"⚠️ Appwrite sync failed: {e}")
            traceback.print_exc()

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
        traceback.print_exc()
