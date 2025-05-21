import os
import json
import torch
import timm
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
from torchvision import transforms
from appwrite.client import Client
from appwrite.services.storage import Storage
from appwrite.services.databases import Databases

# === Load .env ===
load_dotenv()
APPWRITE_API_KEY = os.getenv("APPWRITE_API_KEY")

# === Appwrite Constants ===
APPWRITE_ENDPOINT = "https://syd.cloud.appwrite.io/v1"
APPWRITE_PROJECT_ID = "682b826b003d9cba9018"
BUCKET_ID = "682cfa1a0016991596f5"
DATABASE_ID = "682b89cc0016319fcf30"
COLLECTION_ID = "682d7b240022ba63cd02"
TAG_PATH = "tagged_images.json"

# === Configurable Threshold and Tag Limit ===
TAG_THRESHOLD = 0.35
MAX_TAGS = 20

# === Initialize Appwrite ===
client = Client()
client.set_endpoint(APPWRITE_ENDPOINT)
client.set_project(APPWRITE_PROJECT_ID)
client.set_key(APPWRITE_API_KEY)

storage = Storage(client)
databases = Databases(client)

# === Load model ===
print("🔄 Loading wd-eva02 model from timm...")
model = timm.create_model("hf_hub:SmilingWolf/wd-eva02-large-tagger-v3", pretrained=True)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ Model loaded on {device}")

# === Load id2label.json (offline) ===
print("📄 Loading local id2label.json...")
id2label = {}
try:
    with open("id2label.json", "r") as f:
        id2label_raw = json.load(f)
        id2label = {int(k): v for k, v in id2label_raw.items()}
except Exception as e:
    print(f"⚠️ Failed to load local id2label.json: {e}")

# === Preprocessing for 448x448 ===
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# === Load tag cache ===
if os.path.exists(TAG_PATH):
    with open(TAG_PATH, "r") as f:
        tagged = json.load(f)
else:
    tagged = []

tagged_ids = {entry["imageId"] for entry in tagged}

# === Fetch images ===
print("📦 Fetching image list from Appwrite...")
file_list = storage.list_files(BUCKET_ID)
print(f"Found {len(file_list['files'])} files.")

# === Tagging loop ===
for file in file_list["files"]:
    file_id = file["$id"]
    file_name = file["name"]

    if file_id in tagged_ids:
        print(f"⏩ Skipping {file_name} (already tagged)")
        continue

    print(f"🔍 Tagging {file_name}...")

    try:
        image_bytes = storage.get_file_download(BUCKET_ID, file_id)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)[0]

        if not id2label:
            print("⚠️ Skipping tagging because id2label is unavailable.")
            continue

        tags = [
            (id2label[i], round(probs[i].item(), 4))
            for i in range(len(probs))
            if probs[i].item() > TAG_THRESHOLD
        ]
        tags.sort(key=lambda x: -x[1])

        print("📝 Tags:")
        for tag, score in tags[:20]:
            print(f" - {tag}: {score}")

        tag_data = {
            "imageId": file_id,
            "fileName": file_name,
            "tags": [tag for tag, _ in tags[:MAX_TAGS]]
        }

        tagged.append(tag_data)

        with open(TAG_PATH, "w") as f:
            json.dump(tagged, f, indent=2)

        print("✅ Tags saved locally.")

        # === Upload to Appwrite DB ===
        try:
            databases.create_document(
                database_id=DATABASE_ID,
                collection_id=COLLECTION_ID,
                document_id="unique()",
                data=tag_data
            )
            print("📤 Uploaded tags to Appwrite database.\n")
        except Exception as e:
            print(f"⚠️ Failed to upload tags to Appwrite: {e}\n")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")
