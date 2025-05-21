# wdtagger

`wdtagger` is a local image tagging tool that uses `onnxruntime` and HuggingFace-hosted ONNX models (e.g. `wd-eva02-large-tagger-v3`, `wd-vit-tagger-v3`) to generate Danbooru-style tags for images stored in an Appwrite bucket. It downloads the model and tag CSV on first run, performs OpenCV-based preprocessing to ensure correct input shape, and applies a configurable confidence threshold.

Each tagged image is stored in a local cache (`tagged_images.json`) and uploaded to your Appwrite database collection.

---

## 📦 Requirements

- Python 3.11+
- `uv` package manager (or pip)
- Appwrite project + API key with access to your image bucket
- Internet access (for HuggingFace model download)
- CUDA-compatible GPU (optional)

---

## 🛠 Setup

```bash
git clone https://github.com/Mooshieblob1/wdtagger.git
cd wdtagger
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create your `.env` file:

```env
APPWRITE_API_KEY=your_appwrite_api_key
```

---

## 🚀 One-time Tagging

Run the script to tag all untagged images in the Appwrite bucket:

```bash
python test_tagger.py
```

This will:

- Save tags to `tagged_images.json`
- Upload results to your Appwrite database collection (`imageTags`)

---

## 🔁 Persistent Tagging Loop

To continuously tag every 60 minutes:

```bash
./persistent_runner.sh
```

Or run in the background:

```bash
nohup ./persistent_runner.sh > tagger.log 2>&1 &
```

---

## ♻️ Reprocess All Images

If you change your `TAG_THRESHOLD` and want to regenerate all tags:

```bash
rm tagged_images.json
python test_tagger.py
```

---

## ⚙️ Configuration Options

Edit the top of `test_tagger.py`:

```python
REPO = "SmilingWolf/wd-vit-tagger-v3"  # HuggingFace model
TAG_THRESHOLD = 0.5                    # Minimum confidence to keep tag
TAG_BLACKLIST = ["blue_skin"]         # Tags to ignore completely
```

---

## 📁 Files

| File                  | Purpose                                           |
|-----------------------|---------------------------------------------------|
| `test_tagger.py`      | Tags images from Appwrite bucket & uploads to DB |
| `persistent_runner.sh`| Optional: loops tagging every hour               |
| `requirements.txt`    | Dependencies for the project                      |
| `tagged_images.json`  | Output tag cache (local copy)                     |
| `.env`                | Contains your Appwrite API key                    |

---

## 📄 License

MIT — use freely with credit.
