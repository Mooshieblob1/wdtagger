# wdtagger

`wdtagger` is a local image tagging tool that uses `onnxruntime` and HuggingFace-hosted ONNX models (e.g. `wd-vit-tagger-v3`) to generate Danbooru-style tags for images stored in a BlobPics (Cloudflare Worker + R2) bucket.
It downloads the model and tag CSV on first run, performs OpenCV-based preprocessing, and applies a configurable confidence threshold.

Each tagged image is saved as a local `.json` file and automatically uploaded to your BlobPics backend via a `POST /upload-json` API.

---

## 📦 Requirements

* Python 3.11+
* `uv` package manager (or pip)
* Access to a BlobPics site (Cloudflare Worker + R2, with `/list`, `/original/`, `/upload-json`)
* Internet access (for HuggingFace model download)
* CUDA-compatible GPU (optional)

---

## 🛠 Setup

```bash
git clone https://github.com/Mooshieblob1/wdtagger.git
cd wdtagger
python3 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Set your **BlobPics domain** in the script (`test_tagger.py`):

```python
DOMAIN = "blobpics.tech"
```

---

## 🚀 Run Tagger

Run the script to fetch all images, tag them, and upload tag data to your site:

```bash
python test_tagger.py
```

This will:

* Get image IDs from your BlobPics `/list` endpoint
* Download each image and generate tags
* Save `.json` files locally (one per image)
* POST tags to your BlobPics `/upload-json` API

---

## 🔁 Persistent Tagging Loop

To continuously tag new images every 60 minutes:

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
DOMAIN = "blobpics.tech"                # Your BlobPics site
TAG_THRESHOLD = 0.5                     # Minimum confidence to keep tag
TAG_BLACKLIST = ["blue_skin"]           # Tags to ignore completely
```

---

## 📁 Files

| File                   | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `test_tagger.py`       | Tags images from BlobPics & uploads tags via API |
| `persistent_runner.sh` | Optional: loops tagging every hour               |
| `requirements.txt`     | Dependencies for the project                     |
| `tagged_images.json`   | Output tag cache (local copy)                    |

---

## 📄 License

MIT — use freely with credit.
```