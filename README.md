# wdtagger

`wdtagger` is a local image tagging tool using `timm` and the `wd-eva02-large-tagger-v3` model to generate Danbooru-style tags for images stored in an Appwrite bucket. It supports offline tag label mapping and can run persistently on a loop.

---

## 📦 Requirements

- Python 3.11+
- `uv` package manager (or pip)
- Appwrite project + API key with access to your image bucket
- `id2label.json` in your project root (generated from model CSV or downloaded)
- CUDA-compatible GPU (optional, but recommended)

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

Make sure your `id2label.json` is in the same directory.

---

## 🚀 One-time Tagging

Run the script to tag all untagged images in the Appwrite bucket:

```bash
python test_tagger.py
```

This will append results to `tagged_images.json`.

---

## 🔁 Persistent Tagging Loop

To continuously tag every 60 minutes:

```bash
./persistent_runner.sh
```

Or run in the background with:

```bash
nohup ./persistent_runner.sh > tagger.log 2>&1 &
```

---

## ♻️ Reprocess All Images

If you change your `TAG_THRESHOLD` or `MAX_TAGS`, and want to regenerate all tags:

```bash
rm tagged_images.json
python test_tagger.py
```

---

## ⚙️ Configuration Options

Edit the top of `test_tagger.py`:

```python
TAG_THRESHOLD = 0.5  # Minimum confidence score to include a tag
MAX_TAGS = 100       # Max tags saved per image
```

---

## 🌐 Sync Output to Server (Optional)

Append this to `persistent_runner.sh` to upload results:

```bash
curl -X POST https://yourdomain.com/api/upload-tags \
     -H "Content-Type: application/json" \
     --data-binary "@tagged_images.json"
```

---

## 📁 Files

| File                  | Purpose                                  |
|-----------------------|------------------------------------------|
| `test_tagger.py`      | Tags images from Appwrite bucket         |
| `persistent_runner.sh`| Loops tagging every hour                 |
| `requirements.txt`    | Dependencies for the project             |
| `tagged_images.json`  | Output tag cache                         |
| `id2label.json`       | Label mapping for model outputs          |
| `.env`                | Contains your Appwrite API key           |

---

## 📄 License

MIT — use freely with credit.
