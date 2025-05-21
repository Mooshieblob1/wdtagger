#!/bin/bash

# Move to script directory
cd "$(dirname "$0")"
source .venv/bin/activate

echo "🔁 Starting BlobPics (Cloudflare R2) tagging loop..."

while true; do
  echo "⏱️ $(date): Running tagger for BlobPics images..."
  python test_tagger.py
  echo "✅ Tagging done. Sleeping for 60 minutes..."
  sleep 3600
done
