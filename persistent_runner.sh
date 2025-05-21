#!/bin/bash

cd "$(dirname "$0")"
source .venv/bin/activate

echo "🔁 Starting local tagging loop..."

while true; do
  echo "⏱️ $(date): Running tagger..."
  python test_tagger.py
  echo "✅ Done. Sleeping for 60 minutes..."
  sleep 3600
done
