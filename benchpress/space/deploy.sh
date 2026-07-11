#!/usr/bin/env bash
# Deploy the Composer Space: stage app + SDK + data, upload to a HF Space repo.
# Usage: ./space/deploy.sh <username>/<space-name>   (requires `hf auth login` with write scope)
set -euo pipefail

SPACE_ID="${1:?usage: deploy.sh <username>/<space-name>}"
BENCH="$(cd "$(dirname "$0")/.." && pwd)"   # benchpress/
ROOT="$(cd "$BENCH/.." && pwd)"             # repo root
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$BENCH"/space/app.py "$BENCH"/space/README.md "$BENCH"/space/requirements.txt "$STAGE/"
if [ -f "$BENCH"/space/examples.json ]; then
  cp "$BENCH"/space/examples.json "$STAGE/"
fi
if [ -f "$BENCH"/space/tag_map.json ]; then
  cp "$BENCH"/space/tag_map.json "$STAGE/"
fi
# app.py imports `benchpress_hub` (public name) and `autotagging_loop.runner.hf_sampling`.
mkdir -p "$STAGE/benchpress_hub" "$STAGE/autotagging_loop/runner" "$STAGE/data"
cp "$BENCH"/benchpress_hub/*.py "$STAGE/benchpress_hub/"
cp "$ROOT"/autotagging_loop/__init__.py "$STAGE/autotagging_loop/"
cp "$ROOT"/autotagging_loop/runner/__init__.py "$ROOT"/autotagging_loop/runner/hf_sampling.py "$STAGE/autotagging_loop/runner/"
cp "$ROOT"/data/hf_dataset_map.json "$ROOT"/data/leaderboard_scores.json \
   "$ROOT"/data/cognitive_abilities.json "$STAGE/data/"

python3 - "$SPACE_ID" "$STAGE" <<'EOF'
import sys
from huggingface_hub import HfApi

space_id, stage = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
api.upload_folder(folder_path=stage, repo_id=space_id, repo_type="space",
                  commit_message="Deploy BenchPress Composer")
print(f"https://huggingface.co/spaces/{space_id}")
EOF
