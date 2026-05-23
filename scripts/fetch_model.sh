#!/bin/bash
# Download the Llama-3.2-1B model from HuggingFace and symlink it into tests/model.
# Requires `huggingface-cli login` with access to meta-llama/Llama-3.2-1B.
set -euo pipefail

MODEL="meta-llama/Llama-3.2-1B"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LINK="$REPO_ROOT/tests/model"

if [ -e "$LINK" ] && [ -f "$LINK/config.json" ] && ls "$LINK"/*.safetensors >/dev/null 2>&1; then
    echo "Model already present at $LINK"
    exit 0
fi

if command -v hf >/dev/null 2>&1; then
    HF_CMD="hf download"
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_CMD="huggingface-cli download"
else
    echo "Neither 'hf' nor 'huggingface-cli' found. Install with: brew install huggingface-cli  (or: pip install huggingface_hub)" >&2
    exit 1
fi

SNAPSHOT_DIR="$($HF_CMD "$MODEL" | tail -n 1)"
ln -sfn "$SNAPSHOT_DIR" "$LINK"
echo "Linked $LINK -> $SNAPSHOT_DIR"
