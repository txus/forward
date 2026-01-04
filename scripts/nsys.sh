#!/usr/bin/env bash
set -euo pipefail

OUTPUT="run1"

rm -f "${OUTPUT}.nsys-rep" "${OUTPUT}.sqlite"

nsys profile \
    --trace=nvtx,cuda \
    -o "$OUTPUT" \
    --force-overwrite true \
    ./build/apps/forward --cuda

echo ""
echo "=== NVTX Summary ==="
nsys stats "${OUTPUT}.nsys-rep" 2>&1 | grep -A 20 "NVTX Range Summary"
