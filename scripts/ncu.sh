#!/usr/bin/env bash
set -euo pipefail

TARGET="$1"
KERNEL="$2"

rm -f "${TARGET}.ncu-rep"

echo $TARGET
echo $KERNEL

cmake --preset ninja-nvcc -DCMAKE_BUILD_TYPE=Release
cmake --build build --target test_$TARGET
ncu --set full --kernel-name $KERNEL -o "$TARGET" ./build/tests/$TARGET/test_$TARGET
