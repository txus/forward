#!/usr/bin/env bash
set -e

shopt -s globstar

uv tool run clangd-tidy -p build apps/**/*.cpp src/**/*.cpp tests/**/*.cpp include/**/*.hpp

echo "All good"
