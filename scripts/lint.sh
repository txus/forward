#!/bin/bash
set -e

uv tool run clangd-tidy -p build apps/**/*.cpp src/**/*.cpp tests/**/*.cpp include/**/*.hpp
