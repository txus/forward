#!/bin/bash
# Format all C++ files
find src include -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
echo "Formatted all C++ files"
