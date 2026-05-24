#!/usr/bin/env bash
# Run clang-tidy over our sources using the build's compile_commands.json.
# Uses the toolchain-matching clang-tidy: Homebrew llvm@20 on macOS, PATH on Linux.
set -euo pipefail

if [[ "$(uname -s)" == "Darwin" ]]; then
  TIDY="${HOMEBREW_PREFIX:-/opt/homebrew}/opt/llvm@20/bin/clang-tidy"
else
  TIDY="$(command -v clang-tidy)"
fi

if [[ ! -x "$TIDY" ]]; then
  echo "clang-tidy not found at: $TIDY" >&2
  echo "macOS: brew install llvm@20   Linux: ensure clang-tools is in the nix shell" >&2
  exit 1
fi

if [[ ! -f build/compile_commands.json ]]; then
  echo "build/compile_commands.json missing. Run: cmake --preset \$(uname -s | grep -q Darwin && echo macos || echo linux)" >&2
  exit 1
fi

# Lint our own sources (headers are pulled in transitively and filtered by
# HeaderFilterRegex in .clang-tidy). Explicit files/args win; otherwise lint all
# tracked src/apps .cpp files.
if [[ $# -gt 0 ]]; then
  exec "$TIDY" -p build "$@"
fi
# Avoid `mapfile` -- macOS ships bash 3.2 which lacks it. Our paths have no
# spaces, so word-splitting the git output is safe here.
# shellcheck disable=SC2046
exec "$TIDY" -p build $(git ls-files 'src/*.cpp' 'apps/*.cpp')
