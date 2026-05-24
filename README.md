# forward

[![CI](https://github.com/txus/forward/actions/workflows/ci.yml/badge.svg)](https://github.com/txus/forward/actions/workflows/ci.yml)

A simple inference engine. An excuse to learn C++ really.

Requirements:

* CMake 3.11 or better; 3.14+ highly recommended.
* A C++26 compatible compiler (tested with Clang 20 and GCC 14).
* Git
* Doxygen
* OpenMP (for CPU) (`brew install libomp` on Mac)

To use clang: just export these before you run cmake:

```
export CC=clang
export CXX=clang++
```

To configure:

```
cmake -S . -B build
```

Add `-GNinja` if you have Ninja.

To build:

```
cmake --build build
```

To test (`--target` can be written as `-t` in CMake 3.15+):

```
cmake --build build --target test
```

To build docs (requires Doxygen, output in `build/docs/html`):

```
cmake --build build --target docs
```

## Editor / clangd setup

`.clangd` is **not** committed — it's a per-machine symlink to one of two
checked-in variants, because clangd needs different config per backend:

* **`.clangd.cuda`** (Linux / NVIDIA): the production build compiles `.cu` with
  `nvcc`, which clangd can't parse, so this points clangd at the separate
  clang-based `build/clangd` database (`make dx`) and adds CUDA flag handling
  for `.cu`/`.cuh`.
* **`.clangd.metal`** (macOS / Apple): no `nvcc`, so the regular build's
  `compile_commands.json` is already clang-based. Points clangd at `build/`; no
  `make dx` needed.

Symlink the one for your machine, from the repo root:

```
ln -s .clangd.cuda .clangd    # Linux / NVIDIA
ln -s .clangd.metal .clangd   # macOS / Apple
```

To use an IDE, such as Xcode:

```
cmake -S . -B xbuild -GXcode
cmake --open xbuild
```

The CMakeLists show off several useful design patterns for CMake.
