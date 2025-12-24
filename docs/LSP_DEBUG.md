# CUDA clangd LSP Configuration

## Overview

Getting clangd LSP working properly for CUDA files (.cu, .cuh) on NixOS requires solving three key problems:

1. **Device vs Host analysis** - clangd only does single-pass analysis, but CUDA needs both
2. **NixOS resource-dir mismatch** - clangd uses different Clang headers than clang++
3. **CUDA path configuration** - CMake must find the correct CUDA package with runtime headers

## Solution Summary

### Prerequisites

1. Generate clangd-specific build configuration:
   ```bash
   cmake --preset ninja-clangd
   ```

2. Ensure shell.nix exports required environment variables (see below).

### Files Modified

| File | Purpose |
|------|---------|
| `.clangd` | clangd configuration with PathMatch rules |
| `CMakePresets.json` | Added `CLANG_RESOURCE_DIR` cache variable |
| `shell.nix` | Export `CLANG_RESOURCE_DIR` environment variable |
| `src/tensor/cuda/CMakeLists.txt` | Pass `-resource-dir` flag when building with Clang |

### Key Configuration

#### shell.nix additions:
```nix
# Clang resource directory for clangd (NixOS-specific)
export CLANG_RESOURCE_DIR="$(clang++ -print-resource-dir)"
```

#### CMakePresets.json (ninja-clangd preset):
```json
{
  "cacheVariables": {
    "CUDAToolkit_ROOT": "$env{CUDA_PATH}",
    "CLANG_RESOURCE_DIR": "$env{CLANG_RESOURCE_DIR}"
  }
}
```

#### src/tensor/cuda/CMakeLists.txt:
```cmake
elseif (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  set(CLANG_CUDA_FLAGS -fPIC)
  if(DEFINED ENV{CUDA_PATH})
    list(APPEND CLANG_CUDA_FLAGS --cuda-path=$ENV{CUDA_PATH})
  endif()
  if(DEFINED CLANG_RESOURCE_DIR AND NOT CLANG_RESOURCE_DIR STREQUAL "")
    list(APPEND CLANG_CUDA_FLAGS -resource-dir=${CLANG_RESOURCE_DIR})
  endif()
  target_compile_options(tensor_cuda PRIVATE ${CLANG_CUDA_FLAGS})
endif()
```

#### .clangd:
```yaml
CompileFlags:
  CompilationDatabase: build/clangd

# Kernel files (.cu in kernels/) - device code
---
If:
  PathMatch: .*/kernels/.*\.cu$
CompileFlags:
  Remove:
    - --expt-relaxed-constexpr
    - -Xcompiler=*
    - -G
  Add:
    - -xcuda
    - --cuda-gpu-arch=sm_120
    - --cuda-device-only
Diagnostics:
  Suppress:
    - variadic_device_fn
    - attributes_not_allowed

# Other .cu files - host code (kernel launchers, memory management)
---
If:
  PathMatch: .*\.cu$
  PathExclude: .*/kernels/.*
CompileFlags:
  Remove:
    - --expt-relaxed-constexpr
    - -Xcompiler=*
    - -G
  Add:
    - -xcuda
    - --cuda-gpu-arch=sm_120
    - --cuda-host-only
Diagnostics:
  Suppress:
    - variadic_device_fn
    - attributes_not_allowed
```

## Problem Details

### Problem 1: Device vs Host Analysis

CUDA compilation is a two-pass process:
1. **Device pass** (nvptx64 target) - compiles `__global__`/`__device__` functions
2. **Host pass** (x86_64 target) - compiles regular C++ and kernel launch code

clangd only performs single-pass analysis. By default, it chooses device mode for .cu files, which causes false errors for host-side code like:
- Kernel launch syntax `<<<grid, block>>>(...)`
- CUDA runtime functions (`cudaGetLastError`, etc.)
- Host-only C++ features (`std::runtime_error`, etc.)

**Solution:** Use PathMatch to differentiate:
- `kernels/*.cu` → `--cuda-device-only` (actual GPU kernels)
- Other `*.cu` → `--cuda-host-only` (kernel launchers, memory management)

### Problem 2: NixOS Resource Directory Mismatch

On NixOS, the `clang-tools` package provides clangd, but it's a separate derivation from the `clang` package. They have different resource directories:

```
clangd:     /nix/store/xxx-clang-20.1.8/lib/clang/20/       # Doesn't exist!
clang++:    /nix/store/yyy-clang-wrapper-20.1.8/resource-root/  # Has CUDA headers
```

clangd's default resource-dir doesn't include CUDA builtin headers like:
- `__clang_cuda_builtin_vars.h` (defines `threadIdx`, `blockIdx`, etc.)
- `__clang_cuda_runtime_wrapper.h`

**Solution:** Pass the correct resource-dir via compile_commands.json:
1. Export `CLANG_RESOURCE_DIR=$(clang++ -print-resource-dir)` in shell.nix
2. Pass it to CMake as a cache variable
3. CMake adds `-resource-dir=<path>` to CUDA compile flags

### Problem 3: CUDA Toolkit Path

NixOS has multiple CUDA packages:
- `cuda_nvcc` - Just the nvcc compiler (no runtime headers)
- `cuda-merged` - Complete CUDA toolkit with all headers

CMake's `find_package(CUDAToolkit)` might find `cuda_nvcc` first, which lacks `cuda_runtime.h`.

**Solution:** Set `CUDAToolkit_ROOT` in CMakePresets.json:
```json
"CUDAToolkit_ROOT": "$env{CUDA_PATH}"
```

Where `CUDA_PATH` points to `cuda-merged` in shell.nix.

## Verification

Test clangd analysis on each file type:

```bash
# Device code (kernel files)
clangd --check=src/tensor/cuda/kernels/fill.cu

# Host code (storage management)
clangd --check=src/tensor/cuda/storage.cu

# CPU code (should work out of box)
clangd --check=src/tensor/cpu/ops.cpp
```

All should report 0 errors.

## Known Limitations

1. **IncludeCleaner warnings** - clangd reports many "Failed to get an entry for resolved path" warnings for CUDA internal headers. These are harmless noise.

2. **Suppressed diagnostics** - `variadic_device_fn` and `attributes_not_allowed` are suppressed because clangd incorrectly flags valid CUDA syntax.

3. **Single-pass analysis** - clangd can only analyze device OR host code per file, not both. Our PathMatch rules assume:
   - Files in `kernels/` directory contain actual GPU kernels (device code)
   - Other .cu files contain kernel launchers and host code

## Portability

This configuration is fully portable:
- No hardcoded `/nix/store/` paths in version-controlled files
- Environment variables come from shell.nix or equivalent
- Works on any system where clang++ has CUDA support and exports correct resource-dir
