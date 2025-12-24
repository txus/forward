# CUDA Profiling Guide

## Build Configurations

### Debug (Full Debug Info)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```
- **-G flag**: Device debug info (disables optimizations, step through kernels)
- **-g flag**: Host debug info
- **-lineinfo**: Line-level profiling in NCU
- Use for: Debugging incorrect results, stepping through kernels

### RelWithDebInfo (Recommended for Profiling)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```
- **-G flag**: Device debug info
- **-g flag**: Host debug info
- **-lineinfo**: Line-level profiling
- **--use_fast_math**: Optimizations enabled
- Use for: Performance profiling with NCU/Nsight while seeing source code

### Release (Maximum Performance)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
- **-lineinfo only**: Minimal profiling info (function names only)
- **--use_fast_math**: Full optimizations
- Use for: Production, final benchmarks

## Profiling Tools

### Nsight Compute (NCU) - Kernel Profiler

Profile a single kernel in detail:
```bash
# Basic profiling
ncu ./build/apps/forward

# Profile specific kernel
ncu --kernel-name "my_kernel" ./build/apps/forward

# Full metrics (slow but comprehensive)
ncu --set full ./build/apps/forward

# Launch count (profile first 10 launches)
ncu --launch-count 10 ./build/apps/forward

# Export report
ncu -o profile_report ./build/apps/forward
```

**What NCU shows:**
- Memory throughput (global, shared, L1/L2 cache)
- Compute throughput (FP32, FP16, INT)
- Warp execution efficiency
- Occupancy
- Bottleneck analysis
- Source code line-by-line metrics (with -G flag)

### Nsight Systems (nsys) - Timeline Profiler

Profile entire application timeline:
```bash
# Basic timeline
nsys profile ./build/apps/forward

# Include CPU sampling
nsys profile --sample=cpu ./build/apps/forward

# Export report
nsys profile -o timeline ./build/apps/forward

# View report (GUI required)
nsys-ui timeline.nsys-rep
```

**What Nsys shows:**
- CPU/GPU timeline
- Kernel launch overhead
- Memory transfers (H2D, D2H)
- CPU thread activity
- API calls (CUDA, cuBLAS, etc.)

## Quick Workflow

### 1. Find bottlenecks (Nsight Systems)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
nsys profile -o timeline ./build/apps/forward
```
Look for: kernel durations, memory transfer overhead, CPU gaps

### 2. Optimize slow kernels (Nsight Compute)
```bash
ncu --set full --kernel-name "slow_kernel" ./build/apps/forward
```
Look for: occupancy, memory throughput, warp divergence

### 3. Verify optimizations (Release build)
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
nsys profile ./build/apps/forward
```
Compare timeline before/after

## CUDA Flags Explained

| Flag | Purpose | Build Types | Impact |
|------|---------|-------------|--------|
| `-lineinfo` | Line-level profiling | All | Small binary size increase |
| `-G` | Device debug symbols | Debug, RelWithDebInfo | Large binary, slower execution |
| `-g` | Host debug symbols | Debug, RelWithDebInfo | Moderate binary increase |
| `--use_fast_math` | Fast math operations | RelWithDebInfo, Release | Faster but less precise |

## Common NCU Metrics

- **SM %**: GPU utilization
- **Memory Throughput**: GB/s achieved vs theoretical max
- **Achieved Occupancy**: Active warps vs max warps
- **Warp Execution Efficiency**: Non-divergent warps %
- **L1/L2 Cache Hit Rate**: % of memory accesses hitting cache

## Tips

1. **Always profile with RelWithDebInfo** for best results (debug info + optimizations)
2. **Profile on real data sizes** - small test inputs don't show real bottlenecks
3. **Run multiple times** - first run includes CUDA initialization overhead
4. **Check GPU clock speed** - may throttle if overheating
5. **Use `--launch-count`** - profile only first few kernel launches to save time

## Example: Profiling a matmul kernel

```bash
# Build with debug info
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build

# Get timeline
nsys profile -o matmul_timeline ./build/apps/forward

# Profile matmul kernel specifically
ncu --kernel-name "matmul" --set full -o matmul_profile ./build/apps/forward

# View source-level metrics (GUI)
ncu-ui matmul_profile.ncu-rep
```
