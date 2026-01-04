#pragma once

// NVTX profiling utilities
// Zero-overhead when not profiling - NVTX calls are no-ops unless a profiler is attached

#ifdef BACKEND_CUDA
#include <nvtx3/nvToolsExt.h>

// RAII wrapper for NVTX ranges using the C API
class NvtxRange {
public:
  explicit NvtxRange(const char* name) { nvtxRangePushA(name); }
  ~NvtxRange() { nvtxRangePop(); }
  NvtxRange(const NvtxRange&) = delete;
  NvtxRange& operator=(const NvtxRange&) = delete;
};

// Create an NVTX range that lasts for the current scope
#define NVTX_RANGE(name) NvtxRange nvtx_range_##__LINE__{name}

#else

#define NVTX_RANGE(name)

#endif
