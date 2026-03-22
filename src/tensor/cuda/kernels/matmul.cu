#include "matmul.cuh"
#include "utils.cuh"
#include <cublas_v2.h>
#include <fmt/core.h>
#include "kittens.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace tensor::kernels {

using namespace dtype;

// cuBLAS handle management
//
// Industry practices for cuBLAS handle management:
//
// 1. Thread-local singleton (used here): Each thread gets its own handle, lazily
//    initialized. Simple and works well for most cases. Handles are automatically
//    cleaned up when threads exit.
//
// 2. Global singleton with mutex: One handle shared across all threads. cuBLAS
//    handles are thread-safe, but this can cause contention. Used in simpler apps.
//
// 3. Handle pool: Pre-create N handles, threads check them out/in. Good for
//    high-throughput servers with many threads.
//
// 4. Context/Session object: User creates a "Session" that owns the handle,
//    passes it to all ops. Most explicit, used in TensorFlow/PyTorch internals.
//
// 5. Per-stream handles: One handle per CUDA stream for maximum concurrency.
//    Used in highly optimized inference engines.
//
// We use thread-local here because:
// - Zero contention between threads
// - Lazy initialization (no cost if thread doesn't use cuBLAS)
// - Automatic cleanup
// - Simple API (no handle passing required)

// Check if a 2D tensor is a simple transpose (stride pattern: [1, cols] instead of [cols, 1])
template <typename T, typename D>
bool is_2d_transpose(const TensorView<T, D>& view) {
  if (view.shape.size() != 2) return false;
  // A transposed 2D matrix has stride[0] = 1, stride[1] = shape[0]
  // (stepping through rows goes by 1, stepping through cols jumps by original row count)
  return view.stride[0] == 1 && view.stride[1] == view.shape[0];
}

#ifdef SHITTY

// cuBLAS uses column-major layout, but our tensors are row-major.
// The trick: C = A @ B in row-major is equivalent to C^T = B^T @ A^T in col-major.
// Since the transpose just changes how we interpret the memory layout:
//   - We swap A and B
//   - We swap M and N
// This gives us the correct result without any actual transposition.
//
// For transposed B (common in Linear layers where we store weights as [out, in]
// and want to compute input @ weights.T), we use CUBLAS_OP_T to let cuBLAS
// handle the transpose without copying data.


class CublasHandle {
public:
  static cublasHandle_t get() {
    thread_local CublasHandle instance;
    return instance.handle_;
  }

private:
  cublasHandle_t handle_;

  CublasHandle() {
    cublasStatus_t status = cublasCreate(&handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(fmt::format("cuBLAS initialization failed: {}", static_cast<int>(status)));
    }
    warmup();
  }

  // Warmup bf16 GEMM kernels to avoid JIT compilation on first real call
  void warmup() {
    constexpr int N = 64;
    void* a = nullptr;
    void* b = nullptr;
    void* c = nullptr;
    cudaMalloc(&a, N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&b, N * N * sizeof(__nv_bfloat16));
    cudaMalloc(&c, N * N * sizeof(__nv_bfloat16));

    float alpha = 1.0f, beta = 0.0f;

    // Warmup single GEMM
    cublasGemmEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                 b, CUDA_R_16BF, N, a, CUDA_R_16BF, N, &beta,
                 c, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    // Warmup batched GEMM
    cublasGemmStridedBatchedEx(handle_, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                               b, CUDA_R_16BF, N, N * N,
                               a, CUDA_R_16BF, N, N * N, &beta,
                               c, CUDA_R_16BF, N, N * N, 2,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

    cudaDeviceSynchronize();
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
  }

  ~CublasHandle() {
    cublasDestroy(handle_);
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
};

template <>
Tensor<bfloat16, CUDA> matmul(const TensorView<bfloat16, CUDA>& tensor_a,
                               const TensorView<bfloat16, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && "tensor A must be contiguous");

  size_t a_ndim = tensor_a.shape.size();
  size_t b_ndim = tensor_b.shape.size();

  assert(a_ndim >= 2 && b_ndim >= 2);

  // Check if B is a 2D transpose - if so, we use CUBLAS_OP_T
  bool b_transposed = is_2d_transpose(tensor_b);
  if (!b_transposed) {
    assert(tensor_b.is_contiguous() && "tensor B must be contiguous (or a 2D transpose)");
  }

  size_t M = tensor_a.shape[a_ndim - 2];
  size_t K = tensor_a.shape[a_ndim - 1];
  size_t N = tensor_b.shape[b_ndim - 1];

  assert(K == tensor_b.shape[b_ndim - 2] && "Inner dimensions must match");

  // Calculate batch size from A's leading dimensions
  size_t batch_size = 1;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    batch_size *= tensor_a.shape[i];
  }

  // Build output shape
  Shape out_shape;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    out_shape.push_back(tensor_a.shape[i]);
  }
  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor<bfloat16, CUDA> out{out_shape};

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t handle = CublasHandle::get();

  // For row-major: C = A @ B becomes C^T = B^T @ A^T in col-major
  // We swap A and B in the cuBLAS call.
  //
  // If B is already transposed (view-only, no data copy), we need to "undo" it
  // for cuBLAS by using CUBLAS_OP_T. The physical layout is [N, K] but the view
  // shape is [K, N]. cuBLAS sees it as col-major [K, N], and with OP_T treats it
  // as [N, K] which is what we want.

  // When B is transposed:
  // - Physical data is [N, K] (original weights before transpose view)
  // - View shape is [K, N] (after .transpose())
  // - ldb = N (the leading dimension of the physical layout)
  // - We use CUBLAS_OP_T so cuBLAS reads it as transposed

  cublasOperation_t op_b = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = b_transposed ? static_cast<int>(K) : static_cast<int>(N);

  if (batch_size == 1) {
    // Single matrix multiplication
    cublasStatus_t status = cublasGemmEx(
        handle,
        op_b, CUBLAS_OP_N,
        static_cast<int>(N),       // rows of op(B) and C
        static_cast<int>(M),       // cols of op(A) and C
        static_cast<int>(K),       // cols of op(B), rows of op(A)
        &alpha,
        tensor_b.data, CUDA_R_16BF, ldb,
        tensor_a.data, CUDA_R_16BF, static_cast<int>(K),  // A: lda = K
        &beta,
        out.data(), CUDA_R_16BF, static_cast<int>(N),     // C: ldc = N
        CUBLAS_COMPUTE_32F,        // Accumulate in fp32
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(fmt::format("cuBLAS GEMM failed: {}", static_cast<int>(status)));
    }
  } else {
    // Batched matrix multiplication
    // Note: batched with transposed B not yet supported
    assert(!b_transposed && "Batched matmul with transposed B not yet implemented");

    long long int stride_a = static_cast<long long int>(M * K);
    long long int stride_b = (b_ndim > 2) ? static_cast<long long int>(K * N) : 0;
    long long int stride_c = static_cast<long long int>(M * N);

    cublasStatus_t status = cublasGemmStridedBatchedEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        tensor_b.data, CUDA_R_16BF, static_cast<int>(N), stride_b,
        tensor_a.data, CUDA_R_16BF, static_cast<int>(K), stride_a,
        &beta,
        out.data(), CUDA_R_16BF, static_cast<int>(N), stride_c,
        static_cast<int>(batch_size),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(fmt::format("cuBLAS batched GEMM failed: {}", static_cast<int>(status)));
    }
  }

  return out;
}

template <>
Tensor<float, CUDA> matmul(const TensorView<float, CUDA>& tensor_a,
                            const TensorView<float, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && "tensor A must be contiguous");

  size_t a_ndim = tensor_a.shape.size();
  size_t b_ndim = tensor_b.shape.size();

  assert(a_ndim >= 2 && b_ndim >= 2);

  bool b_transposed = is_2d_transpose(tensor_b);
  if (!b_transposed) {
    assert(tensor_b.is_contiguous() && "tensor B must be contiguous (or a 2D transpose)");
  }

  size_t M = tensor_a.shape[a_ndim - 2];
  size_t K = tensor_a.shape[a_ndim - 1];
  size_t N = tensor_b.shape[b_ndim - 1];

  assert(K == tensor_b.shape[b_ndim - 2] && "Inner dimensions must match");

  size_t batch_size = 1;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    batch_size *= tensor_a.shape[i];
  }

  Shape out_shape;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    out_shape.push_back(tensor_a.shape[i]);
  }
  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor<float, CUDA> out{out_shape};

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t handle = CublasHandle::get();

  cublasOperation_t op_b = b_transposed ? CUBLAS_OP_T : CUBLAS_OP_N;
  int ldb = b_transposed ? static_cast<int>(K) : static_cast<int>(N);

  if (batch_size == 1) {
    cublasStatus_t status = cublasSgemm(
        handle,
        op_b, CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        tensor_b.data, ldb,
        tensor_a.data, static_cast<int>(K),
        &beta,
        out.data(), static_cast<int>(N));

    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(fmt::format("cuBLAS SGEMM failed: {}", static_cast<int>(status)));
    }
  } else {
    assert(!b_transposed && "Batched matmul with transposed B not yet implemented");

    long long int stride_a = static_cast<long long int>(M * K);
    long long int stride_b = (b_ndim > 2) ? static_cast<long long int>(K * N) : 0;
    long long int stride_c = static_cast<long long int>(M * N);

    cublasStatus_t status = cublasSgemmStridedBatched(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(N),
        static_cast<int>(M),
        static_cast<int>(K),
        &alpha,
        tensor_b.data, static_cast<int>(N), stride_b,
        tensor_a.data, static_cast<int>(K), stride_a,
        &beta,
        out.data(), static_cast<int>(N), stride_c,
        static_cast<int>(batch_size));

    if (status != CUBLAS_STATUS_SUCCESS) {
      throw std::runtime_error(fmt::format("cuBLAS batched SGEMM failed: {}", static_cast<int>(status)));
    }
  }

  return out;
}
#endif

template <>
Tensor<float, CUDA> matmul(const TensorView<float, CUDA>& tensor_a,
                            const TensorView<float, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && "tensor A must be contiguous");

  size_t a_ndim = tensor_a.shape.size();
  size_t b_ndim = tensor_b.shape.size();

  assert(a_ndim >= 2 && b_ndim >= 2);

  bool b_transposed = is_2d_transpose(tensor_b);
  if (!b_transposed) {
    assert(tensor_b.is_contiguous() && "tensor B must be contiguous (or a 2D transpose)");
  }

  size_t M = tensor_a.shape[a_ndim - 2];
  size_t K = tensor_a.shape[a_ndim - 1];
  size_t N = tensor_b.shape[b_ndim - 1];

  assert(K == tensor_b.shape[b_ndim - 2] && "Inner dimensions must match");

  size_t batch_size = 1;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    batch_size *= tensor_a.shape[i];
  }

  Shape out_shape;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    out_shape.push_back(tensor_a.shape[i]);
  }
  out_shape.push_back(M);
  out_shape.push_back(N);

  Tensor<float, CUDA> out{out_shape};

  // heyoooooooooooooooooooooooooooooooooooooooo

  return out;
}

using namespace kittens;

// ============================================================================
// MATMUL KERNEL WITH DOUBLE BUFFERING (PIPELINING)
// ============================================================================
//
// OVERVIEW:
// ---------
// Matrix multiplication C = A @ B where:
//   A is [M x K], B is [K x N], C is [M x N]
//
// Each thread block computes one TILE_M x TILE_N output tile.
// We iterate over K in chunks of TILE_K, accumulating partial results.
//
// DOUBLE BUFFERING CONCEPT:
// -------------------------
// Without pipelining (what we had before):
//
//   Iteration 0:  [LOAD tile 0] --> [WAIT] --> [COMPUTE tile 0]
//   Iteration 1:  [LOAD tile 1] --> [WAIT] --> [COMPUTE tile 1]
//   Iteration 2:  [LOAD tile 2] --> [WAIT] --> [COMPUTE tile 2]
//
// The GPU sits idle during LOAD because memory operations have high latency.
//
// With double buffering (pipelining):
//
//   Prologue:     [LOAD tile 0]
//   Iteration 0:  [LOAD tile 1] + [WAIT tile 0] + [COMPUTE tile 0]
//   Iteration 1:  [LOAD tile 2] + [WAIT tile 1] + [COMPUTE tile 1]
//   Epilogue:                     [WAIT tile 2] + [COMPUTE tile 2]
//
// By using TWO buffers, we can load the NEXT tile while computing the CURRENT
// tile. This overlaps memory latency with computation, significantly improving
// throughput.
//
// MEMORY LAYOUT:
// --------------
// We allocate two sets of shared memory buffers:
//   As[0], Bs[0] - Buffer 0
//   As[1], Bs[1] - Buffer 1
//
// We also use two semaphores, one per buffer, to track when each load completes.
//
// ============================================================================

static constexpr int TILE_M = 64;    // Output tile rows
static constexpr int TILE_N = 64;    // Output tile columns
static constexpr int TILE_K = 32;    // Inner dimension chunk size

static constexpr int NUM_WARPS = 4;        // 4 warps per block
static constexpr int WARP_THREADS = 32;    // 32 threads per warp (hardware constant)
static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;  // 128 threads total
static constexpr int WARP_M = TILE_M / NUM_WARPS;  // Each warp handles 16 rows

// Shared memory tile types (st = shared tile, bf = bfloat16)
using a_tile = st_bf<TILE_M, TILE_K>;   // A tile: 64x32 bf16
using b_tile = st_bf<TILE_K, TILE_N>;   // B tile: 32x64 bf16
using d_tile = st_bf<TILE_M, TILE_N>;   // Output tile: 64x64 bf16

// Global memory layout descriptors for TMA (Tensor Memory Accelerator)
// The -1 values are placeholders for runtime dimensions (M, K, N)
using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

// Number of pipeline stages (2 = double buffering)
static constexpr int NUM_STAGES = 2;

__global__
__launch_bounds__(NUM_THREADS)
void kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int M, int N, int K
  ) {

  // -------------------------------------------------------------------------
  // STEP 1: IDENTIFY THIS BLOCK'S OUTPUT TILE
  // -------------------------------------------------------------------------
  // Each block computes one TILE_M x TILE_N output tile.
  // blockIdx.x = column index, blockIdx.y = row index in the output grid.
  int col = blockIdx.x;
  int row = blockIdx.y;

  // Each warp (group of 32 threads) handles a portion of the output tile.
  // With 4 warps and TILE_M=64, each warp computes 16 rows of output.
  int warp_id = threadIdx.x / WARP_THREADS;  // 0, 1, 2, or 3
  [[maybe_unused]] int warp_row_offset = warp_id * WARP_M;  // 0, 16, 32, or 48

  // -------------------------------------------------------------------------
  // STEP 2: ALLOCATE SHARED MEMORY (DOUBLE BUFFERED)
  // -------------------------------------------------------------------------
  // tma_swizzle_allocator handles memory alignment for TMA operations.
  // We allocate TWO sets of A and B tiles for double buffering.
  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);

  // Double buffers: As[0]/Bs[0] and As[1]/Bs[1]
  a_tile (&As)[NUM_STAGES] = al.allocate<a_tile, NUM_STAGES>();
  b_tile (&Bs)[NUM_STAGES] = al.allocate<b_tile, NUM_STAGES>();
  d_tile &Ds = al.allocate<d_tile>();  // Output buffer (only need one)

  // -------------------------------------------------------------------------
  // STEP 3: INITIALIZE SEMAPHORES FOR ASYNC MEMORY OPERATIONS
  // -------------------------------------------------------------------------
  // Semaphores track when TMA loads complete. We need one per buffer.
  // The semaphore mechanism uses a "phase" bit that alternates 0->1->0...
  // A load signals completion by incrementing the semaphore.
  // wait() blocks until the semaphore reaches the expected phase.
  __shared__ semaphore smem_arrived[NUM_STAGES];
  if (threadIdx.x == 0) {
    // Initialize both semaphores with count=0, phase=1
    // Phase=1 means we expect the first arrival to flip it to 0
    init_semaphore(smem_arrived[0], 0, 1);
    init_semaphore(smem_arrived[1], 0, 1);
  }
  __syncthreads();  // Ensure all threads see initialized semaphores

  // -------------------------------------------------------------------------
  // STEP 4: DECLARE REGISTER TILES
  // -------------------------------------------------------------------------
  // rt = register tile (stored in thread-private registers)
  // Each warp has its own register tiles; no sharing between warps.
  rt_bf<WARP_M, TILE_K> A_reg;  // This warp's A slice: 16x32 bf16
  rt_bf<TILE_K, TILE_N> B_reg;  // Full B tile: 32x64 bf16
  rt_bf<TILE_K, TILE_N, ducks::rt_layout::col> B_reg_col;  // B in column layout (for MMA)
  rt_fl<WARP_M, TILE_N> C_accum;  // Accumulator: 16x64 float32 (higher precision)

  // Zero the accumulator before starting
  warp::zero(C_accum);

  int num_tiles = K / TILE_K;  // Number of K-dimension iterations
  int phase[NUM_STAGES] = {0, 0};  // Phase tracking for each buffer's semaphore

  // -------------------------------------------------------------------------
  // STEP 5: PROLOGUE - LOAD FIRST TILE INTO BUFFER 0
  // -------------------------------------------------------------------------
  // Before entering the main loop, kick off the first load.
  // This gets the pipeline started - we'll have data ready when we need it.
  if (threadIdx.x == 0) {
    // Tell the semaphore to expect these bytes
    tma::expect_bytes(smem_arrived[0], sizeof(a_tile) + sizeof(b_tile));
    // Start async loads into buffer 0
    tma::load_async(As[0], A_layout, {row, 0}, smem_arrived[0]);
    tma::load_async(Bs[0], B_layout, {0, col}, smem_arrived[0]);
  }

  // -------------------------------------------------------------------------
  // STEP 6: MAIN LOOP WITH DOUBLE BUFFERING
  // -------------------------------------------------------------------------
  // For each K-tile:
  //   1. Start loading NEXT tile (into buffer 1-current)
  //   2. Wait for CURRENT tile to arrive (buffer current)
  //   3. Compute using CURRENT tile
  //
  // This overlaps the load of tile[i+1] with compute of tile[i].
  for (int tile = 0; tile < num_tiles; ++tile) {
    // Which buffer are we using this iteration?
    // tile 0 -> buffer 0, tile 1 -> buffer 1, tile 2 -> buffer 0, ...
    int cur_buf = tile % NUM_STAGES;
    int next_buf = (tile + 1) % NUM_STAGES;
    int next_tile = tile + 1;

    // --- LAUNCH NEXT LOAD (if there is a next tile) ---
    // This happens BEFORE we wait for the current tile, maximizing overlap.
    if (next_tile < num_tiles && threadIdx.x == 0) {
      // Set up semaphore for next buffer
      tma::expect_bytes(smem_arrived[next_buf], sizeof(a_tile) + sizeof(b_tile));
      // Start async loads for next iteration
      tma::load_async(As[next_buf], A_layout, {row, next_tile}, smem_arrived[next_buf]);
      tma::load_async(Bs[next_buf], B_layout, {next_tile, col}, smem_arrived[next_buf]);
    }

    // --- WAIT FOR CURRENT TILE ---
    // Block until the current buffer's load has completed.
    // The phase bit alternates each time we wait on the same semaphore.
    wait(smem_arrived[cur_buf], phase[cur_buf]);
    phase[cur_buf] ^= 1;  // Flip phase for next use of this buffer

    // --- LOAD FROM SHARED MEMORY TO REGISTERS ---
    // warpgroup::load reads the portion of A that this warp needs.
    // It uses warpid() internally to compute the correct row offset.
    warpgroup::load(A_reg, As[cur_buf]);  // Each warp gets its 16 rows
    warp::load(B_reg, Bs[cur_buf]);       // All warps load the full B tile

    // --- COMPUTE: MATRIX MULTIPLY-ACCUMULATE ---
    // mma_AB requires B in column-major layout for the tensor cores.
    // swap_layout transposes the register layout (not the data).
    warp::swap_layout(B_reg_col, B_reg);

    // Perform: C_accum += A_reg @ B_reg_col
    // This uses HMMA (mma.sync.m16n8k16) tensor core instructions.
    // Each warp computes a 16x64 portion of the output.
    warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);

    // --- SYNCHRONIZE BEFORE REUSING BUFFER ---
    // All threads must finish reading from shared memory before we
    // can overwrite it with the next tile's data.
    __syncthreads();
  }

  // -------------------------------------------------------------------------
  // STEP 7: EPILOGUE - STORE RESULTS
  // -------------------------------------------------------------------------
  // All K-tiles have been processed. C_accum contains the final result.

  // Convert from float32 accumulator back to bf16 and store to shared memory.
  // warpgroup::store handles the warp-to-row mapping automatically.
  warpgroup::store(Ds, C_accum);
  __syncthreads();  // Ensure all warps have written their portion

  // Use TMA to write the result tile back to global memory.
  // Only one thread needs to issue the TMA store command.
  if (threadIdx.x == 0) {
    tma::store_async(D_layout, Ds, {row, col});
    tma::store_async_read_wait();  // Wait for store to complete before kernel exits
  }
}

template <>
Tensor<bfloat16, CUDA> matmul(const TensorView<bfloat16, CUDA>& tensor_a,
                               const TensorView<bfloat16, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && "tensor A must be contiguous");

  size_t a_ndim = tensor_a.shape.size();
  size_t b_ndim = tensor_b.shape.size();

  assert(a_ndim >= 2 && b_ndim >= 2);

  // Check if B is a 2D transpose - if so, we use CUBLAS_OP_T
  bool b_transposed = is_2d_transpose(tensor_b);
  if (!b_transposed) {
    assert(tensor_b.is_contiguous() && "tensor B must be contiguous (or a 2D transpose)");
  }

  size_t M = tensor_a.shape[a_ndim - 2];
  size_t K = tensor_a.shape[a_ndim - 1];
  size_t N = tensor_b.shape[b_ndim - 1];

  assert(K == tensor_b.shape[b_ndim - 2] && "Inner dimensions must match");

  // Calculate batch size from A's leading dimensions
  size_t batch_size = 1;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    batch_size *= tensor_a.shape[i];
  }

  // Build output shape
  Shape out_shape;
  for (size_t i = 0; i < a_ndim - 2; ++i) {
    out_shape.push_back(tensor_a.shape[i]);
  }
  out_shape.push_back(M);
  out_shape.push_back(N);

  auto n_elements = M * N;
  TensorStorage<bfloat16, CUDA> storage(n_elements);
  Tensor<bfloat16, CUDA> out{out_shape, std::move(storage)};

  auto* a_d = reinterpret_cast<Cuda<bfloat16>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<Cuda<bfloat16>*>(tensor_b.data); // NOLINT
  auto* c_d = reinterpret_cast<Cuda<bfloat16>*>(out.data()); // NOLINT
                                                             //
  a_gl A_layout{reinterpret_cast<bf16*>(a_d), nullptr, nullptr, (unsigned long)M, (unsigned long)K};
  b_gl B_layout{reinterpret_cast<bf16*>(b_d), nullptr, nullptr, (unsigned long)K, (unsigned long)N};
  d_gl D_layout{reinterpret_cast<bf16*>(c_d), nullptr, nullptr, (unsigned long)M, (unsigned long)N};

  dim3 blocks(N / TILE_N, M / TILE_M);
  constexpr int smem_size = 48 * 1024;

  auto attr_err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  if (attr_err != cudaSuccess) {
    fmt::print(stderr, "cudaFuncSetAttribute error: {}\n", cudaGetErrorString(attr_err));
  }

  kernel<<<blocks, NUM_THREADS, smem_size>>>(A_layout, B_layout, D_layout, M, N, K);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fmt::print(stderr, "Kernel launch error: {}\n", cudaGetErrorString(err));
  }
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {
    fmt::print(stderr, "Kernel execution error: {}\n", cudaGetErrorString(err));
  }

  return out;
}


} // namespace tensor::kernels
