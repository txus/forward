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

static constexpr int TILE_M = 64;
static constexpr int TILE_N = 64;
static constexpr int TILE_K = 32;

static constexpr int WARP_THREADS = 32;
static constexpr int NUM_THREADS = WARP_THREADS;

using a_tile = st_bf<TILE_M, TILE_K>;
using b_tile = st_bf<TILE_K, TILE_N>;
using d_tile = st_bf<TILE_M, TILE_N>;

using a_gl = gl<bf16, 1, 1, -1, -1, a_tile>;
using b_gl = gl<bf16, 1, 1, -1, -1, b_tile>;
using d_gl = gl<bf16, 1, 1, -1, -1, d_tile>;

__global__
__launch_bounds__(NUM_THREADS, 1)
void kernel(
    const __grid_constant__ a_gl A_layout,
    const __grid_constant__ b_gl B_layout,
    const __grid_constant__ d_gl D_layout,
    int M, int N, int K
  ) {

  int col = blockIdx.x;
  int row = blockIdx.y;

  extern __shared__ int __shm[];
  tma_swizzle_allocator al((int*)&__shm[0]);

  a_tile &As = al.allocate<a_tile>();
  b_tile &Bs = al.allocate<b_tile>();
  d_tile &Ds = al.allocate<d_tile>();

  __shared__ semaphore smem_arrived;
  if (threadIdx.x == 0) {
    init_semaphore(smem_arrived, 0, 1);
  }
  __syncthreads();

  rt_bf<TILE_M, TILE_K> A_reg;
  rt_bf<TILE_K, TILE_N> B_reg;
  rt_bf<TILE_K, TILE_N, ducks::rt_layout::col> B_reg_col;
  rt_fl<TILE_M, TILE_N> C_accum;

  warp::zero(C_accum);
  int num_tiles = K / TILE_K;
  int phase = 0;

  for (int tile = 0; tile < num_tiles; ++tile) {
    if (threadIdx.x == 0) {
      tma::expect_bytes(smem_arrived, sizeof(a_tile) + sizeof(b_tile));
      tma::load_async(As, A_layout, {row, tile}, smem_arrived);
      tma::load_async(Bs, B_layout, {tile, col}, smem_arrived);
    }

    wait(smem_arrived, phase);
    phase ^= 1;

    warp::load(A_reg, As);
    warp::load(B_reg, Bs);

    warp::swap_layout(B_reg_col, B_reg);
    warp::mma_AB(C_accum, A_reg, B_reg_col, C_accum);

    __syncthreads();
  }

  warp::store(Ds, C_accum);
  __syncthreads();

  if (threadIdx.x == 0) {
    tma::store_async(D_layout, Ds, {row, col});
    tma::store_async_read_wait();
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
