#include "matmul.cuh"
#include "utils.cuh"
#include <cublas_v2.h>
#include <fmt/core.h>

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

namespace {

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
  }

  ~CublasHandle() {
    cublasDestroy(handle_);
  }

  CublasHandle(const CublasHandle&) = delete;
  CublasHandle& operator=(const CublasHandle&) = delete;
};

// Check if a 2D tensor is a simple transpose (stride pattern: [1, cols] instead of [cols, 1])
template <typename T, typename D>
bool is_2d_transpose(const TensorView<T, D>& view) {
  if (view.shape.size() != 2) return false;
  // A transposed 2D matrix has stride[0] = 1, stride[1] = shape[0]
  // (stepping through rows goes by 1, stepping through cols jumps by original row count)
  return view.stride[0] == 1 && view.stride[1] == view.shape[0];
}

} // anonymous namespace

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

} // namespace tensor::kernels
