#include "tril.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

template<typename T>
__global__ void tril_kernel(T* out, T* tensor, bool diagonal, size_t cols, size_t rows) {
  // each block handles a row
  size_t row_idx = blockIdx.x;
  size_t row_offset = row_idx * cols;
  size_t cols_to_keep = 0;
  if (diagonal) {
    cols_to_keep = row_idx + 2;
  } else {
    cols_to_keep = row_idx + 1;
  }

  for (size_t col_idx = threadIdx.x; col_idx < cols; col_idx += blockDim.x) {
    if (col_idx < cols) {
      if (col_idx < cols_to_keep) {
        out[row_offset + col_idx] = tensor[row_offset + col_idx];
      } else {
        out[row_offset + col_idx] = T(0);
      }
    }
  }
}

template <typename T> Tensor<T, CUDA> tril(const TensorView<T, CUDA>& tensor, bool diagonal) {
  assert(tensor.shape.size() == 2);
  assert(tensor.is_contiguous());

  auto cols = tensor.shape[1];
  auto rows = tensor.shape[0];

  auto n_elements = rows * cols;

  TensorStorage<T, CUDA> storage(n_elements);

  Shape shape{n_elements};

  Tensor<T, CUDA> out{shape, std::move(storage)};

  size_t block_size = cuda::get_block_size(cols);
  size_t grid_size = rows;

  // Convert to device-native types for kernel call
  auto* device_data = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  auto* in_d = reinterpret_cast<Cuda<T>*>(tensor.data); // NOLINT
  Cuda<T> diagonal_d = to_device_type(diagonal, CUDA{});

  tril_kernel<Cuda<T>><<<grid_size, block_size>>>(device_data, in_d, diagonal_d, cols, rows);

  return out;
}

template Tensor<bfloat16, CUDA> tril(const TensorView<bfloat16, CUDA>& tensor, bool diagonal);
template Tensor<float, CUDA> tril(const TensorView<float, CUDA>& tensor, bool diagonal);
template Tensor<int, CUDA> tril(const TensorView<int, CUDA>& tensor, bool diagonal);

} // namespace tensor::kernels
