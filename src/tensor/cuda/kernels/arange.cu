#include "arange.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

template<typename DeviceT>
__global__ void arange_kernel(DeviceT* out, DeviceT start, DeviceT end, DeviceT step, size_t n) {
  size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;

  if (idx < n) {
    out[idx] = start + DeviceT(idx) * step;
  }
}

template <typename T> Tensor<T, CUDA> arange(T start, T end, T step) {
  auto n_elements = static_cast<size_t>((end - start) / step);

  TensorStorage<T, CUDA> storage(n_elements);

  Shape shape{n_elements};

  Tensor<T, CUDA> out{shape, std::move(storage)};

  size_t block_size = cuda::get_block_size(n_elements);
  size_t grid_size = cuda::get_grid_size(n_elements, block_size);

  // Convert to device-native types for kernel call
  auto* device_data = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  Cuda<T> device_start = to_device_type(start, CUDA{});
  Cuda<T> device_end = to_device_type(end, CUDA{});
  Cuda<T> device_step = to_device_type(step, CUDA{});

  arange_kernel<Cuda<T>><<<grid_size, block_size>>>(device_data, device_start, device_end, device_step, n_elements);

  return out;
}

template Tensor<bfloat16, CUDA> arange(bfloat16 start, bfloat16 end, bfloat16 step);
template Tensor<float, CUDA> arange(float start, float end, float step);
template Tensor<int, CUDA> arange(int start, int end, int step);

} // namespace tensor::kernels
