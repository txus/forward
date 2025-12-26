#include <cstddef>
#include "div.cuh"
#include "utils.cuh"

namespace tensor::kernels {

using namespace dtype;

__global__ void div_float_kernel(Cuda<float>* out, const Cuda<float>* tensor_a, const Cuda<float>* tensor_b, size_t n) {
  // we load 4 fp32 values at a time = 128 bits
  auto base = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto idx = base * 4;

  if (idx + 3 < n) {
    // load 2 doubles = 4 floats = 128 bits
    double2 a_vec = reinterpret_cast<const double2*>(tensor_a)[base]; // NOLINT
    double2 b_vec = reinterpret_cast<const double2*>(tensor_b)[base]; // NOLINT

    // reinterpret as a pair of floats
    float* a2 = reinterpret_cast<float*>(&a_vec); // NOLINT
    float* b2 = reinterpret_cast<float*>(&b_vec); // NOLINT

    double2 out_vec;
    float* out2 = reinterpret_cast<float*>(&out_vec); // NOLINT

    out2[0] = a2[0] / b2[0];
    out2[1] = a2[1] / b2[1];
    out2[2] = a2[2] / b2[2];
    out2[3] = a2[3] / b2[3];

    reinterpret_cast<double2*>(out)[base] = out_vec; // NOLINT
  }
}


Tensor<float, CUDA> div_float(const TensorView<float, CUDA>& tensor_a, const TensorView<float, CUDA>& tensor_b) {
  assert(tensor_a.is_contiguous() && tensor_b.is_contiguous() && "the two tensors should be contiguous");
  assert(tensor_a.shape == tensor_b.shape && "the two tensors should be the same shape");

  size_t n_elements = tensor_a.data_size;
  TensorStorage<float, CUDA> storage(n_elements);

  Tensor<float, CUDA> out{tensor_a.shape, std::move(storage)};

  int block_size = 512;
  // each thread handles 4 elements
  int grid_size = cuda::get_grid_size(n_elements / 4, block_size);

  auto* out_d = reinterpret_cast<Cuda<float>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<const Cuda<float>*>(tensor_a.data); // NOLINT
  auto* b_d = reinterpret_cast<const Cuda<float>*>(tensor_b.data); // NOLINT

  div_float_kernel<<<grid_size, block_size>>>(out_d, a_d, b_d, n_elements);

  return out;
}

__global__ void div_scalar_float_kernel(Cuda<float>* out, const Cuda<float>* tensor, float scalar, size_t n) {
  // we load 4 fp32 values at a time = 128 bits
  auto base = (blockIdx.x * blockDim.x) + threadIdx.x;
  auto idx = base * 4;

  if (idx + 3 < n) {
    // load 2 doubles = 4 floats = 128 bits
    double2 a_vec = reinterpret_cast<const double2*>(tensor)[base]; // NOLINT

    // reinterpret as a pair of floats
    float* a2 = reinterpret_cast<float*>(&a_vec); // NOLINT

    double2 out_vec;
    float* out2 = reinterpret_cast<float*>(&out_vec); // NOLINT

    out2[0] = a2[0] / scalar;
    out2[1] = a2[1] / scalar;
    out2[2] = a2[2] / scalar;
    out2[3] = a2[3] / scalar;

    reinterpret_cast<double2*>(out)[base] = out_vec; // NOLINT
  }
}

Tensor<float, CUDA> div_float(const TensorView<float, CUDA>& tensor, float scalar) {
  assert(tensor.is_contiguous() && "the two tensors should be contiguous");

  size_t n_elements = tensor.data_size;
  TensorStorage<float, CUDA> storage(n_elements);

  Tensor<float, CUDA> out{tensor.shape, std::move(storage)};

  int block_size = 512;
  // each thread handles 4 elements
  int grid_size = cuda::get_grid_size(n_elements / 4, block_size);

  auto* out_d = reinterpret_cast<Cuda<float>*>(out.data()); // NOLINT
  auto* a_d = reinterpret_cast<const Cuda<float>*>(tensor.data); // NOLINT
  Cuda<float> device_scalar = to_device_type(scalar, CUDA{});

  div_scalar_float_kernel<<<grid_size, block_size>>>(out_d, a_d, device_scalar, n_elements);

  return out;
}

} // namespace tensor::kernels
