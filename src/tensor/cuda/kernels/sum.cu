#include "sum.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cuda_bf16.hpp>

namespace tensor::kernels {

using namespace dtype;

const int blockThreads = 256;

__global__ void sum_float_kernel(Cuda<float>* out, Cuda<float>* input, size_t num_reductions, size_t reduce_size, size_t reduce_stride) {
  __shared__ Cuda<float> shmem[blockThreads]; // NOLINT
  size_t tid = threadIdx.x;

  size_t reduction_idx = blockIdx.x; // which reduction are we doing?

  // decompose into outer and inner indices
  size_t outer_idx = reduction_idx / reduce_stride;
  size_t inner_idx = reduction_idx % reduce_stride;

  // base pointer for this reduction
  size_t base = (outer_idx * reduce_size * reduce_stride) + inner_idx;

  // reduce with a grid stride loop to handle reduce_size > blockThreads
  float thread_sum = 0.0;
  for (size_t element = tid; element < reduce_size; element += blockDim.x) {
    thread_sum += input[base + (element * reduce_stride)];
  }
  // now we only have to reduce 'blockThreads' elements, which is easy within a block

  // load partial sums onto shmem
  shmem[tid] = thread_sum;
  __syncthreads();

  // reduce in shared memory
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) { // NOLINT
    if (tid < stride) { shmem[tid] += shmem[tid + stride]; }
    __syncthreads();
  }

  // warp shuffle for the final warp-level reduction
  if (tid < 32) {
    float val = shmem[tid] + shmem[tid + 32];
    for (int offset = 16; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (tid == 0) {
      out[reduction_idx] = val;
    }
  }
}

Tensor<float, CUDA> sum_float(const TensorView<float, CUDA>& input, int dim, bool keepdim) {
  assert(input.is_contiguous() && "the tensor should be contiguous");

  auto shape = input.shape;

  if (dim < 0) {
    dim = shape.size() + dim;
  }

  assert(dim >= 0 && static_cast<size_t>(dim) < shape.size());

  size_t outer_size = 1; // how many reductions will we perform? ("batch size")
  size_t inner_size = 1; // what's the distance between elements to reduce?
  size_t reduce_size = 1; // how many elements each reduction needs to reduce over

  bool found_dim = false;

  // Output shape
  Shape out_shape;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i == static_cast<size_t>(dim)) {
      if (keepdim) {
        out_shape.push_back(1);
      }
      reduce_size = shape[dim];
      found_dim = true;
    } else {
      if (!found_dim) {
        outer_size *= shape[i];
      } else {
        inner_size *= shape[i];
      }

      out_shape.push_back(shape[i]);
    }
  }

  if (out_shape.empty()) {
    out_shape.push_back(1);
  }

  auto n_elements = outer_size * inner_size;

  auto input_strides = get_all_strides(shape);

  TensorStorage<float, CUDA> storage(n_elements);
  Tensor<float, CUDA> out{out_shape, std::move(storage)};

  int block_size = blockThreads;

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<Cuda<float>*>(out.data()); // NOLINT
  auto* input_d = reinterpret_cast<Cuda<float>*>(input.data); // NOLINT

  sum_float_kernel<<<n_elements, block_size>>>(out_d, input_d, n_elements, reduce_size, inner_size);

  return out;
}

} // namespace tensor::kernels
