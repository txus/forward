#include "argmax.cuh"
#include "utils.cuh"
#include <cstddef>
#include <cuda_bf16.hpp>
#include <limits>

namespace tensor::kernels {

using namespace dtype;

const int blockThreads = 256;

__global__ void argmax_bfloat16_kernel(Cuda<int>* out, Cuda<bfloat16>* input, size_t num_reductions, size_t reduce_size, size_t reduce_stride) {
  __shared__ Cuda<bfloat16> shmem_val[blockThreads]; // NOLINT
  __shared__ int shmem_idx[blockThreads]; // NOLINT

  size_t tid = threadIdx.x;

  size_t reduction_idx = blockIdx.x; // which reduction are we doing?

  // decompose into outer and inner indices
  size_t outer_idx = reduction_idx / reduce_stride;
  size_t inner_idx = reduction_idx % reduce_stride;

  // base pointer for this reduction
  size_t base = (outer_idx * reduce_size * reduce_stride) + inner_idx;

  // reduce with a grid stride loop to handle reduce_size > blockThreads
  Cuda<bfloat16> thread_max_val = -std::numeric_limits<float>::infinity();

  int thread_max_idx = tid; // NOLINT

  for (int element = tid; element < reduce_size; element += blockDim.x) { // NOLINT
    Cuda<bfloat16> incoming_val = input[base + (element * reduce_stride)];
    int incoming_idx = element;

    if (incoming_val > thread_max_val) {
      thread_max_idx = incoming_idx;
    }

    thread_max_val = __hmax(thread_max_val, incoming_val);
  }

  // now we only have to reduce 'blockThreads' elements, which is easy within a block

  // load partial argmaxs onto shmem
  shmem_val[tid] = thread_max_val;
  shmem_idx[tid] = thread_max_idx;
  __syncthreads();

  // reduce in shared memory
  for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) { // NOLINT
    if (tid < stride) {
      Cuda<bfloat16> existing_val = shmem_val[tid];
      Cuda<bfloat16> incoming_val = shmem_val[tid + stride];
      int incoming_idx = shmem_idx[tid + stride];
      if (incoming_val > existing_val) {
        shmem_idx[tid] = incoming_idx;
      }

      shmem_val[tid] = __hmax(existing_val, incoming_val);
    }
    __syncthreads();
  }

  // warp shuffle for the final warp-level reduction
  if (tid < 32) {
    Cuda<bfloat16> val = shmem_val[tid];
    int idx = shmem_idx[tid];

    Cuda<bfloat16> incoming_val = shmem_val[tid + 32];
    int incoming_idx = shmem_idx[tid + 32];
    if (incoming_val > val) {
      shmem_idx[tid] = incoming_idx;
    }
    val = __hmax(val, incoming_val);

    for (int offset = 16; offset > 0; offset >>= 1) {
      Cuda<bfloat16> incoming_val = __shfl_down_sync(0xffffffff, val, offset);
      int incoming_idx = __shfl_down_sync(0xffffffff, idx, offset);
      if (incoming_val > val) {
        idx = incoming_idx;
      }
      val = __hmax(val, incoming_val);
    }

    if (tid == 0) {
      out[reduction_idx] = idx;
    }
  }
}

Tensor<int, CUDA> argmax_bfloat16(const TensorView<bfloat16, CUDA>& input, int dim, bool keepdim) {
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

  TensorStorage<int, CUDA> storage(n_elements);
  Tensor<int, CUDA> out{out_shape, std::move(storage)};

  int block_size = blockThreads;

  // Convert to device-native types for kernel call
  auto* out_d = reinterpret_cast<int*>(out.data()); // NOLINT
  auto* input_d = reinterpret_cast<Cuda<bfloat16>*>(input.data); // NOLINT

  argmax_bfloat16_kernel<<<n_elements, block_size>>>(out_d, input_d, n_elements, reduce_size, inner_size);

  return out;
}

} // namespace tensor::kernels
