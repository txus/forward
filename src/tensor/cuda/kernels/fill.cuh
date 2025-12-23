#pragma once

#include <cuda_runtime.h>
#include <tensor/dtype.hpp>
#include <cstddef>

using namespace tensor::dtype;

template<DType T>
__global__ void fill_kernel(T* out, T value, size_t n);
