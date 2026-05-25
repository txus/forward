#include <metal_stdlib>
using namespace metal;

template <typename T>
kernel void arange_kernel(device T* out [[buffer(0)]], constant T& start [[buffer(1)]],
                          constant T& step [[buffer(2)]], constant uint& n [[buffer(3)]],
                          uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    out[gid] = start + T(gid) * step;
  }
}

template [[host_name("arange_f32")]]
kernel void arange_kernel<float>(device float*, constant float&, constant float&, constant uint&,
                                 uint);

template [[host_name("arange_i32")]]
kernel void arange_kernel<int>(device int*, constant int&, constant int&, constant uint&, uint);

template [[host_name("arange_bf16")]]
kernel void arange_kernel<bfloat>(device bfloat*, constant bfloat&, constant bfloat&,
                                  constant uint&, uint);
