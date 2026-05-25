#include <metal_stdlib>
using namespace metal;

// Templated: one source produces multiple specializations
template <typename T>
kernel void fill_kernel(device T* out [[buffer(0)]], constant T& value [[buffer(1)]],
                        constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    out[gid] = value;
  }
}

// Explicit instantiations — MSL requires named entry points per type.
template [[host_name("fill_f32")]]
kernel void fill_kernel<float>(device float*, constant float&, constant uint&, uint);

template [[host_name("fill_i32")]]
kernel void fill_kernel<int>(device int*, constant int&, constant uint&, uint);

template [[host_name("fill_bf16")]]
kernel void fill_kernel<bfloat>(device bfloat*, constant bfloat&, constant uint&, uint);
