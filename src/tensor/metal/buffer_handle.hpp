#pragma once
#include <util/metal.hpp>

// Full definition of the opaque handle that storage.hpp only forward-declares.
// Metal-internal: lives under src/tensor/metal/, so it may name MTL types.
// Include this from metal .cpp files / metal-internal headers that need to
// touch the underlying MTL::Buffer. Public headers keep only the fwd-decl.
namespace tensor::metal_fwd {
struct BufferHandle {
  MTL::Buffer* buf;
};
} // namespace tensor::metal_fwd
