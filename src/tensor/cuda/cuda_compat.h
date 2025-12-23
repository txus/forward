// cuda_compat.h - Workaround for Clang+CUDA+libstdc++ __noinline__ macro conflict
// See: https://github.com/llvm/llvm-project/issues/57544
//
// CUDA's host_defines.h defines __noinline__ as __attribute__((noinline))
// but libstdc++ uses __attribute__((__noinline__)) expecting __noinline__ to be 'noinline'
// This causes: error: use of undeclared identifier 'noinline'
//
// Fix: Save, undef, include problematic headers, then restore the macro
#if defined(__clang__) && defined(__CUDA__) && defined(__noinline__)
#pragma push_macro("__noinline__")
#undef __noinline__
#include <string>
#include <stdexcept>
#pragma pop_macro("__noinline__")
#endif
