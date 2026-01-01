// Single compilation unit for safetensors implementation
// This avoids multiple definition errors when linking tensor_cpu and tensor_cuda together

#define SAFETENSORS_CPP_IMPLEMENTATION
#include <safetensors.hh>
