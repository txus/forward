#pragma once

#include <tensor/tensor.hpp>

namespace tensor {

// Note: Using typename instead of DType/Device concepts in declarations
// to avoid ABI mismatch between NVCC and Clang (different concept mangling)

template <typename T, typename D> Tensor<T, D> arange(T start, T end, T step);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> add(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> sub(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> sub(const TensorView<T, D>& tensor_a,
                                      std::remove_const_t<T> scalar);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> mul(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> mul(const TensorView<T, D>& tensor_a,
                                      std::remove_const_t<T> scalar);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> div(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> div(const TensorView<T, D>& tensor_a,
                                      std::remove_const_t<T> scalar);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> tril(const TensorView<T, D>& tensor, bool diagonal);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> pow(std::remove_const_t<T> scalar,
                                      const TensorView<T, D>& tensor);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> pow(const TensorView<T, D>& tensor,
                                      std::remove_const_t<T> scalar);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> masked_fill(const TensorView<T, D>& tensor_a,
                                              const TensorView<int, D>& mask,
                                              std::remove_const_t<T> masked_value);

template <typename T1, typename T2, typename D>
Tensor<std::remove_const_t<T1>, D> matmul(const TensorView<T1, D>& tensor_a,
                                          const TensorView<T2, D>& tensor_b);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> cat(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b, int dim);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> slice(const TensorView<T, D>& view, int dim, size_t start,
                                        size_t end);

template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> sum(const TensorView<T, D>& input, int dim, bool keepdim);
template <typename T, typename D>
Tensor<std::remove_const_t<T>, D> max(const TensorView<T, D>& input, int dim, bool keepdim);

template <typename T, typename D>
Tensor<int, D> argmax(const TensorView<T, D>& input, int dim, bool keepdim);

// mutations

template <typename T, typename D>
void replace_from_(Tensor<T, D>& out, const TensorView<T, D>& input);

} // namespace tensor
