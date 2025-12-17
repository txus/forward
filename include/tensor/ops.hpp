#pragma once

#include <tensor/tensor.hpp>

namespace tensor {

template <DType T, Device D> Tensor<T, D> arange(T start, T end, T step);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> add(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> sub(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> sub(const TensorView<T, D>& tensor_a,
                                      std::remove_const_t<T> scalar);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> mul(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> mul(const TensorView<T, D>& tensor_a,
                                      std::remove_const_t<T> scalar);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> div(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> div(const TensorView<T, D>& tensor_a,
                                      std::remove_const_t<T> scalar);

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> tril(const TensorView<T, D>& tensor, bool diagonal);

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> pow(std::remove_const_t<T> scalar,
                                      const TensorView<T, D>& tensor);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> pow(const TensorView<T, D>& tensor,
                                      std::remove_const_t<T> scalar);

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> masked_fill(const TensorView<T, D>& tensor_a,
                                              const TensorView<int, D>& mask,
                                              std::remove_const_t<T> masked_value);

template <DType T1, DType T2, Device D>
Tensor<std::remove_const_t<T1>, D> matmul(const TensorView<T1, D>& tensor_a,
                                          const TensorView<T2, D>& tensor_b);

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> cat(const TensorView<T, D>& tensor_a,
                                      const TensorView<T, D>& tensor_b, int dim);

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> slice(const TensorView<T, D>& view, int dim, size_t start,
                                        size_t end);

template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> sum(const TensorView<T, D>& input, int dim, bool keepdim);
template <DType T, Device D>
Tensor<std::remove_const_t<T>, D> max(const TensorView<T, D>& input, int dim, bool keepdim);

template <DType T, Device D>
Tensor<int, D> argmax(const TensorView<T, D>& input, int dim, bool keepdim);

} // namespace tensor
