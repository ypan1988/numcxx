// Copyright 2025 Yi Pan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NUMCXX_NDARRAY_H_
#define NUMCXX_NDARRAY_H_

#include <cstddef>
#include <stdexcept>
#include <utility>  // for std::move
#include <valarray>
#include <vector>

namespace numcxx {

enum class MemoryOrder { kRowMajor, kColMajor };

// 前向声明
template <typename T> class NdArray;
template <typename T> class SliceView;
template <typename T> class GSliceView;
template <typename T> class IndexView;
template <typename T> class MaskView;

namespace internal {

// Computes strides and returns total size
// Returns the total number of elements in the array
inline size_t ComputeStrides(std::vector<size_t>* strides,
                      const std::vector<size_t>& shape, MemoryOrder order) {
  const size_t n_dims = shape.size();
  if (n_dims == 0) {
    strides->clear();
    return 0;
  }

  strides->resize(n_dims);
  size_t total_size = 1;

  if (order == MemoryOrder::kRowMajor) {
    strides->back() = 1;
    for (size_t i = n_dims - 1; i > 0; --i) {
      (*strides)[i - 1] = (*strides)[i] * shape[i];
    }
    total_size = strides->front() * shape.front();
  } else {
    strides->front() = 1;
    for (size_t i = 1; i < n_dims; ++i) {
      (*strides)[i] = (*strides)[i - 1] * shape[i - 1];
    }
    total_size = strides->back() * shape.back();
  }

  return total_size;
}

// Validates the shape parameters
inline void ValidateShape(const std::vector<size_t>& shape) {
  if (shape.empty()) {
    throw std::invalid_argument("Shape cannot be empty");
  }

  for (const auto& dim : shape) {
    if (dim == 0) {
      throw std::invalid_argument("All dimensions must be > 0");
    }
  }
}

}  // namespace internal

// ==================== 表达式模板基础设施 ====================
template <typename Derived>
class Expr {
 public:
  auto operator[](size_t i) const {
    return static_cast<const Derived&>(*this)[i];
  }

  const std::vector<size_t>& shape() const {
    return static_cast<const Derived&>(*this).shape();
  }

  size_t size() const {
    size_t total = 1;
    for (auto dim : shape()) total *= dim;
    return total;
  }
};

// ==================== NdArray 核心实现 ====================
template <typename T>
class NdArray : public Expr<NdArray<T>> {
 public:
  NdArray(std::vector<size_t> shape, const T& init_value = T(),
          MemoryOrder order = MemoryOrder::kRowMajor)
      : shape_(std::move(shape)), order_(order) {
    internal::ValidateShape(shape_);
    const size_t total_size =
        internal::ComputeStrides(&strides_, shape_, order_);
    data_.resize(total_size, init_value);
  }

  // 从表达式构造
  template <typename E,
            typename = std::enable_if_t<std::is_base_of_v<Expr<E>, E>>>
  explicit NdArray(const Expr<E>& expr) : NdArray(expr.shape()) {
    for (size_t i = 0; i < size(); ++i) {
      data_[i] = expr[i];
    }
  }

  // 基础访问
  T& operator[](size_t i) { return data_[i]; }
  const T& operator[](size_t i) const { return data_[i]; }

  // 形状访问
  const std::vector<size_t>& shape() const { return shape_; }
  const std::vector<size_t>& strides() const { return strides_; }
  MemoryOrder memory_order() const { return order_; }
  bool is_row_major() const { return order_ == MemoryOrder::kRowMajor; }
  bool is_column_major() const { return order_ == MemoryOrder::kColMajor; }

  size_t ndim() const { return shape_.size(); }
  size_t size() const { return data_.size(); }

  // Disallow copy and assign
  NdArray(const NdArray&) = delete;
  NdArray& operator=(const NdArray&) = delete;

  // Allow move operations
  NdArray(NdArray&&) = default;
  NdArray& operator=(NdArray&&) = default;

  T* data() { return &data_[0]; }
  const T* data() const { return &data_[0]; }

  // 切片操作
  SliceView<T> slice(const std::vector<std::slice>& slices) {
    return SliceView<T>(*this, slices);
  }

  // 表达式赋值
  template <typename E>
  NdArray& operator=(const Expr<E>& expr) {
    for (size_t i = 0; i < size(); ++i) {
      data_[i] = expr[i];
    }
    return *this;
  }

 private:
  std::valarray<T> data_;
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  MemoryOrder order_;

  friend class SliceView<T>;
};

// ==================== SliceView 实现 ====================
template <typename T>
class SliceView : public Expr<SliceView<T>> {
 public:
  SliceView(NdArray<T>& array, const std::vector<std::slice>& slices)
      : data_(array.data_),
        original_shape_(array.shape_),
        slices_(slices),
        order_(array.order_) {
    compute_view_shape();
  }

  T operator[](size_t linear_index) const {
    const auto indices = unravel_index(linear_index, shape_);
    size_t original_index = 0;
    for (size_t dim = 0; dim < indices.size(); ++dim) {
      const auto& s = slices_[dim];
      original_index +=
          (s.start() + indices[dim] * s.stride()) * original_strides_[dim];
    }
    return data_[original_index];
  }

  const std::vector<size_t>& shape() const { return shape_; }

  // 支持从表达式赋值
  template <typename E>
  SliceView& operator=(const Expr<E>& expr) {
    for (size_t i = 0; i < size(); ++i) {
      data_[unravel_original_index(i)] = expr[i];
    }
    return *this;
  }

 private:
  void compute_view_shape() {
    shape_.resize(slices_.size());
    for (size_t i = 0; i < slices_.size(); ++i) {
      shape_[i] = slices_[i].size();
    }
    original_strides_ = strides_;
  }

  size_t unravel_original_index(size_t linear_index) const {
    const auto indices = unravel_index(linear_index, shape_);
    size_t original_index = 0;
    for (size_t dim = 0; dim < indices.size(); ++dim) {
      const auto& s = slices_[dim];
      original_index +=
          (s.start() + indices[dim] * s.stride()) * original_strides_[dim];
    }
    return original_index;
  }

  static std::vector<size_t> unravel_index(size_t linear_index,
                                           const std::vector<size_t>& shape) {
    std::vector<size_t> indices(shape.size());
    for (int i = shape.size() - 1; i >= 0; --i) {
      indices[i] = linear_index % shape[i];
      linear_index /= shape[i];
    }
    return indices;
  }

  std::valarray<T>& data_;
  std::vector<size_t> original_shape_;
  std::vector<size_t> original_strides_;
  std::vector<std::slice> slices_;
  std::vector<size_t> shape_;
  MemoryOrder order_;
};

// ==================== 表达式操作 ====================
// 标量包装
template <typename T>
class Scalar : public Expr<Scalar<T>> {
 public:
  explicit Scalar(T value) : value_(value) {}
  T operator[](size_t) const { return value_; }
  const std::vector<size_t>& shape() const {
    static const std::vector<size_t> scalar_shape{1};
    return scalar_shape;
  }

 private:
  T value_;
};

// 二元操作
template <typename E1, typename E2, typename Op>
class BinaryOp : public Expr<BinaryOp<E1, E2, Op>> {
 public:
  BinaryOp(E1 lhs, E2 rhs, Op op = {})
      : lhs_(std::move(lhs)), rhs_(std::move(rhs)), op_(op) {}

  auto operator[](size_t i) const { return op_(lhs_[i], rhs_[i]); }

  const std::vector<size_t>& shape() const {
    // 简化：假设形状相同，实际应实现广播
    return lhs_.shape();
  }

 private:
  E1 lhs_;
  E2 rhs_;
  Op op_;
};

// 加法操作
struct AddOp {
  template <typename L, typename R>
  auto operator()(L&& l, R&& r) const {
    return std::forward<L>(l) + std::forward<R>(r);
  }
};

// 运算符重载
template <typename E1, typename E2>
auto operator+(const Expr<E1>& lhs, const Expr<E2>& rhs) {
  return BinaryOp<E1, E2, AddOp>(static_cast<const E1&>(lhs),
                                 static_cast<const E2&>(rhs));
}

}  // namespace numcxx

#endif  // NUMCXX_NDARRAY_H_