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

namespace internal {

// Computes strides and returns total size
// Returns the total number of elements in the array
inline size_t ComputeStrides(std::vector<size_t>* strides,
                             const std::vector<size_t>& shape,
                             MemoryOrder order) {
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

/**
 * Python-style slice implementation with start/stop/step semantics
 * Supports both forward and reverse slicing with exclusive stop boundary
 */
class Slice {
 public:
  // Default constructor creates an empty slice [0:0:1]
  Slice() noexcept : start_(0), stop_(0), step_(1) {}

  /**
   * Main constructor
   * @param start First index (inclusive)
   * @param stop Last index (exclusive)
   * @param step Step size (default=1, cannot be zero)
   */
  Slice(int64_t start, int64_t stop, int64_t step = 1)
      : start_(start), stop_(stop), step_(step) {
    if (step == 0) {
      throw std::invalid_argument("Slice step cannot be zero");
    }
  }

  // Accessors
  int64_t start() const noexcept { return start_; }
  int64_t stop() const noexcept { return stop_; }
  int64_t step() const noexcept { return step_; }

  /**
   * Calculate number of elements in the slice
   * @return Number of elements, 0 if slice is empty
   */
  int64_t size() const noexcept {
    if (step_ == 0) return 0;

    // Check for empty slices
    if ((step_ > 0 && start_ >= stop_) || (step_ < 0 && start_ <= stop_)) {
      return 0;
    }

    const int64_t diff = stop_ - start_;
    return (diff / step_) + ((diff % step_) != 0 ? 1 : 0);
  }

  // Alias for step() to match other libraries
  int64_t stride() const noexcept { return step_; }

  // Comparison operators
  bool operator==(const Slice& other) const noexcept {
    return (start_ == other.start_) && (stop_ == other.stop_) &&
           (step_ == other.step_);
  }
  bool operator!=(const Slice& other) const noexcept {
    return !(*this == other);
  }

 private:
  int64_t start_;  // First index (inclusive)
  int64_t stop_;   // Last index (exclusive)
  int64_t step_;   // Step size (cannot be zero)
};

template <typename T>
class NdArray;
template <typename T>
class SliceArray;
template <typename T>
class GSliceArray;
template <typename T>
class MaskArray;
template <typename T>
class IndirectArray;

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

template <typename Op, typename E>
class UnaryOp : public Expr<UnaryOp<Op, E>> {
 public:
  UnaryOp(Op op, E expr) : op_(op), expr_(std::move(expr)) {}

  auto operator[](size_t i) const { return op_(expr_[i]); }

  const std::vector<size_t>& shape() const { return expr_.shape(); }

 private:
  Op op_;
  E expr_;
};

template <typename Op, typename E1, typename E2>
class BinaryOp : public Expr<BinaryOp<Op, E1, E2>> {
 public:
  BinaryOp(Op op, E1 lhs, E2 rhs)
      : op_(op), lhs_(std::move(lhs)), rhs_(std::move(rhs)) {}

  auto operator[](size_t i) const { return op_(lhs_[i], rhs_[i]); }

  const std::vector<size_t>& shape() const { return lhs_.shape(); }

 private:
  Op op_;
  E1 lhs_;
  E2 rhs_;
};

template <typename T>
class ScalarExpr : public Expr<ScalarExpr<T>> {
 public:
  explicit ScalarExpr(T value) : value_(value) {}

  T operator[](size_t) const { return value_; }

  const std::vector<size_t>& shape() const {
    static const std::vector<size_t> scalar_shape{1};
    return scalar_shape;
  }

 private:
  T value_;
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
  SliceArray<T> slice(const std::vector<std::slice>& slices) {
    return SliceArray<T>(*this, slices);
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

  friend class SliceArray<T>;
};

// ==================== SliceArray 实现 ====================
template <typename T>
class SliceArray : public Expr<SliceArray<T>> {
 public:
  SliceArray(NdArray<T>& array, const std::vector<std::slice>& slices)
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
  SliceArray& operator=(const Expr<E>& expr) {
    for (size_t i = 0; i < this->size(); ++i) {
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
    // Compute strides for the view
    internal::ComputeStrides(&strides_, shape_, order_);
    // Store original strides
    original_strides_ =
        internal::ComputeStrides(&original_strides_, original_shape_, order_);
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
  std::vector<size_t> strides_;
  MemoryOrder order_;
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
  return BinaryOp<AddOp, E1, E2>(static_cast<const E1&>(lhs),
                                 static_cast<const E2&>(rhs));
}

}  // namespace numcxx

#endif  // NUMCXX_NDARRAY_H_