//===-------------------------------numcxx.h-------------------------------===//
//
// Portions of this file are derived from the LLVM libc++ `valarray` code.
// Original libc++ source code is licensed under the Apache License v2.0 with
// LLVM Exceptions (See https://llvm.org/LICENSE.txt).
//
// Modifications, enhancements, and additional code:
//  Copyright (c) 2026 Yi Pan
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// See the LICENSE file for the detail of NumCxx license.
//
//===----------------------------------------------------------------------===//

#ifndef NUMCXX_H_
#define NUMCXX_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <optional>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>
#include <version>

#if __cplusplus >= 202600L
#define NUMCXX_USE_STD 1
#include <mdarray>
#include <mdspan>
#else
#define NUMCXX_USE_STD 0
#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>
#endif

namespace numcxx::detail {
#if NUMCXX_USE_STD
using std::dextents;
using std::extents;
using std::full_extent;
using std::layout_left;
using std::layout_right;
using std::mdarray;
using std::mdspan;
using std::strided_slice;
using std::submdspan;
#else
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::full_extent;
using Kokkos::layout_left;
using Kokkos::layout_right;
using Kokkos::mdspan;
using Kokkos::strided_slice;
using Kokkos::submdspan;
using Kokkos::Experimental::mdarray;
#endif
} // namespace numcxx::detail

namespace numcxx {
// clang-format off
using  size_type = std::size_t;
using index_type = int        ; // TODO: replace int with std::ptrdiff_t

template <size_type       Rank> using dextents = detail::dextents<size_type, Rank>;
template <size_type... Extents> using  extents = detail::extents <size_type, Extents...>;
//clang-format on
}

#ifndef NUMCXX_NO_DEBUG
#define NUMCXX_ASSERT(expr, msg)                                               \
  do {                                                                         \
    if (!(expr)) {                                                             \
      std::cerr << "numcxx assertion failed: " << (msg) << "\n"                \
                << "  at " << __FILE__ << ":" << __LINE__ << " (" << __func__  \
                << ")\n";                                                      \
      std::abort();                                                            \
    }                                                                          \
  } while (0)
#else
#define NUMCXX_ASSERT(expr, msg) ((void)0)
#endif

#ifndef NUMCXX_NO_EXCEPTIONS
#define NUMCXX_THROW(exception_type, msg)                                      \
  do {                                                                         \
    throw exception_type(msg);                                                 \
  } while (0)
#else
#define NUMCXX_THROW(exception_type, msg)                                      \
  do {                                                                         \
    std::cerr << "numcxx critical error: " << (msg) << "\n";                   \
    std::abort();                                                              \
  } while (0)
#endif

namespace numcxx {
// clang-format off
                                        class slice;
template <class Tp, class Ex, class Lp> class ndarray;
template <class Tp, class Ex, class Lp> class slice_view;
template <class Tp, class Ex, class Lp> class mask_view;
template <class Tp, class Ex, class Lp> class indirect_view;

template <class Tp, class Ex, class Lp>       Tp *begin(      ndarray<Tp, Ex, Lp> &v);
template <class Tp, class Ex, class Lp> const Tp *begin(const ndarray<Tp, Ex, Lp> &v);
template <class Tp, class Ex, class Lp>       Tp *end  (      ndarray<Tp, Ex, Lp> &v);
template <class Tp, class Ex, class Lp> const Tp *end  (const ndarray<Tp, Ex, Lp> &v);
// clang-format on

class slice {
private:
  std::optional<index_type> start_;
  std::optional<index_type> stop_;
  index_type step_ = 1;

public:
  slice() = default;

  slice(std::optional<index_type> start, std::optional<index_type> stop,
        index_type step = 1)
      : start_(start), stop_(stop), step_(step) {
    assert(step != 0 && "slice step cannot be zero");
  }

  [[nodiscard]] std::optional<index_type> start() const { return start_; }
  [[nodiscard]] std::optional<index_type> stop() const { return stop_; }
  [[nodiscard]] index_type step() const { return step_; }

  friend bool operator==(const slice &x, const slice &y) {
    return x.start() == y.start() && x.stop() == y.stop() &&
           x.step() == y.step();
  }
};

template <class Op, class A0> struct nc_unary_op {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  A0 a0_;

  nc_unary_op(const Op &op, const A0 &a0) : op_(op), a0_(a0) {}

  result_type operator[](size_t i) const { return op_(a0_[i]); }

  size_t size() const { return a0_.size(); }
};

template <class Op, class A0, class A1> struct nc_binary_op {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  A0 a0_;
  A1 a1_;

  nc_binary_op(const Op &op, const A0 &a0, const A1 &a1)
      : op_(op), a0_(a0), a1_(a1) {}

  result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

  size_t size() const { return a0_.size(); }
};

template <class Tp> class nc_scalar_expr {
public:
  typedef Tp value_type;
  typedef const Tp &result_type;

private:
  const value_type &t_;
  size_t s_;

public:
  explicit nc_scalar_expr(const value_type &t, size_t s) : t_(t), s_(s) {}

  result_type operator[](size_t) const { return t_; }

  size_t size() const { return s_; }
};

template <class Tp> struct nc_unary_plus {
  typedef Tp result_type;
  Tp operator()(const Tp &x) const { return +x; }
};

template <class Tp> struct nc_bit_not {
  typedef Tp result_type;
  Tp operator()(const Tp &x) const { return ~x; }
};

template <class Tp> struct nc_bit_shift_left {
  typedef Tp result_type;
  Tp operator()(const Tp &x, const Tp &y) const { return x << y; }
};

template <class Tp> struct nc_bit_shift_right {
  typedef Tp result_type;
  Tp operator()(const Tp &x, const Tp &y) const { return x >> y; }
};

template <class Tp, class Fp> struct nc_apply_expr {
private:
  Fp __f_;

public:
  typedef Tp result_type;

  explicit nc_apply_expr(Fp __f) : __f_(__f) {}

  Tp operator()(const Tp &x) const { return __f_(x); }
};

// clang-format off
template <class Tp> struct nc_abs_expr   { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::abs  (x); } };
template <class Tp> struct nc_acos_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::acos (x); } };
template <class Tp> struct nc_asin_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::asin (x); } };
template <class Tp> struct nc_atan_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::atan (x); } };
template <class Tp> struct nc_cos_expr   { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::cos  (x); } };
template <class Tp> struct nc_cosh_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::cosh (x); } };
template <class Tp> struct nc_exp_expr   { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::exp  (x); } };
template <class Tp> struct nc_log_expr   { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::log  (x); } };
template <class Tp> struct nc_log10_expr { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::log10(x); } };
template <class Tp> struct nc_sin_expr   { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::sin  (x); } };
template <class Tp> struct nc_sinh_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::sinh (x); } };
template <class Tp> struct nc_sqrt_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::sqrt (x); } };
template <class Tp> struct nc_tan_expr   { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::tan  (x); } };
template <class Tp> struct nc_tanh_expr  { typedef Tp result_type; Tp operator()(const Tp& x) const { return std::tanh (x); } };

template <class Tp> struct nc_atan2_expr { typedef Tp result_type; Tp operator()(const Tp& x, const Tp& y) const { return std::atan2(x, y); } };
template <class Tp> struct nc_pow_expr   { typedef Tp result_type; Tp operator()(const Tp& x, const Tp& y) const { return std::pow  (x, y); } };
// clang-format on

template <class ValExpr> class nc_shift_expr {
  typedef std::remove_reference_t<ValExpr> _RmExpr;

public:
  typedef typename _RmExpr::value_type value_type;
  typedef value_type result_type;

private:
  ValExpr expr_;
  size_t size_;
  ptrdiff_t __ul_;
  ptrdiff_t __sn_;
  ptrdiff_t __n_;
  static const ptrdiff_t _Np =
      static_cast<ptrdiff_t>(sizeof(ptrdiff_t) * __CHAR_BIT__ - 1);

  nc_shift_expr(int n, const _RmExpr &e) : expr_(e), size_(e.size()), __n_(n) {
    ptrdiff_t __neg_n = static_cast<ptrdiff_t>(__n_ >> _Np);
    __sn_ = __neg_n | static_cast<ptrdiff_t>(static_cast<size_t>(-__n_) >> _Np);
    __ul_ = ((size_ - __n_) & ~__neg_n) | ((__n_ + 1) & __neg_n);
  }

public:
  result_type operator[](size_t j) const {
    ptrdiff_t i = static_cast<ptrdiff_t>(j);
    ptrdiff_t __m = (__sn_ * i - __ul_) >> _Np;
    return (expr_[(i + __n_) & __m] & __m) | (value_type() & ~__m);
  }

  size_t size() const { return size_; }

  template <class> friend class nc_val_expr;
};

template <class ValExpr> class nc_cshift_expr {
  typedef std::remove_reference_t<ValExpr> _RmExpr;

public:
  typedef typename _RmExpr::value_type value_type;
  typedef value_type result_type;

private:
  ValExpr expr_;
  size_t size_;
  size_t m_;
  size_t o1_;
  size_t o2_;

  nc_cshift_expr(int n, const _RmExpr &e) : expr_(e), size_(e.size()) {
    n %= static_cast<int>(size_);
    if (n >= 0) {
      m_ = size_ - n;
      o1_ = n;
      o2_ = n - size_;
    } else {
      m_ = -n;
      o1_ = n + size_;
      o2_ = n;
    }
  }

public:
  result_type operator[](size_t i) const {
    if (i < m_)
      return expr_[i + o1_];
    return expr_[i + o2_];
  }

  size_t size() const { return size_; }

  template <class> friend class nc_val_expr;
};

template <typename T>
struct is_slice_or_integral
    : std::bool_constant<std::is_same_v<std::decay_t<T>, slice> ||
                         std::is_integral_v<std::decay_t<T>>> {};

template <typename T>
inline constexpr bool is_slice_or_integral_v = is_slice_or_integral<T>::value;

template <typename... Args>
inline constexpr bool are_all_slice_or_integral_v =
    (is_slice_or_integral_v<Args> && ...);

// clang-format off
template <class ValExpr> class  nc_val_expr                                              ;
template <class ValExpr> struct nc_is_val_expr                       : std::false_type {};
template <class ValExpr> struct nc_is_val_expr<nc_val_expr<ValExpr>> : std::true_type  {};

template <class Tp, class Ex, class Lp> struct nc_is_val_expr<ndarray      <Tp, Ex, Lp>> : std::true_type {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<slice_view   <Tp, Ex, Lp>> : std::true_type {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<mask_view    <Tp, Ex, Lp>> : std::true_type {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<indirect_view<Tp, Ex, Lp>> : std::true_type {};
// clang-format on

namespace detail {

template <typename Extents>
inline constexpr bool is_static_extents_v = Extents::rank_dynamic() == 0;

template <typename NdArrayType>
inline constexpr bool is_static_ndarray_v =
    is_static_extents_v<typename NdArrayType::extents_type>;

template <typename Extents> struct static_extents_size;
template <size_type... Dims>
struct static_extents_size<numcxx::extents<Dims...>>
    : std::integral_constant<size_type, (Dims * ...)> {};

template <class ElementType, class Extents,
          bool IsStaticExtents = is_static_extents_v<Extents>>
struct mdarray_container_selector;

template <class ElementType, class Extents>
struct mdarray_container_selector<ElementType, Extents, true> {
  using type = std::array<ElementType, static_extents_size<Extents>::value>;
};

template <class ElementType, class Extents>
struct mdarray_container_selector<ElementType, Extents, false> {
  using type = std::vector<ElementType>;
};

template <class ElementType, class Extents>
using mdarray_container_t =
    typename mdarray_container_selector<ElementType, Extents>::type;

namespace slice_utils {
inline size_type to_submdspan_arg(index_type idx, size_type dim_len) {
  index_type res = (idx < 0) ? idx + static_cast<index_type>(dim_len) : idx;
  assert(res >= 0 && static_cast<size_type>(res) < dim_len &&
         "Index out of bounds");
  return static_cast<size_type>(res);
}

inline auto to_submdspan_arg(const slice &s, size_type dim_len) {
  auto resolve_index = [dim_len](std::optional<index_type> idx_raw,
                                 index_type default_val) {
    index_type idx = idx_raw.has_value() ? idx_raw.value() : default_val;
    return (idx < 0) ? idx + static_cast<index_type>(dim_len) : idx;
  };

  index_type step = s.step();
  index_type start = resolve_index(s.start(), (step > 0) ? 0 : (dim_len - 1));
  index_type stop = resolve_index(s.stop(), (step > 0) ? dim_len : -1);

  index_type diff = stop - start;
  assert(diff * step > 0 && "invalid slice");

  size_type offset = static_cast<size_type>(start);
  size_type extent = (diff / step) + ((diff % step) != 0 ? 1 : 0);
  size_type stride = static_cast<size_type>(std::abs(s.step()));

  return ::numcxx::detail::strided_slice<size_type, size_type, size_type>{
      offset, extent, stride};
}

template <typename MdSpan, typename... Args, size_t... Is>
auto make_submdspan(MdSpan &&src, std::index_sequence<Is...>, Args &&...args) {
  std::array<size_t, sizeof...(Args)> dims = {src.extent(Is)...};
  return detail::submdspan(std::forward<MdSpan>(src),
                           to_submdspan_arg(args, dims[Is])...);
}

} // namespace slice_utils

} // namespace detail

// [numcxx.ndarray]
template <class ElementType, class Extents,
          class LayoutPolicy = detail::layout_right>
class ndarray {
public:
  using element_type = ElementType;
  using value_type = std::remove_cv_t<element_type>;

  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  using layout_type = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  using mdspan_type = detail::mdspan<element_type, extents_type, layout_type>;
  using const_mdspan_type =
      detail::mdspan<const element_type, extents_type, layout_type>;

  using pointer = element_type *;
  using reference = element_type &;
  using const_pointer = const element_type *;
  using const_reference = const element_type &;

public:
  // clang-format off
  // construct/destroy:
  constexpr ndarray() = default;
  constexpr ndarray(const ndarray &v) = default;
  constexpr ndarray(ndarray &&v) noexcept = default;
  template <class... SizeTypes> explicit constexpr ndarray(SizeTypes... dyn_exts) : elem_(Extents(dyn_exts...)) {}
  ~ndarray() = default;

  // assignment:
  constexpr ndarray &operator=(const ndarray &v) = default;
  constexpr ndarray &operator=(ndarray &&v) noexcept = default;

  // element access
  template <typename... Args> decltype(auto) operator()(Args &&...args) const;
  template <typename... Args> decltype(auto) operator()(Args &&...args);

  // element access (flattened)
  [[nodiscard]] const value_type &operator[](size_t i) const { NUMCXX_ASSERT(i < size(), "ndarray::operator[] index out of bounds"); return data()[i]; }
  [[nodiscard]]       value_type &operator[](size_t i)       { NUMCXX_ASSERT(i < size(), "ndarray::operator[] index out of bounds"); return data()[i]; }

  // observers
  static constexpr          rank_type                     rank()       noexcept { return extents_type::rank()          ; }
  static constexpr          rank_type             rank_dynamic()       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr          size_type static_extent(size_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr          size_type        extent(size_type r) const noexcept { return elem_.extent(r); }
         constexpr          size_type                     size() const noexcept { return elem_.size()   ; }
         constexpr               bool                    empty() const noexcept { return elem_.empty()  ; }
         constexpr          size_type        stride(rank_type r) const          { return elem_.stride(r); }
         constexpr const extents_type&                 extents() const noexcept { return elem_.extents(); }
         constexpr      const_pointer                     data() const noexcept { return elem_.data()   ; }
         constexpr            pointer                     data()       noexcept { return elem_.data()   ; }
  //clang-format on

  // inline explicit ndarray(size_t n);
  // ndarray(const value_type& x, size_t n);
  // ndarray(const value_type* p, size_t n);
  // ndarray(std::initializer_list<value_type> __il);
  // ndarray(const slice_array<value_type>& sa);
  // ndarray(const mask_array<value_type>& ma);
  // ndarray(const indirect_array<value_type>& ia);

  // ndarray& operator=(std::initializer_list<value_type>);
  // ndarray& operator=(const slice_array<value_type>& sa);
  // ndarray& operator=(const mask_array<value_type>& ma);
  // ndarray& operator=(const indirect_array<value_type>& ia);

  // subset operations:
  //[[nodiscard]] nc_val_expr<__slice_expr<const ndarray&> > operator[](slice s)
  // const;
  //[[nodiscard]] slice_array<value_type> operator[](slice s);
  //[[nodiscard]]
  // nc_val_expr<__mask_expr<const ndarray&> > operator[](const ndarray<bool,
  // Extents, LayoutPolicy>& __vb) const;
  //[[nodiscard]] mask_array<value_type> operator[](const ndarray<bool, Extents,
  // LayoutPolicy>& __vb);
  //[[nodiscard]]
  //    nc_val_expr<__mask_expr<const ndarray&> > operator[](ndarray<bool,
  //    Extents, LayoutPolicy>&& __vb) const;
  //[[nodiscard]] mask_array<value_type> operator[](ndarray<bool, Extents,
  // LayoutPolicy>&& __vb);
  //[[nodiscard]]
  // nc_val_expr<__indirect_expr<const ndarray&> > operator[](const
  // ndarray<size_t, Extents, LayoutPolicy>& __vs) const;
  //[[nodiscard]] indirect_array<value_type> operator[](const ndarray<size_t,
  // Extents, LayoutPolicy>& __vs);
  //[[nodiscard]]
  //    nc_val_expr<__indirect_expr<const ndarray&> > operator[](ndarray<size_t,
  //    Extents, LayoutPolicy>&& __vs) const;
  //[[nodiscard]] indirect_array<value_type> operator[](ndarray<size_t, Extents,
  // LayoutPolicy>&& __vs);

  // clang-format off
  // unary operators:
  nc_val_expr<nc_unary_op<nc_unary_plus   <ElementType>, const ndarray &>> operator+() const { return make_unary_expr<nc_unary_plus>   (); }
  nc_val_expr<nc_unary_op<std::negate     <ElementType>, const ndarray &>> operator-() const { return make_unary_expr<std::negate>     (); }
  nc_val_expr<nc_unary_op<nc_bit_not      <ElementType>, const ndarray &>> operator~() const { return make_unary_expr<nc_bit_not>      (); }
  nc_val_expr<nc_unary_op<std::logical_not<ElementType>, const ndarray &>> operator!() const { return make_unary_expr<std::logical_not>(); }

  // computed assignment:
  ndarray &operator=  (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a = b  ; }, x); }
  ndarray &operator+= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a += b ; }, x); }
  ndarray &operator-= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a -= b ; }, x); }
  ndarray &operator*= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a *= b ; }, x); }
  ndarray &operator/= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a /= b ; }, x); }
  ndarray &operator%= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a %= b ; }, x); }
  ndarray &operator&= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a &= b ; }, x); }
  ndarray &operator|= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a |= b ; }, x); }
  ndarray &operator^= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a ^= b ; }, x); }
  ndarray &operator<<=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a <<= b; }, x); }
  ndarray &operator>>=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a >>= b; }, x); }

  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator=  (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator+= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator-= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator*= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator/= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator%= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator&= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator|= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator^= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator<<=(const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> ndarray &operator>>=(const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }
  // clang-format on

  // member functions:
  void swap(ndarray &v) noexcept;

  [[nodiscard]] element_type sum() const;
  [[nodiscard]] element_type min() const;
  [[nodiscard]] element_type max() const;

  //[[nodiscard]] ndarray shift(int i) const;
  //[[nodiscard]] ndarray cshift(int i) const;
  //[[nodiscard]] ndarray apply(value_type __f(value_type)) const;
  //[[nodiscard]] ndarray apply(value_type __f(const value_type&)) const;
  // void resize(size_t n, value_type x = value_type());

private:
  template <class, class, class> friend class ndarray;
  template <class, class, class> friend class slice_view;
  // friend class mask_array;
  // template <class>
  // friend class __mask_expr;
  // template <class>
  // friend class indirect_array;
  // template <class>
  // friend class __indirect_expr;
  template <class> friend class nc_val_expr;

  // template <class Up, class Ex, class Lp>
  // friend Up* begin(ndarray<Up, Ex, Lp>& v);

  // template <class Up, class Ex, class Lp>
  // friend const Up* begin(const ndarray<Up, Ex, Lp>& v);

  // template <class Up, class Ex, class Lp>
  // friend Up* end(ndarray<Up, Ex, Lp>& v);

  // template <class Up, class Ex, class Lp>
  // friend const Up* end(const ndarray<Up, Ex, Lp>& v);

  // void __clear(size_t capacity);
  // ndarray& __assign_range(const value_type* __f, const value_type* __l);

  template <template <typename> class Op> auto make_unary_expr() const {
    using NewExpr = nc_unary_op<Op<value_type>, const ndarray &>;
    return nc_val_expr<NewExpr>(NewExpr(Op<value_type>(), *this));
  }

  template <typename Op>
  ndarray &apply_scalar_op(Op &&op, const value_type &x) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], x);
    return *this;
  }

  template <class Op, class Expr>
  ndarray &apply_expr_op(Op &&op, const Expr &expr) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], expr[i]);
    return *this;
  }

private:
  detail::mdarray<ElementType, Extents, LayoutPolicy,
                  detail::mdarray_container_t<ElementType, Extents>>
      elem_;
};

// template <class Tp, size_t _Size>
// ndarray(const Tp(&)[_Size], size_t) -> ndarray<Tp>;

// extern template void ndarray<size_t>::resize(size_t, size_t);

template <class Op, class Tp, class Ex, class Lp>
struct nc_unary_op<Op, ndarray<Tp, Ex, Lp>> {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  const ndarray<Tp, Ex, Lp> &a0_;

  nc_unary_op(const Op &op, const ndarray<Tp, Ex, Lp> &a0) : op_(op), a0_(a0) {}

  result_type operator[](size_t i) const { return op_(a0_[i]); }

  size_t size() const { return a0_.size(); }
};

template <class Op, class Tp, class Ex, class Lp, class A1>
struct nc_binary_op<Op, ndarray<Tp, Ex, Lp>, A1> {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  const ndarray<Tp, Ex, Lp> &a0_;
  A1 a1_;

  nc_binary_op(const Op &op, const ndarray<Tp, Ex, Lp> &a0, const A1 &a1)
      : op_(op), a0_(a0), a1_(a1) {}

  result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

  size_t size() const { return a0_.size(); }
};

template <class Op, class A0, class Tp, class Ex, class Lp>
struct nc_binary_op<Op, A0, ndarray<Tp, Ex, Lp>> {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  A0 a0_;
  const ndarray<Tp, Ex, Lp> &a1_;

  nc_binary_op(const Op &op, const A0 &a0, const ndarray<Tp, Ex, Lp> &a1)
      : op_(op), a0_(a0), a1_(a1) {}

  result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

  size_t size() const { return a0_.size(); }
};

template <class Op, class Tp1, class Ex1, class Lp1, class Tp2, class Ex2,
          class Lp2>
struct nc_binary_op<Op, ndarray<Tp1, Ex1, Lp1>, ndarray<Tp2, Ex2, Lp2>> {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  const ndarray<Tp1, Ex1, Lp1> &a0_;
  const ndarray<Tp2, Ex2, Lp2> &a1_;

  nc_binary_op(const Op &op, const ndarray<Tp1, Ex1, Lp1> &a0,
               const ndarray<Tp2, Ex2, Lp2> &a1)
      : op_(op), a0_(a0), a1_(a1) {}

  result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

  size_t size() const { return a0_.size(); }
};

// [numcxx.slice_view]
template <class ElementType, class Extents,
          class LayoutPolicy = detail::layout_right>
class slice_view {
public:
  using element_type = ElementType;
  using value_type = std::remove_cv_t<element_type>;

  using extents_type = Extents;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  using layout_type = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  using mdspan_type = detail::mdspan<element_type, extents_type, layout_type>;
  using const_mdspan_type =
      detail::mdspan<const element_type, extents_type, layout_type>;

  using pointer = element_type *;
  using reference = element_type &;
  using const_pointer = const element_type *;
  using const_reference = const element_type &;

public:
  explicit slice_view(mdspan_type span) : span_(span) {}

  // clang-format off
  // element access
  template <typename... Args> decltype(auto) operator()(Args &&...args) const;
  template <typename... Args> decltype(auto) operator()(Args &&...args);

  // element access (flattened)
  [[nodiscard]] const value_type &operator[](size_t i) const { NUMCXX_ASSERT(i < size(), "slice_view::operator[] index out of bounds"); return data_handle()[calc_offset(i)]; }
  [[nodiscard]]       value_type &operator[](size_t i)       { NUMCXX_ASSERT(i < size(), "slice_view::operator[] index out of bounds"); return data_handle()[calc_offset(i)]; }

  // observers
  static constexpr          rank_type                     rank()       noexcept { return extents_type::rank()          ; }
  static constexpr          rank_type             rank_dynamic()       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr          size_type static_extent(size_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr          size_type        extent(size_type r) const noexcept { return span_.extent(r)    ; }
         constexpr          size_type                     size() const noexcept { return span_.size()       ; }
         constexpr               bool                    empty() const noexcept { return span_.empty()      ; }
         constexpr          size_type        stride(rank_type r) const          { return span_.stride(r)    ; }
         constexpr const extents_type&                 extents() const noexcept { return span_.extents()    ; }
         constexpr               auto              data_handle() const noexcept { return span_.data_handle(); }
  // clang-format on

public:
  // clang-format off
  // unary operators:
  nc_val_expr<nc_unary_op<nc_unary_plus   <ElementType>, const slice_view &>> operator+() const { return make_unary_expr<nc_unary_plus>   (); }
  nc_val_expr<nc_unary_op<std::negate     <ElementType>, const slice_view &>> operator-() const { return make_unary_expr<std::negate>     (); }
  nc_val_expr<nc_unary_op<nc_bit_not      <ElementType>, const slice_view &>> operator~() const { return make_unary_expr<nc_bit_not>      (); }
  nc_val_expr<nc_unary_op<std::logical_not<ElementType>, const slice_view &>> operator!() const { return make_unary_expr<std::logical_not>(); }

  // computed assignment:
  slice_view &operator=  (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a = b  ; }, x); }
  slice_view &operator+= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a += b ; }, x); }
  slice_view &operator-= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a -= b ; }, x); }
  slice_view &operator*= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a *= b ; }, x); }
  slice_view &operator/= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a /= b ; }, x); }
  slice_view &operator%= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a %= b ; }, x); }
  slice_view &operator&= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a &= b ; }, x); }
  slice_view &operator|= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a |= b ; }, x); }
  slice_view &operator^= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a ^= b ; }, x); }
  slice_view &operator<<=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a <<= b; }, x); }
  slice_view &operator>>=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a >>= b; }, x); }

  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0> slice_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }
  // clang-format on

private:
  size_type calc_offset(size_type i) const noexcept {
    size_type offset = 0, remaining = i;
    for (rank_type r = rank(); r-- > 0;) {
      offset += (remaining % extent(r)) * span_.stride(r);
      remaining /= extent(r);
    }
    return offset;
  }

  template <template <typename> class Op> auto make_unary_expr() const {
    using NewExpr = nc_unary_op<Op<value_type>, const slice_view &>;
    return nc_val_expr<NewExpr>(NewExpr(Op<value_type>(), *this));
  }

  template <typename Op>
  slice_view &apply_scalar_op(Op &&op, const value_type &x) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], x);
    return *this;
  }

  template <class Op, class Expr>
  slice_view &apply_expr_op(Op &&op, const Expr &expr) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], expr[i]);
    return *this;
  }

private:
  mdspan_type span_;
};

namespace detail {
template <typename MdSpan, typename... Args>
decltype(auto) access_slice(MdSpan &&src, Args &&...args) {
  auto sub_mdspan = detail::slice_utils::make_submdspan(
      std::forward<MdSpan>(src), std::index_sequence_for<Args...>{},
      std::forward<Args>(args)...);
  using sub_mdspan_type = std::decay_t<decltype(sub_mdspan)>;

  if constexpr (sub_mdspan_type::rank() == 0)
    // all dimensions were indexed with integral indices.
    return sub_mdspan();
  else
    return slice_view<typename sub_mdspan_type::element_type,
                      typename sub_mdspan_type::extents_type,
                      typename sub_mdspan_type::layout_type>(sub_mdspan);
}
} // namespace detail

template <typename Tp, typename Ex, typename Lp>
template <typename... Args>
decltype(auto) ndarray<Tp, Ex, Lp>::operator()(Args &&...args) const {
  static_assert(sizeof...(Args) == extents_type::rank(),
                "Number of arguments mush match array rank");
  static_assert(are_all_slice_or_integral_v<Args...>,
                "Each argument must be slice or an integral type");

  return detail::access_slice(elem_.to_mdspan(), std::forward<Args>(args)...);
}

template <typename Tp, typename Ex, typename Lp>
template <typename... Args>
decltype(auto) ndarray<Tp, Ex, Lp>::operator()(Args &&...args) {
  static_assert(sizeof...(Args) == extents_type::rank(),
                "Number of arguments mush match array rank");
  static_assert(are_all_slice_or_integral_v<Args...>,
                "Each argument must be slice or an integral type");

  return detail::access_slice(elem_.to_mdspan(), std::forward<Args>(args)...);
}

template <typename Tp, typename Ex, typename Lp>
template <typename... Args>
decltype(auto) slice_view<Tp, Ex, Lp>::operator()(Args &&...args) const {
  static_assert(sizeof...(Args) == extents_type::rank(),
                "Number of arguments mush match array rank");
  static_assert(are_all_slice_or_integral_v<Args...>,
                "Each argument must be slice or an integral type");

  return detail::access_slice(span_, std::forward<Args>(args)...);
}

template <typename Tp, typename Ex, typename Lp>
template <typename... Args>
decltype(auto) slice_view<Tp, Ex, Lp>::operator()(Args &&...args) {
  static_assert(sizeof...(Args) == extents_type::rank(),
                "Number of arguments mush match array rank");
  static_assert(are_all_slice_or_integral_v<Args...>,
                "Each argument must be slice or an integral type");

  return detail::access_slice(span_, std::forward<Args>(args)...);
}

// slice_array

// template <class Tp>
// class slice_array {
// public:
//     typedef Tp value_type;
//
// private:
//     value_type* vp_;
//     size_t size_;
//     size_t stride_;
//
// public:
//     slice_array(slice_array const&) = default;
//
//     const slice_array& operator=(const slice_array& sa) const;
//
//     void operator=(const value_type& x) const;
//
//     template <class Ex, class Lp>
//     void operator=(const ndarray<value_type, Ex, Lp>& __va) const;
//
//     // Behaves like nc_val_expr::operator[], which returns by value.
//     value_type __get(size_t i) const {
//         _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < size_, "slice_array.__get()
//         index out of bounds"); return vp_[i * stride_];
//     }
//
// private:
//     template <class Ex, class Lp>
//     slice_array(const slice& __sl, const ndarray<value_type, Ex, Lp>& v)
//         : vp_(const_cast<value_type*>(v.begin_ + __sl.start())),
//         size_(__sl.size()), stride_(__sl.stride()) {
//     }
//
//     template <class, class, class>
//     friend class ndarray;
// };
//
// template <class Tp>
// inline const slice_array<Tp>& slice_array<Tp>::operator=(const slice_array&
// sa) const {
//     value_type* t = vp_;
//     const value_type* s = sa.vp_;
//     for (size_t n = size_; n; --n, t += stride_, s += sa.stride_)
//         *t = *s;
//     return *this;
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void slice_array<Tp>::operator=(const Expr& v) const {
//     value_type* t = vp_;
//     for (size_t i = 0; i < size_; ++i, t += stride_)
//         *t = v[i];
// }
//
// template <class Tp>
// template <class Ex, class Lp>
// inline void slice_array<Tp>::operator=(const ndarray<value_type, Ex, Lp>&
// __va) const {
//     value_type* t = vp_;
//     for (size_t i = 0; i < __va.size(); ++i, t += stride_)
//         *t = __va[i];
// }
//
// template <class Tp>
// inline void slice_array<Tp>::operator=(const value_type& x) const {
//     value_type* t = vp_;
//     for (size_t n = size_; n; --n, t += stride_)
//         *t = x;
// }

// mask_array

// template <class Tp>
// class mask_array {
// public:
//     typedef Tp value_type;
//
// private:
//     value_type* vp_;
//     detail::mdarray<size_t, detail::dextents<std::size_t,1>,
//     detail::layout_right> oned_;
//
// public:
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator*=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator/=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator%=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator+=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator-=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator^=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator&=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator|=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator<<=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator>>=(const Expr& v) const;
//
//     mask_array(const mask_array&) = default;
//
//     const mask_array& operator=(const mask_array& ma) const;
//
//     void operator=(const value_type& x) const;
//
//     // Behaves like nc_val_expr::operator[], which returns by value.
//     value_type __get(size_t i) const {
//         _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < oned_.size(),
//         "mask_array.__get() index out of bounds"); return vp_[oned_[i]];
//     }
//
// private:
//     template <class Ex, class Lp>
//     mask_array(const ndarray<bool, Ex, Lp>& __vb, const ndarray<value_type,
//     Ex, Lp>& v)
//         : vp_(const_cast<value_type*>(v.begin_)),
//         oned_(static_cast<size_t>(count(__vb.begin_, __vb.end_, true))) {
//         size_t j = 0;
//         for (size_t i = 0; i < __vb.size(); ++i)
//             if (__vb[i])
//                 oned_[j++] = i;
//     }
//
//     template <class, class, class>
//     friend class ndarray;
// };
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] = v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator*=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] *= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator/=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] /= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator%=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] %= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator+=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] += v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator-=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] -= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator^=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] ^= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator&=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] &= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator|=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] |= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator<<=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] <<= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void mask_array<Tp>::operator>>=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] >>= v[i];
// }
//
// template <class Tp>
// inline const mask_array<Tp>& mask_array<Tp>::operator=(const mask_array& ma)
// const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] = ma.vp_[oned_[i]];
//     return *this;
// }
//
// template <class Tp>
// inline void mask_array<Tp>::operator=(const value_type& x) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] = x;
// }
//
// template <class ValExpr>
// class __mask_expr {
//     typedef std::remove_reference_t<ValExpr> _RmExpr;
//
// public:
//     typedef typename _RmExpr::value_type value_type;
//     typedef value_type result_type;
//
// private:
//     ValExpr expr_;
//     ndarray<size_t, detail::dextents<std::size_t, 1>, detail::layout_right>
//     oned_;
//
//     __mask_expr(const ndarray<bool, detail::dextents<std::size_t, 1>,
//     detail::layout_right>& __vb, const _RmExpr& e)
//         : expr_(e), oned_(static_cast<size_t>(count(__vb.begin_, __vb.end_,
//         true))) { size_t j = 0; for (size_t i = 0; i < __vb.size(); ++i)
//             if (__vb[i])
//                 oned_[j++] = i;
//     }
//
// public:
//     result_type operator[](size_t i) const { return expr_[oned_[i]]; }
//
//     size_t size() const { return oned_.size(); }
//
//     template <class>
//     friend class nc_val_expr;
//     template <class, class, class>
//     friend class ndarray;
// };

// indirect_array

// template <class Tp>
// class indirect_array {
// public:
//     typedef Tp value_type;
//
// private:
//     value_type* vp_;
//     ndarray<size_t, detail::dextents<std::size_t, 1>, detail::layout_right>
//     oned_;
//
// public:
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator*=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator/=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator%=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator+=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator-=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator^=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator&=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator|=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator<<=(const Expr& v) const;
//
//     template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int>
//     = 0> void operator>>=(const Expr& v) const;
//
//     indirect_array(const indirect_array&) = default;
//
//     const indirect_array& operator=(const indirect_array& ia) const;
//
//     void operator=(const value_type& x) const;
//
//     // Behaves like nc_val_expr::operator[], which returns by value.
//     value_type __get(size_t i) const {
//         _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < oned_.size(),
//         "indirect_array.__get() index out of bounds"); return vp_[oned_[i]];
//     }
//
// private:
//     template <class Ex1, class Lp1, class Ex2, class Lp2>
//     indirect_array(const ndarray<size_t, Ex1, Lp1>& ia, const
//     ndarray<value_type, Ex2, Lp2>& v)
//         : vp_(const_cast<value_type*>(v.begin_)), oned_(ia) {
//     }
//
//     template <class Ex1, class Lp1, class Ex2, class Lp2>
//     indirect_array(ndarray<size_t, Ex1, Lp1>&& ia, const ndarray<value_type,
//     Ex2, Lp2>& v)
//         : vp_(const_cast<value_type*>(v.begin_)), oned_(std::move(ia)) {
//     }
//
//     template <class, class, class>
//     friend class ndarray;
// };
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] = v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator*=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] *= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator/=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] /= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator%=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] %= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator+=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] += v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator-=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] -= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator^=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] ^= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator&=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] &= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator|=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] |= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator<<=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] <<= v[i];
// }
//
// template <class Tp>
// template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
// inline void indirect_array<Tp>::operator>>=(const Expr& v) const {
//     size_t n = oned_.size();
//     for (size_t i = 0; i < n; ++i)
//         vp_[oned_[i]] >>= v[i];
// }
//
// template <class Tp>
// inline const indirect_array<Tp>& indirect_array<Tp>::operator=(const
// indirect_array& ia) const {
//     typedef const size_t* _Ip;
//     const value_type* s = ia.vp_;
//     for (_Ip i = oned_.begin_, e = oned_.end_, j = ia.oned_.begin_; i != e;
//     ++i, ++j)
//         vp_[*i] = s[*j];
//     return *this;
// }
//
// template <class Tp>
// inline void indirect_array<Tp>::operator=(const value_type& x) const {
//     typedef const size_t* _Ip;
//     for (_Ip i = oned_.begin_, e = oned_.end_; i != e; ++i)
//         vp_[*i] = x;
// }
//
// template <class ValExpr>
// class __indirect_expr {
//     typedef std::remove_reference_t<ValExpr> _RmExpr;
//
// public:
//     typedef typename _RmExpr::value_type value_type;
//     typedef value_type result_type;
//
// private:
//     ValExpr expr_;
//     ndarray<size_t, detail::dextents<std::size_t, 1>, detail::layout_right>
//     oned_;
//
//     template <class Ex, class Lp>
//     __indirect_expr(const ndarray<size_t, Ex, Lp>& ia, const _RmExpr& e) :
//     expr_(e), oned_(ia) {}
//
//     template <class Ex, class Lp>
//     __indirect_expr(ndarray<size_t, Ex, Lp>&& ia, const _RmExpr& e)
//         : expr_(e), oned_(std::move(ia)) {
//     }
//
// public:
//     result_type operator[](size_t i) const { return expr_[oned_[i]]; }
//
//     size_t size() const { return oned_.size(); }
//
//     template <class>
//     friend class nc_val_expr;
//     template <class, class, class>
//     friend class ndarray;
// };

template <class ValExpr> class nc_val_expr {
  typedef std::remove_reference_t<ValExpr> _RmExpr;

  ValExpr expr_;

public:
  typedef typename _RmExpr::value_type value_type;
  typedef typename _RmExpr::result_type result_type;

  explicit nc_val_expr(const _RmExpr &e) : expr_(e) {}

  result_type operator[](size_t i) const { return expr_[i]; }

  // nc_val_expr<__slice_expr<ValExpr> > operator[](slice s) const {
  //     typedef __slice_expr<ValExpr> NewExpr;
  //     return nc_val_expr< NewExpr >(NewExpr(s, expr_));
  // }

  // template <class Ex, class Lp>
  // nc_val_expr<__mask_expr<ValExpr> > operator[](const ndarray<bool, Ex, Lp>&
  // __vb) const {
  //     typedef __mask_expr<ValExpr> NewExpr;
  //     return nc_val_expr< NewExpr >(NewExpr(__vb, expr_));
  // }

  // template <class Ex, class Lp>
  // nc_val_expr<__indirect_expr<ValExpr> > operator[](const ndarray<size_t, Ex,
  // Lp>& __vs) const {
  //     typedef __indirect_expr<ValExpr> NewExpr;
  //     return nc_val_expr< NewExpr >(NewExpr(__vs, expr_));
  // }

  nc_val_expr<nc_unary_op<nc_unary_plus<value_type>, ValExpr>>
  operator+() const {
    typedef nc_unary_op<nc_unary_plus<value_type>, ValExpr> NewExpr;
    return nc_val_expr<NewExpr>(NewExpr(nc_unary_plus<value_type>(), expr_));
  }

  nc_val_expr<nc_unary_op<std::negate<value_type>, ValExpr>> operator-() const {
    typedef nc_unary_op<std::negate<value_type>, ValExpr> NewExpr;
    return nc_val_expr<NewExpr>(NewExpr(std::negate<value_type>(), expr_));
  }

  nc_val_expr<nc_unary_op<nc_bit_not<value_type>, ValExpr>> operator~() const {
    typedef nc_unary_op<nc_bit_not<value_type>, ValExpr> NewExpr;
    return nc_val_expr<NewExpr>(NewExpr(nc_bit_not<value_type>(), expr_));
  }

  nc_val_expr<nc_unary_op<std::logical_not<value_type>, ValExpr>>
  operator!() const {
    typedef nc_unary_op<std::logical_not<value_type>, ValExpr> NewExpr;
    return nc_val_expr<NewExpr>(NewExpr(std::logical_not<value_type>(), expr_));
  }

  // template<class Ex, class Lp>
  // operator ndarray<nc_val_expr::result_type, Ex, Lp>() const;

  size_t size() const { return expr_.size(); }

  result_type sum() const {
    size_t n = expr_.size();
    result_type r = n ? expr_[0] : result_type();
    for (size_t i = 1; i < n; ++i)
      r += expr_[i];
    return r;
  }

  result_type min() const {
    size_t n = size();
    result_type r = n ? (*this)[0] : result_type();
    for (size_t i = 1; i < n; ++i) {
      result_type x = expr_[i];
      if (x < r)
        r = x;
    }
    return r;
  }

  result_type max() const {
    size_t n = size();
    result_type r = n ? (*this)[0] : result_type();
    for (size_t i = 1; i < n; ++i) {
      result_type x = expr_[i];
      if (r < x)
        r = x;
    }
    return r;
  }

  nc_val_expr<nc_shift_expr<ValExpr>> shift(int i) const {
    return nc_val_expr<nc_shift_expr<ValExpr>>(
        nc_shift_expr<ValExpr>(i, expr_));
  }

  nc_val_expr<nc_cshift_expr<ValExpr>> cshift(int i) const {
    return nc_val_expr<nc_cshift_expr<ValExpr>>(
        nc_cshift_expr<ValExpr>(i, expr_));
  }

  nc_val_expr<nc_unary_op<nc_apply_expr<value_type, value_type (*)(value_type)>,
                          ValExpr>>
  apply(value_type __f(value_type)) const {
    typedef nc_apply_expr<value_type, value_type (*)(value_type)> Op;
    typedef nc_unary_op<Op, ValExpr> NewExpr;
    return nc_val_expr<NewExpr>(NewExpr(Op(__f), expr_));
  }

  nc_val_expr<nc_unary_op<
      nc_apply_expr<value_type, value_type (*)(const value_type &)>, ValExpr>>
  apply(value_type __f(const value_type &)) const {
    typedef nc_apply_expr<value_type, value_type (*)(const value_type &)> Op;
    typedef nc_unary_op<Op, ValExpr> NewExpr;
    return nc_val_expr<NewExpr>(NewExpr(Op(__f), expr_));
  }
};

// template <class ValExpr>
// template <class Ex, class Lp>
// nc_val_expr<ValExpr>::operator ndarray<nc_val_expr::result_type, Ex, Lp>()
// const {
//     ndarray<result_type> r;
//     size_t n = expr_.size();
//     if (n) {
//         r.begin_ = r.end_ = allocator<result_type>().allocate(n);
//         for (size_t i = 0; i != n; ++r.end_, ++i)
//             ::new ((void*)r.end_) result_type(expr_[i]);
//     }
//     return r;
// }

// ndarray

// template <class Tp, class Ex, class Lp>
// inline ndarray<Tp, Ex, Lp>::ndarray(size_t n) : begin_(nullptr),
// end_(nullptr) {
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         for (size_t __n_left = n; __n_left; --__n_left, ++end_)
//             ::new ((void*)end_) value_type();
//         __guard.__complete();
//     }
// }
//
// template <class Tp, class Ex, class Lp>
// inline ndarray<Tp, Ex, Lp>::ndarray(const value_type& x, size_t n) :
// begin_(nullptr), end_(nullptr) {
//     resize(n, x);
// }

// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp>::ndarray(const value_type* p, size_t n) :
// begin_(nullptr), end_(nullptr) {
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         for (size_t __n_left = n; __n_left; ++end_, ++p, --__n_left)
//             ::new ((void*)end_) value_type(*p);
//         __guard.__complete();
//     }
// }

// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp>::ndarray(std::initializer_list<value_type> __il) :
// begin_(nullptr), end_(nullptr) {
//     const size_t n = __il.size();
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         size_t __n_left = n;
//         for (const value_type* p = __il.begin(); __n_left; ++end_, ++p,
//         --__n_left)
//             ::new ((void*)end_) value_type(*p);
//         __guard.__complete();
//     }
// }

// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp>::ndarray(const slice_array<value_type>& sa) :
// begin_(nullptr), end_(nullptr) {
//     const size_t n = sa.size_;
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         size_t __n_left = n;
//         for (const value_type* p = sa.vp_; __n_left; ++end_, p += sa.stride_,
//         --__n_left)
//             ::new ((void*)end_) value_type(*p);
//         __guard.__complete();
//     }
// }
//
// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp>::ndarray(const mask_array<value_type>& ma) :
// begin_(nullptr), end_(nullptr) {
//     const size_t n = ma.oned_.size();
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         typedef const size_t* _Ip;
//         const value_type* s = ma.vp_;
//         for (_Ip i = ma.oned_.begin_, e = ma.oned_.end_; i != e; ++i, ++end_)
//             ::new ((void*)end_) value_type(s[*i]);
//         __guard.__complete();
//     }
// }
//
// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp>::ndarray(const indirect_array<value_type>& ia) :
// begin_(nullptr), end_(nullptr) {
//     const size_t n = ia.oned_.size();
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         typedef const size_t* _Ip;
//         const value_type* s = ia.vp_;
//         for (_Ip i = ia.oned_.begin_, e = ia.oned_.end_; i != e; ++i, ++end_)
//             ::new ((void*)end_) value_type(s[*i]);
//         __guard.__complete();
//     }
// }

// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::__assign_range(const value_type*
// __f, const value_type* __l) {
//     size_t n = __l - __f;
//     if (size() != n) {
//         __clear(size());
//         begin_ = allocator<value_type>().allocate(n);
//         end_ = begin_ + n;
//         std::uninitialized_copy(__f, __l, begin_);
//     }
//     else {
//         std::copy(__f, __l, begin_);
//     }
//     return *this;
// }

// template <class Tp, class Ex, class Lp>
// inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex,
// Lp>::operator=(std::initializer_list<value_type> __il) {
//     return __assign_range(__il.begin(), __il.end());
// }

// template <class Tp, class Ex, class Lp>
// inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const
// slice_array<value_type>& sa) {
//     value_type* t = begin_;
//     const value_type* s = sa.vp_;
//     for (size_t n = sa.size_; n; --n, s += sa.stride_, ++t)
//         *t = *s;
//     return *this;
// }
//
// template <class Tp, class Ex, class Lp>
// inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const
// mask_array<value_type>& ma) {
//     typedef const size_t* _Ip;
//     value_type* t = begin_;
//     const value_type* s = ma.vp_;
//     for (_Ip i = ma.oned_.begin_, e = ma.oned_.end_; i != e; ++i, ++t)
//         *t = s[*i];
//     return *this;
// }
//
// template <class Tp, class Ex, class Lp>
// inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const
// indirect_array<value_type>& ia) {
//     typedef const size_t* _Ip;
//     value_type* t = begin_;
//     const value_type* s = ia.vp_;
//     for (_Ip i = ia.oned_.begin_, e = ia.oned_.end_; i != e; ++i, ++t)
//         *t = s[*i];
//     return *this;
// }

// template <class Tp, class Ex, class Lp>
// inline nc_val_expr<__slice_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex,
// Lp>::operator[](slice s) const {
//     return nc_val_expr<__slice_expr<const ndarray&> >(__slice_expr<const
//     ndarray&>(s, *this));
// }
//
// template <class Tp, class Ex, class Lp>
// inline slice_array<Tp> ndarray<Tp, Ex, Lp>::operator[](slice s) {
//     return slice_array<value_type>(s, *this);
// }
//
// template <class Tp, class Ex, class Lp>
// inline nc_val_expr<__mask_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex,
// Lp>::operator[](const ndarray<bool, Ex, Lp>& __vb) const {
//     return nc_val_expr<__mask_expr<const ndarray&> >(__mask_expr<const
//     ndarray&>(__vb, *this));
// }
//
// template <class Tp, class Ex, class Lp>
// inline mask_array<Tp> ndarray<Tp, Ex, Lp>::operator[](const ndarray<bool, Ex,
// Lp>& __vb) {
//     return mask_array<value_type>(__vb, *this);
// }
//
// template <class Tp, class Ex, class Lp>
// inline nc_val_expr<__mask_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex,
// Lp>::operator[](ndarray<bool, Ex, Lp>&& __vb) const {
//     return nc_val_expr<__mask_expr<const ndarray&> >(__mask_expr<const
//     ndarray&>(std::move(__vb), *this));
// }
//
// template <class Tp, class Ex, class Lp>
// inline mask_array<Tp> ndarray<Tp, Ex, Lp>::operator[](ndarray<bool, Ex, Lp>&&
// __vb) {
//     return mask_array<value_type>(std::move(__vb), *this);
// }
//
// template <class Tp, class Ex, class Lp>
// inline nc_val_expr<__indirect_expr<const ndarray<Tp, Ex, Lp>&> >
// ndarray<Tp, Ex, Lp>::operator[](const ndarray<size_t, Ex, Lp>& __vs) const {
//     return nc_val_expr<__indirect_expr<const ndarray&>
//     >(__indirect_expr<const ndarray&>(__vs, *this));
// }
//
// template <class Tp, class Ex, class Lp>
// inline indirect_array<Tp> ndarray<Tp, Ex, Lp>::operator[](const
// ndarray<size_t, Ex, Lp>& __vs) {
//     return indirect_array<value_type>(__vs, *this);
// }
//
// template <class Tp, class Ex, class Lp>
// inline nc_val_expr<__indirect_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp,
// Ex, Lp>::operator[](ndarray<size_t, Ex, Lp>&& __vs) const {
//     return nc_val_expr<__indirect_expr<const ndarray&>
//     >(__indirect_expr<const ndarray&>(std::move(__vs), *this));
// }
//
// template <class Tp, class Ex, class Lp>
// inline indirect_array<Tp> ndarray<Tp, Ex, Lp>::operator[](ndarray<size_t, Ex,
// Lp>&& __vs) {
//     return indirect_array<value_type>(std::move(__vs), *this);
// }

template <class Tp, class Ex, class Lp>
inline void ndarray<Tp, Ex, Lp>::swap(ndarray &v) noexcept {
  std::swap(elem_, v.elem_);
}

template <class Tp, class Ex, class Lp>
inline Tp ndarray<Tp, Ex, Lp>::sum() const {
  const_pointer first = elem_.data();
  const_pointer last = elem_.data() + elem_.size();
  if (first == last)
    return Tp{};
  const_pointer p = first;
  Tp r = *p;
  for (++p; p != last; ++p)
    r += *p;
  return r;
}

template <class Tp, class Ex, class Lp>
inline Tp ndarray<Tp, Ex, Lp>::min() const {
  const_pointer first = elem_.data();
  const_pointer last = elem_.data() + elem_.size();
  if (first == last)
    return Tp{};
  return *std::min_element(first, last);
}

template <class Tp, class Ex, class Lp>
inline Tp ndarray<Tp, Ex, Lp>::max() const {
  const_pointer first = elem_.data();
  const_pointer last = elem_.data() + elem_.size();
  if (first == last)
    return Tp{};
  return *std::max_element(first, last);
}

// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::shift(int i) const {
//     ndarray<value_type> r;
//     size_t n = size();
//     if (n) {
//         r.begin_ = r.end_ = allocator<value_type>().allocate(n);
//         const value_type* __sb;
//         value_type* __tb;
//         value_type* __te;
//         if (i >= 0) {
//             i = std::min(i, static_cast<int>(n));
//             __sb = begin_ + i;
//             __tb = r.begin_;
//             __te = r.begin_ + (n - i);
//         }
//         else {
//             i = std::min(-i, static_cast<int>(n));
//             __sb = begin_;
//             __tb = r.begin_ + i;
//             __te = r.begin_ + n;
//         }
//         for (; r.end_ != __tb; ++r.end_)
//             ::new ((void*)r.end_) value_type();
//         for (; r.end_ != __te; ++r.end_, ++__sb)
//             ::new ((void*)r.end_) value_type(*__sb);
//         for (__te = r.begin_ + n; r.end_ != __te; ++r.end_)
//             ::new ((void*)r.end_) value_type();
//     }
//     return r;
// }

// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::cshift(int i) const {
//     ndarray<value_type> r;
//     size_t n = size();
//     if (n) {
//         r.begin_ = r.end_ = allocator<value_type>().allocate(n);
//         i %= static_cast<int>(n);
//         const value_type* __m = i >= 0 ? begin_ + i : end_ + i;
//         for (const value_type* s = __m; s != end_; ++r.end_, ++s)
//             ::new ((void*)r.end_) value_type(*s);
//         for (const value_type* s = begin_; s != __m; ++r.end_, ++s)
//             ::new ((void*)r.end_) value_type(*s);
//     }
//     return r;
// }
//
// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::apply(value_type __f(value_type))
// const {
//     ndarray<value_type> r;
//     size_t n = size();
//     if (n) {
//         r.begin_ = r.end_ = allocator<value_type>().allocate(n);
//         for (const value_type* p = begin_; n; ++r.end_, ++p, --n)
//             ::new ((void*)r.end_) value_type(__f(*p));
//     }
//     return r;
// }
//
// template <class Tp, class Ex, class Lp>
// ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::apply(value_type __f(const
// value_type&)) const {
//     ndarray<value_type> r;
//     size_t n = size();
//     if (n) {
//         r.begin_ = r.end_ = allocator<value_type>().allocate(n);
//         for (const value_type* p = begin_; n; ++r.end_, ++p, --n)
//             ::new ((void*)r.end_) value_type(__f(*p));
//     }
//     return r;
// }
//
// template <class Tp, class Ex, class Lp>
// inline void ndarray<Tp, Ex, Lp>::__clear(size_t capacity) {
//     if (begin_ != nullptr) {
//         while (end_ != begin_)
//             (--end_)->~value_type();
//         std::allocator<value_type>().deallocate(begin_, capacity);
//         begin_ = end_ = nullptr;
//     }
// }
//
// template <class Tp, class Ex, class Lp>
// void ndarray<Tp, Ex, Lp>::resize(size_t n, value_type x) {
//     __clear(size());
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         for (size_t __n_left = n; __n_left; --__n_left, ++end_)
//             ::new ((void*)end_) value_type(x);
//         __guard.__complete();
//     }
// }

template <class Tp, class Ex, class Lp>
inline void swap(ndarray<Tp, Ex, Lp> &x, ndarray<Tp, Ex, Lp> &y) noexcept {
  x.swap(y);
}

// clang-format off
#define NUMCXX_MAKE_BINARY_OP(OP, FUNCTOR)                                     \
  template <class Expr1, class Expr2,                                          \
            std::enable_if_t<nc_is_val_expr<Expr1>::value &&                   \
                             nc_is_val_expr<Expr2>::value, int> = 0>           \
  inline nc_val_expr<nc_binary_op<FUNCTOR<typename Expr1::value_type>,         \
                                  Expr1,                                       \
                                  Expr2>>                                      \
  operator OP(const Expr1 & x, const Expr2 & y) {                              \
    typedef typename Expr1::value_type value_type;                             \
    typedef          nc_binary_op<FUNCTOR<value_type>,                         \
                                  Expr1,                                       \
                                  Expr2> Op;                                   \
    return nc_val_expr<Op>    (Op(FUNCTOR<value_type>(),                       \
                                  x,                                           \
                                  y));                                         \
  }                                                                            \
                                                                               \
  template <class Expr,                                                        \
            std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>            \
  inline nc_val_expr<nc_binary_op<FUNCTOR<typename Expr::value_type>,          \
                                  Expr,                                        \
                                  nc_scalar_expr<typename Expr::value_type>>>  \
  operator OP(const Expr & x, const typename Expr::value_type & y) {           \
    typedef typename Expr::value_type value_type;                              \
    typedef          nc_binary_op<FUNCTOR<value_type>,                         \
                                  Expr,                                        \
                                  nc_scalar_expr<value_type>> Op;              \
    return nc_val_expr<Op>    (Op(FUNCTOR<value_type>(),                       \
                                  x,                                           \
                                  nc_scalar_expr<value_type>(y, x.size())));   \
  }                                                                            \
                                                                               \
  template <class Expr,                                                        \
            std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>            \
  inline nc_val_expr<nc_binary_op<FUNCTOR<typename Expr::value_type>,          \
                                  nc_scalar_expr<typename Expr::value_type>,   \
                                  Expr>>                                       \
  operator OP(const typename Expr::value_type & x, const Expr & y) {           \
    typedef typename Expr::value_type value_type;                              \
    typedef          nc_binary_op<FUNCTOR<value_type>,                         \
                                  nc_scalar_expr<value_type>,                  \
                                  Expr> Op;                                    \
    return nc_val_expr<Op>    (Op(FUNCTOR<value_type>(),                       \
                                  nc_scalar_expr<value_type>(x, y.size()),     \
                                  y));                                         \
  }

#define NUMCXX_MAKE_BINARY_FN(FN, FUNCTOR)                                     \
  template <class Expr1, class Expr2,                                          \
            std::enable_if_t<nc_is_val_expr<Expr1>::value &&                   \
                                 nc_is_val_expr<Expr2>::value,                 \
                             int> = 0>                                         \
  inline nc_val_expr<                                                          \
      nc_binary_op<FUNCTOR<typename Expr1::value_type>, Expr1, Expr2>>         \
  FN(const Expr1 &x, const Expr2 &y) {                                         \
    typedef typename Expr1::value_type value_type;                             \
    typedef nc_binary_op<FUNCTOR<value_type>, Expr1, Expr2> Op;                \
    return nc_val_expr<Op>(Op(FUNCTOR<value_type>(), x, y));                   \
  }                                                                            \
                                                                               \
  template <class Expr,                                                        \
            std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>            \
  inline nc_val_expr<nc_binary_op<FUNCTOR<typename Expr::value_type>, Expr,    \
                                  nc_scalar_expr<typename Expr::value_type>>>  \
  FN(const Expr &x, const typename Expr::value_type &y) {                      \
    typedef typename Expr::value_type value_type;                              \
    typedef nc_binary_op<FUNCTOR<value_type>, Expr,                            \
                         nc_scalar_expr<value_type>>                           \
        Op;                                                                    \
    return nc_val_expr<Op>(Op(FUNCTOR<value_type>(), x,                        \
                              nc_scalar_expr<value_type>(y, x.size())));       \
  }                                                                            \
                                                                               \
  template <class Expr,                                                        \
            std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>            \
  inline nc_val_expr<                                                          \
      nc_binary_op<FUNCTOR<typename Expr::value_type>,                         \
                   nc_scalar_expr<typename Expr::value_type>, Expr>>           \
  FN(const typename Expr::value_type &x, const Expr &y) {                      \
    typedef typename Expr::value_type value_type;                              \
    typedef nc_binary_op<FUNCTOR<value_type>, nc_scalar_expr<value_type>,      \
                         Expr>                                                 \
        Op;                                                                    \
    return nc_val_expr<Op>(Op(FUNCTOR<value_type>(),                           \
                              nc_scalar_expr<value_type>(x, y.size()), y));    \
  }

#define NUMCXX_MAKE_UNARY_OP(FN, FUNCTOR)                                      \
  template <class Expr,                                                        \
            std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>            \
  [[nodiscard]] inline nc_val_expr<                                            \
      nc_unary_op<FUNCTOR<typename Expr::value_type>, Expr>>                   \
  FN(const Expr &x) {                                                          \
    typedef typename Expr::value_type value_type;                              \
    typedef nc_unary_op<FUNCTOR<value_type>, Expr> Op;                         \
    return nc_val_expr<Op>(Op(FUNCTOR<value_type>(), x));                      \
  }
// clang-format on

// applies binary operators to each element of two ndarrays, or a ndarray and a
// value
NUMCXX_MAKE_BINARY_OP(+, std::plus)
NUMCXX_MAKE_BINARY_OP(-, std::minus)
NUMCXX_MAKE_BINARY_OP(*, std::multiplies)
NUMCXX_MAKE_BINARY_OP(/, std::divides)
NUMCXX_MAKE_BINARY_OP(%, std::modulus)
NUMCXX_MAKE_BINARY_OP(&, std::bit_and)
NUMCXX_MAKE_BINARY_OP(|, std::bit_or)
NUMCXX_MAKE_BINARY_OP(^, std::bit_xor)
NUMCXX_MAKE_BINARY_OP(<<, nc_bit_shift_left)
NUMCXX_MAKE_BINARY_OP(>>, nc_bit_shift_right)
NUMCXX_MAKE_BINARY_OP(&&, std::logical_and)
NUMCXX_MAKE_BINARY_OP(||, std::logical_or)

// compares two ndarrays or a ndarray with a value
NUMCXX_MAKE_BINARY_OP(==, std::equal_to)
NUMCXX_MAKE_BINARY_OP(!=, std::not_equal_to)
NUMCXX_MAKE_BINARY_OP(<, std::less)
NUMCXX_MAKE_BINARY_OP(<=, std::less_equal)
NUMCXX_MAKE_BINARY_OP(>, std::greater)
NUMCXX_MAKE_BINARY_OP(>=, std::greater_equal)

// absolute function
NUMCXX_MAKE_UNARY_OP(abs, nc_abs_expr)

// exponential functions
NUMCXX_MAKE_UNARY_OP(exp, nc_exp_expr)
NUMCXX_MAKE_UNARY_OP(log, nc_log_expr)
NUMCXX_MAKE_UNARY_OP(log10, nc_log10_expr)

// power function
NUMCXX_MAKE_BINARY_FN(pow, nc_pow_expr)
NUMCXX_MAKE_UNARY_OP(sqrt, nc_sqrt_expr)

// trigonometric functions
NUMCXX_MAKE_UNARY_OP(sin, nc_sin_expr)
NUMCXX_MAKE_UNARY_OP(cos, nc_cos_expr)
NUMCXX_MAKE_UNARY_OP(tan, nc_tan_expr)
NUMCXX_MAKE_UNARY_OP(asin, nc_asin_expr)
NUMCXX_MAKE_UNARY_OP(acos, nc_acos_expr)
NUMCXX_MAKE_UNARY_OP(atan, nc_atan_expr)
NUMCXX_MAKE_BINARY_FN(atan2, nc_atan2_expr)

// hyperbolic functions
NUMCXX_MAKE_UNARY_OP(sinh, nc_sinh_expr)
NUMCXX_MAKE_UNARY_OP(cosh, nc_cosh_expr)
NUMCXX_MAKE_UNARY_OP(tanh, nc_tanh_expr)

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline Tp *begin(ndarray<Tp, Ex, Lp> &v) {
  return v.data();
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline const Tp *begin(const ndarray<Tp, Ex, Lp> &v) {
  return v.data();
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline Tp *end(ndarray<Tp, Ex, Lp> &v) {
  return v.data() + v.size();
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline const Tp *end(const ndarray<Tp, Ex, Lp> &v) {
  return v.data() + v.size();
}

namespace detail {
template <size_t Rank>
auto make_extents(const std::initializer_list<size_t> &shape) {
  if (shape.size() != Rank) {
    NUMCXX_THROW(std::invalid_argument, "shape size does not match array rank");
  }
  std::array<size_t, Rank> dims;
  std::copy(shape.begin(), shape.end(), dims.begin());
  return extents(dims);
}
} // namespace detail

//
// [numcxx.factory] ndarray creation
//
template <typename T>
ndarray<T, dextents<1>> arange(T start, T stop, T step = T(1)) {
  static_assert(std::is_arithmetic_v<T>,
                "arange requires an arithmetic value type");

  const size_t n = std::ceil((stop - start) / step);

  ndarray<T, dextents<1>> arr({n});
  T *p = arr.data();

  T value = start;
  for (size_t i = 0; i < n; ++i) {
    p[i] = value;
    value += step;
  }

  return arr;
}

template <typename T> ndarray<T, dextents<1>> arange(T stop) {
  return arange<T>(T(0), stop, T(1));
}

template <typename NdArrayType,
          typename = std::enable_if_t<detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType ones() {
  NdArrayType arr;
  std::fill_n(arr.data(), arr.size(), typename NdArrayType::value_type(1));
  return arr;
}

template <typename NdArrayType, typename = std::enable_if_t<
                                    !detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType ones(std::initializer_list<size_t> shape) {
  constexpr size_t rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  std::fill_n(arr.data(), arr.size(), typename NdArrayType::value_type(1));
  return arr;
}

template <typename NdArrayType,
          typename = std::enable_if_t<detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType zeros() {
  NdArrayType arr;
  std::fill_n(arr.data(), arr.size(), typename NdArrayType::value_type(0));
  return arr;
}

template <typename NdArrayType, typename = std::enable_if_t<
                                    !detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType zeros(std::initializer_list<size_t> shape) {
  constexpr size_t rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  std::fill_n(arr.data(), arr.size(), typename NdArrayType::value_type(0));
  return arr;
}

//
// [numcxx.random] random number generation
//
namespace random {
inline std::mt19937 &get_engine() {
  static thread_local std::mt19937 engine(std::random_device{}());
  return engine;
}

inline void seed(unsigned int value) { get_engine().seed(value); }

namespace detail {
using ::numcxx::detail::is_static_ndarray_v;
using ::numcxx::detail::make_extents;

template <typename NdArrayType, typename Distribution>
void fill_random(NdArrayType &arr, Distribution &&dist) {
  auto &engine = get_engine();
  for (size_t i = 0; i < arr.size(); ++i) {
    arr.data()[i] = dist(engine);
  }
}
} // namespace detail

// ---------- 1. rand: uniform [0,1) ----------
template <typename NdArrayType,
          typename = std::enable_if_t<detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType rand() {
  NdArrayType arr;
  detail::fill_random(arr, std::uniform_real_distribution<double>(0.0, 1.0));
  return arr;
}

template <typename NdArrayType, typename = std::enable_if_t<
                                    !detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType rand(std::initializer_list<size_t> shape) {
  constexpr size_t rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  detail::fill_random(arr, std::uniform_real_distribution<double>(0.0, 1.0));
  return arr;
}

// ---------- 2. randn: standard normal ----------
template <typename NdArrayType,
          typename = std::enable_if_t<detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType randn() {
  NdArrayType arr;
  detail::fill_random(arr, std::normal_distribution<double>(0.0, 1.0));
  return arr;
}

template <typename NdArrayType, typename = std::enable_if_t<
                                    !detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType randn(std::initializer_list<size_t> shape) {
  constexpr size_t rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  detail::fill_random(arr, std::normal_distribution<double>(0.0, 1.0));
  return arr;
}

// ---------- 3. uniform [low, high) ----------
template <typename NdArrayType,
          typename = std::enable_if_t<detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType uniform(double low, double high) {
  NdArrayType arr;
  detail::fill_random(arr, std::uniform_real_distribution<double>(low, high));
  return arr;
}

template <typename NdArrayType, typename = std::enable_if_t<
                                    !detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType uniform(double low, double high,
                    std::initializer_list<size_t> shape) {
  constexpr size_t rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  detail::fill_random(arr, std::uniform_real_distribution<double>(low, high));
  return arr;
}

// ---------- 4. randint [low, high) ----------
template <typename NdArrayType,
          typename = std::enable_if_t<detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType randint(int low, int high) {
  NdArrayType arr;
  detail::fill_random(arr, std::uniform_int_distribution<int>(low, high - 1));
  return arr;
}

template <typename NdArrayType, typename = std::enable_if_t<
                                    !detail::is_static_ndarray_v<NdArrayType>>>
NdArrayType randint(int low, int high, std::initializer_list<size_t> shape) {
  constexpr size_t rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  detail::fill_random(arr, std::uniform_int_distribution<int>(low, high - 1));
  return arr;
}

} // namespace random

//
// [numcxx.linalg] linear algebra (TODO)
//
namespace linalg {}

//
// [numcxx.aliases] type aliases
//
// clang-format off
template <typename T> using vec  = numcxx::ndarray<T, dextents<1>>;
template <typename T> using mat  = numcxx::ndarray<T, dextents<2>>;
template <typename T> using cube = numcxx::ndarray<T, dextents<3>>;

// int
using ivec  =  vec<int>;
using imat  =  mat<int>;
using icube = cube<int>;

// unsigned int
using uvec  =  vec<unsigned>;
using umat  =  mat<unsigned>;
using ucube = cube<unsigned>;

// double
using dvec  =  vec<double>;
using dmat  =  mat<double>; 
using dcube = cube<double>;

// float
using fvec  =  vec<float>;
using fmat  =  mat<float>;
using fcube = cube<float>;

template<class T, size_type N>                           using  vec_fixed = ndarray<T, extents<N>>      ;
template<class T, size_type M, size_type N>              using  mat_fixed = ndarray<T, extents<M, N>>   ;
template<class T, size_type M, size_type N, size_type K> using cube_fixed = ndarray<T, extents<M, N, K>>;

// int
using ivec2    =  vec_fixed<int,      2>      ; using ivec3    =  vec_fixed<int,      3>      ; using ivec4    =  vec_fixed<int,      4>      ;
using imat22   =  mat_fixed<int,      2, 2>   ; using imat33   =  mat_fixed<int,      3, 3>   ; using imat44   =  mat_fixed<int,      4, 4>   ;
using icube222 = cube_fixed<int,      2, 2, 2>; using icube333 = cube_fixed<int,      3, 3, 3>; using icube444 = cube_fixed<int,      4, 4, 4>;

// unsigned int
using uvec2    =  vec_fixed<unsigned, 2>      ; using uvec3    =  vec_fixed<unsigned, 3>      ; using uvec4    =  vec_fixed<unsigned, 4>      ;
using umat22   =  mat_fixed<unsigned, 2, 2>   ; using umat33   =  mat_fixed<unsigned, 3, 3>   ; using umat44   =  mat_fixed<unsigned, 4, 4>   ;
using ucube222 = cube_fixed<unsigned, 2, 2, 2>; using ucube333 = cube_fixed<unsigned, 3, 3, 3>; using ucube444 = cube_fixed<unsigned, 4, 4, 4>;

// double
using dvec2    =  vec_fixed<double,   2>      ; using dvec3    =  vec_fixed<double,   3>      ; using dvec4    =  vec_fixed<double,   4>      ;
using dmat22   =  mat_fixed<double,   2, 2>   ; using dmat33   =  mat_fixed<double,   3, 3>   ; using dmat44   =  mat_fixed<double,   4, 4>   ;
using dcube222 = cube_fixed<double,   2, 2, 2>; using dcube333 = cube_fixed<double,   3, 3, 3>; using dcube444 = cube_fixed<double,   4, 4, 4>;

// float
using fvec2    =  vec_fixed<float,    2>      ; using fvec3    =  vec_fixed<float,    3>      ; using fvec4    =  vec_fixed<float,    4>      ;
using fmat22   =  mat_fixed<float,    2, 2>   ; using fmat33   =  mat_fixed<float,    3, 3>   ; using fmat44   =  mat_fixed<float,    4, 4>   ;
using fcube222 = cube_fixed<float,    2, 2, 2>; using fcube333 = cube_fixed<float,    3, 3, 3>; using fcube444 = cube_fixed<float,    4, 4, 4>;
// clang-format on

} // namespace numcxx

#endif // NUMCXX_H_