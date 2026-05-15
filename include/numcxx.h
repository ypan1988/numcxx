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

// clang-format off
#if __cplusplus >= 202600L
#define NUMCXX_USE_STD 1
#include <mdarray>
#include <mdspan>
#include <linalg>
#else
#define NUMCXX_USE_STD 0
#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>
#include <experimental/linalg>
#endif
// clang-format on

namespace numcxx {
using size_type = std::size_t;
using index_type = std::ptrdiff_t;

namespace detail {
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
namespace linalg = std::linalg;
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

namespace stdblas_ns =
    MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
namespace linalg = stdblas_ns::linalg;

#endif
} // namespace detail

// clang-format off
template <size_type    Rank   > using dextents     = detail::dextents<size_type, Rank      >;
template <size_type... Extents> using extents      = detail::extents <size_type, Extents...>;
                                using layout_left  = detail::layout_left ;
                                using layout_right = detail::layout_right;
// clang-format on
} // namespace numcxx

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
                                        class         slice;
template <class Tp, class Ex, class Lp> class       ndarray;
template <class Tp, class Ex, class Lp> class    slice_view;
template <class Tp>                     class     mask_view;
template <class Tp>                     class indirect_view;

template <class ValExpr>                class  nc_val_expr ;
template <class Op, class A0>           struct nc_unary_op ;
template <class Op, class A0, class A1> struct nc_binary_op;

template <class ValExpr>                struct nc_is_val_expr                            : std::false_type {};
template <class ValExpr>                struct nc_is_val_expr<nc_val_expr  <ValExpr   >> : std::true_type  {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<ndarray      <Tp, Ex, Lp>> : std::true_type  {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<slice_view   <Tp, Ex, Lp>> : std::true_type  {};
template <class Tp>                     struct nc_is_val_expr<mask_view    <Tp>        > : std::true_type  {};
template <class Tp>                     struct nc_is_val_expr<indirect_view<Tp>        > : std::true_type  {};

// mdspan-like types are ultimately backed by mdarray/mdspan and provide extents and layout semantics.
template <class Tp>                     struct nc_mdspan_like                            : std::false_type {};
template <class Tp, class Ex, class Lp> struct nc_mdspan_like<ndarray      <Tp, Ex, Lp>> : std::true_type  {};
template <class Tp, class Ex, class Lp> struct nc_mdspan_like<slice_view   <Tp, Ex, Lp>> : std::true_type  {};
template <class Tp>                     struct nc_mdspan_like<mask_view    <Tp>        > : std::true_type  {};
template <class Tp>                     struct nc_mdspan_like<indirect_view<Tp>        > : std::true_type  {};
template <class Op, class A0>           struct nc_mdspan_like<nc_unary_op  <Op, A0>    > : nc_mdspan_like<std::decay_t<A0>> {};
template <class Op, class A0, class A1> struct nc_mdspan_like<nc_binary_op <Op, A0, A1>> : std::bool_constant<nc_mdspan_like<std::decay_t<A0>>::value  ||
                                                                                                              nc_mdspan_like<std::decay_t<A1>>::value> {};

template <class Tp> inline constexpr bool nc_is_val_expr_v = nc_is_val_expr<Tp>::value;
template <class Tp> inline constexpr bool nc_mdspan_like_v = nc_mdspan_like<Tp>::value;

template <class Tp, class Ex, class Lp> const Tp *begin(const ndarray<Tp, Ex, Lp> &v);
template <class Tp, class Ex, class Lp>       Tp *begin(      ndarray<Tp, Ex, Lp> &v);
template <class Tp, class Ex, class Lp> const Tp *end  (const ndarray<Tp, Ex, Lp> &v);
template <class Tp, class Ex, class Lp>       Tp *end  (      ndarray<Tp, Ex, Lp> &v);
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
    NUMCXX_ASSERT(step != 0, "slice step cannot be zero");
  }

  [[nodiscard]] std::optional<index_type> start() const { return start_; }
  [[nodiscard]] std::optional<index_type> stop() const { return stop_; }
  [[nodiscard]] index_type step() const { return step_; }

  friend bool operator==(const slice &x, const slice &y) {
    return x.start() == y.start() && x.stop() == y.stop() &&
           x.step() == y.step();
  }
};

// clang-format off
template <class Tp> struct nc_unary_plus { typedef Tp result_type; Tp operator()(const Tp &x) const { return +x; } };
template <class Tp> struct nc_bit_not    { typedef Tp result_type; Tp operator()(const Tp &x) const { return ~x; } };

template <class T> struct is_slice_or_integral : std::bool_constant<std::is_same_v<std::decay_t<T>, slice> || std::is_integral_v<std::decay_t<T>>> {};
template <class T>       inline constexpr bool is_slice_or_integral_v = is_slice_or_integral<T>::value;
template <class... Args> inline constexpr bool are_all_slice_or_integral_v = (is_slice_or_integral_v<Args> && ...);

namespace detail {

template <class Ex>          inline constexpr bool is_static_extents_v = Ex::rank_dynamic() == 0;
template <class NdArrayType> inline constexpr bool is_static_ndarray_v = is_static_extents_v<typename NdArrayType::extents_type>;

template <class Ex>          struct static_extents_size;
template <size_type... Dims> struct static_extents_size<numcxx::extents<Dims...>> : std::integral_constant<size_type, (Dims * ...)> {};

template <class Tp, class Ex, bool IsStaticExtents = is_static_extents_v<Ex>> struct mdarray_container_selector;
template <class Tp, class Ex> struct mdarray_container_selector<Tp, Ex, true > { using type = std::array <Tp, static_extents_size<Ex>::value>; };
template <class Tp, class Ex> struct mdarray_container_selector<Tp, Ex, false> { using type = std::vector<Tp>                                ; };
template <class Tp, class Ex> using  mdarray_container_t = typename mdarray_container_selector<Tp, Ex>::type;

template <class T, class = void> struct is_boolean_expr                                                                                      : std::false_type {};
template <class T>               struct is_boolean_expr<T, std::void_t<decltype(static_cast<bool>(std::declval<const T &>()[size_type{}]))>> : std::true_type  {};

template <class Op, class Expr> auto make_unary_op(const Expr &);

namespace slice_utils {
inline size_type to_submdspan_arg(index_type idx, size_type dim_len) {
  index_type res = (idx < 0) ? idx + static_cast<index_type>(dim_len) : idx;
  NUMCXX_ASSERT(res >= 0,
                static_cast<size_type>(res) < dim_len && "Index out of bounds");
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
  NUMCXX_ASSERT((diff > 0 && step > 0) || (diff < 0 && step < 0),
                "invalid slice");

  size_type offset = static_cast<size_type>(start);
  size_type extent = (diff / step) + ((diff % step) != 0 ? 1 : 0);
  size_type stride = static_cast<size_type>(std::abs(s.step()));

  return ::numcxx::detail::strided_slice<size_type, size_type, size_type>{
      offset, extent, stride};
}

template <typename MdSpan, typename... Args, size_type... Is>
auto make_submdspan(MdSpan &&src, std::index_sequence<Is...>, Args &&...args) {
  std::array<size_type, sizeof...(Args)> dims = {src.extent(Is)...};
  return detail::submdspan(std::forward<MdSpan>(src),
                           to_submdspan_arg(args, dims[Is])...);
}

} // namespace slice_utils

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
// clang-format on

// [numcxx.ndarray]
template <class ElementType, class Extents,
          class LayoutPolicy = detail::layout_right>
class ndarray {
public:
  // clang-format off
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  using extents_type = Extents;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  using  layout_type = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  using const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type>;
  using       mdspan_type = detail::mdspan<      element_type, extents_type, layout_type>;

  using const_reference = typename const_mdspan_type::reference;
  using       reference = typename       mdspan_type::reference;
  using const_pointer   =          const         element_type *;
  using       pointer   =                        element_type *;

  // construct/destroy:
  constexpr ndarray() = default;
  constexpr ndarray(const ndarray &v) = default;
  constexpr ndarray(ndarray &&v) noexcept = default;
  template <class... SizeTypes> explicit constexpr ndarray(SizeTypes... dyn_exts) : elem_(Extents(dyn_exts...)) {}
  ndarray(const    slice_view<ElementType, Extents, LayoutPolicy> &sv) : elem_(sv.to_mdspan()) {}
  ndarray(const     mask_view<ElementType>                        &mv) : elem_(mv.to_mdspan()) {}
  ndarray(const indirect_view<ElementType>                        &iv) : elem_(iv.to_mdspan()) {}
  // template <class Expr, std::enable_if_t<nc_is_val_expr<std::decay_t<Expr>>::value, int> = 0> explicit ndarray(const Expr& expr); // TODO
  ~ndarray() = default;
  // ndarray(std::initializer_list<value_type> __il);

  // assignment:
  constexpr ndarray &operator=(const ndarray &v) = default;
  constexpr ndarray &operator=(ndarray &&v) noexcept = default;
  // ndarray& operator=(std::initializer_list<value_type>);

  // element access (flattened)
  [[nodiscard]] const value_type &operator[](size_type i) const { NUMCXX_ASSERT(i < size(), "ndarray::operator[] index out of bounds"); return data()[i]; }
  [[nodiscard]]       value_type &operator[](size_type i)       { NUMCXX_ASSERT(i < size(), "ndarray::operator[] index out of bounds"); return data()[i]; }

  // subset operations (slice_view):
  template <typename... Args>
  [[nodiscard]] decltype(auto)
  operator()(Args &&...args) const {
    static_assert(sizeof...(Args) == extents_type::rank(), "Number of arguments must match array rank");
    static_assert(are_all_slice_or_integral_v<Args...>, "Each argument must be slice or an integral type");
    return detail::access_slice(to_mdspan(), std::forward<Args>(args)...);
  }
  template <typename... Args>
  [[nodiscard]] decltype(auto)
  operator()(Args &&...args) {
    static_assert(sizeof...(Args) == extents_type::rank(), "Number of arguments must match array rank");
    static_assert(are_all_slice_or_integral_v<Args...>, "Each argument must be slice or an integral type");
    return detail::access_slice(to_mdspan(), std::forward<Args>(args)...);
  }

  // subset operations (mask_view):
  template <typename BoolExpr, std::enable_if_t<detail::is_boolean_expr<std::decay_t<BoolExpr>>::value, int> = 0>
  [[nodiscard]] mask_view<element_type>
  operator[](BoolExpr &&expr) const {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return mask_view<element_type>(to_mdspan(), std::forward<BoolExpr>(expr));
  }
  template <typename BoolExpr, std::enable_if_t<detail::is_boolean_expr<std::decay_t<BoolExpr>>::value, int> = 0>
  [[nodiscard]] mask_view<element_type>
  operator[](BoolExpr &&expr) {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return mask_view<element_type>(to_mdspan(), std::forward<BoolExpr>(expr));
  }

  // subset operations (indirect_view):
  [[nodiscard]] indirect_view<element_type>
  operator[](std::initializer_list<size_type> offsets) const {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets));
  }
  [[nodiscard]] indirect_view<element_type>
  operator[](std::initializer_list<size_type> offsets) {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets));
  }

  [[nodiscard]] indirect_view<element_type>
  operator[](const ndarray<size_type, dextents<1>> &offsets) const {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets.data(), offsets.data() + offsets.size()));
  }
  [[nodiscard]] indirect_view<element_type>
  operator[](const ndarray<size_type, dextents<1>> &offsets) {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets.data(), offsets.data() + offsets.size()));
  }

  // unary operators:
  auto operator+() const { return apply_unary_op<nc_unary_plus>   (); }
  auto operator-() const { return apply_unary_op<std::negate>     (); }
  auto operator~() const { return apply_unary_op<nc_bit_not>      (); }
  auto operator!() const { return apply_unary_op<std::logical_not>(); }

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

  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator=  (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator+= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator-= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator*= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator/= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator%= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator&= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator|= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator^= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator<<=(const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> ndarray &operator>>=(const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

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

  // mdspan interoperability
         constexpr  const_mdspan_type                to_mdspan() const noexcept { return elem_.to_mdspan(); }
         constexpr        mdspan_type                to_mdspan()       noexcept { return elem_.to_mdspan(); }

  // clang-format on

  // member functions:
  void swap(ndarray &v) noexcept;

  [[nodiscard]] value_type sum() const {
    return std::accumulate(data(), data() + size(), value_type{});
  }
  [[nodiscard]] value_type min() const {
    return empty() ? value_type{} : *std::min_element(data(), data() + size());
  }
  [[nodiscard]] value_type max() const {
    return empty() ? value_type{} : *std::max_element(data(), data() + size());
  }

  // void resize(size_type n, value_type x = value_type());

private:
  // clang-format off
  template <class, class, class> friend class       ndarray;
  template <class, class, class> friend class    slice_view;
  template <class>               friend class     mask_view;
  template <class>               friend class indirect_view;
  template <class>               friend class   nc_val_expr;

  template <class Up, class Ex, class Lp> friend const Up* begin(const ndarray<Up, Ex, Lp> &v);
  template <class Up, class Ex, class Lp> friend       Up* begin(      ndarray<Up, Ex, Lp> &v);
  template <class Up, class Ex, class Lp> friend const Up* end  (const ndarray<Up, Ex, Lp> &v);
  template <class Up, class Ex, class Lp> friend       Up* end  (      ndarray<Up, Ex, Lp> &v);
  // clang-format on

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
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

// template <class Tp, size_type _Size>
// ndarray(const Tp(&)[_Size], size_type) -> ndarray<Tp>;

// extern template void ndarray<size_type>::resize(size_type, size_type);

// [numcxx.slice_view]
template <class ElementType, class Extents,
          class LayoutPolicy = detail::layout_right>
class slice_view {
public:
  // clang-format off
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  using extents_type = Extents;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  using  layout_type = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  using const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type>;
  using       mdspan_type = detail::mdspan<      element_type, extents_type, layout_type>;

  using        const_reference = typename const_mdspan_type::reference;
  using              reference = typename       mdspan_type::reference;
  using const_data_handle_type = typename const_mdspan_type::data_handle_type;
  using       data_handle_type = typename       mdspan_type::data_handle_type;

  // construct/destroy:
  slice_view() = delete;
  slice_view(const slice_view &) = default;
  slice_view(slice_view &&) noexcept = default;
  explicit slice_view(mdspan_type span) : span_(span) {}
  ~slice_view() = default;

  // assignment:
  slice_view &operator=(const slice_view &) = default;
  slice_view &operator=(slice_view &&) noexcept = default;

  // element access (flattened)
  [[nodiscard]] const value_type &operator[](size_type i) const { NUMCXX_ASSERT(i < size(), "slice_view::operator[] index out of bounds"); return data_handle()[calc_offset(i)]; }
  [[nodiscard]]       value_type &operator[](size_type i)       { NUMCXX_ASSERT(i < size(), "slice_view::operator[] index out of bounds"); return data_handle()[calc_offset(i)]; }

  // subset operations (slice_view)
  template <typename... Args> decltype(auto) operator()(Args &&...args) const {
    static_assert(sizeof...(Args) == extents_type::rank(), "Number of arguments must match array rank");
    static_assert(are_all_slice_or_integral_v<Args...>, "Each argument must be slice or an integral type");
    return detail::access_slice(span_, std::forward<Args>(args)...);
  }
  template <typename... Args> decltype(auto) operator()(Args &&...args) {
    static_assert(sizeof...(Args) == extents_type::rank(), "Number of arguments must match array rank");
    static_assert(are_all_slice_or_integral_v<Args...>, "Each argument must be slice or an integral type");
    return detail::access_slice(span_, std::forward<Args>(args)...);
  }

  // subset operations (mask_view):
  template <typename BoolExpr, std::enable_if_t<detail::is_boolean_expr<std::decay_t<BoolExpr>>::value, int> = 0>
  [[nodiscard]] mask_view<element_type>
  operator[](BoolExpr &&expr) const {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return mask_view<element_type>(to_mdspan(), std::forward<BoolExpr>(expr));
  }
  template <typename BoolExpr, std::enable_if_t<detail::is_boolean_expr<std::decay_t<BoolExpr>>::value, int> = 0>
  [[nodiscard]] mask_view<element_type>
  operator[](BoolExpr &&expr) {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return mask_view<element_type>(to_mdspan(), std::forward<BoolExpr>(expr));
  }

  // subset operations (indirect_view):
  [[nodiscard]] indirect_view<element_type>
  operator[](std::initializer_list<size_type> offsets) const {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets));
  }
  [[nodiscard]] indirect_view<element_type>
  operator[](std::initializer_list<size_type> offsets) {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets));
  }

  [[nodiscard]] indirect_view<element_type>
  operator[](const ndarray<size_type, dextents<1>> &offsets) const {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets.data(), offsets.data() + offsets.size()));
  }
  [[nodiscard]] indirect_view<element_type>
  operator[](const ndarray<size_type, dextents<1>> &offsets) {
    return indirect_view<element_type>(to_mdspan(), std::vector<size_type>(offsets.data(), offsets.data() + offsets.size()));
  }

  // unary operators:
  auto operator+() const { return apply_unary_op<nc_unary_plus>   (); }
  auto operator-() const { return apply_unary_op<std::negate>     (); }
  auto operator~() const { return apply_unary_op<nc_bit_not>      (); }
  auto operator!() const { return apply_unary_op<std::logical_not>(); }

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

  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> slice_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

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

  // mdspan interoperability
         constexpr  const_mdspan_type                to_mdspan() const noexcept { return span_              ; }
         constexpr        mdspan_type                to_mdspan()       noexcept { return span_              ; }
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

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
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

template <class Tp> class index_accessor {
public:
  using element_type = Tp;
  using reference = element_type &;
  using data_handle_type = element_type *;
  using offset_policy = index_accessor;

  index_accessor() = default;
  explicit index_accessor(const std::vector<size_type> &indices)
      : indices_(indices) {}

  reference access(const data_handle_type p, const size_type i) const noexcept {
    NUMCXX_ASSERT(i < indices_.size(),
                  "index_accessor::access: index out of bounds");
    return p[indices_[i]];
  }

  data_handle_type offset(const data_handle_type p,
                          const size_type i) const noexcept {
    return p + indices_[i];
  }

  friend bool operator==(const index_accessor &a,
                         const index_accessor &b) noexcept {
    return a.indices_ == b.indices_;
  }

  friend bool operator!=(const index_accessor &a,
                         const index_accessor &b) noexcept {
    return !(a == b);
  }

private:
  std::vector<size_type> indices_;
};

} // namespace detail

// [numcxx.mask_view]
template <class ElementType> class mask_view {
public:
  // clang-format off
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  using extents_type = dextents<1>;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  using         layout_type = detail::layout_right;
  using const_accessor_type = detail::index_accessor<const element_type>;
  using       accessor_type = detail::index_accessor<      element_type>;

  using   const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type, const_accessor_type>;
  using         mdspan_type = detail::mdspan<      element_type, extents_type, layout_type,       accessor_type>;

  using        const_reference = typename const_mdspan_type::reference;
  using              reference = typename       mdspan_type::reference;
  using const_data_handle_type = typename const_mdspan_type::data_handle_type;
  using       data_handle_type = typename       mdspan_type::data_handle_type;

  // construct/destroy:
  mask_view() = delete;
  mask_view(const mask_view &) = default;
  mask_view(mask_view &&) noexcept = default;
  ~mask_view() = default;

  // element access
  [[nodiscard]] const value_type &operator[](size_type i) const { NUMCXX_ASSERT(i < size(), "mask_view::operator[] index out of bounds"); return span_[i]; }
  [[nodiscard]]       value_type &operator[](size_type i)       { NUMCXX_ASSERT(i < size(), "mask_view::operator[] index out of bounds"); return span_[i]; }

  // unary operators:
  auto operator+() const { return apply_unary_op<nc_unary_plus>   (); }
  auto operator-() const { return apply_unary_op<std::negate>     (); }
  auto operator~() const { return apply_unary_op<nc_bit_not>      (); }
  auto operator!() const { return apply_unary_op<std::logical_not>(); }

  // computed assignment:
  mask_view &operator=  (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a = b  ; }, x); }
  mask_view &operator+= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a += b ; }, x); }
  mask_view &operator-= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a -= b ; }, x); }
  mask_view &operator*= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a *= b ; }, x); }
  mask_view &operator/= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a /= b ; }, x); }
  mask_view &operator%= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a %= b ; }, x); }
  mask_view &operator&= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a &= b ; }, x); }
  mask_view &operator|= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a |= b ; }, x); }
  mask_view &operator^= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a ^= b ; }, x); }
  mask_view &operator<<=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a <<= b; }, x); }
  mask_view &operator>>=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a >>= b; }, x); }

  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> mask_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

  // observers
  static constexpr          rank_type                     rank()       noexcept { return extents_type::rank()          ; }
  static constexpr          rank_type             rank_dynamic()       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr          size_type static_extent(size_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr          size_type        extent(size_type r) const noexcept { return span_.extent(r); }
         constexpr          size_type                     size() const noexcept { return span_.size()   ; }
         constexpr               bool                    empty() const noexcept { return size() == 0    ; }
         constexpr const extents_type&                 extents() const noexcept { return span_.extents(); }

  // mdspan interoperability
         constexpr  const_mdspan_type                to_mdspan() const noexcept { return span_          ; }
         constexpr        mdspan_type                to_mdspan()       noexcept { return span_          ; }
  // clang-format on

private:
  template <typename MdSpan, typename BoolExpr>
  explicit mask_view(const MdSpan &data_span, const BoolExpr &expr) {
    std::vector<size_type> offsets;
    offsets.reserve(expr.size());

    const auto &mapping = data_span.mapping();
    for (size_type i = 0; i < expr.size(); ++i) {
      if (static_cast<bool>(expr[i]))
        offsets.push_back(mapping(i));
    }

    auto base = data_span.data_handle();
    extents_type ext(offsets.size());
    auto acc = detail::index_accessor<element_type>(base, std::move(offsets));
    span_ = mdspan_type(base, ext, acc);
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }

  template <typename Op>
  mask_view &apply_scalar_op(Op &&op, const value_type &x) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], x);
    return *this;
  }

  template <class Op, class Expr>
  mask_view &apply_expr_op(Op &&op, const Expr &expr) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], expr[i]);
    return *this;
  }

  template <class, class, class> friend class ndarray;

  mdspan_type span_;
};

// [numcxx.indirect_view]
template <class ElementType> class indirect_view {
public:
  // clang-format off
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  using extents_type = dextents<1>;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  using         layout_type = detail::layout_right;
  using const_accessor_type = detail::index_accessor<const element_type>;
  using       accessor_type = detail::index_accessor<      element_type>;

  using   const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type, const_accessor_type>;
  using         mdspan_type = detail::mdspan<      element_type, extents_type, layout_type,       accessor_type>;

  using        const_reference = typename const_mdspan_type::reference;
  using              reference = typename       mdspan_type::reference;
  using const_data_handle_type = typename const_mdspan_type::data_handle_type;
  using       data_handle_type = typename       mdspan_type::data_handle_type;

  // construct/destroy:
  indirect_view() = delete;
  indirect_view(const indirect_view &) = default;
  indirect_view(indirect_view &&) noexcept = default;
  ~indirect_view() = default;

  // element access
  [[nodiscard]] const value_type &operator[](size_type i) const { NUMCXX_ASSERT(i < size(), "indirect_view::operator[] index out of bounds"); return span_[i]; }
  [[nodiscard]]       value_type &operator[](size_type i)       { NUMCXX_ASSERT(i < size(), "indirect_view::operator[] index out of bounds"); return span_[i]; }

  // unary operators:
  auto operator+() const { return apply_unary_op<nc_unary_plus>   (); }
  auto operator-() const { return apply_unary_op<std::negate>     (); }
  auto operator~() const { return apply_unary_op<nc_bit_not>      (); }
  auto operator!() const { return apply_unary_op<std::logical_not>(); }

  // computed assignment:
  indirect_view &operator=  (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a = b  ; }, x); }
  indirect_view &operator+= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a += b ; }, x); }
  indirect_view &operator-= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a -= b ; }, x); }
  indirect_view &operator*= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a *= b ; }, x); }
  indirect_view &operator/= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a /= b ; }, x); }
  indirect_view &operator%= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a %= b ; }, x); }
  indirect_view &operator&= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a &= b ; }, x); }
  indirect_view &operator|= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a |= b ; }, x); }
  indirect_view &operator^= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a ^= b ; }, x); }
  indirect_view &operator<<=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a <<= b; }, x); }
  indirect_view &operator>>=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a >>= b; }, x); }

  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <class Expr, std::enable_if_t<nc_is_val_expr_v<Expr>, int> = 0> indirect_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

  // observers
  static constexpr          rank_type                     rank()       noexcept { return extents_type::rank()          ; }
  static constexpr          rank_type             rank_dynamic()       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr          size_type static_extent(size_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr          size_type        extent(size_type r) const noexcept { return span_.extent(r); }
         constexpr          size_type                     size() const noexcept { return span_.size()   ; }
         constexpr               bool                    empty() const noexcept { return size() == 0    ; }
         constexpr const extents_type&                 extents() const noexcept { return span_.extents(); }

  // mdspan interoperability
         constexpr  const_mdspan_type                to_mdspan() const noexcept { return span_          ; }
         constexpr        mdspan_type                to_mdspan()       noexcept { return span_          ; }
  // clang-format on

private:
  template <typename MdSpan>
  explicit indirect_view(const MdSpan &data_span,
                         std::vector<size_type> offsets) {
    auto base = data_span.data_handle();
    extents_type ext(offsets.size());
    auto acc = detail::index_accessor<element_type>(base, std::move(offsets));
    span_ = mdspan_type(base, ext, acc);
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }

  template <typename Op>
  indirect_view &apply_scalar_op(Op &&op, const value_type &x) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], x);
    return *this;
  }

  template <class Op, class Expr>
  indirect_view &apply_expr_op(Op &&op, const Expr &expr) {
    for (size_type i = 0; i < size(); ++i)
      op((*this)[i], expr[i]);
    return *this;
  }

  template <class, class, class> friend class ndarray;

  mdspan_type span_;
};

// [numcxx.expression_template]
template <class ValExpr> class nc_val_expr {
  typedef std::remove_reference_t<ValExpr> RmExpr;

  ValExpr expr_;

public:
  typedef typename RmExpr::value_type value_type;
  typedef typename RmExpr::result_type result_type;

  explicit nc_val_expr(const RmExpr &e) : expr_(e) {}

  result_type operator[](size_type i) const { return expr_[i]; }

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
  // nc_val_expr<__indirect_expr<ValExpr> > operator[](const ndarray<size_type, Ex,
  // Lp>& __vs) const {
  //     typedef __indirect_expr<ValExpr> NewExpr;
  //     return nc_val_expr< NewExpr >(NewExpr(__vs, expr_));
  // }

  // clang-format off
  auto operator+() const { return apply_unary_op<nc_unary_plus>   (); }
  auto operator-() const { return apply_unary_op<std::negate>     (); }
  auto operator~() const { return apply_unary_op<nc_bit_not>      (); }
  auto operator!() const { return apply_unary_op<std::logical_not>(); }
  // clang-format on

  auto eval() const {
    static_assert(nc_mdspan_like_v<ValExpr>,
                  "Cannot eval() a scalar-only expression (no extents)");
    ndarray<value_type, decltype(expr_.extents()), layout_right> res(
        expr_.extents());
    for (size_type i = 0; i < res.size(); ++i) {
      res[i] = expr_[i];
    }
    return res;
  }

  size_type size() const { return expr_.size(); }
  auto extents() const { return expr_.extents(); }

  result_type sum() const {
    size_type n = expr_.size();
    result_type r = n ? expr_[0] : result_type();
    for (size_type i = 1; i < n; ++i)
      r += expr_[i];
    return r;
  }

  result_type min() const {
    size_type n = size();
    result_type r = n ? (*this)[0] : result_type();
    for (size_type i = 1; i < n; ++i) {
      result_type x = expr_[i];
      if (x < r)
        r = x;
    }
    return r;
  }

  result_type max() const {
    size_type n = size();
    result_type r = n ? (*this)[0] : result_type();
    for (size_type i = 1; i < n; ++i) {
      result_type x = expr_[i];
      if (r < x)
        r = x;
    }
    return r;
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }
};

template <class Tp> class nc_scalar_expr {
public:
  using value_type = std::remove_cv_t<Tp>;

  explicit nc_scalar_expr(const value_type &t, size_type s) : t_(t), s_(s) {}
  const value_type &operator[](size_type) const { return t_; }
  size_type size() const { return s_; }

private:
  const value_type &t_;
  size_type s_;
};

template <class Op, class A0> struct nc_unary_op {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  A0 a0_;

  nc_unary_op(const Op &op, const A0 &a0) : op_(op), a0_(a0) {}

  result_type operator[](size_type i) const { return op_(a0_[i]); }

  size_type size() const { return a0_.size(); }

  auto extents() const {
    static_assert(nc_mdspan_like_v<A0>,
                  "Unary expression has no extents (operand is scalar)");
    return a0_.extents();
  }
};

template <class Op, class A0, class A1> struct nc_binary_op {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  A0 a0_;
  A1 a1_;

  nc_binary_op(const Op &op, const A0 &a0, const A1 &a1)
      : op_(op), a0_(a0), a1_(a1) {}

  result_type operator[](size_type i) const { return op_(a0_[i], a1_[i]); }

  size_type size() const { return a0_.size(); }

  auto extents() const {
    static_assert(nc_mdspan_like_v<nc_binary_op>,
                  "Binary expression has no extents (scalar + scalar)");

    if constexpr (nc_mdspan_like_v<A0>) {
      return a0_.extents();
    } else {
      return a1_.extents();
    }
  }
};

template <class Op, class Tp, class Ex, class Lp>
struct nc_unary_op<Op, ndarray<Tp, Ex, Lp>> {
  typedef typename Op::result_type result_type;
  using value_type = std::decay_t<result_type>;

  Op op_;
  const ndarray<Tp, Ex, Lp> &a0_;

  nc_unary_op(const Op &op, const ndarray<Tp, Ex, Lp> &a0) : op_(op), a0_(a0) {}

  result_type operator[](size_type i) const { return op_(a0_[i]); }

  size_type size() const { return a0_.size(); }
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

  result_type operator[](size_type i) const { return op_(a0_[i], a1_[i]); }

  size_type size() const { return a0_.size(); }
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

  result_type operator[](size_type i) const { return op_(a0_[i], a1_[i]); }

  size_type size() const { return a0_.size(); }
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

  result_type operator[](size_type i) const { return op_(a0_[i], a1_[i]); }

  size_type size() const { return a0_.size(); }
};

// ndarray
template <class Tp, class Ex, class Lp>
inline void ndarray<Tp, Ex, Lp>::swap(ndarray &v) noexcept {
  std::swap(elem_, v.elem_);
}

// template <class Tp, class Ex, class Lp>
// void ndarray<Tp, Ex, Lp>::resize(size_type n, value_type x) {
//     __clear(size());
//     if (n) {
//         begin_ = end_ = allocator<value_type>().allocate(n);
//         auto __guard = std::__make_exception_guard([&] { __clear(n); });
//         for (size_type __n_left = n; __n_left; --__n_left, ++end_)
//             ::new ((void*)end_) value_type(x);
//         __guard.__complete();
//     }
// }

template <class Tp, class Ex, class Lp>
inline void swap(ndarray<Tp, Ex, Lp> &x, ndarray<Tp, Ex, Lp> &y) noexcept {
  x.swap(y);
}

namespace detail {
template <class Op, class Expr> inline auto make_unary_op(const Expr &x) {
  using value_type = typename Expr::value_type;
  using OpType = nc_unary_op<Op, Expr>;
  OpType node(Op{}, x);
  return nc_val_expr<OpType>(node);
}

template <class Op, class Expr1, class Expr2>
inline auto make_expr_expr(const Expr1 &x, const Expr2 &y) {
  using value_type = typename Expr1::value_type;
  using OpType = nc_binary_op<Op, Expr1, Expr2>;
  OpType node(Op{}, x, y);
  return nc_val_expr<OpType>(node);
}

template <class Op, class Expr>
inline auto make_expr_scalar(const Expr &x,
                             const typename Expr::value_type &y) {
  using value_type = typename Expr::value_type;
  using OpType = nc_binary_op<Op, Expr, nc_scalar_expr<value_type>>;
  OpType node(Op{}, x, nc_scalar_expr<value_type>(y, x.size()));
  return nc_val_expr<OpType>(node);
}

template <class Op, class Expr>
inline auto make_scalar_expr(const typename Expr::value_type &x,
                             const Expr &y) {
  using value_type = typename Expr::value_type;
  using OpType = nc_binary_op<Op, nc_scalar_expr<value_type>, Expr>;
  OpType node(Op{}, nc_scalar_expr<value_type>(x, y.size()), y);
  return nc_val_expr<OpType>(node);
}
} // namespace detail

// clang-format off
#define NUMCXX_MAKE_UNARY_FN(FN, FUNCTOR)                                      \
  template <class E, std::enable_if_t<nc_is_val_expr_v<E>, int> = 0>           \
  [[nodiscard]] inline auto FN(const E& x) {                                   \
    return detail::make_unary_op<FUNCTOR<typename E::value_type>>(x);          \
  }

#define NUMCXX_MAKE_BINARY_OP(OP, FUNCTOR)                                     \
  template <                  class E1,                    class E2,           \
    std::enable_if_t<nc_is_val_expr_v<E1> && nc_is_val_expr_v<E2>, int> = 0>   \
  inline auto operator OP(const E1 &x, const E2 &y) {                          \
    return detail::make_expr_expr<FUNCTOR<typename E1::value_type>>(x, y);     \
  }                                                                            \
  template <class E, std::enable_if_t<nc_is_val_expr_v<E>, int> = 0>           \
  inline auto operator OP(const E &x, const typename E::value_type &y) {       \
    return detail::make_expr_scalar<FUNCTOR<typename E::value_type>>(x, y);    \
  }                                                                            \
  template <class E, std::enable_if_t<nc_is_val_expr_v<E>, int> = 0>      \
  inline auto operator OP(const typename E::value_type &x, const E &y) {       \
    return detail::make_scalar_expr<FUNCTOR<typename E::value_type>>(x, y);    \
  }

#define NUMCXX_MAKE_BINARY_FN(FN, FUNCTOR)                                     \
  template <                  class E1,                    class E2,           \
    std::enable_if_t<nc_is_val_expr_v<E1> && nc_is_val_expr_v<E2>, int> = 0>   \
  inline auto FN(const E1 &x, const E2 &y) {                                   \
    return detail::make_expr_expr<FUNCTOR<typename E1::value_type>>(x, y);     \
  }                                                                            \
  template <class E, std::enable_if_t<nc_is_val_expr_v<E>, int> = 0>           \
  inline auto FN(const E &x, const typename E::value_type &y) {                \
    return detail::make_expr_scalar<FUNCTOR<typename E::value_type>>(x, y);    \
  }                                                                            \
  template <class E, std::enable_if_t<nc_is_val_expr_v<E>, int> = 0>           \
  inline auto FN(const typename E::value_type &x, const E &y) {                \
    return detail::make_scalar_expr<FUNCTOR<typename E::value_type>>(x, y);    \
  }

  template <class Tp> struct nc_bit_shift_left  { typedef Tp result_type; Tp operator()(const Tp &x, const Tp &y) const { return x << y; } };
  template <class Tp> struct nc_bit_shift_right { typedef Tp result_type; Tp operator()(const Tp &x, const Tp &y) const { return x >> y; } };

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

// applies binary operators to each element of two ndarrays, or a ndarray and a value
NUMCXX_MAKE_BINARY_OP( +, std::plus         )
NUMCXX_MAKE_BINARY_OP( -, std::minus        )
NUMCXX_MAKE_BINARY_OP( *, std::multiplies   )
NUMCXX_MAKE_BINARY_OP( /, std::divides      )
NUMCXX_MAKE_BINARY_OP( %, std::modulus      )
NUMCXX_MAKE_BINARY_OP( &, std::bit_and      )
NUMCXX_MAKE_BINARY_OP( |, std::bit_or       )
NUMCXX_MAKE_BINARY_OP( ^, std::bit_xor      )
NUMCXX_MAKE_BINARY_OP(<<, nc_bit_shift_left )
NUMCXX_MAKE_BINARY_OP(>>, nc_bit_shift_right)
NUMCXX_MAKE_BINARY_OP(&&, std::logical_and  )
NUMCXX_MAKE_BINARY_OP(||, std::logical_or   )

// compares two ndarrays or a ndarray with a value
NUMCXX_MAKE_BINARY_OP(==, std::equal_to     )
NUMCXX_MAKE_BINARY_OP(!=, std::not_equal_to )
NUMCXX_MAKE_BINARY_OP( <, std::less         )
NUMCXX_MAKE_BINARY_OP(<=, std::less_equal   )
NUMCXX_MAKE_BINARY_OP( >, std::greater      )
NUMCXX_MAKE_BINARY_OP(>=, std::greater_equal)

// absolute function
NUMCXX_MAKE_UNARY_FN( abs, nc_abs_expr )

// exponential functions
NUMCXX_MAKE_UNARY_FN( exp, nc_exp_expr )
NUMCXX_MAKE_UNARY_FN( log, nc_log_expr )
NUMCXX_MAKE_UNARY_FN(log10,nc_log10_expr)

// power function
NUMCXX_MAKE_BINARY_FN(pow, nc_pow_expr )
NUMCXX_MAKE_UNARY_FN(sqrt, nc_sqrt_expr)

// trigonometric functions
NUMCXX_MAKE_UNARY_FN( sin, nc_sin_expr )
NUMCXX_MAKE_UNARY_FN( cos, nc_cos_expr )
NUMCXX_MAKE_UNARY_FN( tan, nc_tan_expr )
NUMCXX_MAKE_UNARY_FN(asin, nc_asin_expr)
NUMCXX_MAKE_UNARY_FN(acos, nc_acos_expr)
NUMCXX_MAKE_UNARY_FN(atan, nc_atan_expr)
NUMCXX_MAKE_BINARY_FN(atan2, nc_atan2_expr)

// hyperbolic functions
NUMCXX_MAKE_UNARY_FN(sinh, nc_sinh_expr)
NUMCXX_MAKE_UNARY_FN(cosh, nc_cosh_expr)
NUMCXX_MAKE_UNARY_FN(tanh, nc_tanh_expr)

template <class Tp, class Ex, class Lp> [[nodiscard]] inline const Tp *begin(const ndarray<Tp, Ex, Lp> &v) { return v.data()           ; }
template <class Tp, class Ex, class Lp> [[nodiscard]] inline       Tp *begin(      ndarray<Tp, Ex, Lp> &v) { return v.data()           ; }
template <class Tp, class Ex, class Lp> [[nodiscard]] inline const Tp *end  (const ndarray<Tp, Ex, Lp> &v) { return v.data() + v.size(); }
template <class Tp, class Ex, class Lp> [[nodiscard]] inline       Tp *end  (      ndarray<Tp, Ex, Lp> &v) { return v.data() + v.size(); }
// clang-format on

namespace detail {
template <size_type Rank>
auto make_extents(const std::initializer_list<size_type> &shape) {
  if (shape.size() != Rank) {
    NUMCXX_THROW(std::invalid_argument, "shape size does not match array rank");
  }
  std::array<size_type, Rank> dims;
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

  const size_type n = std::ceil((stop - start) / step);

  ndarray<T, dextents<1>> arr({n});
  T *p = arr.data();

  T value = start;
  for (size_type i = 0; i < n; ++i) {
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
NdArrayType ones(std::initializer_list<size_type> shape) {
  constexpr size_type rank = NdArrayType::extents_type::rank();
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
NdArrayType zeros(std::initializer_list<size_type> shape) {
  constexpr size_type rank = NdArrayType::extents_type::rank();
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
  for (size_type i = 0; i < arr.size(); ++i) {
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
NdArrayType rand(std::initializer_list<size_type> shape) {
  constexpr size_type rank = NdArrayType::extents_type::rank();
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
NdArrayType randn(std::initializer_list<size_type> shape) {
  constexpr size_type rank = NdArrayType::extents_type::rank();
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
                    std::initializer_list<size_type> shape) {
  constexpr size_type rank = NdArrayType::extents_type::rank();
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
NdArrayType randint(int low, int high, std::initializer_list<size_type> shape) {
  constexpr size_type rank = NdArrayType::extents_type::rank();
  NdArrayType arr(detail::make_extents<rank>(shape));
  detail::fill_random(arr, std::uniform_int_distribution<int>(low, high - 1));
  return arr;
}

} // namespace random

//
// [numcxx.linalg] linear algebra (TODO)
//
namespace linalg {

template <class A, class B> auto matmul(const A &a, const B &b) {
  static_assert(A::rank() == 2, "matmul requires rank-2 lhs");
  static_assert(B::rank() == 2, "matmul requires rank-2 rhs");

  const auto &a_ext = a.extents();
  const auto &b_ext = b.extents();
  if (a_ext.extent(1) != b_ext.extent(0)) {
    NUMCXX_THROW(std::invalid_argument, "matmul: incompatible shapes");
  }

  using value_type = typename A::value_type;
  using extents_type = dextents<2>;
  ndarray<value_type, extents_type, layout_right> c(a_ext.extent(0),
                                                    b_ext.extent(1));
  detail::linalg::matrix_product(a.to_mdspan(), b.to_mdspan(), c.to_mdspan());

  return c;
}

} // namespace linalg

//
// [numcxx.aliases] type aliases
//
// clang-format off
template <typename T> using  vec = numcxx::ndarray<T, dextents<1>>;
template <typename T> using  mat = numcxx::ndarray<T, dextents<2>>;
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