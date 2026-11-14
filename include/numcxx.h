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
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdio>  // std::fprintf, std::fflush
#include <cstdlib> // std::abort
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <source_location>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>
#include <version>

// clang-format off
#if __cplusplus < 202600L
// Uses Kokkos mdspan/submdspan/mdarray/linalg as backend
#define NUMCXX_KOKKOS_MDSPAN_BACKEND
#include <mdspan/mdspan.hpp>
#include <mdspan/mdarray.hpp>
#include <experimental/linalg>
#else
// Uses std mdspan/submdspan/mdarray/linalg as backend (in the future)
#include <mdspan>
#include <mdarray>
#include <linalg>
#endif
// clang-format on

// CBLAS acceleration is optional; set NUMCXX_ENABLE_CBLAS=0 to disable it.
// NUMCXX_CBLAS_HEADER can name a non-standard CBLAS header.
#ifndef NUMCXX_ENABLE_CBLAS
#define NUMCXX_ENABLE_CBLAS 1
#endif

#if NUMCXX_ENABLE_CBLAS
#ifndef NUMCXX_CBLAS_HEADER
#define NUMCXX_CBLAS_HEADER <cblas.h>
#endif
#if defined(__has_include)
#if __has_include(NUMCXX_CBLAS_HEADER)
#include NUMCXX_CBLAS_HEADER
#define NUMCXX_DETAIL_HAS_CBLAS
#endif
#else
#include NUMCXX_CBLAS_HEADER
#define NUMCXX_DETAIL_HAS_CBLAS
#endif
#endif

// LAPACKE is independent of CBLAS; set NUMCXX_ENABLE_LAPACKE=0 to use the
// built-in Gaussian-elimination fallback.
#ifndef NUMCXX_ENABLE_LAPACKE
#define NUMCXX_ENABLE_LAPACKE 1
#endif

#if NUMCXX_ENABLE_LAPACKE
#ifndef NUMCXX_LAPACKE_HEADER
#define NUMCXX_LAPACKE_HEADER <lapacke.h>
#endif
#if !defined(OPENBLAS_USE64BITINT) && defined(__has_include)
#if __has_include(<openblas_config.h>)
#include <openblas_config.h>
#endif
#endif
#if defined(__has_include)
#if __has_include(NUMCXX_LAPACKE_HEADER)
#ifndef HAVE_LAPACK_CONFIG_H
#define NUMCXX_DETAIL_DEFINED_HAVE_LAPACK_CONFIG_H
#define HAVE_LAPACK_CONFIG_H
#endif
#if defined(OPENBLAS_USE64BITINT) && !defined(LAPACK_ILP64)
#define NUMCXX_DETAIL_DEFINED_LAPACK_ILP64
#define LAPACK_ILP64
#endif
#pragma push_macro("I")
#undef I
#include NUMCXX_LAPACKE_HEADER
#pragma pop_macro("I")
#ifdef NUMCXX_DETAIL_DEFINED_LAPACK_ILP64
#undef LAPACK_ILP64
#undef NUMCXX_DETAIL_DEFINED_LAPACK_ILP64
#endif
#ifdef NUMCXX_DETAIL_DEFINED_HAVE_LAPACK_CONFIG_H
#undef HAVE_LAPACK_CONFIG_H
#undef NUMCXX_DETAIL_DEFINED_HAVE_LAPACK_CONFIG_H
#endif
#define NUMCXX_DETAIL_HAS_LAPACKE
#endif
#else
// Without __has_include, assume the configured LAPACKE header is available.
#ifndef HAVE_LAPACK_CONFIG_H
#define NUMCXX_DETAIL_DEFINED_HAVE_LAPACK_CONFIG_H
#define HAVE_LAPACK_CONFIG_H
#endif
#if defined(OPENBLAS_USE64BITINT) && !defined(LAPACK_ILP64)
#define NUMCXX_DETAIL_DEFINED_LAPACK_ILP64
#define LAPACK_ILP64
#endif
#pragma push_macro("I")
#undef I
#include NUMCXX_LAPACKE_HEADER
#pragma pop_macro("I")
#ifdef NUMCXX_DETAIL_DEFINED_LAPACK_ILP64
#undef LAPACK_ILP64
#undef NUMCXX_DETAIL_DEFINED_LAPACK_ILP64
#endif
#ifdef NUMCXX_DETAIL_DEFINED_HAVE_LAPACK_CONFIG_H
#undef HAVE_LAPACK_CONFIG_H
#undef NUMCXX_DETAIL_DEFINED_HAVE_LAPACK_CONFIG_H
#endif
#define NUMCXX_DETAIL_HAS_LAPACKE
#endif
#endif

// [numcxx.configuration]

//===----------------------------------------------------------------------===//
// User Configuration
//===----------------------------------------------------------------------===//
#ifndef NUMCXX_SIZE_TYPE
// Unsigned integer type for sizes and extents (maximum size of any theoretically possible object)
#define NUMCXX_SIZE_TYPE std::size_t
#endif

#ifndef NUMCXX_INDEX_TYPE
// Signed integer type for index arithmetic and NumPy-style indexing (negative indices supported)
#define NUMCXX_INDEX_TYPE std::ptrdiff_t
#endif

#ifndef NUMCXX_DEFAULT_LAYOUT_LEFT
// Row major (C / Numpy style) is used by default.
// To use Column major (Fortran / Matlab style) as default layout, please uncomment the line below.
// #define NUMCXX_DEFAULT_LAYOUT_LEFT
#endif

//===----------------------------------------------------------------------===//
// Error Handling Configuration
//===----------------------------------------------------------------------===//
#ifndef NUMCXX_NO_DEBUG
#define NUMCXX_ASSERT(expr, msg)                                               \
  ((expr) ? (void)0                                                            \
          : ::numcxx::detail::fail("Assertion failed (" #expr ")", (msg)))
#else
#define NUMCXX_ASSERT(expr, msg) ((void)0)
#endif

#define NUMCXX_THROW(exception_type, msg)                                      \
  ::numcxx::detail::throw_exception<exception_type>((msg))

namespace numcxx {
namespace detail {
[[noreturn]]
inline void
fail(const char *kind, const char *msg,
     std::source_location loc = std::source_location::current()) noexcept {
  std::fprintf(stderr, "numcxx:%s:%u: %s: %s: %s\n", loc.file_name(),
               loc.line(), loc.function_name(), kind, msg);
  std::fflush(stderr);
  std::abort();
}

template <class Exception>
[[noreturn]]
inline void throw_exception(const char *msg) {
#ifdef NUMCXX_NO_EXCEPTIONS
  fail("Exception", msg);
#else
  throw Exception(msg);
#endif
}

template <class Exception>
[[noreturn]]
inline void throw_exception(const std::string &msg) {
#ifdef NUMCXX_NO_EXCEPTIONS
  fail("Exception", msg.c_str());
#else
  throw Exception(msg);
#endif
}

template <std::integral Index>
[[noreturn]]
inline void throw_index_out_of_range(Index i, NUMCXX_SIZE_TYPE size,
                                     const char *what) {
  const std::string message =
      std::string(what) + ": index " + std::to_string(i) +
      " out of bounds for size " + std::to_string(size);
  NUMCXX_THROW(std::out_of_range, message);
}

inline void checked_index(NUMCXX_SIZE_TYPE i, NUMCXX_SIZE_TYPE size,
                          const char *what) {
  if (i >= size)
    throw_index_out_of_range(i, size, what);
}

//===----------------------------------------------------------------------===//
// Backend Configuration
//===----------------------------------------------------------------------===//
// Internal: abstract the backend for mdspan, submdspan, mdarray, and linalg.
#ifdef NUMCXX_KOKKOS_MDSPAN_BACKEND
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::full_extent;
using Kokkos::full_extent_t;
using Kokkos::layout_left;
using Kokkos::layout_right;
using Kokkos::mdspan;
using Kokkos::strided_slice;
using Kokkos::submdspan;
using Kokkos::Experimental::mdarray;
namespace linalg = Kokkos::Experimental::linalg;
#else
using std::dextents;
using std::extents;
using std::full_extent;
using std::full_extent_t;
using std::layout_left;
using std::layout_right;
using std::mdarray;
using std::mdspan;
using std::strided_slice;
using std::submdspan;
namespace linalg = std::linalg;
#endif
} // namespace detail

// clang-format off
// [numcxx.declarations]
using  size_type = NUMCXX_SIZE_TYPE;
using index_type = NUMCXX_INDEX_TYPE;

template <size_type    Rank   > using dextents     = detail::dextents<size_type, Rank      >;
template <size_type... Extents> using extents      = detail::extents <size_type, Extents...>;
                                using layout_left  = detail::layout_left ;
                                using layout_right = detail::layout_right;
                                using                detail::full_extent ;
#ifdef NUMCXX_DEFAULT_LAYOUT_LEFT
using default_layout = layout_left;
#else
using default_layout = layout_right;
#endif

                                        class          slice;
template <class Tp, class Ex, class Lp> class        ndarray;
template <class Tp, class Ex, class Lp> class    gslice_view;
template <class Tp>                     class      mask_view;
template <class Tp>                     class  indirect_view;
template <class Tp, size_type Rank>     class broadcast_view;

template <class ValExpr>                struct  nc_val_expr;
template <class Op, class A0>           struct  nc_unary_op;
template <class Op, class A0, class A1> struct nc_binary_op;

// Value-expression types participate in NumCxx expression evaluation.
template <class ValExpr>                struct nc_is_val_expr                             : std::false_type {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<       ndarray<Tp, Ex, Lp>> : std::true_type  {};
template <class Tp, class Ex, class Lp> struct nc_is_val_expr<   gslice_view<Tp, Ex, Lp>> : std::true_type  {};
template <class Tp>                     struct nc_is_val_expr<     mask_view<Tp        >> : std::true_type  {};
template <class Tp>                     struct nc_is_val_expr< indirect_view<Tp        >> : std::true_type  {};
template <class Tp, size_type Rank>     struct nc_is_val_expr<broadcast_view<Tp, Rank  >> : std::true_type  {};
template <class ValExpr>                struct nc_is_val_expr<   nc_val_expr<ValExpr   >> : std::true_type  {};
template <class Op, class A0>           struct nc_is_val_expr<   nc_unary_op<Op, A0    >> : std::true_type  {};
template <class Op, class A0, class A1> struct nc_is_val_expr<  nc_binary_op<Op, A0, A1>> : std::true_type  {};

// Types with multidimensional extents provide shape information.
template <class Tp>                     struct nc_has_extents                             : std::false_type {};
template <class Tp, class Ex, class Lp> struct nc_has_extents<       ndarray<Tp, Ex, Lp>> : std::true_type  {};
template <class Tp, class Ex, class Lp> struct nc_has_extents<   gslice_view<Tp, Ex, Lp>> : std::true_type  {};
template <class Tp>                     struct nc_has_extents<     mask_view<Tp        >> : std::true_type  {};
template <class Tp>                     struct nc_has_extents< indirect_view<Tp        >> : std::true_type  {};
template <class Tp, size_type Rank>     struct nc_has_extents<broadcast_view<Tp, Rank  >> : std::true_type  {};
template <class ValExpr>                struct nc_has_extents<   nc_val_expr<ValExpr   >> : nc_has_extents<std::remove_cvref_t<ValExpr>> {};
template <class Op, class A0>           struct nc_has_extents<   nc_unary_op<Op, A0    >> : nc_has_extents<std::remove_cvref_t<A0>     > {};
template <class Op, class A0, class A1> struct nc_has_extents<  nc_binary_op<Op, A0, A1>> : std::bool_constant<nc_has_extents<std::remove_cvref_t<A0>>::value  ||
                                                                                                               nc_has_extents<std::remove_cvref_t<A1>>::value> {};

template <class Tp> inline constexpr bool nc_is_val_expr_v = nc_is_val_expr<std::remove_cvref_t<Tp>>::value;
template <class Tp> inline constexpr bool nc_has_extents_v = nc_has_extents<std::remove_cvref_t<Tp>>::value;
template <class Tp> struct nc_is_broadcast_view : std::false_type {};
template <class Tp, size_type Rank>
struct nc_is_broadcast_view<broadcast_view<Tp, Rank>> : std::true_type {};
template <class Tp>
inline constexpr bool nc_is_broadcast_view_v =
    nc_is_broadcast_view<std::remove_cvref_t<Tp>>::value;

// Value expressions support element-wise evaluation and have extents.
template <class Tp> concept  value_expression = nc_is_val_expr_v<Tp>;

// Type traits and concepts used to distinguish scalar indexing from slicing operations.
// A slice specifier is scalar index, slice, full_extent, or integral pair range.
template <class Tp>                         concept     scalar_index = std::integral<std::remove_cvref_t<Tp>>;

template <class Tp>                          struct is_integral_pair                  : std::false_type {};
template <std::integral I, std::integral J>  struct is_integral_pair<std::pair<I, J>> : std::true_type  {};
template <class Tp>                         concept    integral_pair = is_integral_pair<std::remove_cvref_t<Tp>>::value;

template <class Tp> concept slice_specifier = scalar_index<Tp> || std::same_as<std::remove_cvref_t<Tp>, slice> ||
                                              std::same_as<std::remove_cvref_t<Tp>, detail::full_extent_t> || integral_pair<Tp>;

template <class... Idxs> concept all_scalar_indices   = (scalar_index<Idxs>  && ...);
template <class... Args> concept all_slice_specifiers = (slice_specifier<Args> && ...);
template <size_type Rank, class... Idxs> concept elem_access_arguments = all_scalar_indices<Idxs...> && sizeof...(Idxs) == Rank;
template <size_type Rank, class... Args> concept gslice_view_arguments = sizeof...(Args) > 0 && (
                                                                        (all_slice_specifiers<Args...> && !all_scalar_indices<Args...> && sizeof...(Args) <= Rank) ||
                                                                        (all_scalar_indices<Args...> && sizeof...(Args) < Rank));
template <size_type Rank, class... Args> concept gslice_arguments      = elem_access_arguments<Rank, Args...> ||
                                                                         gslice_view_arguments<Rank, Args...>;

namespace detail {

template <class Ex   > concept static_extents = requires { Ex::rank_dynamic(); } && (Ex::rank_dynamic() == 0);
template <class Array> concept   static_array = static_extents<typename Array::extents_type>;
template <class Array> concept  dynamic_array = !static_array<Array>;

template <class T> struct is_boolean_expr : std::bool_constant<
    requires(const T &expr) { static_cast<bool>(expr.flat(size_type{})); }> {};

template <class Lhs, class Rhs>
void check_shape_match(const Lhs &lhs, const Rhs &rhs) {
  using lhs_type = std::remove_cvref_t<Lhs>;
  using rhs_type = std::remove_cvref_t<Rhs>;

  if (lhs_type::rank() != rhs_type::rank())
    NUMCXX_THROW(std::invalid_argument,
                 "array operands must have identical shapes");

  for (typename lhs_type::rank_type r = 0; r < lhs_type::rank(); ++r) {
    if (lhs.extent(r) !=
        rhs.extent(static_cast<typename rhs_type::rank_type>(r)))
      NUMCXX_THROW(std::invalid_argument,
                   "array operands must have identical shapes");
  }
}

template <class Array, class Tuple>
[[nodiscard]] constexpr decltype(auto)
apply_indices(Array &array, const Tuple &indices) {
  return std::apply(
      [&array](const auto &...args) -> decltype(auto) {
        return array(args...);
      },
      indices);
}

template <class Op, class Expr> auto make_unary_op(const Expr &);
template <class Tp> struct nc_unary_plus { Tp operator()(const Tp &x) const { return +x; } };
template <class Tp> struct nc_bit_not    { Tp operator()(const Tp &x) const { return ~x; } };

template <typename Tp, std::size_t Rank> struct nested_initializer_list        { using type = std::initializer_list<typename nested_initializer_list<Tp, Rank - 1>::type>; };
template <typename Tp                  > struct nested_initializer_list<Tp, 1> { using type = std::initializer_list<Tp>;                                                   };
template <typename Tp                  > struct nested_initializer_list<Tp, 0> { using type = std::initializer_list<Tp>;                                                   };
template <typename Tp, std::size_t Rank>  using nested_initializer_list_t = typename nested_initializer_list<Tp, Rank>::type;
} // namespace detail

// [numcxx.slice]
/// A NumPy-style slice `start:stop:step`
class slice {
private:
  std::optional<index_type> start_{};
  std::optional<index_type> stop_{};
                index_type  step_ = 1;

public:
  constexpr slice() = default;
  constexpr slice(std::optional<index_type> start,
                  std::optional<index_type> stop ,
                                index_type  step = 1) noexcept
      : start_(start), stop_(stop), step_(step) {
  }

  [[nodiscard]] constexpr                   bool is_valid() const noexcept { return step_ != 0; }
  [[nodiscard]] constexpr std::optional<index_type> start() const noexcept { return start_; }
  [[nodiscard]] constexpr std::optional<index_type> stop () const noexcept { return stop_ ; }
  [[nodiscard]] constexpr               index_type  step () const noexcept { return step_ ; }

  [[nodiscard]] friend constexpr bool operator==(const slice &x, const slice &y) noexcept {
    return x.start_ == y.start_ && x.stop_ == y.stop_ && x.step_ == y.step_;
  }
  [[nodiscard]] friend constexpr bool operator!=(const slice &x, const slice &y) noexcept {
    return !(x == y);
  }

public:
  // ---------------------------------------------------------------------------
  // Internal implementation support.
  //
  // Exposed for testing and other numcxx internals.
  // Not part of the stable public API and subject to change without notice.
  // ---------------------------------------------------------------------------

  struct normalized_slice {
    // Equivalent forward std::strided_slice.
    size_type offset;
    size_type extent; // Length of the interval [offset, offset + extent).
                      // Follows std::strided_slice semantics and is not the number of
                      // selected elements.
    size_type stride;

    // True when the original NumPy slice used a negative step.
    // std::strided_slice only represents forward slices.
    bool reversed;
  };
  [[nodiscard]] constexpr normalized_slice normalize(size_type dim_len) const noexcept;
};

// [numcxx.gslice]
/// Generalized N-dimensional slice descriptor.
///
/// Stores the index and slice specifiers for each dimension as a single
/// selector object, enabling valarray-style selection via operator[].
template <class... Args>
  requires(sizeof...(Args) > 0 && (slice_specifier<Args> && ...))
class gslice {
public:
  constexpr explicit gslice(std::remove_cvref_t<Args>... args)
      : arguments_(std::move(args)...) {}

  // Returns the stored N-dimensional index and slice specifiers.
  [[nodiscard]] constexpr const auto &arguments() const noexcept {
    return arguments_;
  }

private:
  std::tuple<std::remove_cvref_t<Args>...> arguments_;
};

// Deduce and store selector types by value.
template <class... Args> gslice(Args &&...) -> gslice<std::remove_cvref_t<Args>...>;

// [numcxx.ndarray]
namespace detail {
template <class Ex>          struct static_extents_size;
template <size_type... Dims> struct static_extents_size<numcxx::extents<Dims...>> : std::integral_constant<size_type, (Dims * ... * size_type{1})> {};

template <class Tp, class Ex, bool = static_extents<Ex>> struct mdarray_container_selector;
template <class Tp, class Ex> struct mdarray_container_selector<Tp, Ex, true > { using type = std::array <Tp, static_extents_size<Ex>::value>; };
template <class Tp, class Ex> struct mdarray_container_selector<Tp, Ex, false> { using type = std::vector<Tp>                                ; };
template <class Tp, class Ex> using  mdarray_container_t = typename mdarray_container_selector<Tp, Ex>::type;

template <class Tp> struct reduction_mean_type {
  using type = std::conditional_t<std::is_integral_v<Tp>, double, Tp>;
  using real_type = type;
};
template <class Tp> struct reduction_mean_type<std::complex<Tp>> {
  using type = std::complex<Tp>;
  using real_type = Tp;
};
template <class Tp>
using reduction_mean_type_t = typename reduction_mean_type<Tp>::type;
template <class Tp>
using reduction_real_type_t = typename reduction_mean_type<Tp>::real_type;

template <class Tp>
bool reduction_less(const Tp &lhs, const Tp &rhs) {
  if constexpr (requires { lhs.real(); lhs.imag(); }) {
    const bool lhs_real_nan = std::isnan(lhs.real());
    const bool rhs_real_nan = std::isnan(rhs.real());
    if (lhs_real_nan != rhs_real_nan)
      return !lhs_real_nan;
    if (!lhs_real_nan && lhs.real() != rhs.real())
      return lhs.real() < rhs.real();

    const bool lhs_imag_nan = std::isnan(lhs.imag());
    const bool rhs_imag_nan = std::isnan(rhs.imag());
    if (lhs_imag_nan != rhs_imag_nan)
      return !lhs_imag_nan;
    return !lhs_imag_nan && lhs.imag() < rhs.imag();
  } else {
    return lhs < rhs;
  }
}
}

template <typename Layout = default_layout, typename Extents>
std::array<size_type, Extents::rank()>
unravel_index(size_type i, const Extents &extents);

template <class ElementType, class Extents, class LayoutPolicy = default_layout>
class ndarray {
public:
  // element
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  // extents
  using extents_type = Extents;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  // layout
  using  layout_type = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  // mdspan
  using const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type>;
  using       mdspan_type = detail::mdspan<      element_type, extents_type, layout_type>;
  
  // mdarray
  using mdarray_type = detail::mdarray<value_type, extents_type, layout_type,
      detail::mdarray_container_t<value_type, extents_type>>;

  // reference
  using const_reference = typename const_mdspan_type::reference;
  using       reference = typename       mdspan_type::reference;

  // pointer
  using const_pointer   = const element_type *;
  using       pointer   =       element_type *;

  static constexpr  rank_type  rank        (           )       noexcept { return extents_type::rank()          ; }
  static constexpr  rank_type  rank_dynamic(           )       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr  size_type static_extent(rank_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr index_type        extent(rank_type r) const noexcept { return elem_.extent(r)               ; }

  //--------------------------------------------------------------------------------
  // [numcxx.ndarray.cons], ndarray constructors, destructor, and assignment

  // construct/destroy:
  constexpr ndarray() = default;
  constexpr ndarray(const ndarray &v) = default;
  constexpr ndarray(ndarray &&v) noexcept = default;
  template <class... SizeTypes> explicit constexpr ndarray(SizeTypes... dyn_exts) : elem_(Extents(dyn_exts...)) {}
  ndarray(const    gslice_view<ElementType, Extents, LayoutPolicy> &sv) : elem_(sv.to_mdspan()) {}
  ndarray(const     mask_view<ElementType>                        &mv) : elem_(mv.to_mdspan()) {}
  ndarray(const indirect_view<ElementType>                        &iv) : elem_(iv.to_mdspan()) {}
  template <class BroadcastElement, ::numcxx::size_type Rank>
  explicit ndarray(const broadcast_view<BroadcastElement, Rank> &bv);
  // Broadcast views remain explicit-only to preserve their materialization API.
  template <value_expression Expr>
    requires (!std::same_as<std::remove_cvref_t<Expr>,
                            ndarray<ElementType, Extents, LayoutPolicy>> &&
              !nc_is_broadcast_view_v<Expr> &&
              (std::remove_cvref_t<Expr>::rank() == Extents::rank()))
  ndarray(const Expr &expr);
  ndarray(detail::nested_initializer_list_t<element_type, extents_type::rank()> list);

  ~ndarray() = default;

  // assignment:
  constexpr ndarray &operator=(const ndarray &v) = default;
  constexpr ndarray &operator=(ndarray &&v) noexcept = default;
  ndarray& operator=(detail::nested_initializer_list_t<value_type, extents_type::rank()> list);

  // element access using N-dimensional indices.
  template <class... Idxs> requires elem_access_arguments<Extents::rank(), Idxs...>
  [[nodiscard]] const_reference operator() (Idxs&&... idxs) const;
  template <class... Idxs> requires elem_access_arguments<Extents::rank(), Idxs...>
  [[nodiscard]]       reference operator() (Idxs&&... idxs)      ;

  // flattened element access in default_layout order, independent of the physical storage layout.
  [[nodiscard]] const_reference flat_unchecked(size_type i) const;
  [[nodiscard]]       reference flat_unchecked(size_type i)      ;
  [[nodiscard]] const_reference flat          (size_type i) const { detail::checked_index(i, size(), "ndarray::flat"); return flat_unchecked(i); }
  [[nodiscard]]       reference flat          (size_type i)       { detail::checked_index(i, size(), "ndarray::flat"); return flat_unchecked(i); }

  // flattened element access in physical storage order: operator[] is unchecked; at() checks bounds.
  [[nodiscard]] const_reference operator[]    (size_type i) const {                                                    return         data()[i]; }
  [[nodiscard]]       reference operator[]    (size_type i)       {                                                    return         data()[i]; }
  [[nodiscard]] const_reference at            (size_type i) const { detail::checked_index(i, size(), "ndarray::at"  ); return        (*this)[i]; }
  [[nodiscard]]       reference at            (size_type i)       { detail::checked_index(i, size(), "ndarray::at"  ); return        (*this)[i]; }

  // subset operations using N-dimensional slice specifiers.
  template <class... Args> requires gslice_view_arguments<Extents::rank(), Args...>
  [[nodiscard]]          auto  operator()(Args&&... args) const;
  template <class... Args> requires gslice_view_arguments<Extents::rank(), Args...>
  [[nodiscard]]          auto  operator()(Args&&... args)      ;
  
  // subset operations with slice.
  [[nodiscard]]          auto  operator[](const  slice          &selection) const { return                                   (*this)(selection); }
  [[nodiscard]]          auto  operator[](const  slice          &selection)       { return                                   (*this)(selection); }

  // subset operations with gslice (i.e., N-dimensional indices and slice specifiers).
  template <class... Args> requires gslice_arguments     <Extents::rank(), Args...>
  [[nodiscard]] decltype(auto) operator[](const gslice<Args...> &selection) const { return  detail::apply_indices(*this, selection.arguments()); }
  template <class... Args> requires gslice_arguments     <Extents::rank(), Args...>
  [[nodiscard]] decltype(auto) operator[](const gslice<Args...> &selection)       { return  detail::apply_indices(*this, selection.arguments()); }

  // subset operations with boolean mask.
  template <value_expression BoolExpr> requires detail::is_boolean_expr<std::remove_cvref_t<BoolExpr>>::value
  [[nodiscard]] mask_view<const element_type> operator[](BoolExpr &&expr) const {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return      mask_view<const element_type>(to_mdspan(), std::forward<BoolExpr>(expr));
  }
  template <value_expression BoolExpr> requires detail::is_boolean_expr<std::remove_cvref_t<BoolExpr>>::value
  [[nodiscard]] mask_view<      element_type> operator[](BoolExpr &&expr)       {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return      mask_view<      element_type>(to_mdspan(), std::forward<BoolExpr>(expr));
  }

  // subset operations with initializer-list offsets.
  [[nodiscard]] indirect_view<const element_type> operator[](std::initializer_list<size_type> offsets) const {
    return      indirect_view<const element_type>(to_mdspan(), std::vector<size_type>(offsets.begin(), offsets.end()));
  }
  [[nodiscard]] indirect_view<      element_type> operator[](std::initializer_list<size_type> offsets)       {
    return      indirect_view<      element_type>(to_mdspan(), std::vector<size_type>(offsets.begin(), offsets.end()));
  }
  // subset operations with array offsets.
  [[nodiscard]] indirect_view<const element_type> operator[](const ndarray<size_type, dextents<1>> &offsets) const {
    return      indirect_view<const element_type>(to_mdspan(), std::vector<size_type>(offsets.data(), offsets.data() + offsets.size()));
  }
  [[nodiscard]] indirect_view<      element_type> operator[](const ndarray<size_type, dextents<1>> &offsets)       {
    return      indirect_view<      element_type>(to_mdspan(), std::vector<size_type>(offsets.data(), offsets.data() + offsets.size()));
  }

  template <std::integral...                          Dims> [[nodiscard]] auto broadcast_to(                         Dims... dims) const;
  template <std::integral Extent, ::numcxx::size_type Rank> [[nodiscard]] auto broadcast_to(const std::array<Extent, Rank> &shape) const;

  // unary operators:
  auto operator+() const { return apply_unary_op<detail::nc_unary_plus>(); }
  auto operator-() const { return apply_unary_op<std::negate>          (); }
  auto operator~() const { return apply_unary_op<detail::nc_bit_not>   (); }
  auto operator!() const { return apply_unary_op<std::logical_not>     (); }

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

  template <value_expression Expr> ndarray &operator=  (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <value_expression Expr> ndarray &operator+= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <value_expression Expr> ndarray &operator-= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <value_expression Expr> ndarray &operator*= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <value_expression Expr> ndarray &operator/= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <value_expression Expr> ndarray &operator%= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <value_expression Expr> ndarray &operator&= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <value_expression Expr> ndarray &operator|= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <value_expression Expr> ndarray &operator^= (const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <value_expression Expr> ndarray &operator<<=(const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <value_expression Expr> ndarray &operator>>=(const Expr &v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

  // observers
  [[nodiscard]] constexpr          size_type     size() const noexcept { return elem_.size()   ; }
  [[nodiscard]] constexpr               bool    empty() const noexcept { return size() == 0    ; }
  [[nodiscard]] constexpr const extents_type& extents() const noexcept { return elem_.extents(); }
  [[nodiscard]] constexpr      const_pointer     data() const noexcept { return elem_.data()   ; }
  [[nodiscard]] constexpr            pointer     data()       noexcept { return elem_.data()   ; }

  [[nodiscard]] static constexpr bool is_always_unique     () noexcept { return true; }
  [[nodiscard]] static constexpr bool is_always_exhaustive () noexcept { return true; }
  [[nodiscard]] static constexpr bool is_always_strided    () noexcept { return true; }
  [[nodiscard]]        constexpr bool is_unique            () const    { return true; }
  [[nodiscard]]        constexpr bool is_exhaustive        () const    { return true; }
  [[nodiscard]]        constexpr bool is_strided           () const    { return true; }
  [[nodiscard]] constexpr  index_type stride(rank_type     r) const    { return elem_.stride(r); }

  // mdspan interoperability
         constexpr  const_mdspan_type                to_mdspan() const noexcept { return elem_.to_mdspan(); }
         constexpr        mdspan_type                to_mdspan()       noexcept { return elem_.to_mdspan(); }

  // member functions:
  void swap(ndarray &v) noexcept;

  [[nodiscard]] value_type sum() const { return std::accumulate(data(), data() + size(), value_type{})             ; }
  [[nodiscard]] value_type min() const { return empty() ? value_type{} : *std::min_element(data(), data() + size()); }
  [[nodiscard]] value_type max() const { return empty() ? value_type{} : *std::max_element(data(), data() + size()); }
  [[nodiscard]] value_type prod() const {
    if (empty())
      return value_type{1};
    return std::accumulate(data(), data() + size(), value_type{1},
                           std::multiplies<>{});
  }
  [[nodiscard]] detail::reduction_mean_type_t<value_type> mean() const;
  [[nodiscard]] detail::reduction_real_type_t<value_type> var() const;
  [[nodiscard]] detail::reduction_real_type_t<value_type> std() const;
  [[nodiscard]] size_type argmin() const;
  [[nodiscard]] size_type argmax() const;

  [[nodiscard]] auto sum(::numcxx::index_type axis) const;
  [[nodiscard]] auto min(::numcxx::index_type axis) const;
  [[nodiscard]] auto max(::numcxx::index_type axis) const;
  [[nodiscard]] auto prod(::numcxx::index_type axis) const;
  [[nodiscard]] auto mean(::numcxx::index_type axis) const;
  [[nodiscard]] auto var(::numcxx::index_type axis) const;
  [[nodiscard]] auto std(::numcxx::index_type axis) const;
  [[nodiscard]] auto argmin(::numcxx::index_type axis) const;
  [[nodiscard]] auto argmax(::numcxx::index_type axis) const;

  [[nodiscard]] bool any() const;
  [[nodiscard]] bool all() const;
  // Axis reductions return 0/1 values in an unsigned-char ndarray.
  [[nodiscard]] auto any(::numcxx::index_type axis) const;
  [[nodiscard]] auto all(::numcxx::index_type axis) const;
  // void resize(size_type n, value_type x = value_type());

private:
  template <class, class, class> friend class       ndarray;
  template <class, class, class> friend class    gslice_view;
  template <class>               friend class     mask_view;
  template <class>               friend class indirect_view;
  template <class>               friend class   nc_val_expr;
  // clang-format on

  ::numcxx::index_type normalize_reduction_axis(
      ::numcxx::index_type axis, const char *function_name) const {
    const ::numcxx::index_type rank_value =
        static_cast<::numcxx::index_type>(extents_type::rank());
    if (axis < 0)
      axis += rank_value;
    if (axis < 0 || axis >= rank_value)
      NUMCXX_THROW(std::out_of_range,
                   std::string(function_name) + ": axis out of range");
    return axis;
  }

  template <class ResultValue, class Transform, class Operation>
  auto reduce_axis_impl(::numcxx::index_type axis, const char *function_name,
                        ResultValue identity, Transform transform,
                        Operation operation,
                        bool use_first = false) const {
    using result_extents_type =
        dextents<(extents_type::rank() > 0 ? extents_type::rank() - 1 : 0)>;
    using result_type = ndarray<ResultValue, result_extents_type>;

    axis = normalize_reduction_axis(axis, function_name);

    std::array<size_type,
               (extents_type::rank() > 0 ? extents_type::rank() - 1 : 0)>
        output_shape{};
    for (size_type src = 0, dst = 0; src < extents_type::rank(); ++src)
      if (src != static_cast<size_type>(axis))
        output_shape[dst++] = static_cast<size_type>(extent(src));

    auto result = [&]<size_type... Is>(std::index_sequence<Is...>) {
      return result_type(static_cast<index_type>(output_shape[Is])...);
    }(std::make_index_sequence<
       (extents_type::rank() > 0 ? extents_type::rank() - 1 : 0)>{});

    for (size_type output_flat = 0; output_flat < result.size();
         ++output_flat) {
      const auto output_indices =
          unravel_index<default_layout>(output_flat, result.extents());
      std::array<size_type, extents_type::rank()> input_indices{};
      for (size_type src = 0, dst = 0; src < extents_type::rank(); ++src)
        if (src != static_cast<size_type>(axis))
          input_indices[src] = output_indices[dst++];

      ResultValue value = identity;
      bool initialized = false;
      for (size_type reduced = 0;
           reduced < static_cast<size_type>(extent(axis)); ++reduced) {
        input_indices[axis] = reduced;
        const value_type &element = data()[std::apply(
            [&](auto... indices) {
              return elem_.mapping()(indices...);
            },
            input_indices)];
        const ResultValue transformed =
            transform(element, output_flat, reduced);
        if (use_first && !initialized)
          value = transformed;
        else
          value = operation(value, transformed);
        initialized = true;
      }
      result.flat_unchecked(output_flat) = value;
    }
    return result;
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }

  template <typename Op>
  ndarray &apply_scalar_op(Op &&op, const value_type &x) {
    auto ptr = data();
    const size_type n = size();
#pragma omp simd
    for (size_type i = 0; i < n; ++i)
      op(ptr[i], x);
    return *this;
  }

  template <class Op, class Expr>
  ndarray &apply_expr_op(Op &&op, const Expr &expr) {
    detail::check_shape_match(*this, expr);
    const size_type n = size();
#pragma omp simd
    for (size_type i = 0; i < n; ++i)
      op(flat_unchecked(i), expr.flat_unchecked(i));
    return *this;
  }

  template <class Expr, size_type... Is>
  static mdarray_type make_expression_storage(
      const Expr &expr, std::index_sequence<Is...>) {
    if constexpr (detail::static_extents<extents_type>) {
      mdarray_type storage(extents_type{});
      for (rank_type r = 0; r < extents_type::rank(); ++r)
        if (expr.extent(r) != storage.extent(r))
          NUMCXX_THROW(std::invalid_argument,
                       "expression shape does not match static extents");
      return storage;
    } else {
      return mdarray_type(extents_type(
          static_cast<index_type>(expr.extent(Is))...));
    }
  }

private:
  mdarray_type elem_;
  // std::vector<size_type> logical_offset_;
};

// template <class Tp, size_type _Size>
// ndarray(const Tp(&)[_Size], size_type) -> ndarray<Tp>;

// extern template void ndarray<size_type>::resize(size_type, size_type);

// [numcxx.gslice_view]
template <class ElementType, class Extents, class LayoutPolicy = default_layout>
class gslice_view {
public:
  // clang-format off
  // element
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  // extents
  using extents_type = Extents;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  // layout
  using  layout_type = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;

  // mdspan
  using const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type>;
  using       mdspan_type = detail::mdspan<      element_type, extents_type, layout_type>;

  // reference
  using const_reference = typename const_mdspan_type::reference;
  using       reference = typename       mdspan_type::reference;

  // data_handle
  using const_data_handle_type = typename const_mdspan_type::data_handle_type;
  using       data_handle_type = typename       mdspan_type::data_handle_type;

  static constexpr  rank_type  rank        (           )       noexcept { return extents_type::rank()          ; }
  static constexpr  rank_type  rank_dynamic(           )       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr  size_type static_extent(rank_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr index_type        extent(rank_type r) const noexcept { return span_.extent(r)               ; }

  //--------------------------------------------------------------------------------
  // [numcxx.gslice_view.cons], gslice_view constructors, destructor, and assignment

  // construct/destroy:
  gslice_view() = delete;
  gslice_view(const gslice_view &) = default;
  gslice_view(gslice_view &&) noexcept = default;
  explicit gslice_view(mdspan_type span, std::array<bool, extents_type::rank()> reverse = {})
    : span_(span), reverse_dims_(reverse),
      has_reversed_dim_(std::any_of(reverse.begin(), reverse.end(), [](bool b) { return b; })),
      flat_contiguous_(is_flat_contiguous(span, reverse)) {}
  ~gslice_view() = default;

  // assignment:
  gslice_view &operator=(const gslice_view &) = default;
  gslice_view &operator=(gslice_view &&) noexcept = default;

  // element access using N-dimensional indices.
  template <class... Idxs> requires elem_access_arguments<Extents::rank(), Idxs...>
  [[nodiscard]] const_reference operator() (Idxs&&... idxs) const;
  template <class... Idxs> requires elem_access_arguments<Extents::rank(), Idxs...>
  [[nodiscard]]       reference operator() (Idxs&&... idxs)      ;

  // flattened element access in default_layout order, independent of the physical storage layout.
  [[nodiscard]] const_reference flat_unchecked(size_type i) const;
  [[nodiscard]]       reference flat_unchecked(size_type i)      ;
  [[nodiscard]] const_reference flat          (size_type i) const { detail::checked_index(i, size(), "gslice_view::flat"); return             flat_unchecked(i); }
  [[nodiscard]]       reference flat          (size_type i)       { detail::checked_index(i, size(), "gslice_view::flat"); return             flat_unchecked(i); }

  // flattened element access in physical storage order: operator[] is unchecked; at() checks bounds.
  [[nodiscard]] const_reference operator[]    (size_type i) const {                                                        return data_handle()[calc_offset(i)]; }
  [[nodiscard]]       reference operator[]    (size_type i)       {                                                        return data_handle()[calc_offset(i)]; }
  [[nodiscard]] const_reference at            (size_type i) const { detail::checked_index(i, size(), "gslice_view::at"  ); return                    (*this)[i]; }
  [[nodiscard]]       reference at            (size_type i)       { detail::checked_index(i, size(), "gslice_view::at"  ); return                    (*this)[i]; }

  // subset operations using N-dimensional slice specifiers.
  template <typename... Args> requires gslice_view_arguments<Extents::rank(), Args...>
  [[nodiscard]] auto operator()(Args&&... args) const;
  template <typename... Args> requires gslice_view_arguments<Extents::rank(), Args...>
  [[nodiscard]] auto operator()(Args&&... args)      ;

  // subset operations with slice.
  [[nodiscard]] auto operator[](const slice &selection) const {                     return                                  (*this)(selection); }
  [[nodiscard]] auto operator[](const slice &selection)       {                     return                                  (*this)(selection); }

  // subset operations with gslice (i.e., N-dimensional indices and slice specifiers).
  template <class... Args> requires gslice_arguments     <Extents::rank(), Args...>
  [[nodiscard]] decltype(auto) operator[](const gslice<Args...> &selection) const { return detail::apply_indices(*this, selection.arguments()); }
  template <class... Args> requires gslice_arguments     <Extents::rank(), Args...>
  [[nodiscard]] decltype(auto) operator[](const gslice<Args...> &selection)       { return detail::apply_indices(*this, selection.arguments()); }

  // subset operations with boolean mask.
  template <value_expression BoolExpr> requires detail::is_boolean_expr<std::remove_cvref_t<BoolExpr>>::value
  [[nodiscard]] mask_view<const element_type> operator[](BoolExpr &&expr) const {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return mask_view<const element_type>(to_mdspan(), extract_mask_offsets(expr));
  }
  template <value_expression BoolExpr> requires detail::is_boolean_expr<std::remove_cvref_t<BoolExpr>>::value
  [[nodiscard]] mask_view<      element_type> operator[](BoolExpr &&expr)       {
    NUMCXX_ASSERT(expr.size() == size(), "mask size must match array size");
    return mask_view<      element_type>(to_mdspan(), extract_mask_offsets(expr));
  } 

  // subset operations with initializer-list offsets.
  [[nodiscard]] indirect_view<const element_type> operator[](std::initializer_list<size_type> offsets) const {
    return indirect_view<const element_type>(to_mdspan(), extract_indirect_offsets(offsets.begin(), offsets.end()));
  }
  [[nodiscard]] indirect_view<      element_type> operator[](std::initializer_list<size_type> offsets)       {
    return indirect_view<      element_type>(to_mdspan(), extract_indirect_offsets(offsets.begin(), offsets.end()));
  }
  // subset operations with array offsets.
  [[nodiscard]] indirect_view<const element_type> operator[](const ndarray<size_type, dextents<1>> &offsets) const {
    return indirect_view<const element_type>(to_mdspan(), extract_indirect_offsets(offsets.data(), offsets.data() + offsets.size()));
  }
  [[nodiscard]] indirect_view<      element_type> operator[](const ndarray<size_type, dextents<1>> &offsets)       {
    return indirect_view<      element_type>(to_mdspan(), extract_indirect_offsets(offsets.data(), offsets.data() + offsets.size()));
  }

  template <std::integral...                          Dims> [[nodiscard]] auto broadcast_to(                         Dims... dims) const;
  template <std::integral Extent, ::numcxx::size_type Rank> [[nodiscard]] auto broadcast_to(const std::array<Extent, Rank> &shape) const;

  // unary operators:
  auto operator+() const { return apply_unary_op<detail::nc_unary_plus>(); }
  auto operator-() const { return apply_unary_op<std::negate>          (); }
  auto operator~() const { return apply_unary_op<detail::nc_bit_not>   (); }
  auto operator!() const { return apply_unary_op<std::logical_not>     (); }

  // computed assignment:
  gslice_view &operator=  (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a = b  ; }, x); }
  gslice_view &operator+= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a += b ; }, x); }
  gslice_view &operator-= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a -= b ; }, x); }
  gslice_view &operator*= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a *= b ; }, x); }
  gslice_view &operator/= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a /= b ; }, x); }
  gslice_view &operator%= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a %= b ; }, x); }
  gslice_view &operator&= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a &= b ; }, x); }
  gslice_view &operator|= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a |= b ; }, x); }
  gslice_view &operator^= (const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a ^= b ; }, x); }
  gslice_view &operator<<=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a <<= b; }, x); }
  gslice_view &operator>>=(const value_type &x) { return apply_scalar_op([](value_type &a, value_type b) { a >>= b; }, x); }

  template <value_expression Expr> gslice_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <value_expression Expr> gslice_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <value_expression Expr> gslice_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <value_expression Expr> gslice_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <value_expression Expr> gslice_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <value_expression Expr> gslice_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <value_expression Expr> gslice_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <value_expression Expr> gslice_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <value_expression Expr> gslice_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <value_expression Expr> gslice_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <value_expression Expr> gslice_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

  // observers
  [[nodiscard]] constexpr              size_type         size() const noexcept { return span_.size()       ; }
  [[nodiscard]] constexpr                   bool        empty() const noexcept { return span_.empty()      ; }
  [[nodiscard]] constexpr const     extents_type&     extents() const noexcept { return span_.extents()    ; }
  [[nodiscard]] constexpr const data_handle_type& data_handle() const noexcept { return span_.data_handle(); }

  [[nodiscard]] static constexpr bool is_always_unique     () noexcept { return mapping_type::is_always_unique()    ; }
  [[nodiscard]] static constexpr bool is_always_exhaustive () noexcept { return mapping_type::is_always_exhaustive(); }
  [[nodiscard]] static constexpr bool is_always_strided    () noexcept { return mapping_type::is_always_strided()   ; }
  [[nodiscard]]        constexpr bool is_unique            () const    { return span_.mapping().is_unique()         ; }
  [[nodiscard]]        constexpr bool is_exhaustive        () const    { return span_.mapping().is_exhaustive()     ; }
  [[nodiscard]]        constexpr bool is_strided           () const    { return span_.mapping().is_strided()        ; }
  [[nodiscard]] constexpr  index_type stride(rank_type     r) const    { return span_.stride(r)                     ; }
  [[nodiscard]] constexpr bool has_reversed_dim() const noexcept { return has_reversed_dim_; }
  [[nodiscard]] auto eval() const;

  [[nodiscard]] auto sum() const { return eval().sum(); }
  [[nodiscard]] auto min() const { return eval().min(); }
  [[nodiscard]] auto max() const { return eval().max(); }
  [[nodiscard]] auto prod() const { return eval().prod(); }
  [[nodiscard]] auto mean() const { return eval().mean(); }
  [[nodiscard]] auto var() const { return eval().var(); }
  [[nodiscard]] auto std() const { return eval().std(); }
  [[nodiscard]] bool any() const { return eval().any(); }
  [[nodiscard]] bool all() const { return eval().all(); }
  [[nodiscard]] auto argmin() const { return eval().argmin(); }
  [[nodiscard]] auto argmax() const { return eval().argmax(); }

  [[nodiscard]] auto sum(::numcxx::index_type axis) const { return eval().sum(axis); }
  [[nodiscard]] auto min(::numcxx::index_type axis) const { return eval().min(axis); }
  [[nodiscard]] auto max(::numcxx::index_type axis) const { return eval().max(axis); }
  [[nodiscard]] auto prod(::numcxx::index_type axis) const { return eval().prod(axis); }
  [[nodiscard]] auto mean(::numcxx::index_type axis) const { return eval().mean(axis); }
  [[nodiscard]] auto var(::numcxx::index_type axis) const { return eval().var(axis); }
  [[nodiscard]] auto std(::numcxx::index_type axis) const { return eval().std(axis); }
  [[nodiscard]] auto any(::numcxx::index_type axis) const { return eval().any(axis); }
  [[nodiscard]] auto all(::numcxx::index_type axis) const { return eval().all(axis); }
  [[nodiscard]] auto argmin(::numcxx::index_type axis) const {
    return eval().argmin(axis);
  }
  [[nodiscard]] auto argmax(::numcxx::index_type axis) const {
    return eval().argmax(axis);
  }

  // mdspan interoperability
         constexpr  const_mdspan_type                to_mdspan() const noexcept { return span_              ; }
         constexpr        mdspan_type                to_mdspan()       noexcept { return span_              ; }
  // clang-format on

private:
  static bool is_flat_contiguous(
      const mdspan_type &span,
      const std::array<bool, extents_type::rank()> &reverse) noexcept {
    if (span.size() <= 1)
      return true;
    if (!span.mapping().is_exhaustive())
      return false;
    for (rank_type r = 0; r < extents_type::rank(); ++r)
      if (reverse[r] && span.extent(r) > 1)
        return false;

    size_type expected_stride = 1;
    auto check_dim = [&](rank_type r) {
      if (span.extent(r) > 1 && span.stride(r) != expected_stride)
        return false;
      expected_stride *= span.extent(r);
      return true;
    };

    if constexpr (std::is_same_v<default_layout, layout_right>) {
      for (rank_type r = extents_type::rank(); r-- > 0;)
        if (!check_dim(r))
          return false;
    } else {
      for (rank_type r = 0; r < extents_type::rank(); ++r)
        if (!check_dim(r))
          return false;
    }
    return true;
  }

  size_type calc_offset(size_type i) const noexcept {
    if (size() == 0)
      return 0;
    size_type offset = 0, remaining = i;
    auto process_dim = [&](rank_type r) {
      size_type dim_len = extent(r);
      size_type idx = remaining % dim_len;
      if (reverse_dims_[r]) {
        idx = dim_len - 1 - idx;
      }
      offset += idx * span_.stride(r);
      remaining /= dim_len;
    };
    if constexpr (std::is_same_v<layout_type, layout_right>) {
      for (rank_type r = rank(); r-- > 0;)
        process_dim(r);
    } else {
      for (rank_type r = 0; r < rank(); ++r)
        process_dim(r);
    }
    return offset;
  }

  size_type calc_flat_offset(size_type i) const noexcept {
    size_type offset = 0, remaining = i;
    auto process_dim = [&](rank_type r) {
      const size_type dim_len = extent(r);
      size_type idx = remaining % dim_len;
      if (reverse_dims_[r])
        idx = dim_len - 1 - idx;
      offset += idx * span_.stride(r);
      remaining /= dim_len;
    };
    if constexpr (std::is_same_v<default_layout, layout_right>) {
      for (rank_type r = rank(); r-- > 0;)
        process_dim(r);
    } else {
      for (rank_type r = 0; r < rank(); ++r)
        process_dim(r);
    }
    return offset;
  }

  template <typename BoolExpr>
  std::vector<size_type> extract_mask_offsets(const BoolExpr& expr) const {
    std::vector<size_type> offsets;
    offsets.reserve(expr.size());
    for (size_type i = 0; i < expr.size(); ++i)
      if (static_cast<bool>(expr.flat_unchecked(i)))
        offsets.push_back(calc_flat_offset(i));
    return offsets;
  }

  template <typename Iterator>
  std::vector<size_type> extract_indirect_offsets(Iterator first, Iterator last) const {
    std::vector<size_type> offsets(first, last);
    for (auto &offset : offsets)
      offset = calc_flat_offset(offset);
    return offsets;
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }

  template <typename Op>
  gslice_view &apply_scalar_op(Op &&op, const value_type &x) {
#pragma omp simd
    for (size_type i = 0; i < size(); ++i)
      op(flat_unchecked(i), x);
    return *this;
  }

  template <class Op, class Expr>
  gslice_view &apply_expr_op(Op &&op, const Expr &expr) {
    detail::check_shape_match(*this, expr);
#pragma omp simd
    for (size_type i = 0; i < size(); ++i)
      op(flat_unchecked(i), expr.flat_unchecked(i));
    return *this;
  }

private:
  mdspan_type span_;
  std::array<bool, extents_type::rank()> reverse_dims_{};
  bool has_reversed_dim_;
  bool flat_contiguous_;
  // std::vector<size_type> logical_offset_;
};

namespace detail {
struct broadcast_view_factory;
}

// A read-only, non-owning view with explicit zero-stride broadcast axes.
// It intentionally has no mdspan conversion: layout_stride requires unique
// mappings, while a broadcast mapping is non-unique by definition.
template <class ElementType, size_type Rank> class broadcast_view {
public:
  using element_type = std::remove_const_t<ElementType>;
  using value_type = std::remove_cv_t<ElementType>;
  using extents_type = dextents<Rank>;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;
  using const_reference = const value_type &;

  static constexpr rank_type rank() noexcept { return extents_type::rank(); }
  static constexpr rank_type rank_dynamic() noexcept {
    return extents_type::rank_dynamic();
  }
  static constexpr size_type static_extent(rank_type r) noexcept {
    return extents_type::static_extent(r);
  }
  constexpr index_type extent(rank_type r) const noexcept {
    return extents_.extent(r);
  }
  constexpr index_type stride(rank_type r) const noexcept {
    return strides_[r];
  }

  template <typename... Idxs>
    requires all_scalar_indices<Idxs...>
  [[nodiscard]] const_reference operator()(Idxs... idxs) const {
    static_assert(sizeof...(Idxs) == Rank,
                  "Number of arguments must match broadcast_view rank");
    const std::array<index_type, Rank> indices{
        static_cast<index_type>(idxs)...};
    size_type offset = 0;
    for (rank_type r = 0; r < rank(); ++r) {
      const index_type dim_size = static_cast<index_type>(extent(r));
      const index_type index =
          indices[r] < 0 ? indices[r] + dim_size : indices[r];
      NUMCXX_ASSERT(index >= 0 && index < dim_size,
                    "broadcast_view index out of bounds");
      offset +=
          static_cast<size_type>(index) * static_cast<size_type>(stride(r));
    }
    return data_[offset];
  }

  [[nodiscard]] const_reference flat(size_type i) const {
    detail::checked_index(i, size(), "broadcast_view::flat");
    return flat_unchecked(i);
  }

  // Internal API (not stable): unchecked flat access; requires i < size().
  // Broadcast indexing still computes the multidimensional storage offset.
  [[nodiscard]] const_reference flat_unchecked(size_type i) const {
    std::array<size_type, Rank> indices{};
    if constexpr (std::is_same_v<default_layout, layout_right>) {
      for (rank_type r = rank(); r-- > 0;) {
        indices[r] = i % extent(r);
        i /= extent(r);
      }
    } else {
      for (rank_type r = 0; r < rank(); ++r) {
        indices[r] = i % extent(r);
        i /= extent(r);
      }
    }
    size_type offset = 0;
    for (rank_type r = 0; r < rank(); ++r)
      offset += indices[r] * static_cast<size_type>(stride(r));
    return data_[offset];
  }

  [[nodiscard]] constexpr size_type size() const noexcept {
    size_type result = 1;
    for (rank_type r = 0; r < rank(); ++r)
      result *= static_cast<size_type>(extent(r));
    return result;
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }
  [[nodiscard]] constexpr const extents_type &extents() const noexcept {
    return extents_;
  }
  [[nodiscard]] constexpr const value_type *data_handle() const noexcept {
    return data_;
  }
  [[nodiscard]] constexpr bool is_unique() const noexcept {
    for (rank_type r = 0; r < rank(); ++r)
      if (extent(r) > 1 && stride(r) == 0)
        return false;
    return true;
  }
  [[nodiscard]] constexpr bool is_exhaustive() const noexcept {
    return exhaustive_;
  }
  [[nodiscard]] static constexpr bool is_always_unique() noexcept {
    return false;
  }
  [[nodiscard]] static constexpr bool is_always_exhaustive() noexcept {
    return false;
  }
  [[nodiscard]] static constexpr bool is_always_strided() noexcept {
    return true;
  }

  template <std::integral... Dims>
  [[nodiscard]] auto broadcast_to(Dims... dims) const;
  template <std::integral Extent, ::numcxx::size_type TargetRank>
  [[nodiscard]] auto
  broadcast_to(const std::array<Extent, TargetRank> &shape) const;

  [[nodiscard]] auto eval() const {
    return ndarray<value_type, extents_type>(*this);
  }

private:
  friend struct detail::broadcast_view_factory;

  broadcast_view(const value_type *data, const extents_type &extents,
                 const std::array<index_type, Rank> &strides, bool exhaustive)
      : data_(data), extents_(extents), strides_(strides),
        exhaustive_(exhaustive) {}

  const value_type *data_;
  extents_type extents_;
  std::array<index_type, Rank> strides_;
  bool exhaustive_;
};

namespace detail {
struct broadcast_view_factory {
  template <class T, size_type Rank>
  static auto
  create(const T *data, const ::numcxx::dextents<Rank> &extents,
         const std::array<typename ::numcxx::dextents<Rank>::index_type, Rank>
             &strides,
         bool exhaustive) {
    return broadcast_view<T, Rank>(data, extents, strides, exhaustive);
  }
};

template <std::integral Extent>
size_type checked_broadcast_extent(Extent extent) {
  if constexpr (std::is_signed_v<Extent>)
    if (extent < 0)
      NUMCXX_THROW(std::invalid_argument,
                   "broadcast_to: target extents must be non-negative");
  return static_cast<size_type>(extent);
}

template <class Source, std::integral Extent, size_type TargetRank>
auto make_broadcast_view(const Source &source,
                         const std::array<Extent, TargetRank> &target_shape) {
  constexpr size_type source_rank = Source::rank();
  using value_type = typename Source::value_type;
  using index_type = typename Source::index_type;
  using extents_type = ::numcxx::dextents<TargetRank>;

  if constexpr (source_rank > TargetRank) {
    NUMCXX_THROW(std::invalid_argument,
                 "broadcast_to: target rank must be at least the source rank");
  } else {
    std::array<size_type, TargetRank> shape{};
    std::transform(
        target_shape.begin(), target_shape.end(), shape.begin(),
        [](Extent extent) { return checked_broadcast_extent(extent); });

    if constexpr (requires { source.has_reversed_dim(); })
      if (source.has_reversed_dim())
        NUMCXX_THROW(std::invalid_argument,
                     "broadcast_to: reversed gslice_view sources are not "
                     "supported; materialize with eval() first");

    std::array<index_type, TargetRank> strides{};
    constexpr size_type leading_dims = TargetRank - source_rank;
    bool exhaustive = source.is_exhaustive();
    for (size_type r = 0; r < leading_dims; ++r)
      if (shape[r] > 1)
        exhaustive = false;

    for (size_type r = 0; r < source_rank; ++r) {
      const size_type source_extent = static_cast<size_type>(source.extent(r));
      const size_type target_extent = shape[leading_dims + r];
      if (source_extent != target_extent && source_extent != 1) {
        const std::string message =
            "broadcast_to: incompatible extent at dimension " +
            std::to_string(leading_dims + r) + " (source " +
            std::to_string(source_extent) + ", target " +
            std::to_string(target_extent) + ")";
        NUMCXX_THROW(std::invalid_argument, message.c_str());
      }
      strides[leading_dims + r] =
          source_extent == target_extent
              ? static_cast<index_type>(source.stride(r))
              : index_type{};
      if (source_extent != target_extent && target_extent > 1)
        exhaustive = false;
    }

    const extents_type extents(shape);
    const auto *data = [&]() {
      if constexpr (requires { source.to_mdspan(); })
        return source.to_mdspan().data_handle();
      else
        return source.data_handle();
    }();
    return broadcast_view_factory::create<value_type, TargetRank>(
        data, extents, strides, exhaustive);
  }
}

template <class Source, std::integral... Dims>
auto make_broadcast_view(const Source &source, Dims... dims) {
  static_assert(sizeof...(Dims) > 0,
                "broadcast_to requires at least one target extent");
  constexpr size_type target_rank = sizeof...(Dims);
  const std::array<size_type, target_rank> shape{
      checked_broadcast_extent(dims)...};
  return make_broadcast_view(source, shape);
}

template <class T> struct is_broadcast_source : std::false_type {};
template <class Tp, class Ex, class Lp>
struct is_broadcast_source<ndarray<Tp, Ex, Lp>> : std::true_type {};
template <class Tp, class Ex, class Lp>
struct is_broadcast_source<gslice_view<Tp, Ex, Lp>> : std::true_type {};
template <class Tp, size_type Rank>
struct is_broadcast_source<broadcast_view<Tp, Rank>> : std::true_type {};

// clang-format off
template <class Tp> class index_accessor {
public:
  using element_type     = Tp;
  using reference        = element_type &;
  using data_handle_type = element_type *;
  using offset_policy    = index_accessor;

           index_accessor() = default;
  explicit index_accessor(const std::vector<size_type> &idxs) : indices_(idxs) {}

  reference access(const data_handle_type p, const size_type i) const noexcept {
    NUMCXX_ASSERT(i < indices_.size(), "index_accessor::access: index out of bounds");
    return p[indices_[i]];
  }
  
  data_handle_type offset(const data_handle_type p, const size_type i) const noexcept { return p + indices_[i]         ; }
  friend bool operator==(const index_accessor &a, const index_accessor &b)   noexcept { return a.indices_ == b.indices_; }
  friend bool operator!=(const index_accessor &a, const index_accessor &b)   noexcept { return        !(a == b)        ; }

private:
  std::vector<size_type> indices_;
};
// clang-format on
} // namespace detail

// [numcxx.mask_view]
template <class ElementType> class mask_view {
public:
  // clang-format off
  // element
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  // extents
  using extents_type = dextents<1>;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  // layout and accessor
  using         layout_type = default_layout;
  using const_accessor_type = detail::index_accessor<const element_type>;
  using       accessor_type = detail::index_accessor<      element_type>;

  // mdspan
  using   const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type, const_accessor_type>;
  using         mdspan_type = detail::mdspan<      element_type, extents_type, layout_type,       accessor_type>;

  // reference
  using const_reference = typename const_mdspan_type::reference;
  using       reference = typename       mdspan_type::reference;

  // data_handle
  using const_data_handle_type = typename const_mdspan_type::data_handle_type;
  using       data_handle_type = typename       mdspan_type::data_handle_type;

  static constexpr  rank_type  rank        (           )       noexcept { return extents_type::rank()          ; }
  static constexpr  rank_type  rank_dynamic(           )       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr  size_type static_extent(rank_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr index_type        extent(rank_type r) const noexcept { return span_.extent(r)               ; }

  //--------------------------------------------------------------------------------
  // [numcxx.mask_view.cons], mask_view constructors, destructor, and assignment

  // construct/destroy:
  mask_view() = delete;
  mask_view(const mask_view &) = default;
  mask_view(mask_view &&) noexcept = default;
  ~mask_view() = default;

  // Linear access in view order: operator[] is unchecked; at() checks bounds.
  [[nodiscard]] const_reference operator[](size_type i) const {
    return flat_unchecked(i);
  }
  [[nodiscard]] reference operator[](size_type i) {
    return flat_unchecked(i);
  }
  [[nodiscard]] const_reference at(size_type i) const {
    detail::checked_index(i, size(), "mask_view::at");
    return (*this)[i];
  }
  [[nodiscard]] reference at(size_type i) {
    detail::checked_index(i, size(), "mask_view::at");
    return (*this)[i];
  }

  // flattened element access using the default_layout order, independent of the actual storage layout.
  [[nodiscard]] const_reference flat(size_type i) const {
    detail::checked_index(i, size(), "mask_view::flat");
    return flat_unchecked(i);
  }
  [[nodiscard]] reference flat(size_type i) {
    detail::checked_index(i, size(), "mask_view::flat");
    return flat_unchecked(i);
  }
  // Internal API (not stable): unchecked flat access; requires i < size().
  // Bypass mdspan::operator[] to avoid any potential bounds checking.
  [[nodiscard]] const_reference flat_unchecked(size_type i) const {
    return span_.accessor().offset(span_.data_handle(), i)[0];
  }
  [[nodiscard]] reference flat_unchecked(size_type i) {
    return span_.accessor().offset(span_.data_handle(), i)[0];
  }

  // unary operators:
  auto operator+() const { return apply_unary_op<detail::nc_unary_plus>(); }
  auto operator-() const { return apply_unary_op<std::negate>          (); }
  auto operator~() const { return apply_unary_op<detail::nc_bit_not>   (); }
  auto operator!() const { return apply_unary_op<std::logical_not>     (); }

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

  template <value_expression Expr> mask_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <value_expression Expr> mask_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <value_expression Expr> mask_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <value_expression Expr> mask_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <value_expression Expr> mask_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <value_expression Expr> mask_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <value_expression Expr> mask_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <value_expression Expr> mask_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <value_expression Expr> mask_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <value_expression Expr> mask_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <value_expression Expr> mask_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

  // observers
  [[nodiscard]] constexpr              size_type         size() const noexcept { return span_.size()       ; }
  [[nodiscard]] constexpr                   bool        empty() const noexcept { return span_.empty()      ; }
  [[nodiscard]] constexpr const     extents_type&     extents() const noexcept { return span_.extents()    ; }
  [[nodiscard]] constexpr const data_handle_type& data_handle() const noexcept { return span_.data_handle(); }

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
      if (static_cast<bool>(expr.flat_unchecked(i)))
        offsets.push_back(mapping(i));
    }

    auto base = data_span.data_handle();
    extents_type ext(offsets.size());
    auto acc = detail::index_accessor<element_type>(std::move(offsets));
    span_ = mdspan_type(base, ext, acc);
  }

  template <typename MdSpan>
  explicit mask_view(const MdSpan &data_span,
                     std::vector<size_type> offsets) {
    auto base = data_span.data_handle();
    extents_type ext(offsets.size());
    auto acc = detail::index_accessor<element_type>(std::move(offsets));
    span_ = mdspan_type(base, ext, acc);
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }

  template <typename Op>
  mask_view &apply_scalar_op(Op &&op, const value_type &x) {
    for (size_type i = 0; i < size(); ++i)
      op(flat_unchecked(i), x);
    return *this;
  }

  template <class Op, class Expr>
  mask_view &apply_expr_op(Op &&op, const Expr &expr) {
    detail::check_shape_match(*this, expr);
    for (size_type i = 0; i < size(); ++i)
      op(flat_unchecked(i), expr.flat_unchecked(i));
    return *this;
  }

  template <class, class, class> friend class ndarray;
  template <class, class, class> friend class gslice_view;

  mdspan_type span_;
};

// [numcxx.indirect_view]
template <class ElementType> class indirect_view {
public:
  // clang-format off
  // element
  using element_type = ElementType;
  using   value_type = std::remove_cv_t<element_type>;

  // extents
  using extents_type = dextents<1>;
  using   index_type = typename extents_type::index_type;
  using    size_type = typename extents_type::size_type;
  using    rank_type = typename extents_type::rank_type;

  // layout and accessor
  using         layout_type = default_layout;
  using const_accessor_type = detail::index_accessor<const element_type>;
  using       accessor_type = detail::index_accessor<      element_type>;

  // mdspan
  using   const_mdspan_type = detail::mdspan<const element_type, extents_type, layout_type, const_accessor_type>;
  using         mdspan_type = detail::mdspan<      element_type, extents_type, layout_type,       accessor_type>;

  // reference
  using const_reference = typename const_mdspan_type::reference;
  using       reference = typename       mdspan_type::reference;

  // data_handle
  using const_data_handle_type = typename const_mdspan_type::data_handle_type;
  using       data_handle_type = typename       mdspan_type::data_handle_type;

  static constexpr  rank_type  rank        (           )       noexcept { return extents_type::rank()          ; }
  static constexpr  rank_type  rank_dynamic(           )       noexcept { return extents_type::rank_dynamic()  ; }
  static constexpr  size_type static_extent(rank_type r)       noexcept { return extents_type::static_extent(r); }
         constexpr index_type        extent(rank_type r) const noexcept { return span_.extent(r)               ; }

  //--------------------------------------------------------------------------------
  // [numcxx.indirect_view.cons], indirect_view constructors, destructor, and assignment

  // construct/destroy:
  indirect_view() = delete;
  indirect_view(const indirect_view &) = default;
  indirect_view(indirect_view &&) noexcept = default;
  ~indirect_view() = default;

  // Linear access in view order: operator[] is unchecked; at() checks bounds.
  [[nodiscard]] const_reference operator[](size_type i) const {
    return flat_unchecked(i);
  }
  [[nodiscard]] reference operator[](size_type i) {
    return flat_unchecked(i);
  }
  [[nodiscard]] const_reference at(size_type i) const {
    detail::checked_index(i, size(), "indirect_view::at");
    return (*this)[i];
  }
  [[nodiscard]] reference at(size_type i) {
    detail::checked_index(i, size(), "indirect_view::at");
    return (*this)[i];
  }

  // flattened element access using the default_layout order, independent of the actual storage layout.
  [[nodiscard]] const_reference flat(size_type i) const {
    detail::checked_index(i, size(), "indirect_view::flat");
    return flat_unchecked(i);
  }
  [[nodiscard]] reference flat(size_type i) {
    detail::checked_index(i, size(), "indirect_view::flat");
    return flat_unchecked(i);
  }
  // Internal API (not stable): unchecked flat access; requires i < size().
  // Bypass mdspan::operator[] to avoid any potential bounds checking.
  [[nodiscard]] const_reference flat_unchecked(size_type i) const {
    return span_.accessor().offset(span_.data_handle(), i)[0];
  }
  [[nodiscard]] reference flat_unchecked(size_type i) {
    return span_.accessor().offset(span_.data_handle(), i)[0];
  }

  // unary operators:
  auto operator+() const { return apply_unary_op<detail::nc_unary_plus>(); }
  auto operator-() const { return apply_unary_op<std::negate>          (); }
  auto operator~() const { return apply_unary_op<detail::nc_bit_not>   (); }
  auto operator!() const { return apply_unary_op<std::logical_not>     (); }

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

  template <value_expression Expr> indirect_view &operator=  (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a = b  ; }, v); }
  template <value_expression Expr> indirect_view &operator+= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a += b ; }, v); }
  template <value_expression Expr> indirect_view &operator-= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a -= b ; }, v); }
  template <value_expression Expr> indirect_view &operator*= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a *= b ; }, v); }
  template <value_expression Expr> indirect_view &operator/= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a /= b ; }, v); }
  template <value_expression Expr> indirect_view &operator%= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a %= b ; }, v); }
  template <value_expression Expr> indirect_view &operator&= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a &= b ; }, v); }
  template <value_expression Expr> indirect_view &operator|= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a |= b ; }, v); }
  template <value_expression Expr> indirect_view &operator^= (const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a ^= b ; }, v); }
  template <value_expression Expr> indirect_view &operator<<=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a <<= b; }, v); }
  template <value_expression Expr> indirect_view &operator>>=(const Expr& v) { return apply_expr_op([](value_type& a, value_type b) { a >>= b; }, v); }

  // observers
  [[nodiscard]] constexpr              size_type         size() const noexcept { return span_.size()       ; }
  [[nodiscard]] constexpr                   bool        empty() const noexcept { return span_.empty()      ; }
  [[nodiscard]] constexpr const     extents_type&     extents() const noexcept { return span_.extents()    ; }
  [[nodiscard]] constexpr const data_handle_type& data_handle() const noexcept { return span_.data_handle(); }

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
    auto acc = detail::index_accessor<element_type>(std::move(offsets));
    span_ = mdspan_type(base, ext, acc);
  }

  template <template <class> class Op> auto apply_unary_op() const {
    return detail::make_unary_op<Op<value_type>>(*this);
  }

  template <typename Op>
  indirect_view &apply_scalar_op(Op &&op, const value_type &x) {
    for (size_type i = 0; i < size(); ++i)
      op(flat_unchecked(i), x);
    return *this;
  }

  template <class Op, class Expr>
  indirect_view &apply_expr_op(Op &&op, const Expr &expr) {
    detail::check_shape_match(*this, expr);
    for (size_type i = 0; i < size(); ++i)
      op(flat_unchecked(i), expr.flat_unchecked(i));
    return *this;
  }

  template <class, class, class> friend class ndarray;
  template <class, class, class> friend class gslice_view;

  mdspan_type span_;
};

// [numcxx.array_initialization] initialization of ndarrays and slice views from nested initializer_list
namespace detail {
// clang-format off
template <std::size_t N,             typename List> bool                                check_non_jagged(     const List &);
template <std::size_t N,             typename List> std::array<std::size_t, N>            derive_extents(     const List &);
template <std::size_t N, typename I, typename List, std::enable_if_t< N == 1, int> = 0> void add_extents(I &, const List &);
template <std::size_t N, typename I, typename List, std::enable_if_t<(N > 1), int> = 0> void add_extents(I &, const List &);
// clang-format on

template <std::size_t N, typename List>
std::array<std::size_t, N> derive_extents(const List &list) {
  std::array<std::size_t, N> a;
  auto f = a.begin();
  add_extents<N>(f, list); // add sizes (extents) to a
  return a;
}

template <std::size_t N, typename I, typename List,
          std::enable_if_t<N == 1, int>>
void add_extents(I &first, const List &list) {
  *first++ = list.size();
}

template <std::size_t N, typename I, typename List,
          std::enable_if_t<(N > 1), int>>
void add_extents(I &first, const List &list) {
  if (list.size() == 0)
    NUMCXX_THROW(std::invalid_argument, "empty initializer list");
  if (!check_non_jagged<N>(list))
    NUMCXX_THROW(std::invalid_argument, "initializer list is jagged");
  *first++ = list.size(); // store this size (extent)
  add_extents<N - 1>(first, *list.begin());
}

template <std::size_t N, typename List>
bool check_non_jagged(const List &list) {
  auto i = list.begin();
  for (auto j = i + 1; j != list.end(); ++j) {
    if (derive_extents<N - 1>(*i) != derive_extents<N - 1>(*j))
      return false;
  }
  return true;
}

template <typename T, typename Vec>
void add_list(const T *first, const T *last, Vec &vec) {
  vec.insert(vec.end(), first, last);
}

template <typename T, typename Vec>
void add_list(const std::initializer_list<T> *first,
              const std::initializer_list<T> *last, Vec &vec) {
  for (; first != last; ++first)
    add_list(first->begin(), first->end(), vec);
}

template <typename T, typename Vec>
void insert_flat(std::initializer_list<T> list, Vec &vec) {
  add_list(list.begin(), list.end(), vec);
}

template <typename T, typename Iter>
void copy_list(const T *first, const T *last, Iter &iter) {
  iter = std::copy(first, last, iter);
}

template <typename T, typename Iter>
void copy_list(const std::initializer_list<T> *first,
               const std::initializer_list<T> *last, Iter &it) {
  for (; first != last; ++first)
    copy_list(first->begin(), first->end(), it);
}

template <typename T, typename Iter>
void copy_flat(std::initializer_list<T> list, Iter &iter) {
  copy_list(list.begin(), list.end(), iter);
}
} // namespace detail

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>::ndarray(
    detail::nested_initializer_list_t<element_type, extents_type::rank()>
        list) {
  static_assert(std::is_same_v<layout_type, layout_right>,
                "Nested initializer list is only supported for row-major "
                "(layout_right) arrays.");
  if (list.size() == 0)
    NUMCXX_THROW(std::invalid_argument, "empty initializer list not allowed");

  constexpr std::size_t Rank = extents_type::rank();
  auto derived = detail::derive_extents<Rank>(list);

  if constexpr (detail::static_extents<Ex>) {
    // static extents: validate
    Ex expected;
    for (rank_type i = 0; i < Rank; ++i) {
      if (derived[i] != expected.extent(i))
        NUMCXX_THROW(std::invalid_argument,
                     "initializer list shape does not match static extents");
    }

    elem_ = mdarray_type{}; // already has correct size
  } else {
    // dynamic extents: use derived shape
    elem_ = mdarray_type(derived);
  }

  // flatten data
  auto it = elem_.data();
  detail::copy_flat(list, it);
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp> &ndarray<Tp, Ex, Lp>::operator=(
    detail::nested_initializer_list_t<value_type, extents_type::rank()> list) {
  static_assert(std::is_same_v<layout_type, layout_right>,
                "Nested initializer list is only supported for row-major "
                "(layout_right) arrays.");
  if (list.size() == 0)
    NUMCXX_THROW(std::invalid_argument, "empty initializer list not allowed");

  constexpr std::size_t Rank = extents_type::rank();
  auto derived = detail::derive_extents<Rank>(list);

  for (rank_type i = 0; i < Rank; ++i) {
    if (derived[i] != extent(i))
      NUMCXX_THROW(std::invalid_argument,
                   "initializer list shape does not match array extents");
  }

  auto it = data();
  detail::copy_flat(list, it);
  return *this;
}

// [numcxx.ndarray_construction] array creation: factories and random

namespace detail {

template <std::size_t Rank>
std::array<size_type, Rank> make_shape(std::initializer_list<size_type> shape) {
  if (shape.size() != Rank)
    NUMCXX_THROW(std::invalid_argument, "shape size does not match array rank");

  std::array<size_type, Rank> dims{};
  std::copy(shape.begin(), shape.end(), dims.begin());
  return dims;
}

template <typename Array>
Array make_dynamic_array(std::initializer_list<size_type> shape) {
  constexpr auto rank = Array::extents_type::rank();
  NUMCXX_ASSERT(shape.size() == rank, "shape must match array rank");

  auto dims = make_shape<rank>(shape);
  return Array(typename Array::extents_type(dims));
}

} // namespace detail

// -----------------------------
// array creation (factories)
// -----------------------------
template <typename T>
ndarray<T, dextents<1>> arange(T start, T stop, T step = T(1)) {
  static_assert(std::is_arithmetic_v<T>,
                "arange requires an arithmetic value type");

  const size_type n = std::ceil((stop - start) / step);

  ndarray<T, dextents<1>> arr(n);
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

template <detail::static_array Array>
Array ones() {
  Array arr{};
  std::fill_n(arr.data(), arr.size(), typename Array::value_type(1));
  return arr;
}

template <detail::dynamic_array Array>
Array ones(std::initializer_list<size_type> shape) {
  Array arr = detail::make_dynamic_array<Array>(shape);
  std::fill_n(arr.data(), arr.size(), typename Array::value_type(1));
  return arr;
}

template <detail::static_array Array>
Array zeros() {
  Array arr{};
  std::fill_n(arr.data(), arr.size(), typename Array::value_type(0));
  return arr;
}

template <detail::dynamic_array Array>
Array zeros(std::initializer_list<size_type> shape) {
  Array arr = detail::make_dynamic_array<Array>(shape);
  std::fill_n(arr.data(), arr.size(), typename Array::value_type(0));
  return arr;
}

// -----------------------------
// random number generation
// -----------------------------
namespace random {
inline std::mt19937 &get_engine() {
  static thread_local std::mt19937 engine(std::random_device{}());
  return engine;
}

inline void seed(unsigned int value) { get_engine().seed(value); }

namespace detail {
using ::numcxx::detail::static_array;
using ::numcxx::detail::dynamic_array;
using ::numcxx::detail::make_dynamic_array;

template <typename Array, typename Distribution>
void fill_random(Array &arr, Distribution &&dist) {
  auto &engine = get_engine();
  auto *data = arr.data();
  const size_type n = arr.size();
  for (size_type i = 0; i < n; ++i)
    data[i] = dist(engine);
}
} // namespace detail

/// Generates an array of random numbers uniformly distributed in [0,1)
///
/// @tparam Array A floating-point array type (e.g., dvec, dmat).
/// @param shape (for dynamic arrays) The shape as a braced list, e.g., {3,4}.
///      Not used for static arrays (which have fixed shape).
/// @returns An array of random numbers.
///
/// @note For static arrays (fixed extents), use the overload without `shape`.
///       For dynamic arrays (dextents), use the overload with `shape`.
template <detail::static_array Array>
Array rand() {
  using T = typename Array::value_type;
  static_assert(
      std::is_floating_point_v<T>,
      "rand() requires floating-point type (use randint() for integers)");
  Array arr{};
  std::uniform_real_distribution<T> dist(T(0.0), T(1.0));
  detail::fill_random(arr, dist);
  return arr;
}

/// @copydoc rand()
template <detail::dynamic_array Array>
Array rand(std::initializer_list<size_type> shape) {
  using T = typename Array::value_type;
  static_assert(
      std::is_floating_point_v<T>,
      "rand() requires floating-point type (use randint() for integers)");
  Array arr = detail::make_dynamic_array<Array>(shape);
  std::uniform_real_distribution<T> dist(T(0.0), T(1.0));
  detail::fill_random(arr, dist);
  return arr;
}

/// Generates an array of random numbers from the standard normal distribution.
///
/// @tparam Array A floating-point array type (e.g., dvec, dmat).
/// @param shape (for dynamic arrays) The shape as a braced list, e.g., {3,4}.
///      Not used for static arrays (which have fixed shape).
/// @returns An array of random numbers sampled from N(0, 1).
///
/// @note For static arrays (fixed extents), use the overload without `shape`.
///       For dynamic arrays (dextents), use the overload with `shape`.
template <detail::static_array Array>
Array randn() {
  using T = typename Array::value_type;
  static_assert(std::is_floating_point_v<T>,
                "randn() requires floating-point type");
  Array arr{};
  std::normal_distribution<T> dist(T(0.0), T(1.0));
  detail::fill_random(arr, dist);
  return arr;
}

/// @copydoc randn()
template <detail::dynamic_array Array>
Array randn(std::initializer_list<size_type> shape) {
  using T = typename Array::value_type;
  static_assert(std::is_floating_point_v<T>,
                "randn() requires floating-point type");
  Array arr = detail::make_dynamic_array<Array>(shape);
  std::normal_distribution<T> dist(T(0.0), T(1.0));
  detail::fill_random(arr, dist);
  return arr;
}

/// Generates an array of random numbers uniformly distributed in [low, high).
///
/// @tparam Array An array type (e.g., dvec, dmat) with floating-point value type.
/// @param low  Lower bound of the distribution (inclusive).
/// @param high Upper bound of the distribution (exclusive).
/// @param shape (for dynamic arrays) The shape as a braced list, e.g., {3,4}.
///      Not used for static arrays (which have fixed shape).
/// @returns An array of random numbers.
///
/// @note For static arrays (fixed extents), use the overload without `shape`.
///       For dynamic arrays (dextents), use the overload with `shape`.
/// @note Requires `low < high`.
template <detail::static_array Array>
Array uniform(typename Array::value_type low, typename Array::value_type high) {
  using T = typename Array::value_type;
  static_assert(std::is_floating_point_v<T>,
                "uniform() requires floating-point type");
  Array arr{};
  std::uniform_real_distribution<T> dist(low, high);
  detail::fill_random(arr, dist);
  return arr;
}

/// @copydoc uniform()
template <detail::dynamic_array Array>
Array uniform(typename Array::value_type low, typename Array::value_type high,
              std::initializer_list<size_type> shape) {
  using T = typename Array::value_type;
  static_assert(std::is_floating_point_v<T>,
                "uniform() requires floating-point type");
  Array arr = detail::make_dynamic_array<Array>(shape);
  std::uniform_real_distribution<T> dist(low, high);
  detail::fill_random(arr, dist);
  return arr;
}

/// Generates an array of random integers uniformly distributed in [low, high).
///
/// @tparam Array An array type (e.g., vec, mat) with integral value type.
/// @param low  Lower bound of the distribution (inclusive).
/// @param high Upper bound of the distribution (exclusive).
/// @param shape (for dynamic arrays) The shape as a braced list, e.g., {3,4}.
///      Not used for static arrays (which have fixed shape).
/// @returns An array of random integers.
///
/// @note For static arrays (fixed extents), use the overload without `shape`.
///       For dynamic arrays (dextents), use the overload with `shape`.
/// @note Requires `low < high`. The value `high` itself is never generated.
template <detail::static_array Array>
Array randint(typename Array::value_type low, typename Array::value_type high) {
  using T = typename Array::value_type;
  static_assert(std::is_integral_v<T>, "randint() requires integral type");
  if (low >= high)
    NUMCXX_THROW(
        std::invalid_argument,
        "randint: low must be less than high (empty range not supported)");
  Array arr{};
  std::uniform_int_distribution<T> dist(low, high - T(1));
  detail::fill_random(arr, dist);
  return arr;
}

/// @copydoc randint()
template <detail::dynamic_array Array>
Array randint(typename Array::value_type low, typename Array::value_type high,
              std::initializer_list<size_type> shape) {
  using T = typename Array::value_type;
  static_assert(std::is_integral_v<T>, "randint() requires integral type");
  if (low >= high)
    NUMCXX_THROW(
        std::invalid_argument,
        "randint: low must be less than high (empty range not supported)");
  Array arr = detail::make_dynamic_array<Array>(shape);
  std::uniform_int_distribution<T> dist(low, high - T(1));
  detail::fill_random(arr, dist);
  return arr;
}

} // namespace random

// [numcxx.indexing_and_slicing]

// Precondition: is_valid() == true.
constexpr auto slice::normalize(size_type dim_len) const noexcept
    -> normalized_slice {
  const index_type len = static_cast<index_type>(dim_len);
  const bool reversed = step_ < 0;

  index_type start = 0, stop = 0;
  size_type offset = 0, extent = 0;

  // Overflow-safe constexpr absolute value for the reversed case.
  // std::abs is not constexpr for integral types before C++23.
  const size_type stride =
      static_cast<size_type>(reversed ? -(step_ + 1) + 1 : step_);

  if (len == 0)
    return {0, 0, stride, reversed};

  if (!reversed) {
    // Resolve the start index using NumPy forward-slice semantics.
    start = start_.value_or(0);
    if (start < 0)
      start += len;
    start = std::clamp(start, index_type(0), len);

    // Resolve the stop index using NumPy forward-slice semantics.
    stop = stop_.value_or(len);
    if (stop < 0)
      stop += len;
    stop = std::clamp(stop, index_type(0), len);

    // Convert [start, stop) to a forward strided_slice.
    bool empty_slice = start >= stop;
    offset = static_cast<size_type>(start);
    extent = static_cast<size_type>(empty_slice ? 0 : (stop - start));
  } else {
    // Resolve the start index using NumPy reverse-slice semantics.
    start = start_.value_or(len - 1);
    if (start < 0)
      start += len;
    start = std::clamp(start, index_type(0), len - 1);

    // Resolve the stop index.
    //
    // For reverse slices, an omitted stop uses the NumPy sentinel
    // value -1 ("one position before the beginning") and must be
    // preserved. A user-supplied negative stop is treated as a
    // regular negative index and converted relative to the end.
    if (stop_.has_value()) {
      stop = stop_.value();
      if (stop < 0)
        stop += len;
      stop = std::clamp(stop, index_type(-1), len - 1);
    } else {
      stop = -1;
    }

    // NumPy reverse slices are empty when start <= stop.
    bool empty_slice = start <= stop;
    if (empty_slice)
      offset = static_cast<size_type>(start);
    else {
      // Compute the last index visited by the reverse slice.
      // This becomes the offset of the equivalent forward
      // strided_slice; the original ordering is restored later
      // using the reversed flag.
      const index_type n = static_cast<index_type>((start - stop - 1) / stride);
      offset = static_cast<size_type>(start + n * step_);
    }

    extent = static_cast<size_type>(empty_slice ? 0 : (start - stop));
  }

  return {offset, extent, stride, reversed};
}

namespace detail {
namespace index_utils {
inline size_type normalize_index(index_type idx, size_type dim_len) {
  index_type len = static_cast<index_type>(dim_len);
  index_type res = (idx < 0) ? idx + len : idx;
  // Empty dimensions are valid for slicing but cannot be scalar-indexed.
  NUMCXX_ASSERT(res >= 0 && res < len, "Index out of bounds");
  return static_cast<size_type>(res);
}

inline size_type normalize_and_reverse_index(index_type idx, size_type dim_len,
                                             bool reversed) {
  const size_type n = normalize_index(idx, dim_len);
  return reversed ? dim_len - 1 - n : n;
}

// Element access with no reversed source dimensions.
template <class MdSpan, class... Idxs>
  requires all_scalar_indices<Idxs...>
[[nodiscard]] decltype(auto) get_element(MdSpan &&src, Idxs &&...idxs) {
  auto helper =
      [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
    const std::array<size_type, sizeof...(Idxs)> dims = {
        static_cast<size_type>(src.extent(Is))...};
    return std::forward<MdSpan>(src)(
        normalize_index(static_cast<index_type>(idxs), dims[Is])...);
  };

  return helper(std::make_index_sequence<sizeof...(Idxs)>{});
}

// Element access with reversed source dimensions.
template <class MdSpan, class... Idxs>
  requires all_scalar_indices<Idxs...>
[[nodiscard]] decltype(auto)
get_element(MdSpan &&src, const std::array<bool, sizeof...(Idxs)> &reverse_dims,
            Idxs &&...idxs) {
  auto helper =
      [&]<std::size_t... Is>(std::index_sequence<Is...>) -> decltype(auto) {
    const std::array<size_type, sizeof...(Idxs)> dims = {
        static_cast<size_type>(src.extent(Is))...};
    return std::forward<MdSpan>(src)(normalize_and_reverse_index(
        static_cast<index_type>(idxs), dims[Is], reverse_dims[Is])...);
  };

  return helper(std::make_index_sequence<sizeof...(Idxs)>{});
}
} // namespace index_utils

namespace slice_utils {

// Convert a scalar index to a submdspan index argument.
inline size_type to_submdspan_arg(index_type idx, size_type dim_len,
                                  bool &is_index, bool &reversed,
                                  bool src_reversed = false) {
  is_index = true;
  reversed = false;
  return index_utils::normalize_and_reverse_index(idx, dim_len, src_reversed);
}

// Convert a slice to a submdspan strided_slice argument.
inline auto to_submdspan_arg(const slice &s, size_type dim_len, bool &is_index,
                             bool &reversed, bool src_reversed = false) {
  NUMCXX_ASSERT(s.is_valid(), "invalid slice");
  const auto ns = s.normalize(dim_len);
  is_index = false;
  // std::strided_slice only represents forward slices.
  // The resulting dimension is reversed iff exactly one of the
  // normalized slice or source dimension is reversed.
  reversed = ns.reversed != src_reversed;
  size_type offset = ns.offset;
  if (src_reversed && ns.extent != 0) {
    // Reflect the last selected index into the underlying forward
    // coordinate system. This becomes the offset of the equivalent
    // forward std::strided_slice.
    const size_type steps = (ns.extent - 1) / ns.stride;
    const size_type last = ns.offset + steps * ns.stride;
    offset = (dim_len - 1) - last;
  }
  return ::numcxx::detail::strided_slice<size_type, size_type, size_type>{
      offset, ns.extent, ns.stride};
}

// Forward std::full_extent for std::submdspan compatibility.
inline auto to_submdspan_arg(::numcxx::detail::full_extent_t /*tag*/,
                             size_type /*dim_len*/, bool &is_index,
                             bool &reversed, bool src_reversed = false) {
  is_index = false;
  reversed = src_reversed;
  return ::numcxx::detail::full_extent;
}

// Forward a std::pair range [first, second) for std::submdspan compatibility.
// The range is interpreted in the logical coordinate system of the current view.
template <typename I, typename J>
  requires std::integral<I> && std::integral<J>
auto to_submdspan_arg(const std::pair<I, J> &p, size_type dim_len,
                      bool &is_index, bool &reversed,
                      bool src_reversed = false) {
  if constexpr (std::signed_integral<I>)
    NUMCXX_ASSERT(p.first >= 0, "Range start must be non-negative");
  if constexpr (std::signed_integral<J>)
    NUMCXX_ASSERT(p.second >= 0, "Range end must be non-negative");

  const size_type first = static_cast<size_type>(p.first);
  const size_type second = static_cast<size_type>(p.second);

  NUMCXX_ASSERT(first <= second, "Range start must not exceed range end");
  NUMCXX_ASSERT(second <= dim_len, "Range out of bounds");

  is_index = false;
  reversed = src_reversed;

  return src_reversed ? std::pair<size_type, size_type>{dim_len - second,
                                                        dim_len - first}
                      : std::pair<size_type, size_type>{first, second};
}

template <typename MdSpan, std::size_t N, typename... Args>
[[nodiscard]]
auto make_submdspan_impl(MdSpan &&src,
                         const std::array<bool, N> &src_reverse_dims,
                         Args &&...args) {
  static_assert(N == sizeof...(Args),
                "src_reverse_dims must have one entry per argument");

  std::array<bool, N> is_index_arr{};
  std::array<bool, N> reversed_arr{};

  auto helper = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
    const std::array<size_type, N> dims = {
        static_cast<size_type>(src.extent(Is))...};
    return detail::submdspan(std::forward<MdSpan>(src),
                             to_submdspan_arg(args, dims[Is], is_index_arr[Is],
                                              reversed_arr[Is],
                                              src_reverse_dims[Is])...);
  };
  auto sub = helper(std::index_sequence_for<Args...>{});
  return std::tuple{sub, is_index_arr, reversed_arr};
}

template <std::size_t NewRank, std::size_t N>
[[nodiscard]]
auto make_reverse_dims(const std::array<bool, N> &is_index_arr,
                       const std::array<bool, N> &reversed_arr) {
  static_assert(NewRank <= N, "Result rank cannot exceed source rank");
  std::array<bool, NewRank> reverse_dims{};
  std::size_t rev_idx = 0;
  // Scalar indices remove dimensions. Keep reversal flags only for
  // dimensions that remain in the resulting submdspan.
  for (std::size_t i = 0; i < N; ++i) {
    if (!is_index_arr[i]) {
      reverse_dims[rev_idx++] = reversed_arr[i];
    }
  }

  return reverse_dims;
}

template <typename MdSpan, std::size_t N, typename... Args>
[[nodiscard]]
auto make_gslice_view_impl(MdSpan &&src,
                          const std::array<bool, N> &src_reverse_dims,
                          Args &&...args) {
  auto [sub_mdspan, is_index_arr, reversed_arr] = make_submdspan_impl(
      std::forward<MdSpan>(src), src_reverse_dims, std::forward<Args>(args)...);
  using sub_mdspan_type = std::decay_t<decltype(sub_mdspan)>;
  constexpr std::size_t new_rank = sub_mdspan_type::rank();
  const auto reverse_dims =
      make_reverse_dims<new_rank>(is_index_arr, reversed_arr);
  return gslice_view<typename sub_mdspan_type::element_type,
                    typename sub_mdspan_type::extents_type,
                    typename sub_mdspan_type::layout_type>(sub_mdspan,
                                                           reverse_dims);
}

template <typename MdSpan, typename... Args>
[[nodiscard]]
auto make_gslice_view(MdSpan &&src, Args &&...args) {
  // ndarray (or gslice_view with has_reversed_dim_ == false)
  // has no inherited reversed dimensions.
  const std::array<bool, sizeof...(Args)> no_reverse_dims{};
  return make_gslice_view_impl(std::forward<MdSpan>(src), no_reverse_dims,
                              std::forward<Args>(args)...);
}

template <typename MdSpan, std::size_t N, typename... Args>
[[nodiscard]]
auto make_gslice_view_with_reverse_dims(
    MdSpan &&src, const std::array<bool, N> &src_reverse_dims, Args &&...args) {
  return make_gslice_view_impl(std::forward<MdSpan>(src), src_reverse_dims,
                              std::forward<Args>(args)...);
}

template <std::size_t Rank, typename MdSpan, typename... Args>
[[nodiscard]]
auto make_gslice_view_padded(MdSpan &&src, Args &&...args) {
  if constexpr (sizeof...(Args) > Rank) {
    static_assert(sizeof...(Args) <= Rank,
                  "Number of arguments cannot exceed array rank");
  } else {
    auto padded_args = std::tuple_cat(
        std::forward_as_tuple(std::forward<Args>(args)...),
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::tuple{((void)Is, ::numcxx::detail::full_extent)...};
        }(std::make_index_sequence<Rank - sizeof...(Args)>{}));
    return std::apply(
        [&src](auto &&...indices) {
          return make_gslice_view(std::forward<MdSpan>(src),
                                 std::forward<decltype(indices)>(indices)...);
        },
        std::move(padded_args));
  }
}

template <std::size_t Rank, typename MdSpan, std::size_t N, typename... Args>
[[nodiscard]]
auto make_gslice_view_padded_with_reverse_dims(
    MdSpan &&src, const std::array<bool, N> &src_reverse_dims,
    Args &&...args) {
  static_assert(N == Rank, "src_reverse_dims must match array rank");
  if constexpr (sizeof...(Args) > Rank) {
    static_assert(sizeof...(Args) <= Rank,
                  "Number of arguments cannot exceed array rank");
  } else {
    auto padded_args = std::tuple_cat(
        std::forward_as_tuple(std::forward<Args>(args)...),
        [&]<std::size_t... Is>(std::index_sequence<Is...>) {
          return std::tuple{((void)Is, ::numcxx::detail::full_extent)...};
        }(std::make_index_sequence<Rank - sizeof...(Args)>{}));
    return std::apply(
        [&src, &src_reverse_dims](auto &&...indices) {
          return make_gslice_view_with_reverse_dims(
              std::forward<MdSpan>(src), src_reverse_dims,
              std::forward<decltype(indices)>(indices)...);
        },
        std::move(padded_args));
  }
}

} // namespace slice_utils
} // namespace detail

template <typename Tp, typename Extents, typename Lp>
template <typename... Idxs>
  requires elem_access_arguments<Extents::rank(), Idxs...>
[[nodiscard]]
auto ndarray<Tp, Extents, Lp>::operator()(Idxs &&...idxs) const
    -> const_reference {
  return detail::index_utils::get_element(to_mdspan(),
                                          std::forward<Idxs>(idxs)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Idxs>
  requires elem_access_arguments<Extents::rank(), Idxs...>
[[nodiscard]]
auto ndarray<Tp, Extents, Lp>::operator()(Idxs &&...idxs) -> reference {
  return detail::index_utils::get_element(to_mdspan(),
                                          std::forward<Idxs>(idxs)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Idxs>
  requires elem_access_arguments<Extents::rank(), Idxs...>
[[nodiscard]]
auto gslice_view<Tp, Extents, Lp>::operator()(Idxs &&...idxs) const
    -> const_reference {
  return has_reversed_dim_
             ? detail::index_utils::get_element(to_mdspan(), reverse_dims_,
                                                std::forward<Idxs>(idxs)...)
             : detail::index_utils::get_element(to_mdspan(),
                                                std::forward<Idxs>(idxs)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Idxs>
  requires elem_access_arguments<Extents::rank(), Idxs...>
[[nodiscard]]
auto gslice_view<Tp, Extents, Lp>::operator()(Idxs &&...idxs) -> reference {
  return has_reversed_dim_
             ? detail::index_utils::get_element(to_mdspan(), reverse_dims_,
                                                std::forward<Idxs>(idxs)...)
             : detail::index_utils::get_element(to_mdspan(),
                                                std::forward<Idxs>(idxs)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Args>
  requires gslice_view_arguments<Extents::rank(), Args...>
[[nodiscard]] auto ndarray<Tp, Extents, Lp>::operator()(Args &&...args) const {
  return detail::slice_utils::make_gslice_view_padded<Extents::rank()>(
      to_mdspan(), std::forward<Args>(args)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Args>
  requires gslice_view_arguments<Extents::rank(), Args...>
[[nodiscard]] auto ndarray<Tp, Extents, Lp>::operator()(Args &&...args) {
  return detail::slice_utils::make_gslice_view_padded<Extents::rank()>(
      to_mdspan(), std::forward<Args>(args)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Args>
  requires gslice_view_arguments<Extents::rank(), Args...>
[[nodiscard]] auto gslice_view<Tp, Extents, Lp>::operator()(Args &&...args) const {
  return has_reversed_dim_
             ? detail::slice_utils::make_gslice_view_padded_with_reverse_dims<
                   Extents::rank()>(to_mdspan(), reverse_dims_,
                               std::forward<Args>(args)...)
             : detail::slice_utils::make_gslice_view_padded<Extents::rank()>(
                   to_mdspan(), std::forward<Args>(args)...);
}

template <typename Tp, typename Extents, typename Lp>
template <typename... Args>
  requires gslice_view_arguments<Extents::rank(), Args...>
[[nodiscard]] auto gslice_view<Tp, Extents, Lp>::operator()(Args &&...args) {
  return has_reversed_dim_
             ? detail::slice_utils::make_gslice_view_padded_with_reverse_dims<
                   Extents::rank()>(to_mdspan(), reverse_dims_,
                               std::forward<Args>(args)...)
             : detail::slice_utils::make_gslice_view_padded<Extents::rank()>(
                   to_mdspan(), std::forward<Args>(args)...);
}

template <typename Layout, typename Extents>
std::array<size_type, Extents::rank()>
unravel_index(size_type i, const Extents &extents) {
  constexpr std::size_t rank = Extents::rank();
  std::array<size_type, rank> idx{};

  if constexpr (std::is_same_v<Layout, layout_right>) {
    // row-major ('C')
    for (std::size_t r = rank; r-- > 0;) {
      idx[r] = i % extents.extent(r);
      i /= extents.extent(r);
    }
  } else {
    // column-major ('F')
    for (std::size_t r = 0; r < rank; ++r) {
      idx[r] = i % extents.extent(r);
      i /= extents.extent(r);
    }
  }

  return idx;
}

// [numcxx.reduction]
template <class ElementType, class Extents, class LayoutPolicy>
template <value_expression Expr>
  requires (!std::same_as<std::remove_cvref_t<Expr>,
                          ndarray<ElementType, Extents, LayoutPolicy>> &&
            !nc_is_broadcast_view_v<Expr> &&
            (std::remove_cvref_t<Expr>::rank() == Extents::rank()))
ndarray<ElementType, Extents, LayoutPolicy>::ndarray(const Expr &expr)
    : elem_(make_expression_storage(
          expr, std::make_index_sequence<extents_type::rank()>{})) {
  for (size_type i = 0; i < size(); ++i)
    flat_unchecked(i) = static_cast<value_type>(expr.flat_unchecked(i));
}

template <typename Tp, typename Ex, typename Lp>
detail::reduction_mean_type_t<typename ndarray<Tp, Ex, Lp>::value_type>
ndarray<Tp, Ex, Lp>::mean() const {
  using result_type = detail::reduction_mean_type_t<value_type>;
  if (size() == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::mean(): cannot reduce an empty array");
  result_type total{};
  for (size_type i = 0; i < size(); ++i)
    total += static_cast<result_type>(flat_unchecked(i));
  return total / static_cast<result_type>(size());
}

template <typename Tp, typename Ex, typename Lp>
detail::reduction_real_type_t<typename ndarray<Tp, Ex, Lp>::value_type>
ndarray<Tp, Ex, Lp>::var() const {
  using mean_type = detail::reduction_mean_type_t<value_type>;
  using result_type = detail::reduction_real_type_t<value_type>;
  if (size() == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::var(): cannot reduce an empty array");
  const mean_type average = mean();
  result_type total{};
  for (size_type i = 0; i < size(); ++i) {
    const mean_type delta =
        static_cast<mean_type>(flat_unchecked(i)) - average;
    const result_type magnitude =
        static_cast<result_type>(std::abs(delta));
    total += magnitude * magnitude;
  }
  return total / static_cast<result_type>(size());
}

template <typename Tp, typename Ex, typename Lp>
detail::reduction_real_type_t<typename ndarray<Tp, Ex, Lp>::value_type>
ndarray<Tp, Ex, Lp>::std() const {
  if (size() == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::std(): cannot reduce an empty array");
  return std::sqrt(var());
}

template <typename Tp, typename Ex, typename Lp>
bool ndarray<Tp, Ex, Lp>::any() const {
  for (size_type i = 0; i < size(); ++i) {
    const value_type &value = flat_unchecked(i);
    if constexpr (requires { value.real(); value.imag(); }) {
      if (value.real() != 0 || value.imag() != 0)
        return true;
    } else if (static_cast<bool>(value)) {
      return true;
    }
  }
  return false;
}

template <typename Tp, typename Ex, typename Lp>
bool ndarray<Tp, Ex, Lp>::all() const {
  for (size_type i = 0; i < size(); ++i) {
    const value_type &value = flat_unchecked(i);
    if constexpr (requires { value.real(); value.imag(); }) {
      if (value.real() == 0 && value.imag() == 0)
        return false;
    } else if (!static_cast<bool>(value)) {
      return false;
    }
  }
  return true;
}

template <typename Tp, typename Ex, typename Lp>
typename ndarray<Tp, Ex, Lp>::size_type ndarray<Tp, Ex, Lp>::argmin() const {
  if (size() == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::argmin(): cannot reduce an empty array");
  size_type result = 0;
  for (size_type i = 1; i < size(); ++i)
    if (detail::reduction_less(flat_unchecked(i), flat_unchecked(result)))
      result = i;
  return result;
}

template <typename Tp, typename Ex, typename Lp>
typename ndarray<Tp, Ex, Lp>::size_type ndarray<Tp, Ex, Lp>::argmax() const {
  if (size() == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::argmax(): cannot reduce an empty array");
  size_type result = 0;
  for (size_type i = 1; i < size(); ++i)
    if (detail::reduction_less(flat_unchecked(result), flat_unchecked(i)))
      result = i;
  return result;
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::sum(::numcxx::index_type axis) const
{
  return reduce_axis_impl(axis, "ndarray::sum(axis)", value_type{},
                          [](const value_type &x, size_type, size_type) {
                            return x;
                          },
                          std::plus<>{});
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::min(::numcxx::index_type axis) const
{
  return reduce_axis_impl(
      axis, "ndarray::min(axis)", value_type{},
      [](const value_type &x, size_type, size_type) { return x; },
      [](const value_type &a, const value_type &b) {
        return detail::reduction_less(b, a) ? b : a;
      },
      true);
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::max(::numcxx::index_type axis) const
{
  return reduce_axis_impl(
      axis, "ndarray::max(axis)", value_type{},
      [](const value_type &x, size_type, size_type) { return x; },
      [](const value_type &a, const value_type &b) {
        return detail::reduction_less(a, b) ? b : a;
      },
      true);
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::prod(::numcxx::index_type axis) const
{
  return reduce_axis_impl(axis, "ndarray::prod(axis)", value_type{1},
                          [](const value_type &x, size_type, size_type) {
                            return x;
                          },
                          std::multiplies<>{});
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::mean(::numcxx::index_type axis) const
{
  using result_type = detail::reduction_mean_type_t<value_type>;
  axis = normalize_reduction_axis(axis, "ndarray::mean(axis)");
  const size_type count = static_cast<size_type>(extent(axis));
  if (count == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::mean(axis): cannot reduce over empty axis");
  auto result = reduce_axis_impl(
      axis, "ndarray::mean(axis)", result_type{},
      [](const value_type &x, size_type, size_type) {
        return static_cast<result_type>(x);
      },
      std::plus<>{});
  const result_type divisor = static_cast<result_type>(count);
  for (size_type i = 0; i < result.size(); ++i)
    result.flat_unchecked(i) /= divisor;
  return result;
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::var(::numcxx::index_type axis) const
{
  using mean_type = detail::reduction_mean_type_t<value_type>;
  using result_type = detail::reduction_real_type_t<value_type>;
  axis = normalize_reduction_axis(axis, "ndarray::var(axis)");
  const size_type count = static_cast<size_type>(extent(axis));
  if (count == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::var(axis): cannot reduce over empty axis");
  const auto averages = mean(axis);
  auto result = reduce_axis_impl(
      axis, "ndarray::var(axis)", result_type{},
      [&](const value_type &x, size_type output_flat, size_type) {
        const mean_type delta =
            static_cast<mean_type>(x) - averages.flat_unchecked(output_flat);
        const result_type magnitude =
            static_cast<result_type>(std::abs(delta));
        return magnitude * magnitude;
      },
      std::plus<>{});
  const result_type divisor = static_cast<result_type>(count);
  for (size_type i = 0; i < result.size(); ++i)
    result.flat_unchecked(i) /= divisor;
  return result;
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::std(::numcxx::index_type axis) const
{
  axis = normalize_reduction_axis(axis, "ndarray::std(axis)");
  if (extent(axis) == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::std(axis): cannot reduce over empty axis");
  auto result = var(axis);
  for (size_type i = 0; i < result.size(); ++i)
    result.flat_unchecked(i) = std::sqrt(result.flat_unchecked(i));
  return result;
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::any(::numcxx::index_type axis) const {
  return reduce_axis_impl(
      axis, "ndarray::any(axis)", static_cast<unsigned char>(0),
      [](const value_type &x, size_type, size_type) {
        if constexpr (requires { x.real(); x.imag(); })
          return static_cast<unsigned char>(x.real() != 0 || x.imag() != 0);
        else
          return static_cast<unsigned char>(static_cast<bool>(x));
      },
      [](unsigned char a, unsigned char b) {
        return static_cast<unsigned char>(a || b);
      });
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::all(::numcxx::index_type axis) const {
  return reduce_axis_impl(
      axis, "ndarray::all(axis)", static_cast<unsigned char>(1),
      [](const value_type &x, size_type, size_type) {
        if constexpr (requires { x.real(); x.imag(); })
          return static_cast<unsigned char>(x.real() != 0 || x.imag() != 0);
        else
          return static_cast<unsigned char>(static_cast<bool>(x));
      },
      [](unsigned char a, unsigned char b) {
        return static_cast<unsigned char>(a && b);
      });
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::argmin(::numcxx::index_type axis) const
{
  using item_type = std::pair<value_type, size_type>;
  axis = normalize_reduction_axis(axis, "ndarray::argmin(axis)");
  if (extent(axis) == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::argmin(axis): cannot reduce over empty axis");
  auto pairs = reduce_axis_impl(
      axis, "ndarray::argmin(axis)", item_type{},
      [](const value_type &x, size_type, size_type reduced) {
        return item_type{x, reduced};
      },
      [](const item_type &a, const item_type &b) {
        return detail::reduction_less(b.first, a.first) ? b : a;
      },
      true);
  ndarray<size_type, typename decltype(pairs)::extents_type> result(
      pairs.extents());
  for (size_type i = 0; i < result.size(); ++i)
    result.flat_unchecked(i) = pairs.flat_unchecked(i).second;
  return result;
}

template <typename Tp, typename Ex, typename Lp>
auto ndarray<Tp, Ex, Lp>::argmax(::numcxx::index_type axis) const
{
  using item_type = std::pair<value_type, size_type>;
  axis = normalize_reduction_axis(axis, "ndarray::argmax(axis)");
  if (extent(axis) == 0)
    NUMCXX_THROW(std::runtime_error,
                 "ndarray::argmax(axis): cannot reduce over empty axis");
  auto pairs = reduce_axis_impl(
      axis, "ndarray::argmax(axis)", item_type{},
      [](const value_type &x, size_type, size_type reduced) {
        return item_type{x, reduced};
      },
      [](const item_type &a, const item_type &b) {
        return detail::reduction_less(a.first, b.first) ? b : a;
      },
      true);
  ndarray<size_type, typename decltype(pairs)::extents_type> result(
      pairs.extents());
  for (size_type i = 0; i < result.size(); ++i)
    result.flat_unchecked(i) = pairs.flat_unchecked(i).second;
  return result;
}

template <typename Tp, typename Ex, typename Lp>
typename ndarray<Tp, Ex, Lp>::const_reference
ndarray<Tp, Ex, Lp>::flat_unchecked(size_type i) const {
  if constexpr (std::is_same_v<layout_type, default_layout>)
    return data()[i];
  else {
    auto idx = unravel_index<default_layout>(i, extents());

    const auto &mapping = elem_.mapping();
    return std::apply(
        [&](auto... idxs) -> decltype(auto) {
          return data()[mapping(idxs...)];
        },
        idx);
  }
}

template <typename Tp, typename Ex, typename Lp>
typename ndarray<Tp, Ex, Lp>::reference
ndarray<Tp, Ex, Lp>::flat_unchecked(size_type i) {
  if constexpr (std::is_same_v<layout_type, default_layout>)
    return data()[i];
  else {
    auto idx = unravel_index<default_layout>(i, extents());

    const auto &mapping = elem_.mapping();
    return std::apply(
        [&](auto... idxs) -> decltype(auto) {
          return data()[mapping(idxs...)];
        },
        idx);
  }
}

template <typename Tp, typename Ex, typename Lp>
typename gslice_view<Tp, Ex, Lp>::const_reference
gslice_view<Tp, Ex, Lp>::flat_unchecked(size_type i) const {
  return flat_contiguous_ ? data_handle()[i]
                          : data_handle()[calc_flat_offset(i)];
}

template <typename Tp, typename Ex, typename Lp>
typename gslice_view<Tp, Ex, Lp>::reference
gslice_view<Tp, Ex, Lp>::flat_unchecked(size_type i) {
  return flat_contiguous_ ? data_handle()[i]
                          : data_handle()[calc_flat_offset(i)];
}

template <typename Tp, typename Ex, typename Lp>
auto gslice_view<Tp, Ex, Lp>::eval() const {
  using result_type = ndarray<value_type, dextents<rank()>, default_layout>;
  auto result = [&]<size_type... Is>(std::index_sequence<Is...>) {
    return result_type(extent(Is)...);
  }(std::make_index_sequence<rank()>{});
  for (size_type i = 0; i < size(); ++i)
    result.flat_unchecked(i) = flat_unchecked(i);
  return result;
}

template <class Tp, class Ex, class Lp>
template <class BroadcastElement, ::numcxx::size_type Rank>
ndarray<Tp, Ex, Lp>::ndarray(const broadcast_view<BroadcastElement, Rank> &bv)
    : elem_(extents_type(bv.extents())) {
  static_assert(Rank == Ex::rank(),
                "Broadcast view rank must match ndarray rank");
  for (size_type i = 0; i < size(); ++i)
    flat_unchecked(i) = static_cast<value_type>(bv.flat_unchecked(i));
}

template <class Tp, class Ex, class Lp>
template <std::integral... Dims>
auto ndarray<Tp, Ex, Lp>::broadcast_to(Dims... dims) const {
  return detail::make_broadcast_view(*this, dims...);
}

template <class Tp, class Ex, class Lp>
template <std::integral Extent, ::numcxx::size_type Rank>
auto ndarray<Tp, Ex, Lp>::broadcast_to(
    const std::array<Extent, Rank> &shape) const {
  return detail::make_broadcast_view(*this, shape);
}

template <class Tp, class Ex, class Lp>
template <std::integral... Dims>
auto gslice_view<Tp, Ex, Lp>::broadcast_to(Dims... dims) const {
  return detail::make_broadcast_view(*this, dims...);
}

template <class Tp, class Ex, class Lp>
template <std::integral Extent, ::numcxx::size_type Rank>
auto gslice_view<Tp, Ex, Lp>::broadcast_to(
    const std::array<Extent, Rank> &shape) const {
  return detail::make_broadcast_view(*this, shape);
}

template <class Tp, size_type Rank>
template <std::integral... Dims>
auto broadcast_view<Tp, Rank>::broadcast_to(Dims... dims) const {
  return detail::make_broadcast_view(*this, dims...);
}

template <class Tp, size_type Rank>
template <std::integral Extent, ::numcxx::size_type TargetRank>
auto broadcast_view<Tp, Rank>::broadcast_to(
    const std::array<Extent, TargetRank> &shape) const {
  return detail::make_broadcast_view(*this, shape);
}

template <class Source, std::integral... Dims>
  requires detail::is_broadcast_source<std::remove_cvref_t<Source>>::value
[[nodiscard]] auto broadcast_to(const Source &source, Dims... dims) {
  return detail::make_broadcast_view(source, dims...);
}

template <class Source, std::integral Extent, size_type Rank>
  requires detail::is_broadcast_source<std::remove_cvref_t<Source>>::value
[[nodiscard]] auto broadcast_to(const Source &source,
                                const std::array<Extent, Rank> &shape) {
  return detail::make_broadcast_view(source, shape);
}

// [numcxx.expression_template]

// Expression nodes are evaluated in flat default_layout order.
//
// flat(i) accesses the element at linear index i as if the expression
// used default_layout, independent of each operand's physical layout.

// Provides evaluation and reduction APIs for a value expression.
template <class ValExpr> class nc_val_expr {
  using RmExpr = std::remove_cvref_t<ValExpr>;
  static_assert(nc_is_val_expr_v<RmExpr>,
                "nc_val_expr requires a value expression");

  ValExpr expr_;

public:
  // element
  using value_type = typename RmExpr::value_type;
  using element_type = value_type;

  // extents
  using extents_type = typename RmExpr::extents_type;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  // ndarray
  using ndarray_type = ndarray<value_type, extents_type, default_layout>;

  static constexpr rank_type rank() noexcept { return extents_type::rank(); }
  static constexpr rank_type rank_dynamic() noexcept {
    return extents_type::rank_dynamic();
  }
  static constexpr size_type static_extent(rank_type r) noexcept {
    return extents_type::static_extent(r);
  }
  constexpr index_type extent(rank_type r) const noexcept {
    return expr_.extent(r);
  }

  explicit nc_val_expr(const RmExpr &e) : expr_(e) {}

  [[nodiscard]] value_type flat(size_type i) const {
    detail::checked_index(i, size(), "nc_val_expr::flat");
    return flat_unchecked(i);
  }
  // Internal API (not stable): unchecked flat access; requires i < size().
  [[nodiscard]] value_type flat_unchecked(size_type i) const {
    return expr_.flat_unchecked(i);
  }

  // TODO:
  // - Support lazy slicing of expressions, e.g. (a + b)(slice(...)).
  // - Support lazy mask-based selection of expressions.
  // - Support lazy indirect indexing of expressions.

  // clang-format off
  auto operator+() const { return apply_unary_op<detail::nc_unary_plus>(); }
  auto operator-() const { return apply_unary_op<std::negate>          (); }
  auto operator~() const { return apply_unary_op<detail::nc_bit_not>   (); }
  auto operator!() const { return apply_unary_op<std::logical_not>     (); }
  // clang-format on

  [[nodiscard]] auto eval() const {
    ndarray_type res(extents());
    const size_type n = res.size();
    for (size_type i = 0; i < n; ++i) {
      res.flat_unchecked(i) = flat_unchecked(i);
    }
    return res;
  }

  [[nodiscard]] auto prod() const { return eval().prod(); }
  [[nodiscard]] auto mean() const { return eval().mean(); }
  [[nodiscard]] auto var() const { return eval().var(); }
  [[nodiscard]] auto std() const { return eval().std(); }
  [[nodiscard]] bool any() const { return eval().any(); }
  [[nodiscard]] bool all() const { return eval().all(); }
  [[nodiscard]] auto argmin() const { return eval().argmin(); }
  [[nodiscard]] auto argmax() const { return eval().argmax(); }

  [[nodiscard]] auto sum(::numcxx::index_type axis) const { return eval().sum(axis); }
  [[nodiscard]] auto min(::numcxx::index_type axis) const { return eval().min(axis); }
  [[nodiscard]] auto max(::numcxx::index_type axis) const { return eval().max(axis); }
  [[nodiscard]] auto prod(::numcxx::index_type axis) const { return eval().prod(axis); }
  [[nodiscard]] auto mean(::numcxx::index_type axis) const { return eval().mean(axis); }
  [[nodiscard]] auto var(::numcxx::index_type axis) const { return eval().var(axis); }
  [[nodiscard]] auto std(::numcxx::index_type axis) const { return eval().std(axis); }
  [[nodiscard]] auto any(::numcxx::index_type axis) const { return eval().any(axis); }
  [[nodiscard]] auto all(::numcxx::index_type axis) const { return eval().all(axis); }
  [[nodiscard]] auto argmin(::numcxx::index_type axis) const {
    return eval().argmin(axis);
  }
  [[nodiscard]] auto argmax(::numcxx::index_type axis) const {
    return eval().argmax(axis);
  }

  [[nodiscard]] constexpr size_type size() const noexcept {
    return expr_.size();
  }
  [[nodiscard]] constexpr bool empty() const noexcept { return expr_.empty(); }
  [[nodiscard]] constexpr const extents_type &extents() const noexcept {
    return expr_.extents();
  }

  value_type sum() const {
    const size_type n = size();
    value_type r = n ? flat_unchecked(0) : value_type();
    for (size_type i = 1; i < n; ++i)
      r += flat_unchecked(i);
    return r;
  }

  value_type min() const {
    const size_type n = size();
    value_type r = n ? flat_unchecked(0) : value_type();
    for (size_type i = 1; i < n; ++i) {
      const value_type x = flat_unchecked(i);
      if (x < r)
        r = x;
    }
    return r;
  }

  value_type max() const {
    const size_type n = size();
    value_type r = n ? flat_unchecked(0) : value_type();
    for (size_type i = 1; i < n; ++i) {
      const value_type x = flat_unchecked(i);
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
  // element
  using element_type = Tp;
  using value_type = std::remove_cv_t<Tp>;

  explicit nc_scalar_expr(const value_type &t, size_type s) : t_(t), s_(s) {}
  [[nodiscard]] value_type flat(size_type) const { return t_; }
  [[nodiscard]] value_type flat_unchecked(size_type) const { return t_; }
  size_type size() const { return s_; }

private:
  value_type t_;
  size_type s_;
};

namespace detail {

// Keep array-like leaves by reference; expression nodes and scalar operands
// remain inexpensive values.
template <class T> struct nc_operand_storage {
  using type = std::remove_cvref_t<T>;
};

template <class Tp, class Ex, class Lp>
struct nc_operand_storage<ndarray<Tp, Ex, Lp>> {
  using type = const ndarray<Tp, Ex, Lp> &;
};

template <class Tp, class Ex, class Lp>
struct nc_operand_storage<gslice_view<Tp, Ex, Lp>> {
  using type = const gslice_view<Tp, Ex, Lp> &;
};

template <class Tp> struct nc_operand_storage<mask_view<Tp>> {
  using type = const mask_view<Tp> &;
};

template <class Tp> struct nc_operand_storage<indirect_view<Tp>> {
  using type = const indirect_view<Tp> &;
};

template <class T>
using nc_operand_storage_t =
    typename nc_operand_storage<std::remove_cvref_t<T>>::type;

} // namespace detail

template <class Op, class A0> struct nc_unary_op {
  Op op_;
  detail::nc_operand_storage_t<A0> a0_;

  using RmA0 = std::remove_cvref_t<A0>;
  static_assert(nc_has_extents_v<RmA0>,
                "nc_unary_op requires an array operand");

  // element
  using value_type = std::decay_t<decltype(std::declval<Op>()(
      std::declval<const A0 &>().flat(0)))>;
  using element_type = value_type;

  // extents
  using extents_type = typename RmA0::extents_type;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  static constexpr rank_type rank() noexcept { return extents_type::rank(); }
  static constexpr rank_type rank_dynamic() noexcept {
    return extents_type::rank_dynamic();
  }
  static constexpr size_type static_extent(rank_type r) noexcept {
    return extents_type::static_extent(r);
  }
  constexpr index_type extent(rank_type r) const noexcept {
    return a0_.extent(r);
  }

  nc_unary_op(const Op &op, const A0 &a0) : op_(op), a0_(a0) {}

  [[nodiscard]] value_type flat(size_type i) const { return flat_unchecked(i); }

  [[nodiscard]] value_type flat_unchecked(size_type i) const {
    return op_(a0_.flat_unchecked(i));
  }

  [[nodiscard]] constexpr size_type size() const noexcept { return a0_.size(); }

  [[nodiscard]] constexpr bool empty() const noexcept { return a0_.empty(); }

  [[nodiscard]] constexpr const extents_type &extents() const noexcept {
    return a0_.extents();
  }
};

template <class Op, class A0, class A1> struct nc_binary_op {
  Op op_;
  detail::nc_operand_storage_t<A0> a0_;
  detail::nc_operand_storage_t<A1> a1_;

  using RmA0 = std::remove_cvref_t<A0>;
  using RmA1 = std::remove_cvref_t<A1>;
  static_assert(nc_has_extents_v<RmA0> || nc_has_extents_v<RmA1>,
                "nc_binary_op requires at least one array operand");

  // element
  using value_type = std::decay_t<decltype(std::declval<Op>()(
      std::declval<const A0 &>().flat(0), std::declval<const A1 &>().flat(0)))>;
  using element_type = value_type;

  // extents
  using array_operand_type =
      std::conditional_t<nc_has_extents_v<RmA0>, RmA0, RmA1>;
  using extents_type = typename array_operand_type::extents_type;
  using index_type = typename extents_type::index_type;
  using size_type = typename extents_type::size_type;
  using rank_type = typename extents_type::rank_type;

  static constexpr rank_type rank() noexcept { return extents_type::rank(); }
  static constexpr rank_type rank_dynamic() noexcept {
    return extents_type::rank_dynamic();
  }
  static constexpr size_type static_extent(rank_type r) noexcept {
    return extents_type::static_extent(r);
  }
  constexpr index_type extent(rank_type r) const noexcept {
    return extents().extent(r);
  }

  nc_binary_op(const Op &op, const A0 &a0, const A1 &a1)
      : op_(op), a0_(a0), a1_(a1) {
    if constexpr (nc_has_extents_v<RmA0> && nc_has_extents_v<RmA1>)
      detail::check_shape_match(a0, a1);
  }

  [[nodiscard]] value_type flat(size_type i) const { return flat_unchecked(i); }

  [[nodiscard]] value_type flat_unchecked(size_type i) const {
    return op_(a0_.flat_unchecked(i), a1_.flat_unchecked(i));
  }

  [[nodiscard]] constexpr size_type size() const noexcept {
    if constexpr (nc_has_extents_v<RmA0>) {
      return a0_.size();
    } else {
      return a1_.size();
    }
  }

  [[nodiscard]] constexpr bool empty() const noexcept { return size() == 0; }

  [[nodiscard]] constexpr const extents_type &extents() const noexcept {
    if constexpr (nc_has_extents_v<RmA0>) {
      return a0_.extents();
    } else {
      return a1_.extents();
    }
  }
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
inline auto make_expr_scalar(const Expr &x, const auto &y) {
  using scalar_type = std::remove_cvref_t<decltype(y)>;
  using OpType = nc_binary_op<Op, Expr, nc_scalar_expr<scalar_type>>;
  OpType node(Op{}, x, nc_scalar_expr<scalar_type>(y, x.size()));
  return nc_val_expr<OpType>(node);
}

template <class Op, class Expr>
inline auto make_scalar_expr(const auto &x, const Expr &y) {
  using scalar_type = std::remove_cvref_t<decltype(x)>;
  using OpType = nc_binary_op<Op, nc_scalar_expr<scalar_type>, Expr>;
  OpType node(Op{}, nc_scalar_expr<scalar_type>(x, y.size()), y);
  return nc_val_expr<OpType>(node);
}

template <class T> struct complex_component {
  using type = T;
  static constexpr bool is_complex = false;
};

template <class T> struct complex_component<std::complex<T>> {
  using type = T;
  static constexpr bool is_complex = true;
};

template <class T>
using complex_component_t =
    typename complex_component<std::remove_cvref_t<T>>::type;

template <class T, class U,
          bool IsComplex =
              complex_component<std::remove_cvref_t<T>>::is_complex ||
              complex_component<std::remove_cvref_t<U>>::is_complex>
struct promoted_type {
  using type =
      std::common_type_t<std::remove_cvref_t<T>, std::remove_cvref_t<U>>;
};

template <class T, class U> struct promoted_type<T, U, true> {
  using type = std::complex<
      std::common_type_t<complex_component_t<T>, complex_component_t<U>>>;
};

template <class T, class U>
using promoted_type_t = typename promoted_type<T, U>::type;
} // namespace detail

// Expression nodes reference array-like operands; consume them before those
// operands go out of scope or are moved.
// clang-format off
#define NUMCXX_MAKE_UNARY_FN(FN, FUNCTOR)                                      \
  template <value_expression E>                                                \
  [[nodiscard]] inline auto FN(const E& x) {                                   \
    return detail::make_unary_op<FUNCTOR<typename E::value_type>>(x);          \
  }

#define NUMCXX_MAKE_BINARY_OP(OP, FUNCTOR)                                     \
  template <value_expression E1, value_expression E2>                         \
  inline auto operator OP(const E1 &x, const E2 &y) {                          \
    using value_type = detail::promoted_type_t<typename E1::value_type,       \
                                                typename E2::value_type>;      \
    return detail::make_expr_expr<FUNCTOR<value_type>>(x, y);                 \
  }                                                                            \
  template <value_expression E, class S>                                      \
    requires (!value_expression<S>)                                            \
  inline auto operator OP(const E &x, const S &y) {                            \
    using value_type = detail::promoted_type_t<typename E::value_type, S>;     \
    return detail::make_expr_scalar<FUNCTOR<value_type>>(x, y);               \
  }                                                                            \
  template <class S, value_expression E>                                      \
    requires (!value_expression<S>)                                            \
  inline auto operator OP(const S &x, const E &y) {                            \
    using value_type = detail::promoted_type_t<S, typename E::value_type>;     \
    return detail::make_scalar_expr<FUNCTOR<value_type>>(x, y);               \
  }

#define NUMCXX_MAKE_BINARY_FN(FN, FUNCTOR)                                     \
  template <value_expression E1, value_expression E2>                         \
  inline auto FN(const E1 &x, const E2 &y) {                                   \
    using value_type = detail::promoted_type_t<typename E1::value_type,       \
                                                typename E2::value_type>;      \
    return detail::make_expr_expr<FUNCTOR<value_type>>(x, y);                 \
  }                                                                            \
  template <value_expression E, class S>                                      \
    requires (!value_expression<S>)                                            \
  inline auto FN(const E &x, const S &y) {                                     \
    using value_type = detail::promoted_type_t<typename E::value_type, S>;     \
    return detail::make_expr_scalar<FUNCTOR<value_type>>(x, y);               \
  }                                                                            \
  template <class S, value_expression E>                                      \
    requires (!value_expression<S>)                                            \
  inline auto FN(const S &x, const E &y) {                                     \
    using value_type = detail::promoted_type_t<S, typename E::value_type>;     \
    return detail::make_scalar_expr<FUNCTOR<value_type>>(x, y);               \
  }

  template <class Tp> struct nc_bit_shift_left  { Tp operator()(const Tp &x, const Tp &y) const { return x << y; } };
  template <class Tp> struct nc_bit_shift_right { Tp operator()(const Tp &x, const Tp &y) const { return x >> y; } };

  template <class Tp> struct nc_abs_expr   { auto operator()(const Tp& x) const { return std::abs  (x); } };
  template <class Tp> struct nc_acos_expr  { Tp operator()(const Tp& x) const { return std::acos (x); } };
  template <class Tp> struct nc_asin_expr  { Tp operator()(const Tp& x) const { return std::asin (x); } };
  template <class Tp> struct nc_atan_expr  { Tp operator()(const Tp& x) const { return std::atan (x); } };
  template <class Tp> struct nc_cos_expr   { Tp operator()(const Tp& x) const { return std::cos  (x); } };
  template <class Tp> struct nc_cosh_expr  { Tp operator()(const Tp& x) const { return std::cosh (x); } };
  template <class Tp> struct nc_exp_expr   { Tp operator()(const Tp& x) const { return std::exp  (x); } };
  template <class Tp> struct nc_log_expr   { Tp operator()(const Tp& x) const { return std::log  (x); } };
  template <class Tp> struct nc_log10_expr {
    auto operator()(const Tp &x) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return std::log(x) / std::log(Tp(10));
      else
        return std::log10(x);
    }
  };
  template <class Tp> struct nc_sin_expr   { Tp operator()(const Tp& x) const { return std::sin  (x); } };
  template <class Tp> struct nc_sinh_expr  { Tp operator()(const Tp& x) const { return std::sinh (x); } };
  template <class Tp> struct nc_sqrt_expr  { Tp operator()(const Tp& x) const { return std::sqrt (x); } };
  template <class Tp> struct nc_tan_expr   { Tp operator()(const Tp& x) const { return std::tan  (x); } };
  template <class Tp> struct nc_tanh_expr  { Tp operator()(const Tp& x) const { return std::tanh (x); } };

  template <class Tp> struct nc_atan2_expr { Tp operator()(const Tp& x, const Tp& y) const { return std::atan2(x, y); } };
  template <class Tp> struct nc_pow_expr   { Tp operator()(const Tp& x, const Tp& y) const { return std::pow  (x, y); } };
  template <class Tp> struct nc_conj_expr {
    auto operator()(const Tp &x) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return std::conj(x);
      else
        return x;
    }
  };
  template <class Tp> struct nc_real_expr {
    auto operator()(const Tp &x) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return x.real();
      else
        return x;
    }
  };
  template <class Tp> struct nc_imag_expr {
    auto operator()(const Tp &x) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return x.imag();
      else
        return Tp{};
    }
  };
  template <class Tp> struct nc_arg_expr {
    auto operator()(const Tp &x) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return std::arg(x);
      else
        return std::arg(std::complex<Tp>(x));
    }
  };
  template <class Tp> struct nc_less_expr {
    bool operator()(const Tp &x, const Tp &y) const {
      return detail::reduction_less(x, y);
    }
  };
  template <class Tp> struct nc_less_equal_expr {
    bool operator()(const Tp &x, const Tp &y) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return nc_less_expr<Tp>{}(x, y) || x == y;
      else
        return x <= y;
    }
  };
  template <class Tp> struct nc_greater_expr {
    bool operator()(const Tp &x, const Tp &y) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return nc_less_expr<Tp>{}(y, x);
      else
        return x > y;
    }
  };
  template <class Tp> struct nc_greater_equal_expr {
    bool operator()(const Tp &x, const Tp &y) const {
      if constexpr (detail::complex_component<Tp>::is_complex)
        return nc_less_expr<Tp>{}(y, x) || x == y;
      else
        return x >= y;
    }
  };

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
NUMCXX_MAKE_BINARY_OP( <, nc_less_expr         )
NUMCXX_MAKE_BINARY_OP(<=, nc_less_equal_expr   )
NUMCXX_MAKE_BINARY_OP( >, nc_greater_expr      )
NUMCXX_MAKE_BINARY_OP(>=, nc_greater_equal_expr)

// absolute function
NUMCXX_MAKE_UNARY_FN( abs, nc_abs_expr )
NUMCXX_MAKE_UNARY_FN(conj, nc_conj_expr)
NUMCXX_MAKE_UNARY_FN(real, nc_real_expr)
NUMCXX_MAKE_UNARY_FN(imag, nc_imag_expr)
NUMCXX_MAKE_UNARY_FN( arg, nc_arg_expr )
NUMCXX_MAKE_UNARY_FN(angle, nc_arg_expr)

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
// clang-format on

// [numcxx.print]
namespace detail {

// clang-format off
template <typename T> struct printf_format { static constexpr const char *fmt = nullptr; };
template <typename T> struct printf_format<const          T> : printf_format<T> {};
template <typename T> struct printf_format<      volatile T> : printf_format<T> {};
template <typename T> struct printf_format<const volatile T> : printf_format<T> {};
template <typename T> using  printf_format_t = printf_format<std::remove_cv_t<T>> ;

#define NUMCXX_DEF_PRINTF_FMT(T, fmt_str)                                      \
  template <> struct printf_format<T> {                                        \
    static constexpr const char *fmt = fmt_str;                                \
  };

// signed
NUMCXX_DEF_PRINTF_FMT(short             , "%hd"  )
NUMCXX_DEF_PRINTF_FMT(int               , "%d"   )
NUMCXX_DEF_PRINTF_FMT(long              , "%ld"  )
NUMCXX_DEF_PRINTF_FMT(long long         , "%lld" )

// unsigned
NUMCXX_DEF_PRINTF_FMT(unsigned short    , "%hu"  )
NUMCXX_DEF_PRINTF_FMT(unsigned int      , "%u"   )
NUMCXX_DEF_PRINTF_FMT(unsigned long     , "%lu"  )
NUMCXX_DEF_PRINTF_FMT(unsigned long long, "%llu" )

// floating
NUMCXX_DEF_PRINTF_FMT(float             , "%.6g" )
NUMCXX_DEF_PRINTF_FMT(double            , "%.6g" )
NUMCXX_DEF_PRINTF_FMT(long double       , "%.6Lg")

// misc
NUMCXX_DEF_PRINTF_FMT(char              , "%c"   )
NUMCXX_DEF_PRINTF_FMT(bool              , "%d"   )
NUMCXX_DEF_PRINTF_FMT(signed char       , "%hhd" )
NUMCXX_DEF_PRINTF_FMT(unsigned char     , "%hhu" )

template <typename T> void print_complex(const std::complex<T> &z, FILE *file) {
  if constexpr (std::is_same_v<T, long double>) {
    std::fprintf(file, "(%Lg, %Lg)", z.real(), z.imag());
  } else {
    std::fprintf(file, "(%g, %g)", static_cast<double>(z.real()),
                 static_cast<double>(z.imag()));
  }
}

template <typename T> void print_element(const T &x, FILE *file) {
  using U = std::remove_cv_t<std::remove_reference_t<T>>;

  if constexpr (complex_component<U>::is_complex) {
    print_complex(x, file);
  } else {
    static_assert(printf_format_t<U>::fmt != nullptr, "numcxx::print: unsupported type for printf_format");
    std::fprintf(file, printf_format_t<U>::fmt, x);
  }
}

template <typename M>
void print_recursive(const M &arr, FILE *file, size_type dim_idx,
                     size_type offset) {
  constexpr auto R = std::decay_t<M>::rank();
  size_type dim_len = arr.extent(dim_idx);
  size_type stride = arr.stride(dim_idx);

  if (dim_idx == R - 1) {
    std::fprintf(file, "[");
    for (size_type i = 0; i < dim_len; ++i) {
      if (i != 0)
        std::fprintf(file, ", ");
      size_type idx = offset + i * stride;
      print_element(arr[idx], file);
    }
    std::fprintf(file, "]");
  } else {
    std::fprintf(file, "[");
    for (size_type i = 0; i < dim_len; ++i) {
      if (i != 0)
        std::fprintf(file, ",\n ");
      print_recursive(arr, file, dim_idx + 1, offset + i * stride);
    }
    std::fprintf(file, "]");
  }
}

// clang-format off

} // namespace detail

template <value_expression M>
void print(const M &arr, FILE *file = stdout) {
  constexpr auto R = std::decay_t<M>::rank();
  if constexpr (R == 0) {
    detail::print_element(arr[0], file);
  } else {
    detail::print_recursive(arr, file, 0, 0);
  }
  std::fprintf(file, "\n");
}

//
// [numcxx.linalg] linear algebra
//
#ifdef NUMCXX_DETAIL_HAS_CBLAS
namespace detail {
template <class Matrix> struct is_gslice_view : std::false_type {};
template <class ElementType, class Extents, class LayoutPolicy>
struct is_gslice_view<gslice_view<ElementType, Extents, LayoutPolicy>>
    : std::true_type {};
template <class Matrix>
inline constexpr bool is_gslice_view_v = is_gslice_view<std::remove_cvref_t<Matrix>>::value;

// Determine whether a matrix can be passed directly to CBLAS and, if so,
// return its row-major or column-major storage order.
template <class Matrix>
bool cblas_matrix_order(const Matrix &matrix, CBLAS_ORDER &order) {
  const auto span = matrix.to_mdspan();
  const auto &mapping = span.mapping();
  if (!mapping.is_exhaustive() || !mapping.is_strided())
    return false;

  if constexpr (is_gslice_view_v<Matrix>) {
    if (matrix.has_reversed_dim())
      return false;
  }

  const bool row_contiguous =
      span.stride(1) == 1 && span.stride(0) == span.extent(1);
  const bool col_contiguous =
      span.stride(0) == 1 && span.stride(1) == span.extent(0);

  if (row_contiguous) {
    order = CblasRowMajor;
    return true;
  }
  if (col_contiguous) {
    order = CblasColMajor;
    return true;
  }

  return false;
}

// Multiply contiguous BLAS-supported matrices through CBLAS. Return false
// when the operands or dimensions are unsupported so the caller can fall back
// to the generic matrix_product implementation.
template <class A, class B, class C>
bool nc_cblas_gemm(const A &a, const B &b, C &c) {
  using value_type =
      promoted_type_t<typename A::value_type, typename B::value_type>;
  if constexpr ((std::is_same_v<value_type, float> ||
                 std::is_same_v<value_type, double> ||
                 std::is_same_v<value_type, std::complex<float>> ||
                 std::is_same_v<value_type, std::complex<double>>) &&
                std::is_same_v<typename A::value_type, value_type> &&
                std::is_same_v<typename B::value_type, value_type>) {
    const auto &a_ext = a.extents();
    const auto &b_ext = b.extents();
    const auto m = a_ext.extent(0);
    const auto n = b_ext.extent(1);
    const auto k = a_ext.extent(1);
    constexpr auto cblas_max = std::numeric_limits<int>::max();
    if (m <= 0 || n <= 0 || k <= 0 ||
        m > cblas_max || n > cblas_max || k > cblas_max)
      return false;

    CBLAS_ORDER a_order;
    CBLAS_ORDER b_order;
    if (!cblas_matrix_order(a, a_order) ||
        !cblas_matrix_order(b, b_order))
      return false;

    constexpr CBLAS_ORDER result_order =
        std::is_same_v<default_layout, layout_right> ? CblasRowMajor
                                                     : CblasColMajor;
    const auto trans_a =
        a_order == result_order ? CblasNoTrans : CblasTrans;
    const auto trans_b =
        b_order == result_order ? CblasNoTrans : CblasTrans;
    const int cblas_m = static_cast<int>(m);
    const int cblas_n = static_cast<int>(n);
    const int cblas_k = static_cast<int>(k);
    const int lda = result_order == CblasRowMajor
                        ? (trans_a == CblasNoTrans ? cblas_k : cblas_m)
                        : (trans_a == CblasNoTrans ? cblas_m : cblas_k);
    const int ldb = result_order == CblasRowMajor
                        ? (trans_b == CblasNoTrans ? cblas_n : cblas_k)
                        : (trans_b == CblasNoTrans ? cblas_k : cblas_n);
    const int ldc = result_order == CblasRowMajor ? cblas_n : cblas_m;
    const value_type alpha = value_type{1};
    const value_type beta = value_type{};

    if constexpr (std::is_same_v<value_type, float>) {
      cblas_sgemm(result_order, trans_a, trans_b, cblas_m, cblas_n, cblas_k,
                  alpha, a.to_mdspan().data_handle(), lda,
                  b.to_mdspan().data_handle(), ldb, beta,
                  c.to_mdspan().data_handle(), ldc);
    } else if constexpr (std::is_same_v<value_type, double>) {
      cblas_dgemm(result_order, trans_a, trans_b, cblas_m, cblas_n, cblas_k,
                  alpha, a.to_mdspan().data_handle(), lda,
                  b.to_mdspan().data_handle(), ldb, beta,
                  c.to_mdspan().data_handle(), ldc);
    } else if constexpr (std::is_same_v<value_type, std::complex<float>>) {
      cblas_cgemm(result_order, trans_a, trans_b, cblas_m, cblas_n, cblas_k,
                  &alpha, a.to_mdspan().data_handle(), lda,
                  b.to_mdspan().data_handle(), ldb, &beta,
                  c.to_mdspan().data_handle(), ldc);
    } else {
      cblas_zgemm(result_order, trans_a, trans_b, cblas_m, cblas_n, cblas_k,
                  &alpha, a.to_mdspan().data_handle(), lda,
                  b.to_mdspan().data_handle(), ldb, &beta,
                  c.to_mdspan().data_handle(), ldc);
    }
    return true;
  }
  return false;
}
} // namespace detail
#endif

namespace detail {
template <class T> struct is_solve_operand : std::false_type {};
template <class T, class Extents, class LayoutPolicy>
struct is_solve_operand<ndarray<T, Extents, LayoutPolicy>> : std::true_type {};
template <class T, class Extents, class LayoutPolicy>
struct is_solve_operand<gslice_view<T, Extents, LayoutPolicy>>
    : std::true_type {};
template <class T>
inline constexpr bool is_solve_operand_v =
    is_solve_operand<std::remove_cvref_t<T>>::value;

template <class T>
inline constexpr bool is_solve_value_v =
    std::is_same_v<T, float> || std::is_same_v<T, double> ||
    std::is_same_v<T, std::complex<float>> ||
    std::is_same_v<T, std::complex<double>>;

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
template <class Matrix>
bool lapack_matrix_order(const Matrix &matrix, int &order) {
  const auto span = matrix.to_mdspan();
  const auto &mapping = span.mapping();
  if (!mapping.is_exhaustive() || !mapping.is_strided())
    return false;
  if constexpr (requires { matrix.has_reversed_dim(); }) {
    if (matrix.has_reversed_dim())
      return false;
  }

  const bool row_contiguous =
      span.stride(1) == 1 && span.stride(0) == span.extent(1);
  const bool col_contiguous =
      span.stride(0) == 1 && span.stride(1) == span.extent(0);
  if (row_contiguous) {
    order = LAPACK_ROW_MAJOR;
    return true;
  }
  if (col_contiguous) {
    order = LAPACK_COL_MAJOR;
    return true;
  }
  return false;
}

template <class Vector>
bool lapack_vector_contiguous(const Vector &vector) {
  const auto span = vector.to_mdspan();
  if (!span.mapping().is_exhaustive() || !span.mapping().is_strided() ||
      span.stride(0) != 1)
    return false;
  if constexpr (requires { vector.has_reversed_dim(); })
    return !vector.has_reversed_dim();
  return true;
}

template <class T>
lapack_int lapack_gesv(int order, lapack_int n, lapack_int nrhs, T *a,
                       lapack_int lda, lapack_int *ipiv, T *b,
                       lapack_int ldb) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_sgesv(order, n, nrhs, a, lda, ipiv, b, ldb);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dgesv(order, n, nrhs, a, lda, ipiv, b, ldb);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    const size_type a_size = static_cast<size_type>(n) * n;
    const size_type b_size = static_cast<size_type>(n) * nrhs;
    std::vector<lapack_complex_float> lapack_a(a_size);
    std::vector<lapack_complex_float> lapack_b(b_size);
    for (size_type i = 0; i < a_size; ++i)
      lapack_a[i] = lapack_make_complex_float(a[i].real(), a[i].imag());
    for (size_type i = 0; i < b_size; ++i)
      lapack_b[i] = lapack_make_complex_float(b[i].real(), b[i].imag());
    const lapack_int info =
        LAPACKE_cgesv(order, n, nrhs, lapack_a.data(), lda, ipiv,
                      lapack_b.data(), ldb);
    for (size_type i = 0; i < b_size; ++i)
      b[i] = {lapack_complex_float_real(lapack_b[i]),
              lapack_complex_float_imag(lapack_b[i])};
    return info;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    const size_type a_size = static_cast<size_type>(n) * n;
    const size_type b_size = static_cast<size_type>(n) * nrhs;
    std::vector<lapack_complex_double> lapack_a(a_size);
    std::vector<lapack_complex_double> lapack_b(b_size);
    for (size_type i = 0; i < a_size; ++i)
      lapack_a[i] = lapack_make_complex_double(a[i].real(), a[i].imag());
    for (size_type i = 0; i < b_size; ++i)
      lapack_b[i] = lapack_make_complex_double(b[i].real(), b[i].imag());
    const lapack_int info =
        LAPACKE_zgesv(order, n, nrhs, lapack_a.data(), lda, ipiv,
                      lapack_b.data(), ldb);
    for (size_type i = 0; i < b_size; ++i)
      b[i] = {lapack_complex_double_real(lapack_b[i]),
              lapack_complex_double_imag(lapack_b[i])};
    return info;
  } else {
    static_assert(is_solve_value_v<T>,
                  "solve supports float, double, complex<float>, and complex<double>");
    return lapack_int{};
  }
}

template <class T>
lapack_int lapack_potrf(int order, char uplo, lapack_int n, T *a,
                        lapack_int lda) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_spotrf(order, uplo, n, a, lda);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dpotrf(order, uplo, n, a, lda);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    // The extra conversion buffer avoids relying on std::complex/LAPACKE ABI
    // compatibility.
    const size_type matrix_size = static_cast<size_type>(n) * n;
    std::vector<lapack_complex_float> lapack_a(matrix_size);
    for (size_type i = 0; i < matrix_size; ++i)
      lapack_a[i] = lapack_make_complex_float(a[i].real(), a[i].imag());
    const lapack_int info = LAPACKE_cpotrf(order, uplo, n, lapack_a.data(), lda);
    for (size_type i = 0; i < matrix_size; ++i)
      a[i] = {lapack_complex_float_real(lapack_a[i]),
              lapack_complex_float_imag(lapack_a[i])};
    return info;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    const size_type matrix_size = static_cast<size_type>(n) * n;
    std::vector<lapack_complex_double> lapack_a(matrix_size);
    for (size_type i = 0; i < matrix_size; ++i)
      lapack_a[i] = lapack_make_complex_double(a[i].real(), a[i].imag());
    const lapack_int info = LAPACKE_zpotrf(order, uplo, n, lapack_a.data(), lda);
    for (size_type i = 0; i < matrix_size; ++i)
      a[i] = {lapack_complex_double_real(lapack_a[i]),
              lapack_complex_double_imag(lapack_a[i])};
    return info;
  } else {
    static_assert(is_solve_value_v<T>,
                  "cholesky supports float, double, complex<float>, and "
                  "complex<double>");
    return lapack_int{};
  }
}

template <class T>
lapack_int lapack_geqrf(int order, lapack_int m, lapack_int n, T *a,
                        lapack_int lda, T *tau) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_sgeqrf(order, m, n, a, lda, tau);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dgeqrf(order, m, n, a, lda, tau);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    const size_type matrix_size = static_cast<size_type>(m) * n;
    const size_type reflector_count = std::min(m, n);
    std::vector<lapack_complex_float> lapack_a(matrix_size);
    std::vector<lapack_complex_float> lapack_tau(reflector_count);
    for (size_type i = 0; i < matrix_size; ++i)
      lapack_a[i] = lapack_make_complex_float(a[i].real(), a[i].imag());
    const lapack_int info =
        LAPACKE_cgeqrf(order, m, n, lapack_a.data(), lda, lapack_tau.data());
    for (size_type i = 0; i < matrix_size; ++i)
      a[i] = {lapack_complex_float_real(lapack_a[i]),
              lapack_complex_float_imag(lapack_a[i])};
    for (size_type i = 0; i < reflector_count; ++i)
      tau[i] = {lapack_complex_float_real(lapack_tau[i]),
                lapack_complex_float_imag(lapack_tau[i])};
    return info;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    const size_type matrix_size = static_cast<size_type>(m) * n;
    const size_type reflector_count = std::min(m, n);
    std::vector<lapack_complex_double> lapack_a(matrix_size);
    std::vector<lapack_complex_double> lapack_tau(reflector_count);
    for (size_type i = 0; i < matrix_size; ++i)
      lapack_a[i] = lapack_make_complex_double(a[i].real(), a[i].imag());
    const lapack_int info =
        LAPACKE_zgeqrf(order, m, n, lapack_a.data(), lda, lapack_tau.data());
    for (size_type i = 0; i < matrix_size; ++i)
      a[i] = {lapack_complex_double_real(lapack_a[i]),
              lapack_complex_double_imag(lapack_a[i])};
    for (size_type i = 0; i < reflector_count; ++i)
      tau[i] = {lapack_complex_double_real(lapack_tau[i]),
                lapack_complex_double_imag(lapack_tau[i])};
    return info;
  } else {
    static_assert(is_solve_value_v<T>,
                  "qr supports float, double, complex<float>, and "
                  "complex<double>");
    return lapack_int{};
  }
}

template <class T>
lapack_int lapack_orgqr(int order, lapack_int m, lapack_int n, lapack_int k,
                        T *a, lapack_int lda, const T *tau) {
  if constexpr (std::is_same_v<T, float>) {
    return LAPACKE_sorgqr(order, m, n, k, a, lda, tau);
  } else if constexpr (std::is_same_v<T, double>) {
    return LAPACKE_dorgqr(order, m, n, k, a, lda, tau);
  } else if constexpr (std::is_same_v<T, std::complex<float>>) {
    const size_type matrix_size = static_cast<size_type>(m) * n;
    const size_type reflector_count = static_cast<size_type>(k);
    std::vector<lapack_complex_float> lapack_a(matrix_size);
    std::vector<lapack_complex_float> lapack_tau(reflector_count);
    for (size_type i = 0; i < matrix_size; ++i)
      lapack_a[i] = lapack_make_complex_float(a[i].real(), a[i].imag());
    for (size_type i = 0; i < reflector_count; ++i)
      lapack_tau[i] = lapack_make_complex_float(tau[i].real(), tau[i].imag());
    const lapack_int info =
        LAPACKE_cungqr(order, m, n, k, lapack_a.data(), lda, lapack_tau.data());
    for (size_type i = 0; i < matrix_size; ++i)
      a[i] = {lapack_complex_float_real(lapack_a[i]),
              lapack_complex_float_imag(lapack_a[i])};
    return info;
  } else if constexpr (std::is_same_v<T, std::complex<double>>) {
    const size_type matrix_size = static_cast<size_type>(m) * n;
    const size_type reflector_count = static_cast<size_type>(k);
    std::vector<lapack_complex_double> lapack_a(matrix_size);
    std::vector<lapack_complex_double> lapack_tau(reflector_count);
    for (size_type i = 0; i < matrix_size; ++i)
      lapack_a[i] = lapack_make_complex_double(a[i].real(), a[i].imag());
    for (size_type i = 0; i < reflector_count; ++i)
      lapack_tau[i] = lapack_make_complex_double(tau[i].real(), tau[i].imag());
    const lapack_int info =
        LAPACKE_zungqr(order, m, n, k, lapack_a.data(), lda, lapack_tau.data());
    for (size_type i = 0; i < matrix_size; ++i)
      a[i] = {lapack_complex_double_real(lapack_a[i]),
              lapack_complex_double_imag(lapack_a[i])};
    return info;
  } else {
    static_assert(is_solve_value_v<T>,
                  "qr supports float, double, complex<float>, and "
                  "complex<double>");
    return lapack_int{};
  }
}
#endif

template <class T>
void gaussian_solve(std::vector<T> &a, std::vector<T> &b, size_type n,
                    size_type nrhs) {
  using magnitude_type = decltype(std::abs(T{}));
  for (size_type k = 0; k < n; ++k) {
    size_type pivot = k;
    magnitude_type pivot_magnitude = std::abs(a[k * n + k]);
    for (size_type i = k + 1; i < n; ++i) {
      const auto magnitude = std::abs(a[i * n + k]);
      if (magnitude > pivot_magnitude) {
        pivot = i;
        pivot_magnitude = magnitude;
      }
    }
    if (pivot_magnitude == magnitude_type{}) {
      // Match LAPACK gesv: report singularity only for an exactly zero pivot.
      const std::string message =
          "solve: matrix is singular at pivot " + std::to_string(k + 1);
      NUMCXX_THROW(std::runtime_error, message.c_str());
    }

    if (pivot != k) {
      for (size_type j = 0; j < n; ++j)
        std::swap(a[k * n + j], a[pivot * n + j]);
      for (size_type j = 0; j < nrhs; ++j)
        std::swap(b[k * nrhs + j], b[pivot * nrhs + j]);
    }

    for (size_type i = k + 1; i < n; ++i) {
      const T factor = a[i * n + k] / a[k * n + k];
      a[i * n + k] = T{};
#pragma omp simd
      for (size_type j = k + 1; j < n; ++j)
        a[i * n + j] -= factor * a[k * n + j];
#pragma omp simd
      for (size_type j = 0; j < nrhs; ++j)
        b[i * nrhs + j] -= factor * b[k * nrhs + j];
    }
  }

  for (size_type ii = n; ii-- > 0;) {
    for (size_type rhs = 0; rhs < nrhs; ++rhs) {
      for (size_type j = ii + 1; j < n; ++j)
        b[ii * nrhs + rhs] -= a[ii * n + j] * b[j * nrhs + rhs];
      b[ii * nrhs + rhs] /= a[ii * n + ii];
    }
  }
}


template <class T> T cholesky_conjugate(const T &value) {
  if constexpr (std::is_same_v<T, std::complex<float>> ||
                std::is_same_v<T, std::complex<double>>)
    return std::conj(value);
  else
    return value;
}

template <class T> auto cholesky_real_part(const T &value) {
  if constexpr (std::is_same_v<T, std::complex<float>> ||
                std::is_same_v<T, std::complex<double>>)
    return value.real();
  else
    return value;
}

template <class T>
void cholesky_fallback(std::vector<T> &matrix, size_type n, bool upper) {
  using real_type = decltype(cholesky_real_part(T{}));
  for (size_type j = 0; j < n; ++j) {
    T diagonal = matrix[j * n + j];
    for (size_type k = 0; k < j; ++k) {
      if (upper)
        diagonal -= cholesky_conjugate(matrix[k * n + j]) *
                    matrix[k * n + j];
      else
        diagonal -= matrix[j * n + k] *
                    cholesky_conjugate(matrix[j * n + k]);
    }

    const real_type real_diagonal = cholesky_real_part(diagonal);
    if (!(real_diagonal > real_type{})) {
      const std::string message =
          "cholesky: matrix is not positive-definite at leading minor " +
          std::to_string(j + 1);
      NUMCXX_THROW(std::runtime_error, message.c_str());
    }
    const real_type root = std::sqrt(real_diagonal);
    matrix[j * n + j] = T{root};

    if (upper) {
      for (size_type column = j + 1; column < n; ++column) {
        T value = matrix[j * n + column];
        for (size_type k = 0; k < j; ++k)
          value -= cholesky_conjugate(matrix[k * n + j]) *
                   matrix[k * n + column];
        matrix[j * n + column] = value / root;
      }
    } else {
      for (size_type row = j + 1; row < n; ++row) {
        T value = matrix[row * n + j];
        for (size_type k = 0; k < j; ++k)
          value -= matrix[row * n + k] *
                   cholesky_conjugate(matrix[j * n + k]);
        matrix[row * n + j] = value / root;
      }
    }
  }
}

template <class T>
void qr_fallback(std::vector<T> &matrix, size_type m, size_type n,
                 std::vector<T> &tau) {
  using real_type = decltype(cholesky_real_part(T{}));
  const size_type k = std::min(m, n);
  for (size_type j = 0; j < k; ++j) {
    real_type norm{};
    // Accumulating with hypot avoids overflow and underflow in the norm.
    for (size_type i = j; i < m; ++i)
      norm = std::hypot(norm, static_cast<real_type>(
                                   std::abs(matrix[i * n + j])));

    const T alpha = matrix[j * n + j];
    if (norm == real_type{}) {
      tau[j] = T{};
      continue;
    }

    // Choosing beta with the opposite real sign from alpha keeps alpha - beta
    // nonzero whenever the column has nonzero norm.
    const real_type beta_real =
        cholesky_real_part(alpha) >= real_type{} ? -norm : norm;
    const T beta{beta_real};
    const T v0 = alpha - beta;
    tau[j] = (beta - alpha) / beta;
    for (size_type i = j + 1; i < m; ++i)
      matrix[i * n + j] /= v0;
    matrix[j * n + j] = beta;

    for (size_type column = j + 1; column < n; ++column) {
      T dot = matrix[j * n + column];
      for (size_type i = j + 1; i < m; ++i)
        dot += cholesky_conjugate(matrix[i * n + j]) *
               matrix[i * n + column];
      dot *= cholesky_conjugate(tau[j]);
      matrix[j * n + column] -= dot;
      for (size_type i = j + 1; i < m; ++i)
        matrix[i * n + column] -= matrix[i * n + j] * dot;
    }
  }
}

template <class T>
void form_q_fallback(const std::vector<T> &reflectors,
                     const std::vector<T> &tau, std::vector<T> &q,
                     size_type m, size_type n, size_type q_columns,
                     size_type k) {
  for (size_type i = 0; i < std::min(m, q_columns); ++i)
    q[i * q_columns + i] = T{1};

  for (size_type jj = k; jj-- > 0;) {
    for (size_type column = 0; column < q_columns; ++column) {
      T dot{};
      for (size_type i = jj; i < m; ++i) {
        const T v = i == jj ? T{1} : reflectors[i * n + jj];
        dot += cholesky_conjugate(v) * q[i * q_columns + column];
      }
      dot *= tau[jj];
      for (size_type i = jj; i < m; ++i) {
        const T v = i == jj ? T{1} : reflectors[i * n + jj];
        q[i * q_columns + column] -= v * dot;
      }
    }
  }
}

} // namespace detail

namespace linalg {

/// 2D matrix multiplication: C = A x B
///
/// Computes the matrix product of two rank-2 arrays or views.
///
/// @tparam A Type with rank() == 2 and to_mdspan() (ndarray, gslice_view)
/// @tparam B Type with rank() == 2 and to_mdspan() (ndarray, gslice_view)
/// @param  a Left matrix of shape (M × K)
/// @param  b Right matrix of shape (K × N)
/// @returns Result matrix of shape (M × N) in default_layout
///
/// @throws std::invalid_argument If the inner dimensions do not match
template <class A, class B> auto matmul(const A &a, const B &b) {
  static_assert(nc_has_extents_v<A>, "lhs must be ndarray/gslice_view type");
  static_assert(nc_has_extents_v<B>, "rhs must be ndarray/gslice_view type");
  static_assert(A::rank() == 2, "matmul requires rank-2 lhs");
  static_assert(B::rank() == 2, "matmul requires rank-2 rhs");

  const auto &a_ext = a.extents();
  const auto &b_ext = b.extents();
  if (a_ext.extent(1) != b_ext.extent(0)) {
    NUMCXX_THROW(std::invalid_argument, "matmul: incompatible shapes");
  }

  using value_type =
      detail::promoted_type_t<typename A::value_type, typename B::value_type>;
  using extents_type = dextents<2>;
  ndarray<value_type, extents_type, default_layout> c(a_ext.extent(0),
                                                      b_ext.extent(1));

#ifdef NUMCXX_DETAIL_HAS_CBLAS
  // CBLAS requires one storage order per call. Mixed layouts are handled with
  // transpose flags; unsupported layouts and types use the generic fallback.
  if (detail::nc_cblas_gemm(a, b, c))
    return c;
#endif

  if constexpr (std::is_same_v<typename A::value_type, value_type> &&
                std::is_same_v<typename B::value_type, value_type>) {
    detail::linalg::matrix_product(a.to_mdspan(), b.to_mdspan(),
                                   c.to_mdspan());
  } else {
    for (size_type i = 0; i < c.extent(0); ++i) {
      for (size_type j = 0; j < c.extent(1); ++j) {
        value_type sum{};
        for (size_type k = 0; k < a_ext.extent(1); ++k) {
          sum += static_cast<value_type>(a(i, k)) *
                 static_cast<value_type>(b(k, j));
        }
        c(i, j) = sum;
      }
    }
  }

  return c;
}

/// Dot product
///
/// - 1D x 1D: inner product (scalar)
/// - 2D x 1D: matrix-vector product (1D)
/// - 1D x 2D: vector-matrix product (1D)
/// - 2D x 2D: matrix multiplication (2D)
///
/// @tparam A Type with to_mdspan() (ndarray, gslice_view, ...)
/// @tparam B Type with to_mdspan()
/// @param a First operand
/// @param b Second operand
/// @returns Result (scalar, 1D array, or 2D array)
///
/// @throws std::invalid_argument If shapes are incompatible
template <class A, class B> auto dot(const A &a, const B &b) {
  static_assert(nc_has_extents_v<A>, "lhs must be ndarray/gslice_view type");
  static_assert(nc_has_extents_v<B>, "rhs must be ndarray/gslice_view type");

  constexpr int rankA = A::rank();
  constexpr int rankB = B::rank();

  using value_type =
      detail::promoted_type_t<typename A::value_type, typename B::value_type>;

  // 1D x 1D: inner product (scalar)
  if constexpr (rankA == 1 && rankB == 1) {
    const auto &a_ext = a.extents();
    const auto &b_ext = b.extents();
    if (a_ext.extent(0) != b_ext.extent(0)) {
      NUMCXX_THROW(std::invalid_argument, "dot: incompatible shapes for 1D dot product");
    }
    value_type result = value_type{};
    for (size_type i = 0; i < a_ext.extent(0); ++i) {
      result += static_cast<value_type>(a[i]) *
                static_cast<value_type>(b[i]);
    }
    return result;
  }
  // 2D x 1D: matrix-vector product
  else if constexpr (rankA == 2 && rankB == 1) {
    const auto &a_ext = a.extents();
    const auto &b_ext = b.extents();
    if (a_ext.extent(1) != b_ext.extent(0)) {
      NUMCXX_THROW(std::invalid_argument, "dot: matrix columns must match vector size");
    }
    using extents_type = dextents<1>;
    ndarray<value_type, extents_type, default_layout> c(a_ext.extent(0));
    if constexpr (std::is_same_v<typename A::value_type, value_type> &&
                  std::is_same_v<typename B::value_type, value_type>) {
      detail::linalg::matrix_vector_product(a.to_mdspan(), b.to_mdspan(),
                                            c.to_mdspan());
    } else {
      for (size_type i = 0; i < a_ext.extent(0); ++i) {
        value_type sum{};
        for (size_type j = 0; j < a_ext.extent(1); ++j) {
          sum += static_cast<value_type>(a(i, j)) *
                 static_cast<value_type>(b[j]);
        }
        c[i] = sum;
      }
    }
    return c;
  }
  // 1D x 2D: vector-matrix product
  else if constexpr (rankA == 1 && rankB == 2) {
    const auto &a_ext = a.extents();
    const auto &b_ext = b.extents();
    if (a_ext.extent(0) != b_ext.extent(0)) {
      NUMCXX_THROW(std::invalid_argument, "dot: vector size must match matrix rows");
    }
    using extents_type = dextents<1>;
    ndarray<value_type, extents_type, default_layout> c(b_ext.extent(1));
    // For row vector x matrix: (1×M) x (M×N) = (1×N)
    // Use matrix-vector product on transposed perspective, or loop.
    // Simpler: implement directly.
    for (size_type j = 0; j < b_ext.extent(1); ++j) {
      value_type sum = value_type{};
      for (size_type i = 0; i < a_ext.extent(0); ++i) {
        sum += static_cast<value_type>(a[i]) *
               static_cast<value_type>(b(i, j));
      }
      c[j] = sum;
    }
    return c;
  }
  // 2D x 2D: matrix multiplication
  else if constexpr (rankA == 2 && rankB == 2) {
    return matmul(a, b);
  }
  else {
    static_assert((rankA == 1 || rankA == 2) && (rankB == 1 || rankB == 2),
                  "dot currently only supports 1D/2D combinations");
    // The static_assert above will fail; add a friendly message:
    NUMCXX_THROW(std::invalid_argument, "dot: unsupported rank combination");
  }
}

/// Solve A @ x = b using LU factorization with partial pivoting.
///
/// @tparam A Rank-2 ndarray or gslice_view with square extents
/// @tparam B Rank-1 or rank-2 ndarray or gslice_view
/// @param a Coefficient matrix of shape (M, M)
/// @param b Right-hand side of shape (M,) or (M, K)
/// @returns x with the same shape as b, in default_layout
/// @throws std::invalid_argument If the coefficient or RHS shapes are invalid
/// @throws std::runtime_error If the coefficient matrix is singular
template <class A, class B> auto solve(const A &a, const B &b) {
  static_assert(detail::is_solve_operand_v<A>,
                "solve: a must be an ndarray or gslice_view");
  static_assert(detail::is_solve_operand_v<B>,
                "solve: b must be an ndarray or gslice_view");
  static_assert(A::rank() == 2,
                "solve requires a rank-2 coefficient matrix");
  static_assert(B::rank() == 1 || B::rank() == 2,
                "solve requires a rank-1 or rank-2 right-hand side");

  using value_type =
      detail::promoted_type_t<typename A::value_type, typename B::value_type>;
  static_assert(detail::is_solve_value_v<value_type>,
                "solve supports float, double, complex<float>, and "
                "complex<double>");

  const size_type n = a.extent(0);
  if (a.extent(1) != n)
    NUMCXX_THROW(std::invalid_argument, "solve: a must be square");
  if (b.extent(0) != n)
    NUMCXX_THROW(std::invalid_argument,
                 "solve: a and b have incompatible shapes");

  const size_type nrhs = B::rank() == 1 ? 1 : b.extent(1);
  if (n == 0) {
    if constexpr (B::rank() == 1)
      return ndarray<value_type, dextents<1>>(n);
    else
      return ndarray<value_type, dextents<2>>(n, nrhs);
  }

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
  bool use_lapack =
      n <= static_cast<size_type>(std::numeric_limits<lapack_int>::max()) &&
      nrhs <= static_cast<size_type>(std::numeric_limits<lapack_int>::max());
  int lapack_order = LAPACK_ROW_MAJOR;
  if (use_lapack) {
    int a_order;
    if (!detail::lapack_matrix_order(a, a_order)) {
      use_lapack = false;
    } else if constexpr (B::rank() == 1) {
      if (!detail::lapack_vector_contiguous(b))
        use_lapack = false;
      else
        lapack_order = a_order;
    } else {
      int b_order;
      if (!detail::lapack_matrix_order(b, b_order) || a_order != b_order)
        use_lapack = false;
      else
        lapack_order = a_order;
    }
  }
#endif

  const bool column_major =
#ifdef NUMCXX_DETAIL_HAS_LAPACKE
      use_lapack && lapack_order == LAPACK_COL_MAJOR;
#else
      false;
#endif
  std::vector<value_type> a_copy(a.size());
  std::vector<value_type> b_copy(b.size());
  for (size_type i = 0; i < n; ++i) {
    for (size_type j = 0; j < n; ++j) {
      const size_type index = column_major ? j * n + i : i * n + j;
      a_copy[index] = static_cast<value_type>(a(i, j));
    }
    for (size_type j = 0; j < nrhs; ++j) {
      const size_type index = column_major ? j * n + i : i * nrhs + j;
      if constexpr (B::rank() == 1)
        b_copy[index] = static_cast<value_type>(b(i));
      else
        b_copy[index] = static_cast<value_type>(b(i, j));
    }
  }

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
  if (use_lapack) {
    std::vector<lapack_int> ipiv(n);
    const auto lapack_n = static_cast<lapack_int>(n);
    const auto lapack_nrhs = static_cast<lapack_int>(nrhs);
    const lapack_int lda = std::max<lapack_int>(1, lapack_n);
    const lapack_int ldb =
        lapack_order == LAPACK_ROW_MAJOR
            ? std::max<lapack_int>(1, lapack_nrhs)
            : std::max<lapack_int>(1, lapack_n);
    const lapack_int info =
        detail::lapack_gesv(lapack_order, lapack_n, lapack_nrhs,
                            a_copy.data(), lda, ipiv.data(), b_copy.data(),
                            ldb);
    if (info < 0) {
      const std::string message =
          "solve: illegal argument " + std::to_string(-info) +
          " to LAPACK gesv";
      NUMCXX_THROW(std::invalid_argument, message.c_str());
    }
    if (info > 0) {
      const std::string message =
          "solve: matrix is singular at pivot " + std::to_string(info);
      NUMCXX_THROW(std::runtime_error, message.c_str());
    }
  } else
#endif
  {
    detail::gaussian_solve(a_copy, b_copy, n, nrhs);
  }

  if constexpr (B::rank() == 1) {
    ndarray<value_type, dextents<1>> result(n);
    for (size_type i = 0; i < n; ++i)
      result(i) = b_copy[i];
    return result;
  } else {
    ndarray<value_type, dextents<2>> result(n, nrhs);
    for (size_type i = 0; i < n; ++i)
      for (size_type j = 0; j < nrhs; ++j) {
        const size_type index = column_major ? j * n + i : i * nrhs + j;
        result(i, j) = b_copy[index];
      }
    return result;
  }
}

/// Cholesky decomposition of a Hermitian positive-definite matrix.
///
/// @tparam A Rank-2 ndarray or gslice_view with square extents
/// @param a Input matrix of shape (M, M). Only the lower triangle when
///          upper is false, or the upper triangle when upper is true, and
///          the diagonal are read; no Hermitian check is performed.
/// @param upper If true, return U such that A = U^H * U; otherwise return
///              L such that A = L * L^H
/// @returns The triangular factor in a new ndarray, with the other triangle
///          zero-filled
/// @throws std::invalid_argument If the input matrix is not square
/// @throws std::runtime_error If the input matrix is not positive-definite
template <class A> auto cholesky(const A &a, bool upper = false) {
  static_assert(detail::is_solve_operand_v<A>,
                "cholesky: a must be an ndarray or gslice_view");
  static_assert(A::rank() == 2, "cholesky requires a rank-2 matrix");

  using value_type = typename A::value_type;
  static_assert(detail::is_solve_value_v<value_type>,
                "cholesky supports float, double, complex<float>, and "
                "complex<double>");

  const size_type n = a.extent(0);
  if (a.extent(1) != n)
    NUMCXX_THROW(std::invalid_argument, "cholesky: a must be square");
  ndarray<value_type, dextents<2>> result(n, n);
  if (n == 0)
    return result;

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
  bool use_lapack =
      n <= static_cast<size_type>(std::numeric_limits<lapack_int>::max());
  int lapack_order = LAPACK_ROW_MAJOR;
  if (use_lapack && !detail::lapack_matrix_order(a, lapack_order))
    use_lapack = false;
#endif

  const bool column_major =
#ifdef NUMCXX_DETAIL_HAS_LAPACKE
      use_lapack && lapack_order == LAPACK_COL_MAJOR;
#else
      false;
#endif
  std::vector<value_type> factor(a.size());
  for (size_type i = 0; i < n; ++i)
    for (size_type j = 0; j < n; ++j) {
      const size_type index = column_major ? j * n + i : i * n + j;
      factor[index] = static_cast<value_type>(a(i, j));
    }

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
  if (use_lapack) {
    const auto lapack_n = static_cast<lapack_int>(n);
    const lapack_int info = detail::lapack_potrf(
        lapack_order, upper ? 'U' : 'L', lapack_n, factor.data(), lapack_n);
    if (info < 0) {
      const std::string message =
          "cholesky: illegal argument " + std::to_string(-info) +
          " to LAPACK potrf";
      NUMCXX_THROW(std::invalid_argument, message.c_str());
    }
    if (info > 0) {
      const std::string message =
          "cholesky: matrix is not positive-definite at leading minor " +
          std::to_string(info);
      NUMCXX_THROW(std::runtime_error, message.c_str());
    }
  } else
#endif
  {
    detail::cholesky_fallback(factor, n, upper);
  }

  for (size_type i = 0; i < n; ++i)
    for (size_type j = 0; j < n; ++j) {
      const bool in_factor_triangle = upper ? i <= j : i >= j;
      if (in_factor_triangle) {
        const size_type index = column_major ? j * n + i : i * n + j;
        result(i, j) = factor[index];
      } else {
        result(i, j) = value_type{};
      }
    }
  return result;
}

namespace detail {
enum class qr_impl_mode { reduced, complete, r_only };

template <class A>
auto qr_impl(const A &a, qr_impl_mode mode) {
  static_assert(numcxx::detail::is_solve_operand_v<A>,
                "qr: a must be an ndarray or gslice_view");
  static_assert(A::rank() == 2, "qr requires a rank-2 matrix");

  using value_type = typename A::value_type;
  static_assert(numcxx::detail::is_solve_value_v<value_type>,
                "qr supports float, double, complex<float>, and "
                "complex<double>");

  const size_type m = a.extent(0);
  const size_type n = a.extent(1);
  const size_type k = std::min(m, n);
  const bool r_only = mode == qr_impl_mode::r_only;
  const bool complete = mode == qr_impl_mode::complete;
  const size_type q_columns = complete ? m : k;
  ndarray<value_type, dextents<2>> q(r_only ? 0 : m,
                                     r_only ? 0 : q_columns);
  ndarray<value_type, dextents<2>> r(complete ? m : k, n);

  if (m == 0 || n == 0) {
    if (complete && n == 0)
      for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < m; ++j)
          q(i, j) = i == j ? value_type{1} : value_type{};
    return std::make_pair(std::move(q), std::move(r));
  }

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
  bool use_lapack =
      m <= static_cast<size_type>(std::numeric_limits<lapack_int>::max()) &&
      n <= static_cast<size_type>(std::numeric_limits<lapack_int>::max());
  int lapack_order = LAPACK_ROW_MAJOR;
  if (use_lapack &&
      !numcxx::detail::lapack_matrix_order(a, lapack_order))
    use_lapack = false;
#endif

  const bool column_major =
#ifdef NUMCXX_DETAIL_HAS_LAPACKE
      use_lapack && lapack_order == LAPACK_COL_MAJOR;
#else
      false;
#endif
  std::vector<value_type> work(a.size());
  for (size_type i = 0; i < m; ++i)
    for (size_type j = 0; j < n; ++j) {
      const size_type index = column_major ? j * m + i : i * n + j;
      work[index] = static_cast<value_type>(a(i, j));
    }
  std::vector<value_type> tau(k);

#ifdef NUMCXX_DETAIL_HAS_LAPACKE
  if (use_lapack) {
    const auto lapack_m = static_cast<lapack_int>(m);
    const auto lapack_n = static_cast<lapack_int>(n);
    const auto lapack_k = static_cast<lapack_int>(k);
    const lapack_int lda = column_major ? std::max<lapack_int>(1, lapack_m)
                                        : std::max<lapack_int>(1, lapack_n);
    const lapack_int geqrf_info = numcxx::detail::lapack_geqrf(
        lapack_order, lapack_m, lapack_n, work.data(), lda, tau.data());
    if (geqrf_info < 0) {
      const std::string message =
          "qr: illegal argument " + std::to_string(-geqrf_info) +
          " to LAPACK geqrf";
      NUMCXX_THROW(std::invalid_argument, message.c_str());
    }

    for (size_type i = 0; i < r.extent(0); ++i)
      for (size_type j = 0; j < n; ++j)
        r(i, j) = i <= j ? work[column_major ? j * m + i : i * n + j]
                         : value_type{};

    if (!r_only) {
      std::vector<value_type> q_work(m * q_columns, value_type{});
      for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < k && j <= i; ++j) {
          const size_type source = column_major ? j * m + i : i * n + j;
          const size_type target =
              column_major ? j * m + i : i * q_columns + j;
          q_work[target] = work[source];
        }

      const lapack_int q_lda =
          column_major
              ? std::max<lapack_int>(1, lapack_m)
              : std::max<lapack_int>(1, static_cast<lapack_int>(q_columns));
      const lapack_int orgqr_info = numcxx::detail::lapack_orgqr(
          lapack_order, lapack_m, static_cast<lapack_int>(q_columns),
          lapack_k, q_work.data(), q_lda, tau.data());
      if (orgqr_info < 0) {
        const std::string message =
            "qr: illegal argument " + std::to_string(-orgqr_info) +
            " to LAPACK orgqr/ungqr";
        NUMCXX_THROW(std::invalid_argument, message.c_str());
      }

      for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < q_columns; ++j) {
          const size_type index =
              column_major ? j * m + i : i * q_columns + j;
          q(i, j) = q_work[index];
        }
    }
  } else
#endif
  {
    numcxx::detail::qr_fallback(work, m, n, tau);
    for (size_type i = 0; i < r.extent(0); ++i)
      for (size_type j = 0; j < n; ++j)
        r(i, j) = i <= j ? work[i * n + j] : value_type{};

    if (!r_only) {
      std::vector<value_type> q_work(m * q_columns, value_type{});
      numcxx::detail::form_q_fallback(work, tau, q_work, m, n, q_columns, k);
      for (size_type i = 0; i < m; ++i)
        for (size_type j = 0; j < q_columns; ++j)
          q(i, j) = q_work[i * q_columns + j];
    }
  }
  return std::make_pair(std::move(q), std::move(r));
}
} // namespace detail

/// Compute the reduced QR factorization A = Q * R.
///
/// Equivalent to `np.linalg.qr(a)` in NumPy. Q has shape (M, K) and R has
/// shape (K, N), where K = min(M, N).
template <class A> auto qr(const A &a) {
  return detail::qr_impl(a, detail::qr_impl_mode::reduced);
}

/// Compute the complete QR factorization A = Q * R.
///
/// Equivalent to `np.linalg.qr(a, mode="complete")` in NumPy. Q has shape
/// (M, M) and R has shape (M, N).
template <class A> auto qr_complete(const A &a) {
  return detail::qr_impl(a, detail::qr_impl_mode::complete);
}

/// Compute only the R factor without constructing Q.
///
/// Equivalent to `np.linalg.qr(a, mode="r")` in NumPy. R has shape (K, N),
/// where K = min(M, N).
template <class A> auto qr_r(const A &a) {
  auto result = detail::qr_impl(a, detail::qr_impl_mode::r_only);
  return std::move(result.second);
}

} // namespace linalg

//
// [numcxx.aliases] type aliases
//
// clang-format off

// ============================================
// Dynamic array aliases (runtime extents, default layout)
// ============================================
template <class T>                                        using  vec       = ndarray<T, dextents<1>>     ;
template <class T>                                        using  mat       = ndarray<T, dextents<2>>     ;
template <class T>                                        using cube       = ndarray<T, dextents<3>>     ;

// ============================================
// Static array aliases (compile-time extents, default layout)
// ============================================
template <class T, size_type N>                           using  vec_fixed = ndarray<T, extents<N>>      ;
template <class T, size_type M, size_type N>              using  mat_fixed = ndarray<T, extents<M, N>>   ;
template <class T, size_type M, size_type N, size_type K> using cube_fixed = ndarray<T, extents<M, N, K>>;


// ============================================
// Typed aliases (Armadillo-style)
// ============================================

// -------- dynamic --------
// int
using ivec     =  vec<int>     ;
using imat     =  mat<int>     ;
using icube    = cube<int>     ;

// Logical arrays store 0/1 bytes to avoid vector<bool>'s proxy references.
using bvec     =  vec<unsigned char>;
using bmat     =  mat<unsigned char>;
using bcube    = cube<unsigned char>;

// unsigned int
using uvec     =  vec<unsigned>;
using umat     =  mat<unsigned>;
using ucube    = cube<unsigned>;

// double
using dvec     =  vec<double>  ;
using dmat     =  mat<double>  ; 
using dcube    = cube<double>  ;

// float
using fvec     =  vec<float>   ;
using fmat     =  mat<float>   ;
using fcube    = cube<float>   ;

// complex
using cx_float  = std::complex<float> ;
using cx_double = std::complex<double>;
using cx_fvec   = vec<cx_float>       ;
using cx_fmat   = mat<cx_float>       ;
using cx_fcube  = cube<cx_float>      ;
using cx_dvec   = vec<cx_double>      ;
using cx_dmat   = mat<cx_double>      ;
using cx_dcube  = cube<cx_double>    ;
using cx_vec    = cx_dvec             ;
using cx_mat    = cx_dmat             ;
using cx_cube   = cx_dcube            ;

// -------- fixed-size --------
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