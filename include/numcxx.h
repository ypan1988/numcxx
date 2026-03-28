#ifndef NUMCXX_H_
#define NUMCXX_H_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <memory>
#include <type_traits>
#include <utility>
//#  include <__memory/addressof.h>
//#  include <__memory/uninitialized_algorithms.h>
//#  include <__utility/exception_guard.h>

#include <mdspan/mdarray.hpp>

namespace numcxx::detail {
    using Kokkos::Experimental::mdarray;
    using Kokkos::extents;
    using Kokkos::dextents;
    using Kokkos::layout_right;
    using Kokkos::layout_left;
}

namespace numcxx {

template <
    class ElementType,
    class Extents,
    class LayoutPolicy
>
class ndarray;

class slice {
    size_t start_;
    size_t size_;
    size_t stride_;

public:
    slice() : start_(0), size_(0), stride_(0) {}

    slice(size_t __start, size_t __size, size_t __stride)
        : start_(__start), size_(__size), stride_(__stride) {
    }

    [[nodiscard]] size_t start() const { return start_; }
    [[nodiscard]] size_t size() const { return size_; }
    [[nodiscard]] size_t stride() const { return stride_; }

    friend bool operator==(const slice& x, const slice& y) {
        return x.start() == y.start() && x.size() == y.size() && x.stride() == y.stride();
    }
};

template <class Tp>
class slice_array;
template <class Tp>
class mask_array;
template <class Tp>
class indirect_array;

template <class Tp, class Ex, class Lp>
Tp* begin(ndarray<Tp, Ex, Lp>& v);

template <class Tp, class Ex, class Lp>
const Tp* begin(const ndarray<Tp, Ex, Lp>& v);

template <class Tp, class Ex, class Lp>
Tp* end(ndarray<Tp, Ex, Lp>& v);

template <class Tp, class Ex, class Lp>
const Tp* end(const ndarray<Tp, Ex, Lp>& v);

template <class Op, class A0>
struct UnaryOp {
    typedef typename Op::result_type result_type;
    using value_type = std::decay_t<result_type>;

    Op op_;
    A0 a0_;

    UnaryOp(const Op& op, const A0& a0) : op_(op), a0_(a0) {}

    result_type operator[](size_t i) const { return op_(a0_[i]); }

    size_t size() const { return a0_.size(); }
};

template <class Op, class A0, class A1>
struct BinaryOp {
    typedef typename Op::result_type result_type;
    using value_type = std::decay_t<result_type>;

    Op op_;
    A0 a0_;
    A1 a1_;

    BinaryOp(const Op& op, const A0& a0, const A1& a1)
        : op_(op), a0_(a0), a1_(a1) {
    }

    result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

    size_t size() const { return a0_.size(); }
};

template <class Tp>
class __scalar_expr {
public:
    typedef Tp value_type;
    typedef const Tp& result_type;

private:
    const value_type& t_;
    size_t s_;

public:
    explicit __scalar_expr(const value_type& t, size_t s) : t_(t), s_(s) {}

    result_type operator[](size_t) const { return t_; }

    size_t size() const { return s_; }
};

template <class Tp>
struct __unary_plus {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return +x; }
};

template <class Tp>
struct __bit_not {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return ~x; }
};

template <class Tp>
struct __bit_shift_left {
    typedef Tp result_type;
    Tp operator()(const Tp& x, const Tp& y) const { return x << y; }
};

template <class Tp>
struct __bit_shift_right {
    typedef Tp result_type;
    Tp operator()(const Tp& x, const Tp& y) const { return x >> y; }
};

template <class Tp, class Fp>
struct __apply_expr {
private:
    Fp __f_;

public:
    typedef Tp result_type;

    explicit __apply_expr(Fp __f) : __f_(__f) {}

    Tp operator()(const Tp& x) const { return __f_(x); }
};

template <class Tp>
struct __abs_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::abs(x); }
};

template <class Tp>
struct __acos_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::acos(x); }
};

template <class Tp>
struct __asin_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::asin(x); }
};

template <class Tp>
struct __atan_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::atan(x); }
};

template <class Tp>
struct __atan2_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x, const Tp& y) const { return std::atan2(x, y); }
};

template <class Tp>
struct __cos_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::cos(x); }
};

template <class Tp>
struct __cosh_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::cosh(x); }
};

template <class Tp>
struct exp_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::exp(x); }
};

template <class Tp>
struct __log_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::log(x); }
};

template <class Tp>
struct __log10_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::log10(x); }
};

template <class Tp>
struct __pow_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x, const Tp& y) const { return std::pow(x, y); }
};

template <class Tp>
struct __sin_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::sin(x); }
};

template <class Tp>
struct __sinh_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::sinh(x); }
};

template <class Tp>
struct __sqrt_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::sqrt(x); }
};

template <class Tp>
struct __tan_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::tan(x); }
};

template <class Tp>
struct __tanh_expr {
    typedef Tp result_type;
    Tp operator()(const Tp& x) const { return std::tanh(x); }
};

template <class ValExpr>
class __slice_expr {
    typedef std::remove_reference_t<ValExpr> _RmExpr;

public:
    typedef typename _RmExpr::value_type value_type;
    typedef value_type result_type;

private:
    ValExpr expr_;
    size_t start_;
    size_t size_;
    size_t stride_;

    __slice_expr(const slice& __sl, const _RmExpr& e)
        : expr_(e), start_(__sl.start()), size_(__sl.size()), stride_(__sl.stride()) {
    }

public:
    result_type operator[](size_t i) const { return expr_[start_ + i * stride_]; }

    size_t size() const { return size_; }

    template <class>
    friend class nc_val_expr;
    template <class, class, class>
    friend class ndarray;
};

template <class ValExpr>
class __mask_expr;

template <class ValExpr>
class __indirect_expr;

template <class ValExpr>
class __shift_expr {
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
    static const ptrdiff_t _Np = static_cast<ptrdiff_t>(sizeof(ptrdiff_t) * __CHAR_BIT__ - 1);

    __shift_expr(int n, const _RmExpr& e) : expr_(e), size_(e.size()), __n_(n) {
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

    template <class>
    friend class nc_val_expr;
};

template <class ValExpr>
class __cshift_expr {
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

    __cshift_expr(int n, const _RmExpr& e) : expr_(e), size_(e.size()) {
        n %= static_cast<int>(size_);
        if (n >= 0) {
            m_ = size_ - n;
            o1_ = n;
            o2_ = n - size_;
        }
        else {
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

    template <class>
    friend class nc_val_expr;
};

template <class ValExpr>
class nc_val_expr;

template <class ValExpr>
struct nc_is_val_expr : false_type {};

template <class ValExpr>
struct nc_is_val_expr<nc_val_expr<ValExpr> > : true_type {};

template <class Tp, class Ex, class Lp>
struct nc_is_val_expr<ndarray<Tp, Ex, Lp> > : true_type {};

template <class Tp>
struct nc_is_val_expr<slice_array<Tp> > : true_type {};

template <class Tp>
struct nc_is_val_expr<mask_array<Tp> > : true_type {};

template <class Tp>
struct nc_is_val_expr<indirect_array<Tp> > : true_type {};

// The functions using a nc_val_expr access the elements by their index.
// ndarray and the libc++ lazy proxies have an operator[]. The
// Standard proxy array's don't have this operator, instead they have a
// implementation specific accessor
//   __get(size_t)
//
// The functions use the non-member function
//   __get(nc_val_expr, size_t)
//
// If the nc_val_expr is a specialization of nc_val_expr_use_member_functions it
// uses the nc_val_expr's member function
//   __get(size_t)
// else it uses the nc_val_expr's member function
//   operator[](size_t)
template <class ValExpr>
struct nc_val_expr_use_member_functions;

template <class>
struct nc_val_expr_use_member_functions : false_type {};

template <class Tp>
struct nc_val_expr_use_member_functions<slice_array<Tp> > : true_type {};

template <class Tp>
struct nc_val_expr_use_member_functions<mask_array<Tp> > : true_type {};

template <class Tp>
struct nc_val_expr_use_member_functions<indirect_array<Tp> > : true_type {};

template <
    class ElementType,
    class Extents,
    class LayoutPolicy = detail::layout_right
>
class ndarray {
public:
    using value_type = ElementType;
    using result_type = ElementType;
    using extents_type = Extents;
    using layout_type = LayoutPolicy;

    using pointer = ElementType*;
    using const_pointer = const ElementType*;
    using reference = ElementType&;
    using const_reference = const ElementType&;

private:
    detail::mdarray<ElementType, Extents, LayoutPolicy, std::vector<ElementType>> elem_;

    value_type* begin_;
    value_type* end_;

public:
    constexpr ndarray() = default;
    constexpr ndarray(const ndarray& v) = default;
    constexpr ndarray(ndarray&& v) noexcept = default;

    template<class... SizeTypes>
    explicit constexpr ndarray(SizeTypes... dyn_exts)
        : elem_(Extents(dyn_exts...)) {
    }

    constexpr ndarray& operator=(const ndarray& v) = default;
    constexpr ndarray& operator=(ndarray&& v) noexcept = default;

    ~ndarray() = default;

    template<class... SizeTypes>
    constexpr reference operator()(SizeTypes... idxs) noexcept {
        return elem_(idxs...);
    }

    template<class... SizeTypes>
    constexpr const_reference operator()(SizeTypes... idxs) const noexcept {
        return elem_(idxs...);
    }

    constexpr pointer data() noexcept { return elem_.data(); }
    constexpr const_pointer data() const noexcept { return elem_.data(); }

    constexpr const Extents& extents() const noexcept { return elem_.extents(); }
    constexpr size_t extent(size_t r) const noexcept { return elem_.extent(r); }

    // construct/destroy:
    // ndarray() : begin_(nullptr), end_(nullptr) {}
    inline explicit ndarray(size_t n);
    ndarray(const value_type& x, size_t n);
    ndarray(const value_type* p, size_t n);
    //ndarray(const ndarray& v);
    //ndarray(ndarray&& v) noexcept;
    ndarray(std::initializer_list<value_type> __il);
    ndarray(const slice_array<value_type>& sa);
    ndarray(const mask_array<value_type>& ma);
    ndarray(const indirect_array<value_type>& ia);
    //inline ~ndarray();

    // assignment:
    //ndarray& operator=(const ndarray& v);
    //ndarray& operator=(ndarray&& v) noexcept;
    ndarray& operator=(std::initializer_list<value_type>);
    ndarray& operator=(const value_type& x);
    ndarray& operator=(const slice_array<value_type>& sa);
    ndarray& operator=(const mask_array<value_type>& ma);
    ndarray& operator=(const indirect_array<value_type>& ia);
    template <class ValExpr>
    ndarray& operator=(const nc_val_expr<ValExpr>& v);

    // element access:
    [[nodiscard]] const value_type& operator[](size_t i) const {
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < size(), "ndarray::operator[] index out of bounds");
        return begin_[i];
    }

    [[nodiscard]] value_type& operator[](size_t i) {
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < size(), "ndarray::operator[] index out of bounds");
        return begin_[i];
    }

    // subset operations:
    [[nodiscard]] nc_val_expr<__slice_expr<const ndarray&> > operator[](slice s) const;
    [[nodiscard]] slice_array<value_type> operator[](slice s);
    [[nodiscard]]
    nc_val_expr<__mask_expr<const ndarray&> > operator[](const ndarray<bool, Extents, LayoutPolicy>& __vb) const;
    [[nodiscard]] mask_array<value_type> operator[](const ndarray<bool, Extents, LayoutPolicy>& __vb);
    [[nodiscard]]
        nc_val_expr<__mask_expr<const ndarray&> > operator[](ndarray<bool, Extents, LayoutPolicy>&& __vb) const;
    [[nodiscard]] mask_array<value_type> operator[](ndarray<bool, Extents, LayoutPolicy>&& __vb);
    [[nodiscard]]
    nc_val_expr<__indirect_expr<const ndarray&> > operator[](const ndarray<size_t, Extents, LayoutPolicy>& __vs) const;
    [[nodiscard]] indirect_array<value_type> operator[](const ndarray<size_t, Extents, LayoutPolicy>& __vs);
    [[nodiscard]]
        nc_val_expr<__indirect_expr<const ndarray&> > operator[](ndarray<size_t, Extents, LayoutPolicy>&& __vs) const;
    [[nodiscard]] indirect_array<value_type> operator[](ndarray<size_t, Extents, LayoutPolicy>&& __vs);

    // unary operators:
    nc_val_expr<UnaryOp<__unary_plus<ElementType>, const ndarray&> > operator+() const;
    nc_val_expr<UnaryOp<std::negate<ElementType>, const ndarray&> > operator-() const;
    nc_val_expr<UnaryOp<__bit_not<ElementType>, const ndarray&> > operator~() const;
    nc_val_expr<UnaryOp<std::logical_not<ElementType>, const ndarray&> > operator!() const;

    // computed assignment:
    ndarray& operator*=(const value_type& x);
    ndarray& operator/=(const value_type& x);
    ndarray& operator%=(const value_type& x);
    ndarray& operator+=(const value_type& x);
    ndarray& operator-=(const value_type& x);
    ndarray& operator^=(const value_type& x);
    ndarray& operator&=(const value_type& x);
    ndarray& operator|=(const value_type& x);
    ndarray& operator<<=(const value_type& x);
    ndarray& operator>>=(const value_type& x);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator*=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator/=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator%=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator+=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator-=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator^=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator|=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator&=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator<<=(const Expr& v);

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    ndarray& operator>>=(const Expr& v);

    // member functions:
    void swap(ndarray& v) noexcept;

    [[nodiscard]] size_t size() const { return static_cast<size_t>(end_ - begin_); }

    [[nodiscard]] value_type sum() const;
    [[nodiscard]] value_type min() const;
    [[nodiscard]] value_type max() const;

    [[nodiscard]] ndarray shift(int i) const;
    [[nodiscard]] ndarray cshift(int i) const;
    [[nodiscard]] ndarray apply(value_type __f(value_type)) const;
    [[nodiscard]] ndarray apply(value_type __f(const value_type&)) const;
    void resize(size_t n, value_type x = value_type());

private:
    template <class, class, class>
    friend class ndarray;
    template <class>
    friend class slice_array;
    template <class>
    friend class mask_array;
    template <class>
    friend class __mask_expr;
    template <class>
    friend class indirect_array;
    template <class>
    friend class __indirect_expr;
    template <class>
    friend class nc_val_expr;

    template <class Up, class Ex, class Lp>
    friend Up* begin(ndarray<Up, Ex, Lp>& v);

    template <class Up, class Ex, class Lp>
    friend const Up* begin(const ndarray<Up, Ex, Lp>& v);

    template <class Up, class Ex, class Lp>
    friend Up* end(ndarray<Up, Ex, Lp>& v);

    template <class Up, class Ex, class Lp>
    friend const Up* end(const ndarray<Up, Ex, Lp>& v);

    void __clear(size_t capacity);
    ndarray& __assign_range(const value_type* __f, const value_type* __l);
};

//template <class Tp, size_t _Size>
//ndarray(const Tp(&)[_Size], size_t) -> ndarray<Tp>;

template <class Expr,
    std::enable_if_t<nc_is_val_expr<Expr>::value&& nc_val_expr_use_member_functions<Expr>::value, int> = 0>
typename Expr::value_type __get(const Expr& v, size_t i) {
    return v.__get(i);
}

template <class Expr,
    std::enable_if_t<nc_is_val_expr<Expr>::value && !nc_val_expr_use_member_functions<Expr>::value, int> = 0>
typename Expr::value_type __get(const Expr& v, size_t i) {
    return v[i];
}

//extern template void ndarray<size_t>::resize(size_t, size_t);

template <class Op, class Tp, class Ex, class Lp>
struct UnaryOp<Op, ndarray<Tp, Ex, Lp> > {
    typedef typename Op::result_type result_type;
    using value_type = std::decay_t<result_type>;

    Op op_;
    const ndarray<Tp, Ex, Lp>& a0_;

    UnaryOp(const Op& op, const ndarray<Tp, Ex, Lp>& a0) : op_(op), a0_(a0) {}

    result_type operator[](size_t i) const { return op_(a0_[i]); }

    size_t size() const { return a0_.size(); }
};

template <class Op, class Tp, class Ex, class Lp, class A1>
struct BinaryOp<Op, ndarray<Tp, Ex, Lp>, A1> {
    typedef typename Op::result_type result_type;
    using value_type = std::decay_t<result_type>;

    Op op_;
    const ndarray<Tp, Ex, Lp>& a0_;
    A1 a1_;

    BinaryOp(const Op& op, const ndarray<Tp, Ex, Lp>& a0, const A1& a1)
        : op_(op), a0_(a0), a1_(a1) {
    }

    result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

    size_t size() const { return a0_.size(); }
};

template <class Op, class A0, class Tp, class Ex, class Lp>
struct BinaryOp<Op, A0, ndarray<Tp, Ex, Lp> > {
    typedef typename Op::result_type result_type;
    using value_type = std::decay_t<result_type>;

    Op op_;
    A0 a0_;
    const ndarray<Tp, Ex, Lp>& a1_;

    BinaryOp(const Op& op, const A0& a0, const ndarray<Tp, Ex, Lp>& a1)
        : op_(op), a0_(a0), a1_(a1) {
    }

    result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

    size_t size() const { return a0_.size(); }
};

template <class Op,
    class Tp1, class Ex1, class Lp1,
    class Tp2, class Ex2, class Lp2
>
struct BinaryOp<Op,
    ndarray<Tp1, Ex1, Lp1>,
    ndarray<Tp2, Ex2, Lp2>
> {
    typedef typename Op::result_type result_type;
    using value_type = std::decay_t<result_type>;

    Op op_;
    const ndarray<Tp1, Ex1, Lp1>& a0_;
    const ndarray<Tp2, Ex2, Lp2>& a1_;

    BinaryOp(const Op& op, const ndarray<Tp1, Ex1, Lp1>& a0, const ndarray<Tp2, Ex2, Lp2>& a1)
        : op_(op), a0_(a0), a1_(a1) {
    }

    result_type operator[](size_t i) const { return op_(a0_[i], a1_[i]); }

    size_t size() const { return a0_.size(); }
};

// slice_array

template <class Tp>
class slice_array {
public:
    typedef Tp value_type;

private:
    value_type* vp_;
    size_t size_;
    size_t stride_;

public:
    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator*=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator/=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator%=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator+=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator-=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator^=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator&=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator|=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator<<=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator>>=(const Expr& v) const;

    slice_array(slice_array const&) = default;

    const slice_array& operator=(const slice_array& sa) const;

    void operator=(const value_type& x) const;

    template <class Ex, class Lp>
    void operator=(const ndarray<value_type, Ex, Lp>& __va) const;

    // Behaves like nc_val_expr::operator[], which returns by value.
    value_type __get(size_t i) const {
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < size_, "slice_array.__get() index out of bounds");
        return vp_[i * stride_];
    }

private:
    template <class Ex, class Lp>
    slice_array(const slice& __sl, const ndarray<value_type, Ex, Lp>& v)
        : vp_(const_cast<value_type*>(v.begin_ + __sl.start())), size_(__sl.size()), stride_(__sl.stride()) {
    }

    template <class, class, class>
    friend class ndarray;
};

template <class Tp>
inline const slice_array<Tp>& slice_array<Tp>::operator=(const slice_array& sa) const {
    value_type* t = vp_;
    const value_type* s = sa.vp_;
    for (size_t n = size_; n; --n, t += stride_, s += sa.stride_)
        *t = *s;
    return *this;
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t = v[i];
}

template <class Tp>
template <class Ex, class Lp>
inline void slice_array<Tp>::operator=(const ndarray<value_type, Ex, Lp>& __va) const {
    value_type* t = vp_;
    for (size_t i = 0; i < __va.size(); ++i, t += stride_)
        *t = __va[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator*=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t *= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator/=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t /= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator%=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t %= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator+=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t += v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator-=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t -= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator^=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t ^= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator&=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t &= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator|=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t |= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator<<=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t <<= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void slice_array<Tp>::operator>>=(const Expr& v) const {
    value_type* t = vp_;
    for (size_t i = 0; i < size_; ++i, t += stride_)
        *t >>= v[i];
}

template <class Tp>
inline void slice_array<Tp>::operator=(const value_type& x) const {
    value_type* t = vp_;
    for (size_t n = size_; n; --n, t += stride_)
        *t = x;
}

// mask_array

template <class Tp>
class mask_array {
public:
    typedef Tp value_type;

private:
    value_type* vp_;
    detail::mdarray<size_t, detail::dextents<std::size_t,1>, detail::layout_right> oned_;

public:
    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator*=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator/=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator%=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator+=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator-=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator^=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator&=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator|=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator<<=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator>>=(const Expr& v) const;

    mask_array(const mask_array&) = default;

    const mask_array& operator=(const mask_array& ma) const;

    void operator=(const value_type& x) const;

    // Behaves like nc_val_expr::operator[], which returns by value.
    value_type __get(size_t i) const {
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < oned_.size(), "mask_array.__get() index out of bounds");
        return vp_[oned_[i]];
    }

private:
    template <class Ex, class Lp>
    mask_array(const ndarray<bool, Ex, Lp>& __vb, const ndarray<value_type, Ex, Lp>& v)
        : vp_(const_cast<value_type*>(v.begin_)),
        oned_(static_cast<size_t>(count(__vb.begin_, __vb.end_, true))) {
        size_t j = 0;
        for (size_t i = 0; i < __vb.size(); ++i)
            if (__vb[i])
                oned_[j++] = i;
    }

    template <class, class, class>
    friend class ndarray;
};

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] = v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator*=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] *= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator/=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] /= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator%=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] %= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator+=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] += v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator-=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] -= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator^=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] ^= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator&=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] &= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator|=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] |= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator<<=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] <<= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void mask_array<Tp>::operator>>=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] >>= v[i];
}

template <class Tp>
inline const mask_array<Tp>& mask_array<Tp>::operator=(const mask_array& ma) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] = ma.vp_[oned_[i]];
    return *this;
}

template <class Tp>
inline void mask_array<Tp>::operator=(const value_type& x) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] = x;
}

template <class ValExpr>
class __mask_expr {
    typedef std::remove_reference_t<ValExpr> _RmExpr;

public:
    typedef typename _RmExpr::value_type value_type;
    typedef value_type result_type;

private:
    ValExpr expr_;
    ndarray<size_t, detail::dextents<std::size_t, 1>, detail::layout_right> oned_;

    __mask_expr(const ndarray<bool, detail::dextents<std::size_t, 1>, detail::layout_right>& __vb, const _RmExpr& e)
        : expr_(e), oned_(static_cast<size_t>(count(__vb.begin_, __vb.end_, true))) {
        size_t j = 0;
        for (size_t i = 0; i < __vb.size(); ++i)
            if (__vb[i])
                oned_[j++] = i;
    }

public:
    result_type operator[](size_t i) const { return expr_[oned_[i]]; }

    size_t size() const { return oned_.size(); }

    template <class>
    friend class nc_val_expr;
    template <class, class, class>
    friend class ndarray;
};

// indirect_array

template <class Tp>
class indirect_array {
public:
    typedef Tp value_type;

private:
    value_type* vp_;
    ndarray<size_t, detail::dextents<std::size_t, 1>, detail::layout_right> oned_;

public:
    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator*=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator/=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator%=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator+=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator-=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator^=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator&=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator|=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator<<=(const Expr& v) const;

    template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
    void operator>>=(const Expr& v) const;

    indirect_array(const indirect_array&) = default;

    const indirect_array& operator=(const indirect_array& ia) const;

    void operator=(const value_type& x) const;

    // Behaves like nc_val_expr::operator[], which returns by value.
    value_type __get(size_t i) const {
        _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(i < oned_.size(), "indirect_array.__get() index out of bounds");
        return vp_[oned_[i]];
    }

private:
    template <class Ex1, class Lp1, class Ex2, class Lp2>
    indirect_array(const ndarray<size_t, Ex1, Lp1>& ia, const ndarray<value_type, Ex2, Lp2>& v)
        : vp_(const_cast<value_type*>(v.begin_)), oned_(ia) {
    }

    template <class Ex1, class Lp1, class Ex2, class Lp2>
    indirect_array(ndarray<size_t, Ex1, Lp1>&& ia, const ndarray<value_type, Ex2, Lp2>& v)
        : vp_(const_cast<value_type*>(v.begin_)), oned_(std::move(ia)) {
    }

    template <class, class, class>
    friend class ndarray;
};

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] = v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator*=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] *= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator/=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] /= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator%=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] %= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator+=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] += v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator-=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] -= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator^=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] ^= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator&=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] &= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator|=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] |= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator<<=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] <<= v[i];
}

template <class Tp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline void indirect_array<Tp>::operator>>=(const Expr& v) const {
    size_t n = oned_.size();
    for (size_t i = 0; i < n; ++i)
        vp_[oned_[i]] >>= v[i];
}

template <class Tp>
inline const indirect_array<Tp>& indirect_array<Tp>::operator=(const indirect_array& ia) const {
    typedef const size_t* _Ip;
    const value_type* s = ia.vp_;
    for (_Ip i = oned_.begin_, e = oned_.end_, j = ia.oned_.begin_; i != e; ++i, ++j)
        vp_[*i] = s[*j];
    return *this;
}

template <class Tp>
inline void indirect_array<Tp>::operator=(const value_type& x) const {
    typedef const size_t* _Ip;
    for (_Ip i = oned_.begin_, e = oned_.end_; i != e; ++i)
        vp_[*i] = x;
}

template <class ValExpr>
class __indirect_expr {
    typedef std::remove_reference_t<ValExpr> _RmExpr;

public:
    typedef typename _RmExpr::value_type value_type;
    typedef value_type result_type;

private:
    ValExpr expr_;
    ndarray<size_t, detail::dextents<std::size_t, 1>, detail::layout_right> oned_;

    template <class Ex, class Lp>
    __indirect_expr(const ndarray<size_t, Ex, Lp>& ia, const _RmExpr& e) : expr_(e), oned_(ia) {}

    template <class Ex, class Lp>
    __indirect_expr(ndarray<size_t, Ex, Lp>&& ia, const _RmExpr& e)
        : expr_(e), oned_(std::move(ia)) {
    }

public:
    result_type operator[](size_t i) const { return expr_[oned_[i]]; }

    size_t size() const { return oned_.size(); }

    template <class>
    friend class nc_val_expr;
    template <class, class, class>
    friend class ndarray;
};

template <class ValExpr>
class nc_val_expr {
    typedef std::remove_reference_t<ValExpr> _RmExpr;

    ValExpr expr_;

public:
    typedef typename _RmExpr::value_type value_type;
    typedef typename _RmExpr::result_type result_type;

    explicit nc_val_expr(const _RmExpr& e) : expr_(e) {}

    result_type operator[](size_t i) const { return expr_[i]; }

    nc_val_expr<__slice_expr<ValExpr> > operator[](slice s) const {
        typedef __slice_expr<ValExpr> _NewExpr;
        return nc_val_expr< _NewExpr >(_NewExpr(s, expr_));
    }

    template <class Ex, class Lp>
    nc_val_expr<__mask_expr<ValExpr> > operator[](const ndarray<bool, Ex, Lp>& __vb) const {
        typedef __mask_expr<ValExpr> _NewExpr;
        return nc_val_expr< _NewExpr >(_NewExpr(__vb, expr_));
    }

    template <class Ex, class Lp>
    nc_val_expr<__indirect_expr<ValExpr> > operator[](const ndarray<size_t, Ex, Lp>& __vs) const {
        typedef __indirect_expr<ValExpr> _NewExpr;
        return nc_val_expr< _NewExpr >(_NewExpr(__vs, expr_));
    }

    nc_val_expr<UnaryOp<__unary_plus<value_type>, ValExpr> > operator+() const {
        typedef UnaryOp<__unary_plus<value_type>, ValExpr> _NewExpr;
        return nc_val_expr<_NewExpr>(_NewExpr(__unary_plus<value_type>(), expr_));
    }

    nc_val_expr<UnaryOp<std::negate<value_type>, ValExpr> > operator-() const {
        typedef UnaryOp<std::negate<value_type>, ValExpr> _NewExpr;
        return nc_val_expr<_NewExpr>(_NewExpr(std::negate<value_type>(), expr_));
    }

    nc_val_expr<UnaryOp<__bit_not<value_type>, ValExpr> > operator~() const {
        typedef UnaryOp<__bit_not<value_type>, ValExpr> _NewExpr;
        return nc_val_expr<_NewExpr>(_NewExpr(__bit_not<value_type>(), expr_));
    }

    nc_val_expr<UnaryOp<std::logical_not<value_type>, ValExpr> > operator!() const {
        typedef UnaryOp<std::logical_not<value_type>, ValExpr> _NewExpr;
        return nc_val_expr<_NewExpr>(_NewExpr(std::logical_not<value_type>(), expr_));
    }

    template<class Ex, class Lp>
    operator ndarray<nc_val_expr::result_type, Ex, Lp>() const;

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

    nc_val_expr<__shift_expr<ValExpr> > shift(int i) const {
        return nc_val_expr<__shift_expr<ValExpr> >(__shift_expr<ValExpr>(i, expr_));
    }

    nc_val_expr<__cshift_expr<ValExpr> > cshift(int i) const {
        return nc_val_expr<__cshift_expr<ValExpr> >(__cshift_expr<ValExpr>(i, expr_));
    }

    nc_val_expr<UnaryOp<__apply_expr<value_type, value_type(*)(value_type)>, ValExpr> >
        apply(value_type __f(value_type)) const {
        typedef __apply_expr<value_type, value_type(*)(value_type)> Op;
        typedef UnaryOp<Op, ValExpr> _NewExpr;
        return nc_val_expr<_NewExpr>(_NewExpr(Op(__f), expr_));
    }

    nc_val_expr<UnaryOp<__apply_expr<value_type, value_type(*)(const value_type&)>, ValExpr> >
        apply(value_type __f(const value_type&)) const {
        typedef __apply_expr<value_type, value_type(*)(const value_type&)> Op;
        typedef UnaryOp<Op, ValExpr> _NewExpr;
        return nc_val_expr<_NewExpr>(_NewExpr(Op(__f), expr_));
    }
};

template <class ValExpr>
template <class Ex, class Lp>
nc_val_expr<ValExpr>::operator ndarray<nc_val_expr::result_type, Ex, Lp>() const {
    ndarray<result_type> r;
    size_t n = expr_.size();
    if (n) {
        r.begin_ = r.end_ = allocator<result_type>().allocate(n);
        for (size_t i = 0; i != n; ++r.end_, ++i)
            ::new ((void*)r.end_) result_type(expr_[i]);
    }
    return r;
}

// ndarray

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>::ndarray(size_t n) : begin_(nullptr), end_(nullptr) {
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        for (size_t __n_left = n; __n_left; --__n_left, ++end_)
            ::new ((void*)end_) value_type();
        __guard.__complete();
    }
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>::ndarray(const value_type& x, size_t n) : begin_(nullptr), end_(nullptr) {
    resize(n, x);
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>::ndarray(const value_type* p, size_t n) : begin_(nullptr), end_(nullptr) {
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        for (size_t __n_left = n; __n_left; ++end_, ++p, --__n_left)
            ::new ((void*)end_) value_type(*p);
        __guard.__complete();
    }
}

//template <class Tp, class Ex, class Lp>
//ndarray<Tp, Ex, Lp>::ndarray(const ndarray& v) : begin_(nullptr), end_(nullptr) {
//    if (v.size()) {
//        begin_ = end_ = allocator<value_type>().allocate(v.size());
//        auto __guard = std::__make_exception_guard([&] { __clear(v.size()); });
//        for (value_type* p = v.begin_; p != v.end_; ++end_, ++p)
//            ::new ((void*)end_) value_type(*p);
//        __guard.__complete();
//    }
//}

//template <class Tp, class Ex, class Lp>
//inline ndarray<Tp, Ex, Lp>::ndarray(ndarray&& v) noexcept : begin_(v.begin_), end_(v.end_) {
//    v.begin_ = v.end_ = nullptr;
//}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>::ndarray(std::initializer_list<value_type> __il) : begin_(nullptr), end_(nullptr) {
    const size_t n = __il.size();
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        size_t __n_left = n;
        for (const value_type* p = __il.begin(); __n_left; ++end_, ++p, --__n_left)
            ::new ((void*)end_) value_type(*p);
        __guard.__complete();
    }
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>::ndarray(const slice_array<value_type>& sa) : begin_(nullptr), end_(nullptr) {
    const size_t n = sa.size_;
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        size_t __n_left = n;
        for (const value_type* p = sa.vp_; __n_left; ++end_, p += sa.stride_, --__n_left)
            ::new ((void*)end_) value_type(*p);
        __guard.__complete();
    }
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>::ndarray(const mask_array<value_type>& ma) : begin_(nullptr), end_(nullptr) {
    const size_t n = ma.oned_.size();
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        typedef const size_t* _Ip;
        const value_type* s = ma.vp_;
        for (_Ip i = ma.oned_.begin_, e = ma.oned_.end_; i != e; ++i, ++end_)
            ::new ((void*)end_) value_type(s[*i]);
        __guard.__complete();
    }
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>::ndarray(const indirect_array<value_type>& ia) : begin_(nullptr), end_(nullptr) {
    const size_t n = ia.oned_.size();
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        typedef const size_t* _Ip;
        const value_type* s = ia.vp_;
        for (_Ip i = ia.oned_.begin_, e = ia.oned_.end_; i != e; ++i, ++end_)
            ::new ((void*)end_) value_type(s[*i]);
        __guard.__complete();
    }
}

//template <class Tp, class Ex, class Lp>
//inline ndarray<Tp, Ex, Lp>::~ndarray() {
//    __clear(size());
//}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::__assign_range(const value_type* __f, const value_type* __l) {
    size_t n = __l - __f;
    if (size() != n) {
        __clear(size());
        begin_ = allocator<value_type>().allocate(n);
        end_ = begin_ + n;
        std::uninitialized_copy(__f, __l, begin_);
    }
    else {
        std::copy(__f, __l, begin_);
    }
    return *this;
}

//template <class Tp, class Ex, class Lp>
//ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const ndarray& v) {
//    if (this != std::addressof(v))
//        return __assign_range(v.begin_, v.end_);
//    return *this;
//}

//template <class Tp, class Ex, class Lp>
//inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(ndarray&& v) noexcept {
//    __clear(size());
//    begin_ = v.begin_;
//    end_ = v.end_;
//    v.begin_ = nullptr;
//    v.end_ = nullptr;
//    return *this;
//}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(std::initializer_list<value_type> __il) {
    return __assign_range(__il.begin(), __il.end());
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const value_type& x) {
    std::fill(begin_, end_, x);
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const slice_array<value_type>& sa) {
    value_type* t = begin_;
    const value_type* s = sa.vp_;
    for (size_t n = sa.size_; n; --n, s += sa.stride_, ++t)
        *t = *s;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const mask_array<value_type>& ma) {
    typedef const size_t* _Ip;
    value_type* t = begin_;
    const value_type* s = ma.vp_;
    for (_Ip i = ma.oned_.begin_, e = ma.oned_.end_; i != e; ++i, ++t)
        *t = s[*i];
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const indirect_array<value_type>& ia) {
    typedef const size_t* _Ip;
    value_type* t = begin_;
    const value_type* s = ia.vp_;
    for (_Ip i = ia.oned_.begin_, e = ia.oned_.end_; i != e; ++i, ++t)
        *t = s[*i];
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class ValExpr>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator=(const nc_val_expr<ValExpr>& v) {
    size_t n = v.size();
    if (size() != n)
        resize(n);
    value_type* t = begin_;
    for (size_t i = 0; i != n; ++t, ++i)
        *t = result_type(v[i]);
    return *this;
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<__slice_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator[](slice s) const {
    return nc_val_expr<__slice_expr<const ndarray&> >(__slice_expr<const ndarray&>(s, *this));
}

template <class Tp, class Ex, class Lp>
inline slice_array<Tp> ndarray<Tp, Ex, Lp>::operator[](slice s) {
    return slice_array<value_type>(s, *this);
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<__mask_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator[](const ndarray<bool, Ex, Lp>& __vb) const {
    return nc_val_expr<__mask_expr<const ndarray&> >(__mask_expr<const ndarray&>(__vb, *this));
}

template <class Tp, class Ex, class Lp>
inline mask_array<Tp> ndarray<Tp, Ex, Lp>::operator[](const ndarray<bool, Ex, Lp>& __vb) {
    return mask_array<value_type>(__vb, *this);
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<__mask_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator[](ndarray<bool, Ex, Lp>&& __vb) const {
    return nc_val_expr<__mask_expr<const ndarray&> >(__mask_expr<const ndarray&>(std::move(__vb), *this));
}

template <class Tp, class Ex, class Lp>
inline mask_array<Tp> ndarray<Tp, Ex, Lp>::operator[](ndarray<bool, Ex, Lp>&& __vb) {
    return mask_array<value_type>(std::move(__vb), *this);
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<__indirect_expr<const ndarray<Tp, Ex, Lp>&> >
ndarray<Tp, Ex, Lp>::operator[](const ndarray<size_t, Ex, Lp>& __vs) const {
    return nc_val_expr<__indirect_expr<const ndarray&> >(__indirect_expr<const ndarray&>(__vs, *this));
}

template <class Tp, class Ex, class Lp>
inline indirect_array<Tp> ndarray<Tp, Ex, Lp>::operator[](const ndarray<size_t, Ex, Lp>& __vs) {
    return indirect_array<value_type>(__vs, *this);
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<__indirect_expr<const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator[](ndarray<size_t, Ex, Lp>&& __vs) const {
    return nc_val_expr<__indirect_expr<const ndarray&> >(__indirect_expr<const ndarray&>(std::move(__vs), *this));
}

template <class Tp, class Ex, class Lp>
inline indirect_array<Tp> ndarray<Tp, Ex, Lp>::operator[](ndarray<size_t, Ex, Lp>&& __vs) {
    return indirect_array<value_type>(std::move(__vs), *this);
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<UnaryOp<__unary_plus<Tp>, const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator+() const {
    using Op = UnaryOp<__unary_plus<Tp>, const ndarray<Tp>&>;
    return nc_val_expr<Op>(Op(__unary_plus<Tp>(), *this));
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<UnaryOp<std::negate<Tp>, const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator-() const {
    using Op = UnaryOp<std::negate<Tp>, const ndarray<Tp>&>;
    return nc_val_expr<Op>(Op(std::negate<Tp>(), *this));
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<UnaryOp<__bit_not<Tp>, const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator~() const {
    using Op = UnaryOp<__bit_not<Tp>, const ndarray<Tp>&>;
    return nc_val_expr<Op>(Op(__bit_not<Tp>(), *this));
}

template <class Tp, class Ex, class Lp>
inline nc_val_expr<UnaryOp<std::logical_not<Tp>, const ndarray<Tp, Ex, Lp>&> > ndarray<Tp, Ex, Lp>::operator!() const {
    using Op = UnaryOp<std::logical_not<Tp>, const ndarray<Tp>&>;
    return nc_val_expr<Op>(Op(std::logical_not<Tp>(), *this));
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator*=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p *= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator/=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p /= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator%=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p %= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator+=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p += x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator-=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p -= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator^=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p ^= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator&=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p &= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator|=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p |= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator<<=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p <<= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator>>=(const value_type& x) {
    for (value_type* p = begin_; p != end_; ++p)
        *p >>= x;
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator*=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t *= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator/=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t /= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator%=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t %= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator+=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t += std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator-=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t -= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator^=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t ^= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator|=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t |= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator&=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t &= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator<<=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t <<= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> >
inline ndarray<Tp, Ex, Lp>& ndarray<Tp, Ex, Lp>::operator>>=(const Expr& v) {
    size_t i = 0;
    for (value_type* t = begin_; t != end_; ++t, ++i)
        *t >>= std::__get(v, i);
    return *this;
}

template <class Tp, class Ex, class Lp>
inline void ndarray<Tp, Ex, Lp>::swap(ndarray& v) noexcept {
    std::swap(begin_, v.begin_);
    std::swap(end_, v.end_);
}

template <class Tp, class Ex, class Lp>
inline Tp ndarray<Tp, Ex, Lp>::sum() const {
    if (begin_ == end_)
        return value_type();
    const value_type* p = begin_;
    Tp r = *p;
    for (++p; p != end_; ++p)
        r += *p;
    return r;
}

template <class Tp, class Ex, class Lp>
inline Tp ndarray<Tp, Ex, Lp>::min() const {
    if (begin_ == end_)
        return value_type();
    return *std::min_element(begin_, end_);
}

template <class Tp, class Ex, class Lp>
inline Tp ndarray<Tp, Ex, Lp>::max() const {
    if (begin_ == end_)
        return value_type();
    return *std::max_element(begin_, end_);
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::shift(int i) const {
    ndarray<value_type> r;
    size_t n = size();
    if (n) {
        r.begin_ = r.end_ = allocator<value_type>().allocate(n);
        const value_type* __sb;
        value_type* __tb;
        value_type* __te;
        if (i >= 0) {
            i = std::min(i, static_cast<int>(n));
            __sb = begin_ + i;
            __tb = r.begin_;
            __te = r.begin_ + (n - i);
        }
        else {
            i = std::min(-i, static_cast<int>(n));
            __sb = begin_;
            __tb = r.begin_ + i;
            __te = r.begin_ + n;
        }
        for (; r.end_ != __tb; ++r.end_)
            ::new ((void*)r.end_) value_type();
        for (; r.end_ != __te; ++r.end_, ++__sb)
            ::new ((void*)r.end_) value_type(*__sb);
        for (__te = r.begin_ + n; r.end_ != __te; ++r.end_)
            ::new ((void*)r.end_) value_type();
    }
    return r;
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::cshift(int i) const {
    ndarray<value_type> r;
    size_t n = size();
    if (n) {
        r.begin_ = r.end_ = allocator<value_type>().allocate(n);
        i %= static_cast<int>(n);
        const value_type* __m = i >= 0 ? begin_ + i : end_ + i;
        for (const value_type* s = __m; s != end_; ++r.end_, ++s)
            ::new ((void*)r.end_) value_type(*s);
        for (const value_type* s = begin_; s != __m; ++r.end_, ++s)
            ::new ((void*)r.end_) value_type(*s);
    }
    return r;
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::apply(value_type __f(value_type)) const {
    ndarray<value_type> r;
    size_t n = size();
    if (n) {
        r.begin_ = r.end_ = allocator<value_type>().allocate(n);
        for (const value_type* p = begin_; n; ++r.end_, ++p, --n)
            ::new ((void*)r.end_) value_type(__f(*p));
    }
    return r;
}

template <class Tp, class Ex, class Lp>
ndarray<Tp, Ex, Lp> ndarray<Tp, Ex, Lp>::apply(value_type __f(const value_type&)) const {
    ndarray<value_type> r;
    size_t n = size();
    if (n) {
        r.begin_ = r.end_ = allocator<value_type>().allocate(n);
        for (const value_type* p = begin_; n; ++r.end_, ++p, --n)
            ::new ((void*)r.end_) value_type(__f(*p));
    }
    return r;
}

template <class Tp, class Ex, class Lp>
inline void ndarray<Tp, Ex, Lp>::__clear(size_t capacity) {
    if (begin_ != nullptr) {
        while (end_ != begin_)
            (--end_)->~value_type();
        std::allocator<value_type>().deallocate(begin_, capacity);
        begin_ = end_ = nullptr;
    }
}

template <class Tp, class Ex, class Lp>
void ndarray<Tp, Ex, Lp>::resize(size_t n, value_type x) {
    __clear(size());
    if (n) {
        begin_ = end_ = allocator<value_type>().allocate(n);
        auto __guard = std::__make_exception_guard([&] { __clear(n); });
        for (size_t __n_left = n; __n_left; --__n_left, ++end_)
            ::new ((void*)end_) value_type(x);
        __guard.__complete();
    }
}

template <class Tp, class Ex, class Lp>
inline void swap(ndarray<Tp, Ex, Lp>& x, ndarray<Tp, Ex, Lp>& y) noexcept {
    x.swap(y);
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::multiplies<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator*(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::multiplies<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::multiplies<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::multiplies<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator*(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::multiplies<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::multiplies<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::multiplies<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator*(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::multiplies<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::multiplies<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::divides<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator/(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::divides<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::divides<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::divides<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator/(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::divides<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::divides<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::divides<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator/(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::divides<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::divides<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::modulus<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator%(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::modulus<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::modulus<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::modulus<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator%(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::modulus<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::modulus<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::modulus<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator%(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::modulus<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::modulus<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::plus<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator+(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::plus<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::plus<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::plus<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator+(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::plus<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::plus<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::plus<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator+(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::plus<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::plus<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::minus<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator-(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::minus<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::minus<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::minus<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator-(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::minus<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::minus<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::minus<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator-(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::minus<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::minus<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::bit_xor<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator^(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::bit_xor<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::bit_xor<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::bit_xor<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator^(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::bit_xor<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::bit_xor<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::bit_xor<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator^(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::bit_xor<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::bit_xor<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::bit_and<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator&(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::bit_and<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::bit_and<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::bit_and<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator&(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::bit_and<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::bit_and<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::bit_and<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator&(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::bit_and<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::bit_and<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::bit_or<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator|(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::bit_or<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::bit_or<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::bit_or<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator|(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::bit_or<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::bit_or<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::bit_or<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator|(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::bit_or<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::bit_or<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<__bit_shift_left<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator<<(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<__bit_shift_left<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(__bit_shift_left<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr< BinaryOp<__bit_shift_left<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator<<(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__bit_shift_left<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(__bit_shift_left<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr< BinaryOp<__bit_shift_left<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator<<(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__bit_shift_left<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__bit_shift_left<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<__bit_shift_right<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator>>(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<__bit_shift_right<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(__bit_shift_right<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline nc_val_expr<
    BinaryOp<__bit_shift_right<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
    operator>>(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__bit_shift_right<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(__bit_shift_right<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr< BinaryOp<__bit_shift_right<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator>>(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__bit_shift_right<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__bit_shift_right<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::logical_and<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator&&(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::logical_and<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::logical_and<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::logical_and<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator&&(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::logical_and<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::logical_and<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::logical_and<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator&&(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::logical_and<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::logical_and<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::logical_or<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator||(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::logical_or<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::logical_or<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::logical_or<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator||(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::logical_or<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::logical_or<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::logical_or<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator||(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::logical_or<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::logical_or<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::equal_to<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator==(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::equal_to<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::equal_to<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::equal_to<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator==(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::equal_to<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::equal_to<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::equal_to<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator==(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::equal_to<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::equal_to<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::not_equal_to<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator!=(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::not_equal_to<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::not_equal_to<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::not_equal_to<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator!=(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::not_equal_to<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::not_equal_to<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::not_equal_to<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator!=(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::not_equal_to<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::not_equal_to<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::less<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator<(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::less<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::less<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::less<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator<(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::less<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::less<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::less<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator<(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::less<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::less<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::greater<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator>(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::greater<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::greater<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::greater<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator>(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::greater<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::greater<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::greater<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator>(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::greater<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::greater<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::less_equal<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator<=(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::less_equal<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::less_equal<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::less_equal<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator<=(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::less_equal<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::less_equal<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::less_equal<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator<=(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::less_equal<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::less_equal<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
inline nc_val_expr<BinaryOp<std::greater_equal<typename _Expr1::value_type>, _Expr1, _Expr2> >
operator>=(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<std::greater_equal<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(std::greater_equal<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::greater_equal<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
operator>=(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::greater_equal<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(std::greater_equal<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
inline
nc_val_expr<BinaryOp<std::greater_equal<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
operator>=(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<std::greater_equal<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(std::greater_equal<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__abs_expr<typename Expr::value_type>, Expr> >
abs(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__abs_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__abs_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__acos_expr<typename Expr::value_type>, Expr> >
acos(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__acos_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__acos_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__asin_expr<typename Expr::value_type>, Expr> >
asin(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__asin_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__asin_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__atan_expr<typename Expr::value_type>, Expr> >
atan(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__atan_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__atan_expr<value_type>(), x));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
[[nodiscard]] inline
nc_val_expr<BinaryOp<__atan2_expr<typename _Expr1::value_type>, _Expr1, _Expr2> >
atan2(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<__atan2_expr<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(__atan2_expr<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline
nc_val_expr<BinaryOp<__atan2_expr<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
atan2(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__atan2_expr<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(__atan2_expr<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline
nc_val_expr<BinaryOp<__atan2_expr<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
atan2(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__atan2_expr<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__atan2_expr<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__cos_expr<typename Expr::value_type>, Expr> >
cos(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__cos_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__cos_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__cosh_expr<typename Expr::value_type>, Expr> >
cosh(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__cosh_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__cosh_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<exp_expr<typename Expr::value_type>, Expr> >
exp(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<exp_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(exp_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__log_expr<typename Expr::value_type>, Expr> >
log(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__log_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__log_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__log10_expr<typename Expr::value_type>, Expr> >
log10(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__log10_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__log10_expr<value_type>(), x));
}

template <class _Expr1,
    class _Expr2,
    std::enable_if_t<nc_is_val_expr<_Expr1>::value&& nc_is_val_expr<_Expr2>::value, int> = 0>
[[nodiscard]] inline
nc_val_expr<BinaryOp<__pow_expr<typename _Expr1::value_type>, _Expr1, _Expr2> >
pow(const _Expr1& x, const _Expr2& y) {
    typedef typename _Expr1::value_type value_type;
    typedef BinaryOp<__pow_expr<value_type>, _Expr1, _Expr2> Op;
    return nc_val_expr<Op>(Op(__pow_expr<value_type>(), x, y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline
nc_val_expr<BinaryOp<__pow_expr<typename Expr::value_type>, Expr, __scalar_expr<typename Expr::value_type> > >
pow(const Expr& x, const typename Expr::value_type& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__pow_expr<value_type>, Expr, __scalar_expr<value_type> > Op;
    return nc_val_expr<Op>(Op(__pow_expr<value_type>(), x, __scalar_expr<value_type>(y, x.size())));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline
nc_val_expr<BinaryOp<__pow_expr<typename Expr::value_type>, __scalar_expr<typename Expr::value_type>, Expr> >
pow(const typename Expr::value_type& x, const Expr& y) {
    typedef typename Expr::value_type value_type;
    typedef BinaryOp<__pow_expr<value_type>, __scalar_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__pow_expr<value_type>(), __scalar_expr<value_type>(x, y.size()), y));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__sin_expr<typename Expr::value_type>, Expr> >
sin(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__sin_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__sin_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__sinh_expr<typename Expr::value_type>, Expr> >
sinh(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__sinh_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__sinh_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__sqrt_expr<typename Expr::value_type>, Expr> >
sqrt(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__sqrt_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__sqrt_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__tan_expr<typename Expr::value_type>, Expr> >
tan(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__tan_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__tan_expr<value_type>(), x));
}

template <class Expr, std::enable_if_t<nc_is_val_expr<Expr>::value, int> = 0>
[[nodiscard]] inline nc_val_expr<UnaryOp<__tanh_expr<typename Expr::value_type>, Expr> >
tanh(const Expr& x) {
    typedef typename Expr::value_type value_type;
    typedef UnaryOp<__tanh_expr<value_type>, Expr> Op;
    return nc_val_expr<Op>(Op(__tanh_expr<value_type>(), x));
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline Tp* begin(ndarray<Tp, Ex, Lp>& v) {
    return v.begin_;
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline const Tp* begin(const ndarray<Tp, Ex, Lp>& v) {
    return v.begin_;
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline Tp* end(ndarray<Tp, Ex, Lp>& v) {
    return v.end_;
}

template <class Tp, class Ex, class Lp>
[[nodiscard]] inline const Tp* end(const ndarray<Tp, Ex, Lp>& v) {
    return v.end_;
}

}

#endif  // NUMCXX_H_