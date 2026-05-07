# API Documentation for NumCxx
## 0. Overview
+ [Vector, Matrix, Cube and NdArray Classes](#ndarray-classes)
+ [Member functions and Slicing](#member-functions-and-slicing)

<a id="ndarray-classes"></a>
## 1. Vector, Matrix, Cube and NdArray Classes

### 1.0 `ndarray<T, Extents, LayoutPolicy>`
+ `ndarray` is a template class represents a **generic, multidimensional array**.
	+ **`T`**: the type (`double`, `float`, `int`, ...) of the stored elements.
	+ **`Extents`**: describes the rank (i.e., number of dimensions) and size of each dimension. Both dynamic (`dextents<Rank>`) and compile-time (`extents<Dims...>`) extents are supported.
	+ **`LayoutPolicy`**: Specifies how multidimensional indices are mapped to linear storage. Default is `layout_right` (row-major).
+ Constructors:
  ``` cpp
  ndarray(SizeTypes... dyn_exts)           // with dynamic extents (rank must match Extents::rank()).
  ndarray(const ndarray &other)            // copy constructor.
  ndarray(ndarray &&other)                 // move constructor.
  ndarray(const slice_view<T, Ex, Lp> &sv) // from a slice view.
  ndarray(const mask_view<T> &mv)          // from a mask view.
  ndarray(const indirect_view<T> &iv)      // from an indirect view.
  ```
+ Examples:
  ``` cpp
  ndarray<double, dextents<2>, layout_right> m1(3, 4);
  ndarray<double, extents<3, 4>> m2;
  m1(0, 1) = 5.0;
  ```
+ Vector, Matrix, Cube classes are type aliases that fix the `Extents` parameter of `ndarray`:

  |dimensions|dynamic extents|fixed extents|
  |----------|---------------|-------------|
  |1D|`vec<T> = numcxx::ndarray<T, dextents<1>>`|`vec_fixed<T, N> = ndarray<T, extents<N>>`|
  |2D|`mat<T> = numcxx::ndarray<T, dextents<2>>`|`mat_fixed<T, M, N> = ndarray<T, extents<M, N>>`| 
  |3D|`cube<T> = numcxx::ndarray<T, dextents<3>>`|`cube_fixed<T, M, N, K> = ndarray<T, extents<M, N, K>>`|

### 1.1.1 `mat<T>`(dynamic) / `mat_fixed<T, M, N>`(static)
+ Classes for dense matrices with `dynamic` and `static` dimensions. The elements are stored in row-major ordering (i.e., row by row) by default
+ The root ndarray classes are `ndarray<T, dextents<size_t, 2>>`(dynamic) and `ndarray<T, extents<size_t, M, N>>`(static)
+ For convenience, the following matrix typedefs are defined:
	+ typedef for matrix (dynamic)
	``` cpp
	dmat = mat<double>
	fmat = mat<float>
	imat = mat<int>
	umat = mat<unsigned int>
+ typedef for matrix (static) (e.g., `dmat22 = mat_fixed<double, 2, 2>`)

| type   | full type     |
| -----  | ------------  |
| `dmat22`/`dmat33`/`dmat44`/ |`mat_fixed<double, 2, 2>` / `mat_fixed<double, 3, 3>`/ `mat_fixed<double, 4, 4>`|
| `fmat22`/`fmat33`/`fmat44`/ |`mat_fixed<float, 2, 2>` / `mat_fixed<float, 3, 3>` / `mat_fixed<float, 4, 4>`|

### 1.1.2 `mat_fixed<T, M, N>`(static)


<a id="member-functions-and-slicing"></a>
## Member functions and Slicing
