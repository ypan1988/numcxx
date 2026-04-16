# API Documentation for NumCxx
## 0. Overview
+ [Vector, Matrix, Cube and NdArray Classes](#ndarray-classes)
+ [Member functions and Slicing](#member-functions-and-slicing)

<a id="ndarray-classes"></a>
## 1. Vector, Matrix, Cube and NdArray Classes

### 1.0 `ndarray<T, Extents, LayoutPolicy>`
+ template class represents a **generic, multidimensional array**.
+ **`T`**: The type of the stored elements.
+ **`Extents`**: Describes the rank (i.e., number of dimensions) and size of each dimension. Both dynamic (dextents) and compile-time (extents) extents are supported.
+ **`LayoutPolicy`**: Specifies how multidimensional indices are mapped to linear storage. By default, layout_right (row-major ordering) is used.
+ Vector, Matrix, Cube classes are **specializations of ndarray** distinguished by Extents:

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
