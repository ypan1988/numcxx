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
+ vector, matrix, cube classes are type aliases that fix the `Extents` / `LayoutPolicy` parameter of `ndarray`:

  |dimensions|dynamic extents|fixed extents|
  |----------|---------------|-------------|
  |1D|`vec<T> = numcxx::ndarray<T, dextents<1>>`|`vec_fixed<T, N> = ndarray<T, extents<N>>`|
  |2D|`mat<T> = numcxx::ndarray<T, dextents<2>>`|`mat_fixed<T, M, N> = ndarray<T, extents<M, N>>`| 
  |3D|`cube<T> = numcxx::ndarray<T, dextents<3>>`|`cube_fixed<T, M, N, K> = ndarray<T, extents<M, N, K>>`|

### 1.1 `vec<T>` / `vec_fixed<T, N>`
+ Type alias for 1D `ndarray`.
+ For convenience, the following vector typedefs are defined:
	``` cpp
	// dynamic
	dvec = vec<double>
	fvec = vec<float>
	ivec = vec<int>
	uvec = vec<unsigned>

	// static (N = 2, 3, 4)
	dvec2 / dvec3 / dvec4 = vec_fixed<double, N>
	fvec2 / fvec3 / fvec4 = vec_fixed<float, N>
    ```
### 1.2 `mat<T>` / `mat_fixed<T, M, N>`
+ Type alias for 2D `ndarray`.
+ For convenience, the following matrix typedefs are defined:
	``` cpp
	// dynamic
	dmat = mat<double>
	fmat = mat<float>
	imat = mat<int>
	umat = mat<unsigned>

	// static (N = 2, 3, 4)
	dmat22 / dmat33 / dmat44 = mat_fixed<double, N, N>
	fmat22 / fmat33 / fmat44 = mat_fixed<float, N, N>
	```
### 1.3 `cube<T>` / `cube_fixed<T, M, N, K>`
+ Type alias for 3D `ndarray`.
+ For convenience, the following cube typedefs are defined:
	``` cpp
	// dynamic
	dcube = cube<double>
	fcube = cube<float>
	icube = cube<int>
	ucube = cube<unsigned>

	// static (N = 2, 3, 4)
	dcube222 / dcube333 / dcube444 = cube_fixed<double, N, N, N>
	fcube222 / fcube333 / fcube444 = cube_fixed<float, N, N, N>
    ```

<a id="member-functions-and-slicing"></a>
## 2. Member functions and Slicing
