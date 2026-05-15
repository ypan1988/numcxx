# API Documentation for NumCxx
## 0. Overview
+ [Vector, Matrix, Cube and NdArray Classes](#ndarray-classes)
+ [Member functions and Slicing](#member-functions-and-slicing)

<a id="ndarray-classes"></a>
## 1. Vector, Matrix, Cube and NdArray Classes

### 1.0 `ndarray<T, Extents, LayoutPolicy>`
+ `ndarray` is a template class represents a **generic, multidimensional array**.
	+ **`T`**: the type (`double`, `float`, `int`, ...) of the stored elements.
	+ **`Extents`**: describes the rank (i.e., number of dimensions) and size of each dimension. You can choose:
	    + Dynamic extents with `dextents<Rank>` (e.g., dextents<2> for a 2-D ndarray whose shape is decided at run-time).
		+ Compile-time extents with `extents<Dims...>`(e.g., extents<2, 3> for a fixed 2x3 ndarray).
	+ **`LayoutPolicy`**: specifies how multidimensional indices are mapped to linear storage. Two options are available, namely `layout_right`(i.e., row-major ordering, the default) and `layout_left` (i.e., column-major ordering).
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
	// dynamic vectors
	dvec = vec<double>
	fvec = vec<float>
	ivec = vec<int>
	uvec = vec<unsigned>

	// static vectors: vec_fixed<T, N>, with N = 2, 3, 4
	dvec2 / dvec3 / dvec4 = vec_fixed<double, N>
	fvec2 / fvec3 / fvec4 = vec_fixed<float, N>
	ivec2 / ivec3 / ivec4 = vec_fixed<int, N>
	uvec2 / uvec3 / uvec4 = vec_fixed<unsigned, N>
    ```
### 1.2 `mat<T>` / `mat_fixed<T, M, N>`
+ Type alias for 2D `ndarray`.
+ For convenience, the following matrix typedefs are defined:
	``` cpp
	// dynamic matrices
	dmat = mat<double>
	fmat = mat<float>
	imat = mat<int>
	umat = mat<unsigned>

	// static square matrices: mat_fixed<T, N, N>, with N = 2, 3, 4
	dmat22 / dmat33 / dmat44 = mat_fixed<double, N, N>
	fmat22 / fmat33 / fmat44 = mat_fixed<float, N, N>
	imat22 / imat33 / imat44 = mat_fixed<int, N, N>
	umat22 / umat33 / umat44 = mat_fixed<unsigned, N, N>
	```
### 1.3 `cube<T>` / `cube_fixed<T, M, N, K>`
+ Type alias for 3D `ndarray`.
+ For convenience, the following cube typedefs are defined:
	``` cpp
	// dynamic cubes
	dcube = cube<double>
	fcube = cube<float>
	icube = cube<int>
	ucube = cube<unsigned>

	// static cubes: cube_fixed<T, N, N, N>, with N = 2, 3, 4
	dcube222 / dcube333 / dcube444 = cube_fixed<double, N, N, N>
	fcube222 / fcube333 / fcube444 = cube_fixed<float, N, N, N>
	icube222 / icube333 / icube444 = cube_fixed<int, N, N, N>
	ucube222 / ucube333 / ucube444 = cube_fixed<unsigned, N, N, N>
    ```

<a id="member-functions-and-slicing"></a>
## 2. Member functions and Slicing
