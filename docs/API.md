# API Documentation for NumCxx
## 0. Overview
+ [Vector, Matrix, Cube and NdArray Classes](#ndarray-classes)
+ [Member functions and Slicing](#member-functions-and-slicing)

<a id="ndarray-classes"></a>
## 1. Vector, Matrix, Cube and NdArray Classes

### 1.0 `ndarray<T, Extents, LayoutPolicy>`
+ `ndarray` is a template class that represents a **generic, multidimensional array**.
	+ **`T`**: the type (`double`, `float`, `int`, ...) of the stored elements.
	+ **`Extents`**: describes the rank (i.e., number of dimensions) and size of each dimension. You can choose:
	    + Dynamic extents with `dextents<Rank>` (e.g., dextents<2> for a 2-D ndarray whose shape is decided at run-time).
		+ Compile-time extents with `extents<Dims...>`(e.g., extents<2, 3> for a fixed 2x3 ndarray).
	+ **`LayoutPolicy`**: specifies how multidimensional indices are mapped to linear storage. Two options are available, namely `layout_right`(i.e., row-major ordering, the default) and `layout_left` (i.e., column-major ordering).
+ Constructors:
    + `ndarray(SizeTypes... dyn_exts)` - with dynamic extents (rank must match `Extents::rank()`).
    + `ndarray(const ndarray &other)` - copy constructor.
    + `ndarray(ndarray &&other)` - move constructor.
    + `ndarray(const slice_view<T, Ex, Lp> &sv)` - from a slice view.
    + `ndarray(const mask_view<T> &mv)` - from a mask view.
    + `ndarray(const indirect_view<T> &iv)` - from an indirect view.

+ Examples:

```cpp
ndarray<double, dextents<2>, layout_right> m1(3, 4);
ndarray<double, extents<3, 4>> m2;
m1(0, 1) = 5.0;
```

+ vector, matrix, cube classes are type aliases that fix the `Extents` / `LayoutPolicy` parameter of `ndarray`:

  |dimensions|dynamic extents|fixed extents|
  |----------|---------------|-------------|
  |1D|`vec<T> = ndarray<T, dextents<1>>`|`vec_fixed<T, N> = ndarray<T, extents<N>>`|
  |2D|`mat<T> = ndarray<T, dextents<2>>`|`mat_fixed<T, M, N> = ndarray<T, extents<M, N>>`| 
  |3D|`cube<T> = ndarray<T, dextents<3>>`|`cube_fixed<T, M, N, K> = ndarray<T, extents<M, N, K>>`|

### 1.1 `vec<T>` / `vec_fixed<T, N>`
+ Type alias for 1D `ndarray`.
+ For convenience, the following vector typedefs are defined:
    
```cpp
// dynamic vectors
dvec = vec<double>
fvec = vec<float>
ivec = vec<int>
uvec = vec<unsigned>

// static vectors: vec_fixed<T, N>, with N = 2, 3, 4
dvec2 / dvec3 / dvec4 = vec_fixed<double,N>
fvec2 / fvec3 / fvec4 = vec_fixed<float, N>
ivec2 / ivec3 / ivec4 = vec_fixed<int, N>
uvec2 / uvec3 / uvec4 = vec_fixed<unsigned, N>
```

### 1.2 `mat<T>` / `mat_fixed<T, M, N>`
+ Type alias for 2D `ndarray`.
+ For convenience, the following matrix typedefs are defined:

```cpp
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

```cpp
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
### 2.0 `slice`

+ `slice` describes a range of elements in one dimension using `start`, `stop`, and `step`.
+ It follows NumPy-style slicing:
    + `start` is included.
    + `stop` is excluded.
    + `step` controls the interval between selected elements.
    + Negative indices count from the end.
    + A negative `step` selects elements in reverse order.

+ Forms:
    + `slice()` - select the whole dimension.
    + `slice(start, stop)` - select from `start` up to, but not including, `stop`.
    + `slice(start, stop, step)` - select every `step`-th element between `start` and `stop`.

+ Common slices:

  | NumCxx | Equivalent notation | Meaning |
  |--------|---------------------|---------|
  | `slice()` | `[:]` | all elements |
  | `slice(1, 5)` | `[1:5]` | elements 1 through 4 |
  | `slice(0, 10, 2)` | `[0:10:2]` | every second element |
  | `slice(-3, {})` | `[-3:]` | the last three elements |
  | `slice({}, 5)` | `[:5]` | from the beginning up to element 5 |
  | `slice(2, {})` | `[2:]` | from element 2 to the end |
  | `slice({}, {}, 2)` | `[::2]` | every second element |
  | `slice({}, {}, -1)` | `[::-1]` | all elements in reverse order |

+ Missing values are written as `{}`:
    + `slice({}, stop)` - omit `start`.
    + `slice(start, {})` - omit `stop`.
    + `slice({}, {}, step)` - omit both `start` and `stop`.

+ Slicing an `ndarray`:
    + Use `operator()` with one argument for each dimension.
    + An integer selects one element along a dimension.
    + A `slice` selects a range of elements.
    + `full_extent` selects the whole dimension.
    + A pair such as `{1, 4}` is a convenient range equivalent to `slice(1, 4)`.

+ Examples:

      dmat A(4, 5);

      auto rows = A(slice(1, 3), full_extent);
      auto cols = A(full_extent, slice(0, 5, 2));
      auto row  = A(1, full_extent);
      auto rev  = A(slice({}, {}, -1), full_extent);

  In the examples above:
    + `rows` selects rows 1 and 2 and keeps all columns.
    + `cols` keeps all rows and selects every second column.
    + `row` selects row 1 and keeps all columns.
    + `rev` keeps all columns and reverses the row order.

+ A slicing operation returns a `slice_view` of the original array:
    + The selected elements are not copied.
    + Changes made through the view affect the original array.
    + A new independent `ndarray` can be constructed from the view when a copy is needed.

+ Example of modifying the original array through a slice:

      dmat A(4, 5);

      auto middle = A(slice(1, 3), full_extent);
      middle = 0.0;

  This assigns `0.0` to rows 1 and 2 of `A`.

+ Notes:
    + `start` is inclusive.
    + `stop` is exclusive.
    + Negative indices are interpreted relative to the end of the dimension.
    + A negative `step` selects elements in reverse order.
    + `step` must not be zero.
    + Out-of-range `start` and `stop` values are clipped to the valid range.
