# API Documentation for NumCxx 0.1

## Overview
+ [ndarray, vector, matrix and cube classes](#ndarray-classes)
+ [indexing and slicing](#indexing-and-slicing)
+ [member functions and variables](#member-functions-and-variables)
+ [reductions](#reductions)

<a id="ndarray-classes"></a>
## NdArray, Vector, Matrix and Cube Classes
| | |
|---|---|
| **[`ndarray`](#ndarray-def)** | dense N-d array class |
| **[`vec` / `vec_fixed`](#vec-def)** | 1-d vector class (ndarray alias) |
| **[`mat` / `mat_fixed`](#mat-def)** | 2-d matrix class (ndarray alias) |
| **[`cube` / `cube_fixed`](#cube-def)** | 3-d cube class (ndarray alias) |
| **[operators](#operators)** | +, −, *, %, /, ==, !=, <=, >=, <, >, &&, \|\| |

<a id="indexing-and-slicing"></a>
## Indexing and Slicing
| | |
|---|---|
| **[element access](#element-access)** | element access via `()`, `[]` and `flat()` |
| **[`slice`](#slice-def)** | describes a range of elements along one dimension |
| **[`gslice`](#gslice-def)** | packages multidimensional index and slice arguments |
| **[`gslice_view`](#gslice-view-def)** | view created by slicing an array |
| **[`mask_view`](#mask-view-def)** | view selecting elements using a boolean mask |
| **[`indirect_view`](#indirect-view-def)** | view selecting elements using indices |


<a id="member-functions-and-variables"></a>
## Member Functions and Variables

<a id="ndarray-def"></a>
### `ndarray<T, Extents, LayoutPolicy>`
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
    + `ndarray(const gslice_view<T, Ex, Lp> &sv)` - from a slice view.
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

<a id="vec-def"></a>
### `vec<T>` / `vec_fixed<T, N>`
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

<a id="mat-def"></a>
### `mat<T>` / `mat_fixed<T, M, N>`
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

<a id="cube-def"></a>
### `cube<T>` / `cube_fixed<T, M, N, K>`
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

<a id="operators"></a>
### Operators: `+, -, *, %, /, ==, !=, <=, >=, <, >, &&, ||`

- Overloaded operators for `ndarray`, `vec / vec_fixed`, `mat / mat_fixed`, `cube / cube_fixed`, and compatible NumCxx expressions.
- Operations are performed element-wise.
- Array operations require compatible shapes.
- Scalar operations apply the scalar to each element.

**Operations:**

| operator | operation |
|---|---|
| `+` | element-wise addition of two objects, or addition of an object and a scalar |
| `-` | element-wise subtraction of two objects, subtraction involving a scalar, or negation of an object |
| `*` | element-wise multiplication of two objects, or multiplication of an object by a scalar |
| `/` | element-wise division of two objects, or division involving a scalar |
| `%` | element-wise modulo of two objects, or modulo involving a scalar |
| `==` | element-wise equality comparison |
| `!=` | element-wise inequality comparison |
| `>=` | element-wise greater-than-or-equal comparison |
| `<=` | element-wise less-than-or-equal comparison |
| `>` | element-wise greater-than comparison |
| `<` | element-wise less-than comparison |
| `&&` | element-wise logical AND |
| `\|\|` | element-wise logical OR |

Value expressions can directly construct an `ndarray` or be assigned to an
already-sized array. Construction deduces dynamic extents from the expression;
assignment requires the destination shape to match.

<a id="reductions"></a>
## Reductions

`ndarray` provides full-array reductions (`sum`, `min`, `max`, `prod`, `mean`,
`var`, `std`, `any`, `all`, `argmin`, and `argmax`) and single-axis overloads
that take an `index_type`. Axis reductions remove the selected dimension and
return an owning `ndarray` in the default layout. Negative axes count from the
last dimension; an out-of-range axis throws `std::out_of_range` when exceptions
are enabled. `gslice_view` and value expressions provide these reductions by
evaluating to an `ndarray` first.

- `mean` returns a floating-point result for integral inputs. `var` uses
  population variance (`ddof = 0`); complex input produces real variance and
  standard deviation.
- Empty `sum` and `min` / `max` return the existing value-initialized result,
  while empty `prod` is `1`.
- `mean`, `var`, `std`, `argmin`, and `argmax` throw `std::runtime_error` when
  no elements are available to reduce.
- `argmin` and `argmax` return the first matching index in logical
  default-layout order for full reductions, or the index within the selected
  axis for axis reductions.
- `any` and `all` treat nonzero values as true. Full reductions return `bool`;
  axis reductions return an unsigned-char array containing only `0` and `1`.
  Empty inputs follow the logical identities: `any` is false and `all` is true.
- `bvec`, `bmat`, and `bcube` are dynamic `unsigned char` arrays intended for
  logical values; zero means false and nonzero means true.

## Indexing and Slicing

This section covers NumCxx slicing and the view types used to access selected elements without copying data.

<a id="element-access"></a>
### Element access via `()`, `[]`, `at()` and `flat()`

Provides access to individual elements stored in an `ndarray`, `gslice_view`,
`mask_view`, or `indirect_view`.

| | |
|---|---|
| **`(i, j, ...)`** | Access an element or a subarray using up to one index per dimension. Omitted trailing dimensions are retained in a view. |
| **`[i]`** | Unchecked access at flattened index `i`, following the object's physical storage layout. |
| **`[slice]`** | Slice one dimension and retain later dimensions in a `gslice_view`. |
| **`[gslice(...)]`** | Apply a packaged multidimensional selection, equivalent to `(...)`. |
| **`.at(i)`** | Checked access at flattened index `i`, following the same physical storage layout as `[i]`. Throws `std::out_of_range` if `i` is out of range (when exceptions are enabled). |
| **`.flat(i)`** | Checked access at flattened index `i`, using the default NumCxx layout order, independent of the object's physical storage layout. Throws `std::out_of_range` if `i` is out of range (when exceptions are enabled). |

**Notes:**

- indexing starts at `0`
- `()` handles multiple indices and remains the multidimensional indexing and slicing syntax.
- When every dimension is indexed by an integer, `()` returns an element reference.
- When one or more dimensions remain, `()` returns a view; omitted trailing dimensions are treated as `full_extent`. For example, `A(1)` is equivalent to `A(1, full_extent)` for a 2-D array.
- `[]` takes one selector: an integer for flattened access, a `slice` for single-axis slicing, a `gslice` for packaged multidimensional indexing, a boolean expression for mask selection, or an index collection for indirect selection.
- `A[gslice(i, s, t)]` is equivalent to `A(i, s, t)`. A complete set of integer arguments selects an element; otherwise the result is a view.
- Indices always refer to logical axes in order, for both row-major and column-major arrays.
- NumCxx supports integer indices, `slice`, `full_extent`, and integral pair ranges. This is basic partial indexing, not full NumPy indexing; features such as ellipsis, new axes, and advanced array indices are not supported.
- Integer `[]` access follows the C++ container convention: it does not check bounds; using an out-of-range index has undefined behavior.
- `at()` checks bounds and throws `std::out_of_range` when the index is invalid
- Checked accessors report invalid indices as `"<accessor>: index <i> out of bounds for size <size>"`, where `<accessor>` identifies the type and function (for example, `ndarray::flat`)
- Linear `[]` and `at()` use physical storage order; `flat()` uses the default NumCxx layout order.
- these conventions apply to the listed types' linear element access

<a id="slice-def"></a>
### `slice`

- `slice` describes a range of elements in one dimension using `start`, `stop`, and `step`.
- It follows NumPy-style slicing:
  - `start` is included.
  - `stop` is excluded.
  - `step` controls the interval between selected elements.
  - Negative indices count from the end.
  - A negative `step` selects elements in reverse order.

- Forms:
  - `slice()` - select the whole dimension.
  - `slice(start, stop)` - select from `start` up to, but not including, `stop`.
  - `slice(start, stop, step)` - select every `step`-th element between `start` and `stop`.

- Common slices:

  | NumCxx | Equivalent notation | Meaning |
  |---|---|---|
  | `slice()` | `[:]` | all elements |
  | `slice(1, 5)` | `[1:5]` | elements 1 through 4 |
  | `slice(0, 10, 2)` | `[0:10:2]` | every second element |
  | `slice(-3, {})` | `[-3:]` | the last three elements |
  | `slice({}, 5)` | `[:5]` | from the beginning up to element 5 |
  | `slice(2, {})` | `[2:]` | from element 2 to the end |
  | `slice({}, {}, 2)` | `[::2]` | every second element |
  | `slice({}, {}, -1)` | `[::-1]` | all elements in reverse order |

- Missing values are written as `{}`:
  - `slice({}, stop)` - omit `start`.
  - `slice(start, {})` - omit `stop`.
  - `slice({}, {}, step)` - omit both `start` and `stop`.

- Slicing an `ndarray`:
  - Use `operator()` with no more than one argument per dimension, or put the arguments in a `gslice` and pass it to `operator[]`.
  - An integer selects one element along a dimension.
  - A `slice` selects a range of elements.
  - `full_extent` selects the whole dimension.
  - Dimensions omitted from the end of the argument list are selected in full.
  - A pair such as `{1, 4}` is a convenient range equivalent to `slice(1, 4)`.

- Examples:

      dmat A(4, 5);

      auto rows = A(slice(1, 3), full_extent);
      auto cols = A(full_extent, slice(0, 5, 2));
      auto row  = A(1, full_extent);
      auto rev  = A(slice({}, {}, -1), full_extent);
      auto rows_by_subscript = A[gslice(slice(1, 3), full_extent)];

  In the examples above:
  - `rows` selects rows 1 and 2 and keeps all columns.
  - `cols` keeps all rows and selects every second column.
  - `row` selects row 1 and keeps all columns.
  - `rev` keeps all columns and reverses the row order.

- A slicing operation returns a `gslice_view` of the original array:
  - The selected elements are not copied.
  - Changes made through the view affect the original array.
  - A new independent `ndarray` can be constructed from the view when a copy is needed.

- Example of modifying the original array through a slice:

      dmat A(4, 5);

      auto middle = A(slice(1, 3), full_extent);
      middle = 0.0;

  This assigns `0.0` to rows 1 and 2 of `A`.

- Notes:
  - `start` is inclusive.
  - `stop` is exclusive.
  - Negative indices are interpreted relative to the end of the dimension.
  - A negative `step` selects elements in reverse order.
  - `step` must not be zero.
  - Out-of-range `start` and `stop` values are clipped to the valid range.

<a id="gslice-def"></a>
### `gslice`

- `gslice` owns a value copy of one or more scalar indices, `slice` objects, `full_extent` markers, or integral pair ranges.
- It packages the same multidimensional arguments accepted by `operator()`, so they can be passed as one selector to `operator[]`.
- `A[gslice(1, slice(0, 4, 2))]` is equivalent to `A(1, slice(0, 4, 2))`.
- A complete set of scalar indices accesses an element; partial or slicing selections produce a `gslice_view`.
- `gslice` is a NumCxx argument-pack type; its representation is distinct from `std::gslice`.

<a id="gslice-view-def"></a>
### `gslice_view`

- `gslice_view` is a multidimensional, non-owning view created by slicing an `ndarray` or another compatible view.
- It preserves the shape of the selected elements, except for dimensions removed by integer indexing.
- Elements can be read or modified through the view.
- Changes made through the view affect the original array.
- Assigning a scalar sets every selected element to that value.
- Construct an `ndarray` from a `gslice_view` when an independent copy is required.
- A `gslice_view` must not outlive the array or view from which it was created.

<a id="mask-view-def"></a>
### `mask_view`

- `mask_view` is a non-owning view that selects elements for which a mask is `true`.
- Mask selection is element-wise.
- The mask must be compatible with the source array.
- Selected elements can be read or modified through the view.
- Changes made through the view affect the original array.
- Assigning a scalar sets every selected element to that value.
- Construct an `ndarray` from a `mask_view` when an independent copy is required.
- A `mask_view` must not outlive the array and mask from which it was created.

<a id="indirect-view-def"></a>
### `indirect_view`

- `indirect_view` is a non-owning view that selects elements using a collection of indices.
- Elements are accessed in the order specified by the indices.
- Selected elements can be read or modified through the view.
- Changes made through the view affect the original array.
- Assigning a scalar sets every selected element to that value.
- Construct an `ndarray` from an `indirect_view` when an independent copy is required.
- An `indirect_view` must not outlive the array and index collection from which it was created.

<br aria-hidden="true">
