# Slice

Related source section:
- `[NumCxx.slice]`

Related tests:
- `test_slice.cpp`
- `test_slice_normalize.cpp`

Status:
- Stable
- Public API: `slice`
- Internal API: `slice::normalized_slice`, `slice::normalize()`

## Overview

`NumCxx::slice` represents a NumPy-style slice:

```cpp
slice(start, stop, step)
```

equivalent to:

```python
start:stop:step
```

Examples:

```cpp
slice();            // [:]
slice(1, 5);        // [1:5]
slice({}, {}, -1);  // [::-1]
slice(0, 10, 2);    // [0:10:2]
```

A slice stores the user-specified slice parameters independently of any
particular dimension length.

## Public API

### Construction

```cpp
slice();
slice(start, stop, step);
```

Parameters:

- `start`: inclusive starting position.
- `stop`: exclusive ending position.
- `step`: stride between selected elements.

The default step is:

```cpp
1
```

### Validation

```cpp
bool is_valid() const;
```

Returns:

```cpp
step() != 0
```

A slice with a zero step is invalid.

### Accessors

```cpp
start()
stop()
step()
```

These return the original values specified by the user.

### Comparison

```cpp
operator==
operator!=
```

Two slices are equal when `start`, `stop`, and `step` are equal.

## Slice Semantics

NumCxx follows NumPy slice semantics:

- `start` is inclusive.
- `stop` is exclusive.
- Negative indices are interpreted relative to the end.
- Negative steps produce reversed slices.
- Missing `start` and `stop` use NumPy defaults.

For a dimension of length 5:

```cpp
slice()           -> [0,1,2,3,4]
slice(1,4)        -> [1,2,3]
slice(-3,-1)      -> [2,3]
slice({}, {}, -1) -> [4,3,2,1,0]
```

## Normalization

Most NumCxx internals operate on normalized slices rather than directly
using the user-provided `start`, `stop`, and `step`.

Normalization is performed by:

```cpp
slice::normalize(dim_len)
```

where:

```cpp
dim_len
```

is the length of the dimension being sliced.

The result is a:

```cpp
slice::normalized_slice
```

object.

## normalized_slice

```cpp
struct normalized_slice {
    size_type offset;
    size_type extent;
    size_type stride;
    bool reversed;
};
```

### offset

The starting position of the equivalent forward slice.

### extent

The length of the underlying half-open interval:

```cpp
[offset, offset + extent)
```

This follows `std::strided_slice` semantics.

Important:

```cpp
extent
```

is **not** the number of selected elements.

For example:

```cpp
slice(0, 10, 2)
```

produces:

```cpp
offset = 0
extent = 10
stride = 2
```

even though only five elements are selected.

### stride

The normalized stride.

`stride` is always positive.

### reversed

Indicates whether the original slice used a negative step:

```cpp
slice({}, {}, -1)
```

results in:

```cpp
reversed == true
```

## Design Notes

### Why is `reversed` needed?

NumPy supports negative strides directly:

```python
[::-1]
```

However, `std::strided_slice` only represents forward slices.

NumCxx therefore converts every slice into an equivalent forward slice and
records whether the original slice was reversed.

The original ordering is restored later by higher-level slicing and view
machinery.

### Why is `stride` always positive?

Keeping normalized strides positive allows NumCxx to reuse
`std::strided_slice` semantics and simplifies interaction with
`submdspan`.

Reverse traversal is represented separately by:

```cpp
reversed
```

rather than by a negative stride.

### Why is `extent` not the number of selected elements?

`normalized_slice` is designed to match `std::strided_slice`.

For a slice such as:

```cpp
slice(0, 10, 2)
```

the selected indices are:

```cpp
0, 2, 4, 6, 8
```

but the corresponding forward interval is:

```cpp
[0, 10)
```

Therefore:

```cpp
offset = 0
extent = 10
stride = 2
```

This representation makes conversion to `std::strided_slice` straightforward.

## Internal Usage

`normalize()` and `normalized_slice` are exposed for:

- unit testing;
- indexing and slicing internals;
- slice view construction;
- submdspan argument generation.

These APIs are not part of the stable public interface and may change in
future releases.

## Related Tests

### test_slice.cpp

Covers:

- construction;
- accessors;
- validation;
- equality comparison.

### test_slice_normalize.cpp

Covers:

- forward slices;
- reverse slices;
- empty slices;
- negative indices;
- clipping behavior;
- stride handling;
- NumPy compatibility edge cases.