"""Block-mean downsampling. Accumulates in a wider dtype to avoid overflow;
never operates on a memmap (there are none in this codebase -- read chunks
via mrcng.reader first).

factors are given in the same (z, y, x) axis order as the array shape.
"""
from __future__ import annotations

import numpy as np


def _sum_and_count(arr: np.ndarray, factor: int, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Sum `arr` in blocks of `factor` along `axis`, plus the per-block voxel
    count broadcastable against the result. The trailing block is short
    wherever the axis is not divisible by the factor."""
    n = arr.shape[axis]
    indices = np.arange(0, n, factor)
    summed = np.add.reduceat(arr, indices, axis=axis)
    counts = np.diff(np.append(indices, n))
    shape = [1] * arr.ndim
    shape[axis] = len(counts)
    return summed, counts.reshape(shape)


def block_mean(arr: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    """Mean of each (fz, fy, fx) block, as one division at the end.

    All three axes are summed in the accumulator dtype before dividing, so
    there is no intermediate rounding. Reducing one axis at a time and taking
    a mean per axis is *not* equivalent: it truncates a fractional mean back to
    the integer accumulator on every axis after the first, which biases every
    voxel toward zero and -- because the pyramid cascades level from level --
    compounds that bias down the pyramid.
    """
    src_dtype = arr.dtype
    floating = np.issubdtype(src_dtype, np.floating)
    # ponytail: int32 accumulator per spec section 7. Sums prod(factors) voxels,
    # so it holds for int16/uint16 inputs up to prod(factors) ~ 65000; the
    # pyramid only ever passes per-step factors of 1 or 2. Widen to int64 if a
    # caller ever needs a single large-factor reduction.
    acc = arr.astype(np.float64 if floating else np.int32)

    divisor = 1
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        acc, counts = _sum_and_count(acc, factor, axis)
        divisor = divisor * counts

    result = acc / divisor

    if floating:
        return result.astype(src_dtype)

    rounded = np.where(result >= 0, np.floor(result + 0.5), np.ceil(result - 0.5))
    info = np.iinfo(src_dtype)
    clipped = np.clip(rounded, info.min, info.max)
    return clipped.astype(src_dtype)
