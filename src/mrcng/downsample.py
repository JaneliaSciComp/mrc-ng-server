"""Block-mean downsampling. Accumulates in a wider dtype to avoid overflow;
never operates on a memmap (there are none in this codebase -- read chunks
via mrcng.reader first).

factors are given in the same (z, y, x) axis order as the array shape.
"""
from __future__ import annotations

import numpy as np


def _reduceat_mean_1d(arr: np.ndarray, factor: int, axis: int, accum_dtype) -> np.ndarray:
    n = arr.shape[axis]
    indices = np.arange(0, n, factor)
    counts = np.diff(np.append(indices, n))
    summed = np.add.reduceat(arr.astype(accum_dtype), indices, axis=axis)
    shape = [1] * arr.ndim
    shape[axis] = len(counts)
    counts = counts.reshape(shape)
    return summed / counts


def block_mean(arr: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    src_dtype = arr.dtype
    if np.issubdtype(src_dtype, np.floating):
        accum_dtype = np.float64
    else:
        accum_dtype = np.int32

    result = arr
    for axis, factor in enumerate(factors):
        if factor == 1:
            continue
        result = _reduceat_mean_1d(result, factor, axis, accum_dtype)

    if np.issubdtype(src_dtype, np.floating):
        return result.astype(src_dtype)

    rounded = np.where(result >= 0, np.floor(result + 0.5), np.ceil(result - 0.5))
    info = np.iinfo(src_dtype)
    clipped = np.clip(rounded, info.min, info.max)
    return clipped.astype(src_dtype)
