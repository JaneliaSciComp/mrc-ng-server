"""OME-Zarr 0.5 (Zarr v3) metadata, chunk addressing and edge padding.

Derivation module: everything here lands in cache files, so a change to it
needs a fingerprint.DERIVATION_VERSION bump (see CLAUDE.md).

Array order is (z, y, x) on the wire and on disk, matching precomputed raw
chunks byte for byte in the interior. The one difference is the edge: Zarr
chunks are always chunk_shape-sized, padded with fill_value, where precomputed
clips them. Level 0 is never on disk; omezarr/ is a serving cache, not a
standalone store.
"""
from __future__ import annotations

import numpy as np

FORMATS = ("precomputed", "omezarr")

_CODECS = [{"name": "bytes", "configuration": {"endian": "little"}}]


def chunk_rel_path(level: int, kz: int, ky: int, kx: int) -> str:
    """Path of one chunk relative to the omezarr/ directory ("default" key
    encoding, "/" separator)."""
    return f"{level}/c/{kz}/{ky}/{kx}"


def chunk_region(size_xyz, chunk_size_xyz, kz: int, ky: int, kx: int):
    """(x0, x1, y0, y1, z0, z1) of the *data* a grid index covers, clipped to
    the level. Raises ValueError when the chunk origin is outside the level."""
    sx, sy, sz = size_xyz
    cx, cy, cz = chunk_size_xyz
    x0, y0, z0 = kx * cx, ky * cy, kz * cz
    if x0 >= sx or y0 >= sy or z0 >= sz:
        raise ValueError(f"chunk index {(kz, ky, kx)} outside level of size {tuple(size_xyz)}")
    return x0, min(x0 + cx, sx), y0, min(y0 + cy, sy), z0, min(z0 + cz, sz)


def pad_chunk(arr: np.ndarray, chunk_shape_zyx) -> np.ndarray:
    """Zero-pad a clipped (z, y, x) block up to chunk_shape. Interior chunks
    come back as the same object, no copy."""
    if arr.shape == tuple(chunk_shape_zyx):
        return arr
    out = np.zeros(tuple(chunk_shape_zyx), dtype=arr.dtype)
    out[: arr.shape[0], : arr.shape[1], : arr.shape[2]] = arr
    return out


def build_array_json(size_xyz, chunk_size_xyz, dtype: np.dtype) -> dict:
    return {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [int(s) for s in reversed(tuple(size_xyz))],
        # numpy and Zarr v3 spell int8/uint8/int16/uint16/float32 identically.
        "data_type": np.dtype(dtype).name,
        "chunk_grid": {"name": "regular",
                       "configuration": {"chunk_shape": [int(c) for c in reversed(tuple(chunk_size_xyz))]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": _CODECS,
        "dimension_names": ["z", "y", "x"],
        "attributes": {},
    }


def build_group_json(hdr, scales, name: str) -> dict:
    # Same numbers precomputed info uses (angstrom / 10), so both endpoints
    # show the same scale bar. Index order is (x, y, z) in hdr and scales,
    # (z, y, x) in OME.
    base_nm = [a / 10.0 for a in hdr.voxel_size_angstrom]
    datasets = []
    for i, lvl in enumerate(scales):
        fx, fy, fz = lvl.factors
        transforms = [{"type": "scale", "scale": [base_nm[2] * fz, base_nm[1] * fy, base_nm[0] * fx]}]
        if i > 0:
            # NGFF indexes the *centre* of a sample. A mean over f source voxels
            # is centred (f-1)/2 source voxels in from the origin. Precomputed
            # needs no such offset: Neuroglancer's precomputed model is
            # corner-aligned.
            transforms.append({"type": "translation", "translation": [
                (fz - 1) / 2 * base_nm[2], (fy - 1) / 2 * base_nm[1], (fx - 1) / 2 * base_nm[0],
            ]})
        datasets.append({"path": str(i), "coordinateTransformations": transforms})
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "multiscales": [{
                    "name": name,
                    "axes": [{"name": ax, "type": "space", "unit": "nanometer"} for ax in ("z", "y", "x")],
                    "datasets": datasets,
                }],
            },
            # "ome" is reserved by the spec; the non-spec flags precomputed puts
            # at the top level of info live under our own key here.
            "mrcng": {"voxel_size_is_default": hdr.voxel_size_is_default,
                      "is_image_stack": hdr.is_image_stack},
        },
    }
