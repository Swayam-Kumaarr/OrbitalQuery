"""
StackSTAC Adapter — thin bridge between StackSTAC datacube and the existing analysis pipeline.

This module provides a single function `stackstac_mosaic` that:
1. Takes real STAC items + bbox + bands
2. Uses StackSTAC to create a properly aligned datacube
3. Returns numpy arrays compatible with the existing change_detection pipeline

The key advantage over the manual rasterio mosaic path:
- StackSTAC handles CRS alignment, resolution, and bounds automatically
- No shape mismatch between periods (both go through the same grid definition)
- Lazy loading via Dask keeps memory usage controlled

Memory budget (Render free tier = 512 MB):
- 20m resolution for Delhi AOI: ~70 MB per period (safe)
- 10m resolution: ~280 MB (tight but feasible)
- Default: 20m (configurable via resolution parameter)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


def stackstac_mosaic(
    stac_items: list[dict[str, Any]],
    bbox: list[float],
    band_names: list[str],
    resolution: float = 20.0,
    epsg: int = 32643,
    max_dim: Optional[int] = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    """
    Create a mosaicked composite from multiple STAC items using StackSTAC.

    Args:
        stac_items: List of STAC item dicts (must be signed for Planetary Computer)
        bbox: [west, south, east, north] in EPSG:4326
        band_names: Band asset keys to extract (e.g. ["B08", "B11"])
        resolution: Target resolution in meters (default 20.0)
        epsg: Target EPSG code (default 32643 for Delhi region)
        max_dim: Maximum pixel dimension (if set, overrides resolution calculation)
        dtype: Output dtype (default float32)

    Returns:
        dict with:
            - data: np.ndarray (n_bands, height, width) — the mosaicked composite
            - band_names: list of band names (same order as input)
            - transform: affine transform
            - crs: coordinate reference system string
            - shape: [height, width]
            - resolution: actual resolution in meters
            - bounds: (west, south, east, north) in target CRS
            - nodata_mask: boolean mask of nodata pixels
            - scene_count: number of scenes used
            - scene_ids: list of scene IDs used
            - observation_dates: list of acquisition dates
    """
    if not stac_items:
        raise ValueError("No STAC items provided")

    import stackstac

    # Clear poisoned env vars that interfere with rasterio/GDAL
    for key in ['GDAL_CACHEMAX', 'GDAL_DISABLE_READDIR_ON_OPEN',
                'CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'GDAL_HTTP_TIMEOUT', 'GDAL_HTTP_MAX_RETRY']:
        if key in os.environ:
            del os.environ[key]

    logger.info(
        "[StackSTAC] Building mosaic: %d items, bands=%s, bbox=%s, resolution=%sm, epsg=%d",
        len(stac_items), band_names, bbox, resolution, epsg,
    )

    # Build the StackSTAC datacube
    # chunksize=(1, n_bands, 512, 512) keeps memory usage per-chunk at ~1MB
    # Use the requested dtype directly — float32 is fine for spectral indices
    stackstac_dtype = dtype
    try:
        cube = stackstac.stack(
            stac_items,
            assets=band_names,
            bounds_latlon=tuple(bbox),
            epsg=epsg,
            resolution=resolution,
            rescale=False,  # Keep raw integer values
            chunksize=(1, len(band_names), 512, 512),
            dtype=stackstac_dtype,
            fill_value=float("nan"),
        )
    except Exception as e:
        logger.error("[StackSTAC] stack() failed: %s", e)
        raise RuntimeError(f"StackSTAC stack failed: {e}")

    # Log cube metadata
    dims = dict(zip(cube.dims, cube.shape))
    logger.info("[StackSTAC] Cube shape: %s dims: %s", list(cube.shape), dims)

    # If max_dim is set and cube is larger, resample to fit
    if max_dim is not None:
        h, w = dims.get("y", 0), dims.get("x", 0)
        if h > max_dim or w > max_dim:
            scale = min(max_dim / h, max_dim / w) if h > 0 and w > 0 else 1.0
            new_h = int(h * scale)
            new_w = int(w * scale)
            logger.info(
                "[StackSTAC] Resampling from %dx%d to %dx%d (scale=%.2f)",
                w, h, new_w, new_h, scale,
            )
            cube = cube.coarsen(y=int(h / new_h), x=int(w / new_w)).mean()
            dims = dict(zip(cube.dims, cube.shape))

    # Extract the composite (median across time dimension)
    # This creates a single observation from multiple scenes
    logger.info("[StackSTAC] Computing temporal composite (median across %d observations)", dims.get("time", 1))

    # Load into memory — this is where actual data fetching happens
    # Use Dask's compute to materialize the lazy array
    try:
        import dask.array as da
        # Take the median across the time dimension
        if "time" in cube.dims and cube.sizes["time"] > 1:
            composite = cube.median(dim="time", skipna=True)
        else:
            composite = cube.squeeze(dim="time", drop=True) if "time" in cube.dims else cube

        # Compute to numpy
        if isinstance(composite.data, da.Array):
            result_array = composite.compute().values
        else:
            result_array = composite.values

    except Exception as e:
        logger.error("[StackSTAC] Compute failed: %s", e)
        # Fallback: try loading directly
        try:
            if "time" in cube.dims and cube.sizes["time"] > 1:
                result_array = cube.median(dim="time", skipna=True).values
            else:
                result_array = cube.squeeze(dim="time", drop=True).values if "time" in cube.dims else cube.values
        except Exception as e2:
            raise RuntimeError(f"StackSTAC compute failed: {e2}")

    # Ensure shape is (bands, height, width)
    if result_array.ndim == 2:
        result_array = result_array[np.newaxis, ...]  # Add band dim
    elif result_array.ndim == 4:
        # (time, bands, height, width) — already median'd, should be (bands, height, width)
        if result_array.shape[0] == 1:
            result_array = result_array[0]

    # Extract metadata from the xarray object
    try:
        transform = cube.rio.transform() if hasattr(cube, 'rio') else None
        crs_str = f"EPSG:{epsg}"
        bounds = cube.rio.bounds() if hasattr(cube, 'rio') else None
    except Exception:
        transform = None
        crs_str = f"EPSG:{epsg}"
        bounds = None

    # Compute nodata mask
    nodata_mask = np.isnan(result_array).all(axis=0)  # True where ALL bands are NaN

    # Extract scene metadata
    scene_ids = [item.get("id", "unknown") for item in stac_items]
    observation_dates = []
    for item in stac_items:
        dt = item.get("properties", {}).get("datetime", "")
        if dt:
            observation_dates.append(dt[:10])  # Just the date part

    h, w = result_array.shape[1], result_array.shape[2]

    logger.info(
        "[StackSTAC] Mosaic complete: shape=%s, scenes=%d, nodata=%.1f%%",
        result_array.shape, len(stac_items),
        nodata_mask.sum() / nodata_mask.size * 100,
    )

    return {
        "data": result_array,
        "band_names": band_names,
        "transform": transform,
        "crs": crs_str,
        "shape": [h, w],
        "resolution": resolution,
        "bounds": bounds,
        "nodata_mask": nodata_mask,
        "scene_count": len(stac_items),
        "scene_ids": scene_ids,
        "observation_dates": observation_dates,
    }


def stackstac_compute_index(
    stac_items: list[dict[str, Any]],
    bbox: list[float],
    index_name: str,
    band_map: dict[str, str],
    resolution: float = 20.0,
    epsg: int = 32643,
    max_dim: Optional[int] = None,
) -> dict[str, Any]:
    """
    Compute a spectral index directly from StackSTAC datacube.

    Instead of reading bands separately and computing the index, this builds
    a single StackSTAC cube with all required bands and computes the index
    within the aligned datacube. This guarantees perfect band alignment.

    Args:
        stac_items: List of STAC item dicts (must be signed for Planetary Computer)
        bbox: [west, south, east, north] in EPSG:4326
        index_name: Spectral index to compute (e.g. 'NDBI', 'NDVI')
        band_map: Dict mapping logical band names to physical asset keys
                   e.g. {'nir': 'B08', 'swir1': 'B11'} for NDBI
        resolution: Target resolution in meters
        epsg: Target EPSG code
        max_dim: Optional max dimension constraint

    Returns:
        dict with:
            - data: np.ndarray (height, width) — the computed index values
            - transform: affine transform
            - crs: EPSG string
            - shape: [height, width]
            - resolution: meters
            - bounds: (west, south, east, north) in target CRS
            - nodata_mask: boolean mask
            - scene_count: number of scenes
            - scene_ids: list of scene IDs
            - observation_dates: list of dates
    """
    physical_bands = list(band_map.values())
    logger.info(
        "[StackSTAC] Computing index %s: bands=%s (physical=%s), resolution=%sm",
        index_name, list(band_map.keys()), physical_bands, resolution,
    )

    # Build the StackSTAC cube with all required bands
    result = stackstac_mosaic(
        stac_items=stac_items,
        bbox=bbox,
        band_names=physical_bands,
        resolution=resolution,
        epsg=epsg,
        max_dim=max_dim,
        dtype="float32",  # float32 saves 50% memory vs float64 — sufficient precision for spectral indices
    )

    data = result["data"]  # (n_bands, height, width)
    h, w = data.shape[1], data.shape[2]
    nodata_mask = result["nodata_mask"]

    # Extract individual bands
    band_arrays = {}
    for i, phys in enumerate(physical_bands):
        band_arrays[phys] = data[i]

    # Compute the index using the same formulas as spectral_indices.py
    from app.services.spectral_indices import INDEX_DEFINITIONS
    index_def = INDEX_DEFINITIONS.get(index_name)
    if index_def and hasattr(index_def, 'formula'):
        # Build named band references for the formula
        # Map physical band names back to formula variables
        logical_to_physical = band_map  # e.g. {'nir': 'B08', 'swir1': 'B11'}
        formula_vars = {}
        for logical, physical in logical_to_physical.items():
            if physical in band_arrays:
                formula_vars[logical] = band_arrays[physical]
                # Also map by uppercase band name (B08, B11, etc.)
                formula_vars[physical] = band_arrays[physical]

        try:
            # Try direct eval with numpy
            import numpy as _np_local
            safe_dict = {**formula_vars, 'np': _np_local, 'numpy': _np_local, 'nan': _np_local.nan}
            # Also add commonly used band references
            for k, v in band_arrays.items():
                safe_dict[k] = v

            index_array = eval(index_def.formula, {"__builtins__": {}}, safe_dict)
            logger.info(
                "[StackSTAC] Index %s computed via formula: shape=%s mean=%.4f",
                index_name, index_array.shape,
                float(_np_local.nanmean(index_array)),
            )
        except Exception as e:
            logger.warning("[StackSTAC] Formula eval failed: %s, using manual computation", e)
            # Fallback: manual computation for known indices
            if index_name == "NDBI" and "B08" in band_arrays and "B11" in band_arrays:
                nir = band_arrays["B08"].astype(_np_local.float32)
                swir = band_arrays["B11"].astype(_np_local.float32)
                denom = nir + swir
                denom[denom == 0] = _np_local.nan
                index_array = (swir - nir) / denom
            elif index_name == "NDVI" and "B08" in band_arrays and "B04" in band_arrays:
                nir = band_arrays["B08"].astype(_np_local.float32)
                red = band_arrays["B04"].astype(_np_local.float32)
                denom = nir + red
                denom[denom == 0] = _np_local.nan
                index_array = (nir - red) / denom
            else:
                raise RuntimeError(f"Cannot compute {index_name} with available bands: {list(band_arrays.keys())}")
    else:
        raise RuntimeError(f"Index {index_name} not found in INDEX_DEFINITIONS")

    result["data"] = index_array
    result["shape"] = [h, w]
    result["band_names"] = [index_name]
    return result


def stackstac_mosaic_single_band(
    stac_items: list[dict[str, Any]],
    bbox: list[float],
    band_name: str,
    resolution: float = 20.0,
    epsg: int = 32643,
) -> dict[str, Any]:
    """
    Convenience wrapper for single-band mosaicking.
    Returns shape (height, width) instead of (1, height, width).
    """
    result = stackstac_mosaic(
        stac_items=stac_items,
        bbox=bbox,
        band_names=[band_name],
        resolution=resolution,
        epsg=epsg,
    )
    # Squeeze out band dimension
    if result["data"].ndim == 3 and result["data"].shape[0] == 1:
        result["data"] = result["data"][0]
    return result
