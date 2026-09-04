"""Raster analysis service — memory-optimized for Render free tier (512MB).

Uses aggressive GDAL settings and small windows to stay within memory limits.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np

# NOTE: GDAL env vars are set INSIDE rasterio.Env() blocks, not at module level.
# Setting GDAL_CACHEMAX="64" (string) at module level causes
# TypeError: an integer is required when GDAL reads the env var before rasterio.Env().

logger = logging.getLogger(__name__)


def read_raster_window(
    href: str,
    bbox: list[float],
    bands: Optional[list[str]] = None,
    max_dim: int = 512,
    auth: Optional[str] = None,
) -> dict[str, Any]:
    """
    Read a small raster window for a given bounding box.

    Uses aggressive memory limits to stay under 512MB on Render free tier.
    max_dim=1024 → 1024² float32 = 4MB per band (safe).

    Args:
        href: URL or path to raster file
        bbox: Bounding box [west, south, east, north] in WGS-84
        bands: Optional list of band names/indices to read
        max_dim: Maximum dimension for windowed read
        auth: Optional auth provider name (e.g. "earthdata") for authenticated URLs

    Returns dict with:
        - data: numpy array (bands, height, width)
        - band_names: list of band names
        - profile: rasterio profile
        - window_shape: [bands, height, width]
        - transform: affine transform
        - crs: coordinate reference system
    """
    import numpy as np
    import rasterio
    from rasterio.windows import from_bounds as window_from_bounds

    logger.info("Opening raster: %s (max_dim=%d, auth=%s)", href[:120], max_dim, auth)

    # NUCLEAR FIX: Clear ALL GDAL env vars that may have been set as strings
    # at module level by old code. The module-level os.environ.setdefault(
    # "GDAL_CACHEMAX", "64") sets a string, which GDAL reads before
    # rasterio.Env() can override it, causing TypeError.
    import os
    for _gdal_key in ['GDAL_CACHEMAX', 'GDAL_DISABLE_READDIR_ON_OPEN',
                       'CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'GDAL_HTTP_TIMEOUT',
                       'GDAL_HTTP_MAX_RETRY', 'GDAL_HTTP_USERPWD']:
        if _gdal_key in os.environ:
            del os.environ[_gdal_key]

    # Build GDAL environment settings
    gdal_env = {
        "GDAL_CACHEMAX": 64,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "3",
    }

    # Add authentication credentials if needed
    if auth == "earthdata":
        try:
            from app.services.earthdata_auth import get_earthdata_userpwd
            userpwd = get_earthdata_userpwd()
            if userpwd:
                gdal_env["GDAL_HTTP_USERPWD"] = userpwd
                logger.info("[EarthdataAuth] GDAL_HTTP_USERPWD set for rasterio read")
            else:
                logger.warning("[EarthdataAuth] No credentials configured for Earthdata URL")
        except Exception as e:
            logger.warning("[EarthdataAuth] Failed to load credentials: %s", e)

    with rasterio.Env(**gdal_env):
        with rasterio.open(href) as src:
            logger.info(
                "Raster info: %d bands, size=%dx%d, crs=%s",
                src.count, src.width, src.height, src.crs,
            )

            # Transform bbox from WGS-84 to raster CRS if needed
            bbox_native = bbox
            src_crs_str = str(src.crs)
            if "EPSG:4326" not in src_crs_str and src.crs is not None:
                try:
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(
                        "EPSG:4326", src.crs, always_xy=True
                    )
                    west, south = transformer.transform(bbox[0], bbox[1])
                    east, north = transformer.transform(bbox[2], bbox[3])
                    bbox_native = [west, south, east, north]
                except Exception as e:
                    logger.warning("CRS transform failed, using raw bbox: %s", e)

            # Create window — cap for 512MB RAM
            window = None
            try:
                # Compute pixel coordinates from bbox + transform
                inv_transform = ~src.transform
                col_off, row_off = inv_transform * (bbox_native[0], bbox_native[3])
                col_off2, row_off2 = inv_transform * (bbox_native[2], bbox_native[1])
                
                # Ensure integer pixel coords
                col_off = int(max(0, col_off))
                row_off = int(max(0, row_off))
                win_width = int(min(col_off2 - col_off, src.width - col_off))
                win_height = int(min(row_off2 - row_off, src.height - row_off))
                
                if win_width <= 0 or win_height <= 0:
                    logger.warning("Window too small or outside raster, reading full extent")
                else:
                    # Cap to max_dim
                    if win_width > max_dim or win_height > max_dim:
                        scale = min(max_dim / win_width, max_dim / win_height)
                        win_width = int(win_width * scale)
                        win_height = int(win_height * scale)
                    
                    window = rasterio.windows.Window(col_off, row_off, win_width, win_height)
                    logger.info("Window: col=%d, row=%d, w=%d, h=%d", col_off, row_off, win_width, win_height)
            except Exception as e:
                logger.warning("Could not create window: %s", e)
                window = None

            # Determine which bands to read (single band at a time to save memory)
            if bands and src.count >= 1:
                band_names = bands
                band_indices = []
                for b in bands:
                    try:
                        band_indices.append(int(b) if b.isdigit() else 1)
                    except (ValueError, rasterio.errors.BandNotFoundError):
                        band_indices.append(1)
            else:
                band_count = min(src.count, 2)  # Read max 2 bands
                band_indices = list(range(1, band_count + 1))
                band_names = [f"band_{i}" for i in band_indices]

            # Read bands one at a time to minimize memory
            band_arrays = []
            for idx in band_indices:
                if window is not None:
                    data = src.read(indexes=[idx], window=window)
                else:
                    data = src.read(indexes=[idx])
                band_arrays.append(data[0])  # shape: (height, width)

            # Stack into (bands, height, width)
            import numpy as np
            data = np.stack(band_arrays, axis=0) if len(band_arrays) > 1 else band_arrays[0][np.newaxis, ...]

            profile = src.profile.copy()
            if window is not None:
                transform = src.window_transform(window)
            else:
                transform = src.transform

            logger.info("Read data shape: %s, dtype: %s", data.shape, data.dtype)

            return {
                "data": data,
                "band_names": band_names,
                "profile": profile,
                "window_shape": list(data.shape),
                "transform": transform,
                "crs": str(src.crs),
                "dtype": str(data.dtype),
            }


def compute_band_stats(
    data: Any,
    band_names: list[str],
) -> list:
    """Compute statistics for each band."""
    import numpy as np
    from app.models.requests import BandStats

    stats_list = []
    nodata_value = -9999

    for i, name in enumerate(band_names):
        if data.ndim == 3:
            band_data = data[i].astype(np.float64)
        else:
            band_data = data.astype(np.float64)

        valid = band_data[(band_data != 0) & (band_data > nodata_value)]
        nodata_count = int(np.sum((band_data == 0) | (band_data <= nodata_value)))

        if len(valid) == 0:
            stats_list.append(BandStats(
                band=name, dtype=str(data.dtype),
                shape=list(band_data.shape),
                min=0.0, max=0.0, mean=0.0, std=0.0,
                nodata_count=nodata_count,
            ))
        else:
            stats_list.append(BandStats(
                band=name, dtype=str(data.dtype),
                shape=list(band_data.shape),
                min=float(np.min(valid)), max=float(np.max(valid)),
                mean=float(np.mean(valid)), std=float(np.std(valid)),
                nodata_count=nodata_count,
            ))

    return stats_list


def estimate_resolution_meters(profile: dict, crs: str) -> Optional[float]:
    """Estimate spatial resolution in meters from the raster profile."""
    try:
        transform = profile.get("transform")
        if transform is None:
            return None
        pixel_size_x = abs(transform.a)
        pixel_size_y = abs(transform.e)
        if "EPSG:4326" in str(crs):
            return pixel_size_x * 111_000
        return (pixel_size_x + pixel_size_y) / 2
    except Exception:
        return None


def is_rasterio_compatible(href: str) -> bool:
    """Check if a URL can be opened by rasterio."""
    import rasterio
    try:
        with rasterio.open(href) as src:
            return src.count > 0
    except Exception:
        return False


def reproject_to_grid(
    source_array: np.ndarray,
    source_crs: Any,
    source_transform: Any,
    target_crs: Any,
    target_transform: Any,
    target_shape: tuple[int, int],
    resampling_method: str = "bilinear",
) -> np.ndarray:
    """
    Reproject a source array onto a target grid.

    Uses rasterio.warp.reproject for accurate geographic resampling.
    This ensures both periods are compared on the exact same pixel grid.

    Args:
        source_array: 2D numpy array (H, W) to reproject
        source_crs: CRS of the source array
        source_transform: Affine transform of the source array
        target_crs: CRS of the target grid
        target_transform: Affine transform of the target grid
        target_shape: (height, width) of the target grid
        resampling_method: 'bilinear' for continuous data, 'nearest' for categorical

    Returns:
        2D numpy array on the target grid
    """
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS as RCRS

    h, w = target_shape
    destination = np.full((h, w), np.nan, dtype=np.float32)

    # Resolve resampling method
    resampling_map = {
        "bilinear": Resampling.bilinear,
        "nearest": Resampling.nearest,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }
    resampling = resampling_map.get(resampling_method, Resampling.bilinear)

    # Convert CRS if needed
    if not isinstance(source_crs, RCRS):
        source_crs = RCRS.from_user_input(source_crs)
    if not isinstance(target_crs, RCRS):
        target_crs = RCRS.from_user_input(target_crs)

    logger.info(
        "Reprojecting %s array (%dx%d) → target grid (%dx%d, %s)",
        source_crs, source_array.shape[1], source_array.shape[0],
        w, h, target_crs,
    )

    reproject(
        source=source_array.astype(np.float32),
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        resampling=resampling,
    )

    valid_count = int(np.sum(~np.isnan(destination)))
    logger.info(
        "Reprojection complete: %d valid pixels (%.1f%%)",
        valid_count, valid_count / (h * w) * 100,
    )

    return destination


def determine_common_grid(
    href1: str,
    href2: str,
    bbox: list[float],
    target_crs: Optional[str] = None,
    max_dim: int = 512,
    auth: Optional[str] = None,
) -> dict[str, Any]:
    """
    Determine a common analysis grid from two raster sources.

    Opens both rasters to read their CRS, resolution, and bounds.
    Then constructs a deterministic common grid that covers the AOI.

    Args:
        href1: URL/path to first raster
        href2: URL/path to second raster
        bbox: AOI bounding box [west, south, east, north] in WGS-84
        target_crs: Optional target CRS override
        max_dim: Maximum pixel dimension
        auth: Optional auth provider name (e.g. "earthdata")

    The common grid uses:
    - CRS: from the first raster (or target_crs if specified)
    - Resolution: native resolution of the first raster (no degradation)
    - Bounds: intersection of both rasters' overlap with the AOI bbox
    - Transform: computed from the above

    Returns dict with:
        - crs, transform, width, height, bounds
        - resolution_meters
        - crs1, crs2 (both source CRS for logging)
    """
    import rasterio
    import os
    from rasterio.crs import CRS as RCRS
    from pyproj import Transformer

    # Clear poisoned env vars
    for key in ['GDAL_CACHEMAX', 'GDAL_DISABLE_READDIR_ON_OPEN',
                'CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'GDAL_HTTP_TIMEOUT',
                'GDAL_HTTP_MAX_RETRY', 'GDAL_HTTP_USERPWD']:
        if key in os.environ:
            del os.environ[key]

    # Build GDAL env settings
    gdal_env = {
        "GDAL_CACHEMAX": 64,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "3",
    }

    # Add auth if needed
    if auth == "earthdata":
        try:
            from app.services.earthdata_auth import get_earthdata_userpwd
            userpwd = get_earthdata_userpwd()
            if userpwd:
                gdal_env["GDAL_HTTP_USERPWD"] = userpwd
        except Exception:
            pass

    rasters = []
    for href in [href1, href2]:
        with rasterio.Env(**gdal_env):
            with rasterio.open(href) as src:
                rasters.append({
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                    "resolution_x": abs(src.transform.a),
                    "resolution_y": abs(src.transform.e),
                })

    # Use first raster's CRS as analysis CRS (unless overridden)
    analysis_crs = RCRS.from_user_input(target_crs) if target_crs else rasters[0]["crs"]
    analysis_resolution = rasters[0]["resolution_x"]  # preserve native resolution

    # Transform AOI bbox to analysis CRS
    transformer_to_analysis = Transformer.from_crs(
        RCRS.from_epsg(4326), analysis_crs, always_xy=True
    )
    west_a, south_a = transformer_to_analysis.transform(bbox[0], bbox[1])
    east_a, north_a = transformer_to_analysis.transform(bbox[2], bbox[3])

    # Compute pixel grid from AOI bounds + resolution
    width = max(1, int(abs(east_a - west_a) / analysis_resolution))
    height = max(1, int(abs(north_a - south_a) / analysis_resolution))

    # Cap dimensions for memory safety
    if width > max_dim or height > max_dim:
        scale = min(max_dim / width, max_dim / height)
        width = int(width * scale)
        height = int(height * scale)
        # Recalculate resolution to fit
        analysis_resolution = abs(east_a - west_a) / width

    from rasterio.transform import from_bounds
    analysis_transform = from_bounds(west_a, south_a, east_a, north_a, width, height)

    logger.info(
        "Common grid: crs=%s, res=%.2fm, %dx%d, bounds=(%.4f,%.4f,%.4f,%.4f)",
        analysis_crs, analysis_resolution, width, height,
        west_a, south_a, east_a, north_a,
    )
    logger.info(
        "Source CRS: scene1=%s, scene2=%s",
        rasters[0]["crs"], rasters[1]["crs"],
    )

    return {
        "crs": analysis_crs,
        "transform": analysis_transform,
        "width": width,
        "height": height,
        "bounds": (west_a, south_a, east_a, north_a),
        "resolution_meters": analysis_resolution,
        "source_crs1": str(rasters[0]["crs"]),
        "source_crs2": str(rasters[1]["crs"]),
    }
