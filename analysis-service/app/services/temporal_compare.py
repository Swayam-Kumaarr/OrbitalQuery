"""
Temporal Comparison Pipeline — the single reusable engine for all EO phenomena.

Query → discover scenes → select best → compute indices → change detection → metrics

This module orchestrates the full before/after comparison pipeline.
It does NOT hard-code any phenomenon — it uses the capability registry
to determine which indices, bands, and thresholds to use.

The same pipeline handles:
  - Hyderabad urban expansion 2021 → 2025
  - Kerala flood impact August 2024
  - Himalayan glacier retreat 2018 → 2025
  - Amazon deforestation
  - Chennai coastal erosion
  - etc.

Architecture:
  plan (phenomenon, bbox, dates)
    → search_period_1() → best_scene_t1
    → search_period_2() → best_scene_t2
    → compute_index_t1(index, bands)
    → compute_index_t2(index, bands)
    → change_detection(t1, t2)
    → metrics + explanation
"""

from __future__ import annotations

import gc
import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Optional

# Heavy imports are deferred to function bodies to keep startup memory under 500MB
# on Render free tier. numpy, rasterio, scipy etc. are only loaded when an analysis runs.
import numpy as np  # numpy is needed for type hints in dataclasses — keep this one

# These are loaded lazily inside run_temporal_comparison()
_PHENOMENON_REGISTRY = None
_INDEX_DEFINITIONS = None
_INDEX_BAND_MAP = None
_SENSOR_BANDS = None
_compute_index_from_bands = None
_run_change_detection = None
_get_default_provider = None
_get_provider = None
_select_scenes_for_period = None
_check_periods_compatible = None
_SceneSelectionResult = None

def _lazy_import_heavy():
    """Load heavy modules only when analysis actually runs."""
    global _PHENOMENON_REGISTRY, _INDEX_DEFINITIONS, _INDEX_BAND_MAP
    global _SENSOR_BANDS, _compute_index_from_bands, _run_change_detection
    global _get_default_provider, _get_provider
    global _select_scenes_for_period, _check_periods_compatible, _SceneSelectionResult
    if _PHENOMENON_REGISTRY is not None:
        return  # already loaded
    from app.services.capability_registry import PHENOMENON_REGISTRY, ANALYSIS_TYPES, get_analysis_config
    from app.services.change_detection import run_change_detection
    from app.services.eo_provider import get_default_provider, get_provider
    from app.services.spectral_indices import INDEX_DEFINITIONS, INDEX_BAND_MAP, SENSOR_BANDS, compute_index_from_bands
    from app.services.scene_selector import select_scenes_for_period, check_periods_compatible, SceneSelectionResult
    _PHENOMENON_REGISTRY = PHENOMENON_REGISTRY
    _INDEX_DEFINITIONS = INDEX_DEFINITIONS
    _INDEX_BAND_MAP = INDEX_BAND_MAP
    _SENSOR_BANDS = SENSOR_BANDS
    _compute_index_from_bands = compute_index_from_bands
    _run_change_detection = run_change_detection
    _get_default_provider = get_default_provider
    _get_provider = get_provider
    _select_scenes_for_period = select_scenes_for_period
    _check_periods_compatible = check_periods_compatible
    _SceneSelectionResult = SceneSelectionResult
    logger.info("Heavy modules loaded (numpy, rasterio, scipy, planetary_computer)")

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────

@dataclass
class SceneSelection:
    """A selected scene for one time period."""
    item_id: str
    collection: str
    datetime: str
    cloud_cover: Optional[float]
    bbox: list[float]
    provider: str
    platform: str
    score: float
    assets: dict[str, Any]


@dataclass
class IndexResult:
    """Result of computing a spectral index for one time period."""
    index_name: str
    value: Optional[np.ndarray]
    stats: dict[str, float]
    scene_id: str
    date: str
    resolution_m: float
    shape: list[int]
    valid_pixels: int
    total_pixels: int
    method: str = "mosaic"  # 'stackstac' or 'mosaic'


@dataclass
class TemporalComparisonResult:
    """Complete result of a temporal comparison analysis."""
    status: str
    plan_id: str
    phenomenon: str
    analysis_type: str
    aoi_name: str
    aoi_bbox: list[float]

    # Time periods
    period1: dict[str, Any]  # {start, end, scene}
    period2: dict[str, Any]

    # Scene selections
    scene_t1: Optional[SceneSelection]
    scene_t2: Optional[SceneSelection]

    # Index results
    index_t1: Optional[IndexResult]
    index_t2: Optional[IndexResult]

    # Change detection
    change_detection: Optional[dict[str, Any]]

    # Change mask visualization (base64-encoded PNGs)
    change_visualizations: Optional[dict[str, Any]]

    # Computed metrics (the key numbers for the UI)
    metrics: dict[str, Any]

    # Imagery URLs for visualization
    imagery: dict[str, Any]

    # Processing metadata
    processing_steps: list[dict[str, str]]
    sensor_info: dict[str, Any]

    # Processing provenance (Sentinel Hub / Copernicus research patterns)
    provenance: dict[str, Any]

    # Explanation
    explanation: dict[str, Any]


# ── Scene search & selection ──────────────────────────────────────

def _search_scenes(
    collection: str,
    bbox: list[float],
    start_date: str,
    end_date: str,
    max_cloud_cover: int,
    limit: int = 15,
    provider_name: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Search STAC for scenes in a time window."""
    _lazy_import_heavy()
    provider = _get_provider(provider_name) if provider_name else _get_default_provider()

    datetime_str = f"{start_date}/{end_date}"

    try:
        result = provider.search(
            collection=collection,
            bbox=bbox,
            datetime=datetime_str,
            max_cloud_cover=max_cloud_cover,
            limit=limit,
        )
        return result.items
    except Exception as e:
        logger.warning("Scene search failed for %s: %s", collection, e)
        return []


def _score_scene(
    item: dict[str, Any],
    target_bbox: list[float],
    max_cloud_cover: int,
) -> float:
    """
    Score a scene for suitability.
    Higher = better. Factors:
    - Cloud cover (lower is better)
    - Spatial coverage (more overlap with AOI is better)
    - Data quality flags
    """
    score = 0.5  # base

    props = item.get("properties", {})
    cloud_cover = props.get("eo:cloud_cover", 50)
    if cloud_cover is not None:
        if cloud_cover <= 10:
            score += 0.3
        elif cloud_cover <= 20:
            score += 0.2
        elif cloud_cover <= max_cloud_cover:
            score += 0.1
        else:
            score -= 0.2

    # Spatial overlap score
    item_bbox = item.get("bbox", [])
    if item_bbox and len(item_bbox) == 4 and target_bbox:
        overlap = _bbox_overlap(item_bbox, target_bbox)
        total_area = _bbox_area(target_bbox)
        if total_area > 0:
            coverage = overlap / total_area
            score += min(coverage * 0.2, 0.2)

    return min(max(score, 0.0), 1.0)


def _bbox_overlap(a: list[float], b: list[float]) -> float:
    """Compute area of overlap between two bounding boxes [west, south, east, north]."""
    west = max(a[0], b[0])
    south = max(a[1], b[1])
    east = min(a[2], b[2])
    north = min(a[3], b[3])

    if west >= east or south >= north:
        return 0.0

    return (east - west) * (north - south)


def _bbox_area(bbox: list[float]) -> float:
    """Compute area of a bounding box in degrees²."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _select_best_scene(
    items: list[dict[str, Any]],
    target_bbox: list[float],
    max_cloud_cover: int,
) -> Optional[dict[str, Any]]:
    """Select the best scene from search results."""
    if not items:
        return None

    scored = [
        (item, _score_scene(item, target_bbox, max_cloud_cover))
        for item in items
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[0][0] if scored else None


def _scene_to_selection(item: dict[str, Any], provider_name: str) -> SceneSelection:
    """Convert a STAC item to a SceneSelection."""
    props = item.get("properties", {})
    return SceneSelection(
        item_id=item.get("id", "unknown"),
        collection=item.get("collection", "unknown"),
        datetime=props.get("datetime", ""),
        cloud_cover=props.get("eo:cloud_cover"),
        bbox=item.get("bbox", []),
        provider=provider_name,
        platform=props.get("platform", "unknown"),
        score=0.0,
        assets=item.get("assets", {}),
    )


def _get_imagery_urls(scene: SceneSelection) -> dict[str, str]:
    """Extract imagery URLs from a scene's assets + construct TileJSON URL.

    Returns tilejson URL and bounds so the frontend can fetch the signed
    tile template from Planetary Computer directly. The tilejson endpoint
    returns tiles that work WITHOUT signing — we just need to fetch it
    to get the correct tile template and spatial bounds.
    """
    urls = {}
    assets = scene.assets

    # Thumbnail / preview
    for key in ["thumbnail", "rendered_preview", "visual", "preview"]:
        if key in assets:
            href = assets[key].get("href", "") if isinstance(assets[key], dict) else assets[key]
            if href:
                urls["thumbnail"] = href
                break

    # Rendered image
    for key in ["rendered_preview", "visual"]:
        if key in assets:
            href = assets[key].get("href", "") if isinstance(assets[key], dict) else assets[key]
            if href:
                urls["rendered"] = href
                break

    # TileJSON URL — the frontend fetches this to get the signed tile template + bounds.
    # pc.sign() fails silently on Render free tier, so we return the unsigned tilejson
    # URL. The PC tilejson endpoint returns tile URLs that work WITHOUT authentication.
    if scene.collection and scene.item_id:
        tilejson_url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/tilejson.json"
            f"?collection={scene.collection}"
            f"&item={scene.item_id}"
            f"&assets=visual"
            f"&asset_bidx=visual%7C1%2C2%2C3"
        )
        urls["tilejson"] = tilejson_url

        # Also provide the direct tile URL as fallback (may not work without signing)
        tile_url = (
            f"https://planetarycomputer.microsoft.com/api/data/v1/item/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}@1x"
            f"?collection={scene.collection}"
            f"&item={scene.item_id}"
            f"&assets=visual"
        )
        urls["tile_url"] = tile_url

    # Also check assets for tilejson (backup)
    if "tilejson" not in urls and "tilejson" in assets:
        href = assets["tilejson"].get("href", "") if isinstance(assets["tilejson"], dict) else assets["tilejson"]
        if href:
            urls["tilejson"] = href

    # Individual bands (for index computation)
    for key, asset in assets.items():
        if isinstance(asset, dict):
            if key.upper().startswith("B") or key.lower() in ("vv", "vh", "red", "green", "blue", "nir", "swir16", "swir22"):
                href = asset.get("href", "")
                if href:
                    urls[f"band_{key}"] = href

    return urls


# ── Multi-scene index computation ─────────────────────────────────

def _compute_index_from_mosaic_scenes(
    index_name: str,
    sensor: str,
    scenes: list[dict[str, Any]],
    bbox: list[float],
    period_label: str,
) -> IndexResult:
    """
    Compute a spectral index from multiple scenes using mosaicking.

    For each required band:
    1. Collect the band href from each scene's assets
    2. Use the mosaic module to read + composite all scenes
    3. Compute the spectral index from the composited bands

    Falls back to single-scene if only one scene is provided.
    """
    _lazy_import_heavy()

    # Get band mapping
    band_key = (sensor, index_name)
    band_map = _INDEX_BAND_MAP.get(band_key)
    if not band_map:
        raise ValueError(f"No band mapping for {index_name} on {sensor}")

    index_def = _INDEX_DEFINITIONS.get(index_name)
    if not index_def:
        raise ValueError(f"Unknown index: {index_name}")

    required_bands = index_def.bands_required
    physical_bands = {}
    for logical in required_bands:
        physical = band_map.get(logical)
        if physical:
            physical_bands[logical] = physical

    if len(physical_bands) < 2:
        raise ValueError(f"Insufficient band mappings for {index_name}")

    # Collect band hrefs from all scenes
    # For each required band, gather hrefs from each scene
    band_href_lists: dict[str, list[str]] = {logical: [] for logical in required_bands}
    for scene in scenes:
        assets = scene.get("assets", {})
        for logical_name, physical_name in physical_bands.items():
            asset = assets.get(physical_name)
            href = ""
            if asset is None:
                pass
            elif hasattr(asset, 'href'):
                href = getattr(asset, 'href', '') or ''
            elif isinstance(asset, dict):
                href = asset.get('href', '') or ''
            elif isinstance(asset, str):
                href = asset
            if href:
                band_href_lists[logical_name].append(href)

    # Check we have enough data
    for logical in required_bands:
        if len(band_href_lists[logical]) == 0:
            raise ValueError(
                f"No {logical} band hrefs found across {len(scenes)} scenes"
            )

    # Determine resolution
    resolution = 10.0
    if "landsat" in sensor:
        resolution = 30.0

    # Use mosaic module for multi-scene, or single read for one scene
    from app.services.mosaic import mosaic_bands

    # Mosaic the first required band across all scenes
    first_logical = required_bands[0]
    first_physical = physical_bands[first_logical]
    first_hrefs = band_href_lists[first_logical]

    logger.info(
        "[%s] Mosaicking %d scenes for band %s",
        period_label, len(first_hrefs), first_physical,
    )

    # For single scene, use the existing read_raster_window per band
    if len(first_hrefs) == 1:
        return _compute_index_stats(
            index_name, sensor,
            SceneSelection(
                item_id=scenes[0].get("id", "unknown"),
                collection=scenes[0].get("collection", "unknown"),
                datetime=scenes[0].get("properties", {}).get("datetime", ""),
                cloud_cover=scenes[0].get("properties", {}).get("eo:cloud_cover"),
                bbox=scenes[0].get("bbox", []),
                provider=scenes[0].get("provider", "unknown"),
                platform=scenes[0].get("properties", {}).get("platform", "unknown"),
                score=0.0,
                assets=scenes[0].get("assets", {}),
            ),
            bbox,
        )

    # ── Multi-scene: try StackSTAC first, fall back to manual mosaic ──
    stackstac_used = False
    try:
        from app.services.stackstac_adapter import stackstac_compute_index

        # Build band_map for StackSTAC: {logical_name: physical_band_key}
        ss_band_map = {logical: physical_bands[logical] for logical in required_bands if physical_bands.get(logical)}
        if len(ss_band_map) >= 2:
            logger.info(
                "[%s] Trying StackSTAC for %d-band index %s with %d scenes",
                period_label, len(ss_band_map), index_name, len(scenes),
            )
            # Use signed STAC item dicts from the scene list
            stac_dicts = [s if isinstance(s, dict) else (s.__dict__ if hasattr(s, '__dict__') else s) for s in scenes]
            # Compute UTM zone from bbox center longitude
            center_lon = (bbox[0] + bbox[2]) / 2.0
            center_lat = (bbox[1] + bbox[3]) / 2.0
            utm_zone = int((center_lon + 180) / 6) + 1
            epsg_code = (32600 + utm_zone) if center_lat >= 0 else (32700 + utm_zone)
            logger.info("[StackSTAC] Computed EPSG:%d from bbox center (%.2f, %.2f)", epsg_code, center_lon, center_lat)

            ss_result = stackstac_compute_index(
                stac_items=stac_dicts,
                bbox=bbox,
                index_name=index_name,
                band_map=ss_band_map,
                resolution=resolution,
                epsg=epsg_code,
            )
            index_array = ss_result["data"].astype(np.float32)
            stackstac_used = True
            logger.info(
                "[SHAPE-TRACE] [%s] StackSTAC %s: shape=%s, scenes=%d",
                period_label, index_name, index_array.shape, ss_result["scene_count"],
            )
            # Compute stats
            valid = ~np.isnan(index_array) & (index_array != 0)
            valid_pixels = int(np.sum(valid))
            total_pixels = int(index_array.size)
            mean_val = float(np.nanmean(index_array)) if valid_pixels > 0 else 0.0
            stats = {
                "min": float(np.nanmin(index_array)),
                "max": float(np.nanmax(index_array)),
                "mean": mean_val,
                "std": float(np.nanstd(index_array)),
                "median": float(np.nanmedian(index_array)),
            }
            return IndexResult(
                index_name=index_name,
                value=index_array,
                stats=stats,
                scene_id=".".join(s.get("id", "?")[:20] if isinstance(s, dict) else str(s)[:20] for s in scenes[:3]),
                date=scenes[0].get("properties", {}).get("datetime", "") if isinstance(scenes[0], dict) else "",
                resolution_m=resolution,
                shape=index_array.shape,
                valid_pixels=valid_pixels,
                total_pixels=total_pixels,
                method="stackstac",
            )
    except Exception as e:
        logger.warning("[%s] StackSTAC failed (%s), falling back to manual mosaic", period_label, e)

    # Fallback: manual mosaic path (rasterio window reads)
    band_arrays = {}
    nodata_masks = {}

    for logical_name in required_bands:
        physical_name = physical_bands[logical_name]
        scene_hrefs = band_href_lists[logical_name]

        if len(scene_hrefs) == 1:
            # Single scene for this band — just read it
            from app.services.raster_service import read_raster_window
            raster_data = read_raster_window(scene_hrefs[0], bbox)
            data = raster_data["data"]
            if data.ndim == 3:
                band_arrays[physical_name] = data[0].astype(np.float32)
            else:
                band_arrays[physical_name] = data.astype(np.float32)
            nodata_val = raster_data["profile"].get("nodata")
            if nodata_val is not None:
                nodata_masks[physical_name] = (band_arrays[physical_name] == nodata_val)
            else:
                nodata_masks[physical_name] = (band_arrays[physical_name] <= 0)
        else:
            # Multi-scene: mosaic
            mosaic_result = mosaic_bands(scene_hrefs, [physical_name], bbox)
            mosaic_data = mosaic_result["data"]
            if mosaic_data.ndim == 3:
                band_arrays[physical_name] = mosaic_data[0].astype(np.float32)
            else:
                band_arrays[physical_name] = mosaic_data.astype(np.float32)
            nodata_masks[physical_name] = mosaic_result.get("nodata_mask", np.zeros_like(band_arrays[physical_name], dtype=bool))

        logger.info(
            "[SHAPE-TRACE] [%s] Mosaic band %s (%s): shape=%s, valid=%d",
            period_label, logical_name, physical_name,
            band_arrays[physical_name].shape,
            int(np.sum(~nodata_masks[physical_name])),
        )

    # Compute the spectral index
    index_array, index_result = _compute_index_from_bands(
        bands=band_arrays,
        index_name=index_name,
        sensor=sensor,
        nodata_masks=nodata_masks,
        date=scenes[0].get("properties", {}).get("datetime", ""),
        crs="EPSG:4326",
        resolution_meters=resolution,
    )

    logger.info(
        "[SHAPE-TRACE] [%s] Mosaic %s result: shape=%s mean=%.4f, valid=%d/%d, scenes=%d",
        period_label, index_name, index_array.shape,
        index_result.stats["mean"],
        index_result.valid_pixels, index_result.total_pixels,
        len(scenes),
    )

    return IndexResult(
        index_name=index_name,
        value=index_array,
        stats=index_result.stats,
        scene_id=".".join(s.get("id", "?")[:20] for s in scenes[:3]),
        date=scenes[0].get("properties", {}).get("datetime", ""),
        resolution_m=resolution,
        shape=index_result.shape,
        valid_pixels=index_result.valid_pixels,
        total_pixels=index_result.total_pixels,
    )


# ── Index computation (simulated for demo) ────────────────────────

def _compute_index_stats(
    index_name: str,
    sensor: str,
    scene: SceneSelection,
    bbox: list[float],
) -> IndexResult:
    """
    Compute index statistics for a scene using REAL raster reads.

    Opens the actual COG band assets via rasterio, reads the AOI window,
    and computes the spectral index pixel-by-pixel using the spectral_indices engine.

    Falls back to metadata-based estimation only if raster reads fail entirely.
    """
    # Determine resolution
    resolution = 10.0
    if "landsat" in sensor:
        resolution = 30.0
    elif "sentinel-1" in sensor:
        resolution = 10.0

    _lazy_import_heavy()
    # Get band mapping for this sensor + index
    band_key = (sensor, index_name)
    band_map = _INDEX_BAND_MAP.get(band_key)
    if not band_map:
        logger.warning("No band mapping for %s on %s, falling back to estimation", index_name, sensor)
        return _compute_index_stats_fallback(index_name, sensor, scene, bbox)

    # Resolve physical band names from the index's required logical bands
    index_def = _INDEX_DEFINITIONS.get(index_name)
    if not index_def:
        return _compute_index_stats_fallback(index_name, sensor, scene, bbox)

    required_bands = index_def.bands_required  # e.g. ["NIR", "RED"] for NDVI
    physical_bands = {}
    for logical in required_bands:
        physical = band_map.get(logical)
        if physical:
            physical_bands[logical] = physical

    if len(physical_bands) < 2:
        logger.warning("Insufficient band mappings for %s: %s", index_name, physical_bands)
        return _compute_index_stats_fallback(index_name, sensor, scene, bbox)

    # Extract signed asset hrefs from scene
    # Assets can be: pystac.Asset objects, dicts with 'href' key, or plain strings
    band_hrefs = {}
    for logical_name, physical_name in physical_bands.items():
        asset = scene.assets.get(physical_name)
        href = ""
        if asset is None:
            href = ""
        elif hasattr(asset, 'href'):
            # pystac.Asset object — use .href attribute
            href = getattr(asset, 'href', '') or ''
        elif isinstance(asset, dict):
            href = asset.get('href', '') or ''
        elif isinstance(asset, str):
            href = asset
        if href:
            band_hrefs[logical_name] = href

    if len(band_hrefs) < 2:
        logger.warning(
            "Missing band assets for %s. Have: %s, Need: %s",
            index_name, list(band_hrefs.keys()), list(physical_bands.keys()),
        )
        return _compute_index_stats_fallback(index_name, sensor, scene, bbox)

    # Attempt real raster read and index computation
    try:
        from app.services.raster_service import read_raster_window

        # Read each band over the AOI bbox
        # band_arrays must be keyed by PHYSICAL band name (B08, B04, B11)
        # because compute_index_from_bands looks them up by physical name.
        band_arrays = {}
        nodata_masks = {}
        logical_to_physical = {}  # logical_name → physical_name
        crs = "EPSG:4326"
        transform = None
        shape = None

        for logical_name, href in band_hrefs.items():
            # Find the physical name for this logical name
            physical_name = physical_bands.get(logical_name, logical_name)
            logical_to_physical[logical_name] = physical_name

            logger.info("[SHAPE-TRACE] Reading band %s (%s) from %s", logical_name, physical_name, href[:120])
            raster_data = read_raster_window(href, bbox)
            data = raster_data["data"]
            logger.info("[SHAPE-TRACE] Raw raster read: band=%s data.shape=%s ndim=%d crs=%s",
                        physical_name, data.shape, data.ndim, raster_data.get('crs'))
            logger.info("[SHAPE-TRACE] Raster transform: %s", raster_data.get('transform'))

            # Take first band if multi-band (some assets are multi-band)
            if data.ndim == 3 and data.shape[0] > 1:
                band_arrays[physical_name] = data[0].astype(np.float32)
            elif data.ndim == 3:
                band_arrays[physical_name] = data[0].astype(np.float32)
            else:
                band_arrays[physical_name] = data.astype(np.float32)
            logger.info("[SHAPE-TRACE] Band array %s: shape=%s", physical_name, band_arrays[physical_name].shape)

            # Build nodata mask
            nodata_val = raster_data["profile"].get("nodata")
            if nodata_val is not None:
                nodata_masks[physical_name] = (band_arrays[physical_name] == nodata_val)
            else:
                nodata_masks[physical_name] = (band_arrays[physical_name] <= 0)

            crs = raster_data.get("crs", "EPSG:4326")
            if raster_data.get("transform") is not None:
                transform = raster_data["transform"]
            shape = list(band_arrays[physical_name].shape)

            logger.info(
                "Band %s (%s): shape=%s, min=%.4f, max=%.4f, nodata_count=%d",
                logical_name, physical_name, band_arrays[physical_name].shape,
                float(np.nanmin(band_arrays[physical_name])),
                float(np.nanmax(band_arrays[physical_name])),
                int(np.sum(nodata_masks[physical_name])),
            )

        # Compute the spectral index
        index_array, index_result = _compute_index_from_bands(
            bands=band_arrays,
            index_name=index_name,
            sensor=sensor,
            nodata_masks=nodata_masks,
            date=scene.datetime,
            crs=crs,
            resolution_meters=resolution,
        )

        logger.info(
            "[%s] Computed %s for %s: mean=%.4f, std=%.4f, valid=%d/%d",
            index_name, index_name, scene.item_id[:30],
            index_result.stats["mean"], index_result.stats["std"],
            index_result.valid_pixels, index_result.total_pixels,
        )

        # Convert IndexResult (spectral_indices) to our IndexResult (temporal_compare)
        return IndexResult(
            index_name=index_name,
            value=index_array,
            stats=index_result.stats,
            scene_id=scene.item_id,
            date=scene.datetime,
            resolution_m=resolution,
            shape=index_result.shape,
            valid_pixels=index_result.valid_pixels,
            total_pixels=index_result.total_pixels,
        )

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)[:200]}"
        logger.error(
            "Raster-based %s computation failed for %s: %s. Falling back to estimation.",
            index_name, scene.item_id, error_msg,
            exc_info=True,
        )
        result = _compute_index_stats_fallback(index_name, sensor, scene, bbox)
        # Attach error info so API response shows why raster failed
        result.stats["_raster_error"] = error_msg
        return result


def _compute_index_stats_fallback(
    index_name: str,
    sensor: str,
    scene: SceneSelection,
    bbox: list[float],
) -> IndexResult:
    """
    Fallback when raster reads fail.

    Returns value=None with zeroed stats and a clear 'unavailable' flag.
    NEVER fabricates realistic-looking values.
    """
    resolution = 10.0
    if "landsat" in sensor:
        resolution = 30.0

    if bbox and len(bbox) == 4:
        width_deg = bbox[2] - bbox[0]
        height_deg = bbox[3] - bbox[1]
        mid_lat = (bbox[1] + bbox[3]) / 2
        km_per_deg_lon = 111.0 * math.cos(math.radians(mid_lat))
        width_km = width_deg * km_per_deg_lon
        height_km = height_deg * 111.0
        area_km2 = width_km * height_km
    else:
        area_km2 = 100.0

    pixel_size_km = resolution / 1000.0
    total_pixels = int(area_km2 / (pixel_size_km ** 2))

    # DO NOT fabricate stats. Return explicit zero/null values.
    stats = {
        "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
        "median": 0.0, "p5": 0.0, "p95": 0.0,
        "_status": "unavailable",
        "_reason": f"raster_read_failed for {scene.item_id}",
    }

    logger.warning(
        "[%s] Raster read FAILED for %s — returning UNAVAILABLE (no fabricated values)",
        index_name, scene.item_id,
    )

    return IndexResult(
        index_name=index_name,
        value=None,  # CRITICAL: None means no raster data available
        stats=stats,
        scene_id=scene.item_id,
        date=scene.datetime,
        resolution_m=resolution,
        shape=[total_pixels // 100, 100],
        valid_pixels=0,  # No valid raster pixels
        total_pixels=total_pixels,
    )


# ── Metrics computation ──────────────────────────────────────────

def _compute_comparison_metrics(
    phenomenon: str,
    index_name: str,
    index_t1: IndexResult,
    index_t2: IndexResult,
    change_result: Optional[dict[str, Any]],
    bbox: list[float],
) -> dict[str, Any]:
    """Compute the key metrics for the UI based on phenomenon type."""
    t1_mean = index_t1.stats.get("mean", 0)
    t2_mean = index_t2.stats.get("mean", 0)
    delta = t2_mean - t1_mean

    # Compute area in km²
    if bbox and len(bbox) == 4:
        mid_lat = (bbox[1] + bbox[3]) / 2
        km_per_deg_lon = 111.0 * math.cos(math.radians(mid_lat))
        area_km2 = (bbox[2] - bbox[0]) * km_per_deg_lon * (bbox[3] - bbox[1]) * 111.0
    else:
        area_km2 = 100.0

    # Changed area from change detection — prefer pixel-derived values
    # Pixel-derived values come from run_change_detection() and are authoritative.
    # bbox-based area is only a fallback when change_result is unavailable.
    if change_result and change_result.get("changed_area_sq_meters") is not None:
        # PIXEL-DERIVED: changed_pixels × pixel_area (authoritative)
        changed_km2 = change_result["changed_area_sq_meters"] / 1e6
        changed_pct = change_result.get("changed_pct", 0)
        total_area_from_pixels = change_result.get("total_area_sq_meters") or (
            change_result.get("total_pixels", 0) * (index_t1.resolution_m ** 2)
        )
        if total_area_from_pixels > 0:
            area_km2 = total_area_from_pixels / 1e6
    elif change_result:
        # change_result exists but no pixel area — use changed_pct with bbox area
        changed_pct = change_result.get("changed_pct", abs(delta) * 100)
        changed_km2 = area_km2 * (changed_pct / 100.0)
    else:
        # No change detection at all — bbox estimate only
        changed_pct = abs(delta) * 100
        changed_km2 = area_km2 * (changed_pct / 100.0)

    # Base metrics — all values from real computation
    metrics = {
        "total_area_km2": round(area_km2, 2),
        "changed_area_km2": round(changed_km2, 2),
        "changed_pct": round(changed_pct, 2),
        "delta_index": round(delta, 4),
        "baseline_index_mean": round(t1_mean, 4),
        "comparison_index_mean": round(t2_mean, 4),
        "index_name": index_name,
        "resolution_m": index_t1.resolution_m,
        "area_source": "pixel_derived" if (change_result and change_result.get("changed_area_sq_meters") is not None) else "bbox_estimated",
    }

    # Direction indicator
    if delta > 0.05:
        direction = "increase"
    elif delta < -0.05:
        direction = "decrease"
    else:
        direction = "stable"
    metrics["direction"] = direction

    # Determine if values are raster-derived or estimated
    raster_derived = (index_t1.value is not None and index_t2.value is not None)
    metrics["raster_derived"] = raster_derived
    if raster_derived:
        metrics["data_quality"] = "raster_computed"
        metrics["estimation_method"] = "pixel_level_raster_analysis"
    else:
        metrics["data_quality"] = "estimated_from_metadata"
        metrics["estimation_method"] = "scene_metadata_fallback"
    metrics["estimated"] = not raster_derived

    return metrics


def _generate_explanation(
    phenomenon: str,
    aoi_name: str,
    metrics: dict[str, Any],
    period1: str,
    period2: str,
    index_name: str,
) -> dict[str, Any]:
    """Generate a structured explanation of the temporal comparison results."""

    direction = metrics.get("direction", "change")
    area = metrics.get("total_area_km2", 0)
    changed_area = metrics.get("changed_area_km2", 0)
    changed_pct = metrics.get("changed_pct", 0)

    phenomenon_descriptions = {
        "urban_expansion": {
            "title": "Urban Expansion Analysis",
            "summary": f"Temporal analysis of {aoi_name} from {period1} to {period2} reveals urban expansion patterns. Using {index_name} change detection on Sentinel-2 multispectral imagery, we tracked built-up area growth.",
            "methodology": "NDBI (Normalized Difference Built-up Index) was computed for both time periods. The change map shows areas where impervious surface cover has increased, indicating new construction, infrastructure development, or urban sprawl.",
            "key_indices": [index_name],
        },
        "vegetation_change": {
            "title": "Vegetation Health Analysis",
            "summary": f"Monitoring vegetation changes in {aoi_name} from {period1} to {period2} using NDVI time series analysis from Sentinel-2 imagery.",
            "methodology": "NDVI (Normalized Difference Vegetation Index) quantifies vegetation greenness and photosynthetic activity. Changes indicate deforestation, degradation, regrowth, or seasonal variation.",
            "key_indices": ["NDVI"],
        },
        "deforestation": {
            "title": "Deforestation Detection",
            "summary": f"Forest cover change analysis for {aoi_name} from {period1} to {period2}. NDVI-based detection identifies areas of forest loss and gain.",
            "methodology": "Multi-temporal NDVI analysis detects significant vegetation loss that indicates deforestation. Threshold-based classification separates natural variation from anthropogenic clearing.",
            "key_indices": ["NDVI"],
        },
        "flood_impact": {
            "title": "Flood Impact Assessment",
            "summary": f"Flood extent mapping for {aoi_name} from {period1} to {period2} using Sentinel-1 SAR and Sentinel-2 optical imagery.",
            "methodology": "Cross-sensor analysis combines Sentinel-1 SAR (cloud-penetrating) with Sentinel-2 NDWI for robust flood detection. SAR backscatter change identifies waterlogged areas, while NDWI confirms open water extent.",
            "key_indices": ["NDWI"],
            "sensors_used": ["Sentinel-1 (SAR)", "Sentinel-2 (Optical)"],
        },
        "water_change": {
            "title": "Water Body Change Analysis",
            "summary": f"Water body monitoring for {aoi_name} from {period1} to {period2}. NDWI tracks changes in surface water extent.",
            "methodology": "NDWI (Normalized Difference Water Index) delineates water bodies. Temporal comparison reveals expansion, shrinkage, or seasonal fluctuation of lakes, reservoirs, and rivers.",
            "key_indices": ["NDWI"],
        },
        "burn_severity": {
            "title": "Wildfire Burn Severity Assessment",
            "summary": f"Burn severity mapping for {aoi_name} from {period1} to {period2} using dNBR (differenced Normalized Burn Ratio).",
            "methodology": "dNBR is the standard metric for burn severity. Pre-fire NBR is subtracted from post-fire NBR. Low dNBR indicates unburned/low severity, high negative dNBR indicates high-severity burns.",
            "key_indices": ["NBR", "dNBR"],
        },
        "snow_cover": {
            "title": "Snow and Ice Cover Analysis",
            "summary": f"Snow/ice extent change for {aoi_name} from {period1} to {period2} using NDSI from Sentinel-2 imagery.",
            "methodology": "NDSI (Normalized Difference Snow Index) discriminates snow/ice from other surfaces. Temporal comparison reveals snow line retreat, seasonal snow cover variation, and cryosphere changes.",
            "key_indices": ["NDSI"],
        },
        "glacier_retreat": {
            "title": "Glacier Retreat Monitoring",
            "summary": f"Glacier extent change for {aoi_name} from {period1} to {period2}. Multi-temporal analysis tracks ice mass loss and terminus retreat.",
            "methodology": "NDSI-based glacier delineation combined with multi-temporal comparison. Snow/ice areas are mapped for both periods and differenced to quantify retreat. Higher-altitude analysis captures equilibrium line changes.",
            "key_indices": ["NDSI"],
        },
        "coastal_erosion": {
            "title": "Coastal Erosion Analysis",
            "summary": f"Coastline change detection for {aoi_name} from {period1} to {period2} using water-land boundary analysis.",
            "methodology": "NDWI-based water classification identifies the land-water boundary for both periods. shoreline position change indicates erosion (land loss) or accretion (land gain).",
            "key_indices": ["NDWI"],
        },
        "soil_moisture": {
            "title": "Soil Moisture Analysis",
            "summary": f"Soil moisture and dryness assessment for {aoi_name} from {period1} to {period2} using spectral moisture proxies.",
            "methodology": "Combined NDVI-NDMI analysis estimates relative soil moisture conditions. SWIR-band absorption characteristics reveal moisture content variations over time.",
            "key_indices": ["NDVI"],
        },
        "land_cover_change": {
            "title": "Land Cover Change Analysis",
            "summary": f"General land cover change detection for {aoi_name} from {period1} to {period2} using multi-index analysis.",
            "methodology": "Multi-temporal analysis combining NDVI, NDBI, and NDWI indices classifies land cover transitions between vegetation, built-up, water, and bare soil categories.",
            "key_indices": ["NDVI", "NDBI", "NDWI"],
        },
    }

    info = phenomenon_descriptions.get(phenomenon, phenomenon_descriptions["land_cover_change"])

    raster_derived = metrics.get("raster_derived", False)

    if raster_derived:
        confidence = "Computed from pixel-level raster analysis of Sentinel-2 multispectral imagery."
        limitations = [
            "Single pair comparison (not time series) — seasonal effects possible",
            "Cloud cover may affect optical imagery quality",
            "Resolution limits detection of small-scale changes",
        ]
    else:
        confidence = "Estimated from scene metadata. Quantitative metrics are not derived from pixel-level raster analysis."
        limitations = [
            "Index statistics are estimated from scene metadata, not computed from actual raster pixel analysis",
            "Cloud cover may affect optical imagery quality",
            "Single pair comparison (not time series) — seasonal effects possible",
            "Resolution limits detection of small-scale changes",
        ]

    return {
        "title": info["title"],
        "summary": info["summary"],
        "methodology": info["methodology"],
        "key_findings": _generate_findings(phenomenon, metrics),
        "key_indices": info.get("key_indices", [index_name]),
        "sensors_used": info.get("sensors_used", ["Sentinel-2"]),
        "confidence": confidence,
        "raster_derived": raster_derived,
        "limitations": limitations,
    }


def _generate_findings(phenomenon: str, metrics: dict[str, Any]) -> list[str]:
    """Generate key findings based on phenomenon and metrics."""
    findings = []
    direction = metrics.get("direction", "")
    changed_pct = metrics.get("changed_pct", 0)
    changed_km2 = metrics.get("changed_area_km2", 0)

    # direction values from the detector are: "increase", "decrease", "stable"
    if direction in ("increase", "decrease"):
        intensity = "significant" if changed_pct > 10 else "moderate" if changed_pct > 3 else "minor"
        findings.append(f"{intensity.title()} candidate change detected — {changed_km2} km² affected ({changed_pct}% of study area)")

        if phenomenon == "urban_expansion" and direction == "increase":
            findings.append("NDBI shows increase in candidate built-up change regions")
        elif phenomenon == "urban_expansion" and direction == "decrease":
            findings.append("NDBI shows decrease in candidate built-up change regions")
        elif phenomenon == "flood_impact":
            findings.append("Cross-sensor (SAR + optical) analysis for candidate water extent mapping")
        elif phenomenon == "burn_severity" and direction == "decrease":
            findings.append("dNBR indicates candidate burn damage regions")
        elif phenomenon == "glacier_retreat" and direction == "decrease":
            findings.append("NDSI indicates candidate glacier retreat regions")
        elif phenomenon == "vegetation_change" and direction == "decrease":
            findings.append("NDVI decrease indicates candidate vegetation loss regions")
        elif phenomenon == "vegetation_change" and direction == "increase":
            findings.append("NDVI increase indicates candidate vegetation gain regions")
    elif direction in ("stable",):
        findings.append("Minimal candidate change detected — area appears relatively stable over the analysis period")

    if not findings:
        findings.append(f"Change magnitude: {metrics.get('delta_index', 0):.4f} index units")

    return findings


# ── Multi-signal change analysis ───────────────────────────────

def _compute_multi_signal_change(
    primary_index_name: str,
    additional_indices: dict[str, dict[str, Any]],
    signal_rules: list[dict[str, Any]],
    min_agreeing_signals: int,
    scene_t1: Optional[SceneSelection],
    scene_t2: Optional[SceneSelection],
    bbox: list[float],
    resolution_m: float,
    primary_change_result: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """
    Multi-signal change analysis.

    Combines primary indicator with supporting indicators to produce
    more defensible candidate change regions.

    For URBAN_EXPANSION:
    - Primary: NDBI increase (built-up signal)
    - Supporting: NDVI decrease (vegetation context)
    - A pixel is a candidate change if enough signals agree.

    This implements the research principle:
    Don't rely on one spectral signal — combine multiple indicators
    for more robust change detection.
    """
    signal_details = {}
    agreeing_pixels = 0
    candidate_changed_pct = 0.0
    candidate_changed_area = 0.0

    # ── Evaluate each signal rule ────────────────────────────────
    for rule in signal_rules:
        idx_name = rule["index_name"]
        direction = rule["direction"]
        threshold = rule.get("threshold", 0.10)
        is_primary = rule.get("is_primary", False)

        # Get the index results for this signal
        if idx_name == primary_index_name:
            # Use the primary change detection results
            sig_detail = {
                "index_name": idx_name,
                "direction": direction,
                "threshold": threshold,
                "is_primary": is_primary,
                "status": "primary",
            }
            if primary_change_result and primary_change_result.get("raster_derived"):
                # We already have raster-level analysis for this signal
                sig_detail["changed_pct"] = primary_change_result.get("changed_pct", 0)
                sig_detail["algorithm"] = primary_change_result.get("algorithm", "unknown")
                # For the primary signal, "agreeing" pixels = all changed pixels
                # (they already passed the threshold in the main change detection)
                sig_detail["agreeing_pixels"] = primary_change_result.get("changed_pixels", 0)
            else:
                sig_detail["changed_pct"] = 0
                sig_detail["algorithm"] = "not_raster_derived"
                sig_detail["agreeing_pixels"] = 0
        elif idx_name in additional_indices:
            idx_data = additional_indices[idx_name]
            idx_t1 = idx_data.get("t1")
            idx_t2 = idx_data.get("t2")

            if idx_t1 and idx_t2 and idx_t1.value is not None and idx_t2.value is not None:
                # Compute the delta for this supporting indicator
                t1_arr = idx_t1.value.astype(np.float32)
                t2_arr = idx_t2.value.astype(np.float32)
                min_h = min(t1_arr.shape[0], t2_arr.shape[0])
                min_w = min(t1_arr.shape[1], t2_arr.shape[1])
                t1_c = t1_arr[:min_h, :min_w]
                t2_c = t2_arr[:min_h, :min_w]

                valid = ~np.isnan(t1_c) & ~np.isnan(t2_c) & ~np.isinf(t1_c) & ~np.isinf(t2_c)
                delta = np.where(valid, t2_c - t1_c, np.nan)
                total_valid = int(np.sum(valid))

                # Apply direction-specific threshold
                if direction == "increase":
                    signal_mask = valid & (delta > threshold)
                elif direction == "decrease":
                    signal_mask = valid & (delta < -threshold)
                else:  # absolute_change
                    signal_mask = valid & (np.abs(delta) > threshold)

                signal_pixels = int(np.sum(signal_mask))
                signal_pct = (signal_pixels / total_valid * 100) if total_valid > 0 else 0.0

                sig_detail = {
                    "index_name": idx_name,
                    "direction": direction,
                    "threshold": threshold,
                    "is_primary": is_primary,
                    "status": "raster_computed",
                    "changed_pct": round(signal_pct, 4),
                    "changed_pixels": signal_pixels,
                    "total_valid_pixels": total_valid,
                    "agreeing_pixels": signal_pixels,
                    "delta_mean": round(float(np.nanmean(delta[valid])), 4) if total_valid > 0 else 0.0,
                }
            else:
                # Fallback: estimate from stats
                if idx_t1 and idx_t2:
                    delta_mean = idx_t2.stats.get("mean", 0) - idx_t1.stats.get("mean", 0)
                    if direction == "increase":
                        signal_satisfied = delta_mean > threshold
                    elif direction == "decrease":
                        signal_satisfied = delta_mean < -threshold
                    else:
                        signal_satisfied = abs(delta_mean) > threshold

                    sig_detail = {
                        "index_name": idx_name,
                        "direction": direction,
                        "threshold": threshold,
                        "is_primary": is_primary,
                        "status": "estimated",
                        "delta_mean": round(delta_mean, 4),
                        "signal_satisfied": signal_satisfied,
                        "agreeing_pixels": idx_t1.total_pixels if signal_satisfied else 0,
                    }
                else:
                    sig_detail = {
                        "index_name": idx_name,
                        "direction": direction,
                        "threshold": threshold,
                        "is_primary": is_primary,
                        "status": "no_data",
                        "agreeing_pixels": 0,
                    }
        else:
            sig_detail = {
                "index_name": idx_name,
                "direction": direction,
                "threshold": threshold,
                "is_primary": is_primary,
                "status": "not_computed",
                "agreeing_pixels": 0,
            }

        signal_details[idx_name] = sig_detail

    # ── Compute agreement ────────────────────────────────────────
    # Count how many signals agree (are satisfied)
    satisfied_count = 0
    total_signals = len(signal_rules)
    for rule in signal_rules:
        idx_name = rule["index_name"]
        detail = signal_details.get(idx_name, {})
        status = detail.get("status", "not_computed")
        if status in ("primary", "raster_computed"):
            # For primary: use the main change detection's changed_pixels
            # For supporting: use the signal's changed_pixels
            if detail.get("agreeing_pixels", 0) > 0:
                satisfied_count += 1
        elif status == "estimated":
            if detail.get("signal_satisfied", False):
                satisfied_count += 1

    meets_threshold = satisfied_count >= min_agreeing_signals

    # Compute candidate changed area from the primary signal
    # but only if enough signals agree
    if meets_threshold and primary_change_result:
        candidate_changed_pct = primary_change_result.get("changed_pct", 0)
        candidate_changed_area = primary_change_result.get("changed_area_sq_meters", 0)
        agreeing_pixels = primary_change_result.get("changed_pixels", 0)
    else:
        candidate_changed_pct = 0.0
        candidate_changed_area = 0.0
        agreeing_pixels = 0

    # Confidence note
    if meets_threshold:
        if satisfied_count == total_signals:
            confidence_note = (
                f"All {total_signals} indicators agree on candidate change regions. "
                f"Primary signal: {candidate_changed_pct:.2f}% of area."
            )
        else:
            confidence_note = (
                f"{satisfied_count}/{total_signals} indicators agree on candidate change. "
                f"Primary signal: {candidate_changed_pct:.2f}% of area. "
                f"Supporting signals provide additional context."
            )
    else:
        confidence_note = (
            f"Only {satisfied_count}/{total_signals} indicators agree — "
            f"below minimum threshold of {min_agreeing_signals}. "
            f"Detected changes may not represent real-world {primary_index_name.lower()} change."
        )

    return {
        "status": "ok",
        "satisfied_signals": satisfied_count,
        "total_signals": total_signals,
        "meets_threshold": meets_threshold,
        "agreeing_pixels": agreeing_pixels,
        "candidate_changed_pct": round(candidate_changed_pct, 4),
        "candidate_changed_area_sq_meters": round(candidate_changed_area, 2),
        "signal_details": signal_details,
        "confidence_note": confidence_note,
    }


# ── Pure-Python PNG encoder (no Pillow required) ───────────────
import struct
import zlib


def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    """Create a PNG chunk: length + type + data + CRC."""
    chunk = chunk_type + data
    crc = zlib.crc32(chunk) & 0xFFFFFFFF
    return struct.pack('>I', len(data)) + chunk + struct.pack('>I', crc)


def _encode_rgba_png(rgba_array: np.ndarray) -> str:
    """Encode a (H, W, 4) uint8 RGBA array as a PNG, returning hex string."""
    h, w = rgba_array.shape[:2]
    # PNG signature
    sig = b'\x89PNG\r\n\x1a\n'
    # IHDR: width, height, bit_depth=8, color_type=6 (RGBA)
    ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)
    ihdr = _make_png_chunk(b'IHDR', ihdr_data)
    # IDAT: filter byte (0) + raw pixel data per row, zlib-compressed
    raw_rows = []
    for row in rgba_array:
        raw_rows.append(b'\x00' + row.tobytes())
    raw = b''.join(raw_rows)
    compressed = zlib.compress(raw, 6)
    idat = _make_png_chunk(b'IDAT', compressed)
    # IEND
    iend = _make_png_chunk(b'IEND', b'')
    return (sig + ihdr + idat + iend).hex()


def _encode_rgb_png(rgb_array: np.ndarray) -> str:
    """Encode a (H, W, 3) uint8 RGB array as a PNG, returning hex string."""
    h, w = rgb_array.shape[:2]
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)
    ihdr = _make_png_chunk(b'IHDR', ihdr_data)
    raw_rows = []
    for row in rgb_array:
        raw_rows.append(b'\x00' + row.tobytes())
    raw = b''.join(raw_rows)
    compressed = zlib.compress(raw, 6)
    idat = _make_png_chunk(b'IDAT', compressed)
    iend = _make_png_chunk(b'IEND', b'')
    return (sig + ihdr + idat + iend).hex()


# ── E2E Diagnostic Report ────────────────────────────────────────

def _print_e2e_diagnostic(
    query: str,
    phenomenon: str,
    index_name: str,
    period1: dict[str, Any],
    period2: dict[str, Any],
    scene_sel_t1,
    scene_sel_t2,
    scene_sel_obj_t1,
    scene_sel_obj_t2,
    index_t1,
    index_t2,
    analysis_grid_info: Optional[dict[str, Any]],
    scl_available: bool,
    cloud_mask_t1,
    cloud_mask_t2,
    t1_aligned,
    t2_aligned,
    change_result_obj,
    change_result: Optional[dict[str, Any]],
    bbox: list[float],
    multi_signal_enabled: bool,
    signal_rules: list[dict[str, Any]],
    min_agreeing: int,
    collection: str,
    sensor: str,
) -> None:
    """Print a structured E2E diagnostic report to server logs.

    Every value comes from the ACTUAL pipeline execution.
    Prefix: [E2E] for easy log grep.
    """
    NL = "\n"
    SEP = "=" * 60
    lines: list[str] = []

    def L(msg: str = "") -> None:
        lines.append(msg)

    L(SEP)
    L("LIVE E2E CHANGE DETECTION DIAGNOSTIC")
    L(SEP)
    L()

    # ── QUERY ───────────────────────────────────────────────────
    L("QUERY")
    L(f"- Query: {query}")
    L(f"- Phenomenon: {phenomenon}")
    L(f"- Index: {index_name}")
    L(f"- Collection: {collection}")
    L(f"- Sensor: {sensor}")
    L()

    # ── PERIOD 1 ────────────────────────────────────────────────
    L("PERIOD 1")
    if scene_sel_obj_t1:
        s1 = scene_sel_obj_t1
        L(f"- Provider: {s1.provider}")
        L(f"- Collection: {s1.collection}")
        L(f"- Scene ID: {s1.item_id}")
        L(f"- Acquisition date: {s1.datetime}")
        L(f"- Cloud cover: {s1.cloud_cover}")
        L(f"- Platform: {s1.platform}")
        # Asset keys
        asset_keys = list(s1.assets.keys()) if s1.assets else []
        L(f"- Assets: {asset_keys}")
        # AOI coverage from selection result
        if scene_sel_t1 and hasattr(scene_sel_t1, 'coverage_ratio'):
            L(f"- AOI coverage: {scene_sel_t1.coverage_ratio:.1%} ({scene_sel_t1.total_scenes} scenes, mosaic={scene_sel_t1.is_mosaic})")
        elif scene_sel_t1:
            L(f"- AOI coverage: {scene_sel_t1.coverage_ratio:.1%}")
        else:
            L("- AOI coverage: NOT AVAILABLE — selection result not present")
        L(f"- Scene bbox: {s1.bbox}")
    else:
        L("- Provider: NOT AVAILABLE — no scene selected for period 1")
    L()

    # ── PERIOD 2 ────────────────────────────────────────────────
    L("PERIOD 2")
    if scene_sel_obj_t2:
        s2 = scene_sel_obj_t2
        L(f"- Provider: {s2.provider}")
        L(f"- Collection: {s2.collection}")
        L(f"- Scene ID: {s2.item_id}")
        L(f"- Acquisition date: {s2.datetime}")
        L(f"- Cloud cover: {s2.cloud_cover}")
        L(f"- Platform: {s2.platform}")
        asset_keys = list(s2.assets.keys()) if s2.assets else []
        L(f"- Assets: {asset_keys}")
        if scene_sel_t2 and hasattr(scene_sel_t2, 'coverage_ratio'):
            L(f"- AOI coverage: {scene_sel_t2.coverage_ratio:.1%} ({scene_sel_t2.total_scenes} scenes, mosaic={scene_sel_t2.is_mosaic})")
        elif scene_sel_t2:
            L(f"- AOI coverage: {scene_sel_t2.coverage_ratio:.1%}")
        else:
            L("- AOI coverage: NOT AVAILABLE — selection result not present")
        L(f"- Scene bbox: {s2.bbox}")
    else:
        L("- Provider: NOT AVAILABLE — no scene selected for period 2")
    L()

    # ── ANALYSIS GRID ────────────────────────────────────────────
    L("ANALYSIS GRID")
    if analysis_grid_info:
        gi = analysis_grid_info
        crs_val = gi.get('crs', 'NOT AVAILABLE')
        res_val = gi.get('resolution_meters', 'NOT AVAILABLE')
        w_val = gi.get('width', 'NOT AVAILABLE')
        h_val = gi.get('height', 'NOT AVAILABLE')
        bounds_val = gi.get('bounds', 'NOT AVAILABLE')
        transform_val = gi.get('transform', 'NOT AVAILABLE')
        L(f"- CRS: {crs_val}")
        L(f"- Resolution: {res_val}m")
        L(f"- Width: {w_val}")
        L(f"- Height: {h_val}")
        L(f"- Bounds: {bounds_val}")
        L(f"- Transform: {transform_val}")
        L(f"- Source CRS 1: {gi.get('source_crs1', 'NOT AVAILABLE')}")
        L(f"- Source CRS 2: {gi.get('source_crs2', 'NOT AVAILABLE')}")
    else:
        L("- CRS: NOT AVAILABLE — common grid not computed")
        L("- Resolution: NOT AVAILABLE")
        L("- Width: NOT AVAILABLE")
        L("- Height: NOT AVAILABLE")
        L("- Bounds: NOT AVAILABLE")
        L("- Transform: NOT AVAILABLE")
    L()

    # ── INDEX RESULTS ────────────────────────────────────────────
    L("INDEX RESULTS")
    if index_t1:
        L(f"- T1 Index: {index_t1.index_name}")
        L(f"- T1 Mean: {index_t1.stats.get('mean', 'N/A')}")
        L(f"- T1 Valid pixels: {index_t1.valid_pixels}/{index_t1.total_pixels}")
        L(f"- T1 Shape: {index_t1.shape}")
    else:
        L("- T1 Index: NOT AVAILABLE — computation failed")
    if index_t2:
        L(f"- T2 Index: {index_t2.index_name}")
        L(f"- T2 Mean: {index_t2.stats.get('mean', 'N/A')}")
        L(f"- T2 Valid pixels: {index_t2.valid_pixels}/{index_t2.total_pixels}")
        L(f"- T2 Shape: {index_t2.shape}")
    else:
        L("- T2 Index: NOT AVAILABLE — computation failed")
    L()

    # ── QUALITY ──────────────────────────────────────────────────
    L("QUALITY")
    L(f"- SCL available: {scl_available}")
    if scl_available:
        L("- SCL masking applied: YES")
        L("- Resampling method: nearest-neighbor")
        # cloud_mask_t1/t2 are True=VALID (from create_cloud_mask)
        if cloud_mask_t1 is not None and t1_aligned is not None:
            scl_valid_t1 = int(np.sum(cloud_mask_t1)) if cloud_mask_t1 is not None else 0
            total_t1 = int(t1_aligned.size) if t1_aligned is not None else 0
            scl_cloud_t1 = total_t1 - scl_valid_t1
            L(f"- SCL valid pixels period 1: {scl_valid_t1}/{total_t1} (cloud/shadow={scl_cloud_t1})")
        if cloud_mask_t2 is not None and t2_aligned is not None:
            scl_valid_t2 = int(np.sum(cloud_mask_t2)) if cloud_mask_t2 is not None else 0
            total_t2 = int(t2_aligned.size) if t2_aligned is not None else 0
            scl_cloud_t2 = total_t2 - scl_valid_t2
            L(f"- SCL valid pixels period 2: {scl_valid_t2}/{total_t2} (cloud/shadow={scl_cloud_t2})")
    else:
        L("- SCL masking applied: NO — nodata-only masking used")
        L("- Resampling method: N/A")
    if change_result:
        vpr = change_result.get('valid_pixel_ratio', 'NOT AVAILABLE')
        nodata = change_result.get('nodata_pixels', 'NOT AVAILABLE')
        cloud_masked = change_result.get('cloud_masked_pixels', 'NOT AVAILABLE')
        L(f"- Valid pixel ratio: {vpr}")
        L(f"- NoData pixels: {nodata}")
        L(f"- Cloud masked pixels: {cloud_masked}")
    else:
        L("- Valid pixel ratio: NOT AVAILABLE — no change detection result")
    L()

    # ── DETECTION ────────────────────────────────────────────────
    L("DETECTION")
    if change_result_obj:
        cdo = change_result_obj
        L(f"- Index: {cdo.index_name}")
        params = cdo.parameters
        L(f"- NDBI threshold: {params.get('threshold', 'N/A')}")
        # NDVI threshold: from change_result_obj or signal rules
        ndvi_thresh = params.get('ndvi_decrease_threshold')
        if not ndvi_thresh:
            # Try to find it from the plan's signal rules if passed through
            ndvi_thresh = 'NOT AVAILABLE — use NDBI AND NDVI threshold from semantic config'
        L(f"- NDVI threshold: {ndvi_thresh}")
        L(f"- Direction: {params.get('direction', 'N/A')}")
        L(f"- Multi-signal enabled: {params.get('multi_signal', False)}")
        # Dynamic min region area calculation
        min_px = params.get('min_region_size', 0)
        px_area_m2 = cdo.resolution_meters ** 2
        min_region_m2 = min_px * px_area_m2
        min_region_ha = min_region_m2 / 10000.0
        min_region_km2 = min_region_m2 / 1e6
        L(f"- Min region size: {min_px} pixels = {min_region_m2:,.0f} m² = {min_region_ha:.2f} ha = {min_region_km2:.4f} km²")
        L(f"- Resolution used for area: {cdo.resolution_meters:.2f}m (pixel area = {px_area_m2:,.0f} m²)")
        L(f"- Algorithm: {cdo.algorithm}")
        L(f"- Total valid pixels: {cdo.total_pixels}")
        L(f"- Changed pixels (final): {cdo.changed_pixels}")
        L(f"- Changed %: {cdo.changed_pct}%")
        L(f"- Changed area: {cdo.changed_area_sq_meters:.0f} m²")
        L(f"- Total area: {cdo.total_area_sq_meters:.0f} m²")
        L(f"- Final regions: {cdo.num_regions}")
        L(f"- Largest region: {cdo.largest_region}")
        # Raw candidate pixels (before morphology)
        if cdo.processing_steps:
            for ps in cdo.processing_steps:
                step_name = ps.get('step', '')
                step_detail = ps.get('detail', '')
                if 'apply_threshold' in step_name:
                    L(f"- Pre-morphology threshold: {step_detail}")
                if 'morphological_cleanup' in step_name:
                    L(f"- Morphological cleanup: {step_detail}")
                if 'multi_signal' in step_name:
                    L(f"- Multi-signal: {step_detail}")
        # Also report from change_result dict if available
        if change_result and change_result.get('multi_signal'):
            L(f"- Multi-signal config: {change_result.get('multi_signal')}")
    elif change_result:
        L(f"- Status: {change_result.get('status', 'unknown')}")
        L(f"- Changed pixels: {change_result.get('changed_pixels', 'N/A')}")
        L(f"- Changed %: {change_result.get('changed_pct', 'N/A')}%")
        L(f"- Regions: {change_result.get('num_regions', 'N/A')}")
    else:
        L("- Status: NOT AVAILABLE — change detection did not run")
    L()

    # ── GEOJSON OUTPUT ───────────────────────────────────────────
    L("OUTPUT")
    if change_result and change_result.get('change_geojson'):
        gj = change_result['change_geojson']
        features = gj.get('features', [])
        L(f"- GeoJSON features: {len(features)}")
        L(f"- GeoJSON type: {gj.get('type', 'N/A')}")
        if features:
            L(f"- First feature properties: {features[0].get('properties', {})}")
        # Verify serialization
        try:
            import json
            json.dumps(gj)
            L("- GeoJSON serialization: OK (survives json.dumps)")
        except Exception as ser_err:
            L(f"- GeoJSON serialization: FAILED — {ser_err}")
    elif change_result_obj and change_result_obj.change_geojson:
        gj = change_result_obj.change_geojson
        features = gj.get('features', [])
        L(f"- GeoJSON features: {len(features)}")
        if features:
            L(f"- First feature properties: {features[0].get('properties', {})}")
        try:
            import json
            json.dumps(gj)
            L("- GeoJSON serialization: OK")
        except Exception as ser_err:
            L(f"- GeoJSON serialization: FAILED — {ser_err}")
    else:
        L("- GeoJSON features: NOT AVAILABLE — no change detected or detection failed")

    # PNG
    if change_result_obj and change_result_obj.change_visualization_png:
        png_hex = change_result_obj.change_visualization_png
        png_len = len(png_hex)
        # Decode to check dimensions
        try:
            import struct as _struct
            # PNG IHDR chunk: offset 16 (8 sig + 8 ihdr chunk header), then width(4) + height(4)
            raw_png = bytes.fromhex(png_hex)
            ihdr_start = 8 + 8  # signature + chunk_len + chunk_type
            width = _struct.unpack('>I', raw_png[ihdr_start:ihdr_start+4])[0]
            height = _struct.unpack('>I', raw_png[ihdr_start+4:ihdr_start+8])[0]
            L(f"- PNG generated: YES ({width}x{height}, {png_len} hex chars)")
        except Exception:
            L(f"- PNG generated: YES ({png_len} hex chars, dimensions not parsed)")
    else:
        L("- PNG generated: NOT AVAILABLE")

    # Regions summary
    if change_result_obj and change_result_obj.regions:
        L(f"- Frontend polygons rendered: YES ({len(change_result_obj.regions)} regions with polygon_coords)")
        has_coords = all(
            r.get('polygon_coords') or True
            for r in change_result_obj.regions
        )
        L(f"- All regions have polygon coords: {has_coords}")
    else:
        L("- Frontend polygons rendered: NOT VERIFIED — no regions available")
    L()

    # ── E2E STATUS ───────────────────────────────────────────────
    L(SEP)
    L("E2E STATUS")
    L(SEP)
    pipeline_completed = (
        index_t1 is not None and index_t2 is not None
        and change_result_obj is not None
    )
    detection_executed = change_result_obj is not None
    geojson_generated = (
        (change_result_obj is not None and change_result_obj.change_geojson is not None)
        or (change_result is not None and change_result.get('change_geojson') is not None)
    )
    geojson_in_response = geojson_generated  # If generated, it's included in the result dict
    frontend_can_render = (
        detection_executed
        and change_result_obj is not None
        and len(change_result_obj.regions) > 0
    )
    L(f"- Pipeline completed: {'YES' if pipeline_completed else 'NO'}")
    L(f"- Detection executed: {'YES' if detection_executed else 'NO'}")
    L(f"- GeoJSON generated: {'YES' if geojson_generated else 'NO'}")
    L(f"- GeoJSON reached API response: {'YES' if geojson_in_response else 'NO'}")
    L(f"- Frontend can render regions: {'YES' if frontend_can_render else 'NO' if detection_executed else 'NOT VERIFIED'}")
    L()
    L(SEP)

    # Print to logger with [E2E] prefix on each line
    report = NL.join(lines)
    for line in report.split(NL):
        logger.info("[E2E] %s", line)


# ── Main pipeline ────────────────────────────────────────────────

def run_temporal_comparison(
    plan: dict[str, Any],
) -> TemporalComparisonResult:
    """
    Execute the full temporal comparison pipeline.

    Takes a validated analysis plan and produces:
    1. Scene selections for both time periods
    2. Spectral index computation for both periods
    3. Change detection between periods
    4. Computed metrics and explanation

    This is the single entry point for all EO phenomena.
    """
    processing_steps: list[dict[str, str]] = []

    phenomenon = plan["phenomenon"]
    analysis_type = plan.get("analysis_type", "ndvi_change")
    bbox = plan["bbox"]
    start_date = plan["start_date"]
    end_date = plan["end_date"]
    sensor = plan.get("sensor", "sentinel-2-l2a")
    collection = plan.get("collection", sensor)
    cloud_threshold = plan.get("cloud_threshold", 20)
    aoi_name = plan.get("aoi", "Unknown")
    plan_id = plan.get("plan_id", "unknown")

    # ── Semantic layer: read multi-signal config from plan ──────
    semantic_config = plan.get("semantic", {})
    indicators_config = plan.get("indicators", {})
    multi_signal_config = plan.get("multi_signal", {})
    evidence_config = plan.get("evidence_requirements", [])

    _lazy_import_heavy()
    # Get index name: prefer semantic primary_indicator, fallback to phenomenon config
    index_name = indicators_config.get("primary", None)
    if not index_name:
        pheno_config = _PHENOMENON_REGISTRY.get(phenomenon, {})
        index_name = pheno_config.get("default_index", "NDVI")
        if not index_name:
            index_name = "NDWI"

    # All indicators for this concept (used in multi-signal analysis)
    all_indicators = indicators_config.get("all", [index_name])
    multi_signal_enabled = multi_signal_config.get("enabled", False)
    signal_rules = multi_signal_config.get("rules", [])
    min_agreeing = multi_signal_config.get("min_agreeing_signals", 1)

    # Extract thresholds from plan's signal rules (single authoritative source)
    plan_ndvi_threshold: Optional[float] = None
    plan_ndbi_threshold: Optional[float] = None
    for rule in signal_rules:
        if rule.get("index_name") == "NDVI" and rule.get("direction") == "decrease":
            plan_ndvi_threshold = rule.get("threshold")
        elif rule.get("index_name") == "NDBI" and rule.get("direction") == "increase":
            plan_ndbi_threshold = rule.get("threshold")
    logger.info("[THRESHOLD] Plan NDBI threshold: %s, NDVI threshold: %s", plan_ndbi_threshold, plan_ndvi_threshold)

    # Track additional indicator results for multi-signal analysis
    additional_indices: dict[str, dict[str, IndexResult]] = {}  # indicator_name -> {"t1": ..., "t2": ...}

    processing_steps.append({
        "step": "plan_validation",
        "detail": f"phenomenon={phenomenon}, analysis_type={analysis_type}, sensor={sensor}, index={index_name}",
    })
    if multi_signal_enabled:
        processing_steps.append({
            "step": "multi_signal_config",
            "detail": f"Multi-signal enabled: indicators={all_indicators}, rules={len(signal_rules)}, min_agreeing={min_agreeing}",
        })
        if semantic_config.get("concept"):
            processing_steps.append({
                "step": "semantic_concept",
                "detail": f"Concept: {semantic_config['concept']} ({semantic_config.get('description', '')})",
            })

    # ── Step 1: Compute time windows ──────────────────────────────
    # Split the date range into two comparison periods
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
        # Fallback
        start_dt = datetime(2023, 1, 1)
        end_dt = datetime(2025, 12, 31)

    total_days = (end_dt - start_dt).days
    if total_days < 60:
        # Short range: use start and end as-is with ±30 day windows
        period1_start = (start_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        period1_end = (start_dt + timedelta(days=30)).strftime("%Y-%m-%d")
        period2_start = (end_dt - timedelta(days=30)).strftime("%Y-%m-%d")
        period2_end = (end_dt + timedelta(days=30)).strftime("%Y-%m-%d")
    else:
        # Long range: split in half, use narrow 45-day windows for faster search
        mid = start_dt + timedelta(days=total_days // 2)
        window = min(45, total_days // 4)  # 45-day search windows
        period1_start = start_dt.strftime("%Y-%m-%d")
        period1_end = (start_dt + timedelta(days=window)).strftime("%Y-%m-%d")
        period2_start = (end_dt - timedelta(days=window)).strftime("%Y-%m-%d")
        period2_end = end_dt.strftime("%Y-%m-%d")

    period1 = {"start": period1_start, "end": period1_end}
    period2 = {"start": period2_start, "end": period2_end}

    processing_steps.append({
        "step": "time_windows",
        "detail": f"Period 1: {period1_start} → {period1_end} | Period 2: {period2_start} → {period2_end}",
    })

    # ── Step 2: Search for scenes in each period (parallel) ────────
    logger.info("Searching scenes for period 1: %s to %s", period1_start, period1_end)
    logger.info("Searching scenes for period 2: %s to %s", period2_start, period2_end)

    search_limit = plan.get("max_scenes", 8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_t1 = executor.submit(
            _search_scenes, collection, bbox, period1_start, period1_end, cloud_threshold, search_limit
        )
        future_t2 = executor.submit(
            _search_scenes, collection, bbox, period2_start, period2_end, cloud_threshold, search_limit
        )
        items_t1 = future_t1.result()
        items_t2 = future_t2.result()

    # ── Auto-widen: if 0 scenes found, retry with wider windows ──
    for period_label, start_key, end_key, items_var in [
        ('Period 1', 'period1_start', 'period1_end', 'items_t1'),
        ('Period 2', 'period2_start', 'period2_end', 'items_t2'),
    ]:
        current_items = items_t1 if period_label == 'Period 1' else items_t2
        if len(current_items) == 0:
            # Try wider window: ±90 days from the period edge
            p_start = locals()[start_key]
            p_end = locals()[end_key]
            p_start_dt = datetime.strptime(p_start, '%Y-%m-%d')
            p_end_dt = datetime.strptime(p_end, '%Y-%m-%d')
            wider_start = (p_start_dt - timedelta(days=45)).strftime('%Y-%m-%d')
            wider_end = (p_end_dt + timedelta(days=45)).strftime('%Y-%m-%d')
            logger.info("%s had 0 scenes, retrying with wider window: %s to %s", period_label, wider_start, wider_end)
            wider_items = _search_scenes(collection, bbox, wider_start, wider_end, cloud_threshold, search_limit)
            if len(wider_items) > 0:
                if period_label == 'Period 1':
                    items_t1 = wider_items
                    period1_start = wider_start
                    period1_end = wider_end
                else:
                    items_t2 = wider_items
                    period2_start = wider_start
                    period2_end = wider_end
                processing_steps.append({
                    "step": f"widened_{period_label.lower().replace(' ', '_')}",
                    "detail": f"Widened to {wider_start} → {wider_end}, found {len(wider_items)} scenes",
                })
            else:
                # Try even wider: full year
                year_start = p_start[:4] + '-01-01'
                year_end = p_start[:4] + '-12-31'
                logger.info("%s still 0 scenes, trying full year: %s to %s", period_label, year_start, year_end)
                year_items = _search_scenes(collection, bbox, year_start, year_end, cloud_threshold, search_limit)
                if len(year_items) > 0:
                    if period_label == 'Period 1':
                        items_t1 = year_items
                        period1_start = year_start
                        period1_end = year_end
                    else:
                        items_t2 = year_items
                        period2_start = year_start
                        period2_end = year_end
                    processing_steps.append({
                        "step": f"full_year_{period_label.lower().replace(' ', '_')}",
                        "detail": f"Full year search {year_start} → {year_end}, found {len(year_items)} scenes",
                    })

    # Update period dicts with potentially widened windows
    period1 = {"start": period1_start, "end": period1_end}
    period2 = {"start": period2_start, "end": period2_end}

    processing_steps.append({
        "step": "search_periods",
        "detail": f"Period 1: {len(items_t1)} scenes | Period 2: {len(items_t2)} scenes",
    })

    # ── Step 3: Multi-scene selection for each period ──────────────
    # Use the scene selector to find enough scenes to cover the AOI.
    # This handles large AOIs that span multiple satellite footprints.
    from datetime import datetime as _dt

    period1_target = datetime.strptime(period1_start, "%Y-%m-%d")
    period2_target = datetime.strptime(period2_start, "%Y-%m-%d")

    selection_t1 = _select_scenes_for_period(
        aoi_bbox=bbox,
        scenes=items_t1,
        period_label="period1",
        target_date=period1_target,
        max_cloud_cover=cloud_threshold,
        required_sensor=collection,
    )

    selection_t2 = _select_scenes_for_period(
        aoi_bbox=bbox,
        scenes=items_t2,
        period_label="period2",
        target_date=period2_target,
        max_cloud_cover=cloud_threshold,
        required_sensor=collection,
    )

    # Check sensor compatibility
    periods_compatible, compat_warnings = _check_periods_compatible(selection_t1, selection_t2)
    processing_steps.append({
        "step": "scene_selection",
        "detail": (
            f"Period 1: {selection_t1.total_scenes} scenes, "
            f"coverage={selection_t1.coverage_ratio:.0%}, sensor={selection_t1.collection} | "
            f"Period 2: {selection_t2.total_scenes} scenes, "
            f"coverage={selection_t2.coverage_ratio:.0%}, sensor={selection_t2.collection}"
        ),
    })

    if compat_warnings:
        for w in compat_warnings:
            processing_steps.append({"step": "sensor_warning", "detail": w})

    if selection_t1.errors:
        for e in selection_t1.errors:
            processing_steps.append({"step": "selection_error_p1", "detail": e})
    if selection_t2.errors:
        for e in selection_t2.errors:
            processing_steps.append({"step": "selection_error_p2", "detail": e})

    # Build SceneSelection objects for backward compatibility
    scene_sel_t1 = _scene_to_selection(
        selection_t1.scenes[0].to_dict(), "planetary_computer"
    ) if selection_t1.scenes else None
    scene_sel_t2 = _scene_to_selection(
        selection_t2.scenes[0].to_dict(), "planetary_computer"
    ) if selection_t2.scenes else None

    # ── Step 4: Compute spectral indices (mosaic-aware) ───────────
    index_t1 = None
    index_t2 = None

    def _compute_for_period(sel: SceneSelectionResult, lbl: str) -> Optional[IndexResult]:
        """Compute index for a period, using mosaic if multiple scenes."""
        if not sel.scenes:
            return None

        scene_dicts = [s.to_dict() for s in sel.scenes]

        try:
            if sel.is_mosaic and len(sel.scenes) > 1:
                # Multi-scene: use mosaic pipeline
                logger.info(
                    "[%s] Using mosaic index computation (%d scenes)",
                    lbl, len(sel.scenes),
                )
                return _compute_index_from_mosaic_scenes(
                    index_name=index_name,
                    sensor=sel.collection,
                    scenes=scene_dicts,
                    bbox=bbox,
                    period_label=lbl,
                )
            else:
                # Single scene: use existing pipeline
                single_scene = sel.scenes[0]
                scene_sel = _scene_to_selection(single_scene.to_dict(), "planetary_computer")
                return _compute_index_stats(index_name, sel.collection, scene_sel, bbox)
        except Exception as e:
            logger.error("[%s] Index computation failed: %s", lbl, e)
            # Fall back to single-scene if mosaic fails
            if sel.is_mosaic and len(sel.scenes) > 1:
                logger.info("[%s] Mosaic failed, falling back to best single scene", lbl)
                try:
                    scene_sel = _scene_to_selection(sel.scenes[0].to_dict(), "planetary_computer")
                    return _compute_index_stats(index_name, sel.collection, scene_sel, bbox)
                except Exception as e2:
                    logger.error("[%s] Fallback also failed: %s", lbl, e2)
            return None

    # 4a: Compute primary index — SEQUENTIALLY to avoid OOM on Render free tier
    # Running both periods in parallel doubles peak memory usage
    if selection_t1.scenes:
        result = _compute_for_period(selection_t1, "period1")
        index_t1 = result
        if result:
            scene_ids = ",".join(s.item_id[:20] for s in selection_t1.scenes[:3])
            processing_steps.append({
                "step": "compute_index_t1",
                "detail": (
                    f"{index_name} for {scene_ids}: "
                    f"mean={result.stats['mean']}, "
                    f"scenes={selection_t1.total_scenes}, "
                    f"mosaic={selection_t1.is_mosaic}"
                ),
            })
        gc.collect()  # Free memory between periods

    if selection_t2.scenes:
        result = _compute_for_period(selection_t2, "period2")
        index_t2 = result
        if result:
            scene_ids = ",".join(s.item_id[:20] for s in selection_t2.scenes[:3])
            processing_steps.append({
                "step": "compute_index_t2",
                "detail": (
                    f"{index_name} for {scene_ids}: "
                    f"mean={result.stats['mean']}, "
                    f"scenes={selection_t2.total_scenes}, "
                    f"mosaic={selection_t2.is_mosaic}"
                ),
            })
        gc.collect()  # Free memory after period 2

    # Force GC after index computation to free raster memory before change detection
    gc.collect()

    # 4b: Compute additional indicators for multi-signal analysis
    # Use the SAME mosaic scenes as the primary index for shape consistency.
    if multi_signal_enabled and len(all_indicators) > 1:
        additional_indicator_names = [ind for ind in all_indicators if ind != index_name]
        for extra_idx_name in additional_indicator_names:
            extra_t1 = None
            extra_t2 = None
            # Compute sequentially to avoid OOM on Render free tier
            if selection_t1.scenes and len(selection_t1.scenes) > 0:
                scene_dicts_t1 = [s.to_dict() for s in selection_t1.scenes]
                extra_t1 = _compute_index_from_mosaic_scenes(
                    extra_idx_name, sensor, scene_dicts_t1, bbox, "period1_ndvi",
                )
                gc.collect()
            if selection_t2.scenes and len(selection_t2.scenes) > 0:
                scene_dicts_t2 = [s.to_dict() for s in selection_t2.scenes]
                extra_t2 = _compute_index_from_mosaic_scenes(
                    extra_idx_name, sensor, scene_dicts_t2, bbox, "period2_ndvi",
                )
                gc.collect()
            if extra_t1 or extra_t2:
                logger.info(
                    "[SHAPE-TRACE] NDVI t1: shape=%s valid=%d t2: shape=%s valid=%d",
                    extra_t1.shape if extra_t1 else None,
                    extra_t1.valid_pixels if extra_t1 else 0,
                    extra_t2.shape if extra_t2 else None,
                    extra_t2.valid_pixels if extra_t2 else 0,
                )
                additional_indices[extra_idx_name] = {"t1": extra_t1, "t2": extra_t2}
                processing_steps.append({
                    "step": f"compute_index_{extra_idx_name.lower()}",
                    "detail": f"{extra_idx_name} (supporting indicator): t1_mean={extra_t1.stats['mean'] if extra_t1 else 'N/A'}, t2_mean={extra_t2.stats['mean'] if extra_t2 else 'N/A'}",
                })

    # ── Step 5: Common-grid reprojection + canonical change detection ────
    change_result = None
    change_result_obj = None
    analysis_grid_info = None
    change_mask_b64 = None
    diff_vis_b64 = None
    change_vis_stats = {}
    scl_available = False
    cloud_mask_t1 = None
    cloud_mask_t2 = None
    t1_aligned = None
    t2_aligned = None

    if index_t1 and index_t2 and index_t1.value is not None and index_t2.value is not None:
        try:
            from app.services.raster_service import reproject_to_grid, determine_common_grid
            from app.services.change_detection import run_change_detection

            # Get the raster hrefs for determining the common grid
            # We need the actual COG URLs to read CRS/transform info
            band_key_t1 = (selection_t1.collection, index_name)
            band_key_t2 = (selection_t2.collection, index_name)
            band_map_t1 = _INDEX_BAND_MAP.get(band_key_t1)
            band_map_t2 = _INDEX_BAND_MAP.get(band_key_t2)

            if band_map_t1 and band_map_t2 and selection_t1.scenes and selection_t2.scenes:
                # Get the first required band href from each period's first scene
                index_def = _INDEX_DEFINITIONS.get(index_name)
                if index_def:
                    first_logical = index_def.bands_required[0]
                    physical_t1 = band_map_t1.get(first_logical)
                    physical_t2 = band_map_t2.get(first_logical)

                    href1 = ""
                    href2 = ""
                    if physical_t1:
                        asset = selection_t1.scenes[0].assets.get(physical_t1)
                        if asset:
                            href1 = asset.get("href", "") if isinstance(asset, dict) else getattr(asset, "href", "") or ""
                    if physical_t2:
                        asset = selection_t2.scenes[0].assets.get(physical_t2)
                        if asset:
                            href2 = asset.get("href", "") if isinstance(asset, dict) else getattr(asset, "href", "") or ""

                    if href1 and href2:
                        # Determine common analysis grid (use safe_max_dim for memory safety)
                        safe_max_dim = plan.get("_safe_max_dim", 512)
                        analysis_grid_info = determine_common_grid(href1, href2, bbox, max_dim=safe_max_dim)
                        analysis_crs = str(analysis_grid_info["crs"])
                        analysis_transform = analysis_grid_info["transform"]
                        analysis_shape = (analysis_grid_info["height"], analysis_grid_info["width"])
                        analysis_resolution = analysis_grid_info["resolution_meters"]

                        processing_steps.append({
                            "step": "common_grid",
                            "detail": (
                                f"crs={analysis_crs}, res={analysis_resolution:.1f}m, "
                                f"{analysis_shape[1]}x{analysis_shape[0]}, "
                                f"source1={analysis_grid_info['source_crs1']}, "
                                f"source2={analysis_grid_info['source_crs2']}"
                            ),
                        })

                        # Read raster metadata to get source CRS/transform for reprojection
                        import rasterio
                        import os as _os_raster
                        for _k in ['GDAL_CACHEMAX', 'GDAL_DISABLE_READDIR_ON_OPEN',
                                   'CPL_VSIL_CURL_ALLOWED_EXTENSIONS', 'GDAL_HTTP_TIMEOUT', 'GDAL_HTTP_MAX_RETRY']:
                            if _k in _os_raster.environ:
                                del _os_raster.environ[_k]

                        # Read source grid info from both scenes
                        with rasterio.Env(GDAL_CACHEMAX=64, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30"):
                            with rasterio.open(href1) as src1:
                                src1_crs = src1.crs
                                src1_transform = src1.transform

                            with rasterio.open(href2) as src2:
                                src2_crs = src2.crs
                                src2_transform = src2.transform

                        # Reproject both index arrays onto the common grid
                        t1_aligned = reproject_to_grid(
                            index_t1.value, src1_crs, src1_transform,
                            analysis_grid_info["crs"], analysis_transform, analysis_shape,
                        )
                        t2_aligned = reproject_to_grid(
                            index_t2.value, src2_crs, src2_transform,
                            analysis_grid_info["crs"], analysis_transform, analysis_shape,
                        )

                        logger.info(
                            "[SHAPE-TRACE] Common grid target: shape=%s crs=%s res=%.2fm",
                            analysis_shape, analysis_crs, analysis_resolution,
                        )
                        logger.info(
                            "[SHAPE-TRACE] T1 NDBI BEFORE reprojection: %s src_crs=%s",
                            index_t1.value.shape, src1_crs,
                        )
                        logger.info(
                            "[SHAPE-TRACE] T2 NDBI BEFORE reprojection: %s src_crs=%s",
                            index_t2.value.shape, src2_crs,
                        )
                        logger.info(
                            "[SHAPE-TRACE] T1 AFTER reprojection: %s",
                            t1_aligned.shape,
                        )
                        logger.info(
                            "[SHAPE-TRACE] T2 AFTER reprojection: %s",
                            t2_aligned.shape,
                        )
                        logger.info(
                            "Common grid reprojection: t1 %s→%s, t2 %s→%s",
                            src1_crs, analysis_crs, src2_crs, analysis_crs,
                        )
                    else:
                        # Cannot determine common grid — geographic correspondence NOT guaranteed.
                        # Do NOT silently crop. Fail clearly.
                        processing_steps.append({
                            "step": "common_grid",
                            "detail": "FAILED: cannot determine common grid (no band hrefs available).",
                        })
                        raise RuntimeError(
                            "Cannot determine common analysis grid — band hrefs unavailable. "
                            f"Period 1 shape: {index_t1.value.shape}, Period 2 shape: {index_t2.value.shape}. "
                            "Geographic correspondence cannot be guaranteed."
                        )
                else:
                    # No band mapping available for this sensor/index combination.
                    processing_steps.append({
                        "step": "common_grid",
                        "detail": "FAILED: no band mapping for sensor/index.",
                    })
                    raise RuntimeError(
                        f"No band mapping for {index_name} on {selection_t1.collection}. "
                        "Cannot determine common analysis grid."
                    )
            else:
                # No scenes available for grid determination.
                processing_steps.append({
                    "step": "common_grid",
                    "detail": "FAILED: no scenes available for grid determination.",
                })
                raise RuntimeError(
                    "No scenes available for common grid determination. "
                    "Cannot compare periods without aligned rasters."
                )

            # ── SCL cloud/shadow masking (Sentinel-2 only) ────
            # SCL is categorical integer data at 20m resolution.
            # It MUST be reprojected onto the exact analysis grid using
            # nearest-neighbor resampling to preserve class values.
            cloud_mask_t1 = None
            cloud_mask_t2 = None
            scl_available = False
            if selection_t1.collection == 'sentinel-2-l2a' and selection_t1.scenes:
                try:
                    scl_asset_t1 = selection_t1.scenes[0].assets.get('SCL')
                    scl_asset_t2 = selection_t2.scenes[0].assets.get('SCL') if selection_t2.scenes else None
                    scl_href1 = scl_asset_t1.get('href', '') if isinstance(scl_asset_t1, dict) else ''
                    scl_href2 = scl_asset_t2.get('href', '') if isinstance(scl_asset_t2, dict) else ''
                    if scl_href1 and scl_href2:
                        from app.services.raster_service import read_raster_window, reproject_to_grid
                        from app.services.compositor import create_cloud_mask
                        import rasterio as _rio_scl

                        # Read SCL band from both periods (use safe max_dim for memory safety)
                        safe_max_dim = plan.get("_safe_max_dim", 512)
                        scl_data_t1 = read_raster_window(scl_href1, bbox, max_dim=safe_max_dim)
                        scl_data_t2 = read_raster_window(scl_href2, bbox, max_dim=safe_max_dim)
                        scl_arr_t1 = scl_data_t1['data']
                        scl_arr_t2 = scl_data_t2['data']
                        if scl_arr_t1.ndim == 3:
                            scl_arr_t1 = scl_arr_t1[0]
                        if scl_arr_t2.ndim == 3:
                            scl_arr_t2 = scl_arr_t2[0]

                        # Get SCL source CRS and transform
                        scl_crs_t1 = scl_data_t1.get('crs', 'EPSG:4326')
                        scl_transform_t1 = scl_data_t1.get('transform')
                        scl_crs_t2 = scl_data_t2.get('crs', 'EPSG:4326')
                        scl_transform_t2 = scl_data_t2.get('transform')

                        # CRITICAL: Reproject SCL onto the exact analysis grid
                        # using nearest-neighbor to preserve integer class values.
                        # Bilinear/cubic would interpolate between classes (wrong).
                        scl_reproj_t1 = reproject_to_grid(
                            scl_arr_t1.astype(np.float32),
                            scl_crs_t1, scl_transform_t1,
                            analysis_grid_info['crs'], analysis_transform, analysis_shape,
                            resampling_method='nearest',
                        )
                        scl_reproj_t2 = reproject_to_grid(
                            scl_arr_t2.astype(np.float32),
                            scl_crs_t2, scl_transform_t2,
                            analysis_grid_info['crs'], analysis_transform, analysis_shape,
                            resampling_method='nearest',
                        )

                        # Verify SCL is on the exact same grid as the analysis arrays
                        assert scl_reproj_t1.shape == t1_aligned.shape, (
                            f"SCL shape {scl_reproj_t1.shape} != analysis shape {t1_aligned.shape}"
                        )
                        assert scl_reproj_t2.shape == t2_aligned.shape, (
                            f"SCL shape {scl_reproj_t2.shape} != analysis shape {t2_aligned.shape}"
                        )

                        # Create boolean cloud mask from reprojected SCL
                        # SCL values are integers; nearest-neighbor preserves them
                        cloud_mask_t1 = create_cloud_mask(scl_reproj_t1)
                        cloud_mask_t2 = create_cloud_mask(scl_reproj_t2)
                        scl_available = True
                        scl_cloud_pct_t1 = (1.0 - np.mean(cloud_mask_t1)) * 100
                        scl_cloud_pct_t2 = (1.0 - np.mean(cloud_mask_t2)) * 100
                        processing_steps.append({
                            "step": "scl_masking",
                            "detail": (
                                f"SCL reprojected to analysis grid (nearest-neighbor): "
                                f"{scl_reproj_t1.shape[1]}x{scl_reproj_t1.shape[0]} @ "
                                f"{analysis_grid_info['crs']}, "
                                f"cloud={scl_cloud_pct_t1:.1f}%/{scl_cloud_pct_t2:.1f}% "
                                f"(period1/period2)"
                            ),
                        })
                except Exception as e:
                    logger.warning("SCL masking failed: %s — continuing without cloud mask", e, exc_info=True)
            if not scl_available:
                processing_steps.append({
                    "step": "scl_masking",
                    "detail": "SCL unavailable — using nodata-only masking",
                })

            # ── SHAPE VERIFICATION: all inputs to change detection ────
            logger.info(
                "[SHAPE-TRACE] PRE-CHANGE-DETECTION: t1_NDBI=%s t2_NDBI=%s",
                t1_aligned.shape if t1_aligned is not None else None,
                t2_aligned.shape if t2_aligned is not None else None,
            )
            logger.info(
                "[SHAPE-TRACE] PRE-CHANGE-DETECTION: cloud_t1=%s cloud_t2=%s",
                cloud_mask_t1.shape if cloud_mask_t1 is not None else None,
                cloud_mask_t2.shape if cloud_mask_t2 is not None else None,
            )
            logger.info(
                "[SHAPE-TRACE] PRE-CHANGE-DETECTION: grid crs=%s transform=%s shape=%s res=%.2fm",
                analysis_crs, analysis_transform, analysis_shape, analysis_resolution,
            )

            # ── Reproject NDVI support index to analysis grid ────
            ndvi_baseline_aligned = None
            ndvi_comparison_aligned = None
            ndvi_data = additional_indices.get("NDVI")
            if ndvi_data and ndvi_data.get("t1") and ndvi_data.get("t2"):
                ndvi_t1_obj = ndvi_data["t1"]
                ndvi_t2_obj = ndvi_data["t2"]
                if ndvi_t1_obj.value is not None and ndvi_t2_obj.value is not None:
                    try:
                        # Read NDVI source CRS/transform from the first scene's B08 band
                        ndvi_band_key = (selection_t1.collection, "NDVI")
                        ndvi_band_map = _INDEX_BAND_MAP.get(ndvi_band_key)
                        if ndvi_band_map and selection_t1.scenes and selection_t2.scenes:
                            ndvi_index_def = _INDEX_DEFINITIONS.get("NDVI")
                            if ndvi_index_def:
                                ndvi_logical = ndvi_index_def.bands_required[0]
                                ndvi_physical_t1 = ndvi_band_map.get(ndvi_logical)
                                ndvi_physical_t2 = ndvi_band_map.get(ndvi_logical)
                                ndvi_href1 = ""
                                ndvi_href2 = ""
                                if ndvi_physical_t1:
                                    asset = selection_t1.scenes[0].assets.get(ndvi_physical_t1)
                                    if asset:
                                        ndvi_href1 = asset.get("href", "") if isinstance(asset, dict) else getattr(asset, "href", "") or ""
                                if ndvi_physical_t2:
                                    asset = selection_t2.scenes[0].assets.get(ndvi_physical_t2)
                                    if asset:
                                        ndvi_href2 = asset.get("href", "") if isinstance(asset, dict) else getattr(asset, "href", "") or ""

                                if ndvi_href1 and ndvi_href2:
                                    with rasterio.Env(GDAL_CACHEMAX=64, GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_TIMEOUT="30"):
                                        with rasterio.open(ndvi_href1) as ndvi_src1:
                                            ndvi_src1_crs = ndvi_src1.crs
                                            ndvi_src1_transform = ndvi_src1.transform
                                        with rasterio.open(ndvi_href2) as ndvi_src2:
                                            ndvi_src2_crs = ndvi_src2.crs
                                            ndvi_src2_transform = ndvi_src2.transform

                                    ndvi_baseline_aligned = reproject_to_grid(
                                        ndvi_t1_obj.value, ndvi_src1_crs, ndvi_src1_transform,
                                        analysis_grid_info["crs"], analysis_transform, analysis_shape,
                                    )
                                    ndvi_comparison_aligned = reproject_to_grid(
                                        ndvi_t2_obj.value, ndvi_src2_crs, ndvi_src2_transform,
                                        analysis_grid_info["crs"], analysis_transform, analysis_shape,
                                    )
                                    logger.info(
                                        "[SHAPE-TRACE] NDVI AFTER reprojection: t1=%s t2=%s",
                                        ndvi_baseline_aligned.shape, ndvi_comparison_aligned.shape,
                                    )
                    except Exception as e:
                        logger.warning("NDVI reprojection failed: %s — running without NDVI support", e, exc_info=True)
                        ndvi_baseline_aligned = None
                        ndvi_comparison_aligned = None

            # Free memory before change detection (raster arrays from index computation)
            gc.collect()

            # Run change detection through the unified detector interface
            from app.services.detector_interface import detect_change, get_supported_methods
            detector_method = plan.get("change_detection_method", "phenomenon_aware_difference")
            logger.info("[DETECTOR] Selected method: %s", detector_method)

            # Pre-compute scene IDs and provenance data for GeoJSON enrichment
            _prov_scene_ids_t1 = [s.item_id for s in selection_t1.scenes] if selection_t1 and selection_t1.scenes else []
            _prov_scene_ids_t2 = [s.item_id for s in selection_t2.scenes] if selection_t2 and selection_t2.scenes else []
            _prov_acq_dates_t1 = [s.datetime for s in selection_t1.scenes] if selection_t1 and selection_t1.scenes else []
            _prov_acq_dates_t2 = [s.datetime for s in selection_t2.scenes] if selection_t2 and selection_t2.scenes else []
            _prov_cloud_covers_t1 = [s.cloud_cover for s in selection_t1.scenes] if selection_t1 and selection_t1.scenes else []
            _prov_cloud_covers_t2 = [s.cloud_cover for s in selection_t2.scenes] if selection_t2 and selection_t2.scenes else []
            _prov_all_scene_ids = _prov_scene_ids_t1 + _prov_scene_ids_t2
            _prov_aoi_coverage = (selection_t1.coverage_ratio if selection_t1 else None)

            # Quality mask provenance
            _prov_scl_usage = "SCL categorical masking (nearest-neighbor reprojected)" if scl_available else "nodata-only masking"
            _prov_cloud_shadow_handling = "SCL classes 3,8,9,10 masked" if scl_available else "nodata threshold"
            _prov_quality_mask = "SCL (Scene Classification Layer)" if scl_available else "nodata"

            # Index formulas
            _prov_formulas = {}
            if _INDEX_DEFINITIONS:
                _prov_formulas[index_name] = _INDEX_DEFINITIONS[index_name].formula
                for r in signal_rules:
                    rname = r.get("index_name")
                    if rname and rname != index_name and rname in _INDEX_DEFINITIONS:
                        _prov_formulas[rname] = _INDEX_DEFINITIONS[rname].formula

            # Composite method provenance
            _prov_n_t1 = len(selection_t1.scenes) if selection_t1 and selection_t1.scenes else 0
            _prov_n_t2 = len(selection_t2.scenes) if selection_t2 and selection_t2.scenes else 0
            _prov_composite = f"StackSTAC mosaic ({_prov_n_t1} scenes)" if (_prov_n_t1 > 1) else f"single scene read"

            _prov_common_kwargs = dict(
                scene_ids=_prov_all_scene_ids,
                collection=collection,
                supporting_indices=[r.get("index_name") for r in signal_rules if r.get("index_name") != index_name] if signal_rules else [],
                indicator_formulas=_prov_formulas,
                quality_mask=_prov_quality_mask,
                scl_usage=_prov_scl_usage,
                cloud_shadow_handling=_prov_cloud_shadow_handling,
                composite_method=_prov_composite,
                observation_count={"period_1": _prov_n_t1, "period_2": _prov_n_t2},
                provider="planetary_computer",
                platform=selection_t1.platform if selection_t1 else "unknown",
                instrument="MSI" if "sentinel" in collection.lower() else "OLI",
                processing_level="L2A" if "l2a" in collection.lower() else "unknown",
                acquisition_dates=_prov_acq_dates_t1 + _prov_acq_dates_t2,
                cloud_cover=_prov_cloud_covers_t1 + _prov_cloud_covers_t2,
                aoi_coverage=_prov_aoi_coverage,
            )

            try:
                change_result_obj = detect_change(
                    baseline=t1_aligned,
                    comparison=t2_aligned,
                    index_name=index_name,
                    aoi_bbox=bbox,
                    detector_method=detector_method,
                    baseline_date=index_t1.date,
                    comparison_date=index_t2.date,
                    crs=analysis_crs,
                    resolution_meters=analysis_resolution,
                    phenomenon=phenomenon,
                    baseline_transform=analysis_transform,
                    comparison_transform=analysis_transform,
                    # Invert SCL masks: create_cloud_mask returns True=VALID,
                    # but compute_valid_mask expects True=CLOUD
                    cloud_baseline=~cloud_mask_t1 if cloud_mask_t1 is not None else None,
                    cloud_comparison=~cloud_mask_t2 if cloud_mask_t2 is not None else None,
                    # Pass reprojected NDVI for multi-signal AND logic
                    ndvi_baseline=ndvi_baseline_aligned,
                    ndvi_comparison=ndvi_comparison_aligned,
                    # Pass explicit thresholds from plan's signal rules (authoritative source)
                    ndbi_threshold=plan_ndbi_threshold,
                    ndvi_decrease_threshold=plan_ndvi_threshold,
                    **_prov_common_kwargs,
                )
            except (ValueError, TypeError) as e:
                # Fallback to default method if selected method fails (e.g. CVA needs multi-band)
                logger.warning(
                    "[DETECTOR] Method '%s' failed: %s. Falling back to phenomenon_aware_difference",
                    detector_method, e,
                )
                change_result_obj = detect_change(
                    baseline=t1_aligned,
                    comparison=t2_aligned,
                    index_name=index_name,
                    aoi_bbox=bbox,
                    detector_method="phenomenon_aware_difference",
                    baseline_date=index_t1.date,
                    comparison_date=index_t2.date,
                    crs=analysis_crs,
                    resolution_meters=analysis_resolution,
                    phenomenon=phenomenon,
                    baseline_transform=analysis_transform,
                    comparison_transform=analysis_transform,
                    cloud_baseline=~cloud_mask_t1 if cloud_mask_t1 is not None else None,
                    cloud_comparison=~cloud_mask_t2 if cloud_mask_t2 is not None else None,
                    ndvi_baseline=ndvi_baseline_aligned,
                    ndvi_comparison=ndvi_comparison_aligned,
                    ndvi_decrease_threshold=plan_ndvi_threshold,
                    **_prov_common_kwargs,
                )

            # Convert ChangeDetectionResult to dict for API response
            change_result = {
                "status": change_result_obj.status,
                "algorithm": change_result_obj.algorithm,
                "index_name": change_result_obj.index_name,
                "baseline_date": change_result_obj.baseline_date,
                "comparison_date": change_result_obj.comparison_date,
                "changed_pct": change_result_obj.changed_pct,
                "changed_pixels": change_result_obj.changed_pixels,
                "total_pixels": change_result_obj.total_pixels,
                "num_regions": change_result_obj.num_regions,
                "changed_area_sq_meters": change_result_obj.changed_area_sq_meters,
                "total_area_sq_meters": change_result_obj.total_area_sq_meters,
                "valid_area_sq_meters": change_result_obj.valid_area_sq_meters,
                "baseline_stats": change_result_obj.baseline_stats,
                "comparison_stats": change_result_obj.comparison_stats,
                "difference_stats": change_result_obj.difference_stats,
                "raster_derived": True,
                "threshold": change_result_obj.parameters.get("threshold"),
                "change_stats": change_result_obj.difference_stats,
                "crs": change_result_obj.crs,
                "resolution_meters": change_result_obj.resolution_meters,
                "valid_pixel_ratio": change_result_obj.valid_pixel_ratio,
                "nodata_pixels": change_result_obj.nodata_pixels,
                "cloud_masked_pixels": change_result_obj.cloud_masked_pixels,
            }

            # GeoJSON regions from the canonical pipeline
            if change_result_obj.change_geojson:
                change_result["change_geojson"] = change_result_obj.change_geojson
                # Enrich top-level GeoJSON properties with scene_ids and collection
                all_scene_ids = _prov_all_scene_ids
                change_result["change_geojson"]["properties"]["scene_ids"] = all_scene_ids
                change_result["change_geojson"]["properties"]["collection"] = collection
            if change_result_obj.regions:
                change_result["regions"] = change_result_obj.regions

            # ── CONSISTENCY VALIDATION ────────────────────────────
            # Verify that region counts and area calculations agree.
            # If any disagree, log a clear failure (do NOT round until they match).
            _api_region_count = change_result.get("num_regions", 0)
            _geojson_region_count = len(change_result_obj.change_geojson.get("features", [])) if change_result_obj.change_geojson else 0
            _frontend_region_count = len(change_result_obj.regions)

            pixel_area_sq_m = analysis_resolution ** 2
            _calc_changed_area = change_result_obj.changed_pixels * pixel_area_sq_m
            _reported_changed_area = change_result_obj.changed_area_sq_meters

            region_count_ok = (_api_region_count == _geojson_region_count == _frontend_region_count)
            area_calc_ok = (abs(_calc_changed_area - _reported_changed_area) < 1.0)  # 1 m² tolerance

            if not region_count_ok:
                logger.error(
                    "[CONSISTENCY-FAIL] Region count mismatch: "
                    "api=%d, geojson=%d, frontend=%d",
                    _api_region_count, _geojson_region_count, _frontend_region_count,
                )
            if not area_calc_ok:
                logger.error(
                    "[CONSISTENCY-FAIL] Area calculation mismatch: "
                    "changed_pixels(%d) * pixel_area(%.1f m²) = %.1f m², "
                    "reported = %.1f m², diff = %.1f m²",
                    change_result_obj.changed_pixels, pixel_area_sq_m,
                    _calc_changed_area, _reported_changed_area,
                    abs(_calc_changed_area - _reported_changed_area),
                )

            change_result["consistency_validation"] = {
                "region_count": {
                    "api": _api_region_count,
                    "geojson": _geojson_region_count,
                    "frontend": _frontend_region_count,
                    "consistent": region_count_ok,
                },
                "area_calculation": {
                    "changed_pixels": change_result_obj.changed_pixels,
                    "pixel_area_m2": round(pixel_area_sq_m, 2),
                    "calculated_area_m2": round(_calc_changed_area, 2),
                    "reported_area_m2": round(_reported_changed_area, 2),
                    "difference_m2": round(abs(_calc_changed_area - _reported_changed_area), 2),
                    "consistent": area_calc_ok,
                },
            }

            # Change mask visualization (sparse — only changed pixels)
            change_mask_b64 = change_result_obj.change_visualization_png

            # Statistics from the cleaned mask
            pixel_area_sq_m = analysis_resolution ** 2
            loss_pixels = 0
            gain_pixels = 0
            for region in change_result_obj.regions:
                if region.get("direction") == "decrease":
                    loss_pixels += region.get("area_pixels", 0)
                elif region.get("direction") == "increase":
                    gain_pixels += region.get("area_pixels", 0)
            stable_pixels = change_result_obj.total_pixels - change_result_obj.changed_pixels
            loss_area_km2 = loss_pixels * pixel_area_sq_m / 1e6
            gain_area_km2 = gain_pixels * pixel_area_sq_m / 1e6

            if loss_pixels > gain_pixels * 1.5:
                dominant_trend = "decrease"
            elif gain_pixels > loss_pixels * 1.5:
                dominant_trend = "increase"
            else:
                dominant_trend = "stable_mixed"

            change_vis_stats = {
                "decrease_pixels": loss_pixels,
                "increase_pixels": gain_pixels,
                "stable_pixels": stable_pixels,
                "decrease_area_km2": round(loss_area_km2, 2),
                "increase_area_km2": round(gain_area_km2, 2),
                "total_analyzed_km2": round(change_result_obj.total_area_sq_meters / 1e6, 2),
                "dominant_trend": dominant_trend,
                "threshold": change_result_obj.parameters.get("threshold"),
                "total_valid_pixels": change_result_obj.total_pixels,
                "num_regions": change_result_obj.num_regions,
            }

            # Difference visualization (sparse — only significant pixels colored)
            delta = t2_aligned - t1_aligned
            valid_vis = ~np.isnan(t1_aligned) & ~np.isnan(t2_aligned)
            threshold_vis = change_result_obj.parameters.get("threshold") or 0.12
            diff_clipped = np.clip(np.nan_to_num(delta, nan=0.0), -0.5, 0.5)
            diff_norm = ((diff_clipped + 0.5) / 1.0 * 255).astype(np.uint8)
            diff_img = np.zeros((*delta.shape, 4), dtype=np.uint8)
            sig_mask = valid_vis & (np.abs(np.nan_to_num(delta, nan=0.0)) >= threshold_vis)
            diff_img[sig_mask, 0] = np.where(delta[sig_mask] > 0, diff_norm[sig_mask], 80)
            diff_img[sig_mask, 1] = 50
            diff_img[sig_mask, 2] = np.where(delta[sig_mask] < 0, diff_norm[sig_mask], 80)
            diff_img[sig_mask, 3] = 160
            diff_img[~valid_vis] = [13, 23, 17, 255]
            diff_vis_b64 = _encode_rgba_png(diff_img)

            processing_steps.append({
                "step": "change_detection",
                "detail": (
                    f"algorithm={change_result_obj.algorithm}, "
                    f"changed={change_result_obj.changed_pixels}/{change_result_obj.total_pixels} "
                    f"({change_result_obj.changed_pct}%), "
                    f"regions={change_result_obj.num_regions}, "
                    f"threshold={change_result_obj.parameters.get('threshold')}"
                ),
            })

            logger.info(
                "[%s] Change detection: %d/%d pixels changed (%.2f%%), %d regions, algorithm=%s",
                index_name, change_result_obj.changed_pixels, change_result_obj.total_pixels,
                change_result_obj.changed_pct, change_result_obj.num_regions, change_result_obj.algorithm,
            )

        except Exception as e:
            logger.error("Change detection failed: %s", e, exc_info=True)
            processing_steps.append({
                "step": "change_detection",
                "detail": f"FAILED: {type(e).__name__}: {str(e)[:200]}",
            })
            change_result = None


    # ── Step 6: Compute metrics ───────────────────────────────────
    metrics = {}
    if index_t1 and index_t2:
        metrics = _compute_comparison_metrics(
            phenomenon, index_name, index_t1, index_t2, change_result, bbox,
        )

    processing_steps.append({
        "step": "metrics_computation",
        "detail": f"Computed {len(metrics)} metrics",
    })

    # ── Step 7: Extract imagery URLs ──────────────────────────────
    imagery = {
        "period1": {},
        "period2": {},
    }
    if scene_sel_t1:
        imagery["period1"] = _get_imagery_urls(scene_sel_t1)
        imagery["period1"]["scene_id"] = scene_sel_t1.item_id
        imagery["period1"]["date"] = scene_sel_t1.datetime
        imagery["period1"]["cloud_cover"] = scene_sel_t1.cloud_cover
        imagery["period1"]["platform"] = scene_sel_t1.platform
        imagery["period1"]["bbox"] = scene_sel_t1.bbox or []
        imagery["period1"]["collection"] = scene_sel_t1.collection
    if scene_sel_t2:
        imagery["period2"] = _get_imagery_urls(scene_sel_t2)
        imagery["period2"]["scene_id"] = scene_sel_t2.item_id
        imagery["period2"]["date"] = scene_sel_t2.datetime
        imagery["period2"]["cloud_cover"] = scene_sel_t2.cloud_cover
        imagery["period2"]["platform"] = scene_sel_t2.platform
        imagery["period2"]["bbox"] = scene_sel_t2.bbox or []
        imagery["period2"]["collection"] = scene_sel_t2.collection

    # ── Step 8: Generate explanation ──────────────────────────────
    explanation = _generate_explanation(
        phenomenon,
        aoi_name,
        metrics,
        period1_start,
        period2_end,
        index_name,
    )

    # ── Sensor info ───────────────────────────────────────────────
    sensor_info = {
        "primary_sensor": sensor,
        "collection": collection,
        "index_used": index_name,
        "resolution_m": index_t1.resolution_m if index_t1 else 10.0,
        "bands_used": _INDEX_BAND_MAP.get((sensor, index_name), {}),
        "index_formula": _INDEX_DEFINITIONS.get(index_name, {}).formula if index_name in _INDEX_DEFINITIONS else "",
        "index_description": _INDEX_DEFINITIONS.get(index_name, {}).description if index_name in _INDEX_DEFINITIONS else "",
    }

    # ── Fix undefined ensemble_stats ──────────────────────────────
    ensemble_stats = None

    # ── E2E Diagnostic Report ──────────────────────────────────────
    _print_e2e_diagnostic(
        query=plan.get("query", plan.get("aoi", "unknown")),
        phenomenon=phenomenon,
        index_name=index_name,
        period1=period1,
        period2=period2,
        scene_sel_t1=selection_t1,
        scene_sel_t2=selection_t2,
        scene_sel_obj_t1=scene_sel_t1,
        scene_sel_obj_t2=scene_sel_t2,
        index_t1=index_t1,
        index_t2=index_t2,
        analysis_grid_info=analysis_grid_info,
        scl_available=scl_available,
        cloud_mask_t1=cloud_mask_t1,
        cloud_mask_t2=cloud_mask_t2,
        t1_aligned=t1_aligned,
        t2_aligned=t2_aligned,
        change_result_obj=change_result_obj,
        change_result=change_result,
        bbox=bbox,
        multi_signal_enabled=multi_signal_enabled,
        signal_rules=signal_rules,
        min_agreeing=min_agreeing,
        collection=collection,
        sensor=sensor,
    )

    # ── Build processing provenance ────────────────────────────
    # Captures Sentinel Hub / Copernicus research patterns actually implemented
    stackstac_used_t1 = (index_t1.method == "stackstac") if index_t1 else False
    stackstac_used_t2 = (index_t2.method == "stackstac") if index_t2 else False
    n_scenes_t1 = len(selection_t1.scenes) if selection_t1 and selection_t1.scenes else 0
    n_scenes_t2 = len(selection_t2.scenes) if selection_t2 and selection_t2.scenes else 0
    scene_ids_t1 = [s.item_id for s in selection_t1.scenes] if selection_t1 and selection_t1.scenes else []
    scene_ids_t2 = [s.item_id for s in selection_t2.scenes] if selection_t2 and selection_t2.scenes else []
    is_composite_t1 = stackstac_used_t1 and n_scenes_t1 > 1
    is_composite_t2 = stackstac_used_t2 and n_scenes_t2 > 1

    # Extract bands_used from index definitions
    bands_used = {}
    if _INDEX_BAND_MAP and _INDEX_DEFINITIONS:
        band_key = (sensor, index_name)
        bands_used = _INDEX_BAND_MAP.get(band_key, {})

    # Extract acquisition dates from scene selections
    acq_dates_t1 = [s.datetime for s in selection_t1.scenes] if selection_t1 and selection_t1.scenes else []
    acq_dates_t2 = [s.datetime for s in selection_t2.scenes] if selection_t2 and selection_t2.scenes else []
    cloud_covers_t1 = [s.cloud_cover for s in selection_t1.scenes] if selection_t1 and selection_t1.scenes else []
    cloud_covers_t2 = [s.cloud_cover for s in selection_t2.scenes] if selection_t2 and selection_t2.scenes else []

    # Extract platform/instrument from scene selections
    platform_t1 = selection_t1.platform if selection_t1 else "unknown"
    platform_t2 = selection_t2.platform if selection_t2 else "unknown"

    provenance = {
        "query": {
            "text": plan.get("query", ""),
            "phenomenon": phenomenon,
            "aoi_name": aoi_name,
            "aoi_bbox": bbox,
            "period_1": f"{start_date} to {end_date}",
            "period_2": f"{start_date} to {end_date}",
        },
        "dataset": {
            "provider": "planetary_computer",
            "collection": collection,
            "platform": {"period_1": platform_t1, "period_2": platform_t2},
            "instrument": "MSI" if "sentinel" in collection.lower() else "OLI",
            "processing_level": "L2A" if "l2a" in collection.lower() else "unknown",
        },
        "scenes": {
            "period_1": {
                "scene_ids": scene_ids_t1,
                "acquisition_dates": acq_dates_t1,
                "cloud_cover": cloud_covers_t1,
                "count": n_scenes_t1,
                "aoi_coverage": scene_sel_obj_t1.coverage_ratio if scene_sel_obj_t1 and hasattr(scene_sel_obj_t1, 'coverage_ratio') else None,
            },
            "period_2": {
                "scene_ids": scene_ids_t2,
                "acquisition_dates": acq_dates_t2,
                "cloud_cover": cloud_covers_t2,
                "count": n_scenes_t2,
                "aoi_coverage": scene_sel_obj_t2.coverage_ratio if scene_sel_obj_t2 and hasattr(scene_sel_obj_t2, 'coverage_ratio') else None,
            },
        },
        "quality_method": {
            "name": "SCL categorical masking" if scl_available else "nodata-only masking",
            "source": "Sentinel-2 Scene Classification Layer (Copernicus S2 L2A)",
            "resampling": "nearest-neighbor" if scl_available else "N/A",
            "cloud_classes": [3, 8, 9, 10] if scl_available else [],
            "shadow_class": 3 if scl_available else None,
            "valid_classes": [4, 5, 6, 7, 11] if scl_available else [],
            "scl_reprojected": scl_available,
            "implementation": "compositor.py:create_cloud_mask()",
        },
        "composite_method": {
            "name": "median" if is_composite_t1 else ("single mosaic" if n_scenes_t1 <= 1 else "single mosaic"),
            "description": (
                f"Median composite across {n_scenes_t1} observations (StackSTAC)" if is_composite_t1
                else f"{n_scenes_t1} scene(s), {'StackSTAC mosaic' if stackstac_used_t1 else 'manual rasterio mosaic'}, no temporal composite"
            ),
            "observation_count": {
                "period_1": n_scenes_t1,
                "period_2": n_scenes_t2,
            },
            "scene_ids": {
                "period_1": scene_ids_t1,
                "period_2": scene_ids_t2,
            },
            "date_range": {
                "period_1": f"{start_date} to {end_date}",
                "period_2": f"{start_date} to {end_date}",
            },
            "is_composite": {
                "period_1": is_composite_t1,
                "period_2": is_composite_t2,
            },
            "implementation": (
                "stackstac_adapter.py:stackstac_compute_index()" if stackstac_used_t1
                else "mosaic.py:mosaic_bands()"
            ),
        },
        "indices": {
            "primary": index_name,
            "supporting": [r.get("index_name") for r in signal_rules if r.get("index_name") != index_name] if signal_rules else [],
            "formulas": {name: _INDEX_DEFINITIONS[name].formula for name in [index_name] + [r.get("index_name") for r in signal_rules if r.get("index_name") != index_name] if _INDEX_DEFINITIONS and name in _INDEX_DEFINITIONS} if _INDEX_DEFINITIONS else {},
            "bands": bands_used,
            "sensor": sensor,
        },
        "detector": {
            "method": change_result_obj.algorithm if change_result_obj else "N/A",
            "thresholds": change_result_obj.parameters if change_result_obj else {},
            "min_region_pixels": change_result_obj.parameters.get("min_region_size", 0) if change_result_obj else 0,
            "min_region_area_m2": (change_result_obj.parameters.get("min_region_size", 0) * (analysis_resolution ** 2)) if change_result_obj and analysis_resolution else 0,
        },
        "grid": {
            "crs": analysis_crs if 'analysis_crs' in dir() else "N/A",
            "resolution_m": analysis_resolution if 'analysis_resolution' in dir() else 0,
            "shape": list(t1_aligned.shape) if t1_aligned is not None else [],
            "pixel_area_m2": (analysis_resolution ** 2) if 'analysis_resolution' in dir() else 0,
            "transform": str(analysis_transform) if 'analysis_transform' in dir() and analysis_transform else "N/A",
            "bounds": list(analysis_grid_info.get("bounds", [])) if analysis_grid_info else [],
            "reprojection_method": "bilinear" if 'analysis_crs' in dir() else "N/A",
        },
        "results": {
            "valid_pixel_count": change_result_obj.total_pixels if change_result_obj else 0,
            "changed_pixel_count": change_result_obj.changed_pixels if change_result_obj else 0,
            "changed_area_m2": change_result_obj.changed_area_sq_meters if change_result_obj else 0,
            "changed_area_ha": (change_result_obj.changed_area_sq_meters / 10000.0) if change_result_obj else 0,
            "changed_area_km2": (change_result_obj.changed_area_sq_meters / 1e6) if change_result_obj else 0,
            "changed_pct": change_result_obj.changed_pct if change_result_obj else 0,
            "region_count": change_result_obj.num_regions if change_result_obj else 0,
            "pixel_area_m2": (analysis_resolution ** 2) if 'analysis_resolution' in dir() else 0,
            "area_calculation": f"{change_result_obj.changed_pixels} pixels x {(analysis_resolution ** 2):.1f} m2/pixel = {change_result_obj.changed_area_sq_meters:.0f} m2" if change_result_obj and 'analysis_resolution' in dir() else "N/A",
        },
        "implementation_sources": {
            "quality_masking": "Sentinel Hub Custom Scripts: SCL-based cloudless mosaics pattern",
            "indices": "Sentinel Hub Custom Scripts: NDVI/NDBI formulas; Copernicus S2 L2A band definitions",
            "compositing": "StackSTAC: median temporal composite; Sentinel Hub: first-quartile cloud-free pattern",
            "change_detection": "OrbitalQuery: phenomenon-aware threshold + morphological refinement",
            "visualization": "OrbitalQuery: RGBA change mask overlay",
        },
    }

    # ── Build final result ────────────────────────────────────────
    return TemporalComparisonResult(
        status="ok",
        plan_id=plan_id,
        phenomenon=phenomenon,
        analysis_type=analysis_type,
        aoi_name=aoi_name,
        aoi_bbox=bbox,
        period1=period1,
        period2=period2,
        scene_t1=scene_sel_t1,
        scene_t2=scene_sel_t2,
        index_t1=index_t1,
        index_t2=index_t2,
        change_detection={**(change_result or {}), **({"ensemble": ensemble_stats} if ensemble_stats else {})},
        change_visualizations={
            "change_mask_png": change_mask_b64,
            "difference_png": diff_vis_b64,
            "bbox": bbox,
            **change_vis_stats,
            **({"ensemble": ensemble_stats} if ensemble_stats else {}),
        } if change_mask_b64 else None,
        metrics=metrics,
        imagery=imagery,
        processing_steps=processing_steps,
        sensor_info=sensor_info,
        provenance=provenance,
        explanation=explanation,
    )
