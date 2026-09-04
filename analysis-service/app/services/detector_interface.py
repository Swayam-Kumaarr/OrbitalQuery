"""
Unified Change Detection Interface.

Routes to the appropriate detector method based on analysis plan configuration.

Supported methods:
  1. phenomenon_aware_difference — current production method (threshold + morphology)
  2. cva — Change Vector Analysis with PIF normalization
  3. object_based — multi-scale object-based change detection

The unified interface accepts the same primary inputs as run_change_detection(),
with optional multi-band arrays for CVA. It returns a ChangeDetectionResult.

All methods produce:
  - change_mask (H, W) boolean
  - morphological cleanup
  - connected components
  - region extraction
  - GeoJSON
  - visualization
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


# ── Supported methods ───────────────────────────────────────────────

SUPPORTED_METHODS = {
    "phenomenon_aware_difference": {
        "description": "Element-wise threshold difference with multi-signal support",
        "requires_multiband": False,
        "default": True,
    },
    "cva": {
        "description": "Change Vector Analysis with PIF normalization",
        "requires_multiband": True,
        "default": False,
    },
    "object_based": {
        "description": "Multi-scale object-based change detection",
        "requires_multiband": False,
        "default": False,
    },
}


def get_supported_methods() -> dict[str, dict[str, Any]]:
    """Return the list of supported detector methods and their capabilities."""
    return SUPPORTED_METHODS.copy()


def detect_change(
    # Primary inputs (required for all methods)
    baseline: np.ndarray,
    comparison: np.ndarray,
    index_name: str,
    aoi_bbox: list[float],
    # Method selection
    detector_method: str = "phenomenon_aware_difference",
    # Thresholds
    threshold: Optional[float] = None,
    min_region_size: Optional[int] = None,
    direction: Optional[str] = None,
    # Dates and metadata
    baseline_date: str = "unknown",
    comparison_date: str = "unknown",
    crs: str = "EPSG:4326",
    resolution_meters: float = 10.0,
    phenomenon: Optional[str] = None,
    # Masks
    nodata_baseline: Optional[np.ndarray] = None,
    nodata_comparison: Optional[np.ndarray] = None,
    cloud_baseline: Optional[np.ndarray] = None,
    cloud_comparison: Optional[np.ndarray] = None,
    # Transforms
    baseline_transform: Any = None,
    comparison_transform: Any = None,
    # Multi-signal support (primary method)
    ndvi_baseline: Optional[np.ndarray] = None,
    ndvi_comparison: Optional[np.ndarray] = None,
    ndbi_threshold: Optional[float] = None,
    ndvi_decrease_threshold: Optional[float] = None,
    # CVA-specific inputs
    bands_t1: Optional[np.ndarray] = None,
    bands_t2: Optional[np.ndarray] = None,
    band_names: Optional[list[str]] = None,
    # Object-based specific inputs
    scale_sigmas: Optional[list[float]] = None,
    min_agreement: int = 2,
    confidence_threshold: float = 0.4,
    # Full provenance passthrough
    scene_ids: Optional[list[str]] = None,
    collection: str = "",
    supporting_indices: Optional[list[str]] = None,
    indicator_formulas: Optional[dict[str, str]] = None,
    quality_mask: str = "",
    scl_usage: str = "",
    cloud_shadow_handling: str = "",
    composite_method: str = "",
    observation_count: Optional[dict[str, int]] = None,
    provider: str = "",
    platform: str = "",
    instrument: str = "",
    processing_level: str = "",
    acquisition_dates: Optional[list[str]] = None,
    cloud_cover: Optional[list[float]] = None,
    aoi_coverage: Optional[float] = None,
) -> Any:
    """
    Unified change detection entry point.

    Routes to the appropriate detector method and returns a ChangeDetectionResult
    (from change_detection.py) with the algorithm field set to the actual method used.

    Args:
        baseline: Primary index array (H, W) — e.g., NDBI_t1
        comparison: Primary index array (H, W) — e.g., NDBI_t2
        index_name: Name of the primary index (e.g., "NDBI")
        aoi_bbox: [west, south, east, north]
        detector_method: One of SUPPORTED_METHODS keys
        ... (other args passed through to the appropriate method)

    Returns:
        ChangeDetectionResult with algorithm set to detector_method
    """
    if detector_method not in SUPPORTED_METHODS:
        raise ValueError(
            f"Unknown detector method: '{detector_method}'. "
            f"Supported: {list(SUPPORTED_METHODS.keys())}"
        )

    method_info = SUPPORTED_METHODS[detector_method]

    if method_info["requires_multiband"]:
        if bands_t1 is None or bands_t2 is None:
            raise ValueError(
                f"Method '{detector_method}' requires multi-band inputs "
                f"(bands_t1, bands_t2). None provided."
            )

    logger.info(
        "[DETECTOR] Using method: %s (requires_multiband=%s)",
        detector_method, method_info["requires_multiband"],
    )

    # ── Route to the appropriate method ─────────────────────────
    if detector_method == "phenomenon_aware_difference":
        return _run_phenomenon_aware(
            baseline=baseline,
            comparison=comparison,
            index_name=index_name,
            aoi_bbox=aoi_bbox,
            threshold=threshold,
            min_region_size=min_region_size,
            direction=direction,
            baseline_date=baseline_date,
            comparison_date=comparison_date,
            crs=crs,
            resolution_meters=resolution_meters,
            phenomenon=phenomenon,
            nodata_baseline=nodata_baseline,
            nodata_comparison=nodata_comparison,
            cloud_baseline=cloud_baseline,
            cloud_comparison=cloud_comparison,
            baseline_transform=baseline_transform,
            comparison_transform=comparison_transform,
            ndvi_baseline=ndvi_baseline,
            ndvi_comparison=ndvi_comparison,
            ndbi_threshold=ndbi_threshold,
            ndvi_decrease_threshold=ndvi_decrease_threshold,
            scene_ids=scene_ids,
            collection=collection,
            supporting_indices=supporting_indices,
            indicator_formulas=indicator_formulas,
            quality_mask=quality_mask,
            scl_usage=scl_usage,
            cloud_shadow_handling=cloud_shadow_handling,
            composite_method=composite_method,
            observation_count=observation_count,
            provider=provider,
            platform=platform,
            instrument=instrument,
            processing_level=processing_level,
            acquisition_dates=acquisition_dates,
            cloud_cover=cloud_cover,
            aoi_coverage=aoi_coverage,
        )
    elif detector_method == "cva":
        return _run_cva_detector(
            baseline=baseline,
            comparison=comparison,
            index_name=index_name,
            aoi_bbox=aoi_bbox,
            threshold=threshold,
            min_region_size=min_region_size,
            baseline_date=baseline_date,
            comparison_date=comparison_date,
            crs=crs,
            resolution_meters=resolution_meters,
            phenomenon=phenomenon,
            bands_t1=bands_t1,
            bands_t2=bands_t2,
            band_names=band_names,
            ndvi_baseline=ndvi_baseline,
            ndvi_comparison=ndvi_comparison,
            baseline_transform=baseline_transform,
            comparison_transform=comparison_transform,
            nodata_baseline=nodata_baseline,
            nodata_comparison=nodata_comparison,
            cloud_baseline=cloud_baseline,
            cloud_comparison=cloud_comparison,
        )
    elif detector_method == "object_based":
        return _run_object_based_detector(
            baseline=baseline,
            comparison=comparison,
            index_name=index_name,
            aoi_bbox=aoi_bbox,
            threshold=threshold,
            min_region_size=min_region_size,
            direction=direction,
            baseline_date=baseline_date,
            comparison_date=comparison_date,
            crs=crs,
            resolution_meters=resolution_meters,
            phenomenon=phenomenon,
            scale_sigmas=scale_sigmas,
            min_agreement=min_agreement,
            confidence_threshold=confidence_threshold,
            baseline_transform=baseline_transform,
            comparison_transform=comparison_transform,
            nodata_baseline=nodata_baseline,
            nodata_comparison=nodata_comparison,
            cloud_baseline=cloud_baseline,
            cloud_comparison=cloud_comparison,
            ndvi_baseline=ndvi_baseline,
            ndvi_comparison=ndvi_comparison,
            ndvi_decrease_threshold=ndvi_decrease_threshold,
        )
    else:
        raise ValueError(f"Unhandled detector method: '{detector_method}'")


# ── Method implementations ──────────────────────────────────────────

def _run_phenomenon_aware(**kwargs) -> Any:
    """Delegate to the existing run_change_detection."""
    from app.services.change_detection import run_change_detection
    return run_change_detection(**kwargs)


def _run_cva_detector(
    baseline: np.ndarray,
    comparison: np.ndarray,
    index_name: str,
    aoi_bbox: list[float],
    threshold: Optional[float],
    min_region_size: Optional[int],
    baseline_date: str,
    comparison_date: str,
    crs: str,
    resolution_meters: float,
    phenomenon: Optional[str],
    bands_t1: np.ndarray,
    bands_t2: np.ndarray,
    band_names: Optional[list[str]],
    ndvi_baseline: Optional[np.ndarray],
    ndvi_comparison: Optional[np.ndarray],
    baseline_transform: Any,
    comparison_transform: Any,
    nodata_baseline: Optional[np.ndarray],
    nodata_comparison: Optional[np.ndarray],
    cloud_baseline: Optional[np.ndarray],
    cloud_comparison: Optional[np.ndarray],
) -> Any:
    """
    CVA-based change detection.

    1. Run CVA on multi-band data to get magnitude + change mask
    2. Apply morphological cleanup
    3. Connected components + region extraction
    4. Return as ChangeDetectionResult
    """
    from app.services.cva import run_cva
    from app.services.change_detection import (
        ChangeDetectionResult, morphological_cleanup, extract_regions,
        build_geojson, generate_change_visualization, compute_valid_mask,
        compute_array_stats, align_rasters,
    )

    processing_steps = []

    # Determine effective min_region_size
    from app.services.change_detection import PHENOMENON_CONFIG, DEFAULT_CONFIG
    config = PHENOMENON_CONFIG.get(phenomenon, DEFAULT_CONFIG) if phenomenon else DEFAULT_CONFIG
    effective_min_size = min_region_size if min_region_size is not None else config["min_region_pixels"]

    # Run CVA
    cva_result = run_cva(
        bands_t1=bands_t1,
        bands_t2=bands_t2,
        band_names=band_names or [],
        ndvi_t1=ndvi_baseline,
        ndvi_t2=ndvi_comparison,
        apply_normalization=True,
        threshold_method="otsu",
    )

    processing_steps.extend(cva_result.processing_steps)

    # Use CVA's change mask as the primary mask
    raw_change_mask = cva_result.change_mask

    # If user provided an explicit threshold, override CVA's adaptive threshold
    if threshold is not None:
        from app.services.change_detection import compute_difference, apply_threshold
        # Recompute difference using primary index and apply threshold
        diff = compute_difference(baseline, comparison)
        raw_change_mask = apply_threshold(diff, threshold, "absolute")
        processing_steps.append({
            "step": "cva_threshold_override",
            "detail": f"Using explicit threshold={threshold} instead of CVA Otsu",
        })

    # Build valid mask for statistics
    valid_mask = np.ones(baseline.shape, dtype=bool)
    if nodata_baseline is not None:
        valid_mask &= ~nodata_baseline
    if nodata_comparison is not None:
        valid_mask &= ~nodata_comparison
    if cloud_baseline is not None:
        valid_mask &= ~cloud_baseline
    if cloud_comparison is not None:
        valid_mask &= ~cloud_comparison

    # Compute difference for region stats
    diff = np.where(valid_mask, comparison.astype(np.float32) - baseline.astype(np.float32), np.nan)

    # Morphological cleanup
    cleaned_mask, num_regions = morphological_cleanup(raw_change_mask, effective_min_size)

    processing_steps.append({
        "step": "morphological_cleanup",
        "detail": f"min_region={effective_min_size}, raw={int(np.sum(raw_change_mask))}, cleaned={int(np.sum(cleaned_mask))}, regions={num_regions}",
    })

    # Connected components
    labeled, _ = ndimage.label(cleaned_mask)

    # Region extraction
    regions = extract_regions(labeled, diff, valid_mask, num_regions, resolution_meters, baseline_transform, crs)

    regions_dicts = []
    for r in regions:
        regions_dicts.append({
            "region_id": r.region_id,
            "area_pixels": r.area_pixels,
            "area_sq_meters": round(r.area_sq_meters, 2),
            "bbox": r.bbox,
            "centroid": [round(r.centroid[0], 2), round(r.centroid[1], 2)],
            "mean_delta": round(r.mean_delta, 4),
            "max_delta": round(r.max_delta, 4),
            "min_delta": round(r.min_delta, 4),
            "direction": r.direction,
        })

    largest_region = regions_dicts[0] if regions_dicts else None

    # GeoJSON
    change_geojson = build_geojson(regions, aoi_bbox, index_name, "cva")

    # Visualization
    visualization_png = generate_change_visualization(labeled, diff, valid_mask, num_regions)

    # Area calculations
    total_valid_pixels = int(np.sum(valid_mask))
    changed_pixels = int(np.sum(cleaned_mask))
    total_area = total_valid_pixels * (resolution_meters ** 2)
    changed_area = changed_pixels * (resolution_meters ** 2)
    changed_pct = (changed_pixels / total_valid_pixels * 100) if total_valid_pixels > 0 else 0.0

    # Stats
    baseline_stats = compute_array_stats(baseline, valid_mask)
    comparison_stats = compute_array_stats(comparison, valid_mask)
    difference_stats = compute_array_stats(diff, valid_mask)

    # Pixel counts
    pixel_counts = {
        "total_pixels": baseline.size,
        "valid_pixels": total_valid_pixels,
        "nodata_pixels": int(np.sum(~valid_mask)) - (int(np.sum(cloud_baseline)) if cloud_baseline is not None else 0) - (int(np.sum(cloud_comparison)) if cloud_comparison is not None else 0),
        "cloud_masked_pixels": (int(np.sum(cloud_baseline)) if cloud_baseline is not None else 0) + (int(np.sum(cloud_comparison)) if cloud_comparison is not None else 0),
    }
    valid_ratio = total_valid_pixels / max(baseline.size, 1)

    return ChangeDetectionResult(
        status="ok",
        algorithm="cva",
        parameters={
            "index_name": index_name,
            "threshold": threshold if threshold is not None else cva_result.threshold,
            "min_region_size": effective_min_size,
            "direction": "absolute",
            "phenomenon": phenomenon,
            "cva_magnitude_threshold": cva_result.threshold,
            "cva_normalized": cva_result.normalized,
            "multi_signal": False,
        },
        baseline_date=baseline_date,
        comparison_date=comparison_date,
        index_name=index_name,
        aoi_bbox=aoi_bbox,
        crs=crs,
        resolution_meters=resolution_meters,
        baseline_shape=list(baseline.shape),
        comparison_shape=list(comparison.shape),
        difference_shape=list(diff.shape) if diff is not None else list(baseline.shape),
        mask_shape=list(cleaned_mask.shape),
        total_pixels=total_valid_pixels,
        changed_pixels=changed_pixels,
        unchanged_pixels=total_valid_pixels - changed_pixels,
        changed_pct=round(changed_pct, 4),
        total_area_sq_meters=round(total_area, 2),
        changed_area_sq_meters=round(changed_area, 2),
        valid_area_sq_meters=round(total_area, 2),
        baseline_stats=baseline_stats,
        comparison_stats=comparison_stats,
        difference_stats=difference_stats,
        num_regions=num_regions,
        regions=regions_dicts,
        largest_region=largest_region,
        change_geojson=change_geojson,
        change_visualization_png=visualization_png,
        processing_steps=processing_steps,
        reproducibility={
            "algorithm": "cva",
            "inputs": {
                "baseline_date": baseline_date,
                "comparison_date": comparison_date,
                "index_name": index_name,
                "aoi_bbox": aoi_bbox,
                "phenomenon": phenomenon,
            },
            "parameters": {
                "threshold": threshold if threshold is not None else cva_result.threshold,
                "min_region_size": effective_min_size,
                "resolution_meters": resolution_meters,
            },
            "deterministic": True,
        },
        valid_pixel_ratio=round(valid_ratio, 4),
        nodata_pixels=pixel_counts["nodata_pixels"],
        cloud_masked_pixels=pixel_counts["cloud_masked_pixels"],
    )


def _run_object_based_detector(
    baseline: np.ndarray,
    comparison: np.ndarray,
    index_name: str,
    aoi_bbox: list[float],
    threshold: Optional[float],
    min_region_size: Optional[int],
    direction: Optional[str],
    baseline_date: str,
    comparison_date: str,
    crs: str,
    resolution_meters: float,
    phenomenon: Optional[str],
    scale_sigmas: Optional[list[float]],
    min_agreement: int,
    confidence_threshold: float,
    baseline_transform: Any,
    comparison_transform: Any,
    nodata_baseline: Optional[np.ndarray],
    nodata_comparison: Optional[np.ndarray],
    cloud_baseline: Optional[np.ndarray],
    cloud_comparison: Optional[np.ndarray],
    ndvi_baseline: Optional[np.ndarray],
    ndvi_comparison: Optional[np.ndarray],
    ndvi_decrease_threshold: Optional[float],
) -> Any:
    """
    Object-based multi-scale change detection.

    1. Compute delta from primary index
    2. Run object-based CD on the delta
    3. Apply morphological cleanup
    4. Connected components + region extraction
    5. Return as ChangeDetectionResult
    """
    from app.services.object_cd import run_object_cd
    from app.services.change_detection import (
        ChangeDetectionResult, morphological_cleanup, extract_regions,
        build_geojson, generate_change_visualization, compute_valid_mask,
        compute_array_stats, compute_difference, apply_threshold,
    )
    from app.services.change_detection import PHENOMENON_CONFIG, DEFAULT_CONFIG

    processing_steps = []

    config = PHENOMENON_CONFIG.get(phenomenon, DEFAULT_CONFIG) if phenomenon else DEFAULT_CONFIG
    effective_min_size = min_region_size if min_region_size is not None else config["min_region_pixels"]

    # Build valid mask
    valid_mask = np.ones(baseline.shape, dtype=bool)
    if nodata_baseline is not None:
        valid_mask &= ~nodata_baseline
    if nodata_comparison is not None:
        valid_mask &= ~nodata_comparison
    if cloud_baseline is not None:
        valid_mask &= ~cloud_baseline
    if cloud_comparison is not None:
        valid_mask &= ~cloud_comparison

    # Compute delta
    diff = compute_difference(baseline, comparison)
    diff[~valid_mask] = np.nan

    # Object-based detection uses the absolute delta
    delta_for_obj = np.nan_to_num(diff, nan=0.0)

    # Run object-based CD
    obj_result = run_object_cd(
        delta=delta_for_obj,
        scale_sigmas=scale_sigmas,
        min_agreement=min_agreement,
        min_object_size=effective_min_size,
        confidence_threshold=confidence_threshold,
    )

    processing_steps.extend(obj_result.processing_steps)

    # Use object-based mask as primary
    raw_change_mask = obj_result.change_mask

    # If user provided an explicit threshold, apply it on top
    if threshold is not None:
        direction = direction or config.get("direction", "absolute")
        raw_change_mask = apply_threshold(diff, threshold, direction)
        processing_steps.append({
            "step": "object_threshold_override",
            "detail": f"Using explicit threshold={threshold} dir={direction} instead of object-based mask",
        })

    # Multi-signal AND logic for urban expansion
    effective_ndvi_threshold = ndvi_decrease_threshold if ndvi_decrease_threshold is not None else config.get("ndvi_decrease_threshold", 0.10)
    if config.get("multi_signal") and ndvi_baseline is not None and ndvi_comparison is not None:
        ndvi_diff = ndvi_comparison - ndvi_baseline
        ndvi_decrease = valid_mask & (ndvi_diff < -effective_ndvi_threshold)
        raw_change_mask = raw_change_mask & ndvi_decrease
        processing_steps.append({
            "step": "multi_signal",
            "detail": f"AND logic with NDVI decrease >= {effective_ndvi_threshold}",
        })

    # Morphological cleanup
    cleaned_mask, num_regions = morphological_cleanup(raw_change_mask, effective_min_size)

    processing_steps.append({
        "step": "morphological_cleanup",
        "detail": f"min_region={effective_min_size}, raw={int(np.sum(raw_change_mask))}, cleaned={int(np.sum(cleaned_mask))}, regions={num_regions}",
    })

    # Connected components
    labeled, _ = ndimage.label(cleaned_mask)

    # Region extraction
    regions = extract_regions(labeled, diff, valid_mask, num_regions, resolution_meters, baseline_transform, crs)

    regions_dicts = []
    for r in regions:
        regions_dicts.append({
            "region_id": r.region_id,
            "area_pixels": r.area_pixels,
            "area_sq_meters": round(r.area_sq_meters, 2),
            "bbox": r.bbox,
            "centroid": [round(r.centroid[0], 2), round(r.centroid[1], 2)],
            "mean_delta": round(r.mean_delta, 4),
            "max_delta": round(r.max_delta, 4),
            "min_delta": round(r.min_delta, 4),
            "direction": r.direction,
        })

    largest_region = regions_dicts[0] if regions_dicts else None

    # GeoJSON
    change_geojson = build_geojson(regions, aoi_bbox, index_name, "object_based")

    # Visualization
    visualization_png = generate_change_visualization(labeled, diff, valid_mask, num_regions)

    # Area calculations
    total_valid_pixels = int(np.sum(valid_mask))
    changed_pixels = int(np.sum(cleaned_mask))
    total_area = total_valid_pixels * (resolution_meters ** 2)
    changed_area = changed_pixels * (resolution_meters ** 2)
    changed_pct = (changed_pixels / total_valid_pixels * 100) if total_valid_pixels > 0 else 0.0

    # Stats
    baseline_stats = compute_array_stats(baseline, valid_mask)
    comparison_stats = compute_array_stats(comparison, valid_mask)
    difference_stats = compute_array_stats(diff, valid_mask)

    pixel_counts = {
        "total_pixels": baseline.size,
        "valid_pixels": total_valid_pixels,
        "nodata_pixels": int(np.sum(~valid_mask)),
        "cloud_masked_pixels": (int(np.sum(cloud_baseline)) if cloud_baseline is not None else 0) + (int(np.sum(cloud_comparison)) if cloud_comparison is not None else 0),
    }
    valid_ratio = total_valid_pixels / max(baseline.size, 1)

    return ChangeDetectionResult(
        status="ok",
        algorithm="object_based",
        parameters={
            "index_name": index_name,
            "threshold": threshold,
            "min_region_size": effective_min_size,
            "direction": direction or "absolute",
            "phenomenon": phenomenon,
            "scale_sigmas": obj_result.scale_sigmas,
            "confidence_threshold": confidence_threshold,
            "min_agreement": min_agreement,
            "multi_signal": config.get("multi_signal", False),
            "ndvi_decrease_threshold": effective_ndvi_threshold,
        },
        baseline_date=baseline_date,
        comparison_date=comparison_date,
        index_name=index_name,
        aoi_bbox=aoi_bbox,
        crs=crs,
        resolution_meters=resolution_meters,
        baseline_shape=list(baseline.shape),
        comparison_shape=list(comparison.shape),
        difference_shape=list(diff.shape),
        mask_shape=list(cleaned_mask.shape),
        total_pixels=total_valid_pixels,
        changed_pixels=changed_pixels,
        unchanged_pixels=total_valid_pixels - changed_pixels,
        changed_pct=round(changed_pct, 4),
        total_area_sq_meters=round(total_area, 2),
        changed_area_sq_meters=round(changed_area, 2),
        valid_area_sq_meters=round(total_area, 2),
        baseline_stats=baseline_stats,
        comparison_stats=comparison_stats,
        difference_stats=difference_stats,
        num_regions=num_regions,
        regions=regions_dicts,
        largest_region=largest_region,
        change_geojson=change_geojson,
        change_visualization_png=visualization_png,
        processing_steps=processing_steps,
        reproducibility={
            "algorithm": "object_based",
            "inputs": {
                "baseline_date": baseline_date,
                "comparison_date": comparison_date,
                "index_name": index_name,
                "aoi_bbox": aoi_bbox,
                "phenomenon": phenomenon,
            },
            "parameters": {
                "threshold": threshold,
                "min_region_size": effective_min_size,
                "resolution_meters": resolution_meters,
                "scale_sigmas": obj_result.scale_sigmas,
                "confidence_threshold": confidence_threshold,
            },
            "deterministic": True,
        },
        valid_pixel_ratio=round(valid_ratio, 4),
        nodata_pixels=pixel_counts["nodata_pixels"],
        cloud_masked_pixels=pixel_counts["cloud_masked_pixels"],
    )
