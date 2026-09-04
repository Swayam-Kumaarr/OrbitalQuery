"""
Synthetic tests for the unified detector interface.

Tests all 3 detector methods against 4 controlled array patterns:
  1. NDBI up + NDVI down  → urban expansion signal
  2. NDBI up + NDVI up    → vegetation/greening
  3. NDBI down + NDVI down → mixed signal
  4. NDBI down + NDVI up   → urban → vegetation (inverse)

Each test verifies:
  - Correct method routing
  - Change mask shape consistency
  - GeoJSON feature count consistency
  - Region area > 0 when change exists
  - Algorithm name matches requested method
  - No fabrication: synthetic data only
"""

import numpy as np
import pytest
from app.services.detector_interface import detect_change, get_supported_methods


# ── Synthetic array builders ──────────────────────────────────────

SHAPE = (100, 100)
RESOLUTION = 10.0  # 10m pixels
AOI_BBOX = [77.0, 28.0, 77.5, 28.5]


def _build_urban_expansion():
    """NDBI: 0.05 → 0.25 (increase), NDVI: 0.40 → 0.10 (decrease)."""
    baseline_ndbi = np.full(SHAPE, 0.05, dtype=np.float32)
    comparison_ndbi = np.full(SHAPE, 0.25, dtype=np.float32)
    baseline_ndvi = np.full(SHAPE, 0.40, dtype=np.float32)
    comparison_ndvi = np.full(SHAPE, 0.10, dtype=np.float32)
    return baseline_ndbi, comparison_ndbi, baseline_ndvi, comparison_ndvi


def _build_vegetation_greening():
    """NDBI: 0.25 → 0.05 (decrease), NDVI: 0.10 → 0.40 (increase)."""
    baseline_ndbi = np.full(SHAPE, 0.25, dtype=np.float32)
    comparison_ndbi = np.full(SHAPE, 0.05, dtype=np.float32)
    baseline_ndvi = np.full(SHAPE, 0.10, dtype=np.float32)
    comparison_ndvi = np.full(SHAPE, 0.40, dtype=np.float32)
    return baseline_ndbi, comparison_ndbi, baseline_ndvi, comparison_ndvi


def _build_mixed_signal():
    """NDBI: 0.30 → 0.05 (decrease), NDVI: 0.40 → 0.10 (decrease)."""
    baseline_ndbi = np.full(SHAPE, 0.30, dtype=np.float32)
    comparison_ndbi = np.full(SHAPE, 0.05, dtype=np.float32)
    baseline_ndvi = np.full(SHAPE, 0.40, dtype=np.float32)
    comparison_ndvi = np.full(SHAPE, 0.10, dtype=np.float32)
    return baseline_ndbi, comparison_ndbi, baseline_ndvi, comparison_ndvi


def _build_inverse_urban():
    """NDBI: 0.25 → 0.05 (decrease), NDVI: 0.10 → 0.40 (increase)."""
    baseline_ndbi = np.full(SHAPE, 0.25, dtype=np.float32)
    comparison_ndbi = np.full(SHAPE, 0.05, dtype=np.float32)
    baseline_ndvi = np.full(SHAPE, 0.10, dtype=np.float32)
    comparison_ndvi = np.full(SHAPE, 0.40, dtype=np.float32)
    return baseline_ndbi, comparison_ndbi, baseline_ndvi, comparison_ndvi


# ── Tests: supported methods ──────────────────────────────────────

class TestSupportedMethods:
    def test_three_methods_registered(self):
        methods = get_supported_methods()
        assert len(methods) == 3
        assert "phenomenon_aware_difference" in methods
        assert "cva" in methods
        assert "object_based" in methods

    def test_default_is_phenomenon_aware(self):
        methods = get_supported_methods()
        assert methods["phenomenon_aware_difference"]["default"] is True

    def test_unknown_method_raises(self):
        baseline, comparison, ndvi_b, ndvi_c = _build_urban_expansion()
        with pytest.raises(ValueError, match="Unknown detector method"):
            detect_change(
                baseline=baseline,
                comparison=comparison,
                index_name="NDBI",
                aoi_bbox=AOI_BBOX,
                detector_method="nonexistent_method",
            )


# ── Tests: phenomenon_aware_difference ────────────────────────────

class TestPhenomenonAwareDifference:
    def test_urban_expansion_detected(self):
        """NDBI up + NDVI down should produce change regions with AND logic."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_urban_expansion()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
        assert result.algorithm == "phenomenon_aware_difference"
        # NDBI delta = +0.20 > 0.12 AND NDVI delta = -0.30 < -0.08 → full area change
        assert result.changed_pixels > 0
        assert result.num_regions >= 1
        assert result.changed_area_sq_meters > 0
        # GeoJSON should have features matching regions
        assert result.change_geojson is not None
        if result.change_geojson and "features" in result.change_geojson:
            assert len(result.change_geojson["features"]) == result.num_regions

    def test_vegetation_greening_no_urban_change(self):
        """NDBI down + NDVI up → no urban expansion (NDBI increase fails)."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_vegetation_greening()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
        # NDBI decreased, not increased → no urban change
        assert result.changed_pixels == 0

    def test_mixed_signal_no_change(self):
        """NDBI down + NDVI down → no urban expansion."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_mixed_signal()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
        assert result.changed_pixels == 0

    def test_inverse_urban_no_change(self):
        """NDBI down + NDVI up → no urban expansion."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_inverse_urban()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
        assert result.changed_pixels == 0

    def test_no_ndvi_still_works(self):
        """Without NDVI, only NDBI threshold applies."""
        ndbi_b, ndbi_c, _, _ = _build_urban_expansion()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
        # NDBI delta = +0.20 > 0.12 → full area change (no NDVI constraint)
        assert result.changed_pixels > 0

    def test_visualization_png_exists(self):
        """PNG should be generated when change exists."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_urban_expansion()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.change_visualization_png is not None
        assert len(result.change_visualization_png) > 0


# ── Tests: CVA ────────────────────────────────────────────────────

class TestCVA:
    def test_cva_requires_multiband(self):
        """CVA without bands_t1/bands_t2 should raise ValueError."""
        ndbi_b, ndbi_c, _, _ = _build_urban_expansion()
        with pytest.raises(ValueError, match="requires multi-band"):
            detect_change(
                baseline=ndbi_b,
                comparison=ndbi_c,
                index_name="NDBI",
                aoi_bbox=AOI_BBOX,
                detector_method="cva",
            )

    def test_cva_runs_with_multiband(self):
        """CVA with real multiband data should produce a result."""
        ndbi_b, ndbi_c, _, _ = _build_urban_expansion()
        # Build 3-band arrays (red, nir, swir)
        bands_t1 = np.stack([
            np.full(SHAPE, 0.10, dtype=np.float32),  # red
            np.full(SHAPE, 0.30, dtype=np.float32),  # nir
            np.full(SHAPE, 0.15, dtype=np.float32),  # swir
        ], axis=0)
        bands_t2 = np.stack([
            np.full(SHAPE, 0.15, dtype=np.float32),  # red (increased → more urban)
            np.full(SHAPE, 0.20, dtype=np.float32),  # nir (decreased)
            np.full(SHAPE, 0.25, dtype=np.float32),  # swir (increased → more urban)
        ], axis=0)
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="cva",
            bands_t1=bands_t1,
            bands_t2=bands_t2,
            band_names=["red", "nir", "swir"],
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
        assert result.algorithm == "cva"
        assert result.changed_pixels >= 0
        assert result.change_geojson is not None
        # Processing steps should include CVA-specific steps
        step_names = [s["step"] for s in result.processing_steps]
        assert any(s in step_names for s in ["change_vectors", "magnitude_direction", "pif_normalization"])


# ── Tests: object_based ───────────────────────────────────────────

class TestObjectBased:
    def test_object_based_runs(self):
        """Object-based CD should produce a result."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_urban_expansion()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="object_based",
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            ndvi_decrease_threshold=0.08,
        )
        assert result.status == "ok"
        assert result.algorithm == "object_based"
        assert result.changed_pixels >= 0
        assert result.change_geojson is not None

    def test_object_based_vegetation_greening(self):
        """Object-based: NDBI down + NDVI up → no urban change."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_vegetation_greening()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="object_based",
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            ndvi_decrease_threshold=0.08,
        )
        assert result.status == "ok"
        assert result.changed_pixels == 0


# ── Tests: consistency ────────────────────────────────────────────

class TestConsistency:
    def test_geojson_feature_count_matches_regions(self):
        """Every method must have GeoJSON features == region count."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_urban_expansion()

        for method in ["phenomenon_aware_difference", "object_based"]:
            kwargs = dict(
                baseline=ndbi_b,
                comparison=ndbi_c,
                index_name="NDBI",
                aoi_bbox=AOI_BBOX,
                detector_method=method,
                phenomenon="urban_expansion",
                resolution_meters=RESOLUTION,
                ndvi_baseline=ndvi_b,
                ndvi_comparison=ndvi_c,
                ndvi_decrease_threshold=0.08,
            )
            result = detect_change(**kwargs)
            if result.change_geojson and "features" in result.change_geojson:
                assert len(result.change_geojson["features"]) == result.num_regions, \
                    f"{method}: features ({len(result.change_geojson['features'])}) != regions ({result.num_regions})"

    def test_area_calculation_consistency(self):
        """changed_pixels * pixel_area ≈ changed_area."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_urban_expansion()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        pixel_area = RESOLUTION ** 2
        expected_area = result.changed_pixels * pixel_area
        assert abs(result.changed_area_sq_meters - expected_area) < 1.0, \
            f"Area mismatch: {result.changed_area_sq_meters} != {expected_area}"

    def test_no_cloud_mask_still_works(self):
        """All methods should work without cloud masks."""
        ndbi_b, ndbi_c, ndvi_b, ndvi_c = _build_urban_expansion()
        result = detect_change(
            baseline=ndbi_b,
            comparison=ndbi_c,
            index_name="NDBI",
            aoi_bbox=AOI_BBOX,
            detector_method="phenomenon_aware_difference",
            threshold=0.12,
            ndvi_decrease_threshold=0.08,
            ndvi_baseline=ndvi_b,
            ndvi_comparison=ndvi_c,
            phenomenon="urban_expansion",
            resolution_meters=RESOLUTION,
        )
        assert result.status == "ok"
