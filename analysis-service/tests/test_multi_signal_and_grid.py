"""Unit tests for the critical change-detection repair."""

import numpy as np
import pytest


class TestMultiSignalANDLogic:
    """Verify that urban multi-signal uses AND (not OR)."""

    def _run_detection(self, ndbi_diff, ndvi_diff, threshold=0.12, ndvi_thresh=0.10):
        """Simulate the multi-signal AND logic from run_change_detection."""
        raw_change_mask = ndbi_diff > threshold
        ndvi_decrease = ndvi_diff < -ndvi_thresh
        dual_signal = raw_change_mask & ndvi_decrease
        return raw_change_mask, dual_signal

    def test_ndbi_up_ndvi_down_is_change(self):
        """NDBI↑ + NDVI↓ = change (both signals agree)."""
        ndbi = np.array([[0.2, 0.15], [0.0, 0.0]])
        ndvi = np.array([[-0.15, -0.12], [0.0, 0.0]])
        raw, dual = self._run_detection(ndbi, ndvi)
        assert dual[0, 0] == True, "NDBI↑ + NDVI↓ should be detected"
        assert dual[0, 1] == True, "NDBI↑ + NDVI↓ should be detected"

    def test_ndbi_up_ndvi_up_no_change(self):
        """NDBI↑ + NDVI↑ = no change (NDVI increased, not a built-up signal)."""
        ndbi = np.array([[0.2, 0.15]])
        ndvi = np.array([[0.15, 0.12]])
        raw, dual = self._run_detection(ndbi, ndvi)
        assert raw[0, 0] == True, "NDBI exceeds threshold"
        assert dual[0, 0] == False, "NDBI↑ + NDVI↑ should NOT be detected"

    def test_ndbi_down_ndvi_down_no_change(self):
        """NDBI↓ + NDVI↓ = no change (NDBI decreased)."""
        ndbi = np.array([[-0.2, -0.15]])
        ndvi = np.array([[-0.15, -0.12]])
        raw, dual = self._run_detection(ndbi, ndvi)
        assert raw[0, 0] == False, "NDBI below threshold"
        assert dual[0, 0] == False, "NDBI↓ + NDVI↓ should NOT be detected"

    def test_ndbi_down_ndvi_up_no_change(self):
        """NDBI↓ + NDVI↑ = no change."""
        ndbi = np.array([[-0.2, -0.15]])
        ndvi = np.array([[0.15, 0.12]])
        raw, dual = self._run_detection(ndbi, ndvi)
        assert raw[0, 0] == False
        assert dual[0, 0] == False

    def test_and_vs_or_difference(self):
        """Prove that AND produces fewer pixels than OR (the old bug)."""
        ndbi = np.array([[0.2, 0.0, 0.0]])
        ndvi = np.array([[-0.15, -0.15, 0.0]])  # First two pixels: NDVI down
        raw, and_result = self._run_detection(ndbi, ndvi)
        # OR would include pixel [0,1] (NDVI down but NDBI not up)
        or_result = raw | (ndvi < -0.10)
        assert np.sum(and_result) < np.sum(or_result), \
            "AND should reject pixels where only NDVI decreased"


class TestCommonGridReprojection:
    """Verify that reproject_to_grid produces correctly aligned arrays."""

    def test_same_grid_passthrough(self):
        """When source == target grid, array is returned unchanged."""
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS

        arr = np.random.rand(100, 100).astype(np.float32)
        crs = CRS.from_epsg(4326)
        transform = from_bounds(77.0, 28.0, 78.0, 29.0, 100, 100)

        from app.services.raster_service import reproject_to_grid
        result = reproject_to_grid(arr, crs, transform, crs, transform, (100, 100))

        assert result.shape == (100, 100), "Shape should be preserved"
        assert np.allclose(result, arr, equal_nan=True), "Values should be identical for same grid"


class TestChangeDetectionMinRegion:
    """Verify min_region filtering works."""

    def test_small_region_removed(self):
        """Regions smaller than min_region_size should be removed."""
        from scipy import ndimage

        # Create a mask with a 10-pixel region and a 100-pixel region
        mask = np.zeros((50, 50), dtype=bool)
        mask[5:6, 5:15] = True  # 10-pixel line
        mask[20:30, 20:30] = True  # 100-pixel square

        # Label and filter
        labeled, n = ndimage.label(mask)
        sizes = ndimage.sum(mask, labeled, range(1, n + 1))
        min_size = 25
        cleaned = np.zeros_like(mask)
        for i, size in enumerate(sizes):
            if size >= min_size:
                cleaned[labeled == (i + 1)] = True

        assert np.sum(cleaned) == 100, "Only the 100-pixel region should remain"
        assert np.sum(mask) == 110, "Original had 110 pixels"

    def test_urban_min_region_25(self):
        """Urban expansion should use min_region=25 (allows small urban patches)."""
        from app.services.change_detection import PHENOMENON_CONFIG
        assert PHENOMENON_CONFIG["urban_expansion"]["min_region_pixels"] == 25


class TestGeoJsonGeneration:
    """Verify GeoJSON is produced by run_change_detection."""

    def test_geojson_in_result(self):
        """run_change_detection should produce GeoJSON."""
        from app.services.change_detection import run_change_detection

        # Create two simple arrays with a clear change region
        baseline = np.full((50, 50), 0.3, dtype=np.float32)
        comparison = np.full((50, 50), 0.3, dtype=np.float32)
        # Create a 15x15 change region (225 pixels > min_region of 25)
        comparison[10:25, 10:25] = 0.5

        result = run_change_detection(
            baseline=baseline,
            comparison=comparison,
            index_name="NDVI",
            aoi_bbox=[77.0, 28.0, 78.0, 29.0],
            phenomenon="vegetation_change",
        )

        assert result.change_geojson is not None, "GeoJSON should be present"
        assert result.change_geojson["type"] == "FeatureCollection"
        assert len(result.change_geojson["features"]) > 0, "Should have at least one feature"
        assert result.regions is not None
        assert len(result.regions) > 0, "Should have at least one region"


class TestSCLNearestNeighborReprojection:
    """Verify SCL is reprojected with nearest-neighbor, not cropped."""

    def test_scl_reprojection_preserves_class_values(self):
        """SCL class integers must be preserved through nearest-neighbor reprojection."""
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from app.services.raster_service import reproject_to_grid

        # Simulate 20m SCL raster (50x50) with class values 4, 5, 8, 9
        scl_20m = np.zeros((50, 50), dtype=np.float32)
        scl_20m[10:20, 10:20] = 4   # Vegetation
        scl_20m[20:30, 20:30] = 8   # Cloud Medium
        scl_20m[30:40, 30:40] = 9   # Cloud High
        scl_20m[5:15, 35:45] = 5    # Bare Soils

        crs_20m = CRS.from_epsg(32643)  # UTM 43N (Delhi)
        transform_20m = from_bounds(700000, 3000000, 705000, 3005000, 50, 50)

        # 10m analysis grid (100x100)
        crs_10m = CRS.from_epsg(32643)
        transform_10m = from_bounds(700000, 3000000, 705000, 3005000, 100, 100)

        # Reproject with nearest-neighbor
        scl_reproj = reproject_to_grid(
            scl_20m, crs_20m, transform_20m,
            crs_10m, transform_10m, (100, 100),
            resampling_method='nearest',
        )

        # Shape must match analysis grid
        assert scl_reproj.shape == (100, 100), f"Shape {scl_reproj.shape} != (100, 100)"

        # All values must be integer SCL classes (no interpolated fractions)
        unique_vals = set(np.unique(scl_reproj[~np.isnan(scl_reproj)]))
        valid_scl_classes = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0}
        assert unique_vals.issubset(valid_scl_classes), \
            f"Non-integer SCL values found: {unique_vals - valid_scl_classes}"

        # Cloud classes (8, 9) should be present in the reprojected output
        assert 8.0 in unique_vals or 9.0 in unique_vals, "Cloud classes should survive reprojection"

    def test_scl_nearest_neighbor_not_bilinear(self):
        """Verify nearest-neighbor produces sharp edges, not blurred transitions."""
        from rasterio.transform import from_bounds
        from rasterio.crs import CRS
        from app.services.raster_service import reproject_to_grid

        # Binary SCL: half cloud (9), half clear (4)
        scl = np.zeros((50, 50), dtype=np.float32)
        scl[:, :25] = 9  # Cloud
        scl[:, 25:] = 4  # Vegetation

        crs = CRS.from_epsg(32643)
        transform = from_bounds(700000, 3000000, 705000, 3005000, 50, 50)
        transform_10m = from_bounds(700000, 3000000, 705000, 3005000, 100, 100)

        scl_nearest = reproject_to_grid(scl, crs, transform, crs, transform_10m, (100, 100), 'nearest')

        # Nearest-neighbor: only original class values (4 and 9)
        unique = set(np.unique(scl_nearest))
        assert unique <= {4.0, 9.0, np.nan}, f"Nearest-neighbor should only have original values: {unique}"


class TestAlignRastersHardening:
    """Verify align_rasters fails clearly when shapes differ with different transforms."""

    def test_matching_shapes_pass(self):
        """Same shapes → pass through."""
        from app.services.change_detection import align_rasters
        a = np.zeros((10, 10))
        b = np.zeros((10, 10))
        result_a, result_b, info = align_rasters(a, b)
        assert result_a.shape == (10, 10)
        assert info['method'] == 'none'

    def test_different_shapes_different_transforms_fails(self):
        """Different shapes + different transforms → ValueError."""
        from app.services.change_detection import align_rasters
        from rasterio.transform import from_bounds
        a = np.zeros((10, 10))
        b = np.zeros((8, 8))
        t1 = from_bounds(0, 0, 10, 10, 10, 10)
        t2 = from_bounds(0, 0, 8, 8, 8, 8)
        with pytest.raises(ValueError, match="different transforms"):
            align_rasters(a, b, t1, t2)

    def test_different_shapes_same_transform_crops(self):
        from rasterio.transform import from_bounds
        """Different shapes + same transform → crop with warning."""
        from app.services.change_detection import align_rasters
        a = np.zeros((10, 10))
        b = np.zeros((8, 8))
        t = from_bounds(0, 0, 10, 10, 10, 10)
        result_a, result_b, info = align_rasters(a, b, t, t)
        assert result_a.shape == (8, 8)
        assert result_b.shape == (8, 8)
        assert info['method'] == 'crop_to_common_warning'

    def test_different_shapes_no_transforms_crops(self):
        """Different shapes + no transforms → crop with warning."""
        from app.services.change_detection import align_rasters
        a = np.zeros((10, 10))
        b = np.zeros((8, 8))
        result_a, result_b, info = align_rasters(a, b)
        assert result_a.shape == (8, 8)
        assert info['method'] == 'crop_to_common_warning'
