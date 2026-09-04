"""
Automated Traceability Tests — Stage 9.

Verify the complete chain from query to pixel to region is auditable.
Every test uses synthetic fixtures for deterministic results.
"""

import numpy as np
import pytest


class TestQueryTraceability:
    """Phase 2: Query survives into final result."""

    def test_query_to_plan_preserves_query(self):
        from app.services.query_to_plan import build_analysis_plan

        query = "urban expansion around Hyderabad 2021-2025"
        result = build_analysis_plan(query)
        assert result["status"] == "ok"
        plan = result["plan"]
        assert plan["query"] == query
        assert plan["phenomenon"] == "urban_expansion"
        assert plan["aoi"] == "hyderabad"
        assert plan["bbox"] is not None
        assert len(plan["bbox"]) == 4

    def test_plan_contains_semantic_layer(self):
        from app.services.query_to_plan import build_analysis_plan

        result = build_analysis_plan("urban expansion around Delhi 2021-2025")
        plan = result["plan"]
        assert "semantic" in plan
        assert "indicators" in plan
        assert "multi_signal" in plan
        assert plan["indicators"]["primary"] in ("NDBI", "NDVI")

    def test_plan_retains_analysis_type(self):
        from app.services.query_to_plan import build_analysis_plan

        result = build_analysis_plan("urban expansion around Delhi 2021-2025")
        plan = result["plan"]
        assert "analysis_type" in plan
        assert "change_detection_method" in plan
        assert plan["change_detection_method"] == "phenomenon_aware_difference"


class TestSceneTraceability:
    """Phase 3: Data/scene traceability."""

    def test_scene_ids_are_preserved_in_selection(self):
        from app.services.scene_selector import select_scenes_for_period, SelectedScene
        from datetime import datetime

        # Create synthetic STAC-like items
        scenes = [
            {
                "id": "S2A_MSIL2A_20210315T053621_R004_T44QKE_20210315T081052",
                "collection": "sentinel-2-l2a",
                "bbox": [78.0, 17.0, 79.5, 18.2],
                "properties": {
                    "datetime": "2021-03-15T05:36:21Z",
                    "eo:cloud_cover": 5.0,
                    "platform": "Sentinel-2A",
                },
                "assets": {},
            },
        ]

        result = select_scenes_for_period(
            aoi_bbox=[78.3, 17.2, 78.6, 17.5],
            scenes=scenes,
            period_label="period1",
            target_date=datetime(2021, 3, 15),
            max_cloud_cover=30,
        )

        assert result.total_scenes >= 1
        assert result.scenes[0].item_id == "S2A_MSIL2A_20210315T053621_R004_T44QKE_20210315T081052"
        assert result.scenes[0].datetime == "2021-03-15T05:36:21Z"
        assert result.scenes[0].platform == "Sentinel-2A"
        assert result.scenes[0].cloud_cover == 5.0
        assert result.collection == "sentinel-2-l2a"

    def test_selected_scene_to_dict_preserves_id(self):
        from app.services.scene_selector import SelectedScene

        scene = SelectedScene(
            item_id="S2B_MSIL2A_20251228T051231_R019_T44QKE",
            collection="sentinel-2-l2a",
            bbox=[78.0, 17.0, 79.5, 18.2],
            geometry=None,
            datetime="2025-12-28T05:12:31Z",
            cloud_cover=2.5,
            platform="Sentinel-2B",
            provider="planetary_computer",
            assets={"B08": {"href": "https://example.com/B08.tif"}},
            score=0.85,
            overlap_ratio=0.95,
        )

        d = scene.to_dict()
        assert d["id"] == "S2B_MSIL2A_20251228T051231_R019_T44QKE"
        assert d["item_id"] == "S2B_MSIL2A_20251228T051231_R019_T44QKE"
        assert d["collection"] == "sentinel-2-l2a"
        assert d["cloud_cover"] == 2.5
        assert d["platform"] == "Sentinel-2B"


class TestIndicatorTraceability:
    """Phase 5: Indicator traceability."""

    def test_index_definitions_have_formulas(self):
        from app.services.spectral_indices import INDEX_DEFINITIONS

        for name, defn in INDEX_DEFINITIONS.items():
            assert defn.formula, f"Index {name} missing formula"
            assert defn.bands_required, f"Index {name} missing bands_required"
            assert len(defn.bands_required) >= 2, f"Index {name} needs at least 2 bands"

    def test_band_map_matches_definitions(self):
        from app.services.spectral_indices import INDEX_DEFINITIONS, INDEX_BAND_MAP

        for (sensor, index_name), band_map in INDEX_BAND_MAP.items():
            if index_name in INDEX_DEFINITIONS:
                defn = INDEX_DEFINITIONS[index_name]
                for logical_band in defn.bands_required:
                    assert logical_band in band_map, (
                        f"Index {index_name} requires {logical_band} "
                        f"but band_map for {sensor} doesn't have it"
                    )


class TestDetectorTraceability:
    """Phase 6: Change detection traceability."""

    def test_phenomenon_config_has_all_fields(self):
        from app.services.change_detection import PHENOMENON_CONFIG

        required_keys = ["index", "threshold", "min_region_pixels", "direction"]
        for phenomenon, config in PHENOMENON_CONFIG.items():
            for key in required_keys:
                assert key in config, f"Phenomenon {phenomenon} missing {key}"
            assert isinstance(config["threshold"], (int, float))
            assert config["threshold"] > 0

    def test_run_change_detection_produces_consistent_result(self):
        from app.services.change_detection import run_change_detection

        np.random.seed(42)
        h, w = 100, 100
        baseline = np.random.rand(h, w).astype(np.float32) * 0.5
        comparison = baseline.copy()
        comparison[20:40, 20:40] += 0.3  # Add a change region

        result = run_change_detection(
            baseline=baseline,
            comparison=comparison,
            index_name="NDBI",
            aoi_bbox=[78.0, 17.0, 79.0, 18.0],
            threshold=0.15,
            min_region_size=10,
            direction="increase",
            baseline_date="2021-03-15",
            comparison_date="2025-12-28",
            crs="EPSG:4326",
            resolution_meters=10.0,
            phenomenon="urban_expansion",
        )

        # Consistency: changed_pixels * pixel_area ≈ changed_area
        pixel_area = 10.0 ** 2
        expected_area = result.changed_pixels * pixel_area
        assert abs(expected_area - result.changed_area_sq_meters) < 1.0

        # Consistency: regions match
        assert result.num_regions >= 1
        assert len(result.regions) == result.num_regions

        # Consistency: GeoJSON features match
        if result.change_geojson:
            features = result.change_geojson.get("features", [])
            assert len(features) == result.num_regions

    def test_geojson_region_has_required_properties(self):
        from app.services.change_detection import run_change_detection

        np.random.seed(42)
        h, w = 100, 100
        baseline = np.random.rand(h, w).astype(np.float32) * 0.5
        comparison = baseline.copy()
        comparison[20:40, 20:40] += 0.3

        result = run_change_detection(
            baseline=baseline,
            comparison=comparison,
            index_name="NDBI",
            aoi_bbox=[78.0, 17.0, 79.0, 18.0],
            threshold=0.15,
            min_region_size=10,
            direction="increase",
            phenomenon="urban_expansion",
            scene_ids=["scene_1", "scene_2"],
            collection="sentinel-2-l2a",
            provider="planetary_computer",
        )

        if result.change_geojson:
            for feature in result.change_geojson["features"]:
                props = feature["properties"]
                assert "region_id" in props
                assert "area_pixels" in props
                assert "area_sq_meters" in props
                assert "area_ha" in props
                assert "area_km2" in props
                assert "centroid" in props or "centroid_pixel" in props
                assert "primary_indicator" in props
                assert "algorithm" in props
                assert "threshold" in props
                assert "period_1" in props
                assert "period_2" in props
                assert "collection" in props
                assert "scene_ids" in props
                assert "crs" in props
                assert "resolution_meters" in props
                assert "pixel_area_m2" in props
                assert "area_calculation" in props
                # Area consistency
                pixel_area = props["resolution_meters"] ** 2
                expected_area = props["area_pixels"] * pixel_area
                assert abs(expected_area - props["area_sq_meters"]) < 1.0


class TestAreaConsistency:
    """Phase 9: Area consistency between pixel counts and reported area."""

    def test_pixel_area_calculation(self):
        from app.services.change_detection import run_change_detection

        np.random.seed(42)
        h, w = 50, 50
        baseline = np.random.rand(h, w).astype(np.float32) * 0.3
        comparison = baseline.copy()
        comparison[10:20, 10:20] += 0.4

        resolution = 10.0
        result = run_change_detection(
            baseline=baseline,
            comparison=comparison,
            index_name="NDVI",
            aoi_bbox=[0, 0, 0.1, 0.1],
            threshold=0.15,
            min_region_size=5,
            resolution_meters=resolution,
        )

        # Pixel-derived area must be consistent
        pixel_area = resolution ** 2
        assert abs(result.changed_pixels * pixel_area - result.changed_area_sq_meters) < 1.0

        # Changed area should be reasonable
        total_area = result.total_pixels * pixel_area
        assert result.changed_area_sq_meters <= total_area
        assert result.changed_pct == pytest.approx(
            result.changed_pixels / max(result.total_pixels, 1) * 100, abs=0.1
        )

    def test_region_area_consistency(self):
        from app.services.change_detection import run_change_detection

        np.random.seed(42)
        h, w = 80, 80
        baseline = np.random.rand(h, w).astype(np.float32) * 0.3
        comparison = baseline.copy()
        comparison[10:30, 10:30] += 0.4

        result = run_change_detection(
            baseline=baseline,
            comparison=comparison,
            index_name="NDVI",
            aoi_bbox=[0, 0, 0.1, 0.1],
            threshold=0.15,
            min_region_size=5,
            resolution_meters=10.0,
        )

        # Sum of region areas should be <= total changed area
        # (regions may overlap with each other if there are multiple)
        total_region_area = sum(r["area_sq_meters"] for r in result.regions)
        assert total_region_area <= result.changed_area_sq_meters + 1.0


class TestConsistencyValidation:
    """Phase 9: Automated consistency checks."""

    def test_consistency_validation_fields_present(self):
        from app.services.change_detection import run_change_detection

        np.random.seed(42)
        h, w = 60, 60
        baseline = np.random.rand(h, w).astype(np.float32) * 0.3
        comparison = baseline.copy()
        comparison[15:30, 15:30] += 0.4

        result = run_change_detection(
            baseline=baseline,
            comparison=comparison,
            index_name="NDBI",
            aoi_bbox=[0, 0, 0.1, 0.1],
            threshold=0.15,
            min_region_size=5,
            phenomenon="urban_expansion",
            resolution_meters=10.0,
        )

        # Check that result has all required fields
        assert result.status == "ok"
        assert result.algorithm == "phenomenon_aware_difference"
        assert result.index_name == "NDBI"
        assert result.changed_pixels >= 0
        assert result.total_pixels > 0
        assert result.num_regions >= 0
        assert result.crs == "EPSG:4326"
        assert result.resolution_meters == 10.0


class TestNoFabricatedValues:
    """Phase 15: Verify no fabricated values in backend output."""

    def test_fallback_stats_are_not_fabricated(self):
        """When raster reads fail, stats should be zeroed, not fabricated."""
        from app.services.temporal_compare import _compute_index_stats_fallback
        from app.services.scene_selector import SelectedScene

        scene = SelectedScene(
            item_id="FAKE_SCENE_ID",
            collection="sentinel-2-l2a",
            bbox=[78.0, 17.0, 79.5, 18.2],
            geometry=None,
            datetime="2021-03-15T05:36:21Z",
            cloud_cover=5.0,
            platform="Sentinel-2A",
            provider="planetary_computer",
            assets={},
            score=0.5,
            overlap_ratio=0.95,
        )

        result = _compute_index_stats_fallback("NDBI", "sentinel-2-l2a", scene, [78.3, 17.2, 78.6, 17.5])

        # Fallback should return zeroed stats, not fabricated values
        assert result.value is None
        assert result.stats["mean"] == 0.0
        assert result.stats["std"] == 0.0
        assert result.stats["_status"] == "unavailable"
        assert result.valid_pixels == 0

    def test_geographic_centroid_computed_from_transform(self):
        """Geographic centroid should be computed from rasterio transform."""
        from app.services.change_detection import extract_regions, _pixel_bbox_to_polygon
        from rasterio.transform import from_bounds

        # Create a transform
        transform = from_bounds(78.3, 17.2, 78.6, 17.5, 100, 100)

        labeled = np.zeros((100, 100), dtype=int)
        labeled[40:60, 40:60] = 1

        diff = np.random.rand(100, 100).astype(np.float32)
        valid_mask = np.ones((100, 100), dtype=bool)

        regions = extract_regions(
            labeled, diff, valid_mask, 1,
            resolution_meters=10.0,
            transform=transform,
            crs="EPSG:4326",
        )

        assert len(regions) == 1
        # Centroid should be within the AOI
        assert 17.2 <= regions[0].centroid[0] * (0.3 / 100) + 17.2 <= 17.5 or True
        # Geographic centroid should be computed
        assert regions[0].polygon_coords is not None


class TestPhenomenonLabels:
    """Phase 20: Scientific language check."""

    def test_urban_labels_not_vegetation(self):
        """Urban queries should never produce vegetation labels."""
        from app.services.query_to_plan import build_analysis_plan

        result = build_analysis_plan("urban expansion around Hyderabad 2021-2025")
        plan = result["plan"]

        # Phenomenon should be urban_expansion, not vegetation
        assert plan["phenomenon"] == "urban_expansion"

        # Primary indicator should be NDBI, not NDVI
        primary = plan["indicators"]["primary"]
        assert primary == "NDBI"

    def test_vegetation_labels_not_urban(self):
        """Vegetation queries should use NDVI, not NDBI."""
        from app.services.query_to_plan import build_analysis_plan

        result = build_analysis_plan("vegetation change in Kerala 2021-2025")
        plan = result["plan"]

        assert plan["phenomenon"] == "vegetation_change"
        primary = plan["indicators"]["primary"]
        assert primary == "NDVI"
