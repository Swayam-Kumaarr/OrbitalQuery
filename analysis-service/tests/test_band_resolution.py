"""
Regression tests for Sentinel-2 / Landsat Band and Asset Resolution.

Proves:
- B04 -> correct STAC asset
- B08 -> correct STAC asset
- B11 -> correct STAC asset
- SCL -> correct STAC asset
- Case-insensitivity and eo:bands metadata resolution work
- Missing/invalid bands raise KeyError (never silently defaulting to another band)
- Single-band rasters (src.count == 1) return band 1
- Multi-band rasters resolve correctly or raise ValueError (never silently defaulting to band 1)
"""

import pytest
from app.services.stac_service import resolve_band_asset
from app.services.mosaic import _resolve_band_index


@pytest.fixture
def sentinel2_sample_assets():
    """Real-world Planetary Computer Sentinel-2 L2A STAC asset dictionary."""
    return {
        "AOT": {"href": "https://example.com/AOT.tif", "type": "image/tiff", "eo:bands": []},
        "B01": {"href": "https://example.com/B01.tif", "type": "image/tiff", "eo:bands": [{"name": "B01", "common_name": "coastal"}]},
        "B02": {"href": "https://example.com/B02.tif", "type": "image/tiff", "eo:bands": [{"name": "B02", "common_name": "blue"}]},
        "B03": {"href": "https://example.com/B03.tif", "type": "image/tiff", "eo:bands": [{"name": "B03", "common_name": "green"}]},
        "B04": {"href": "https://example.com/B04.tif", "type": "image/tiff", "eo:bands": [{"name": "B04", "common_name": "red"}]},
        "B08": {"href": "https://example.com/B08.tif", "type": "image/tiff", "eo:bands": [{"name": "B08", "common_name": "nir"}]},
        "B11": {"href": "https://example.com/B11.tif", "type": "image/tiff", "eo:bands": [{"name": "B11", "common_name": "swir16"}]},
        "B12": {"href": "https://example.com/B12.tif", "type": "image/tiff", "eo:bands": [{"name": "B12", "common_name": "swir22"}]},
        "SCL": {"href": "https://example.com/SCL.tif", "type": "image/tiff", "eo:bands": []},
        "visual": {"href": "https://example.com/visual.tif", "type": "image/tiff"},
    }


class TestSTACBandResolution:
    """Test suite for resolve_band_asset."""

    def test_resolve_exact_bands(self, sentinel2_sample_assets):
        """Exact band keys B04, B08, B11, SCL resolve to their corresponding assets."""
        key_b04, asset_b04 = resolve_band_asset(sentinel2_sample_assets, "B04", "sentinel-2-l2a")
        assert key_b04 == "B04"
        assert asset_b04["href"].endswith("B04.tif")

        key_b08, asset_b08 = resolve_band_asset(sentinel2_sample_assets, "B08", "sentinel-2-l2a")
        assert key_b08 == "B08"
        assert asset_b08["href"].endswith("B08.tif")

        key_b11, asset_b11 = resolve_band_asset(sentinel2_sample_assets, "B11", "sentinel-2-l2a")
        assert key_b11 == "B11"
        assert asset_b11["href"].endswith("B11.tif")

        key_scl, asset_scl = resolve_band_asset(sentinel2_sample_assets, "SCL", "sentinel-2-l2a")
        assert key_scl == "SCL"
        assert asset_scl["href"].endswith("SCL.tif")

    def test_resolve_case_insensitive(self, sentinel2_sample_assets):
        """Case variations like 'b04', 'b08', 'scl' resolve properly."""
        key, asset = resolve_band_asset(sentinel2_sample_assets, "b04", "sentinel-2-l2a")
        assert key == "B04"

        key, asset = resolve_band_asset(sentinel2_sample_assets, "scl", "sentinel-2-l2a")
        assert key == "SCL"

    def test_resolve_by_common_name_and_alias(self, sentinel2_sample_assets):
        """Logical names like 'RED', 'NIR', 'SWIR', 'red', 'nir' map to correct physical assets."""
        key, _ = resolve_band_asset(sentinel2_sample_assets, "RED", "sentinel-2-l2a")
        assert key == "B04"

        key, _ = resolve_band_asset(sentinel2_sample_assets, "NIR", "sentinel-2-l2a")
        assert key == "B08"

        key, _ = resolve_band_asset(sentinel2_sample_assets, "SWIR", "sentinel-2-l2a")
        assert key == "B11"

        key, _ = resolve_band_asset(sentinel2_sample_assets, "swir16", "sentinel-2-l2a")
        assert key == "B11"

    def test_unresolvable_band_raises_keyerror(self, sentinel2_sample_assets):
        """Missing or unresolvable bands must raise KeyError with diagnostic details (never default)."""
        with pytest.raises(KeyError) as exc_info:
            resolve_band_asset(sentinel2_sample_assets, "NON_EXISTENT_BAND", "sentinel-2-l2a")

        err_msg = str(exc_info.value)
        assert "NON_EXISTENT_BAND" in err_msg
        assert "sentinel-2-l2a" in err_msg
        assert "Available asset keys" in err_msg


class MockRasterSource:
    """Mock for rasterio DatasetReader."""

    def __init__(self, count: int, descriptions: list[str] = None):
        self.count = count
        self.descriptions = descriptions or [None] * count


class TestRasterBandIndexResolution:
    """Test suite for _resolve_band_index in mosaic.py."""

    def test_single_band_raster_returns_index_1(self):
        """Single-band raster (COG band asset) returns band index 1 without error."""
        single_src = MockRasterSource(count=1)
        assert _resolve_band_index(single_src, "B08") == 1
        assert _resolve_band_index(single_src, "B04") == 1
        assert _resolve_band_index(single_src, "B11") == 1
        assert _resolve_band_index(single_src, "SCL") == 1

    def test_multiband_raster_resolves_known_bands(self):
        """Multi-band raster (e.g. 12-band scene) resolves S2 bands to correct 1-based index."""
        multi_src = MockRasterSource(count=12)
        assert _resolve_band_index(multi_src, "B04") == 4
        assert _resolve_band_index(multi_src, "B08") == 8
        assert _resolve_band_index(multi_src, "B11") == 11

    def test_multiband_raster_resolves_descriptions(self):
        """Multi-band raster resolves by band descriptions when available."""
        multi_src = MockRasterSource(count=3, descriptions=["Red band", "Green band", "Near-Infrared B08"])
        assert _resolve_band_index(multi_src, "B08") == 3
        assert _resolve_band_index(multi_src, "Red") == 1

    def test_multiband_unresolvable_band_raises_value_error(self):
        """Multi-band raster must raise ValueError when a band cannot be resolved (no silent fallback to band 1)."""
        multi_src = MockRasterSource(count=3, descriptions=["Red", "Green", "Blue"])
        with pytest.raises(ValueError) as exc_info:
            _resolve_band_index(multi_src, "B11")  # B11 is index 11, out of bounds for count=3

        err_msg = str(exc_info.value)
        assert "Cannot resolve band 'B11'" in err_msg
        assert "3 bands" in err_msg
