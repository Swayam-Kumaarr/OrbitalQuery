"""
Regression tests for TASK 1 and TASK 2 fixes:

1. StackSTAC dtype/fill_value handling — float32 must not fall back to float64
2. Sentinel-2 / Landsat band resolution — no silent fallback to band 1
"""
from __future__ import annotations

import sys
import numpy as np
import pytest


def test_stackstac_mosaic_requires_valid_items():
    from app.services.stackstac_adapter import stackstac_mosaic

    items = [
        {
            "id": "S2A_2024",
            "collection": "sentinel-2-l2a",
            "bbox": [77.0, 28.4, 77.4, 28.8],
            "geometry": {"type": "Polygon", "coordinates": [[[77.0, 28.4], [77.4, 28.4], [77.4, 28.8], [77.0, 28.8], [77.0, 28.4]]]},
            "properties": {"datetime": "2024-01-01T00:00:00Z", "platform": "sentinel-2a"},
            "assets": {},
        }
    ]

    with pytest.raises(Exception):
        stackstac_mosaic(
            stac_items=items,
            bbox=[77.0, 28.4, 77.4, 28.8],
            band_names=["B04"],
            resolution=20.0,
            epsg=32643,
            dtype="float32",
        )


def test_stackstac_fill_value_is_float32_compatible():
    """StackSTAC must use a float32-compatible fill value, not float64 NaN."""
    source = open("app/services/stackstac_adapter.py").read()
    assert 'fill_value=float("nan")' not in source
    assert 'fill_value=np.nan)' not in source
    assert 'fill_value=fill_val' in source
    assert 'fill_val = np.float32(np.nan)' in source


def test_stackstac_fill_sentinel_is_float32_compatible():
    import app.services.stackstac_adapter as mod
    assert hasattr(mod, "stackstac_mosaic")
    source = open(mod.__file__).read()
    assert "fill_value=fill_val" in source or "np.float32(np.nan)" in source
    assert 'fill_value=float("nan")' not in source


def test_stackstac_dtype_remains_float32():
    """The adapter must not silently cast to float64."""
    source = open("app/services/stackstac_adapter.py").read()
    assert 'dtype="float64"' not in source
    assert 'np.float64' not in source


def test_resolve_band_index_raises_for_unknown_band():
    from app.services.mosaic import _resolve_band_index

    class FakeRaster:
        count = 3
        descriptions = ["Band 1 Coastal", "Band 2 Blue", "Band 3 Green"]

    with pytest.raises(ValueError, match="Cannot resolve band"):
        _resolve_band_index(FakeRaster(), "B99")


def test_resolve_band_index_sentinel2_mapping():
    from app.services.mosaic import _resolve_band_index

    class FakeS2:
        count = 12
        descriptions = [
            "B1 Coastal aerosol", "B2 Blue", "B3 Green", "B4 Red",
            "B5 Red edge 1", "B6 Red edge 2", "B7 Red edge 3",
            "B8 NIR", "B8A Narrow NIR", "B9 Water vapour",
            "B11 SWIR1", "B12 SWIR2",
        ]

    assert _resolve_band_index(FakeS2(), "B04") == 4
    assert _resolve_band_index(FakeS2(), "B08") == 8
    assert _resolve_band_index(FakeS2(), "B11") == 11

    # Single-band SCL asset (count=1) resolves to index 1
    class FakeSCLAsset:
        count = 1
        descriptions = [None]

    assert _resolve_band_index(FakeSCLAsset(), "SCL") == 1

    with pytest.raises(ValueError):
        _resolve_band_index(FakeS2(), "SCL")
