"""
STAC search service — delegates to EOProvider interface.

All existing function signatures are preserved for backward compatibility.
Internally delegates to the registered EOProvider (default: Planetary Computer).

This ensures the analysis engine does not depend directly on any
provider-specific library. Provider logic lives in eo_provider.py.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from app.services.eo_provider import (
    EOProvider,
    get_default_provider,
    get_provider,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════
# Backward-compatible API (delegates to provider)
# ══════════════════════════════════════════════════════════════════

def get_stac_client():
    """
    Get the underlying STAC client from the default provider.

    NOTE: This leaks the provider-specific client. New code should use
    the EOProvider interface directly. Kept for backward compatibility
    with preprocessing, sentinel1, and temporal_engine modules.
    """
    provider = get_default_provider()
    if hasattr(provider, '_get_client'):
        return provider._get_client()
    raise NotImplementedError(
        f"Provider '{provider.get_name()}' does not expose a raw STAC client. "
        "Use the EOProvider.search() interface instead."
    )


def search_stac(
    collection: str,
    bbox: Optional[list[float]] = None,
    datetime: Optional[str] = None,
    max_cloud_cover: Optional[int] = None,
    limit: int = 10,
    query: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Search STAC API for items matching the given parameters.

    Delegates to the registered EOProvider.

    Returns a dict with 'items' (list of signed STAC items) and 'total'.
    """
    provider = get_default_provider()
    result = provider.search(
        collection=collection,
        bbox=bbox,
        datetime=datetime,
        max_cloud_cover=max_cloud_cover,
        limit=limit,
        query=query,
    )
    return {
        "items": result.items,
        "total": result.total,
    }


def get_item_assets(item_dict: dict[str, Any]) -> dict[str, dict]:
    """Extract assets from a STAC item dict."""
    provider = get_default_provider()
    return provider.get_assets(item_dict)


# Asset types that are NOT rasterio-compatible (JPEG/PNG renders)
NON_RASTER_ASSETS = {
    "visual", "rendered_preview", "thumbnail", "preview",
    "quicklook", "tilejson",
}

# Raster band names (GeoTIFF, windowed-read compatible)
RASTER_BANDS = {
    "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08",
    "B8A", "B09", "B11", "B12", "AOT", "SCL", "WVP",
    "red", "green", "blue", "nir", "swir16", "swir22",
}


def select_best_asset(
    assets: dict[str, dict],
    preferred_bands: Optional[list[str]] = None,
    mode: str = "analysis",
) -> tuple[str, dict]:
    """
    Select the best asset for raster access.

    mode='analysis': prefer GeoTIFF raster bands for windowed reads.
    mode='preview': prefer visual/rendered thumbnails.

    Returns (asset_key, asset_dict).
    """
    if preferred_bands:
        for band in preferred_bands:
            if band in assets:
                return band, assets[band]

    if mode == "analysis":
        for key in ["B04", "B08", "B03", "B02", "B05", "B11", "B12"]:
            if key in assets:
                return key, assets[key]
        for key, asset in assets.items():
            if key.lower() not in NON_RASTER_ASSETS:
                return key, asset

    elif mode == "preview":
        for key in ["visual", "rendered_preview", "thumbnail", "preview"]:
            if key in assets:
                return key, assets[key]

    if assets:
        first_key = next(iter(assets))
        return first_key, assets[first_key]

    raise ValueError("No assets found in STAC item")


def resolve_band_asset(
    assets: dict[str, Any],
    band_name: str,
    collection: Optional[str] = None,
) -> tuple[str, Any]:
    """
    Resolve a logical or physical band name to its corresponding STAC asset.

    Resolution strategy:
    1. Exact key match (e.g. "B04", "B08", "B11", "SCL", "QA_PIXEL")
    2. Case-insensitive key match (e.g. "b04" -> "B04", "scl" -> "SCL")
    3. STAC eo:bands metadata matching (by 'name' or 'common_name')
    4. Canonical band alias matching (e.g. "red" -> "B04", "nir" -> "B08" for Sentinel-2)
    5. Asset title / description matching (e.g. "Band 4 - Red")

    Raises KeyError if the band cannot be resolved. NEVER silently defaults to another band.

    Returns:
        tuple of (resolved_asset_key, asset_dict_or_object)
    """
    if not assets:
        raise KeyError(
            f"Cannot resolve band '{band_name}': STAC item contains no assets. "
            f"Collection: '{collection or 'unknown'}'"
        )

    # 1. Exact key match
    if band_name in assets:
        return band_name, assets[band_name]

    # 2. Case-insensitive key match
    band_name_upper = band_name.strip().upper()
    for key, asset in assets.items():
        if key.strip().upper() == band_name_upper:
            return key, asset

    # 3. STAC eo:bands metadata matching
    band_name_lower = band_name.strip().lower()
    for key, asset in assets.items():
        eo_bands = []
        if isinstance(asset, dict):
            eo_bands = asset.get("eo:bands") or asset.get("extra_fields", {}).get("eo:bands", [])
        elif hasattr(asset, "extra_fields"):
            eo_bands = asset.extra_fields.get("eo:bands", [])
        for b_meta in eo_bands:
            if isinstance(b_meta, dict):
                c_name = str(b_meta.get("common_name", "")).strip().lower()
                n_name = str(b_meta.get("name", "")).strip().upper()
                if c_name == band_name_lower or n_name == band_name_upper:
                    return key, asset

    # 4. Known canonical aliases for Sentinel-2 / Landsat
    S2_ALIASES = {
        "RED": "B04", "NIR": "B08", "SWIR": "B11", "SWIR1": "B11", "SWIR16": "B11",
        "SWIR2": "B12", "SWIR22": "B12", "GREEN": "B03", "BLUE": "B02",
        "COASTAL": "B01", "REDEDGE1": "B05", "REDEDGE2": "B06", "REDEDGE3": "B07",
        "NIR08": "B8A", "NARROW_NIR": "B8A", "WVP": "B09", "WATER_VAPOUR": "B09",
        "SCL": "SCL", "SCENE_CLASSIFICATION": "SCL", "AOT": "AOT",
    }
    LANDSAT_ALIASES = {
        "RED": "B4", "NIR": "B5", "SWIR": "B6", "SWIR1": "B6", "SWIR16": "B6",
        "SWIR2": "B7", "SWIR22": "B7", "GREEN": "B3", "BLUE": "B2",
        "COASTAL": "B1", "QA": "QA_PIXEL", "QA_PIXEL": "QA_PIXEL",
    }

    target_alias = None
    if collection and "landsat" in collection.lower():
        target_alias = LANDSAT_ALIASES.get(band_name_upper)
    else:
        # Default to Sentinel-2 or check both
        target_alias = S2_ALIASES.get(band_name_upper) or LANDSAT_ALIASES.get(band_name_upper)

    if target_alias:
        if target_alias in assets:
            return target_alias, assets[target_alias]
        for key, asset in assets.items():
            if key.strip().upper() == target_alias.upper():
                return key, asset

    # 5. Asset title / description matching
    for key, asset in assets.items():
        title = ""
        desc = ""
        if isinstance(asset, dict):
            title = str(asset.get("title", ""))
            desc = str(asset.get("description", ""))
        elif hasattr(asset, "title"):
            title = str(getattr(asset, "title", "") or "")
            desc = str(getattr(asset, "description", "") or "")

        combined = f"{title} {desc}".upper()
        if band_name_upper in combined:
            return key, asset

    # Failed to resolve — raise explicit error with full diagnostics
    available_keys = list(assets.keys())
    raise KeyError(
        f"Could not resolve STAC asset for requested band '{band_name}'. "
        f"Collection: '{collection or 'unknown'}'. "
        f"Available asset keys: {available_keys}"
    )


def get_asset_href(asset: dict[str, Any]) -> str:
    """Get the href from a STAC asset, preferring signed href."""
    provider = get_default_provider()
    return provider.get_asset_href(asset)


def check_stac_api_reachable() -> bool:
    """Check if the STAC API is reachable via the default provider."""
    try:
        provider = get_default_provider()
        return provider.is_reachable()
    except Exception as e:
        logger.error("Provider unreachable: %s", e)
        return False
