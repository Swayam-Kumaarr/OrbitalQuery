"""
NASA Earthdata authentication helper for HLS asset access.

HLS data is hosted on LP DAAC and requires Earthdata Login authentication.
This module provides secure helpers for:

1. Loading credentials from environment variables
2. Providing HTTP Basic Auth headers for httpx/requests
3. Setting GDAL_HTTP_USERPWD for rasterio authenticated reads
4. Testing authentication against a known HLS asset

CRITICAL SECURITY RULES:
- NEVER log credentials
- NEVER return credentials in API responses
- NEVER expose credentials to the frontend
- Use environment variables only — no hardcoded values
- Clear credentials from memory when possible
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

import base64

logger = logging.getLogger(__name__)

# Environment variable names
EARTHDATA_USER_ENV = "EARTHDATA_USER"
EARTHDATA_PASS_ENV = "EARTHDATA_PASS"

# Known HLS test asset (Delhi AOI, 2024) — used for auth verification only
HLS_TEST_ASSET_URL = (
    "https://lpdaac.earthdata.nasa.gov/lp-prod-data/"
    "HLS.S30.T43RGM.2024059T052749.v2.0/HLS.S30.T43RGM.2024059T052749.v2.0.B04.tif"
)


def get_earthdata_credentials() -> Tuple[Optional[str], Optional[str]]:
    """
    Load Earthdata credentials from environment variables.

    Returns:
        (user, pass) tuple, or (None, None) if not configured.
    """
    user = os.environ.get(EARTHDATA_USER_ENV)
    password = os.environ.get(EARTHDATA_PASS_ENV)

    if user and password:
        logger.info("[EarthdataAuth] Credentials loaded from environment")
        return user, password
    else:
        logger.warning(
            "[EarthdataAuth] No credentials found — set %s and %s",
            EARTHDATA_USER_ENV,
            EARTHDATA_PASS_ENV,
        )
        return None, None


def get_earthdata_auth_header() -> Optional[dict]:
    """
    Return HTTP Basic Auth header dict for Earthdata.

    Returns:
        {"Authorization": "Basic <base64>"} or None if no credentials.
    """
    user, password = get_earthdata_credentials()
    if not user or not password:
        return None

    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("utf-8")
    return {"Authorization": f"Basic {token}"}


def get_earthdata_userpwd() -> Optional[str]:
    """
    Return 'user:password' string for GDAL_HTTP_USERPWD.

    Returns:
        'user:password' or None if no credentials.
    """
    user, password = get_earthdata_credentials()
    if not user or not password:
        return None
    return f"{user}:{password}"


def get_earthdata_rasterio_env() -> dict:
    """
    Return GDAL environment settings for authenticated rasterio reads.

    Sets GDAL_HTTP_USERPWD for Earthdata-authenticated COG access.
    Also includes the standard memory-safe GDAL settings.

    Returns:
        Dict of env vars to pass to rasterio.Env(), or empty dict if no credentials.
    """
    env_settings = {
        "GDAL_CACHEMAX": 64,
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_TIMEOUT": "30",
        "GDAL_HTTP_MAX_RETRY": "3",
    }

    userpwd = get_earthdata_userpwd()
    if userpwd:
        env_settings["GDAL_HTTP_USERPWD"] = userpwd

    return env_settings


def is_earthdata_url(url: Optional[str]) -> bool:
    """
    Check if a URL requires Earthdata authentication.

    Returns True for LP DAAC, Earthdata, or urs.earthdata.nasa.gov URLs.
    """
    if not url:
        return False
    earthdata_domains = [
        "lpdaac.earthdata.nasa.gov",
        "urs.earthdata.nasa.gov",
        "earthdata.nasa.gov",
    ]
    return any(domain in url for domain in earthdata_domains)


def test_earthdata_authentication(test_url: Optional[str] = None) -> dict:
    """
    Test Earthdata authentication by attempting a HEAD request to an HLS asset.

    Returns dict with:
        - authenticated: bool
        - http_status: int or None
        - error: str or None
        - url: str tested
    """
    import httpx

    url = test_url or HLS_TEST_ASSET_URL
    user, password = get_earthdata_credentials()

    result = {
        "authenticated": False,
        "http_status": None,
        "error": None,
        "url": url,
    }

    if not user or not password:
        result["error"] = "No Earthdata credentials configured"
        return result

    try:
        auth = (user, password)
        headers = {}
        with httpx.Client(timeout=30.0, follow_redirects=False) as client:
            resp = client.head(url, auth=auth, headers=headers)
            result["http_status"] = resp.status_code
            result["authenticated"] = resp.status_code in (200, 302, 303)
            if not result["authenticated"]:
                result["error"] = f"HTTP {resp.status_code}: {resp.reason_phrase}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:200]}"

    return result


def test_rasterio_earthdata_read(test_url: Optional[str] = None) -> dict:
    """
    Test that rasterio can open an Earthdata-authenticated HLS asset.

    Returns dict with:
        - readable: bool
        - bands: int or None
        - shape: [height, width] or None
        - crs: str or None
        - resolution: float or None
        - dtype: str or None
        - nodata: int/float or None
        - error: str or None
    """
    import rasterio

    url = test_url or HLS_TEST_ASSET_URL
    user, password = get_earthdata_credentials()

    result = {
        "readable": False,
        "bands": None,
        "shape": None,
        "crs": None,
        "resolution": None,
        "dtype": None,
        "nodata": None,
        "error": None,
        "url": url,
    }

    if not user or not password:
        result["error"] = "No Earthdata credentials configured"
        return result

    env = get_earthdata_rasterio_env()
    try:
        with rasterio.Env(**env):
            with rasterio.open(url) as src:
                result["readable"] = True
                result["bands"] = src.count
                result["shape"] = [src.height, src.width]
                result["crs"] = str(src.crs) if src.crs else None
                result["resolution"] = abs(src.transform.a) if src.transform else None
                result["dtype"] = str(src.dtypes[0]) if src.dtypes else None
                result["nodata"] = src.nodata
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    return result
