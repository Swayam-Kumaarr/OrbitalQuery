"""
Memory guard for Render free tier (512MB limit).

Monitors RSS memory and provides early warnings before OOM kills the process.
"""

from __future__ import annotations

import logging
import os
import resource
from typing import Optional

logger = logging.getLogger(__name__)

# Render free tier limit (with some safety margin)
RENDER_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "460"))  # 460MB safety margin of 512MB


def get_rss_mb() -> float:
    """Get current RSS (Resident Set Size) in MB."""
    # resource.getrusage returns values in KB on Linux, bytes on macOS
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On Linux, ru_maxrss is in KB; on macOS it's in bytes
    if os.name == 'posix':
        # Linux: ru_maxrss in KB
        rss_bytes = usage.ru_maxrss * 1024
    else:
        rss_bytes = usage.ru_maxrss
    return rss_bytes / (1024 * 1024)


def check_memory_headroom(required_mb: float = 100) -> tuple[bool, float, str]:
    """
    Check if there's enough memory headroom for an operation.

    Returns:
        (ok, current_mb, message)
    """
    current_mb = get_rss_mb()
    remaining = RENDER_LIMIT_MB - current_mb

    if remaining < required_mb:
        msg = (
            f"Low memory: {current_mb:.0f}MB used, {remaining:.0f}MB remaining "
            f"(need {required_mb:.0f}MB). Limit={RENDER_LIMIT_MB}MB"
        )
        logger.warning("[MEMORY] %s", msg)
        return False, current_mb, msg

    return True, current_mb, f"Memory OK: {current_mb:.0f}MB used, {remaining:.0f}MB remaining"


def estimate_array_memory_mb(height: int, width: int, n_bands: int = 1, dtype_bytes: int = 4) -> float:
    """Estimate memory for a numpy array in MB."""
    return (height * width * n_bands * dtype_bytes) / (1024 * 1024)


def get_safe_max_dim(bbox_area_deg2: float = 0.15) -> int:
    """
    Compute a safe max_dim based on available memory.

    For Render free tier (460MB safety margin), we need to keep total
    array memory under ~300MB to leave room for GDAL/rasterio overhead.

    Args:
        bbox_area_deg2: Area of the AOI in degrees² (approximate)
    """
    # Memory budget for arrays: 300MB
    # Typical analysis loads: NDBI t1, NDBI t2, NDVI t1, NDVI t2,
    # SCL t1, SCL t2, change mask, morphology = ~8 arrays
    # Plus 2 arrays for aligned rasters = 10 arrays total
    BUDGET_MB = 300
    N_ARRAYS = 10
    BYTES_PER_PIXEL = 4  # float32

    # pixels_per_array = BUDGET_MB * 1024^2 / (N_ARRAYS * BYTES_PER_PIXEL)
    pixels_budget = (BUDGET_MB * 1024 * 1024) / (N_ARRAYS * BYTES_PER_PIXEL)
    # max_dim = sqrt(pixels_budget)
    max_dim = int(pixels_budget ** 0.5)

    # Clamp between reasonable bounds
    max_dim = max(256, min(max_dim, 1024))

    logger.info(
        "[MEMORY] Safe max_dim=%d (budget=%dMB, %d arrays, bbox_area=%.3f deg²)",
        max_dim, BUDGET_MB, N_ARRAYS, bbox_area_deg2,
    )
    return max_dim
