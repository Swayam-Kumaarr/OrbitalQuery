"""
Memory guard for Render free tier (512MB limit).

Monitors RSS memory and provides early warnings before OOM kills the process.
Designed to prevent Render from SIGKILL-ing the process.
"""

from __future__ import annotations

import logging
import os
import gc
from typing import Optional

logger = logging.getLogger(__name__)

# Render free tier = 512MB. Use 420MB as hard limit to leave room for overhead.
RENDER_LIMIT_MB = int(os.getenv("MEMORY_LIMIT_MB", "420"))


def get_rss_mb() -> float:
    """Get current RSS (Resident Set Size) in MB via /proc."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Format: "VmRSS:    123456 kB"
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except (FileNotFoundError, IndexError, ValueError):
        pass
    # Fallback: try resource module
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return (usage.ru_maxrss / 1024.0) if os.name == 'posix' else (usage.ru_maxrss / (1024 * 1024))
    except Exception:
        return 0.0


def check_memory_headroom(required_mb: float = 150) -> tuple[bool, float, str]:
    """
    Check if there's enough memory headroom for an operation.
    
    Returns:
        (ok, current_mb, message)
    """
    current_mb = get_rss_mb()
    remaining = RENDER_LIMIT_MB - current_mb

    if remaining < required_mb:
        gc.collect()  # Try to reclaim memory first
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
    
    For Render free tier (420MB hard limit), keep total array memory under 120MB
    to leave maximum room for Python runtime + GDAL + rasterio overhead (~300MB).
    
    Args:
        bbox_area_deg2: Area of the AOI in degrees² (approximate)
    """
    current_mb = get_rss_mb()
    available_for_arrays = max(50, RENDER_LIMIT_MB - current_mb - 200)  # 200MB for overhead
    available_for_arrays = min(available_for_arrays, 120)  # Cap at 120MB for arrays
    
    N_ARRAYS = 12  # NDBI t1/t2, NDVI t1/t2, SCL t1/t2, change masks, aligned arrays
    BYTES_PER_PIXEL = 4  # float32
    
    pixels_budget = (available_for_arrays * 1024 * 1024) / (N_ARRAYS * BYTES_PER_PIXEL)
    max_dim = int(pixels_budget ** 0.5)
    
    # Hard clamp: 256–512 for Render free tier
    max_dim = max(256, min(max_dim, 512))
    
    logger.info(
        "[MEMORY] RSS=%.0fMB, available_for_arrays=%dMB, safe_max_dim=%d",
        current_mb, available_for_arrays, max_dim,
    )
    return max_dim
