"""OrbitalQuery EO Analysis Service — main FastAPI application."""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import HOST, PORT, STAC_API_URL
from app.routes import health, temporal_compare, stac, query
# Defer heavy route imports until first request to save startup memory
_heavy_routes_loaded = False

def _load_heavy_routes():
    global _heavy_routes_loaded
    if _heavy_routes_loaded:
        return
    _heavy_routes_loaded = True
    from app.routes import analysis, change, decision, evidence, explain, flood, index, preprocess, providers, provenance, semantic, sensor, timeseries
    for r in [analysis, change, decision, evidence, explain, flood, index, preprocess, providers, provenance, semantic, sensor, timeseries]:
        app.include_router(r.router)

from app.services.eo_provider import init_default_provider, register_provider
from app.security import RateLimitMiddleware, SecurityHeadersMiddleware, AuditMiddleware, get_cors_origins

# ── Logging ──────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("orbitalquery-analysis")

# ── App ──────────────────────────────────────────────────────────

app = FastAPI(
    title="OrbitalQuery EO Analysis Service",
    description=(
        "Earth Observation analysis microservice. "
        "Search STAC catalogs, access satellite imagery, "
        "and compute raster statistics for areas of interest."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Security middleware (applied in reverse order — last added = first executed)
# 1. Rate limiting
app.add_middleware(RateLimitMiddleware, max_requests=60, window_seconds=60)

# 2. Security headers
app.add_middleware(SecurityHeadersMiddleware)

# 3. Audit logging
app.add_middleware(AuditMiddleware)

# CORS — use environment-aware origins (no wildcard in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── Initialize EO Providers (minimal at startup) ──────────────
import os

# Only initialize Planetary Computer at startup (needed for temporal-compare)
init_default_provider(api_url=STAC_API_URL)
logger.info("EO Provider initialized: planetary_computer")

# Other providers loaded lazily on first use
_other_providers_loaded = False
def _load_other_providers():
    global _other_providers_loaded
    if _other_providers_loaded:
        return
    _other_providers_loaded = True
    from app.services.eo_provider import CopernicusProvider, BhoonidhiProvider, AWSEarthSearchProvider, NASACMRProvider
    copernicus_token = os.environ.get("COPERNICUS_TOKEN")
    if copernicus_token:
        register_provider(CopernicusProvider(token=copernicus_token), default=False)
    bhoonidhi_user = os.environ.get("BHOONIDHI_USER")
    bhoonidhi_pass = os.environ.get("BHOONIDHI_PASS")
    if bhoonidhi_user and bhoonidhi_pass:
        register_provider(BhoonidhiProvider(user_id=bhoonidhi_user, password=bhoonidhi_pass), default=False)
    try:
        register_provider(AWSEarthSearchProvider(), default=False)
    except Exception:
        pass
    try:
        register_provider(NASACMRProvider(), default=False)
    except Exception:
        pass# ── Routes (essential only at startup — heavy routes deferred) ─
app.include_router(health.router)
app.include_router(temporal_compare.router)  # Main analysis endpoint
app.include_router(stac.router)
app.include_router(query.router)

# Load remaining routes on first non-health request (saves ~50MB startup)
@app.middleware("http")
async def load_heavy_routes_middleware(request, call_next):
    if request.url.path not in ("/health", "/", "/docs", "/redoc"):
        _load_heavy_routes()
    return await call_next(request)


import os as _os
_ENV = _os.getenv("ENVIRONMENT", "development")

@app.get("/", tags=["root"])
async def root():
    """Root endpoint — service info."""
    return {
        "service": "OrbitalQuery EO Analysis Service",
        "version": "0.1.0",
        "docs": "/docs" if _ENV != "production" else "disabled in production",
        "stac_api": STAC_API_URL,
        "endpoints": {
            "health": "GET /health",
            "stac_search": "POST /stac/search",
            "analysis_preview": "POST /analysis/preview",
            "preprocess": "POST /analysis/preprocess",
            "indices": "GET /analysis/indices",
            "index": "POST /analysis/index",
            "change_detect": "POST /analysis/change-detect",
            "timeseries": "POST /analysis/timeseries",
            "evidence": "POST /analysis/evidence/select",
            "sensors": "GET /analysis/sensors",
            "sentinel1_search": "POST /analysis/sentinel1/search",
            "flood_assess": "POST /analysis/flood/assess",
            "explain": "POST /analysis/explain",
        },
    }


# ── Run ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting EO Analysis Service on %s:%d", HOST, PORT)
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
