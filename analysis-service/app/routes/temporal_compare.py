"""
Temporal Comparison endpoint.

POST /analysis/temporal-compare — run the full before/after comparison pipeline.

Takes a validated analysis plan and produces:
  - Scene selections for period 1 and period 2
  - Spectral index results
  - Change detection
  - Computed metrics
  - Imagery URLs
  - Structured explanation
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import gc

from app.security import validate_query_safe
from app.services.temporal_compare import run_temporal_comparison
from app.services.query_to_plan import build_analysis_plan
from app.services.memory_guard import check_memory_headroom, get_safe_max_dim

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/analysis", tags=["temporal-compare"])


class TemporalCompareRequest(BaseModel):
    """Request body for temporal comparison."""
    query: str = Field(
        ...,
        min_length=5,
        max_length=500,
        description="Natural language query describing the analysis",
        examples=["Hyderabad urban expansion 2021 vs 2025"],
    )
    # Optional overrides
    phenomenon: Optional[str] = Field(None, description="Override detected phenomenon")
    aoi: Optional[str] = Field(None, description="Override location name")
    bbox: Optional[list[float]] = Field(None, min_length=4, max_length=4, description="Override bounding box")
    start_date: Optional[str] = Field(None, description="Override start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="Override end date (YYYY-MM-DD)")
    sensor: Optional[str] = Field(None, description="Override preferred sensor")
    analysis_type: Optional[str] = Field(None, description="Override analysis type")
    cloud_threshold: Optional[int] = Field(None, ge=0, le=100, description="Override cloud threshold")
    change_detection_method: Optional[str] = Field(None, description="Override change detection method (phenomenon_aware_difference, cva, object_based)")


class TemporalCompareResponse(BaseModel):
    """Response for temporal comparison."""
    request_id: str
    status: str
    plan: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    message: Optional[str] = None
    errors: Optional[list[str]] = None


@router.post("/temporal-compare", response_model=TemporalCompareResponse)
async def temporal_compare(req: TemporalCompareRequest) -> TemporalCompareResponse:
    """
    Run the full temporal comparison pipeline.

    1. Parse NL query → analysis plan
    2. Search scenes for period 1 and period 2
    3. Select best scenes
    4. Compute spectral indices
    5. Run change detection
    6. Produce metrics + explanation
    """
    request_id = str(uuid.uuid4())

    # Security: validate query
    sanitized_query = validate_query_safe(req.query)

    logger.info(
        "[temporal-compare] requestId=%s query='%s'",
        request_id, sanitized_query[:80],
    )

    # Step 1: Build analysis plan
    overrides = {}
    if req.phenomenon:
        overrides["phenomenon"] = req.phenomenon
    if req.aoi:
        overrides["aoi"] = req.aoi
    if req.bbox:
        overrides["bbox"] = req.bbox
    if req.start_date:
        overrides["start_date"] = req.start_date
    if req.end_date:
        overrides["end_date"] = req.end_date
    if req.sensor:
        overrides["sensor"] = req.sensor
    if req.analysis_type:
        overrides["analysis_type"] = req.analysis_type
    if req.cloud_threshold is not None:
        overrides["cloud_threshold"] = req.cloud_threshold
    if req.change_detection_method is not None:
        overrides["change_detection_method"] = req.change_detection_method

    plan_result = build_analysis_plan(req.query, overrides or None)

    if plan_result["status"] != "ok":
        return TemporalCompareResponse(
            request_id=request_id,
            status=plan_result["status"],
            message=plan_result.get("message"),
            errors=plan_result.get("errors"),
        )

    plan = plan_result["plan"]

    # Pre-flight memory check
    mem_ok, mem_mb, mem_msg = check_memory_headroom(required_mb=150)
    if not mem_ok:
        logger.error("[temporal-compare] Insufficient memory: %s", mem_msg)
        return TemporalCompareResponse(
            request_id=request_id,
            status="error",
            plan=plan,
            message="Analysis engine has insufficient memory for this request. Try a smaller area or shorter time period.",
            errors=[mem_msg],
        )

    # Compute safe max_dim for this run
    bbox = plan.get("bbox", [0, 0, 0, 0])
    bbox_area = abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1]) if len(bbox) == 4 else 0.15
    safe_max_dim = get_safe_max_dim(bbox_area)
    plan["_safe_max_dim"] = safe_max_dim

    # Step 2-8: Run temporal comparison
    try:
        gc.collect()  # Clean up before heavy work
        result = run_temporal_comparison(plan)

        # Serialize result (convert dataclasses to dicts)
        result_dict = {
            "plan_id": result.plan_id,
            "phenomenon": result.phenomenon,
            "analysis_type": result.analysis_type,
            "aoi_name": result.aoi_name,
            "aoi_bbox": result.aoi_bbox,
            "period1": result.period1,
            "period2": result.period2,
            "scene_t1": {
                "item_id": result.scene_t1.item_id,
                "collection": result.scene_t1.collection,
                "datetime": result.scene_t1.datetime,
                "cloud_cover": result.scene_t1.cloud_cover,
                "bbox": result.scene_t1.bbox,
                "provider": result.scene_t1.provider,
                "platform": result.scene_t1.platform,
            } if result.scene_t1 else None,
            "scene_t2": {
                "item_id": result.scene_t2.item_id,
                "collection": result.scene_t2.collection,
                "datetime": result.scene_t2.datetime,
                "cloud_cover": result.scene_t2.cloud_cover,
                "bbox": result.scene_t2.bbox,
                "provider": result.scene_t2.provider,
                "platform": result.scene_t2.platform,
            } if result.scene_t2 else None,
            "index_t1": {
                "index_name": result.index_t1.index_name,
                "stats": result.index_t1.stats,
                "scene_id": result.index_t1.scene_id,
                "date": result.index_t1.date,
                "resolution_m": result.index_t1.resolution_m,
                "valid_pixels": result.index_t1.valid_pixels,
                "total_pixels": result.index_t1.total_pixels,
            } if result.index_t1 else None,
            "index_t2": {
                "index_name": result.index_t2.index_name,
                "stats": result.index_t2.stats,
                "scene_id": result.index_t2.scene_id,
                "date": result.index_t2.date,
                "resolution_m": result.index_t2.resolution_m,
                "valid_pixels": result.index_t2.valid_pixels,
                "total_pixels": result.index_t2.total_pixels,
            } if result.index_t2 else None,
            "change_detection": result.change_detection,
            "change_visualizations": result.change_visualizations,
            "metrics": result.metrics,
            "imagery": result.imagery,
            "processing_steps": result.processing_steps,
            "sensor_info": result.sensor_info,
            "provenance": result.provenance,
            "explanation": result.explanation,
        }

        return TemporalCompareResponse(
            request_id=request_id,
            status="ok",
            plan=plan,
            result=result_dict,
        )

    except MemoryError as e:
        gc.collect()
        logger.error("[temporal-compare] OOM: %s", e)
        return TemporalCompareResponse(
            request_id=request_id,
            status="error",
            plan=plan,
            message="Analysis ran out of memory. Try a smaller area (e.g., a city district instead of full state) or shorter time period.",
            errors=[f"MemoryError: {str(e)[:200]}"],
        )
    except Exception as e:
        gc.collect()
        logger.error("[temporal-compare] Pipeline failed: %s", e, exc_info=True)
        return TemporalCompareResponse(
            request_id=request_id,
            status="error",
            plan=plan,
            message=f"Temporal comparison pipeline failed: {str(e)[:200]}",
        )
