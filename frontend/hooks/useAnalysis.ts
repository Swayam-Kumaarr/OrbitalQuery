'use client';

import { useState, useCallback } from 'react';

class AnalysisError extends Error {
  code: string | null;
  constructor(message: string, code: string | null = null) {
    super(message);
    this.name = 'AnalysisError';
    this.code = code;
  }
}

// ── Types ───────────────────────────────────────────────────────

export interface AnalysisPlan {
  plan_id: string;
  phenomenon: string;
  phenomenon_description: string;
  analysis_type: string;
  sensor: string;
  bands: string[];
  aoi: string;
  bbox: number[];
  start_date: string;
  end_date: string;
  cloud_threshold: number;
  comparison_strategy: string;
  min_scenes: number;
  max_scenes: number;
  output_requirements: string[];
  required_indices: string[];
  validation: {
    status: string;
    phenomenon: string;
    analysis_type: string;
    sensor: string;
    bands: string[];
    dates_provided: boolean;
    bbox_provided: boolean;
  };
}

export interface SceneInfo {
  item_id: string;
  collection: string;
  datetime: string;
  cloud_cover: number | null;
  bbox: number[];
  provider: string;
  platform: string;
}

export interface IndexInfo {
  index_name: string;
  stats: Record<string, number>;
  scene_id: string;
  date: string;
  resolution_m: number;
  valid_pixels: number;
  total_pixels: number;
}

export interface GeoJsonRegion {
  region_id: number;
  area_pixels: number;
  area_sq_meters: number;
  area_ha?: number;
  area_km2?: number;
  bbox: number[];
  centroid: number[];
  centroid_pixel?: number[];
  mean_delta: number;
  max_delta: number;
  min_delta: number;
  direction: string;
  primary_indicator?: string;
  index_name?: string;
  supporting_indicators?: string[];
  indicator_formulas?: Record<string, string>;
  algorithm?: string;
  threshold?: number | null;
  ndvi_threshold?: number | null;
  period_1?: string;
  period_2?: string;
  collection?: string;
  provider?: string;
  platform?: string;
  instrument?: string;
  processing_level?: string;
  crs?: string;
  resolution_meters?: number;
  pixel_area_m2?: number;
  scene_ids?: string[];
  acquisition_dates?: string[];
  cloud_cover?: number[];
  aoi_coverage?: number | null;
  quality_mask?: string;
  scl_usage?: string;
  composite_method?: string;
  observation_count?: { period_1?: number; period_2?: number };
  phenomenon?: string;
  area_calculation?: string;
  polygon_coords?: number[][][];
}

export interface TemporalComparisonResult {
  plan_id: string;
  phenomenon: string;
  analysis_type: string;
  aoi_name: string;
  aoi_bbox: number[];
  period1: { start: string; end: string };
  period2: { start: string; end: string };
  scene_t1: SceneInfo | null;
  scene_t2: SceneInfo | null;
  index_t1: IndexInfo | null;
  index_t2: IndexInfo | null;
  change_detection: Record<string, any> | null;
  change_visualizations: {
    change_mask_png?: string;
    difference_png?: string;
    bbox?: number[];
    change_geojson?: any;
    regions?: GeoJsonRegion[];
  } | null;
  metrics: Record<string, any>;
  imagery: {
    period1: Record<string, string>;
    period2: Record<string, string>;
  };
  processing_steps: Array<{ step: string; detail: string }>;
  sensor_info: Record<string, any>;
  provenance?: {
    query?: { text: string; phenomenon: string; aoi_name: string; aoi_bbox: number[]; period_1: string; period_2: string };
    dataset?: { provider: string; collection: string; platform: Record<string, string>; instrument: string; processing_level: string };
    scenes?: { period_1: { scene_ids: string[]; acquisition_dates: string[]; cloud_cover: number[]; count: number; aoi_coverage: number | null }; period_2: { scene_ids: string[]; acquisition_dates: string[]; cloud_cover: number[]; count: number; aoi_coverage: number | null } };
    quality_method?: { name: string; source: string; resampling: string; cloud_classes: number[]; scl_reprojected: boolean };
    composite_method?: { name: string; description: string; observation_count: { period_1: number; period_2: number }; implementation: string };
    indices?: { primary: string; supporting: string[]; formulas: Record<string, string>; bands: Record<string, string>; sensor: string };
    detector?: { method: string; thresholds: Record<string, any>; min_region_pixels: number; min_region_area_m2: number };
    grid?: { crs: string; resolution_m: number; shape: number[]; pixel_area_m2: number; transform: string; bounds: number[]; reprojection_method: string };
    results?: { valid_pixel_count: number; changed_pixel_count: number; changed_area_m2: number; changed_area_ha: number; changed_area_km2: number; changed_pct: number; region_count: number; pixel_area_m2: number; area_calculation: string };
  };
  explanation: {
    title: string;
    summary: string;
    methodology: string;
    key_findings: string[];
    key_indices: string[];
    sensors_used: string[];
    confidence: string;
    limitations: string[];
  };
}

export type AnalysisStep =
  | 'idle'
  | 'planning'
  | 'searching'
  | 'ranking'
  | 'processing'
  | 'deciding'
  | 'explaining'
  | 'complete'
  | 'error';

export interface AnalysisState {
  step: AnalysisStep;
  query: string;
  plan: AnalysisPlan | null;
  scenes: SceneInfo[];
  result: TemporalComparisonResult | null;
  error: string | null;
  errorCode: string | null;
  detail: string | null;
  processingSteps: Array<{ step: string; detail: string }>;
}

// ── Constants ───────────────────────────────────────────────────

/** 120s timeout — Vercel rewrite + Render cold start (40s) + retry (2s) + analysis (40s) + overhead */
const FETCH_TIMEOUT_MS = 120_000;

// ── Helper: single fetch attempt ────────────────────────────────

export interface AnalysisOverrides {
  bbox?: number[];
  start_date?: string;
  end_date?: string;
  phenomenon?: string;
}

async function fetchAnalysis(query: string, overrides?: AnalysisOverrides): Promise<any> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  const body: any = { query };
  if (overrides?.bbox) body.bbox = overrides.bbox;
  if (overrides?.start_date) body.start_date = overrides.start_date;
  if (overrides?.end_date) body.end_date = overrides.end_date;
  if (overrides?.phenomenon) body.phenomenon = overrides.phenomenon;

  let res: Response;
  try {
    res = await fetch('/api/analysis/temporal-compare', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (fetchErr: any) {
    clearTimeout(timeout);
    if (fetchErr.name === 'AbortError') {
      throw new AnalysisError(
        'TIMEOUT',
        'TIMEOUT',
      );
    }
    // "Failed to fetch" can mean CORS, network, or the backend is unreachable
    throw new AnalysisError('NETWORK', 'NETWORK');
  }
  clearTimeout(timeout);

  let data: any;
  try {
    data = await res.json();
  } catch {
    throw new AnalysisError('Invalid server response.', 'PARSE');
  }

  if (!res.ok) {
    const errMsg = data?.detail || data?.message || data?.error || `Server error (${res.status})`;
    if (
      errMsg.includes('starting up') ||
      errMsg.includes('unavailable') ||
      errMsg.includes('PYTHON_UNAVAILABLE') ||
      res.status === 502 ||
      res.status === 503
    ) {
      throw new AnalysisError('PYTHON_UNAVAILABLE', 'HTTP_503');
    }
    throw new AnalysisError(errMsg, `HTTP_${res.status}`);
  }

  if (data.status === 'error' && !data.plan) {
    throw new AnalysisError(data.message || 'Query could not be processed.', 'ANALYSIS');
  }

  return data;
}

// ── Hook ────────────────────────────────────────────────────────

export function useAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    step: 'idle',
    query: '',
    plan: null,
    scenes: [],
    result: null,
    error: null,
    errorCode: null,
    detail: null,
    processingSteps: [],
  });

  const reset = useCallback(() => {
    setState({
      step: 'idle',
      query: '',
      plan: null,
      scenes: [],
      result: null,
      error: null,
      errorCode: null,
      detail: null,
      processingSteps: [],
    });
  }, []);

  const analyze = useCallback(async (query: string, overrides?: AnalysisOverrides) => {
    setState(prev => ({
      ...prev,
      step: 'planning',
      query,
      error: null,
      detail: 'Parsing your natural language query...',
      processingSteps: [],
    }));

    try {
      await new Promise(r => setTimeout(r, 50));

      setState(prev => ({ ...prev, step: 'searching', detail: 'Querying satellite archives...' }));

      // ── Fetch with automatic retry for cold-start errors ──
      // Fetch with single retry for cold-start recovery
      let data: any = null;
      for (let attempt = 0; attempt <= 1; attempt++) {
        try {
          data = await fetchAnalysis(query, overrides);
          break;
        } catch (err: any) {
          const wrapped = err instanceof AnalysisError ? err : new AnalysisError(err.message, 'UNKNOWN');
          const isColdStart = wrapped.code === 'PYTHON_UNAVAILABLE' || wrapped.code === 'HTTP_503' || wrapped.code === 'TIMEOUT' || wrapped.code === 'UPSTREAM_TIMEOUT' || wrapped.code === 'UPSTREAM_UNAVAILABLE';
          if (attempt === 0 && isColdStart) {
            setState(prev => ({
              ...prev,
              step: 'searching',
              detail: 'Engine waking up — retrying in 10s...',
            }));
            await new Promise(r => setTimeout(r, 10_000));
            continue;
          }
          throw wrapped;
        }
      }
      if (!data) {
        throw new AnalysisError('No response from analysis engine.', 'UNKNOWN');
      }

      // ── Process successful response ───────────────────────
      const plan = data.plan;
      if (!plan) {
        throw new AnalysisError('No analysis plan generated. Please provide a location and phenomenon.', 'ANALYSIS');
      }

      const planDetail = [
        plan.phenomenon && `Phenomenon: ${plan.phenomenon.replace(/_/g, ' ')}`,
        plan.aoi && `Location: ${plan.aoi}`,
        plan.start_date && plan.end_date && `Period: ${plan.start_date} — ${plan.end_date}`,
      ].filter(Boolean).join(' · ');

      setState(prev => ({ ...prev, step: 'ranking', plan, detail: planDetail || 'Plan created' }));

      const result: TemporalComparisonResult = data.result;
      const backendSteps = result?.processing_steps || [];
      const sceneCount = [result?.scene_t1, result?.scene_t2].filter(Boolean).length;

      setState(prev => ({
        ...prev,
        step: 'processing',
        detail: sceneCount > 0
          ? `Selected ${sceneCount} observation${sceneCount > 1 ? 's' : ''} for analysis`
          : 'Processing observations...',
        processingSteps: backendSteps,
      }));

      await new Promise(r => setTimeout(r, 100));

      const changedPct = result?.metrics?.changed_pct;
      setState(prev => ({
        ...prev,
        step: 'deciding',
        detail: changedPct > 0
          ? `Change detected: ${typeof changedPct === 'number' ? changedPct.toFixed(1) : changedPct}% of area`
          : 'Running change detection...',
      }));

      await new Promise(r => setTimeout(r, 80));

      const findingCount = result?.explanation?.key_findings?.length || 0;
      setState(prev => ({
        ...prev,
        step: 'explaining',
        detail: findingCount > 0
          ? `Generated ${findingCount} key finding${findingCount > 1 ? 's' : ''}`
          : 'Generating analysis summary...',
      }));

      await new Promise(r => setTimeout(r, 80));

      const scenes: SceneInfo[] = [];
      if (result.scene_t1) scenes.push(result.scene_t1);
      if (result.scene_t2) scenes.push(result.scene_t2);

      setState(prev => ({
        ...prev,
        step: 'complete',
        scenes,
        result,
        detail: null,
      }));

      if (data.fallback) {
        console.warn('[OrbitalQuery] Running in degraded mode — Python analysis engine unavailable.');
      }

    } catch (err: any) {
      // Map error codes to user-friendly messages
      let errorMessage = err.message || 'Analysis failed';
      const code = err.code ?? null;

      if (code === 'TIMEOUT') {
        errorMessage =
          'Request timed out. The analysis engine may be experiencing high load. Please try again.';
      } else if (code === 'NETWORK') {
        errorMessage =
          'Could not reach the analysis server. Please try again.';
      } else if (code === 'PYTHON_UNAVAILABLE' || code === 'HTTP_503') {
        errorMessage =
          'The analysis engine is currently unavailable. Showing results from local dataset catalog.';
      }

      setState(prev => ({
        ...prev,
        step: 'error',
        error: errorMessage,
        errorCode: code,
        detail: null,
      }));
    }
  }, []);

  return { state, analyze, reset };
}
