'use client';

import { useEffect, useRef, useState } from 'react';
import type { TemporalComparisonResult, SceneInfo, IndexInfo } from '@/hooks/useAnalysis';
import SwipeMap from '@/components/SwipeMap';
import dynamic from 'next/dynamic';
const RechartsTrendChart = dynamic(() => import('./RechartsTrendChart'), { ssr: false });
import {
  loadSatelliteTiles,
  buildTileJsonUrl,
  parseBbox,
  boundsToLatLng,
} from '@/lib/satellite-tiles';

interface Props {
  result: TemporalComparisonResult;
}

// ── Phenomenon display config ─────────────────────────────────
const PHENOMENON_CONFIG: Record<string, { color: string; label: string; indexLabel: string }> = {
  urban_expansion: { color: '#8B6CF6', label: 'Urban Expansion', indexLabel: 'NDBI' },
  vegetation_change: { color: '#22C55E', label: 'Vegetation Change', indexLabel: 'NDVI' },
  deforestation: { color: '#EF4444', label: 'Deforestation', indexLabel: 'NDVI' },
  flood_impact: { color: '#60A5FA', label: 'Flood Impact', indexLabel: 'NDWI' },
  water_change: { color: '#06B6D4', label: 'Water Body Change', indexLabel: 'NDWI' },
  burn_severity: { color: '#F97316', label: 'Burn Severity', indexLabel: 'NBR' },
  snow_cover: { color: '#CBD5E1', label: 'Snow Cover', indexLabel: 'NDSI' },
  glacier_retreat: { color: '#67E8F9', label: 'Glacier Retreat', indexLabel: 'NDSI' },
  coastal_erosion: { color: '#0EA5E9', label: 'Coastal Erosion', indexLabel: 'NDWI' },
  soil_moisture: { color: '#D97706', label: 'Soil Moisture', indexLabel: 'NDVI' },
  land_cover_change: { color: '#8B6CF6', label: 'Land Cover Change', indexLabel: 'NDVI' },
};

type ViewMode = 'side-by-side' | 'swipe' | 'difference' | 'change-mask';

const GOOGLE_TILE = 'https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}';

// ══════════════════════════════════════════════════════════════
// ── Synchronized Dual Map ───────────────────────────────────
// ══════════════════════════════════════════════════════════════
function SynchronizedDualMap({
  bbox, sceneT1, sceneT2, thumbnailT1, thumbnailT2, signedTileUrl1, signedTileUrl2, tilejsonUrl1, tilejsonUrl2, sceneBboxT1, sceneBboxT2,
}: {
  bbox: number[];
  sceneT1: SceneInfo | null;
  sceneT2: SceneInfo | null;
  thumbnailT1?: string;
  thumbnailT2?: string;
  signedTileUrl1?: string;
  signedTileUrl2?: string;
  tilejsonUrl1?: string;
  tilejsonUrl2?: string;
  sceneBboxT1?: any;
  sceneBboxT2?: any;
}) {
  const leftRef = useRef<HTMLDivElement>(null);
  const rightRef = useRef<HTMLDivElement>(null);
  const leftMapRef = useRef<any>(null);
  const rightMapRef = useRef<any>(null);
  const syncingRef = useRef(false);
  const [leftLoading, setLeftLoading] = useState(true);
  const [rightLoading, setRightLoading] = useState(true);

  useEffect(() => {
    if (!leftRef.current || !rightRef.current || leftMapRef.current) return;
    const leftRect = leftRef.current.getBoundingClientRect();
    if (leftRect.width < 10 || leftRect.height < 10) return;
    let cancelled = false;

    import('leaflet').then(async (L) => {
      if (cancelled || !leftRef.current || !rightRef.current) return;

      // Fetch TileJSON to get correct tile template + bounds for each scene
      async function fetchTilejson(url: string): Promise<{ tileTemplate: string; bounds: number[] } | null> {
        try {
          const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
          if (!r.ok) return null;
          const tj = await r.json();
          return { tileTemplate: tj.tiles?.[0] || '', bounds: tj.bounds || [] };
        } catch { return null; }
      }

      const [tj1, tj2] = await Promise.all([
        tilejsonUrl1 ? fetchTilejson(tilejsonUrl1) : Promise.resolve(null),
        tilejsonUrl2 ? fetchTilejson(tilejsonUrl2) : Promise.resolve(null),
      ]);

      // Use TileJSON bounds for init, fallback to scene bbox, then AOI
      const initBounds = (tj1?.bounds && tj1.bounds.length === 4 ? tj1.bounds : null)
        || parseBbox(sceneBboxT1) || parseBbox(sceneBboxT2) || bbox;
      const [west, south, east, north] = initBounds;
      const center: [number, number] = [(south + north) / 2, (west + east) / 2];
      const latDiff = north - south;
      const lngDiff = east - west;
      const initZoom = Math.min(12, Math.max(6, Math.floor(Math.log2(360 / Math.max(latDiff, lngDiff)))));

      const leftMap = L.map(leftRef.current, { center, zoom: initZoom, zoomControl: false, attributionControl: false });
      L.tileLayer(GOOGLE_TILE, { maxZoom: 22, subdomains: ['0', '1', '2', '3'] }).addTo(leftMap);

      const rightMap = L.map(rightRef.current, { center, zoom: initZoom, zoomControl: false, attributionControl: false });
      L.tileLayer(GOOGLE_TILE, { maxZoom: 22, subdomains: ['0', '1', '2', '3'] }).addTo(rightMap);

      // Use TileJSON tile templates if available (they work without signing)
      const tileUrl1 = tj1?.tileTemplate || signedTileUrl1;
      const tileUrl2 = tj2?.tileTemplate || signedTileUrl2;

      const [leftResult, rightResult] = await Promise.all([
        loadSatelliteTiles(leftMap, { L, signedTileUrl: tileUrl1, tilejsonUrl: tilejsonUrl1, sceneCollection: sceneT1?.collection, sceneItemId: sceneT1?.item_id, thumbnailUrl: thumbnailT1, sceneBbox: tj1?.bounds?.length === 4 ? tj1.bounds : sceneBboxT1, aoiBbox: bbox, opacity: 0.9 }),
        loadSatelliteTiles(rightMap, { L, signedTileUrl: tileUrl2, tilejsonUrl: tilejsonUrl2, sceneCollection: sceneT2?.collection, sceneItemId: sceneT2?.item_id, thumbnailUrl: thumbnailT2, sceneBbox: tj2?.bounds?.length === 4 ? tj2.bounds : sceneBboxT2, aoiBbox: bbox, opacity: 0.9 }),
      ]);

      if (cancelled) return;
      setLeftLoading(false);
      setRightLoading(false);

      // AOI rectangle on both maps
      const aoiBounds = boundsToLatLng(bbox);
      L.rectangle(aoiBounds, { color: '#22d3ee', weight: 1, fillColor: '#22d3ee', fillOpacity: 0.03, dashArray: '6 3' }).addTo(leftMap);
      L.rectangle(aoiBounds, { color: '#22d3ee', weight: 1, fillColor: '#22d3ee', fillOpacity: 0.03, dashArray: '6 3' }).addTo(rightMap);

      const fitBounds = leftResult.hasImagery ? leftResult.bounds : rightResult.bounds;
      leftMap.fitBounds(fitBounds, { padding: [20, 20] });
      rightMap.fitBounds(fitBounds, { padding: [20, 20] });

      setTimeout(() => { leftMap.invalidateSize(); rightMap.invalidateSize(); }, 100);

      // Synchronize navigation
      leftMap.on('move', () => {
        if (syncingRef.current) return;
        syncingRef.current = true;
        rightMap.setView(leftMap.getCenter(), leftMap.getZoom(), { animate: false });
        syncingRef.current = false;
      });
      rightMap.on('move', () => {
        if (syncingRef.current) return;
        syncingRef.current = true;
        leftMap.setView(rightMap.getCenter(), rightMap.getZoom(), { animate: false });
        syncingRef.current = false;
      });

      leftMapRef.current = leftMap;
      rightMapRef.current = rightMap;
    });

    return () => {
      cancelled = true;
      leftMapRef.current?.remove();
      rightMapRef.current?.remove();
      leftMapRef.current = null;
      rightMapRef.current = null;
    };
  }, [bbox, thumbnailT1, thumbnailT2, tilejsonUrl1, tilejsonUrl2, signedTileUrl1, signedTileUrl2, sceneBboxT1, sceneBboxT2]);

  return (
    <div className="w-full grid grid-cols-2 gap-[2px]" style={{ height: 'clamp(400px, 65vh, 700px)' }}>
      <div className="relative rounded-l-lg overflow-hidden bg-oq-950">
        <div ref={leftRef} className="absolute inset-0" />
        <div className="absolute top-2 left-2 z-[1000] px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-widest bg-oq-950/80" style={{ color: '#60A5FA', border: '1px solid rgba(96,165,250,0.2)' }}>Before</div>
        {leftLoading && (
          <div className="absolute inset-0 z-[1000] flex items-center justify-center bg-oq-950/60">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-oq-900/90 border border-oq-700/30">
              <svg className="animate-spin h-3 w-3 text-lime" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
              <span className="text-[9px] text-oq-200">Loading imagery...</span>
            </div>
          </div>
        )}
        <div className="absolute bottom-2 right-2 z-[1000] flex flex-col rounded overflow-hidden border border-oq-700/30" style={{ padding: 12 }}>
          <button onClick={() => leftMapRef.current?.zoomIn()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime hover:bg-oq-800/80 transition-colors text-xs font-bold bg-oq-950/70 backdrop-blur-sm rounded">+</button>
          <div className="h-px bg-oq-700/30 my-0.5" />
          <button onClick={() => leftMapRef.current?.zoomOut()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime hover:bg-oq-800/80 transition-colors text-xs font-bold bg-oq-950/70 backdrop-blur-sm rounded">−</button>
        </div>
      </div>
      <div className="relative rounded-r-lg overflow-hidden bg-oq-950">
        <div ref={rightRef} className="absolute inset-0" />
        <div className="absolute top-2 left-2 z-[1000] px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-widest bg-oq-950/80" style={{ color: '#FB923C', border: '1px solid rgba(251,146,60,0.2)' }}>After</div>
        {rightLoading && (
          <div className="absolute inset-0 z-[1000] flex items-center justify-center bg-oq-950/60">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-oq-900/90 border border-oq-700/30">
              <svg className="animate-spin h-3 w-3 text-lime" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
              <span className="text-[9px] text-oq-200">Loading imagery...</span>
            </div>
          </div>
        )}
        <div className="absolute bottom-2 right-2 z-[1000] flex flex-col rounded overflow-hidden border border-oq-700/30" style={{ padding: 12 }}>
          <button onClick={() => rightMapRef.current?.zoomIn()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime hover:bg-oq-800/80 transition-colors text-xs font-bold bg-oq-950/70 backdrop-blur-sm rounded">+</button>
          <div className="h-px bg-oq-700/30 my-0.5" />
          <button onClick={() => rightMapRef.current?.zoomOut()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime hover:bg-oq-800/80 transition-colors text-xs font-bold bg-oq-950/70 backdrop-blur-sm rounded">−</button>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ── Difference View ─────────────────────────────────────────
// ══════════════════════════════════════════════════════════════
function DifferenceView({
  bbox, changeDetection, config, metrics,
  sceneT1, sceneT2, thumbnailT1, thumbnailT2, signedTileUrl1, signedTileUrl2, sceneBboxT1, sceneBboxT2,
  changeVisualizations, tilejsonUrl1, tilejsonUrl2,
}: {
  bbox: number[];
  changeDetection: Record<string, any> | null;
  config: typeof PHENOMENON_CONFIG[string];
  metrics: Record<string, any>;
  sceneT1?: SceneInfo | null;
  sceneT2?: SceneInfo | null;
  thumbnailT1?: string;
  thumbnailT2?: string;
  signedTileUrl1?: string;
  signedTileUrl2?: string;
  sceneBboxT1?: any;
  sceneBboxT2?: any;
  changeVisualizations?: Record<string, any> | null;
  tilejsonUrl1?: string;
  tilejsonUrl2?: string;
}) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<any>(null);
  const [viewMode, setViewMode] = useState<'blend' | 'diff'>('blend');
  const [overlayOpacity, setOverlayOpacity] = useState(0.5);
  const overlayLayerRef = useRef<any>(null);
  const diffLayerRef = useRef<any>(null);

  // Decode the backend difference PNG
  const decodeVis = (hex: string | null): string | null => {
    if (!hex) return null;
    try {
      const bytes = new Uint8Array(hex.match(/.{1,2}/g)!.map(b => parseInt(b, 16)));
      return URL.createObjectURL(new Blob([bytes], { type: 'image/png' }));
    } catch { return null; }
  };

  const diffPngUrl = decodeVis(changeVisualizations?.difference_png || null);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;
    let cancelled = false;

    import('leaflet').then(async (L) => {
      if (cancelled || !mapRef.current) return;

      // Fetch TileJSON for correct bounds
      async function fetchTilejson(url: string): Promise<{ tileTemplate: string; bounds: number[] } | null> {
        try {
          const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
          if (!r.ok) return null;
          const tj = await r.json();
          return { tileTemplate: tj.tiles?.[0] || '', bounds: tj.bounds || [] };
        } catch { return null; }
      }

      const [tj1, tj2] = await Promise.all([
        tilejsonUrl1 ? fetchTilejson(tilejsonUrl1) : Promise.resolve(null),
        tilejsonUrl2 ? fetchTilejson(tilejsonUrl2) : Promise.resolve(null),
      ]);

      const initBounds = (tj1?.bounds && tj1.bounds.length === 4 ? tj1.bounds : null)
        || parseBbox(sceneBboxT1) || parseBbox(sceneBboxT2) || bbox;
      const [west, south, east, north] = initBounds;
      const center: [number, number] = [(south + north) / 2, (west + east) / 2];

      const tileUrl1 = tj1?.tileTemplate || signedTileUrl1;
      const tileUrl2 = tj2?.tileTemplate || signedTileUrl2;

      const map = L.map(mapRef.current, { center, zoom: 10, zoomControl: false, attributionControl: false });
      L.tileLayer(GOOGLE_TILE, { maxZoom: 22, subdomains: ['0', '1', '2', '3'] }).addTo(map);

      const baseResult = await loadSatelliteTiles(map, { L, signedTileUrl: tileUrl1, tilejsonUrl: tilejsonUrl1, sceneCollection: sceneT1?.collection, sceneItemId: sceneT1?.item_id, thumbnailUrl: thumbnailT1, sceneBbox: tj1?.bounds?.length === 4 ? tj1.bounds : sceneBboxT1, aoiBbox: bbox, opacity: 0.9, zIndex: 400 });
      const overlayResult = await loadSatelliteTiles(map, { L, signedTileUrl: tileUrl2, tilejsonUrl: tilejsonUrl2, sceneCollection: sceneT2?.collection, sceneItemId: sceneT2?.item_id, thumbnailUrl: thumbnailT2, sceneBbox: tj2?.bounds?.length === 4 ? tj2.bounds : sceneBboxT2, aoiBbox: bbox, opacity: overlayOpacity, zIndex: 500 });

      if (cancelled) return;

      if (overlayResult.layer) overlayLayerRef.current = overlayResult.layer;

      const fitBounds = baseResult.hasImagery ? baseResult.bounds : overlayResult.bounds;
      map.fitBounds(fitBounds, { padding: [30, 30], maxZoom: 14 });

      // AOI boundary
      L.rectangle([[south, west], [north, east]], { color: '#22d3ee', weight: 1, fillColor: '#22d3ee', fillOpacity: 0.02, dashArray: '6 4' }).addTo(map);

      mapInstanceRef.current = map;
    });

    return () => { cancelled = true; mapInstanceRef.current?.remove(); mapInstanceRef.current = null; overlayLayerRef.current = null; diffLayerRef.current = null; };
  }, [bbox, signedTileUrl1, signedTileUrl2, tilejsonUrl1, tilejsonUrl2, sceneT1?.item_id, sceneT2?.item_id]);

  useEffect(() => {
    if (overlayLayerRef.current?.setOpacity) overlayLayerRef.current.setOpacity(overlayOpacity);
  }, [overlayOpacity]);

  // Add/remove the difference PNG overlay
  useEffect(() => {
    if (!mapInstanceRef.current || !diffPngUrl) return;
    const map = mapInstanceRef.current;

    if (viewMode === 'diff') {
      if (diffLayerRef.current) { map.removeLayer(diffLayerRef.current); diffLayerRef.current = null; }
      const visBbox = changeVisualizations?.bbox || bbox;
      const bounds: [[number, number], [number, number]] = [[visBbox[1], visBbox[0]], [visBbox[3], visBbox[2]]];
      import('leaflet').then((L) => {
        const leaflet = (L as any).default || L;
        const img = leaflet.imageOverlay(diffPngUrl, bounds as any, { opacity: 0.85, zIndex: 600 }).addTo(map);
        diffLayerRef.current = img;
      });
    } else {
      if (diffLayerRef.current) { map.removeLayer(diffLayerRef.current); diffLayerRef.current = null; }
    }
    return () => { if (diffLayerRef.current) { map.removeLayer(diffLayerRef.current); diffLayerRef.current = null; } };
  }, [viewMode, diffPngUrl, bbox]);

  return (
    <div className="relative w-full rounded-lg overflow-hidden bg-oq-950" style={{ height: 'clamp(400px, 65vh, 700px)' }}>
      <div ref={mapRef} className="absolute inset-0" />
      <div className="absolute top-2 left-2 z-[1000] px-2 py-0.5 rounded text-[8px] font-bold uppercase tracking-widest bg-oq-950/80 text-oq-200 border border-oq-700/30 backdrop-blur-sm">
        {viewMode === 'diff' ? 'Difference — NDVI Overlay' : `Before / After — ${config.indexLabel} Blend`}
      </div>
      {/* View mode toggle */}
      <div className="absolute top-10 left-2 z-[1000] bg-oq-950/85 backdrop-blur-sm rounded border border-oq-700/30 px-2 py-1.5">
        <div className="flex gap-1 mb-1">
          <button
            onClick={() => setViewMode('blend')}
            className={`px-2 py-0.5 rounded text-[8px] font-semibold uppercase ${viewMode === 'blend' ? 'bg-lime text-oq-950' : 'text-oq-300 hover:text-oq-100'}`}
          >Blend</button>
          {diffPngUrl && (
            <button
              onClick={() => setViewMode('diff')}
              className={`px-2 py-0.5 rounded text-[8px] font-semibold uppercase ${viewMode === 'diff' ? 'bg-lime text-oq-950' : 'text-oq-300 hover:text-oq-100'}`}
            >Diff</button>
          )}
        </div>
        {viewMode === 'blend' && (
          <div className="flex items-center gap-1.5">
            <span className="text-[8px] text-semantic-before w-7">Before</span>
            <input type="range" min={0} max={100} value={overlayOpacity * 100} onChange={(e) => setOverlayOpacity(parseInt(e.target.value) / 100)} className="w-16 h-0.5 accent-lime cursor-pointer" />
            <span className="text-[8px] text-semantic-after w-7 text-right">After</span>
          </div>
        )}
        {viewMode === 'diff' && (
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 text-[7px]"><span className="w-2 h-1 rounded-sm" style={{background:'rgba(80,50,50,0.8)'}} /> Decrease</span>
            <span className="flex items-center gap-1 text-[7px]"><span className="w-2 h-1 rounded-sm" style={{background:'rgba(50,50,80,0.8)'}} /> Increase</span>
          </div>
        )}
      </div>
      {/* Zoom */}
      <div className="absolute top-2 right-2 z-[1000] flex flex-col rounded overflow-hidden border border-oq-700/30" style={{ padding: 12 }}>
        <button onClick={() => mapInstanceRef.current?.zoomIn()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime hover:bg-oq-800/80 transition-colors text-xs font-bold bg-oq-950/70 backdrop-blur-sm rounded">+</button>
        <div className="h-px bg-oq-700/30 my-0.5" />
        <button onClick={() => mapInstanceRef.current?.zoomOut()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime hover:bg-oq-800/80 transition-colors text-xs font-bold bg-oq-950/70 backdrop-blur-sm rounded">−</button>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ── Scene Evidence Strip ────────────────────────────────────
// ══════════════════════════════════════════════════════════════
function SceneStrip({ scene, indexStats, label, color }: {
  scene: SceneInfo | null; indexStats: IndexInfo | null; label: string; color: string;
}) {
  if (!scene) return <div className="p-4 rounded-lg text-[10px] text-oq-300 text-center" style={{ background: 'rgba(13,23,17,0.6)', border: '1px solid rgba(42,58,47,0.5)' }}>No scene available</div>;

  const dateStr = scene.datetime ? new Date(scene.datetime).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' }) : '—';

  return (
    <div className="p-4 rounded-lg" style={{ background: 'rgba(13,23,17,0.6)', border: '1px solid rgba(42,58,47,0.5)' }}>
      <div className="flex items-center gap-1.5 mb-3">
        <span className="w-2 h-2 rounded-full" style={{ background: color }} />
        <span className="text-[9px] font-semibold uppercase tracking-wider" style={{ color: '#E5E7EB' }}>{label}</span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-x-4 gap-y-2 mb-3">
        <div><div className="text-[8px] uppercase tracking-wider" style={{ color: '#9CA3AF' }}>Sensor</div><div className="text-[11px] font-medium" style={{ color: '#FFFFFF' }}>{scene.platform || '—'}</div></div>
        <div><div className="text-[8px] uppercase tracking-wider" style={{ color: '#9CA3AF' }}>Date</div><div className="text-[11px] font-medium" style={{ color: '#FFFFFF' }}>{dateStr}</div></div>
        <div><div className="text-[8px] uppercase tracking-wider" style={{ color: '#9CA3AF' }}>Cloud</div><div className="text-[11px] font-mono" style={{ color: '#FFFFFF' }}>{scene.cloud_cover != null ? `${scene.cloud_cover.toFixed(1)}%` : '—'}</div></div>
        <div><div className="text-[8px] uppercase tracking-wider" style={{ color: '#9CA3AF' }}>Collection</div><div className="text-[11px] font-mono truncate" style={{ color: '#FFFFFF' }}>{scene.collection}</div></div>
      </div>
      {indexStats && (
        <div className="pt-3" style={{ borderTop: '1px solid rgba(42,58,47,0.4)' }}>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
            {Object.entries(indexStats.stats).slice(0, 6).map(([key, val]) => (
              <div key={key} className="text-center">
                <div className="text-[7px] uppercase tracking-wider mb-0.5" style={{ color: '#9CA3AF' }}>{key}</div>
                <div className="text-[11px] font-mono font-medium" style={{ color: '#FFFFFF' }}>{typeof val === 'number' ? val.toFixed(4) : String(val)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ── Processing Pipeline ─────────────────────────────────────
// ══════════════════════════════════════════════════════════════
const PIPELINE_STAGES = [
  { num: '01', label: 'Query interpretation' },
  { num: '02', label: 'Area of interest extraction' },
  { num: '03', label: 'Satellite imagery discovery' },
  { num: '04', label: 'Cloud filtering' },
  { num: '05', label: 'Band selection' },
  { num: '06', label: 'Spectral index computation' },
  { num: '07', label: 'Temporal comparison' },
  { num: '08', label: 'Change detection' },
];

function ProcessingPipeline({ steps }: { steps: Array<{ step: string; detail: string }> }) {
  const totalBackendSteps = steps.length;
  const totalFrontendStages = PIPELINE_STAGES.length;
  return (
    <div className={`grid gap-[2px] ${totalFrontendStages <= 8 ? 'grid-cols-2 md:grid-cols-8' : 'grid-cols-3 md:grid-cols-9'}`}>
      {PIPELINE_STAGES.map((stage, i) => {
        // Map backend steps to frontend stages proportionally
        const completed = totalBackendSteps > 0 && i < totalBackendSteps;
        return (
          <div key={stage.num} className={`p-2 rounded text-center ${completed ? 'bg-lime/8 border border-lime/15' : 'bg-oq-800/20 border border-oq-700/10'}`}>
            <div className={`text-[9px] font-mono font-bold mb-0.5 ${completed ? 'text-lime' : 'text-oq-400'}`}>{stage.num}</div>
            <div className={`text-[8px] leading-tight ${completed ? 'text-oq-100' : 'text-oq-400'}`}>{stage.label}</div>
          </div>
        );
      })}
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ── Change Mask View — GeoJSON-primary with sparse PNG ──────
// ══════════════════════════════════════════════════════════════
function ChangeMaskView({
  bbox, sceneT1, sceneT2, thumbnailT1, thumbnailT2, sceneBboxT1, sceneBboxT2,
  changeMaskB64, changeVisBbox, changeVisStats, config, tilejsonUrl1, tilejsonUrl2,
  changeGeojson, regions,
}: {
  bbox: number[];
  sceneT1: SceneInfo | null; sceneT2: SceneInfo | null;
  thumbnailT1?: string; thumbnailT2?: string;
  sceneBboxT1?: any; sceneBboxT2?: any;
  changeMaskB64: string | null;
  changeVisBbox: number[] | null;
  changeVisStats: Record<string, any>;
  config: { color: string; label: string; indexLabel: string };
  tilejsonUrl1?: string;
  tilejsonUrl2?: string;
  changeGeojson?: any;
  regions?: Array<{ region_id: number; area_pixels: number; area_sq_meters: number; direction: string; mean_delta: number; bbox: number[]; centroid: number[] }>;
}) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapRefInst = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedRegion, setSelectedRegion] = useState<any>(null);

  // Derive labels from phenomenon (not hardcoded to vegetation)
  const isUrban = config.label.toLowerCase().includes('urban');
  const decreaseLabel = isUrban ? 'Built-up Decrease' : 'Decrease';
  const increaseLabel = isUrban ? 'Built-up Increase' : 'Increase';

  useEffect(() => {
    if (!mapRef.current || mapRefInst.current) return;
    let cancelled = false;

    import('leaflet').then(async (L) => {
      if (cancelled || !mapRef.current) return;

      async function fetchTilejson(url: string): Promise<{ tileTemplate: string; bounds: number[] } | null> {
        try {
          const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
          if (!r.ok) return null;
          const tj = await r.json();
          return { tileTemplate: tj.tiles?.[0] || '', bounds: tj.bounds || [] };
        } catch { return null; }
      }

      const [tj1] = await Promise.all([
        tilejsonUrl1 ? fetchTilejson(tilejsonUrl1) : Promise.resolve(null),
      ]);

      const initBounds = (tj1?.bounds && tj1.bounds.length === 4 ? tj1.bounds : null)
        || parseBbox(sceneBboxT1) || parseBbox(sceneBboxT2) || bbox;
      const [west, south, east, north] = initBounds;
      const center: [number, number] = [(south + north) / 2, (west + east) / 2];
      const latDiff = north - south;
      const lngDiff = east - west;
      const initZoom = Math.min(12, Math.max(6, Math.floor(Math.log2(360 / Math.max(latDiff, lngDiff)))));

      const tileUrl1 = tj1?.tileTemplate;

      const map = L.map(mapRef.current, { center, zoom: initZoom, zoomControl: false, attributionControl: false });
      L.tileLayer(GOOGLE_TILE, { maxZoom: 22, subdomains: ['0', '1', '2', '3'] }).addTo(map);

      const satResult = await loadSatelliteTiles(map, {
        L, signedTileUrl: tileUrl1, tilejsonUrl: tilejsonUrl1,
        sceneCollection: sceneT1?.collection, sceneItemId: sceneT1?.item_id,
        thumbnailUrl: thumbnailT1,
        sceneBbox: tj1?.bounds?.length === 4 ? tj1.bounds : sceneBboxT1,
        aoiBbox: bbox, opacity: 0.5, zIndex: 300,
      });

      if (cancelled) return;
      setLoading(false);

      // AOI boundary
      const aoiBounds = boundsToLatLng(bbox);
      L.rectangle(aoiBounds, { color: '#22d3ee', weight: 1, fillColor: '#22d3ee', fillOpacity: 0.02, dashArray: '6 3' }).addTo(map);

      // ── Render GeoJSON regions as Leaflet polygons ──────────
      if (changeGeojson?.features?.length > 0) {
        changeGeojson.features.forEach((feature: any) => {
          const coords = feature.geometry?.coordinates;
          const props = feature.properties || {};
          if (!coords || coords.length === 0) return;

          // GeoJSON Polygon coordinates are [lng, lat] pairs
          // Leaflet expects [[lat, lng], ...]
          const latLngs = coords[0].map((c: number[]) => [c[1], c[0]] as [number, number]);

          const dir = props.direction || 'unknown';
          const isIncrease = dir === 'increase';
          const fillColor = isIncrease ? '#22B45A' : '#DC3C3C';
          const fillOpacity = 0.5;

          const polygon = L.polygon(latLngs, {
            color: fillColor,
            weight: 2,
            fillColor: fillColor,
            fillOpacity,
            opacity: 0.9,
          }).addTo(map);

          // Region label
          const regionLabel = `#${props.region_id || '?'} — ${(props.area_sq_meters / 10000).toFixed(1)} ha — ${dir}`;
          polygon.bindTooltip(regionLabel, { className: 'oq-tooltip' });

          // Click to show region details with full provenance
          polygon.on('click', () => {
            setSelectedRegion({
              region_id: props.region_id,
              direction: dir,
              area_pixels: props.area_pixels,
              area_sq_meters: props.area_sq_meters,
              area_ha: (props.area_sq_meters / 10000).toFixed(2),
              area_km2: ((props.area_sq_meters || 0) / 1e6).toFixed(6),
              mean_delta: props.mean_delta,
              max_delta: props.max_delta,
              min_delta: props.min_delta,
              index_name: props.index_name,
              algorithm: props.algorithm,
              threshold: props.threshold,
              ndvi_threshold: props.ndvi_threshold,
              phenomenon: props.phenomenon,
              period_1: props.period_1,
              period_2: props.period_2,
              collection: props.collection,
              provider: props.provider,
              platform: props.platform,
              instrument: props.instrument,
              processing_level: props.processing_level,
              crs: props.crs,
              resolution_meters: props.resolution_meters,
              pixel_area_m2: props.pixel_area_m2,
              scene_ids: props.scene_ids,
              centroid: props.centroid,
              area_calculation: props.area_calculation,
              supporting_indicators: props.supporting_indicators,
              quality_mask: props.quality_mask,
              composite_method: props.composite_method,
            });
          });
        });
      }

      map.fitBounds(satResult.bounds, { padding: [20, 20] });
      setTimeout(() => map.invalidateSize(), 100);
      mapRefInst.current = map;
    });

    return () => { cancelled = true; mapRefInst.current?.remove(); mapRefInst.current = null; };
  }, [bbox, thumbnailT1, sceneBboxT1]);

  // ── Metrics from GeoJSON regions (source of truth) ──────────
  const regionList = regions || changeGeojson?.features?.map((f: any) => f.properties) || [];
  const numRegions = regionList.length;
  const totalRegionAreaSqM = regionList.reduce((sum: number, r: any) => sum + (r.area_sq_meters || 0), 0);
  const totalRegionAreaKm2 = totalRegionAreaSqM / 1e6;
  const decreaseRegions = regionList.filter((r: any) => r.direction === 'decrease');
  const increaseRegions = regionList.filter((r: any) => r.direction === 'increase');
  const decreaseAreaSqM = decreaseRegions.reduce((s: number, r: any) => s + (r.area_sq_meters || 0), 0);
  const increaseAreaSqM = increaseRegions.reduce((s: number, r: any) => s + (r.area_sq_meters || 0), 0);
  const decreaseAreaKm2 = decreaseAreaSqM / 1e6;
  const increaseAreaKm2 = increaseAreaSqM / 1e6;
  const threshold = changeVisStats.threshold ?? changeGeojson?.properties?.threshold ?? 0.15;
  const totalAnalyzed = changeVisStats.total_analyzed_km2 ?? totalRegionAreaKm2;

  const dominantTrend = decreaseAreaSqM > increaseAreaSqM * 1.5 ? 'decrease'
    : increaseAreaSqM > decreaseAreaSqM * 1.5 ? 'increase' : 'stable_mixed';

  const trendLabel = dominantTrend === 'decrease' ? `${decreaseLabel.toUpperCase()} DOMINANT`
    : dominantTrend === 'increase' ? `${increaseLabel.toUpperCase()} DOMINANT`
    : 'STABLE / MIXED';
  const trendColor = dominantTrend === 'decrease' ? '#DC3C3C'
    : dominantTrend === 'increase' ? '#22B45A' : '#9CA3AF';

  return (
    <div className="relative">
      {/* Full-width change mask map */}
      <div className="relative rounded-lg overflow-hidden bg-oq-950" style={{ height: 'clamp(400px, 65vh, 700px)' }}>
        <div ref={mapRef} className="absolute inset-0" />

        {/* Title badge */}
        <div className="absolute top-3 left-3 z-[1000] px-3 py-1 rounded bg-oq-950/85 backdrop-blur-sm border border-oq-700/30">
          <span className="text-[9px] font-bold uppercase tracking-widest" style={{ color: '#10B981' }}>Change Regions</span>
          <span className="text-[8px] text-oq-300 ml-2">{config.indexLabel}-based detection / {numRegions} regions</span>
        </div>

        {/* Legend — phenomenon-aware labels */}
        <div className="absolute top-3 right-12 z-[1000] bg-oq-950/85 backdrop-blur-sm rounded border border-oq-700/30 px-3 py-2">
          <div className="text-[7px] uppercase tracking-wider mb-1.5 font-medium" style={{ color: '#9CA3AF' }}>Legend</div>
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-sm" style={{ background: 'rgba(220,60,60,0.85)' }} />
              <span className="text-[9px] font-medium" style={{ color: '#E5E7EB' }}>{decreaseLabel} ({decreaseRegions.length} regions / {decreaseAreaKm2.toFixed(2)} km²)</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-sm" style={{ background: 'rgba(34,180,90,0.85)' }} />
              <span className="text-[9px] font-medium" style={{ color: '#E5E7EB' }}>{increaseLabel} ({increaseRegions.length} regions / {increaseAreaKm2.toFixed(2)} km²)</span>
            </div>
          </div>
        </div>

        {/* Selected region popup — full provenance */}
        {selectedRegion && (
          <div className="absolute bottom-14 left-3 z-[1000] bg-oq-950/95 backdrop-blur-sm rounded border border-oq-700/30 px-3 py-2.5 max-w-[320px] max-h-[400px] overflow-y-auto">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[10px] font-bold" style={{ color: selectedRegion.direction === 'increase' ? '#22B45A' : '#DC3C3C' }}>
                Region {String(selectedRegion.region_id).padStart(2, '0')}
              </span>
              <button onClick={() => setSelectedRegion(null)} className="text-oq-400 hover:text-white text-[10px]">x</button>
            </div>
            <div className="text-[8px] text-oq-200 space-y-1">
              {/* Location */}
              <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider">Location & Area</div>
              <div>Area: {selectedRegion.area_ha} ha ({selectedRegion.area_km2} km²)</div>
              <div>Pixels: {selectedRegion.area_pixels}</div>
              {selectedRegion.area_calculation && <div className="text-oq-400">{selectedRegion.area_calculation}</div>}
              {selectedRegion.centroid && <div>Centroid: [{selectedRegion.centroid.map((c: number) => c.toFixed(4)).join(', ')}]</div>}

              {/* Signal */}
              <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider pt-1">Signal</div>
              <div>Primary: {selectedRegion.index_name} ({selectedRegion.direction})</div>
              {selectedRegion.mean_delta != null && <div>Mean Δ: {selectedRegion.mean_delta.toFixed(4)}</div>}
              {selectedRegion.max_delta != null && <div>Max Δ: {selectedRegion.max_delta.toFixed(4)}</div>}
              {selectedRegion.supporting_indicators && selectedRegion.supporting_indicators.length > 0 && (
                <div>Supporting: {selectedRegion.supporting_indicators.join(', ')}</div>
              )}

              {/* Period */}
              <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider pt-1">Period</div>
              <div>Baseline: {selectedRegion.period_1 || '—'}</div>
              <div>Comparison: {selectedRegion.period_2 || '—'}</div>

              {/* Dataset */}
              <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider pt-1">Dataset</div>
              <div>Collection: {selectedRegion.collection || '—'}</div>
              {selectedRegion.provider && <div>Provider: {selectedRegion.provider}</div>}
              {selectedRegion.platform && <div>Platform: {selectedRegion.platform}</div>}

              {/* Method */}
              <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider pt-1">Method</div>
              <div>Algorithm: {selectedRegion.algorithm || '—'}</div>
              <div>Threshold: {selectedRegion.threshold != null ? selectedRegion.threshold : '—'}</div>
              {selectedRegion.resolution_meters && <div>Resolution: {selectedRegion.resolution_meters}m</div>}
              {selectedRegion.crs && <div>CRS: {selectedRegion.crs}</div>}

              {/* Scene IDs */}
              {selectedRegion.scene_ids && selectedRegion.scene_ids.length > 0 && (
                <>
                  <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider pt-1">Scenes</div>
                  {selectedRegion.scene_ids.map((sid: string, idx: number) => (
                    <div key={idx} className="text-oq-400 font-mono truncate">{sid}</div>
                  ))}
                </>
              )}

              {/* Quality */}
              {selectedRegion.quality_mask && (
                <>
                  <div className="text-[9px] font-semibold text-oq-100 uppercase tracking-wider pt-1">Quality</div>
                  <div>Mask: {selectedRegion.quality_mask}</div>
                  {selectedRegion.composite_method && <div>Composite: {selectedRegion.composite_method}</div>}
                </>
              )}
            </div>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="absolute inset-0 z-[1000] flex items-center justify-center bg-oq-950/60">
            <div className="flex items-center gap-2 px-3 py-1.5 rounded bg-oq-900/90 border border-oq-700/30">
              <svg className="animate-spin h-3 w-3 text-lime" viewBox="0 0 24 24" fill="none"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
              <span className="text-[9px] text-oq-200">Loading change regions...</span>
            </div>
          </div>
        )}

        {/* Zoom controls */}
        <div className="absolute bottom-3 right-3 z-[1000] flex flex-col rounded overflow-hidden border border-oq-700/30" style={{ padding: 10 }}>
          <button onClick={() => mapRefInst.current?.zoomIn()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime bg-oq-950/70 backdrop-blur-sm rounded text-xs font-bold">+</button>
          <div className="h-px bg-oq-700/30 my-0.5" />
          <button onClick={() => mapRefInst.current?.zoomOut()} className="w-7 h-7 flex items-center justify-center text-oq-200 hover:text-lime bg-oq-950/70 backdrop-blur-sm rounded text-xs font-bold">−</button>
        </div>
      </div>

      {/* Change statistics bar — derived from GeoJSON regions */}
      <div className="mt-3 flex items-stretch gap-3">
        {/* Dominant trend */}
        <div className="flex-shrink-0 px-4 py-3 rounded-lg border border-oq-700/20 bg-oq-800/20">
          <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: '#9CA3AF' }}>Dominant Trend</div>
          <div className="text-[15px] font-bold tracking-tight" style={{ color: trendColor }}>{trendLabel}</div>
        </div>
        {/* Decrease */}
        <div className="flex-1 px-4 py-3 rounded-lg border border-oq-700/20 bg-oq-800/20">
          <div className="flex items-center gap-1.5 mb-1">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#DC3C3C' }} />
            <span className="text-[8px] uppercase tracking-wider" style={{ color: '#9CA3AF' }}>{decreaseLabel}</span>
          </div>
          <div className="text-[18px] font-bold leading-none" style={{ color: '#DC3C3C' }}>{decreaseAreaKm2.toFixed(2)} km²</div>
          <div className="text-[9px] mt-0.5 font-mono" style={{ color: '#68756E' }}>{decreaseRegions.length} regions</div>
        </div>
        {/* Increase */}
        <div className="flex-1 px-4 py-3 rounded-lg border border-oq-700/20 bg-oq-800/20">
          <div className="flex items-center gap-1.5 mb-1">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#22B45A' }} />
            <span className="text-[8px] uppercase tracking-wider" style={{ color: '#9CA3AF' }}>{increaseLabel}</span>
          </div>
          <div className="text-[18px] font-bold leading-none" style={{ color: '#22B45A' }}>{increaseAreaKm2.toFixed(2)} km²</div>
          <div className="text-[9px] mt-0.5 font-mono" style={{ color: '#68756E' }}>{increaseRegions.length} regions</div>
        </div>
        {/* Total */}
        <div className="flex-1 px-4 py-3 rounded-lg border border-oq-700/20 bg-oq-800/20">
          <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: '#9CA3AF' }}>Total Regions</div>
          <div className="text-[18px] font-bold leading-none" style={{ color: '#FFFFFF' }}>{numRegions}</div>
          <div className="text-[9px] mt-0.5 font-mono" style={{ color: '#68756E' }}>{totalRegionAreaKm2.toFixed(2)} km² total / threshold: {typeof threshold === 'number' ? threshold.toFixed(3) : threshold}</div>
        </div>
      </div>
    </div>
  );
}

// ══════════════════════════════════════════════════════════════
// ── Annual Trend Chart ──────────────────────────────────────
// ══════════════════════════════════════════════════════════════
function AnnualTrendChart({ result, config }: { result: TemporalComparisonResult; config: { color: string; indexLabel: string } }) {
  const t1 = result.index_t1;
  const t2 = result.index_t2;
  if (!t1 || !t2) return null;

  const meanT1 = t1.stats?.mean ?? 0;
  const meanT2 = t2.stats?.mean ?? 0;
  const dateT1 = result.scene_t1?.datetime ? new Date(result.scene_t1.datetime).getFullYear() : 0;
  const dateT2 = result.scene_t2?.datetime ? new Date(result.scene_t2.datetime).getFullYear() : 0;
  if (!dateT1 || !dateT2) return null;

  // ONLY real observations — no interpolation between years
  const trendData = [
    { year: String(dateT1), index: parseFloat(meanT1.toFixed(4)), area: 0 },
    { year: String(dateT2), index: parseFloat(meanT2.toFixed(4)), area: parseFloat(((result.metrics?.changed_area_km2 as number) || 0).toFixed(1)) },
  ];

  return (
    <details className="rounded-lg border border-oq-700/15 bg-oq-800/15 overflow-hidden group" open>
      <summary className="px-4 py-2.5 text-[10px] font-semibold text-oq-200 uppercase tracking-wider cursor-pointer hover:text-oq-50 transition-colors flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <svg className="w-3 h-3 text-oq-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z" /></svg>
          Observed Change ({dateT1} → {dateT2})
        </span>
        <svg className="w-3.5 h-3.5 text-oq-300 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
      </summary>
      <div className="px-4 pb-4">
        <div className="mb-2">
          <span className="text-[9px] text-oq-300">Two measured observations — values between are NOT interpolated</span>
        </div>
        <RechartsTrendChart data={trendData} indexLabel={config.indexLabel} />
      </div>
    </details>
  );
}

// ══════════════════════════════════════════════════════════════
// ── MAIN COMPONENT ──────────────────────────────────────────
// ══════════════════════════════════════════════════════════════
export default function TemporalComparisonView({ result }: Props) {
  const config = PHENOMENON_CONFIG[result.phenomenon] || PHENOMENON_CONFIG.land_cover_change;
  const metrics = result.metrics || {};
  const explanation = result.explanation || {};
  const sensorInfo = result.sensor_info || {};
  const [viewMode, setViewMode] = useState<ViewMode>('side-by-side');

  const isFallback = (metrics as any).fallbackMode;

  // ── Extract values from result (never fabricate) ──────────
  const changedArea = metrics.changed_area_km2;
  const changedPct = metrics.changed_pct;
  const deltaIndex = metrics.delta_index;
  const direction = metrics.direction;
  const changedPixels = result.change_detection?.changed_pixels || result.change_detection?.changedPixels || 0;
  const totalPixels = result.change_detection?.total_pixels || 0;

  // Format direction as trend label — phenomenon-aware
  const trendLabel = (() => {
    if (!direction || direction === 'N/A') return null;
    const phenomenon = result.phenomenon || '';
    // Phenomenon-specific direction labels
    // NEVER show vegetation labels for urban queries
    if (phenomenon === 'urban_expansion') {
      const urbanMap: Record<string, string> = {
        increase: 'DETECTED URBAN CHANGE', decrease: 'URBAN DECREASE', mixed: 'MIXED URBAN SIGNAL',
      };
      return urbanMap[direction] || direction.toUpperCase();
    }
    if (phenomenon === 'vegetation_change' || phenomenon === 'deforestation') {
      const vegMap: Record<string, string> = {
        increase: 'VEGETATION GAIN', decrease: 'VEGETATION LOSS', mixed: 'MIXED SIGNAL',
      };
      return vegMap[direction] || direction.toUpperCase();
    }
    if (phenomenon === 'water_change' || phenomenon === 'flood_impact') {
      const waterMap: Record<string, string> = {
        increase: 'WATER EXPANSION', decrease: 'WATER SHRINKING', mixed: 'MIXED SIGNAL',
      };
      return waterMap[direction] || direction.toUpperCase();
    }
    if (phenomenon === 'burn_severity') {
      const burnMap: Record<string, string> = {
        decrease: 'BURN DAMAGE', increase: 'RECOVERY', mixed: 'MIXED SIGNAL',
      };
      return burnMap[direction] || direction.toUpperCase();
    }
    // Generic fallback
    const genericMap: Record<string, string> = {
      increase: 'INCREASE', decrease: 'DECREASE', mixed: 'MIXED SIGNAL',
      stable: 'STABLE',
    };
    return genericMap[direction] || direction.toUpperCase();
  })();

  // ── Has valid imagery? ───────────────────────────────────
  const hasImagery = !isFallback && result.aoi_bbox && result.aoi_bbox.length === 4;

  return (
    <div className="space-y-0" style={{ background: 'var(--color-bg-deep)' }}>

      {/* ════════════════════════════════════════════════════════ */}
      {/* ── MAP AREA (dominant element) ─────────────────────── */}
      {/* ════════════════════════════════════════════════════════ */}
      {hasImagery && (
        <>
          {/* ── Map container with mode switcher overlaid ── */}
          <div className="relative border border-oq-700/20 rounded-lg overflow-hidden">
            {/* Mode switcher — centered at top of map */}
            <div className="absolute top-3 left-1/2 -translate-x-1/2 z-[1010]">
              <div className="inline-flex bg-oq-950/85 backdrop-blur-sm rounded border border-oq-700/30 p-[2px]">
                {(['side-by-side', 'swipe', 'difference', 'change-mask'] as ViewMode[]).map((mode) => (
                  <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-1 rounded text-[9px] font-semibold uppercase tracking-wider transition-all ${
                      viewMode === mode ? 'bg-lime text-oq-950' : 'text-oq-300 hover:text-oq-100 hover:bg-oq-800/50'
                    }`}
                  >
                    {mode === 'side-by-side' && 'Side by Side'}
                    {mode === 'swipe' && 'Swipe'}
                    {mode === 'difference' && 'Difference'}
                    {mode === 'change-mask' && 'Change Mask'}
                  </button>
                ))}
              </div>
            </div>

            {/* Map views */}
          {viewMode === 'side-by-side' && (
            <SynchronizedDualMap
              bbox={result.aoi_bbox}
              sceneT1={result.scene_t1} sceneT2={result.scene_t2}
              thumbnailT1={result.imagery?.period1?.thumbnail} thumbnailT2={result.imagery?.period2?.thumbnail}
              signedTileUrl1={result.imagery?.period1?.tile_url as string}
              signedTileUrl2={result.imagery?.period2?.tile_url as string}
              tilejsonUrl1={result.imagery?.period1?.tilejson as string}
              tilejsonUrl2={result.imagery?.period2?.tilejson as string}
              sceneBboxT1={result.imagery?.period1?.bbox || result.scene_t1?.bbox}
              sceneBboxT2={result.imagery?.period2?.bbox || result.scene_t2?.bbox}
            />
          )}
          {viewMode === 'swipe' && (
            <SwipeMap
              bbox={result.aoi_bbox}
              thumbnailT1={result.imagery?.period1?.thumbnail} thumbnailT2={result.imagery?.period2?.thumbnail}
              signedTileUrl1={result.imagery?.period1?.tile_url as string}
              signedTileUrl2={result.imagery?.period2?.tile_url as string}
              tilejsonUrl1={result.imagery?.period1?.tilejson as string}
              tilejsonUrl2={result.imagery?.period2?.tilejson as string}
              sceneT1={result.scene_t1} sceneT2={result.scene_t2}
              sceneBboxT1={result.imagery?.period1?.bbox || result.scene_t1?.bbox}
              sceneBboxT2={result.imagery?.period2?.bbox || result.scene_t2?.bbox}
            />
          )}
          {viewMode === 'difference' && (
            <DifferenceView
              bbox={result.aoi_bbox}
              changeDetection={result.change_detection} config={config} metrics={metrics}
              sceneT1={result.scene_t1} sceneT2={result.scene_t2}
              thumbnailT1={result.imagery?.period1?.thumbnail} thumbnailT2={result.imagery?.period2?.thumbnail}
              signedTileUrl1={result.imagery?.period1?.tile_url as string}
              signedTileUrl2={result.imagery?.period2?.tile_url as string}
              sceneBboxT1={result.imagery?.period1?.bbox || result.scene_t1?.bbox}
              sceneBboxT2={result.imagery?.period2?.bbox || result.scene_t2?.bbox}
              changeVisualizations={result.change_visualizations || null}
              tilejsonUrl1={result.imagery?.period1?.tilejson as string}
              tilejsonUrl2={result.imagery?.period2?.tilejson as string}
            />
          )}
          {viewMode === 'change-mask' && (
            <ChangeMaskView
              bbox={result.aoi_bbox}
              sceneT1={result.scene_t1} sceneT2={result.scene_t2}
              thumbnailT1={result.imagery?.period1?.thumbnail} thumbnailT2={result.imagery?.period2?.thumbnail}
              sceneBboxT1={result.imagery?.period1?.bbox || result.scene_t1?.bbox}
              sceneBboxT2={result.imagery?.period2?.bbox || result.scene_t2?.bbox}
              changeMaskB64={result.change_visualizations?.change_mask_png || null}
              changeVisBbox={result.change_visualizations?.bbox || null}
              changeVisStats={result.change_visualizations || {}}
              config={config}
              tilejsonUrl1={result.imagery?.period1?.tilejson as string}
              tilejsonUrl2={result.imagery?.period2?.tilejson as string}
              changeGeojson={result.change_visualizations?.change_geojson || result.change_detection?.change_geojson}
              regions={result.change_visualizations?.regions || result.change_detection?.regions}
            />
          )}

          </div>{/* end map container */}

          {/* ── CHANGE SUMMARY (below map) ─── */}
          <div className="mt-4 px-1">
            <div className="max-w-3xl mx-auto">
              {/* Phenomenon label */}
              <div className="mb-2">
                <span className="inline-flex items-center px-2 py-0.5 rounded text-[9px] font-semibold uppercase tracking-wider" style={{ background: `${config.color}15`, color: config.color, border: `1px solid ${config.color}25` }}>
                  {config.label}
                </span>
                <span className="text-[10px] text-oq-300 ml-2 font-mono">{result.aoi_name}</span>
              </div>

              {/* Key metrics row */}
              <div className="flex items-end gap-6 flex-wrap">
                {/* Changed area */}
                {changedArea != null && changedArea > 0 ? (
                  <div>
                    <div className="text-[8px] uppercase tracking-wider font-medium mb-0.5" style={{ color: '#E5E7EB' }}>Change Detected</div>
                    <div className="text-[28px] font-bold leading-none tracking-tight" style={{ color: '#FFFFFF' }}>
                      {typeof changedArea === 'number' ? changedArea.toLocaleString(undefined, { maximumFractionDigits: 0 }) : changedArea} km²
                    </div>
                    {changedPct != null && (
                      <div className="text-[10px] mt-0.5" style={{ color: '#E5E7EB' }}>{typeof changedPct === 'number' ? changedPct.toFixed(1) : changedPct}% of study area</div>
                    )}
                  </div>
                ) : changedPct != null && changedPct > 0 ? (
                  <div>
                    <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-0.5">Change Detected</div>
                    <div className="text-[28px] font-bold text-oq-50 leading-none tracking-tight">{typeof changedPct === 'number' ? changedPct.toFixed(1) : changedPct}%</div>
                    <div className="text-[10px] text-oq-300 mt-0.5">of study area</div>
                  </div>
                ) : (
                  <div>
                    <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-0.5">Status</div>
                    <div className="text-[28px] font-bold text-oq-50 leading-none tracking-tight">STABLE</div>
                    <div className="text-[10px] text-oq-300 mt-0.5">minimal change detected</div>
                  </div>
                )}

                {/* Index change */}
                {deltaIndex != null && (
                  <div>
                    <div className="text-[8px] uppercase tracking-wider font-medium mb-0.5" style={{ color: '#E5E7EB' }}>{config.indexLabel} Change</div>
                    <div className="text-[22px] font-bold leading-none tracking-tight" style={{ color: config.color }}>
                      {deltaIndex > 0 ? '+' : ''}{typeof deltaIndex === 'number' ? deltaIndex.toFixed(2) : deltaIndex}
                    </div>
                  </div>
                )}

                {/* Direction / trend */}
                {trendLabel && (
                  <div>
                    <div className="text-[8px] uppercase tracking-wider font-medium mb-0.5" style={{ color: '#E5E7EB' }}>Detected Trend</div>
                    <div className="text-[16px] font-bold leading-none tracking-tight" style={{ color: '#FFFFFF' }}>{trendLabel}</div>
                  </div>
                )}
              </div>

              {/* Changed pixels (if raster-derived) */}
              {changedPixels > 0 && (
                <div className="mt-2 flex items-center gap-3">
                  <span className="text-[9px] font-mono" style={{ color: '#E5E7EB' }}>{changedPixels.toLocaleString()} changed pixels</span>
                  {totalPixels > 0 && <span className="text-[9px]" style={{ color: '#9CA3AF' }}>/ {totalPixels.toLocaleString()} total</span>}
                  {metrics.raster_derived && <span className="text-[8px] text-lime/70 font-medium">RASTER-DERIVED</span>}
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* ════════════════════════════════════════════════════════ */}
      {/* ── PROGRESSIVE DISCLOSURE SECTIONS ─────────────────── */}
      {/* ════════════════════════════════════════════════════════ */}
      <div className="max-w-[1400px] mx-auto px-6 py-4 space-y-2">

        {/* ── 1. Analysis Summary ──────────────────────────── */}
        <details className="rounded-lg border border-oq-700/15 bg-oq-800/15 overflow-hidden group" open>
          <summary className="px-4 py-2.5 text-[10px] font-semibold text-oq-200 uppercase tracking-wider cursor-pointer hover:text-oq-50 transition-colors flex items-center justify-between">
            <span className="flex items-center gap-1.5">
              <svg className="w-3 h-3 text-oq-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>
              Analysis Summary
            </span>
            <svg className="w-3.5 h-3.5 text-oq-300 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
          </summary>
          <div className="px-4 pb-3 space-y-3">
            {/* Executive summary */}
            {explanation.summary && (
              <p className="text-[11px] text-oq-200 leading-relaxed" style={{ maxWidth: '75ch' }}>{explanation.summary}</p>
            )}
            {/* Key findings */}
            {explanation.key_findings && Array.isArray(explanation.key_findings) && explanation.key_findings.length > 0 && (
              <div className="space-y-1">
                {explanation.key_findings.map((f: string, i: number) => (
                  <div key={i} className="flex items-start gap-1.5 text-[11px] text-oq-200">
                    <span className="mt-0.5 flex-shrink-0" style={{ color: config.color }}>▸</span>
                    <span>{f}</span>
                  </div>
                ))}
              </div>
            )}
            {/* Confidence */}
            {explanation.confidence && (
              <div className="p-2.5 rounded bg-oq-800/30 border border-oq-700/15">
                <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Confidence</div>
                <p className="text-[10px] text-oq-200 leading-relaxed">{explanation.confidence}</p>
              </div>
            )}
            {/* Limitations */}
            {explanation.limitations && explanation.limitations.length > 0 && (
              <div className="flex flex-wrap gap-1">
                {explanation.limitations.map((lim: string, i: number) => (
                  <span key={i} className="text-[8px] text-oq-300 bg-oq-800/30 px-1.5 py-0.5 rounded border border-oq-700/10">{lim}</span>
                ))}
              </div>
            )}
          </div>
        </details>

        {/* ── 2. Scene Evidence ─────────────────────────────── */}
        {!isFallback && (
          <details className="rounded-lg border border-oq-700/15 bg-oq-800/15 overflow-hidden group">
            <summary className="px-4 py-2.5 text-[10px] font-semibold text-oq-200 uppercase tracking-wider cursor-pointer hover:text-oq-50 transition-colors flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <svg className="w-3 h-3 text-oq-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" /></svg>
                Scene Evidence
              </span>
              <svg className="w-3.5 h-3.5 text-oq-300 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
            </summary>
            <div className="px-4 pb-3">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                <SceneStrip scene={result.scene_t1} indexStats={result.index_t1} label="Baseline" color="var(--color-before)" />
                <SceneStrip scene={result.scene_t2} indexStats={result.index_t2} label="Comparison" color="var(--color-after)" />
              </div>
              {/* Change detection details */}
              {result.change_detection && (
                <div className="mt-2 p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                  <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1.5">Change Detection</div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-x-3 gap-y-1">
                    <div><div className="text-[7px] text-oq-300 uppercase">Algorithm</div><div className="text-[10px] text-oq-100 font-mono">{result.change_detection.algorithm || 'difference_threshold'}</div></div>
                    <div><div className="text-[7px] text-oq-300 uppercase">Index</div><div className="text-[10px] text-oq-100 font-mono">{config.indexLabel}</div></div>
                    <div><div className="text-[7px] text-oq-300 uppercase">Changed Pixels</div><div className="text-[10px] text-oq-100 font-mono">{(result.change_detection.changed_pixels || result.change_detection.changedPixels || 0).toLocaleString()}</div></div>
                    <div><div className="text-[7px] text-oq-300 uppercase">Regions</div><div className="text-[10px] text-oq-100 font-mono">{result.change_detection.num_regions || result.change_detection.numRegions || 0}</div></div>
                  </div>
                </div>
              )}
            </div>
          </details>
        )}

        {/* ── 3. Annual Trend ──────────────────────────────── */}
        <AnnualTrendChart result={result} config={config} />

        {/* ── 4. Methodology ────────────────────────────────── */}
        {(explanation.methodology || sensorInfo.index_formula) && (
          <details className="rounded-lg border border-oq-700/15 bg-oq-800/15 overflow-hidden group">
            <summary className="px-4 py-2.5 text-[10px] font-semibold text-oq-200 uppercase tracking-wider cursor-pointer hover:text-oq-50 transition-colors flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <svg className="w-3 h-3 text-oq-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" /></svg>
                Methodology
              </span>
              <svg className="w-3.5 h-3.5 text-oq-300 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
            </summary>
            <div className="px-4 pb-3 space-y-2">
              {explanation.methodology && (
                <p className="text-[11px] text-oq-200 leading-relaxed">{explanation.methodology}</p>
              )}
              <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                <div>
                  <div className="text-[8px] text-oq-300 uppercase">Index</div>
                  <div className="text-[11px] text-oq-100 font-mono">{sensorInfo.index_used || config.indexLabel}</div>
                </div>
                {sensorInfo.index_formula && (
                  <div className="col-span-2">
                    <div className="text-[8px] text-oq-300 uppercase">Formula</div>
                    <div className="text-[11px] text-oq-100 font-mono">{sensorInfo.index_formula}</div>
                  </div>
                )}
                <div>
                  <div className="text-[8px] text-oq-300 uppercase">Resolution</div>
                  <div className="text-[11px] text-oq-100">{sensorInfo.resolution_m || 10}m</div>
                </div>
              </div>
            </div>
          </details>
        )}

        {/* ── 5. Processing Pipeline ────────────────────────── */}
        {result.processing_steps && result.processing_steps.length > 0 && (
          <details className="rounded-lg border border-oq-700/15 bg-oq-800/15 overflow-hidden group">
            <summary className="px-4 py-2.5 text-[10px] font-semibold text-oq-200 uppercase tracking-wider cursor-pointer hover:text-oq-50 transition-colors flex items-center justify-between">
              <span className="flex items-center gap-1.5">
                <svg className="w-3 h-3 text-oq-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" /></svg>
                Processing Pipeline — {result.processing_steps.length} stages
              </span>
              <svg className="w-3.5 h-3.5 text-oq-300 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
            </summary>
            <div className="px-4 pb-3">
              <ProcessingPipeline steps={result.processing_steps} />
            </div>
          </details>
        )}

        {/* ── 6. Provenance Trace (query → pixel → region) ─── */}
        {(() => {
          const prov = result.provenance;
          const consistency = result.change_detection?.consistency_validation;
          if (!prov && !consistency) return null;
          return (
            <details className="rounded-lg border border-oq-700/15 bg-oq-800/15 overflow-hidden group">
              <summary className="px-4 py-2.5 text-[10px] font-semibold text-oq-200 uppercase tracking-wider cursor-pointer hover:text-oq-50 transition-colors flex items-center justify-between">
                <span className="flex items-center gap-1.5">
                  <svg className="w-3 h-3 text-oq-300" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" /></svg>
                  Full Provenance
                </span>
                <svg className="w-3.5 h-3.5 text-oq-300 group-open:rotate-180 transition-transform" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" /></svg>
              </summary>
              <div className="px-4 pb-3 space-y-3">
                {prov && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {/* Query */}
                    <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                      <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Query</div>
                      <div className="text-[10px] text-oq-100">{prov.query?.text || '—'}</div>
                      <div className="text-[9px] text-oq-300 mt-0.5">Phenomenon: {prov.query?.phenomenon || '—'}</div>
                    </div>
                    {/* Dataset */}
                    {prov.dataset && (
                      <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                        <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Dataset</div>
                        <div className="text-[10px] text-oq-100">{prov.dataset.provider} / {prov.dataset.collection}</div>
                        <div className="text-[9px] text-oq-300 mt-0.5">{prov.dataset.instrument} · {prov.dataset.processing_level}</div>
                      </div>
                    )}
                    {/* Grid */}
                    {prov.grid && prov.grid.crs && (
                      <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                        <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Analysis Grid</div>
                        <div className="text-[10px] text-oq-100 font-mono">CRS: {prov.grid.crs}</div>
                        <div className="text-[9px] text-oq-300 mt-0.5">Resolution: {prov.grid.resolution_m}m · Shape: {prov.grid.shape?.[1]}x{prov.grid.shape?.[0]}</div>
                        <div className="text-[9px] text-oq-300">Pixel area: {prov.grid.pixel_area_m2} m²</div>
                        {prov.grid.bounds && prov.grid.bounds.length === 4 && (
                          <div className="text-[8px] text-oq-400 font-mono mt-0.5">Bounds: [{prov.grid.bounds.map((b: number) => b.toFixed(4)).join(', ')}]</div>
                        )}
                        <div className="text-[8px] text-oq-400">Reprojection: {prov.grid.reprojection_method || 'N/A'}</div>
                      </div>
                    )}
                    {/* Detector */}
                    {prov.detector && (
                      <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                        <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Detector</div>
                        <div className="text-[10px] text-oq-100">Method: {prov.detector.method}</div>
                        <div className="text-[9px] text-oq-300 mt-0.5">Min region: {prov.detector.min_region_pixels}px = {prov.detector.min_region_area_m2?.toFixed(0)} m²</div>
                        {prov.detector.thresholds?.threshold != null && (
                          <div className="text-[9px] text-oq-300">Threshold: {prov.detector.thresholds.threshold}</div>
                        )}
                      </div>
                    )}
                    {/* Indices */}
                    {prov.indices && (
                      <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                        <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Indices</div>
                        <div className="text-[10px] text-oq-100">Primary: {prov.indices.primary}</div>
                        {prov.indices.supporting?.length > 0 && (
                          <div className="text-[9px] text-oq-300">Supporting: {prov.indices.supporting.join(', ')}</div>
                        )}
                        {prov.indices.formulas && Object.entries(prov.indices.formulas).map(([name, formula]) => (
                          <div key={name} className="text-[8px] text-oq-400 font-mono mt-0.5">{name}: {String(formula)}</div>
                        ))}
                      </div>
                    )}
                    {/* Quality */}
                    {prov.quality_method && (
                      <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                        <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Quality Mask</div>
                        <div className="text-[10px] text-oq-100">{prov.quality_method.name}</div>
                        <div className="text-[9px] text-oq-300 mt-0.5">Resampling: {prov.quality_method.resampling}</div>
                      </div>
                    )}
                    {/* Results */}
                    {prov.results && (
                      <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                        <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1">Results</div>
                        <div className="text-[10px] text-oq-100 font-mono">Changed: {prov.results.changed_pixel_count?.toLocaleString()} / {prov.results.valid_pixel_count?.toLocaleString()} pixels</div>
                        <div className="text-[10px] text-oq-100">Changed area: {prov.results.changed_area_ha?.toFixed(2)} ha ({prov.results.changed_area_km2?.toFixed(4)} km²)</div>
                        <div className="text-[9px] text-oq-300 mt-0.5">Regions: {prov.results.region_count}</div>
                        {prov.results.area_calculation && (
                          <div className="text-[8px] text-oq-400 font-mono mt-0.5">{prov.results.area_calculation}</div>
                        )}
                      </div>
                    )}
                  </div>
                )}
                {/* Consistency validation */}
                {consistency && (
                  <div className="p-2.5 rounded bg-oq-800/20 border border-oq-700/10">
                    <div className="text-[8px] text-oq-300 uppercase tracking-wider font-medium mb-1.5">Consistency Validation</div>
                    <div className="grid grid-cols-2 gap-2">
                      <div className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${consistency.region_count?.consistent ? 'bg-lime' : 'bg-semantic-error'}`} />
                        <span className="text-[9px] text-oq-200">
                          Region count: {consistency.region_count?.consistent ? 'CONSISTENT' : 'MISMATCH'}
                          {' '}(api={consistency.region_count?.api}, geojson={consistency.region_count?.geojson}, frontend={consistency.region_count?.frontend})
                        </span>
                      </div>
                      <div className="flex items-center gap-1.5">
                        <span className={`w-1.5 h-1.5 rounded-full ${consistency.area_calculation?.consistent ? 'bg-lime' : 'bg-semantic-error'}`} />
                        <span className="text-[9px] text-oq-200">
                          Area calc: {consistency.area_calculation?.consistent ? 'CONSISTENT' : 'MISMATCH'}
                          {' '}(diff: {consistency.area_calculation?.difference_m2?.toFixed(1)} m²)
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </details>
          );
        })()}
      </div>
    </div>
  );
}
