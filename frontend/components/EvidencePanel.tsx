'use client';

import { SceneInfo } from '@/hooks/useAnalysis';

interface EvidencePanelProps {
  scenes: SceneInfo[];
}

export default function EvidencePanel({ scenes }: EvidencePanelProps) {
  if (scenes.length === 0) {
    return (
      <div className="oq-card p-5">
        <h3 className="text-sm font-bold text-oq-50 mb-3">Evidence</h3>
        <p className="text-xs text-oq-300">No scenes discovered yet</p>
      </div>
    );
  }

  const avgCloudCover = scenes.length > 0
    ? scenes.reduce((sum, s) => sum + (s.cloud_cover ?? 0), 0) / scenes.length
    : null;

  return (
    <div className="oq-card p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-oq-50">Evidence</h3>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-lime animate-pulse" />
          <span className="text-[10px] text-oq-300 font-medium">
            {scenes.length} observation{scenes.length !== 1 ? 's' : ''}
            {avgCloudCover !== null && avgCloudCover > 0 ? ` · avg cloud ${avgCloudCover.toFixed(1)}%` : ''}
          </span>
        </div>
      </div>

      <div className="space-y-2">
        {scenes.map((scene, i) => {
          const roleLabel = i === 0 ? 'Period 1 (Before)' : 'Period 2 (After)';
          const cloudColor = (scene.cloud_cover ?? 0) <= 10 ? 'text-lime' : (scene.cloud_cover ?? 0) <= 20 ? 'text-oq-200' : 'text-semantic-warning';

          return (
            <div key={scene.item_id || i} className="flex items-center gap-3 p-3 rounded-xl bg-oq-800/50 border border-oq-700/30">
              <div className="w-8 h-8 rounded-lg bg-lime/10 flex items-center justify-center shrink-0">
                <svg className="w-4 h-4 text-lime" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-[11px] font-medium text-oq-50 truncate">
                  {scene.platform} — {scene.datetime ? new Date(scene.datetime).toLocaleDateString() : '—'}
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className={`text-[9px] ${cloudColor}`}>
                    ☁ {scene.cloud_cover != null ? `${scene.cloud_cover.toFixed(1)}%` : '—'}
                  </span>
                  <span className="text-[9px] text-oq-600">•</span>
                  <span className={`text-[9px] ${i === 0 ? 'text-semantic-before' : 'text-semantic-after'}`}>{roleLabel}</span>
                  {scene.item_id && (
                    <>
                      <span className="text-[9px] text-oq-600">•</span>
                      <span className="text-[8px] text-oq-500 font-mono truncate max-w-[140px]">{scene.item_id}</span>
                    </>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
