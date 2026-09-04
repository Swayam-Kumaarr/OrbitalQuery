#!/bin/sh
set -e
export PORT=${PORT:-8080}
# Render free tier: 512MB RAM limit
# Single worker, no reload, limited threads
# GC threshold lowered to reclaim memory more aggressively
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
echo "Starting uvicorn on 0.0.0.0:${PORT} (single worker, memory-optimized)"
python -m uvicorn app.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --workers 1 \
  --limit-concurrency 2 \
  --timeout-keep-alive 300 \
  --log-level info
