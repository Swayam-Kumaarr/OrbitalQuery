FROM python:3.11-slim

# System dependencies for rasterio, geopandas, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ gfortran make libgomp1 libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (no cache to save image size)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build deps not needed at runtime
RUN apt-get purge -y gcc g++ gfortran make && apt-get autoremove -y

# Copy application
COPY . .

# Remove test files and docs to save space
RUN find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
RUN rm -rf tests/ docs/ .pytest_cache/ venv/ 2>/dev/null || true

# Environment: memory-optimized for Render free tier (512MB)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MALLOC_TRIM_THRESHOLD_=100000
ENV MALLOC_MMAP_THRESHOLD_=100000

RUN chmod +x start.sh 2>/dev/null || true
EXPOSE 8080

CMD ["./start.sh"]
