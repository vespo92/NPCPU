# NPCPU - Non-Player Cognitive Processing Unit
# Multi-stage Docker build for production deployment

# =============================================================================
# Stage 1: Base Python image with dependencies
# =============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base as dependencies

# Copy requirements files
COPY requirements.txt requirements-full.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# =============================================================================
# Stage 3: Full dependencies (for development/full features)
# =============================================================================
FROM dependencies as full

RUN pip install --no-cache-dir -r requirements-full.txt

# =============================================================================
# Stage 4: Production image (minimal)
# =============================================================================
FROM dependencies as production

# Create non-root user
RUN useradd --create-home --shell /bin/bash npcpu

# Copy application code
COPY --chown=npcpu:npcpu . .

# Switch to non-root user
USER npcpu

# Create output directories
RUN mkdir -p /app/simulation_output /app/simulation_saves

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import protocols.consciousness; print('OK')" || exit 1

# Default command: run simulation
CMD ["python", "run_simulation.py", "--population", "20", "--ticks", "1000"]

# =============================================================================
# Stage 5: Development image (full features + dev tools)
# =============================================================================
FROM full as development

# Install development dependencies
COPY requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy application code
COPY . .

# Create output directories
RUN mkdir -p /app/simulation_output /app/simulation_saves

# Expose monitoring port
EXPOSE 8765 8080

# Default command for development
CMD ["python", "-m", "pytest", "tests/", "-v"]

# =============================================================================
# Stage 6: Monitoring server image
# =============================================================================
FROM full as monitoring

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash npcpu
USER npcpu

# Expose WebSocket and HTTP ports
EXPOSE 8765 8080

# Run monitoring server
CMD ["python", "-m", "monitoring.server", "--host", "0.0.0.0", "--port", "8765"]
