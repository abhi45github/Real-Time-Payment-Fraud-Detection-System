# Multi-stage build for optimized production image
FROM python:3.10-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV APP_HOME=/app

# Create app directory
WORKDIR $APP_HOME

# Create non-root user
RUN useradd -m -u 1000 fraud_detector && \
    chown -R fraud_detector:fraud_detector $APP_HOME

# Copy application code
COPY --chown=fraud_detector:fraud_detector . .

# Switch to non-root user
USER fraud_detector

# Expose ports
EXPOSE 8000 9090

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run the application
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--loop", "uvloop"]

# Updated: 2025-01-08 22:00:16 - Enhancement #4529

# Updated: 2025-01-17 10:28:37 - Enhancement #7782

# Updated: 2025-01-25 16:18:00 - Enhancement #7990

# Updated: 2025-01-26 10:26:23 - Enhancement #6351

# Updated: 2025-02-18 16:29:11 - Enhancement #4937

# Updated: 2025-03-05 19:32:34 - Enhancement #3768

# Updated: 2025-03-28 09:20:53 - Enhancement #8197

# Updated: 2025-04-02 21:48:00 - Enhancement #1599

# Updated: 2025-04-03 21:40:41 - Enhancement #8301
