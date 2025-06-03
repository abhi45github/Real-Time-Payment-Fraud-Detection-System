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

# Updated: 2025-04-04 21:09:47 - Enhancement #8813

# Updated: 2025-04-09 20:08:41 - Enhancement #4048

# Updated: 2025-04-17 12:21:31 - Enhancement #1334

# Updated: 2025-04-30 12:58:53 - Enhancement #3240

# Updated: 2025-05-02 22:51:11 - Enhancement #9758

# Updated: 2025-05-14 12:12:09 - Enhancement #6815

# Updated: 2025-05-15 19:48:20 - Enhancement #2846

# Updated: 2025-01-04 08:12:53 - Enhancement #4011

# Updated: 2025-01-07 11:11:44 - Enhancement #9573

# Updated: 2025-01-12 15:59:19 - Enhancement #9172

# Updated: 2025-01-16 11:58:26 - Enhancement #1324

# Updated: 2025-01-20 09:06:01 - Enhancement #1922

# Updated: 2025-02-15 15:42:55 - Enhancement #9260

# Updated: 2025-02-16 21:38:02 - Enhancement #7628

# Updated: 2025-02-16 13:55:21 - Enhancement #5683

# Updated: 2025-02-17 17:49:29 - Enhancement #8894

# Updated: 2025-03-09 14:04:00 - Enhancement #1502

# Updated: 2025-03-12 09:38:42 - Enhancement #1476

# Updated: 2025-03-13 22:30:47 - Enhancement #1196

# Updated: 2025-03-13 16:36:06 - Enhancement #3375

# Updated: 2025-03-16 20:28:58 - Enhancement #9872

# Updated: 2025-03-20 22:37:05 - Enhancement #1290

# Updated: 2025-03-25 15:10:09 - Enhancement #8026

# Updated: 2025-03-25 22:18:50 - Enhancement #5086

# Updated: 2025-04-09 17:13:37 - Enhancement #2189

# Updated: 2025-04-17 08:50:09 - Enhancement #9279

# Updated: 2025-04-18 15:17:05 - Enhancement #7700

# Updated: 2025-04-23 12:04:22 - Enhancement #3030

# Updated: 2025-04-28 20:52:07 - Enhancement #5933

# Updated: 2025-05-17 21:41:05 - Enhancement #8058

# Auto-generated update: 2024-12-03 12:16:27 - Task #965

# Auto-generated update: 2024-12-05 18:40:00 - Task #451

# Auto-generated update: 2024-12-06 13:56:38 - Task #166

# Auto-generated update: 2024-12-24 17:26:13 - Task #766

# Auto-generated update: 2025-01-18 09:20:34 - Task #390

# Auto-generated update: 2025-01-30 21:31:27 - Task #140

# Auto-generated update: 2025-02-11 17:55:34 - Task #237

# Auto-generated update: 2025-02-13 19:32:18 - Task #235

# Auto-generated update: 2025-02-21 11:25:32 - Task #471

# Auto-generated update: 2025-05-11 20:10:31 - Task #731

# Auto-generated update: 2025-06-02 13:35:32 - Task #341

# Auto-generated update: 2025-06-03 19:26:00 - Task #327
