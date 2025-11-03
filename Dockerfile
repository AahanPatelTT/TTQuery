# Multi-stage build for Synapse MCP Server
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Final stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 synapse && \
    mkdir -p /app/artifacts /app/sessions && \
    chown -R synapse:synapse /app

# Copy Python packages from builder
COPY --from=builder /root/.local /home/synapse/.local

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=synapse:synapse . .

# Add local bin to PATH for user-installed packages
ENV PATH=/home/synapse/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Switch to non-root user
USER synapse

# Expose ports
EXPOSE 8880

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

# Default command (can be overridden)
CMD ["python", "mcp_server.py", "--http-port", "8880", "--production"]

