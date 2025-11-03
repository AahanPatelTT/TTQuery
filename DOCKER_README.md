# 🐳 Synapse MCP Server - Docker Deployment Guide

This guide explains how to deploy the Synapse MCP Server using Docker for production use.

## 📋 Prerequisites

- Docker Engine 20.10+ or Docker Desktop
- Docker Compose 2.0+ (optional, for easier management)
- Knowledge base artifacts in `./artifacts/` directory
- Environment variables configured (see below)

## 🚀 Quick Start

### Using Docker Compose (Recommended)

1. **Create a `.env` file** (optional - MCP server doesn't require LiteLLM):
```bash
# Optional: Only needed if you plan to use other Synapse components that need LLM
# LITELLM_API_KEY=your_api_key_here
# LITELLM_BASE_URL=http://your-litellm-server:4000
# LITELLM_MODEL=gemini/gemini-2.5-pro

# Recommended configuration
EMBED_MODEL=BAAI/bge-large-en-v1.5
LOG_LEVEL=INFO
GUNICORN_WORKERS=4
GUNICORN_TIMEOUT=120
```

2. **Build and start the container**:
```bash
docker-compose up -d
```

3. **Check status**:
```bash
docker-compose ps
docker-compose logs -f synapse-mcp
```

4. **Access the server**:
- HTTP: `http://localhost:8880/mcp`
- Health: `http://localhost:8880/health`

### Using Docker Directly

1. **Build the image**:
```bash
docker build -t synapse-mcp-server .
```

2. **Run the container**:
```bash
docker run -d \
  --name synapse-mcp \
  -p 8880:8880 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/sessions:/app/sessions \
  synapse-mcp-server
```

## 🔧 Configuration

### Environment Variables

**Note:** LiteLLM configuration is **not required** for the MCP server. The server only performs retrieval and returns context for external LLM generation.

**Optional (for other Synapse components):**
- `LITELLM_API_KEY`: API key for LiteLLM proxy (if using chat.py or other LLM features)
- `LITELLM_BASE_URL`: Base URL for LiteLLM proxy server (if using chat.py or other LLM features)

**Optional (MCP Server):**
- `LITELLM_MODEL`: LLM model to use (default: `gemini/gemini-2.5-pro`)
- `EMBED_MODEL`: Embedding model (default: `BAAI/bge-large-en-v1.5`)
- `LOG_LEVEL`: Logging level (default: `INFO`)
- `GUNICORN_WORKERS`: Number of worker processes (default: `4`)
- `GUNICORN_TIMEOUT`: Request timeout in seconds (default: `120`)

### Volumes

The following directories should be mounted as volumes:

- `./artifacts` → `/app/artifacts`: Knowledge base embeddings and database
- `./sessions` → `/app/sessions`: Chat session files
- `./mcp_server.log` → `/app/mcp_server.log`: Server logs (optional)

## 📊 Monitoring

### Health Checks

The container includes a health check endpoint:
```bash
curl http://localhost:8880/health
```

### View Logs

```bash
# Docker Compose
docker-compose logs -f synapse-mcp

# Docker
docker logs -f synapse-mcp
```

### Container Status

```bash
# Docker Compose
docker-compose ps

# Docker
docker ps | grep synapse-mcp
```

## 🛠️ Development Mode

For development, you may want to disable production mode:

```bash
docker run -d \
  --name synapse-mcp-dev \
  -p 8880:8880 \
  -e LITELLM_API_KEY=your_key \
  -e LITELLM_BASE_URL=http://your-server:4000 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/sessions:/app/sessions \
  synapse-mcp-server \
  python mcp_server.py --http-port 8880 --debug
```

## 🔄 Updating

1. **Stop the container**:
```bash
docker-compose down
# or
docker stop synapse-mcp
```

2. **Rebuild the image**:
```bash
docker-compose build
# or
docker build -t synapse-mcp-server .
```

3. **Start the container**:
```bash
docker-compose up -d
# or
docker start synapse-mcp
```

## 🧪 Testing

After deployment, verify the server is working:

```bash
# Health check
curl http://localhost:8880/health

# Server info
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2024-11-05",
      "capabilities": {"tools": {}},
      "clientInfo": {"name": "test-client", "version": "1.0.0"}
    }
  }'
```

## 🚨 Troubleshooting

### Container won't start

1. Check logs: `docker logs synapse-mcp`
2. Verify environment variables are set
3. Ensure artifacts directory exists and is accessible
4. Check port availability: `netstat -an | grep 8880`

### Health check failing

1. Wait longer (startup can take 30-40 seconds)
2. Check if knowledge bases are loading correctly
3. Verify database file exists in artifacts directory

### Performance issues

1. Increase `GUNICORN_WORKERS` (default: 4)
2. Increase `GUNICORN_TIMEOUT` for long-running queries
3. Monitor resource usage: `docker stats synapse-mcp`

## 📚 Production Best Practices

1. **Use a reverse proxy** (nginx, traefik) for SSL/TLS termination
2. **Set up log rotation** for container logs
3. **Monitor resource usage** and adjust worker count accordingly
4. **Backup volumes** regularly (artifacts, sessions)
5. **Use secrets management** for sensitive environment variables
6. **Enable health checks** in orchestration platforms (Kubernetes, Docker Swarm)

## 🎯 Next Steps

- Review [PRODUCTION_CHECKLIST.md](Docs/PRODUCTION_CHECKLIST.md)
- See [MCP_SERVER_GUIDE.md](Docs/MCP_SERVER_GUIDE.md) for API documentation
- Check [MCP_USAGE_GUIDE.md](Docs/MCP_USAGE_GUIDE.md) for usage examples

