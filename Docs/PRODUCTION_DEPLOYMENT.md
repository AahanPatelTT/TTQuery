# 🚀 Production Deployment Guide

## ✅ Production-Ready Status

The Synapse MCP Server is now **production-ready** with the following improvements:

### 1. Production WSGI Server
- ✅ Integrated Gunicorn as production WSGI server
- ✅ Configurable worker processes and timeouts
- ✅ Automatic fallback to Flask dev server if Gunicorn unavailable
- ✅ Environment-based configuration

### 2. Docker Support
- ✅ Multi-stage Dockerfile for optimized builds
- ✅ Docker Compose configuration
- ✅ Health checks and monitoring
- ✅ Non-root user for security
- ✅ Volume mounts for persistence

### 3. Enhanced Configuration
- ✅ Production mode flag (`--production`)
- ✅ Configurable via environment variables
- ✅ Updated startup scripts
- ✅ Comprehensive logging

## 📋 Quick Start

### Local Production Deployment

1. **Install dependencies** (including Gunicorn):
```bash
pip install -r requirements.txt
```

2. **Start server in production mode**:
```bash
./start_mcp_server.sh --production
```

Or directly:
```bash
python mcp_server.py --transport both --production
```

### Docker Deployment

1. **Using Docker Compose** (recommended):
```bash
# Create .env file with your configuration
docker-compose up -d
```

2. **Using Docker directly**:
```bash
docker build -t synapse-mcp-server .
docker run -d -p 8880:8880 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/sessions:/app/sessions \
  synapse-mcp-server
```

## ⚙️ Configuration

### Environment Variables

**Production Server (Gunicorn):**
- `GUNICORN_WORKERS`: Number of worker processes (default: 4)
- `GUNICORN_TIMEOUT`: Request timeout in seconds (default: 120)
- `LOG_LEVEL`: Logging level (default: info)

**Application (Optional - not required for MCP server):**
- `LITELLM_API_KEY`: API key for LiteLLM proxy (only needed for chat.py/LLM features)
- `LITELLM_BASE_URL`: Base URL for LiteLLM proxy (only needed for chat.py/LLM features)
- `LITELLM_MODEL`: LLM model (default: gemini/gemini-2.5-pro)
- `EMBED_MODEL`: Embedding model (default: BAAI/bge-large-en-v1.5)

### Command-Line Options

```bash
# Production mode with custom workers
python mcp_server.py --transport both --production

# Development mode (Flask dev server)
python mcp_server.py --transport both --debug

# Custom ports
python mcp_server.py --transport both --http-port 8080 --ws-port 8081 --production
```

## 🧪 Testing

### Verify Production Setup

1. **Run test suite**:
```bash
python test_mcp_server.py --quick
```

2. **Health check**:
```bash
curl http://localhost:8880/health
```

3. **Server info**:
```bash
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2024-11-05"}
  }'
```

## 📊 Monitoring

### Logs

- Server logs: `mcp_server.log`
- Docker logs: `docker logs synapse-mcp-server`
- Docker Compose: `docker-compose logs -f`

### Health Checks

- HTTP endpoint: `GET /health`
- Returns: `{"status": "healthy", "timestamp": "..."}`

### Metrics to Monitor

- Request latency
- Worker process health
- Memory usage
- Knowledge base loading times
- Session count

## 🔒 Security Considerations

1. **Non-root execution**: Docker container runs as non-root user
2. **Environment variables**: Sensitive data in environment, not code
3. **CORS**: Configurable CORS for cross-origin requests
4. **Input validation**: All inputs validated and sanitized
5. **Error handling**: No sensitive data exposed in error messages

## 🚀 Deployment Checklist

Before deploying to production:

- [ ] All tests pass (`python test_mcp_server.py`)
- [ ] Environment variables configured
- [ ] Knowledge bases initialized and accessible
- [ ] Gunicorn installed (`pip install gunicorn`)
- [ ] Health check endpoint responding
- [ ] Logs directory writable
- [ ] Volumes properly mounted (Docker)
- [ ] Ports not conflicting
- [ ] Firewall rules configured
- [ ] Reverse proxy configured (if needed)
- [ ] SSL/TLS certificates (if needed)
- [ ] Backup strategy for artifacts/sessions

## 📚 Additional Resources

- [DOCKER_README.md](DOCKER_README.md) - Docker deployment guide
- [Docs/PRODUCTION_CHECKLIST.md](Docs/PRODUCTION_CHECKLIST.md) - Complete production checklist
- [Docs/MCP_SERVER_GUIDE.md](Docs/MCP_SERVER_GUIDE.md) - API documentation
- [TEST_README.md](TEST_README.md) - Testing guide

## 🎯 What Changed

### Before (Development)
- Flask development server (single-threaded)
- Warning about production server
- Basic error handling
- No Docker support

### After (Production)
- ✅ Gunicorn production WSGI server
- ✅ Configurable workers and timeouts
- ✅ Comprehensive Docker support
- ✅ Health checks and monitoring
- ✅ Production-ready configuration
- ✅ Security best practices

## 💡 Tips

1. **Worker Count**: Set `GUNICORN_WORKERS` based on CPU cores (typically 2-4x cores)
2. **Timeout**: Adjust `GUNICORN_TIMEOUT` based on expected query duration
3. **Monitoring**: Set up log aggregation (e.g., ELK, Datadog)
4. **Scaling**: Use Docker Swarm or Kubernetes for horizontal scaling
5. **Backups**: Regularly backup `artifacts/` and `sessions/` directories

---

**Status**: ✅ **PRODUCTION READY**

The server now uses Gunicorn in production mode and is fully containerized with Docker support.

