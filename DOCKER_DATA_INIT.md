# Docker Data Initialization Guide

## Quick Answers

### Do I need to restart the container when uploading files to Data?

**No restart needed for uploading files**, but:
- Files uploaded to `./Data` folder are immediately visible in the container (volume mount)
- However, **raw files need to be processed** (chunked + embedded) before they become queryable
- After processing completes, you can reload knowledge bases without restarting

### How to initialize new data in the container?

You have **3 options**:

## Option 1: Via MCP Tool (Recommended - No Restart)

Use the MCP server's built-in tools:

```bash
# 1. Process a specific folder from Data directory
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -H "mcp-session-id: init-session" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "initialize_knowledge_base",
      "arguments": {
        "folder": "MyNewFolder"
      }
    }
  }'

# 2. After processing, reload knowledge bases (detects new KBs)
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "reload_knowledge_bases",
      "arguments": {}
    }
  }'
```

**Note:** The `initialize_knowledge_base` tool automatically reloads KBs after successful processing.

## Option 2: Via Docker Exec (Direct)

```bash
# 1. Upload files to ./Data folder (on host)
# Files are immediately visible in container via volume mount

# 2. Execute initialization inside container
docker exec synapse-mcp-server python initialize_fast.py --folder "MyNewFolder" --verbose

# 3. Reload knowledge bases via MCP tool (Option 1, step 2)
# OR restart container to reload
docker restart synapse-mcp-server
```

## Option 3: Process All Folders

```bash
# Process all folders in Data directory
docker exec synapse-mcp-server python initialize_fast.py --verbose

# Then reload KBs
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "reload_knowledge_bases",
      "arguments": {}
    }
  }'
```

## Workflow Summary

```
1. Upload files → ./Data/MyFolder/  (on host)
                 ↓ (volume mount)
2. Files visible → /app/Data/MyFolder/  (in container)

3. Process files → initialize_knowledge_base tool
                 ↓
4. Creates embeddings → /app/artifacts/embedded_MyFolder.jsonl

5. Reload KBs → reload_knowledge_bases tool (or restart)
                 ↓
6. New KB available → Can query immediately
```

## Key Points

- ✅ **No restart needed** after uploading raw files to Data
- ✅ **No restart needed** after processing (use `reload_knowledge_bases` tool)
- ✅ **Restart only needed** if you want to reload KBs the old way
- ✅ The `initialize_knowledge_base` tool auto-reloads KBs after success
- ✅ The `reload_knowledge_bases` tool scans artifacts and updates available KBs

## Troubleshooting

**Check if KBs are loaded:**
```bash
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "list_knowledge_bases",
      "arguments": {}
    }
  }'
```

**Check processing status:**
```bash
curl -X POST http://localhost:8880/mcp \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/call",
    "params": {
      "name": "get_processing_status",
      "arguments": {}
    }
  }'
```

