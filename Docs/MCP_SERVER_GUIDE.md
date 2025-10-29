# 🌐 Synapse MCP Server Guide

This guide explains how to use the **Synapse MCP (Model Context Protocol) Server** to expose all CLI functionality via HTTP and WebSocket transports.

## 🎯 **Overview**

The Synapse MCP Server wraps all existing CLI functionality and exposes it through standardized MCP tools. This enables AI models and external applications to interact with your knowledge base programmatically.

### **Key Features**
- ✅ **All CLI functionality exposed as MCP tools**
- ✅ **HTTP transport with JSON-RPC 2.0**
- ✅ **WebSocket transport for real-time communication**
- ✅ **Session persistence and management**
- ✅ **Knowledge base switching and management**
- ✅ **Comprehensive logging and error handling**
- ✅ **Document processing status monitoring**

## 🚀 **Quick Start**

### **1. Install Dependencies**
```bash
# Install MCP server dependencies
pip install websockets flask flask-cors

# Or install from requirements.txt
pip install -r requirements.txt
```

### **2. Start the MCP Server**
```bash
# Start both HTTP and WebSocket servers (recommended)
./start_mcp_server.sh

# Or start manually with specific options
python mcp_server.py --transport both --http-port 3000 --ws-port 3001

# Start only HTTP server
python mcp_server.py --transport http --port 3000

# Start only WebSocket server  
python mcp_server.py --transport websocket --port 3001
```

### **3. Test the Connection**
```bash
# Test with example client
python mcp_client_example.py --demo

# Interactive mode
python mcp_client_example.py --interactive

# WebSocket test
python mcp_client_example.py --transport websocket --port 3001 --interactive

# Run comprehensive test suite
python test_mcp_server.py
```

## 🛠️ **Available MCP Tools**

The MCP server exposes the following tools that mirror all CLI functionality:

### **Core Query Tools**
- **`ask_question`** - Ask questions with conversation context
- **`set_verbose_mode`** - Enable/disable detailed retrieval information
- **`get_server_info`** - Get server status and information

### **Knowledge Base Management**
- **`list_knowledge_bases`** - List all available knowledge bases
- **`switch_knowledge_base`** - Switch to a different knowledge base
- **`get_kb_stats`** - Get detailed statistics about a knowledge base

### **Session Management**
- **`create_session`** - Create new conversation session
- **`load_session`** - Load existing session from file
- **`list_sessions`** - List all available sessions
- **`get_session_history`** - Get conversation history
- **`clear_session_history`** - Clear session history
- **`export_session`** - Export session to JSON file

### **Document Processing**
- **`get_processing_status`** - Get document processing status
- **`initialize_knowledge_base`** - Run knowledge base initialization

## 📡 **Transport Options**

### **HTTP Transport**
```bash
# Default HTTP endpoint
POST http://localhost:3000/mcp

# Server information
GET http://localhost:3000/mcp

# Health check
GET http://localhost:3000/health
```

**Example HTTP Request:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "ask_question",
    "arguments": {
      "question": "What is Synapse?",
      "session_id": "demo_session"
    }
  }
}
```

### **WebSocket Transport**
```bash
# Default WebSocket endpoint
ws://localhost:3001
```

**Example WebSocket Message:**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "list_knowledge_bases",
    "arguments": {}
  }
}
```

## 🔧 **Configuration**

### **Configuration File** (`mcp_config.json`)
```json
{
  "server": {
    "name": "synapse-rag-server",
    "version": "1.0.0"
  },
  "transports": {
    "http": {
      "enabled": true,
      "port": 3000,
      "cors": true
    },
    "websocket": {
      "enabled": true,
      "port": 3001
    }
  },
  "knowledge_base": {
    "artifacts_dir": "artifacts",
    "auto_initialize": true
  },
  "logging": {
    "level": "INFO",
    "file": "mcp_server.log"
  }
}
```

### **Command Line Options**
```bash
python mcp_server.py --help

Options:
  --transport {http,websocket,both}  Transport protocol
  --http-port PORT                   HTTP server port [default: 3000]
  --ws-port PORT                     WebSocket server port [default: 3001]
  --artifacts-dir DIR                Artifacts directory [default: artifacts]
  --debug                            Enable debug mode
  --cors                             Enable CORS for HTTP [default: True]
```

## 💡 **Usage Examples**

### **Basic Query Example**
```python
import asyncio
from mcp_client_example import MCPClient

async def ask_question():
    client = MCPClient("http", "localhost", 3000)
    
    # Initialize
    await client.initialize()
    
    # Ask a question
    response = await client.call_tool("ask_question", {
        "question": "How does the RAG pipeline work?",
        "session_id": "my_session",
        "verbose": True
    })
    
    print(response["result"]["content"][0]["text"])
    await client.close()

asyncio.run(ask_question())
```

### **Knowledge Base Switching Example**
```python
async def switch_kb():
    client = MCPClient("http", "localhost", 3000)
    await client.initialize()
    
    # List available KBs
    kbs = await client.call_tool("list_knowledge_bases", {})
    print("Available KBs:", kbs)
    
    # Switch to a specific KB
    result = await client.call_tool("switch_knowledge_base", {
        "knowledge_base": "Aahan_s_Notes"
    })
    print("Switched:", result)
    
    await client.close()
```

### **Session Management Example**
```python
async def manage_sessions():
    client = MCPClient("websocket", "localhost", 3001)
    await client.initialize()
    
    # Create session
    session = await client.call_tool("create_session", {
        "session_id": "research_session"
    })
    
    # Ask questions in session
    for question in ["What is RISC-V?", "How does it compare to ARM?"]:
        response = await client.call_tool("ask_question", {
            "question": question,
            "session_id": "research_session"
        })
        print(f"Q: {question}")
        print(f"A: {response['result']['content'][0]['text']}")
    
    # Export session
    await client.call_tool("export_session", {
        "session_id": "research_session",
        "output_file": "research_conversation.json"
    })
    
    await client.close()
```

## 📊 **Monitoring and Logging**

### **Server Logs**
The MCP server generates comprehensive logs in `mcp_server.log`:

```
2024-01-15 10:30:15 [MCP-INFO] MCPServer: MCP Server initialized
2024-01-15 10:30:15 [MCP-INFO] MCPServer: Available KBs: ['Aahan_s_Notes', 'Confluence']
2024-01-15 10:30:20 [MCP-INFO] MCPServer: Handling method: tools/call with params: {'name': 'ask_question', 'arguments': {...}}
2024-01-15 10:30:25 [MCP-INFO] MCPServer: WebSocket client connected: ('127.0.0.1', 54321)
```

### **Health Monitoring**
```bash
# Check server health
curl http://localhost:3000/health

# Get server information
curl http://localhost:3000/mcp

# Monitor processing status
python mcp_client_example.py --interactive
> status
```

## 🔒 **Security Considerations**

### **Default Security Settings**
- **CORS enabled** for HTTP transport (configurable)
- **Localhost binding** by default
- **No authentication** (suitable for local development)

### **Production Deployment**
For production use, consider:

```python
# Enable authentication
app.before_request(authenticate_request)

# Restrict origins
CORS(app, origins=["https://your-domain.com"])

# Use HTTPS
app.run(ssl_context='adhoc')

# Bind to specific interface
app.run(host='127.0.0.1')  # localhost only
```

## 🐛 **Troubleshooting**

### **Common Issues**

**1. Connection Refused**
```bash
# Check if server is running
curl http://localhost:3000/health

# Check logs
tail -f mcp_server.log

# Restart server
./start_mcp_server.sh
```

**2. Missing Dependencies**
```bash
# Install WebSocket support
pip install websockets

# Install Flask for HTTP transport
pip install flask flask-cors
```

**3. Knowledge Base Not Found**
```bash
# Initialize knowledge base
python initialize_fast.py

# Check available KBs
python mcp_client_example.py --interactive
> list-kbs
```

**4. Environment Variables**
```bash
# Required environment variables
export LITELLM_API_KEY=your_api_key
export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/
```

### **Debug Mode**
```bash
# Start server in debug mode
python mcp_server.py --debug --transport http

# Enable verbose logging in client
python mcp_client_example.py --demo --verbose
```

## 🔄 **Integration with AI Models**

The MCP server is designed to work with AI models that support the Model Context Protocol:

### **Tool Definitions**
AI models can discover available tools:
```json
{
  "method": "tools/list",
  "result": {
    "tools": [
      {
        "name": "ask_question",
        "description": "Ask a question to the knowledge base",
        "inputSchema": {
          "type": "object",
          "properties": {
            "question": {"type": "string"},
            "session_id": {"type": "string"},
            "verbose": {"type": "boolean"}
          }
        }
      }
    ]
  }
}
```

### **Tool Execution**
AI models can execute tools and get structured responses:
```json
{
  "method": "tools/call",
  "params": {
    "name": "ask_question",
    "arguments": {
      "question": "What are the key features of Synapse?",
      "session_id": "ai_session"
    }
  },
  "result": {
    "content": [
      {
        "type": "text", 
        "text": "{\"answer\": \"Synapse is a RAG system...\", \"sources\": [...]}"
      }
    ]
  }
}
```

## 🚀 **Advanced Usage**

### **Batch Requests**
The HTTP transport supports batch requests:
```json
[
  {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "list_knowledge_bases", "arguments": {}}},
  {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "get_server_info", "arguments": {}}}
]
```

### **Streaming Responses**
For long-running operations, the server supports streaming via Server-Sent Events (SSE):
```bash
GET http://localhost:3000/mcp?stream=true
```

### **Custom Extensions**
Extend the MCP server with custom tools:
```python
# Add custom tool to MCPServer.define_tools()
MCPTool(
    name="custom_analysis",
    description="Run custom analysis on documents", 
    inputSchema={
        "type": "object",
        "properties": {
            "analysis_type": {"type": "string"},
            "parameters": {"type": "object"}
        }
    }
)
```

## 📚 **Additional Resources**

- **[Model Context Protocol Specification](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)**
- **[MCP Client Example Code](mcp_client_example.py)**
- **[Server Configuration Reference](mcp_config.json)**
- **[Production Checklist](PRODUCTION_CHECKLIST.md#code-maintenance--cleanup)**

## 💬 **Support**

For issues with the MCP server:

1. **Check logs**: `mcp_server.log` and console output
2. **Test connectivity**: Use the example client with `--debug`
3. **Verify dependencies**: Ensure all required packages are installed
4. **Check environment**: Verify `LITELLM_API_KEY` and `LITELLM_BASE_URL`
5. **Review configuration**: Check `mcp_config.json` settings

The MCP server provides a complete bridge between the Synapse CLI functionality and external AI systems, enabling seamless integration with any MCP-compatible client.
