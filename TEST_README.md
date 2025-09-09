# 🧪 Synapse MCP Server - Testing Guide

This document explains the testing tools available for the Synapse MCP Server implementation.

## 📋 **Testing Tools**

### **1. MCP Server Test Suite** (`test_mcp_server.py`)

Comprehensive automated test suite for validating MCP server functionality.

```bash
# Run all tests
python test_mcp_server.py

# Quick essential tests only
python test_mcp_server.py --quick

# Verbose output with timing
python test_mcp_server.py --verbose

# Run specific test category
python test_mcp_server.py --test essential
python test_mcp_server.py --test core
python test_mcp_server.py --test integration
```

**Test Categories:**

- **Essential** (3 tests): Core imports, server init, environment
- **Core** (3 tests): Message handling, tool schemas, error handling  
- **Integration** (2 tests): Configuration files, knowledge base discovery

**Expected Output:**
```
🚀 Synapse MCP Server Test Suite
============================================================
📋 Running ESSENTIAL tests (3 tests)
✅ All tests passed! MCP server is ready for deployment.
```

### **2. MCP Client Example** (`mcp_client_example.py`)

Interactive client for testing and demonstrating MCP server functionality.

```bash
# Full demo (recommended first test)
python mcp_client_example.py --demo

# Interactive mode for manual testing
python mcp_client_example.py --interactive

# WebSocket transport test
python mcp_client_example.py --transport websocket --demo

# Custom server connection
python mcp_client_example.py --host localhost --port 8080 --demo
```

**Demo Features:**
- Server initialization and tool discovery
- Knowledge base listing and switching
- Conversation with session management
- Processing status monitoring

**Interactive Commands:**
- `ask <question>` - Ask a question
- `list-kbs` - List knowledge bases
- `switch-kb <name>` - Switch knowledge base
- `history` - Show conversation history
- `status` - Show processing status
- `server-info` - Show server information
- `quit` - Exit

## 🚀 **Quick Testing Workflow**

### **Step 1: Environment Setup**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install MCP dependencies if needed
pip install websockets flask flask-cors
```

### **Step 2: Run Test Suite**
```bash
# Quick validation
python test_mcp_server.py --quick

# Expected: 3/3 tests passed
```

### **Step 3: Start MCP Server**
```bash
# In a separate terminal
./launch.sh --mcp

# Server will start on:
# HTTP: http://localhost:3000/mcp
# WebSocket: ws://localhost:3001
```

### **Step 4: Test Client Connection**
```bash
# Run demo in main terminal
python mcp_client_example.py --demo

# Expected: Successful connection and demo completion
```

### **Step 5: Interactive Testing** (Optional)
```bash
# Start interactive mode
python mcp_client_example.py --interactive

# Try commands:
💬 You: ask What is Synapse?
💬 You: list-kbs
💬 You: switch-kb Aahan_s_Notes
💬 You: history
💬 You: quit
```

## 🔍 **Test Coverage**

### **Automated Tests Coverage**

| Component | Test Coverage |
|-----------|---------------|
| **Imports & Dependencies** | ✅ All MCP components |
| **Server Initialization** | ✅ Tool registration, KB discovery |
| **Message Handling** | ✅ JSON-RPC 2.0 protocol |
| **Tool Schemas** | ✅ All 14 tool definitions |
| **Error Handling** | ✅ Invalid requests, error responses |
| **Configuration** | ✅ Config files, startup scripts |
| **Knowledge Bases** | ✅ KB discovery and validation |
| **Environment** | ✅ Required variables, dependencies |

### **Manual Test Scenarios**

The interactive client enables testing of:

- **Conversation Flow**: Multi-turn Q&A with context
- **Session Management**: Create, load, export sessions
- **Knowledge Base Switching**: Real-time KB hot-swapping
- **Error Recovery**: Invalid commands, connection issues
- **Performance**: Response times, large query handling

## ⚡ **Performance Testing**

### **Response Time Benchmarks**
```bash
# Add timing to test runs
python test_mcp_server.py --verbose

# Expected timings:
# Message Handling: < 5s
# Tool Schemas: < 1s  
# Error Handling: < 1s
```

### **Load Testing** (Manual)
```bash
# Start server
./launch.sh --mcp

# Run multiple clients simultaneously
for i in {1..5}; do
  python mcp_client_example.py --demo &
done

# Wait for completion
wait
```

## 🐛 **Common Test Issues**

### **Test Failures**

**1. Import Errors**
```bash
# Error: No module named 'websockets'
pip install websockets flask flask-cors

# Error: No module named 'mcp_server'
# Ensure you're in the Synapse root directory
```

**2. Environment Variable Errors**
```bash
# Error: Missing environment variables
export LITELLM_API_KEY=your_api_key
export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/
```

**3. Knowledge Base Errors**
```bash
# Error: No knowledge bases found
python initialize_fast.py

# Check initialization status
python initialize_fast.py --status
```

### **Connection Issues**

**1. Server Not Running**
```bash
# Check if server is running
curl http://localhost:3000/health

# Start server if needed
./launch.sh --mcp
```

**2. Port Conflicts**
```bash
# Check port usage
lsof -i :3000
lsof -i :3001

# Use different ports
python mcp_server.py --http-port 8080 --ws-port 8081
```

**3. Firewall Issues**
```bash
# Ensure ports are accessible
telnet localhost 3000
telnet localhost 3001
```

## 📊 **Test Results Interpretation**

### **Success Indicators**
- ✅ All automated tests pass (6/6 or specified subset)
- ✅ Demo client connects and completes successfully  
- ✅ Interactive client responds to all commands
- ✅ Server logs show no errors
- ✅ Health check returns "healthy" status

### **Warning Signs**
- ⚠️ Slow response times (>10s for simple queries)
- ⚠️ WebSocket connection failures  
- ⚠️ Missing knowledge bases
- ⚠️ Session management errors
- ⚠️ Tool schema validation failures

### **Critical Issues**
- ❌ Server fails to start
- ❌ Client cannot connect
- ❌ Tool calls return errors
- ❌ Knowledge base switching fails
- ❌ Session persistence broken

## 🔧 **Advanced Testing**

### **Custom Test Scenarios**
```python
# Create custom test script
from mcp_client_example import MCPClient
import asyncio

async def custom_test():
    client = MCPClient("http", "localhost", 3000)
    await client.initialize()
    
    # Test specific functionality
    result = await client.call_tool("ask_question", {
        "question": "Your custom test question",
        "session_id": "test_session"
    })
    
    print(f"Result: {result}")
    await client.close()

asyncio.run(custom_test())
```

### **Integration Testing**
```bash
# Test complete workflow
python test_mcp_server.py --test essential
./launch.sh --mcp &
sleep 5
python mcp_client_example.py --demo
pkill -f mcp_server.py
```

### **Stress Testing**
```bash
# Test with many concurrent requests
for i in {1..10}; do
  python -c "
import asyncio
from mcp_client_example import MCPClient

async def stress_test():
    client = MCPClient('http', 'localhost', 3000)
    await client.initialize()
    for j in range(10):
        await client.call_tool('get_server_info', {})
    await client.close()

asyncio.run(stress_test())
" &
done
wait
```

## ✅ **Testing Checklist**

Before deploying the MCP server:

- [ ] All automated tests pass (`python test_mcp_server.py`)
- [ ] Demo client runs successfully (`python mcp_client_example.py --demo`)
- [ ] Interactive client works (`python mcp_client_example.py --interactive`)
- [ ] Both HTTP and WebSocket transports function
- [ ] Knowledge base switching works
- [ ] Session management functions correctly
- [ ] Error handling responds appropriately
- [ ] Server health check returns "healthy"
- [ ] All 14 MCP tools are accessible
- [ ] Performance is within acceptable ranges

**Ready for Production**: All items checked ✅

The testing suite provides comprehensive validation of the MCP server implementation, ensuring reliable operation before deployment to production environments.
