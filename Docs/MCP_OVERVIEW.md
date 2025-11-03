# 🌐 Synapse MCP Server - Documentation Overview

Welcome to the comprehensive documentation for the **Synapse Model Context Protocol (MCP) Server**. This overview will guide you through all available documentation and help you find exactly what you need.

## 📚 **Documentation Structure**

### **🚀 Getting Started**
- **[MCP Usage Guide](MCP_USAGE_GUIDE.md)** - Complete user guide with examples
  - Quick start and setup
  - All 14 MCP tools explained
  - HTTP transport with FastMCP support (SSE)
  - Client integration examples
  - Session management
  - Troubleshooting

### **🏗️ Technical Deep Dive**  
- **[MCP Architecture](MCP_ARCHITECTURE.md)** - Technical architecture and framework internals
  - System architecture overview
  - Core components and design
  - MCP protocol implementation
  - HTTP transport with SSE streaming
  - FastMCP session management
  - Performance considerations
  - Extension patterns


## 🎯 **Choose Your Path**

### **👨‍💻 I'm a Developer** 
**Want to integrate Synapse with AI models?**

1. **Start here:** [MCP Usage Guide - Quick Start](MCP_USAGE_GUIDE.md#quick-start)
2. **Learn the tools:** [Available MCP Tools](MCP_USAGE_GUIDE.md#available-mcp-tools)
3. **See examples:** [Client Integration](MCP_USAGE_GUIDE.md#client-integration)
4. **Go advanced:** [Advanced Usage Patterns](MCP_USAGE_GUIDE.md#advanced-usage-patterns)

### **🏗️ I'm a System Architect**
**Want to understand the technical architecture?**

1. **Architecture overview:** [MCP Architecture - Overview](MCP_ARCHITECTURE.md#architecture-overview)
2. **Core components:** [Core Components](MCP_ARCHITECTURE.md#core-components)
3. **Protocol details:** [MCP Protocol Implementation](MCP_ARCHITECTURE.md#mcp-protocol-implementation)
4. **Performance:** [Performance Considerations](MCP_ARCHITECTURE.md#performance-considerations)

### **🔧 I'm a DevOps Engineer**
**Want to deploy in production?**

1. **Production deployment:** [Production Deployment](MCP_USAGE_GUIDE.md#production-deployment)
2. **Configuration:** [Server Configuration](MCP_USAGE_GUIDE.md#server-configuration)
3. **Monitoring:** [Monitoring & Observability](MCP_ARCHITECTURE.md#monitoring--observability)
4. **Security:** [Security Architecture](MCP_ARCHITECTURE.md#security-architecture)

### **🧪 I'm Testing the System**
**Want to validate everything works?**

1. **Run tests:** `python test_mcp_server.py` ✅ **All tests passing**
2. **Try demo:** `python mcp_client_example.py --demo`
3. **Interactive mode:** `python mcp_client_example.py --interactive`
4. **HTTP testing:** `curl -X POST http://localhost:3000/mcp -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ask_question", "arguments": {"question": "What is Synapse?"}}}'`
5. **Troubleshooting:** [Troubleshooting Guide](MCP_USAGE_GUIDE.md#troubleshooting)

## 🛠️ **What is the MCP Server?**

The **Synapse MCP Server** is a production-ready implementation of the [Model Context Protocol](https://modelcontextprotocol.io) that exposes all of Synapse's CLI functionality as standardized tools for AI models.

### **Key Benefits**
- ✅ **Complete CLI Exposure**: All 14 CLI functions available as MCP tools
- ✅ **Dual Transport**: HTTP (JSON-RPC 2.0) and WebSocket support
- ✅ **Session Persistence**: Conversation memory and context
- ✅ **Real-time KB Switching**: Hot-swap knowledge bases mid-conversation
- ✅ **Production Ready**: Comprehensive logging, error handling, monitoring
- ✅ **AI Model Integration**: Direct integration with MCP-compatible AI systems

### **Use Cases**
1. **AI Agent Integration**: Enable AI models to query your knowledge base
2. **Conversational RAG**: Multi-turn conversations with memory
3. **Knowledge Base Management**: Programmatic KB switching and management
4. **Document Processing**: Real-time processing status and control
5. **Session Analytics**: Conversation export and analysis

## 📊 **Feature Matrix**

| Feature | CLI | Web GUI | **MCP Server** |
|---------|-----|---------|----------------|
| Question Answering | ✅ | ✅ | ✅ |
| Conversation Memory | ✅ | ✅ | ✅ |
| Knowledge Base Switching | ✅ | ✅ | ✅ |
| Session Management | ✅ | ✅ | ✅ Enhanced |
| Verbose Mode | ✅ | ✅ | ✅ Configurable |
| Export Capabilities | JSON | JSON + UI | ✅ Full API |
| **AI Model Integration** | ❌ | ❌ | ✅ **Native MCP** |
| **Programmatic Access** | ❌ | Limited | ✅ **Full API** |
| **Real-time Communication** | ❌ | HTTP Only | ✅ **WebSocket** |
| **Batch Processing** | ❌ | ❌ | ✅ **Supported** |

## 🚀 **Quick Start Summary**

```bash
# 1. Install dependencies
pip install websockets flask flask-cors

# 2. Start MCP server  
./launch.sh --mcp

# 3. Test connection
python mcp_client_example.py --demo

# 4. Try interactive mode
python mcp_client_example.py --interactive
```

**Server Endpoints:**
- **HTTP**: `POST http://localhost:3000/mcp`
- **WebSocket**: `ws://localhost:3001`
- **Health**: `GET http://localhost:3000/health`

## 🔧 **MCP Tools Overview**

The server exposes **14 MCP tools** organized into 4 categories:

### **🧠 Query Tools (3)**
- `ask_question` - Core RAG with session context
- `set_verbose_mode` - Toggle detailed retrieval  
- `get_server_info` - Server status and config

### **📚 Knowledge Base Tools (3)**
- `list_knowledge_bases` - KB discovery
- `switch_knowledge_base` - Real-time KB switching
- `get_kb_stats` - Detailed KB statistics

### **💾 Session Tools (6)**
- `create_session` / `load_session` - Session lifecycle
- `list_sessions` - Session discovery
- `get_session_history` / `clear_session_history` - History management
- `export_session` - Session persistence

### **⚙️ Processing Tools (2)**
- `get_processing_status` - Real-time processing status
- `initialize_knowledge_base` - KB initialization control

## 📖 **Documentation Quick Links**

### **Getting Started**
- [Quick Start Guide](MCP_USAGE_GUIDE.md#quick-start)
- [Installation & Setup](MCP_USAGE_GUIDE.md#installation--setup)
- [First Steps Tutorial](MCP_USAGE_GUIDE.md#transport-protocols)

### **Development**
- [Client Integration Examples](MCP_USAGE_GUIDE.md#client-integration)
- [Python Client Code](MCP_USAGE_GUIDE.md#python-client-example)
- [JavaScript/Node.js Client](MCP_USAGE_GUIDE.md#javascriptnodejs-client)
- [Tool Reference](MCP_USAGE_GUIDE.md#available-mcp-tools)

### **Architecture & Framework**
- [System Architecture](MCP_ARCHITECTURE.md#architecture-overview)
- [Protocol Implementation](MCP_ARCHITECTURE.md#mcp-protocol-implementation)
- [Transport Layer Design](MCP_ARCHITECTURE.md#transport-layer-architecture)
- [Extension Patterns](MCP_ARCHITECTURE.md#extension--customization)

### **Operations**
- [Production Deployment](MCP_USAGE_GUIDE.md#production-deployment)
- [Docker & Kubernetes](MCP_USAGE_GUIDE.md#docker-deployment)
- [Monitoring & Logging](MCP_ARCHITECTURE.md#monitoring--observability)
- [Troubleshooting](MCP_USAGE_GUIDE.md#troubleshooting)

## 🆘 **Need Help?**

### **Common Questions**

**Q: How do I start the MCP server?**  
A: Run `./launch.sh --mcp` for both HTTP and WebSocket, or see [Server Configuration](MCP_USAGE_GUIDE.md#server-configuration)

**Q: How do I test if it's working?**  
A: Run `python test_mcp_server.py` or `python mcp_client_example.py --demo`

**Q: How do I integrate with my AI model?**  
A: See [Client Integration](MCP_USAGE_GUIDE.md#client-integration) for examples in Python and JavaScript

**Q: How do I switch knowledge bases?**  
A: Use the `switch_knowledge_base` tool - see [Knowledge Base Operations](MCP_USAGE_GUIDE.md#knowledge-base-operations)

**Q: How does session management work?**  
A: See [Session Management](MCP_USAGE_GUIDE.md#session-management) for complete lifecycle details

**Q: What if I get connection errors?**  
A: Check [Troubleshooting Guide](MCP_USAGE_GUIDE.md#troubleshooting) for common issues and solutions

### **More Resources**
- **Test Suite**: `python test_mcp_server.py --help`
- **Example Client**: `python mcp_client_example.py --help`
- **Server Logs**: `tail -f mcp_server.log`
- **Health Check**: `curl http://localhost:3000/health`

## 🎉 **Ready to Go!**

The Synapse MCP Server provides a complete, production-ready bridge between your RAG knowledge base and AI models. Pick your starting point above and dive in!

**Core Team Resources:**
- 📖 **Complete Usage Guide**: [MCP_USAGE_GUIDE.md](MCP_USAGE_GUIDE.md)
- 🏗️ **Technical Architecture**: [MCP_ARCHITECTURE.md](MCP_ARCHITECTURE.md)
- 🚀 **Quick Reference**: [MCP_SERVER_GUIDE.md](MCP_SERVER_GUIDE.md)

---

*The MCP Server transforms Synapse from a powerful local RAG system into a fully programmable knowledge service that AI models can interact with directly. Welcome to the future of AI-powered document intelligence!* 🌟
