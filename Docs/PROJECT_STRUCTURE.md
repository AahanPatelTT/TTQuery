# Synapse Project Structure

This document outlines the production-ready structure of the Synapse RAG pipeline system.

## 📁 **Repository Structure**

```
Synapse/
├── 📁 Core Application
│   ├── chat.py                     # Main interactive chat interface (CLI + Web GUI)
│   ├── initialize_fast.py          # Fast incremental initialization pipeline
│   ├── mcp_server.py               # MCP server for AI model integration ⭐ **NEW**
│   ├── mcp_client_example.py       # MCP client example and testing tool ⭐ **NEW**
│   ├── test_mcp_server.py          # Comprehensive MCP server test suite ⭐ **NEW**
│   ├── web_upload.py               # Web upload server for real-time processing
│   ├── launch.sh                   # Unified launch script (includes MCP options)
│   └── start_mcp_server.sh         # Dedicated MCP server startup script ⭐ **NEW**
│
├── 📁 Pipeline Components
│   ├── pipeline/
│   │   ├── parse.py               # Document parsing (PDF, PPTX, DOCX, etc.)
│   │   ├── chunk.py               # Text chunking with overlap
│   │   ├── embed.py               # Multi-vector embedding generation
│   │   ├── query.py               # Hybrid search and retrieval
│   │   ├── image_extractor.py     # Image extraction and processing
│   │   ├── database.py            # SQLite database for fast caching
│   │   ├── fast_embed.py          # Real-time embedding service
│   │   └── progress.py            # CLI progress tracking
│
├── 📁 Web Interface
│   ├── templates/
│   │   └── index.html             # Main GUI template
│   └── static/
│       ├── css/main.css           # Modern dark theme styles
│       └── js/main.js             # Frontend JavaScript
│
├── 📁 Configuration
│   ├── requirements.txt           # Python dependencies (includes MCP dependencies)
│   ├── default_config.json       # Default RAG configuration
│   ├── mcp_config.json           # MCP server configuration ⭐ **NEW**
│   ├── litellm_config.yaml       # LiteLLM proxy configuration
│   └── .gitignore                # Git ignore patterns
│
├── 📁 Documentation
│   ├── README.md                  # Main project documentation
│   ├── HOW_IT_WORKS.md           # Technical implementation details
│   ├── GUI.md                    # Web interface documentation
│   ├── CONFIG_SESSION_FEATURES.md # Session management guide
│   ├── FAST_INITIALIZATION_GUIDE.md # Fast processing documentation
│   ├── IMAGE_DEDUPLICATION_GUIDE.md # Image processing guide
│   ├── KNOWLEDGE_BASE_USAGE.md   # Knowledge base management
│   ├── MCP_OVERVIEW.md           # MCP documentation overview ⭐ **NEW**
│   ├── MCP_USAGE_GUIDE.md        # Complete MCP usage guide ⭐ **NEW**
│   ├── MCP_ARCHITECTURE.md       # MCP technical architecture ⭐ **NEW**
│   ├── MCP_SERVER_GUIDE.md       # MCP quick reference ⭐ **NEW**
│   ├── PROJECT_STRUCTURE.md      # This file
│   └── PRODUCTION_CHECKLIST.md   # Production readiness checklist
│
├── 📁 Data & Artifacts (Generated)
│   ├── Data/                     # Input documents (user-provided)
│   ├── artifacts/                # Generated artifacts
│   │   ├── *.jsonl              # Processed data files
│   │   ├── *.db                 # Database files
│   │   └── extracted_images/    # Extracted images
│   └── sessions/                # Chat session history
│
└── 📁 Development & Testing
    ├── .venv/                   # Python virtual environment
    ├── *.log                    # Application logs (including mcp_server.log)
    ├── TEST_README.md           # Testing guide and procedures ⭐ **NEW**
    └── Installer/               # Installation utilities
```

## 🔧 **Core Components**

### **Entry Points**
- `chat.py` - Main application (CLI + GUI)
- `launch.sh` - Unified launcher with all modes
- `web_upload.py` - Dedicated upload server

### **Pipeline Modules**
- `pipeline/parse.py` - Document ingestion and parsing
- `pipeline/chunk.py` - Text chunking and segmentation  
- `pipeline/embed.py` - Vector embedding generation
- `pipeline/query.py` - Search and retrieval engine

### **Advanced Features**
- `pipeline/database.py` - Fast incremental processing
- `pipeline/fast_embed.py` - Real-time embedding service
- `pipeline/image_extractor.py` - Image processing and OCR

### **User Interfaces**
- **CLI**: Interactive command-line chat
- **Web GUI**: Modern web interface with real-time config
- **Upload Interface**: Drag & drop document management

## ✅ **MCP Integration Complete** ⭐ **PRODUCTION READY**

### **MCP Server** (`mcp_server.py`) ✅ **FULLY FUNCTIONAL**
```bash
# Start MCP server
./start_mcp_server.sh                    # Both HTTP + WebSocket
python mcp_server.py --transport both    # Manual start

# Test MCP server
python test_mcp_server.py                # Comprehensive test suite
python mcp_client_example.py --demo      # Interactive demo
```

### **MCP Endpoints** ✅ **TESTED**
```http
# HTTP Transport (JSON-RPC 2.0)
POST http://localhost:8880/mcp           # Main MCP endpoint
GET  http://localhost:8880/health        # Health check

# WebSocket Transport
ws://localhost:3001                      # Real-time MCP communication
```

### **MCP Tools** ✅ **14 TOOLS AVAILABLE**
- **Query Tools**: `ask_question`, `set_verbose_mode`, `get_server_info`
- **Knowledge Base**: `list_knowledge_bases`, `switch_knowledge_base`, `get_kb_stats`
- **Session Management**: `create_session`, `load_session`, `list_sessions`, `get_session_history`, `clear_session_history`, `export_session`
- **Processing**: `get_processing_status`, `initialize_knowledge_base`

### **Legacy Interfaces** (Still Available)
```python
# CLI Interface
python chat.py --question "Your question" --kb "knowledge_base"

# Query API
from pipeline.query import answer
result = answer(question, embeddings_path, ...)

# Web GUI
python chat.py --test_gui
```

## 🔄 **Data Flow**

```
Documents → Parse → Chunk → Embed → Database → Query → Response
    ↓         ↓       ↓       ↓        ↓        ↓        ↓
   JSON     JSONL   JSONL   SQLite   FAISS   LLM    Markdown
```

## 📊 **Production Readiness**

### ✅ **Completed**
- Modular architecture
- Database-driven caching
- Real-time processing
- Web interface
- Documentation
- Error handling
- Progress tracking

### 🔄 **MCP Preparation**
- CLI interface (ready)
- HTTP API (ready)
- Programmatic access (ready)
- Transfer protocols (to be enhanced)

## 🚀 **Next Steps for MCP**

1. **CLI Wrapper**: Create MCP-compatible CLI interface
2. **HTTP Enhancement**: Extend HTTP framework for MCP protocols
3. **Transfer Methods**: Add efficient data transfer mechanisms
4. **Node Integration**: Package as MCP node
