# Synapse Project Structure

This document outlines the production-ready structure of the Synapse RAG pipeline system.

## 📁 **Repository Structure**

```
Synapse/
├── 📁 Core Application
│   ├── chat.py                     # Main interactive chat interface (CLI + Web GUI)
│   ├── initialize.py               # Standard initialization pipeline (legacy)
│   ├── initialize_fast.py          # Fast incremental initialization (recommended)
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

## 🎯 **MCP Integration Points**

### **CLI Interface** (`chat.py`)
```python
# Primary MCP integration target
python chat.py --question "Your question" --kb "knowledge_base"
```

### **Query API** (`pipeline/query.py`)
```python
# Direct programmatic access
from pipeline.query import answer
result = answer(question, embeddings_path, ...)
```

### **HTTP Endpoints** (`chat.py --test_gui`)
```http
POST /api/ask              # Query endpoint
GET  /api/knowledge-bases  # List available KBs
POST /api/upload          # Upload documents
GET  /api/upload/status   # Processing status
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
