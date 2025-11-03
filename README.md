# Synapse: Enterprise RAG Pipeline

**Intelligent Document Q&A with Conversation Memory and Advanced Retrieval**

Synapse is a complete RAG (Retrieval-Augmented Generation) system that transforms your document collections into an intelligent, conversational knowledge base. Built for engineering teams, it provides precise, cited answers through an intuitive chat interface.

## ✨ **Key Features**

### **🧠 Intelligent Chat Interface**
- **CLI & Web GUI**: Choose between command-line or modern web interface
- **Conversation Memory**: Maintains context across multiple questions
- **Interactive Commands**: Rich command system with session management
- **Verbose Mode**: Shows detailed retrieval and ranking steps
- **Session Export**: Save and resume conversations
- **Real-time Configuration**: Adjust RAG parameters on-the-fly (GUI)
- **Markdown Rendering**: Beautiful formatting for code, tables, and text (GUI)
- **Session-Specific Configs**: Each conversation maintains its own RAG configuration
- **Default Config Management**: Save and load default configurations for new sessions

### **🔍 Advanced Retrieval**
- **Hybrid Search**: Dense (FAISS) + Sparse (BM25) retrieval with RRF fusion
- **Multi-Vector Embeddings**: Summary + full-content vectors for optimal recall
- **Smart Reranking**: Cross-encoder reranking with MMR diversification
- **Precise Citations**: Page-level source attribution with inline references
- **Table-aware prompting**: CSV/table chunks include a compact Markdown table preview (first rows/cols) during generation to preserve structure
- **PPTX-native tables**: Extracts real PPTX tables into CSV at parse time for accurate table retrieval and reconstruction
- **Slide cohesion**: Keeps slides atomic and adds small slide-window chunks for context; prompt stitches multiple chunks from the same deck coherently

### **🖼️ Visual Content Processing** ⭐ **NEW**
- **Image Extraction**: Automatically extracts diagrams, charts, and illustrations from PDFs and PPTXs
- **Image Deduplication**: Intelligent duplicate detection prevents saving the same image multiple times
- **Rich Image Metadata**: OCR text, document context, technical keywords, and categorization
- **Smart Image Retrieval**: Returns relevant diagrams when queries reference visual content
- **Visual Citation**: Full image paths provided for viewing extracted diagrams
- **Technical Diagram Understanding**: Specialized processing for block diagrams, flowcharts, and technical illustrations

### **📚 Multi-Knowledge Base System** ⭐ **NEW**
- **Specialized Knowledge Bases**: Each folder becomes a focused knowledge base
- **Multi-KB Selection**: Select multiple knowledge bases simultaneously (GUI checkboxes)
- **Interactive Switching**: Change knowledge bases mid-conversation (CLI commands)
- **Cross-KB Search**: Search across selected knowledge bases with unified results
- **Smart Result Fusion**: Intelligently combines and attributes results from multiple sources
- **Domain-Specific Filtering**: Focus on specific knowledge domains or search comprehensively
- **Cross-Folder Deduplication**: Images deduplicated across all folders for efficiency

## 🎯 **Quick Start**

### **Prerequisites**
```bash
# macOS dependencies for advanced parsing
brew install libmagic poppler tesseract

# Python 3.9+ required
python3 --version
```

### **🚀 Easy Launch (Automated Setup)** ⭐ **NEW**
```bash
# One-command launch with automatic setup
./launch.sh --gui     # Launch web GUI with auto-setup
./launch.sh --fast    # Fast incremental initialization (recommended)
./launch.sh --status  # Check processing status
./launch.sh           # Launch CLI interface with auto-setup
```

### **1. Manual Installation & Setup**
```bash
# Clone and setup environment
git clone <repository-url>
cd Synapse
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### **2. Environment Configuration** 
Set up your LiteLLM credentials for the TensTorrent proxy:

```bash
# Required for query and chat functionality
export LITELLM_API_KEY=*your_api_key*
export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/

# Optional: specify model (defaults to gemini-2.5-pro)
export LITELLM_MODEL=gemini/gemini-2.5-pro
```

**For persistent setup**, add to your shell profile:
```bash
echo 'export LITELLM_API_KEY=*your_api_key*' >> ~/.zshrc
echo 'export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/' >> ~/.zshrc
source ~/.zshrc
```

### **3. Initialize Knowledge Base**
Add desired files in a simple (.csv/.md) or regular formats like .pdf, .pptx, .docx etc. These files will processed to construct the knowledge base.

```bash
# Single knowledge base (traditional approach)
python initialize_fast.py

# Fast incremental initialization (recommended)
python initialize_fast.py

# Advanced options
python initialize_fast.py --embed-provider openai --verbose --cleanup
```

The initialization script:
- ✅ Validates dependencies and environment
- ✅ Processes documents with progress tracking
- ✅ Uses intelligent caching for efficiency  
- ✅ Provides detailed status and error recovery

### **4. Start Chatting!**

**🚀 Recommended: Use Launch Script** ⭐ **NEW**
```bash
# Fast incremental initialization (recommended for updates)
./launch.sh --fast

# Launch web GUI (handles all setup automatically)
./launch.sh --gui

# Check processing status
./launch.sh --status

# Launch CLI interface (handles all setup automatically) 
./launch.sh
```

**Option A: Command Line Interface (Manual)**
```bash
# Launch interactive CLI chat interface
python chat.py

# With specific knowledge base selection
python chat.py --kb "Aahan_s_Notes"
python chat.py --select-kb

# With session persistence and verbose mode
python chat.py --verbose --session research_session.json
```

**Option B: Web GUI Interface (Manual)** ⭐ **NEW**
```bash
# Launch modern web-based GUI
python chat.py --test_gui

# Then open http://127.0.0.1:7860 in your browser
```

**Option C: MCP Server Interface** ⭐ **NEW** ✅ **PRODUCTION READY**
```bash
# Launch MCP server (for AI model integration)
./launch.sh --mcp              # Both HTTP and WebSocket
./launch.sh --mcp-http         # HTTP only (port 3000)

# Or manually:
python mcp_server.py --transport both --http-port 3000 --ws-port 3001

# Test the MCP server
python tests/test_mcp_server.py      # Comprehensive test suite
python tests/test_fastmcp.py         # FastMCP/SSE tests
python mcp_client_example.py --demo  # Interactive demo

# Quick HTTP test
curl -X POST http://localhost:3000/mcp -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "ask_question", "arguments": {"question": "What is Synapse?"}}}'
```

📖 **[Complete MCP Documentation →](Docs/MCP_OVERVIEW.md)**

📖 **See [CONFIG_SESSION_FEATURES.md](Docs/CONFIG_SESSION_FEATURES.md) for complete web interface documentation**

## ⚡ **Fast Incremental Updates** ⭐ **NEW**

Synapse now includes a **database-driven incremental system** that makes updates **10-100x faster**:

### **🔥 Key Benefits**
- **Only processes changed files** - skips unchanged documents automatically
- **Real-time embedding generation** - background processing for instant updates  
- **Database persistence** - no more slow file scanning
- **Web upload interface** - drag & drop documents for instant processing
- **Status monitoring** - see processing progress in real-time

### **🚀 Usage**
```bash
# Fast incremental initialization (recommended)
./launch.sh --fast

# Check what's being processed
./launch.sh --status

# Advanced: Direct script usage
python initialize_fast.py                    # Process all folders
python initialize_fast.py --folder "MyDocs"  # Process specific folder  
python initialize_fast.py --status           # Show detailed status

# Web upload interface (real-time document processing)
python web_upload.py                         # Start web server at http://localhost:5000
```

### **📊 How It Works**
1. **Document Tracking**: SQLite database tracks file changes (mtime, size, hash)
2. **Incremental Parsing**: Only parses new/modified files
3. **Background Embedding**: Automatic embedding generation in background threads
4. **Smart Caching**: Database-driven caching eliminates slow file scans
5. **Real-time Updates**: Upload documents via web interface for instant processing

### **🔄 Migration from Old System**
The new system is **fully compatible** with existing setups:
- First run will migrate existing files to database
- Old file-based caching still works as fallback
- All existing knowledge bases remain accessible

## 💬 **Chat Interface Guide**

> 🌐 **For Web GUI users**: See [GUI.md](Docs/GUI.md) for complete web interface documentation  
> 📟 **CLI users**: Continue reading below for command-line interface guide

### **Persistent Memory System**
Synapse automatically saves every conversation and resumes where you left off:

- 🔄 **Auto-Resume**: Continues your most recent session (within 24 hours)
- 💾 **Auto-Save**: Every exchange is immediately saved to disk
- 📂 **Session Management**: All conversations stored in `sessions/` directory
- 🆕 **Fresh Start**: Use `--new-session` to force a new conversation

```bash
# Default: auto-resume recent session
python chat.py

# Force new session 
python chat.py --new-session

# Load specific session
python chat.py --session sessions/chat_session_20241215_143022.json
```

### **Basic Usage**
```
💬 You: Fetch the concept approval checklist for Alexandria

🤖 Assistant:
| ...reconstructed table... |
[1] Slide 5: Concept Approval Checklist
```

### **Interactive Commands**
| Command | Description |
|---------|-------------|
| `/help` | Show all available commands |
| `/verbose` | Toggle detailed retrieval view |
| `/clear` | Clear conversation history |
| `/history` | Show recent Q&A exchanges |
| `/sessions` | List all available sessions with details |
| `/new` | Start a new session (clears current context) |
| `/stats` | Display session and knowledge base statistics |
| `/export <file>` | Export conversation to JSON |
| `/kb` | List available knowledge bases ⭐ **NEW** |
| `/switch-kb` | Switch to different knowledge base ⭐ **NEW** |
| `/quit` | Exit chat interface |

### **Verbose Mode Example**
When you enable verbose mode (`/verbose`), you'll see:

```
🔍 QUERY ENCODING
Question: What are the performance metrics for Tensix?
Model: BAAI/bge-large-en-v1.5

🔍 DENSE RETRIEVAL
📊 Summary vector results (top 10)…

🔍 PPTX TABLES
📑 Detected native PPTX table chunks (as CSV) on slides 5, 6…

🔍 FINAL CONTEXT (COHERENT)
📊 Selected 10 contexts (prioritizing top document) stitched coherently
```

## 🔧 **Advanced Configuration**

### **Initialization Options**
```bash
# Different embedding providers
python initialize_fast.py --provider openai --embed-model text-embedding-3-small
python initialize_fast.py --provider colbert --colbert-model colbert-ir/colbertv2.0

# Custom data processing
python initialize_fast.py --data-dir /path/to/docs --artifacts-dir custom_output
python initialize_fast.py --skip-parse --skip-chunk  # Only run embedding step

# Performance tuning
python initialize_fast.py --force-reprocess  # Bypass all caching
python initialize_fast.py --verbose          # Detailed output for debugging
```

### **Manual Pipeline Execution**
If you prefer step-by-step control:

```bash
# Option 1: Single knowledge base (traditional)
# 1. Parse documents with image extraction and deduplication
python pipeline/parse.py \
  --input "Data/" \
  --output "artifacts/parsed.jsonl" \
  --extract-images \
  --verbose

# Option 2: Folder-based knowledge bases (recommended)
# 1. Parse with folder-based organization
python pipeline/parse.py \
  --input "Data/" \
  --output "artifacts/parsed.jsonl" \
  --folder-based \
  --extract-images \
  --verbose

# For high-quality OCR processing (slower):
# python pipeline/parse.py --engine unstructured --folder-based --extract-images --input "Data/"

# Disable deduplication if needed:
# python pipeline/parse.py --folder-based --extract-images --disable-image-deduplication --input "Data/"

# 2. Create chunks (folder-based automatically processes all parsed files)
python pipeline/chunk.py --folder-based --verbose

# 3. Generate embeddings (folder-based creates separate embeddings per folder)
python pipeline/embed.py --folder-based --provider local --verbose

# 4. Query specific knowledge bases
python pipeline/query.py \
  --list-kb  # List available knowledge bases

python pipeline/query.py \
  --kb "Aahan_s_Notes" \
  --question "What is RISC-V?" \
  --topk 10

python pipeline/query.py \
  --select-kb \
  --question "What are the timing constraints?"
```

## 🧠 **System Architecture**

### **Pipeline Overview**
```
📁 Documents (PDF, PPTX, MD, CSV, Images)
    ↓
🔍 Parse (folder-based or single; basic/unstructured engines)
    ↓
🖼️  Extract Images (deduplication + OCR + metadata; optional AI captioning)
    ↓
✂️  Chunk (folder-based; heading-aware + token-targeted; slides atomic + windowed)
    ↓  
🧠 Embed (folder-based; multi-vector per knowledge base)
    ↓
📚 Knowledge Base Selection (interactive or programmatic)
    ↓
💬 Chat Interface (CLI/GUI with KB switching)
    ↓
🔍 Retrieve (dense + sparse + RRF + rerank + MMR + doc-coherence + images)
    ↓
🤖 Generate (Gemini 2.5 Pro + citations + relevant image paths)
```

### **Retrieval Process**
1. **Query Encoding**: Transform question into vector representation
2. **Dense Retrieval**: FAISS search over summary and full-content embeddings
3. **Sparse Retrieval**: BM25 keyword matching for exact terms
4. **Fusion**: Reciprocal Rank Fusion combines dense and sparse results
5. **Reranking**: Cross-encoder scores query-document relevance
6. **Coherent Contexting**: Prefer multiple chunks from the top document to maximize continuity
7. **Image Matching**: Identify relevant diagrams based on query content and document context
8. **Generation**: LLM produces cited answer from selected contexts, preserving tables/lists and including relevant image paths

## ⚡ **Intelligent Caching System**

Synapse includes smart caching at every pipeline stage:

### **Cache Features**
- **File Modification Tracking**: Only reprocess changed documents
- **Configuration Awareness**: Invalidates cache when settings change
- **Automatic Cache Paths**: Generated based on output filenames
- **Error Resilience**: Graceful fallback if cache operations fail

### **Cache Commands**
```bash
# Normal operation (caching enabled by default)
python initialize_fast.py

# Force reprocessing (disable caching)
python initialize_fast.py --force-reprocess

# Custom cache locations  
python pipeline/parse.py --cache-path "custom/cache.pkl"
```

### **Cache Performance Impact**
- ⚡ **First run**: Full processing (parse + chunk + embed)
- ⚡ **Second run**: ~95% time reduction via cache hits
- ⚡ **Incremental changes**: Only reprocess affected stages

## 🌐 **Interface Comparison**

| Feature | CLI | Web GUI | MCP Server |
|---------|-----|---------|------------|
| **Conversation Memory** | ✅ | ✅ | ✅ |
| **Session Management** | ✅ | ✅ Enhanced | ✅ Full API |
| **Verbose Mode** | ✅ | ✅ Console | ✅ Configurable |
| **Configuration** | Command-line args | ✅ Real-time | ✅ Programmatic |
| **Markdown Rendering** | Plain text | ✅ Rich HTML | ✅ JSON Response |
| **Real-time Parameter Tuning** | ❌ | ✅ | ✅ |
| **Visual Feedback** | Text-based | ✅ Modern UI | ✅ Structured Data |
| **Export Options** | JSON | ✅ JSON + UI | ✅ Full Export API |
| **Knowledge Base Selection** | Commands | ✅ Checkboxes | ✅ API Switching |
| **Multi-KB Search** | ❌ | ✅ Cross-KB Results | ✅ |
| **Session Configs** | ❌ | ✅ | ✅ |
| **Default Configs** | ❌ | ✅ | ✅ |
| **AI Integration** | ❌ | ❌ | ✅ **MCP Protocol** |
| **Transport Options** | ❌ | HTTP Only | ✅ **HTTP + WebSocket** |
| **Production Ready** | ✅ | ✅ | ✅ **Fully Tested** |
| **Test Suite** | ❌ | ❌ | ✅ **Comprehensive** |

📖 **[Web GUI Documentation →](Docs/CONFIG_SESSION_FEATURES.md)**  
📖 **[MCP Server Documentation →](Docs/MCP_OVERVIEW.md)**

## 📊 **Output Artifacts**

### **Pipeline Outputs**
- `artifacts/parsed.jsonl`: Normalized document elements with metadata (includes PPTX tables as CSV and image descriptions)
- `artifacts/chunked.jsonl`: Token-targeted chunks with overlap; slides atomic + windowed
- `artifacts/embeddings.jsonl`: Multi-vector embeddings with text

### **Image Processing Outputs** ⭐ **NEW**
- `artifacts/extracted_images/`: Individual image files (PNG format) with descriptive filenames
- `artifacts/image_metadata.json`: Rich metadata for all extracted images including OCR text, document context, and technical keywords
- **Deduplication**: Only unique images saved; duplicates create references to originals

### **Session Files**
- `sessions/`: Conversation history with retrieval metadata (CLI & GUI compatible)
- `default_config.json`: Default configuration for new sessions

### **Example Query Response with Images**
```
🤖 Assistant: The Ascalon cluster architecture consists of:
- Eight RISC-V CPU Cores
- Shared Cache system
- Cluster Internal Network
...

📚 Sources:
[1] ascalon_manual.pdf (pages 8, 30, 36)

🖼️  Relevant Images (3):
  [1] ascalon_manual_p048_img00_846dc90c.png
  [2] ascalon_manual_p042_img00_4b9150d6.png  
  [3] ascalon_manual_p006_img00_d93dabb2.png
```

## 🛠️ **Troubleshooting**

### **General Issues**
- If tables appear missing in answers, ensure you re-parsed after the PPTX table update and re-embedded.
- Set `TOKENIZERS_PARALLELISM=false` to silence HF fork warnings.

### **Image Processing** ⭐ **NEW**
- **Missing images**: Re-run parsing with `--extract-images` flag to extract diagrams from PDFs/PPTXs
- **Rate limiting errors**: Disable AI captioning with `--extract-images` (without `--enable-image-captioning`)
- **No relevant images**: Images are only returned when contextually relevant to your query
- **Duplicate images**: Use `--disable-image-deduplication` if you need all images saved separately
- **Dependencies**: Install `PyMuPDF` and `python-pptx` for image extraction: `pip install PyMuPDF python-pptx`

### **Knowledge Base Selection** ⭐ **NEW**
- **No knowledge bases found**: Run `python initialize_fast.py` to create folder-based knowledge bases
- **Wrong knowledge base**: Use `python chat.py --list-kb` to see available options
- **Switching knowledge bases**: Use `/switch-kb` command in chat or restart with `--kb` flag
