# TTQuery: Enterprise RAG Pipeline

**Intelligent Document Q&A with Conversation Memory and Advanced Retrieval**

TTQuery is a complete RAG (Retrieval-Augmented Generation) system that transforms your document collections into an intelligent, conversational knowledge base. Built for engineering teams, it provides precise, cited answers through an intuitive chat interface.

## ✨ **Key Features**

### **🧠 Intelligent Chat Interface**
- **CLI & Web GUI**: Choose between command-line or modern web interface
- **Conversation Memory**: Maintains context across multiple questions
- **Interactive Commands**: Rich command system with session management
- **Verbose Mode**: Shows detailed retrieval and ranking steps
- **Session Export**: Save and resume conversations
- **Real-time Configuration**: Adjust RAG parameters on-the-fly (GUI)
- **Markdown Rendering**: Beautiful formatting for code, tables, and text (GUI)

### **🔍 Advanced Retrieval**
- **Hybrid Search**: Dense (FAISS) + Sparse (BM25) retrieval with RRF fusion
- **Multi-Vector Embeddings**: Summary + full-content vectors for optimal recall
- **Smart Reranking**: Cross-encoder reranking with MMR diversification
- **Precise Citations**: Page-level source attribution with inline references
- **Table-aware prompting**: CSV/table chunks include a compact Markdown table preview (first rows/cols) during generation to preserve structure
- **PPTX-native tables**: Extracts real PPTX tables into CSV at parse time for accurate table retrieval and reconstruction
- **Slide cohesion**: Keeps slides atomic and adds small slide-window chunks for context; prompt stitches multiple chunks from the same deck coherently

## 🎯 **Quick Start**

### **Prerequisites**
```bash
# macOS dependencies for advanced parsing
brew install libmagic poppler tesseract

# Python 3.9+ required
python3 --version
```

### **1. Installation & Setup**
```bash
# Clone and setup environment
git clone <repository-url>
cd TTQuery
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
```bash
# Automated pipeline: Parse → Chunk → Embed
python initialize.py

# Advanced options
python initialize.py --provider openai --verbose --force-reprocess
```

The initialization script:
- ✅ Validates dependencies and environment
- ✅ Processes documents with progress tracking
- ✅ Uses intelligent caching for efficiency  
- ✅ Provides detailed status and error recovery

### **4. Start Chatting!**

**Option A: Command Line Interface**
```bash
# Launch interactive CLI chat interface
python chat.py

# With session persistence and verbose mode
python chat.py --verbose --session research_session.json
```

**Option B: Web GUI Interface** ⭐ **NEW**
```bash
# Launch modern web-based GUI
python chat.py --test_gui

# Then open http://127.0.0.1:7860 in your browser
```

📖 **See [GUI.md](GUI.md) for complete web interface documentation**

## 💬 **Chat Interface Guide**

> 🌐 **For Web GUI users**: See [GUI.md](GUI.md) for complete web interface documentation  
> 📟 **CLI users**: Continue reading below for command-line interface guide

### **Persistent Memory System**
TTQuery automatically saves every conversation and resumes where you left off:

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
python initialize.py --provider openai --embed-model text-embedding-3-small
python initialize.py --provider colbert --colbert-model colbert-ir/colbertv2.0

# Custom data processing
python initialize.py --data-dir /path/to/docs --artifacts-dir custom_output
python initialize.py --skip-parse --skip-chunk  # Only run embedding step

# Performance tuning
python initialize.py --force-reprocess  # Bypass all caching
python initialize.py --verbose          # Detailed output for debugging
```

### **Manual Pipeline Execution**
If you prefer step-by-step control:

```bash
# 1. Parse documents
python pipeline/parse.py \
  --input "Data/" \
  --output "artifacts/parsed.jsonl" \
  --engine unstructured --verbose

# 2. Create chunks  
python pipeline/chunk.py \
  --input "artifacts/parsed.jsonl" \
  --output "artifacts/chunked.jsonl" \
  --target-tokens 300 --overlap 0.12 --verbose

# 3. Generate embeddings
python pipeline/embed.py \
  --input "artifacts/chunked.jsonl" \
  --output "artifacts/embeddings.jsonl" \
  --provider local --verbose

# 4. Single query (without chat interface)
python pipeline/query.py \
  --question "Your question here" \
  --embeddings "artifacts/embeddings.jsonl" \
  --topk 10 --timeout 60
```

## 🧠 **System Architecture**

### **Pipeline Overview**
```
📁 Documents (PDF, PPTX, MD, CSV, Images)
    ↓
🔍 Parse (unstructured.io + OCR + PPTX native table extraction)
    ↓
✂️  Chunk (heading-aware + token-targeted + overlap; slides atomic + windowed)
    ↓  
🧠 Embed (multi-vector: summary + full-content)
    ↓
💬 Chat Interface
    ↓
🔍 Retrieve (dense + sparse + RRF + rerank + MMR + doc-coherence)
    ↓
🤖 Generate (Gemini 2.5 Pro + citations)
```

### **Retrieval Process**
1. **Query Encoding**: Transform question into vector representation
2. **Dense Retrieval**: FAISS search over summary and full-content embeddings
3. **Sparse Retrieval**: BM25 keyword matching for exact terms
4. **Fusion**: Reciprocal Rank Fusion combines dense and sparse results
5. **Reranking**: Cross-encoder scores query-document relevance
6. **Coherent Contexting**: Prefer multiple chunks from the top document to maximize continuity
7. **Generation**: LLM produces cited answer from selected contexts, preserving tables/lists

## ⚡ **Intelligent Caching System**

TTQuery includes smart caching at every pipeline stage:

### **Cache Features**
- **File Modification Tracking**: Only reprocess changed documents
- **Configuration Awareness**: Invalidates cache when settings change
- **Automatic Cache Paths**: Generated based on output filenames
- **Error Resilience**: Graceful fallback if cache operations fail

### **Cache Commands**
```bash
# Normal operation (caching enabled by default)
python initialize.py

# Force reprocessing (disable caching)
python initialize.py --force-reprocess

# Custom cache locations  
python pipeline/parse.py --cache-path "custom/cache.pkl"
```

### **Cache Performance Impact**
- ⚡ **First run**: Full processing (parse + chunk + embed)
- ⚡ **Second run**: ~95% time reduction via cache hits
- ⚡ **Incremental changes**: Only reprocess affected stages

## 🌐 **Interface Comparison**

| Feature | CLI | Web GUI |
|---------|-----|---------|
| **Conversation Memory** | ✅ | ✅ |
| **Session Management** | ✅ | ✅ Enhanced |
| **Verbose Mode** | ✅ | ✅ Console |
| **Configuration** | Command-line args | ✅ Real-time |
| **Markdown Rendering** | Plain text | ✅ Rich HTML |
| **Real-time Parameter Tuning** | ❌ | ✅ |
| **Visual Feedback** | Text-based | ✅ Modern UI |
| **Export Options** | JSON | ✅ JSON + UI |

📖 **[Detailed GUI Documentation →](GUI.md)**

## 📊 **Output Artifacts**

### **Pipeline Outputs**
- `artifacts/parsed.jsonl`: Normalized document elements with metadata (includes PPTX tables as CSV)
- `artifacts/chunked.jsonl`: Token-targeted chunks with overlap; slides atomic + windowed
- `artifacts/embeddings.jsonl`: Multi-vector embeddings with text

### **Session Files**
- `sessions/`: Conversation history with retrieval metadata (CLI & GUI compatible)

## 🛠️ **Troubleshooting**

- If tables appear missing in answers, ensure you re-parsed after the PPTX table update and re-embedded.
- Set `TOKENIZERS_PARALLELISM=false` to silence HF fork warnings.
