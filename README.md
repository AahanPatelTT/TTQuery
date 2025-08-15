# TTQuery: Enterprise RAG Pipeline

**Intelligent Document Q&A with Conversation Memory and Advanced Retrieval**

TTQuery is a complete RAG (Retrieval-Augmented Generation) system that transforms your document collections into an intelligent, conversational knowledge base. Built for engineering teams, it provides precise, cited answers through an intuitive chat interface.

## ✨ **Key Features**

### **🧠 Intelligent Chat Interface**
- **Conversation Memory**: Maintains context across multiple questions
- **Interactive Commands**: Rich command system with session management
- **Verbose Mode**: Shows detailed retrieval and ranking steps
- **Session Export**: Save and resume conversations

### **🔍 Advanced Retrieval**
- **Hybrid Search**: Dense (FAISS) + Sparse (BM25) retrieval with RRF fusion
- **Multi-Vector Embeddings**: Summary + full-content vectors for optimal recall
- **Smart Reranking**: Cross-encoder reranking with MMR diversification
- **Precise Citations**: Page-level source attribution with inline references

### **🚀 Automated Pipeline**
- **One-Command Setup**: Automated Parse → Chunk → Embed → Chat workflow
- **Intelligent Caching**: Only reprocess changed files for rapid iteration
- **Progress Tracking**: Real-time status with detailed error handling
- **Multiple Providers**: Local, OpenAI, ColBERT, and BERT embedding options

### **📚 Document Support**
- **High-Fidelity Parsing**: PDF (with OCR), PowerPoint, Markdown, CSV, images
- **Table Extraction**: Converts tables to searchable text with structure preservation
- **Image Understanding**: Vision captioning + OCR + contextual enhancement
- **Metadata Rich**: Preserves page numbers, headings, and document structure

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
```bash
# Launch interactive chat interface
python chat.py

# With session persistence and verbose mode
python chat.py --verbose --session research_session.json
```

## 💬 **Chat Interface Guide**

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
💬 You: What is Ascalon's cache hierarchy?

🤖 Assistant (2.1s):
Ascalon features a multi-level cache hierarchy designed for high performance...
[1] According to the cache overview, it includes L1 instruction and data caches...

📚 Sources:
[1] ascalon_manual.pdf (page 15)
[2] IPS-Ascalon CPU IP Cache Hierarchy Overview.pdf (page 3)

💬 You: How does this compare to ARM processors?

🤖 Assistant (1.8s):
Based on our previous discussion about Ascalon's cache hierarchy, compared to ARM processors...
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
📊 Summary vector results (top 10):
  1. [1234] tensix_manual.pdf - Performance benchmarks show...
  2. [5678] performance_analysis.md - Tensix achieves...

🔍 SPARSE RETRIEVAL (BM25)
📊 BM25 keyword results (top 10):
  1. [9012] metrics_report.pdf - Key performance indicators...

🔍 RECIPROCAL RANK FUSION (RRF)
📊 Fused results after per-document capping (top 15):
  1. [1234] tensix_manual.pdf
  2. [5678] performance_analysis.md

🔍 RERANKING
📊 Cross-encoder reranked results (top 15):
  1. [1234] tensix_manual.pdf
       Performance benchmarks show Tensix achieving...

🔍 FINAL CONTEXT (MMR DIVERSIFIED)
📊 Selected contexts for LLM (8 chunks):
  [1] tensix_manual.pdf (page 42)
      Performance metrics indicate sustained throughput...
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
  --topk 8 --timeout 60
```

## 🧠 **System Architecture**

### **Pipeline Overview**
```
📁 Documents (PDF, PPTX, MD, CSV, Images)
    ↓
🔍 Parse (unstructured.io + OCR + table extraction)
    ↓
✂️  Chunk (heading-aware + token-targeted + overlap)
    ↓  
🧠 Embed (multi-vector: summary + full-content)
    ↓
💬 Chat Interface
    ↓
🔍 Retrieve (dense + sparse + RRF + rerank + MMR)
    ↓
🤖 Generate (Gemini 2.5 Pro + citations)
```

### **Retrieval Process**
1. **Query Encoding**: Transform question into vector representation
2. **Dense Retrieval**: FAISS search over summary and full-content embeddings
3. **Sparse Retrieval**: BM25 keyword matching for exact terms
4. **Fusion**: Reciprocal Rank Fusion combines dense and sparse results
5. **Reranking**: Cross-encoder scores query-document relevance
6. **Diversification**: MMR reduces redundancy and increases coverage
7. **Generation**: LLM produces cited answer from selected contexts

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

Cache files are stored in `artifacts/` with descriptive names:
- `parsed_cache.pkl` - Parser cache (per-file metadata + parsed chunks)
- `chunked_chunk_cache.pkl` - Chunking cache (input metadata + chunks)
- `embeddings_embed_cache.pkl` - Embedding cache (input metadata + vectors)

## 🎛️ **Embedding Providers**

### **Local (Recommended)**
```bash
python initialize.py --provider local --embed-model BAAI/bge-large-en-v1.5
```
- High-quality retrieval model
- No API costs
- Privacy-preserving

### **OpenAI**
```bash
python initialize.py --provider openai --embed-model text-embedding-3-small
export OPENAI_API_KEY=your_key
```
- Managed service
- Consistent performance
- API costs apply

### **ColBERT** 
```bash
python initialize.py --provider colbert --colbert-model colbert-ir/colbertv2.0
```
- Token-level embeddings
- Late interaction scoring
- Higher precision

### **BERT (Fallback)**
```bash
python initialize.py --provider bert --bert-model bert-base-uncased
```
- Simple mean-pooled vectors
- Lightweight option
- Basic functionality

## 📊 **Output Artifacts**

### **Pipeline Outputs**
- `artifacts/parsed.jsonl`: Normalized document elements with metadata
- `artifacts/chunked.jsonl`: Token-targeted chunks with overlap
- `artifacts/embeddings.jsonl`: Multi-vector embeddings with text

### **Session Files**
- `session.json`: Conversation history with retrieval metadata
- Export format includes timestamps, sources, and retrieval statistics

### **Cache Files**
- Automatic cache management with metadata validation
- Persistent across runs with intelligent invalidation

## 🛠️ **Troubleshooting**

### **Common Issues**

**Environment Variables Not Set**
```bash
echo "API Key: $LITELLM_API_KEY"
echo "Base URL: $LITELLM_BASE_URL"
# Should show your configured values
```

**Missing Dependencies**
```bash
pip install -r requirements.txt
# On macOS: brew install libmagic poppler tesseract
```

**Empty Knowledge Base**
```bash
# Check if embeddings file exists and has content
ls -la artifacts/embeddings.jsonl
wc -l artifacts/embeddings.jsonl
```

**LiteLLM Connection Issues**
```bash
# Test proxy connection
python LiteLLM.py --prompt "Test connection" --timeout 10
```

### **Debug Mode**
```bash
# Verbose initialization
python initialize.py --verbose

# Verbose chat with session logging
python chat.py --verbose --session debug_session.json
```

## 🚀 **Performance Tips**

### **Optimization Strategies**
- Use caching for iterative development (default behavior)
- Adjust chunk size for your document types (`--target-tokens`)
- Choose embedding provider based on accuracy vs. cost needs
- Enable verbose mode only for debugging (impacts performance)

### **Hardware Recommendations**
- **Memory**: 8GB+ RAM for large document collections
- **Storage**: SSD recommended for cache performance
- **CPU**: Multi-core benefits embedding computation
- **GPU**: Optional for local embedding models

## 📚 **Documentation**

- `HOW_IT_WORKS.md`: Technical implementation details
- `pipeline/`: Individual stage documentation in docstrings
- `/help` in chat: Interactive command reference

## 🔐 **Security Notes**

- Environment variables avoid secrets in code
- Local processing preserves data privacy
- Optional cloud embedding providers (OpenAI)
- Session files may contain sensitive conversation data

---

**🎯 Ready to transform your documents into intelligent conversations? Start with `python initialize.py` and then `python chat.py`!**