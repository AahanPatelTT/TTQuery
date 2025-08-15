## How TTQuery Works

TTQuery is a comprehensive RAG (Retrieval-Augmented Generation) system that transforms document collections into intelligent, conversational knowledge bases. This document explains the technical implementation, design decisions, and architectural choices.

## System Overview

TTQuery provides two primary interfaces:

### **🚀 Automated Setup (`initialize.py`)**
- **One-command initialization**: Runs complete Parse → Chunk → Embed pipeline
- **Dependency validation**: Checks all requirements and environment setup
- **Progress tracking**: Real-time status with detailed error handling
- **Intelligent caching**: Only reprocess changed files for rapid iteration

### **💬 Interactive Chat (`chat.py`)**
- **Conversation memory**: Maintains context across multiple questions
- **Verbose retrieval**: Shows detailed ranking and retrieval steps
- **Session management**: Save/load conversation history
- **Interactive commands**: Rich command system for user control

### Pipeline at a glance
1. Parse (Data → parsed.jsonl) **[Cached]**
   - Walks the `Data/` directory and normalizes documents into a unified JSONL format with rich metadata for citations.
   - Default engine is unstructured.io for high‑fidelity parsing. A lightweight basic engine exists for resilience.
   - **PPTX native tables**: Extracts real PowerPoint tables (via `python-pptx`) per slide and serializes as CSV with `metadata.content_format="csv"`, `slide_number`, and `table_index`.
   - **Caching**: Tracks file modification times and sizes. Only reparses changed files.

2. Chunk (parsed.jsonl → chunked.jsonl) **[Cached]**
   - Converts parsed elements into embedding‑ready chunks targeting ~200–400 tokens with ~10–15% overlap.
   - Heading‑aware for Markdown; sentence‑aware packing for other text; row‑aware for CSV tables.
   - **Slides atomic + windowed**: Keeps PPTX slides as single chunks and creates small slide-window chunks (e.g., 2 slides) to preserve context across slides.
   - **Caching**: Invalidates cache when input file or chunking configuration (tokens, overlap, encoding) changes.

3. Embed (chunked.jsonl → embeddings.jsonl) **[Cached]**
   - Produces two vectors per chunk for multi‑vector retrieval:
     - Summary Vector: a concise, human‑readable summary (broad recall)
     - Full‑Content Vector: embedding of the complete, linearized chunk text (precise recall)
   - With ColBERT provider, we also write native token‑level vectors:
     - `embedding_summary_mv` and `embedding_full_mv` (lists of token embeddings) alongside pooled single vectors for compatibility.
   - **Caching**: Invalidates cache when input file or embedding configuration (provider, model, summary mode) changes.

4. Interactive Chat Interface **[Memory-Enabled]**
   - **Conversation management**: Maintains context across multiple exchanges
   - **Session persistence**: Save/load conversation history to JSON files
   - **Interactive commands**: `/help`, `/verbose`, `/clear`, `/history`, `/stats`, `/export`
   - **Verbose mode**: Detailed retrieval and ranking step visualization

5. Retrieve + Generate **[Enhanced]**
   - **Hybrid recall** (dense + sparse):
     - Dense multi‑vector search over `embedding_summary` (broad) and `embedding_full` (precise) using FAISS (inner product with L2‑normalized vectors).
     - Sparse BM25 over `full_text`.
     - Reciprocal Rank Fusion (RRF) to combine the above lists, with per‑document caps to avoid domination.
   - **Rerank + diversify**:
     - Local cross‑encoder rerank (Flashrank or `cross-encoder/ms-marco-MiniLM-L-6-v2`) with robust error handling.
     - Maximal Marginal Relevance (MMR) reduces redundancy and increases coverage.
   - **Coherent context selection**:
     - Prefer multiple chunks from the top-ranked document to maximize continuity for slide decks and long sections.
   - **Generate**:
     - Uses Gemini 2.5 Pro via TensTorrent LiteLLM proxy (`LITELLM_BASE_URL`, `LITELLM_API_KEY`).
     - **Conversation context**: Previous Q&A exchanges inform new responses.
     - Strict inline citations `[n]` per claim; bottom Sources block includes `source_path` and page/slide when available.
     - **Table-aware prompting**: CSV/table chunks render a compact Markdown table preview (first rows/cols). Native PPTX tables are included via parsed CSV.

    Quick smoke test for the proxy is available via `LiteLLM.py`:
```bash
export LITELLM_API_KEY=*your key*
export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/
python LiteLLM.py --prompt "Hello from LiteLLM proxy!" --model gemini/gemini-2.5-pro --timeout 30
```

### Intelligent Caching System

All pipeline stages implement smart caching to dramatically improve development velocity and iterative workflows:

#### Cache Architecture
- **File-based persistence**: Uses Python pickle for efficient serialization
- **Metadata tracking**: Stores file modification times, sizes, and configuration hashes
- **Configuration awareness**: Invalidates cache when settings change (chunk sizes, models, etc.)
- **Automatic paths**: Cache files are auto-generated based on output filenames
- **Error resilience**: Graceful fallback if cache operations fail

#### Cache Implementation per Stage
1. **Parser Cache** (`parsed_cache.pkl`)
   - **Tracks**: Individual file mtime/size in input directory
   - **Invalidates**: When source files are modified or parser settings change
   - **Stores**: Map of `file_path` → `CacheEntry(chunks, metadata)`
   - **Performance**: Skip expensive OCR, table extraction, image captioning

2. **Chunking Cache** (`chunked_chunk_cache.pkl`)  
   - **Tracks**: Input JSONL file mtime/size and configuration hash
   - **Invalidates**: When parsed.jsonl changes or chunking config (tokens, overlap, encoding) changes
   - **Stores**: Single cache entry with all chunked results
   - **Performance**: Skip token-aware text splitting and overlap processing

3. **Embedding Cache** (`embeddings_embed_cache.pkl`)
   - **Tracks**: Input JSONL file mtime/size and configuration hash  
   - **Invalidates**: When chunked.jsonl changes or embedding config (provider, model, summary mode) changes
   - **Stores**: Single cache entry with all embedding vectors
   - **Performance**: Skip expensive model inference (local or API calls)

#### Cache Control
```bash
# Default: caching enabled
python pipeline/parse.py --input "Data/" --output "artifacts/parsed.jsonl"

# Force reprocessing (bypass cache)  
python pipeline/chunk.py --no-cache --input "artifacts/parsed.jsonl" --output "artifacts/chunked.jsonl"

# Custom cache location
python pipeline/embed.py --cache-path "custom/embed_cache.pkl" --input "artifacts/chunked.jsonl" --output "artifacts/embeddings.jsonl"
```

#### Cache Performance Impact
- **First run**: Full processing (parse + chunk + embed)
- **Second run** (no changes): ~95% time reduction via cache hits
- **Incremental changes**: Only reprocess affected stages and downstream dependencies
- **Config changes**: Intelligent invalidation ensures correctness

### Artifacts and flow
- `artifacts/parsed.jsonl`: One line per parsed element with `id`, `document_id`, `source_path`, `source_type`, `content`, and `metadata` (page/slide numbers, heading paths, element type, coordinates when available). Includes native PPTX tables serialized to CSV with `content_format="csv"` and `table_index`.
- `artifacts/chunked.jsonl`: One line per chunk with `num_tokens` and `chunk_index`, preserving provenance metadata for citations. Includes atomic slide chunks and windowed slide chunks for PPTX.
- `artifacts/embeddings.jsonl`: One line per chunk with `summary_text`, `full_text`, and their embeddings (`embedding_summary`, `embedding_full`). These are consumed by the query stage for FAISS and BM25 indexing at runtime.
  - When the ColBERT provider is used, token-level vectors are also emitted as `embedding_summary_mv` and `embedding_full_mv` and pooled vectors are kept for compatibility.

### Important design choices (and why)
- Unstructured‑first parsing
  - Extracts high‑fidelity elements (titles, text, tables, figures) with page numbers and optional OCR.
  - Basic parsers remain available as a fallback and for lighter environments.

- Deterministic, citation‑friendly IDs
  - `document_id` is a hash of the absolute path; `chunk id` is a hash of document id + locator + content.
  - Ensures reproducibility, deduplication, and stable citations.

- Tables → structured CSV and sentence linearization
  - PDF tables are extracted as CSV (default via pdfplumber). Unstructured HTML tables can be converted to CSV.
  - **PPTX tables** are extracted natively and serialized to CSV, enabling true table retrieval and reconstruction.
  - Before embedding, tables are linearized into sentences that name columns and sample rows—this makes them retrievable by meaning, not layout.

- Images → captioning + OCR + context
  - Images are converted to text by combining a vision caption (e.g., BLIP) with OCR.
  - We also augment captions with nearby page text to reflect surrounding context.

- Smart chunking with overlap
  - Markdown is split with heading awareness; other text uses sentence‑aware packing.
  - CSV is row‑aware to avoid splitting rows across chunks.
  - PPTX slides remain atomic with additional windowed chunks to preserve flow across slides.
  - Overlap preserves context across boundaries to reduce answer truncation.

- Multi‑vector retrieval
  - Summary vectors capture “what this is about” for broad/ambiguous queries.
  - Full‑content vectors capture details for precise questions.
  - Default local embedding model is a state‑of‑the‑art retriever: `BAAI/bge-large-en-v1.5`.
  - Optional providers: OpenAI dense vectors; ColBERT token‑level vectors (plus pooled), and BERT as a fallback.
  - With ColBERT, use token‑level late interaction scoring for higher precision.
  - OpenAI embeddings are supported via `--provider openai` and `--openai-embed-model` if you prefer managed embedding APIs.

### Extensibility
- Code ingestion: add tree‑sitter parsers to produce function‑level elements; reuse the same chunking and embedding flows.
- Table fidelity: integrate camelot/tabula for specialized PDFs if needed.
- Summaries: swap heuristic summaries for LLM‑generated ones when permitted.
- Retrieval: add reranking, hybrid keyword + vector search, and temporal/source filters.

### Reliability and trust
- Rich metadata (page/slide/heading/coordinates) is preserved end‑to‑end for inline citations.
- Deterministic IDs and sorted outputs make runs reproducible across environments.

### Operational knobs (non‑exhaustive)
- Parser: `--engine [unstructured|basic]`, `--pdf-strategy`, `--ocr-languages`, table extractor and CSV conversion flags, image captioning flags.
- Chunker: `--target-tokens`, `--overlap`, tokenizer selection.
- Embedder: local sentence-transformers or OpenAI; summary mode (heuristic/LLM).
  - ColBERT provider adds native multi-vector (token-level) embeddings with pooled compatibility vectors.

## Enhanced User Experience

### **Interactive Chat Interface**

**Conversation Memory**
- **Context window**: Maintains last 3 Q&A exchanges by default
- **Contextual responses**: Previous answers inform new questions  
- **Follow-up support**: Natural conversation flow with references to earlier discussions
- **Context pruning**: Automatic truncation of long answers for manageable context

**Session Management**
- **Persistent storage**: JSON-based conversation history
- **Resume capability**: Load previous sessions and continue conversations
- **Export functionality**: Save conversations for analysis or sharing
- **Metadata tracking**: Timestamps, retrieval statistics, and performance metrics

**Interactive Commands**
```
/help         - Show all available commands and usage tips
/verbose      - Toggle detailed retrieval step visualization  
/clear        - Reset conversation history and start fresh
/history      - Display recent question-answer exchanges
/stats        - Show session and knowledge base statistics
/export <file> - Save complete conversation to JSON file
/quit         - Exit chat interface gracefully
```

### **Verbose Retrieval Mode**

When enabled (`/verbose` in chat or `--verbose` flag), the system displays:

**1. Query Encoding**
- Original question and preprocessing steps
- Embedding model used and query vector generation
- Query prefixing for instruction-tuned models

**2. Dense Retrieval Results**
- Top candidates from summary vector search (broad recall)
- Top candidates from full-content vector search (precise recall) 
- Similarity scores and source document attribution

**3. Sparse Retrieval Results**
- BM25 keyword matching results with relevance scores
- Term overlap analysis and frequency statistics
- Keyword-based ranking compared to semantic ranking

**4. Fusion Analysis**
- Reciprocal Rank Fusion (RRF) score computation
- Weight application: [0.9, 1.2, 0.8] for [summary, full, sparse]
- Per-document cap application (default: 4 chunks per document)

**5. Reranking Process**
- Cross-encoder relevance scoring (Flashrank or MiniLM)
- Query-document pair analysis up to 2000 characters
- Confidence scores and re-ordering results

**6. Final Context Selection**
- Maximal Marginal Relevance (MMR) diversification
- Lambda parameter balancing (0.7 relevance vs 0.3 diversity)
- Selected contexts with metadata and source attribution

**7. LLM Generation**
- Prompt construction details and context integration
- Generation timing and token usage statistics
- Citation enforcement and quality metrics

### **Automation and Setup**

**Initialize Script (`initialize.py`)**
- **Pre-flight checks**: Validates dependencies, environment variables, and data directory
- **Progress tracking**: Real-time status updates with time estimates
- **Error handling**: Graceful failure recovery with specific troubleshooting guidance
- **Flexible execution**: Skip stages, force reprocessing, custom providers/models
- **Success validation**: Verifies output files and provides next steps

**Advanced Configuration**
- **Multiple providers**: Local, OpenAI, ColBERT, BERT embedding options
- **Custom models**: Configurable embedding models for different domains
- **Performance tuning**: Adjustable batch sizes, timeout values, and cache settings
- **Resource management**: Memory usage monitoring and optimization suggestions

### **Reliability and Error Handling**

**Robust Processing**
- **Flashrank compatibility**: Handles both object and dictionary returns
- **Network resilience**: Retry logic for API-based providers
- **Partial failures**: Continue processing when individual files fail
- **Cache validation**: Integrity checks and automatic recovery

**Debugging Support**
- **Verbose logging**: Detailed information for troubleshooting
- **Performance metrics**: Timing analysis and bottleneck identification
- **Health checks**: System validation and configuration verification
- **Error context**: Comprehensive error messages with solution suggestions

### **Security and Privacy**

**Environment-Based Configuration**
- **API key management**: Environment variables prevent secrets in code
- **Local processing**: Documents remain on local system by default
- **Session data**: Conversation history stored locally with user control
- **Provider choice**: Option for fully local processing without external APIs

### **Extensibility Framework**

**Modular Architecture**
- **Plugin support**: Easy addition of new embedding providers
- **Custom parsers**: Extensible document format support
- **Provider abstraction**: Swappable backends for different components
- **API consistency**: Uniform interfaces across all components

### **Future Enhancements**

**Planned Features**
- **Multi-modal enhancement**: Advanced image and diagram understanding
- **Temporal awareness**: Time-based relevance and freshness scoring
- **User personalization**: Adaptive ranking based on usage patterns
- **Distributed processing**: Scale to larger document collections

### **Maintenance and Updates**

Keep this document updated as new features are added, especially:
- New interactive commands and chat capabilities
- Additional embedding providers and models
- Enhanced retrieval strategies and ranking methods
- Performance optimizations and caching improvements
- User experience enhancements and automation features

### **Security Notes**
- Do not commit real API keys. Use environment variables for `LITELLM_API_KEY` and `LITELLM_BASE_URL`
- Session files may contain sensitive conversation data - manage access appropriately
- The optional `litellm_config.yaml` is not required by the pipeline and should avoid storing sensitive values
- Consider local-only processing for highly sensitive documents


