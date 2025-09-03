## How Synapse Works

Synapse is a comprehensive RAG (Retrieval-Augmented Generation) system that transforms document collections into intelligent, conversational knowledge bases. This document explains the technical implementation, design decisions, and architectural choices.

## System Overview

Synapse provides two primary interfaces:

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
- **Session-specific configs**: Each session stores its own configuration
- **Default config management**: Save and load default settings for new sessions

### Pipeline at a glance
1. Parse (Data → parsed.jsonl) **[Cached]**
   - Walks the `Data/` directory and normalizes documents into a unified JSONL format with rich metadata for citations.
   - Default engine is unstructured.io for high‑fidelity parsing. A lightweight basic engine exists for resilience.
   - **PPTX native tables**: Extracts real PowerPoint tables (via `python-pptx`) per slide and serializes as CSV with `metadata.content_format="csv"`, `slide_number`, and `table_index`.
   - **Image extraction** ⭐ **NEW**: Extracts diagrams, charts, and technical illustrations from PDFs and PPTXs with rich metadata including OCR text, document context, and AI captions (optional).
   - **Image deduplication** ⭐ **NEW**: Content-based duplicate detection prevents saving the same image multiple times across documents.
   - **Folder-based processing** ⭐ **NEW**: Creates specialized knowledge bases per folder for domain-specific retrieval.
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
   - **Multi-KB selection**: Choose multiple knowledge bases via GUI checkboxes ⭐ **NEW**
   - **Cross-KB search**: Unified results from multiple knowledge sources ⭐ **NEW**

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
   - **Image matching** ⭐ **NEW**:
     - Identifies relevant diagrams based on query content and document context from retrieved chunks.
     - Includes full image paths for viewing extracted technical diagrams and illustrations.
   - **Generate**:
     - Uses Gemini 2.5 Pro via TensTorrent LiteLLM proxy (`LITELLM_BASE_URL`, `LITELLM_API_KEY`).
     - **Conversation context**: Previous Q&A exchanges inform new responses.
     - Strict inline citations `[n]` per claim; bottom Sources block includes `source_path` and page/slide when available.
     - **Table-aware prompting**: CSV/table chunks render a compact Markdown table preview (first rows/cols). Native PPTX tables are included via parsed CSV.
     - **Visual content inclusion**: Relevant image paths are automatically included when queries reference diagrams or visual content.

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
- `artifacts/parsed.jsonl`: One line per parsed element with `id`, `document_id`, `source_path`, `source_type`, `content`, and `metadata` (page/slide numbers, heading paths, element type, coordinates when available). Includes native PPTX tables serialized to CSV with `content_format="csv"` and `table_index`. Image elements include rich visual metadata.
- `artifacts/chunked.jsonl`: One line per chunk with `num_tokens` and `chunk_index`, preserving provenance metadata for citations. Includes atomic slide chunks and windowed slide chunks for PPTX, plus image description chunks.
- `artifacts/embeddings.jsonl`: One line per chunk with `summary_text`, `full_text`, and their embeddings (`embedding_summary`, `embedding_full`). These are consumed by the query stage for FAISS and BM25 indexing at runtime.
  - When the ColBERT provider is used, token-level vectors are also emitted as `embedding_summary_mv` and `embedding_full_mv` and pooled vectors are kept for compatibility.

### Image Processing Artifacts ⭐ **NEW**
- `artifacts/extracted_images/`: Individual image files (PNG format) with descriptive filenames including document name, page number, and unique ID.
- `artifacts/image_metadata.json`: Rich metadata for all extracted images including:
  - OCR text extracted from the image
  - Document context from surrounding text  
  - Technical keywords and image categorization (diagram, chart, flowchart, etc.)
  - Source document, page number, and extraction method
  - Image dimensions, file size, and format information

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

- Images → extraction + OCR + context + optional AI captioning ⭐ **ENHANCED**
  - Images are extracted as independent PNG files with rich metadata for direct viewing.
  - OCR text is extracted from images using Tesseract for searchable technical content.
  - Document context from surrounding page text is preserved for relevance matching.
  - Optional AI captioning (BLIP) can be enabled but is disabled by default to avoid rate limiting.
  - Images are categorized by type (diagram, chart, flowchart, table) for better organization.
  - Technical keywords are extracted to improve retrieval relevance for engineering queries.

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
- Parser: `--engine [unstructured|basic]`, `--pdf-strategy`, `--ocr-languages`, table extractor and CSV conversion flags.
  - **Image extraction** ⭐ **NEW**: `--extract-images`, `--enable-image-captioning`, `--images-output-dir` for controlling visual content processing.
- Chunker: `--target-tokens`, `--overlap`, tokenizer selection.
- Embedder: local sentence-transformers or OpenAI; summary mode (heuristic/LLM).
  - ColBERT provider adds native multi-vector (token-level) embeddings with pooled compatibility vectors.
- Query: `--chunked` flag required for image inclusion in responses.

## Visual Content Processing ⭐ **NEW**

### **Image Extraction Pipeline**

**PDF Image Extraction**
- **PyMuPDF Integration**: Uses PyMuPDF (fitz) for fast, reliable image extraction from PDF documents
- **Page Context Preservation**: Captures surrounding text from the PDF page to provide document context
- **Multiple Format Support**: Handles various embedded image formats (PNG, JPEG, etc.)
- **Metadata Enrichment**: Generates comprehensive metadata including page numbers, coordinates, and document hierarchy

**PPTX Image Extraction**  
- **Native PowerPoint Processing**: Uses python-pptx for direct access to slide images
- **Slide Context Integration**: Associates images with slide titles and bullet points for semantic understanding
- **Presentation Flow Awareness**: Maintains slide sequence context for better retrieval relevance

**OCR Text Extraction**
- **Tesseract Integration**: Extracts searchable text from technical diagrams and charts
- **Technical Content Optimization**: Specialized character whitelisting for engineering terminology
- **Error Correction**: Common OCR error corrections for technical terms (I2C, GPIO, CPU, etc.)

**AI Caption Generation (Optional)**
- **BLIP Model Integration**: Uses Salesforce BLIP for natural language image descriptions
- **Rate Limiting Protection**: Disabled by default to prevent HuggingFace API throttling
- **Model Caching**: Efficient model loading and reuse across multiple images
- **Graceful Degradation**: Continues processing if AI captioning fails

### **Image Metadata Architecture**

**Rich Metadata Structure**
- **Unique Identification**: SHA-256 hash-based IDs for deterministic image tracking
- **Source Attribution**: Full document path, page/slide numbers, and extraction method
- **Visual Properties**: Dimensions, file size, format, and quality metrics
- **Content Analysis**: OCR text, document context, and optional AI captions
- **Technical Classification**: Automatic categorization (diagram, chart, flowchart, table)
- **Keyword Extraction**: Technical terms relevant for engineering queries

**Contextual Information**
- **Document Context**: Surrounding page text that provides semantic meaning
- **Hierarchical Location**: Section headings, slide titles, and document structure
- **Technical Keywords**: Domain-specific terms extracted for improved retrieval
- **Image Relationships**: Cross-references to related text and tables

### **Smart Image Retrieval**

**Context-Based Matching**
- **Query Relevance**: Matches user questions to image content and surrounding context
- **Document Coherence**: Prioritizes images from highly relevant retrieved text chunks
- **Technical Domain Awareness**: Enhanced matching for engineering and technical queries
- **Semantic Understanding**: Uses both OCR text and document context for relevance scoring

**Integration with Retrieval Pipeline**
- **Hybrid Search Support**: Images participate in both dense and sparse retrieval
- **Reranking Integration**: Image relevance considered in cross-encoder reranking
- **MMR Diversification**: Images contribute to response diversity and coverage
- **Citation Integration**: Image paths provided alongside traditional text citations

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

**Default Config Management**
- **`default_config.json`**: Stores the default configuration template
- **Save as Default**: Overwrites `default_config.json` with the current session's settings
- **Reset to Default**: Reverts the current session's config to the values in `default_config.json`

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
- **Advanced image understanding**: Enhanced diagram analysis and technical illustration processing
- **Temporal awareness**: Time-based relevance and freshness scoring  
- **User personalization**: Adaptive ranking based on usage patterns
- **Distributed processing**: Scale to larger document collections
- **Interactive image viewing**: In-browser image display and annotation capabilities

### **Maintenance and Updates**

Keep this document updated as new features are added, especially:
- New interactive commands and chat capabilities
- Additional embedding providers and models
- Enhanced retrieval strategies and ranking methods
- **Visual content processing improvements and new image formats** ⭐ **NEW**
- Performance optimizations and caching improvements
- User experience enhancements and automation features

## 🆕 **New Features (Latest Updates)**

### **📚 Folder-Based Knowledge Bases**
Synapse now supports creating specialized knowledge bases for each folder in your Data directory:

**Benefits:**
- **Domain-specific retrieval**: Query only relevant documents for your topic
- **Reduced noise**: Avoid irrelevant results from other domains
- **Better performance**: Smaller knowledge bases = faster search
- **Specialized expertise**: Each knowledge base focuses on its content area

**Usage:**
```bash
# Create folder-based knowledge bases
python initialize_folders.py

# List available knowledge bases
python chat.py --list-kb
python pipeline/query.py --list-kb

# Use specific knowledge base
python chat.py --kb "Aahan_s_Notes"
python pipeline/query.py --kb "hash_Confluence_IPS" --question "IPS requirements"

# Interactive selection
python chat.py --select-kb
python pipeline/query.py --select-kb --question "Your question"
```

**Knowledge Base Naming:**
- Regular folders: `Aahan_s_Notes`, `Ascalon_Docs`
- Folders with `#` prefix: `hash_Confluence_IPS`, `hash_Confluence_PSE`
- Display names: `Aahan's Notes`, `#Confluence/IPS`

### **🖼️ Image Deduplication System**
Intelligent duplicate detection prevents saving the same image multiple times:

**How It Works:**
1. **Content-based hashing**: Images normalized (RGB, 512x512) and hashed
2. **Cross-document detection**: Works across all files and folders
3. **Smart references**: Duplicates point to original files instead of creating copies
4. **Metadata preservation**: OCR text, captions, and context preserved for all references

**Benefits:**
- **Storage savings**: Can reduce image storage by 50%+ for documents with repeated diagrams
- **Faster processing**: Duplicates skip expensive OCR and captioning
- **Cleaner organization**: No redundant files cluttering the output directory
- **Maintained searchability**: All image references remain fully searchable

**Configuration:**
```bash
# Default: deduplication enabled
python initialize_folders.py --extract-images

# Disable deduplication if needed
python initialize_folders.py --extract-images --disable-image-deduplication
```

**Deduplication Statistics Example:**
```
🖼️  IMAGE EXTRACTION:
   📷 Total images processed: 150
   ✨ Unique images saved: 95
   🔗 Duplicate references: 55
   💡 Deduplication saved ~55 duplicate files
```

### **🔄 Dynamic Knowledge Base Management**

**CLI Commands:**
- `/kb` - List available knowledge bases with current selection
- `/switch-kb` - Interactively switch to a different knowledge base

**Web GUI Multi-KB Selector** ⭐ **NEW**
- **Checkbox interface**: Select multiple knowledge bases simultaneously
- **Real-time status**: Visual indicators for selected KBs
- **Control buttons**: Select All, Clear All, Refresh options
- **Cross-KB search**: Query multiple knowledge bases with unified results
- **Smart result fusion**: Intelligently combines results from different sources
- **Source attribution**: Clear indication of which KB provided each result

**Technical Implementation:**
- `enhanced_answer_multi_kb()` function handles multi-KB queries
- Results distributed fairly across selected knowledge bases
- Automatic fallback to single-KB optimization when only one KB selected
- Combined source blocks with clear KB attribution

### **Security Notes**
- Do not commit real API keys. Use environment variables for `LITELLM_API_KEY` and `LITELLM_BASE_URL`
- Session files may contain sensitive conversation data - manage access appropriately
- The optional `litellm_config.yaml` is not required by the pipeline and should avoid storing sensitive values
- Consider local-only processing for highly sensitive documents


