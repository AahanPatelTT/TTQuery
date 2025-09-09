# Synapse Web GUI Documentation

**Modern Web Interface for Synapse RAG System**

The Synapse web GUI provides a polished, user-friendly interface for interacting with your document knowledge base. Built with modern web technologies, it offers all CLI functionality plus enhanced visual features and real-time configuration.

## 🚀 **Quick Start**

### **Launch the GUI**
```bash
# Activate your environment
source .venv/bin/activate

# Start the web server
python chat.py --test_gui

# Open in your browser
open http://127.0.0.1:7860
```

### **Prerequisites**
- Completed [initial setup](README.md#quick-start) and knowledge base initialization
- Environment variables configured (`LITELLM_API_KEY`, `LITELLM_BASE_URL`)
- Flask and markdown dependencies (automatically installed via `requirements.txt`)

## 🎨 **Interface Overview**

The GUI is split into two main areas:

### **📋 Left Sidebar: Configuration & Session Management**
- **RAG Admin Panel**: Real-time configuration of retrieval parameters
- **Session Control**: Create, load, and manage conversation sessions
- **Debug Options**: Toggle verbose mode and view system info

### **💬 Right Panel: Chat Interface**
- **Message History**: Conversation with markdown rendering
- **Input Area**: Multi-line text input with auto-resize
- **Status Indicators**: Connection state and loading feedback

![Synapse GUI Layout](https://i.imgur.com/placeholder.png)

## ⚙️ **Configuration Panel**

### **System Prompt Customization**
```
Custom system prompt override for specialized use cases:

Example: "You are a hardware engineer assistant. Focus on 
technical specifications and always include performance 
metrics when available."
```

### **Retrieval Parameters**

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| **Top K Contexts** | Number of final contexts sent to LLM | 1-20 | 10 |
| **Per-Document Cap** | Max chunks per document in results | 1-12 | 8 |
| **MMR Lambda (λ)** | Diversity vs relevance balance | 0.0-1.0 | 0.8 |
| **Timeout** | LLM generation timeout (seconds) | 10-180 | 60 |
| **Verbose Retrieval** | Show detailed retrieval steps | ✓/✗ | ✗ |

### **Parameter Effects**

**Top K Contexts**: Higher values provide more context but may include noise
- 🔹 **5-8**: Focused, precise answers
- 🔹 **10-15**: Comprehensive context (recommended)
- 🔹 **16-20**: Maximum context for complex queries

**Per-Document Cap**: Controls document diversity
- 🔹 **1-3**: Force diversity across multiple documents
- 🔹 **4-8**: Balanced approach (recommended)
- 🔹 **9-12**: Allow deep focus on single document

**MMR Lambda**: Balances relevance vs diversity
- 🔹 **0.5-0.6**: High diversity, broad coverage
- 🔹 **0.7-0.8**: Balanced (recommended)
- 🔹 **0.9-1.0**: High relevance, focused results

## 📝 **Session Management**

### **Session Features**
- **Auto-Resume**: Continues your most recent session automatically
- **Session List**: View all available sessions with exchange counts
- **Load Previous**: Switch between different conversation contexts
- **Export**: Download session history as JSON

### **Session Operations**

```bash
# Session files are stored in sessions/ directory
sessions/
├── chat_session_20241215_143022.json
├── chat_session_20241215_150830.json
└── chat_session_20241216_091245.json
```

**Create New Session**: Starts fresh conversation, clearing current context

**Load Session**: Switch to previous conversation, maintaining all history

**Export Session**: Download complete conversation with metadata:
```json
{
  "created_at": "2024-12-15T14:30:22",
  "exported_at": "2024-12-15T16:45:10",
  "total_exchanges": 12,
  "history": [
    {
      "timestamp": "2024-12-15T14:30:45",
      "question": "What are the Alexandria performance metrics?",
      "answer": "Based on the documentation...",
      "sources": "[1] IPM-Alexandria-270725-192147.pdf...",
      "retrieval_info": {
        "model_used": "BAAI/bge-large-en-v1.5",
        "final_contexts": 8,
        "generation_time_s": 2.3
      }
    }
  ]
}
```

## 📚 **Knowledge Base Selection** ⭐ **NEW**

The GUI provides an intuitive interface for selecting which knowledge bases to search:

### **Multi-KB Selector**
- **Checkbox Interface**: Select one or multiple knowledge bases simultaneously
- **Visual Indicators**: Current KB highlighted, selection status clearly shown
- **Real-time Status**: Dynamic status updates showing selected KBs
- **KB Information**: Each KB shows size, chunk count, and description

### **Control Options**
- **🔄 Refresh**: Reload the knowledge base list
- **✅ Select All**: Check all available knowledge bases
- **❌ Clear All**: Deselect all knowledge bases
- **Status Display**: Shows number of selected KBs with color coding

### **Search Behavior**
- **Single KB**: Optimized single-knowledge-base search
- **Multiple KBs**: Cross-KB search with unified results
- **Source Attribution**: Results clearly indicate which KB provided each source
- **Fair Distribution**: Results distributed evenly across selected KBs

### **Example Usage**
1. **Domain-Specific Search**: Select only "Technical_Specs" KB for hardware questions
2. **Comprehensive Search**: Select all KBs for broad research queries
3. **Comparative Analysis**: Select specific KBs to compare information across domains

```
Selected: 3 knowledge bases
✅ Technical_Specs (1,234 chunks, 15.2 MB)
✅ User_Manuals (856 chunks, 8.7 MB) 
✅ Research_Papers (2,341 chunks, 28.1 MB)
❌ Meeting_Notes (445 chunks, 3.2 MB)
```

## 📤 **Document Upload** ⭐ **NEW**

The GUI now includes a powerful document upload system for real-time knowledge base updates:

### **Upload Interface**
- **Drag & Drop**: Simply drag files onto the upload area
- **File Browser**: Click to select multiple files at once
- **Visual Feedback**: See selected files with icons, names, and sizes
- **Progress Tracking**: Real-time upload progress with status updates

### **Knowledge Base Management**
- **Target Selection**: Choose which knowledge base to upload to
- **Existing KBs**: Select from current knowledge bases
- **New KB Creation**: Type a new name to create a fresh knowledge base
- **Smart Organization**: Files automatically organized by folder structure

### **Supported File Types**
- **Documents**: PDF, PPTX, DOCX, TXT, MD
- **Data**: CSV files with automatic table processing
- **Images**: PNG, JPG, JPEG, TIFF, BMP with OCR and captioning

### **Real-Time Processing**
- **Instant Upload**: Files uploaded immediately to selected folder
- **Auto-Reinitialization**: Automatic parsing and embedding after upload
- **Duplicate Detection**: Prevents storing the same document twice
- **Background Processing**: Non-blocking processing in background threads
- **Live Progress**: Real-time progress tracking with detailed steps
- **Status Updates**: Live feedback on parsing and embedding progress
- **Auto-Refresh**: Knowledge base list updates automatically

### **Upload Workflow**
1. **Select Target**: Choose existing KB or create new one
2. **Add Files**: Drag & drop or click to select documents
3. **Review Selection**: See file list with sizes and types
4. **Upload**: Click "Upload & Process" for instant processing
5. **Monitor**: Watch real-time status updates

### **Example Usage**
```
📁 Target: Technical_Specifications
📄 Selected Files:
   📄 new_datasheet.pdf (2.3 MB)
   📊 performance_metrics.pptx (1.8 MB)
   📈 benchmark_results.csv (245 KB)

📤 Upload & Process → ✅ 3 files uploaded successfully
🔄 Processing in background... → ✅ 127 new chunks added
```

### **Advanced Features**
- **File Validation**: Automatic filtering of unsupported file types
- **Duplicate Detection**: 
  - Content-based: Prevents uploading files with identical content
  - Name-based: Handles files with same names intelligently
  - Cross-folder: Checks for duplicates across the entire knowledge base
- **Auto-Processing**: 
  - Immediate parsing after upload
  - Background embedding generation
  - Real-time progress tracking
  - Automatic knowledge base updates
- **Error Recovery**: Detailed error messages for failed uploads
- **Batch Processing**: Efficient handling of multiple files simultaneously
- **Progress Monitoring**: Live updates on parsing and embedding progress

## 💬 **Chat Interface**

### **Message Rendering**
The GUI automatically renders responses with proper formatting:

**Markdown Support**:
- ✅ **Headers** (`# ## ###`)
- ✅ **Bold/Italic** (`**bold**`, `*italic*`)
- ✅ **Code blocks** with syntax highlighting
- ✅ **Tables** with proper borders and alignment
- ✅ **Lists** (bullet and numbered)
- ✅ **Links** (clickable)
- ✅ **Collapsible Sources** ⭐ **NEW**: Sources hidden by default with expandable dropdown

**Example Rendered Output**:
```markdown
## Alexandria Performance Metrics

| Metric | Value | Unit |
|--------|-------|------|
| Clock Speed | 2.5 | GHz |
| Cache Size | 1MB | L2 |
| Power | 15W | TDP |

Key features:
- Advanced vector processing
- Hardware-accelerated inference
- Low-latency memory access

[1] IPM-Alexandria-270725-192147.pdf (page 15)
[2] Alexandria Concept Review.pptx (slide 8)
```

### **Input Features**
- **Auto-resize**: Text area expands as you type
- **Enter to send**: Press Enter to submit (Shift+Enter for new line)
- **Loading states**: Visual feedback during processing
- **Error handling**: Clear error messages with retry options

## 🔍 **Verbose Mode**

When enabled, verbose mode shows detailed retrieval steps in the browser console:

```javascript
// Console output example
Retrieval info: {
  model_used: "BAAI/bge-large-en-v1.5",
  dense_summary_results: 60,
  dense_full_results: 60,
  sparse_results: 60,
  fused_results: 45,
  reranked_results: 20,
  final_contexts: 8,
  generation_time_s: 2.34,
  sources_count: 8
}
```

### **Collapsible Sources** ⭐ **NEW**

The GUI now features a cleaner interface with **collapsible source citations**:

**Benefits**:
- **Cleaner Interface**: Sources are hidden by default to reduce clutter
- **On-Demand Access**: Click the "📚 Sources" button to expand citations
- **Smooth Animations**: Elegant expand/collapse transitions
- **Persistent State**: Each message has its own expandable sources section

**Usage**:
1. **Ask a Question**: Get your answer as usual
2. **View Sources**: Click the "📚 Sources ▼" button at the bottom of each response
3. **Expand/Collapse**: Toggle sources visibility as needed
4. **Multiple Messages**: Each message has independent source controls

**Example**:
```
AI Response: The Alexandria processor features a 2.5 GHz clock speed...

[📚 Sources ▼]  ← Click to expand
└── [1] alexandria_specs.pdf (page 15)
    [2] performance_metrics.pptx (slide 3)
    [3] technical_overview.md (section 2.1)
```

## 🎯 **Advanced Usage**

### **Real-time Configuration**
Changes to configuration are applied immediately to subsequent queries. The system auto-saves your preferences with smart debouncing.

### **Keyboard Shortcuts**
- **Enter**: Send message
- **Shift+Enter**: New line in message
- **Ctrl+/**: Focus message input
- **Esc**: Clear current input

### **URL Parameters** (Future Enhancement)
```
http://127.0.0.1:7860/?session=research&verbose=true&topk=15
```

## 🏗️ **Technical Architecture**

### **Frontend Stack**
- **HTML5**: Semantic structure with accessibility features
- **Modern CSS**: CSS Grid, Flexbox, CSS Variables for theming
- **Vanilla JavaScript**: Class-based ES6+ for maintainability
- **Flask Templates**: Server-side rendering with Jinja2

### **File Structure**
```
Synapse/
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   ├── css/
│   │   └── main.css        # Modern CSS with design system
│   └── js/
│       └── main.js         # SynapseApp class and utilities
└── chat.py                 # Flask server with API endpoints
```

### **API Endpoints**

| Endpoint | Method | Purpose |
|----------|---------|---------|
| `/` | GET | Main interface |
| `/api/config` | GET/POST | Configuration management |
| `/api/ask` | POST | Submit questions |
| `/api/history` | GET | Load session history |
| `/api/sessions` | GET | List available sessions |
| `/api/session/new` | POST | Create new session |
| `/api/session/load` | POST | Load specific session |
| `/api/clear` | POST | Clear current session |
| `/api/export` | GET | Export session data |

### **JavaScript Architecture**
```javascript
class SynapseApp {
  constructor()           // Initialize app and state
  init()                 // Setup event listeners and load data
  sendQuestion()         // Handle message submission
  loadConfig()          // Manage configuration state
  loadSessions()        // Session management
  showNotification()    // User feedback system
}
```

## 🎨 **Customization**

### **Theming**
The interface uses CSS custom properties for easy theming:

```css
:root {
  --bg-primary: #0a0e13;        /* Main background */
  --bg-secondary: #151921;       /* Panel backgrounds */
  --text-primary: #f3f4f6;       /* Main text */
  --accent-primary: #3b82f6;     /* Interactive elements */
  /* ... more variables */
}
```

### **Adding Custom Features**
The modular architecture makes it easy to extend:

1. **CSS**: Add styles to `static/css/main.css`
2. **JavaScript**: Extend `SynapseApp` class in `static/js/main.js`
3. **Backend**: Add new routes in `run_gui()` function
4. **Templates**: Modify `templates/index.html`

## 🚨 **Troubleshooting**

### **Common Issues**

**GUI won't start**:
```bash
# Check if Flask is installed
pip list | grep -i flask

# Verify environment variables
echo $LITELLM_API_KEY
echo $LITELLM_BASE_URL
```

**Markdown not rendering**:
```bash
# Install markdown package
pip install markdown>=3.6
```

**Session loading errors**:
- Check that `sessions/` directory exists
- Verify session file permissions
- Ensure JSON files aren't corrupted

**API errors**:
- Verify LiteLLM proxy connectivity
- Check browser developer tools for detailed errors
- Ensure embeddings file exists at specified path

### **Performance Optimization**

**For large knowledge bases**:
- Reduce `topk` to 8-10 for faster responses
- Lower `per_doc` cap to 4-6 for better diversity
- Use shorter timeout values (30-45s) for quicker feedback

**For slower systems**:
- Disable verbose mode for production use
- Consider reducing embedding dimensions in the pipeline
- Use smaller models for embedding if available

## 🔮 **Future Enhancements**

- **Streaming responses**: Real-time token generation
- **Advanced theming**: Light/dark mode toggle
- **Collaboration**: Multi-user sessions
- **Export formats**: PDF, Word, Markdown export
- **Search history**: Full-text search across all sessions
- **Bookmarks**: Save and organize important exchanges
- **Admin dashboard**: System metrics and performance monitoring

---

📚 **[Back to Main Documentation](README.md)**
