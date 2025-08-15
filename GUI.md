# TTQuery Web GUI Documentation

**Modern Web Interface for TTQuery RAG System**

The TTQuery web GUI provides a polished, user-friendly interface for interacting with your document knowledge base. Built with modern web technologies, it offers all CLI functionality plus enhanced visual features and real-time configuration.

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

![TTQuery GUI Layout](https://i.imgur.com/placeholder.png)

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
TTQuery/
├── templates/
│   └── index.html          # Main HTML template
├── static/
│   ├── css/
│   │   └── main.css        # Modern CSS with design system
│   └── js/
│       └── main.js         # TTQueryApp class and utilities
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
class TTQueryApp {
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
2. **JavaScript**: Extend `TTQueryApp` class in `static/js/main.js`
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
