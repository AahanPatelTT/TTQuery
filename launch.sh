#!/bin/bash

# Synapse Launch Script
# This script initializes the virtual environment and launches the Synapse RAG application

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    if command_exists python3; then
        python_version=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Found Python version: $python_version"
        
        # Check if version is 3.9 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
            print_success "Python version is compatible (3.9+)"
            return 0
        else
            print_error "Python version must be 3.9 or higher. Found: $python_version"
            return 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.9 or higher."
        return 1
    fi
}

# Function to check macOS dependencies
check_macos_dependencies() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        print_status "Checking macOS dependencies..."
        
        local missing_deps=()
        
        if ! command_exists brew; then
            print_warning "Homebrew not found. Please install Homebrew first: https://brew.sh/"
            print_warning "Then run: brew install libmagic poppler tesseract"
            missing_deps+=("brew")
        else
            if ! brew list libmagic >/dev/null 2>&1; then
                missing_deps+=("libmagic")
            fi
            if ! brew list poppler >/dev/null 2>&1; then
                missing_deps+=("poppler")
            fi
            if ! brew list tesseract >/dev/null 2>&1; then
                missing_deps+=("tesseract")
            fi
        fi
        
        if [ ${#missing_deps[@]} -gt 0 ]; then
            print_warning "Missing macOS dependencies: ${missing_deps[*]}"
            print_status "Installing missing dependencies..."
            brew install "${missing_deps[@]}"
            print_success "macOS dependencies installed"
        else
            print_success "All macOS dependencies are installed"
        fi
    fi
}

# Function to setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ ! -d ".venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv .venv
        print_success "Virtual environment created"
    else
        print_status "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source .venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    print_success "Virtual environment setup complete"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ ! -f "requirements.txt" ]; then
        print_error "requirements.txt not found in current directory"
        exit 1
    fi
    
    # Install dependencies
    pip install -r requirements.txt
    
    print_success "Dependencies installed successfully"
}

# Function to check environment variables
check_environment_vars() {
    print_status "Checking environment variables..."
    
    local missing_vars=()
    
    if [ -z "$LITELLM_API_KEY" ]; then
        missing_vars+=("LITELLM_API_KEY")
    fi
    
    if [ -z "$LITELLM_BASE_URL" ]; then
        missing_vars+=("LITELLM_BASE_URL")
    fi
    
    if [ ${#missing_vars[@]} -gt 0 ]; then
        print_warning "Missing environment variables: ${missing_vars[*]}"
        print_status "Please set the following environment variables:"
        echo "export LITELLM_API_KEY=your_api_key_here"
        echo "export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/"
        echo ""
        print_status "You can also create a .env file in the project root with these variables."
        
        # Check if .env file exists
        if [ -f ".env" ]; then
            print_status "Loading environment variables from .env file..."
            export $(cat .env | grep -v '^#' | xargs)
            
            # Check again after loading .env
            if [ -z "$LITELLM_API_KEY" ] || [ -z "$LITELLM_BASE_URL" ]; then
                print_error "Required environment variables still missing after loading .env file"
                print_status "Please set LITELLM_API_KEY and LITELLM_BASE_URL and try again"
                exit 1
            fi
        else
            print_error "Required environment variables not set. Please set them and try again."
            exit 1
        fi
    fi
    
    print_success "Environment variables are properly configured"
}

# Function to initialize knowledge base
initialize_knowledge_base() {
    print_status "Initializing knowledge base..."
    
    # Use fast initialization by default
    print_status "Running fast incremental initialization..."
    python initialize_fast.py --verbose
        
    
    if [ $? -eq 0 ]; then
        print_success "Knowledge base initialized successfully"
    else
        print_error "Knowledge base initialization failed"
        print_status "Falling back to legacy initialization..."
        python initialize_fast.py
        if [ $? -ne 0 ]; then
            exit 1
        fi
    fi
}

# Function to launch the application
launch_app() {
    print_status "Launching Synapse application..."
    
    # Check which mode is requested
    case "$1" in
        "--gui"|"-g"|"gui")
            print_status "Starting web GUI interface..."
            python chat.py --test_gui
            ;;
        "fast")
            print_status "Running fast incremental initialization..."
            python initialize_fast.py --verbose
            ;;
        "status")
            print_status "Checking processing status..."
            python initialize_fast.py --status
            ;;
        "upload")
            print_status "Starting web upload server..."
            echo "📍 URL: http://localhost:5000"
            echo "⏹️  Press Ctrl+C to stop"
            python web_upload.py
            ;;
        "mcp")
            print_status "Starting MCP server (both HTTP and WebSocket)..."
            echo "📍 HTTP: http://localhost:3000/mcp"
            echo "📍 WebSocket: ws://localhost:3001"
            echo "⏹️  Press Ctrl+C to stop"
            exec ./start_mcp_server.sh
            ;;
        "mcp-http")
            print_status "Starting MCP server (HTTP only)..."
            echo "📍 URL: http://localhost:3000/mcp"
            echo "⏹️  Press Ctrl+C to stop"
            python mcp_server.py --transport http --port 3000
            ;;
        "mcp-websocket")
            print_status "Starting MCP server (WebSocket only)..."
            echo "📍 URL: ws://localhost:3001"
            echo "⏹️  Press Ctrl+C to stop"
            python mcp_server.py --transport websocket --port 3001
            ;;
        *)
            print_status "Starting CLI interface..."
            python chat.py
            ;;
    esac
}

# Main execution
main() {
    echo ""
    echo "██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗"
    echo "██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝"
    echo "██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   "
    echo "██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   "
    echo "██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   "
    echo "╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   "
    echo ""
    echo "              SYNAPSE LAUNCH SCRIPT"
    echo "================================================================"
    echo ""
    
    # Check if we're in the right directory
    if [ ! -f "chat.py" ] || [ ! -f "initialize_fast.py" ]; then
        print_error "This script must be run from the Synapse project root directory"
        print_error "Please navigate to the Synapse directory and try again"
        exit 1
    fi
    
    # Check Python version
    check_python_version || exit 1
    
    # Check macOS dependencies
    check_macos_dependencies
    
    # Setup virtual environment
    setup_venv
    
    # Install dependencies
    install_dependencies
    
    # Check environment variables
    check_environment_vars
    
    # Initialize knowledge base
    initialize_knowledge_base
    
    # Launch the application
    launch_app "$1"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --gui, -g     Launch the web GUI interface"
        echo "  --fast        Use fast incremental initialization (recommended)"
        echo "  --status      Show processing status"
        echo "  --upload      Start web upload server for real-time document uploads"
        echo "  --mcp         Start MCP server (both HTTP and WebSocket transports)"
        echo "  --mcp-http    Start MCP server (HTTP transport only)"
        echo "  --mcp-ws      Start MCP server (WebSocket transport only)"
        echo "  --help, -h    Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0              # Launch CLI interface"
        echo "  $0 --gui        # Launch web GUI interface"
        echo "  $0 --fast       # Fast incremental initialization"
        echo "  $0 --status     # Show processing status"
        echo "  $0 --upload     # Start web upload server"
        echo "  $0 --mcp        # Start MCP server (HTTP + WebSocket)"
        echo "  $0 --mcp-http   # Start MCP server (HTTP only)"
        exit 0
        ;;
    --fast)
        main "fast"
        ;;
    --status)
        main "status"
        ;;
    --upload)
        main "upload"
        ;;
    --mcp)
        main "mcp"
        ;;
    --mcp-http)
        main "mcp-http"
        ;;
    --mcp-ws|--mcp-websocket)
        main "mcp-websocket"
        ;;
    *)
        main "$1"
        ;;
esac
