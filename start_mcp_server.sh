#!/bin/bash

# Synapse MCP Server Startup Script
# This script starts the MCP server with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[MCP]${NC} $1"
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

# Default configuration
TRANSPORT="both"
HTTP_PORT=3000
WS_PORT=3001
ARTIFACTS_DIR="artifacts"
DEBUG=false
CONFIG_FILE="mcp_config.json"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --transport)
            TRANSPORT="$2"
            shift 2
            ;;
        --http-port)
            HTTP_PORT="$2"
            shift 2
            ;;
        --ws-port)
            WS_PORT="$2"
            shift 2
            ;;
        --artifacts-dir)
            ARTIFACTS_DIR="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --transport TRANSPORT     Transport to use (http|websocket|both) [default: both]"
            echo "  --http-port PORT          HTTP server port [default: 3000]"
            echo "  --ws-port PORT            WebSocket server port [default: 3001]"
            echo "  --artifacts-dir DIR       Artifacts directory [default: artifacts]"
            echo "  --debug                   Enable debug mode"
            echo "  --config FILE             Configuration file [default: mcp_config.json]"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                        # Start both HTTP and WebSocket servers"
            echo "  $0 --transport http       # Start only HTTP server"
            echo "  $0 --debug               # Start with debug logging"
            echo "  $0 --http-port 8080      # Use custom HTTP port"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print banner
echo ""
echo "██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗"
echo "██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝"
echo "██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   "
echo "██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   "
echo "██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   "
echo "╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   "
echo ""
echo "              MCP SERVER STARTUP"
echo "================================================================"
echo ""

# Check if we're in the right directory
if [ ! -f "mcp_server.py" ]; then
    print_error "mcp_server.py not found. Please run from the Synapse root directory."
    exit 1
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
print_status "Python version: $python_version"

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status "Virtual environment: $VIRTUAL_ENV"
else
    print_warning "No virtual environment detected. Consider activating .venv"
fi

# Check required environment variables
missing_vars=()
if [ -z "$LITELLM_API_KEY" ]; then
    missing_vars+=("LITELLM_API_KEY")
fi
if [ -z "$LITELLM_BASE_URL" ]; then
    missing_vars+=("LITELLM_BASE_URL")
fi

if [ ${#missing_vars[@]} -gt 0 ]; then
    print_error "Missing required environment variables: ${missing_vars[*]}"
    print_status "Please set:"
    echo "  export LITELLM_API_KEY=your_api_key"
    echo "  export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/"
    exit 1
fi

print_success "Environment variables configured"

# Check if artifacts directory exists
if [ ! -d "$ARTIFACTS_DIR" ]; then
    print_warning "Artifacts directory '$ARTIFACTS_DIR' not found"
    print_status "Running fast initialization to create knowledge base..."
    python3 initialize_fast.py --verbose
    if [ $? -ne 0 ]; then
        print_error "Failed to initialize knowledge base"
        exit 1
    fi
fi

# Check for available knowledge bases
kb_count=$(find "$ARTIFACTS_DIR" -name "embedded_*.jsonl" 2>/dev/null | wc -l)
if [ "$kb_count" -eq 0 ]; then
    print_warning "No knowledge bases found in '$ARTIFACTS_DIR'"
    print_status "You may need to run initialization first:"
    echo "  python3 initialize_fast.py"
else
    print_success "Found $kb_count knowledge base(s)"
fi

# Check dependencies
print_status "Checking MCP server dependencies..."
python3 -c "
try:
    import flask, websockets, json, asyncio
    print('✅ All dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    print('Run: pip install flask flask-cors websockets')
    exit(1)
" || exit 1

# Display server configuration
print_status "Server configuration:"
echo "  Transport: $TRANSPORT"
if [[ "$TRANSPORT" == "http" || "$TRANSPORT" == "both" ]]; then
    echo "  HTTP Port: $HTTP_PORT"
    echo "  HTTP URL: http://localhost:$HTTP_PORT/mcp"
fi
if [[ "$TRANSPORT" == "websocket" || "$TRANSPORT" == "both" ]]; then
    echo "  WebSocket Port: $WS_PORT"
    echo "  WebSocket URL: ws://localhost:$WS_PORT"
fi
echo "  Artifacts: $ARTIFACTS_DIR"
echo "  Debug: $DEBUG"

# Build command
CMD_ARGS=(
    "--transport" "$TRANSPORT"
    "--artifacts-dir" "$ARTIFACTS_DIR"
)

if [[ "$TRANSPORT" == "http" || "$TRANSPORT" == "both" ]]; then
    CMD_ARGS+=("--http-port" "$HTTP_PORT")
fi

if [[ "$TRANSPORT" == "websocket" || "$TRANSPORT" == "both" ]]; then
    CMD_ARGS+=("--ws-port" "$WS_PORT")
fi

if [ "$DEBUG" = true ]; then
    CMD_ARGS+=("--debug")
fi

print_status "Starting MCP server..."
echo "Command: python3 mcp_server.py ${CMD_ARGS[*]}"
echo ""

# Start the server
exec python3 mcp_server.py "${CMD_ARGS[@]}"
