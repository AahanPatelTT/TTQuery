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

# Check if we're in the right directory
if [ ! -f "mcp_server.py" ]; then
    print_error "This script must be run from the Synapse project root directory"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment not found. Please run ./launch.sh first to set up the environment."
    exit 1
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate

# Load environment variables
if [ -f ".env" ]; then
    print_status "Loading environment variables from .env file..."
    set -a
    source .env
    set +a
    print_success "Environment variables loaded"
else
    print_warning "No .env file found. Make sure LITELLM_API_KEY and LITELLM_BASE_URL are set."
fi

# Check required environment variables
if [ -z "$LITELLM_API_KEY" ] || [ -z "$LITELLM_BASE_URL" ]; then
    print_error "Required environment variables not set:"
    print_error "  LITELLM_API_KEY: ${LITELLM_API_KEY:+SET}"
    print_error "  LITELLM_BASE_URL: ${LITELLM_BASE_URL:+SET}"
    print_error "Please set these variables and try again."
    exit 1
fi

# Parse command line arguments
TRANSPORT="both"
HTTP_PORT=3000
WS_PORT=3001
DEBUG=false
CORS=true

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
        --debug)
            DEBUG=true
            shift
            ;;
        --no-cors)
            CORS=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --transport {http|websocket|both}  Transport protocol (default: both)"
            echo "  --http-port PORT                   HTTP port (default: 3000)"
            echo "  --ws-port PORT                     WebSocket port (default: 3001)"
            echo "  --debug                           Enable debug mode"
            echo "  --no-cors                         Disable CORS"
            echo "  --help, -h                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                # Start with both transports"
            echo "  $0 --transport http               # HTTP only"
            echo "  $0 --transport websocket          # WebSocket only"
            echo "  $0 --debug                        # Debug mode"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display startup information
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

print_status "Starting MCP server with configuration:"
print_status "  Transport: $TRANSPORT"
print_status "  HTTP Port: $HTTP_PORT"
print_status "  WebSocket Port: $WS_PORT"
print_status "  Debug Mode: $DEBUG"
print_status "  CORS Enabled: $CORS"
echo ""

# Start the MCP server
print_status "Starting MCP server..."
echo "📍 HTTP: http://localhost:$HTTP_PORT/mcp"
echo "📍 WebSocket: ws://localhost:$WS_PORT"
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Build the command
CMD="python mcp_server.py --transport $TRANSPORT --http-port $HTTP_PORT --ws-port $WS_PORT"
if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi
if [ "$CORS" = false ]; then
    CMD="$CMD --no-cors"
fi

# Execute the command
exec $CMD
