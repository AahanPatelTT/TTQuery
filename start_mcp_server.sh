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
    print_warning "No .env file found. (LiteLLM config is optional - not required for MCP server)"
fi

# Note: LiteLLM configuration is not required for MCP server
# The server only performs retrieval and returns context for external LLM generation

# Parse command line arguments
HTTP_PORT=8880
DEBUG=false
CORS=true
PRODUCTION=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --http-port)
            HTTP_PORT="$2"
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
        --production)
            PRODUCTION=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --http-port PORT                   HTTP port (default: 8880)"
            echo "  --debug                           Enable debug mode"
            echo "  --no-cors                         Disable CORS"
            echo "  --production                      Use production WSGI server (gunicorn)"
            echo "  --help, -h                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                # Start with default settings"
            echo "  $0 --http-port 8080               # Use custom port"
            echo "  $0 --debug                        # Debug mode"
            echo "  $0 --production                   # Production mode"
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
print_status "  HTTP Port: $HTTP_PORT"
print_status "  Debug Mode: $DEBUG"
print_status "  Production Mode: $PRODUCTION"
print_status "  CORS Enabled: $CORS"
echo ""

# Start the MCP server
print_status "Starting MCP server..."
echo "📍 HTTP: http://localhost:$HTTP_PORT/mcp"
echo "⏹️  Press Ctrl+C to stop"
echo ""

# Build the command
CMD="python mcp_server.py --http-port $HTTP_PORT"
if [ "$DEBUG" = true ]; then
    CMD="$CMD --debug"
fi
if [ "$CORS" = false ]; then
    CMD="$CMD --no-cors"
fi
if [ "$PRODUCTION" = true ]; then
    CMD="$CMD --production"
fi

# Execute the command
exec $CMD
