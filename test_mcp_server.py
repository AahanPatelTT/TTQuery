#!/usr/bin/env python3
"""
Synapse MCP Server Test Suite

A comprehensive test suite to validate MCP server functionality, covering:
- Import validation and dependency checking
- Server initialization and configuration
- MCP protocol message handling
- Knowledge base discovery and management
- Environment and file validation

Usage:
    python test_mcp_server.py                  # Run all tests
    python test_mcp_server.py --verbose        # Run with detailed output
    python test_mcp_server.py --quick          # Run essential tests only
    python test_mcp_server.py --test imports   # Run specific test category
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Optional

def test_imports():
    """Test that all required imports work"""
    print("🔍 Testing imports...")
    
    try:
        # Core MCP server imports
        from mcp_server import MCPServer, HTTPTransport, WebSocketTransport
        print("✅ MCP server imports successful")
        
        # Required dependencies 
        import flask
        import websockets
        import asyncio
        print("✅ Transport dependencies available")
        
        # Synapse components
        from chat import ChatSession
        from pipeline.query import load_corpus, list_available_knowledge_bases
        print("✅ Synapse components available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_mcp_server_init():
    """Test MCP server initialization"""
    print("\n🔍 Testing MCP server initialization...")
    
    try:
        from mcp_server import MCPServer
        
        # Initialize server
        server = MCPServer(artifacts_dir="artifacts")
        
        # Check basic properties
        assert len(server.tools) > 0, "No tools defined"
        assert server.artifacts_dir == "artifacts", "Artifacts dir not set"
        
        print(f"✅ MCP server initialized with {len(server.tools)} tools")
        
        # Test tool definitions
        tool_names = [tool.name for tool in server.tools]
        expected_tools = [
            "ask_question", "list_knowledge_bases", "switch_knowledge_base",
            "create_session", "load_session", "export_session", "get_server_info"
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"
        
        print("✅ All expected tools are defined")
        return True
        
    except Exception as e:
        print(f"❌ MCP server initialization failed: {e}")
        return False


def test_message_handling():
    """Test MCP message handling"""
    print("\n🔍 Testing MCP message handling...")
    
    try:
        import asyncio
        from mcp_server import MCPServer
        
        server = MCPServer(artifacts_dir="artifacts")
        
        # Test initialize message
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        async def test_async():
            response = await server.handle_message(init_message)
            assert response["jsonrpc"] == "2.0", "Invalid JSON-RPC version"
            assert response["id"] == 1, "Incorrect message ID"
            assert "result" in response, "No result in response"
            return response
        
        response = asyncio.run(test_async())
        print("✅ Initialize message handled correctly")
        
        # Test tools/list message
        list_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }
        
        async def test_list_tools():
            response = await server.handle_message(list_message)
            assert "result" in response, "No result in response"
            assert "tools" in response["result"], "No tools in response"
            return response
        
        response = asyncio.run(test_list_tools())
        tools_count = len(response["result"]["tools"])
        print(f"✅ Tools list retrieved ({tools_count} tools)")
        
        return True
        
    except Exception as e:
        print(f"❌ Message handling test failed: {e}")
        return False


def test_configuration_files():
    """Test configuration files exist and are valid"""
    print("\n🔍 Testing configuration files...")
    
    try:
        # Test mcp_config.json
        if os.path.exists("mcp_config.json"):
            with open("mcp_config.json") as f:
                config = json.load(f)
            
            assert "server" in config, "Missing server config"
            assert "transports" in config, "Missing transports config"
            print("✅ mcp_config.json is valid")
        else:
            print("⚠️  mcp_config.json not found (optional)")
        
        # Test startup script
        if os.path.exists("start_mcp_server.sh"):
            print("✅ start_mcp_server.sh exists")
        else:
            print("❌ start_mcp_server.sh not found")
            
        # Test example client
        if os.path.exists("mcp_client_example.py"):
            print("✅ mcp_client_example.py exists")
        else:
            print("❌ mcp_client_example.py not found")
            
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


def test_environment():
    """Test required environment variables"""
    print("\n🔍 Testing environment...")
    
    required_vars = ["LITELLM_API_KEY", "LITELLM_BASE_URL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("   Set these for full MCP server functionality:")
        print("   export LITELLM_API_KEY=your_api_key")
        print("   export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/")
        return False
    else:
        print("✅ All required environment variables are set")
        return True


def test_knowledge_base():
    """Test knowledge base availability"""
    print("\n🔍 Testing knowledge base...")
    
    try:
        artifacts_dir = "artifacts"
        
        if not os.path.exists(artifacts_dir):
            print(f"⚠️  Artifacts directory '{artifacts_dir}' not found")
            print("   Run: python initialize_fast.py")
            return False
            
        # Look for embedding files
        import glob
        embedding_files = glob.glob(os.path.join(artifacts_dir, "embedded_*.jsonl"))
        
        if embedding_files:
            print(f"✅ Found {len(embedding_files)} knowledge base(s)")
            for file in embedding_files[:3]:  # Show first 3
                print(f"   • {os.path.basename(file)}")
            if len(embedding_files) > 3:
                print(f"   • ... and {len(embedding_files) - 3} more")
            return True
        else:
            print("⚠️  No knowledge bases found")
            print("   Run: python initialize_fast.py")
            return False
            
    except Exception as e:
        print(f"❌ Knowledge base test failed: {e}")
        return False


def test_tool_schemas():
    """Test that all MCP tools have valid schemas"""
    print("\n🔍 Testing tool schemas...")
    
    try:
        from mcp_server import MCPServer
        
        server = MCPServer(artifacts_dir="artifacts")
        
        required_fields = ["name", "description", "inputSchema"]
        schema_issues = []
        
        for tool in server.tools:
            # Check required fields
            for field in required_fields:
                if not hasattr(tool, field) or not getattr(tool, field):
                    schema_issues.append(f"Tool '{tool.name}' missing {field}")
            
            # Validate input schema structure
            if hasattr(tool, 'inputSchema'):
                schema = tool.inputSchema
                if not isinstance(schema, dict):
                    schema_issues.append(f"Tool '{tool.name}' has invalid schema type")
                elif "type" not in schema:
                    schema_issues.append(f"Tool '{tool.name}' schema missing 'type' field")
        
        if schema_issues:
            for issue in schema_issues:
                print(f"❌ {issue}")
            return False
        
        print(f"✅ All {len(server.tools)} tool schemas are valid")
        return True
        
    except Exception as e:
        print(f"❌ Tool schema validation failed: {e}")
        return False


def test_server_error_handling():
    """Test MCP server error handling"""
    print("\n🔍 Testing error handling...")
    
    try:
        import asyncio
        from mcp_server import MCPServer
        
        server = MCPServer(artifacts_dir="artifacts")
        
        # Test invalid JSON-RPC message
        invalid_messages = [
            {"invalid": "message"},  # Missing jsonrpc
            {"jsonrpc": "1.0", "method": "test"},  # Wrong version
            {"jsonrpc": "2.0", "method": "nonexistent"},  # Invalid method
        ]
        
        async def test_error_messages():
            for msg in invalid_messages:
                response = await server.handle_message(msg)
                if "error" not in response:
                    return False
            return True
        
        result = asyncio.run(test_error_messages())
        
        if result:
            print("✅ Error handling works correctly")
            return True
        else:
            print("❌ Error handling failed")
            return False
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


class TestRunner:
    """Enhanced test runner with filtering and reporting"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.tests = {
            "essential": [
                ("Import Test", test_imports),
                ("MCP Server Init", test_mcp_server_init),
                ("Environment Variables", test_environment),
            ],
            "core": [
                ("Message Handling", test_message_handling),
                ("Tool Schemas", test_tool_schemas),
                ("Error Handling", test_server_error_handling),
            ],
            "integration": [
                ("Configuration Files", test_configuration_files),
                ("Knowledge Base", test_knowledge_base),
            ]
        }
    
    def run_category(self, category: str) -> tuple[int, int]:
        """Run tests in a specific category"""
        if category not in self.tests:
            print(f"❌ Unknown test category: {category}")
            return 0, 0
        
        tests = self.tests[category]
        passed = 0
        
        print(f"\n📋 Running {category.upper()} tests ({len(tests)} tests)")
        print("-" * 40)
        
        for test_name, test_func in tests:
            try:
                start_time = time.time()
                result = test_func()
                duration = time.time() - start_time
                
                if result:
                    passed += 1
                    if self.verbose:
                        print(f"   ✅ {test_name} ({duration:.2f}s)")
                else:
                    if self.verbose:
                        print(f"   ❌ {test_name} ({duration:.2f}s)")
                        
            except Exception as e:
                if self.verbose:
                    print(f"   💥 {test_name} crashed: {e}")
                print(f"❌ {test_name} crashed: {e}")
        
        return passed, len(tests)
    
    def run_all_tests(self) -> bool:
        """Run all test categories"""
        print("🚀 Synapse MCP Server Test Suite")
        print("=" * 60)
        
        total_passed = 0
        total_tests = 0
        
        for category in ["essential", "core", "integration"]:
            passed, count = self.run_category(category)
            total_passed += passed
            total_tests += count
        
        # Summary
        print("\n" + "=" * 60)
        print(f"📊 OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            print("🎉 All tests passed! MCP server is ready for deployment.")
            self._print_next_steps()
            return True
        else:
            print("⚠️  Some tests failed. See details above.")
            self._print_troubleshooting()
            return False
    
    def _print_next_steps(self):
        """Print next steps for successful tests"""
        print("\n🚀 NEXT STEPS:")
        print("   1. Start MCP server:")
        print("      ./start_mcp_server.sh")
        print("      # or ./launch.sh --mcp")
        print("   2. Test with demo client:")
        print("      python mcp_client_example.py --demo")
        print("   3. Try interactive mode:")
        print("      python mcp_client_example.py --interactive")
        print("   4. Integrate with AI models using MCP protocol")
    
    def _print_troubleshooting(self):
        """Print troubleshooting information"""
        print("\n🔧 TROUBLESHOOTING:")
        print("   • Missing dependencies:")
        print("     pip install -r requirements.txt")
        print("   • Environment variables not set:")
        print("     export LITELLM_API_KEY=your_key")
        print("     export LITELLM_BASE_URL=your_url")
        print("   • No knowledge bases found:")
        print("     python initialize_fast.py")
        print("   • Permission issues:")
        print("     chmod +x start_mcp_server.sh")


def main():
    parser = argparse.ArgumentParser(description="Synapse MCP Server Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with timing information")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run essential tests only")
    parser.add_argument("--test", choices=["essential", "core", "integration"],
                       help="Run specific test category")
    
    args = parser.parse_args()
    
    runner = TestRunner(verbose=args.verbose)
    
    if args.test:
        # Run specific category
        passed, total = runner.run_category(args.test)
        success = passed == total
        print(f"\n📊 {args.test.upper()} RESULTS: {passed}/{total} tests passed")
    elif args.quick:
        # Run essential tests only
        passed, total = runner.run_category("essential")
        success = passed == total
        print(f"\n📊 QUICK TEST RESULTS: {passed}/{total} tests passed")
    else:
        # Run all tests
        success = runner.run_all_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
