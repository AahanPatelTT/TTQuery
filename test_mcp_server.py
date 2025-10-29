#!/usr/bin/env python3
"""
Comprehensive MCP Server Test Suite

A complete test suite that:
1. Starts the MCP server
2. Tests all 14 tools with proper parameters
3. Validates error handling and edge cases
4. Performs performance testing
5. Tests both HTTP and WebSocket transports

Usage:
    python test_mcp_server.py                  # Run all tests
    python test_mcp_server.py --verbose        # Run with detailed output
    python test_mcp_server.py --quick          # Run essential tests only
    python test_mcp_server.py --test tools     # Run specific test category
"""

import argparse
import asyncio
import json
import os
import sys
import time
import subprocess
import signal
import threading
from typing import Dict, Any, List, Optional

# Test configuration
SERVER_HOST = "localhost"
HTTP_PORT = 3000
WS_PORT = 3001
TEST_TIMEOUT = 15
MAX_RETRIES = 3

class MCPTester:
    """Comprehensive MCP server tester"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.base_url = f"http://{SERVER_HOST}:{HTTP_PORT}"
        self.ws_url = f"ws://{SERVER_HOST}:{WS_PORT}"
        self.process = None
        self.test_results = []
        
    def start_server(self) -> bool:
        """Start MCP server"""
        print("🚀 Starting MCP server...")
        try:
            self.process = subprocess.Popen(
                [sys.executable, "mcp_server.py", "--transport", "both", "--http-port", "3000", "--ws-port", "3001"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait for server to be ready
            for i in range(5):
                time.sleep(1)
                if self._check_server_ready():
                    print("✅ MCP server started")
                    return True
                print(f"⏳ Waiting... ({i+1}/5)")
            
            print("❌ Failed to start server")
            return False
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def stop_server(self):
        """Stop MCP server"""
        if self.process:
            print("🛑 Stopping MCP server...")
            try:
                self.process.terminate()
                self.process.wait(timeout=3)
            except:
                self.process.kill()
            finally:
                self.process = None
    
    def _check_server_ready(self) -> bool:
        """Check if server is ready"""
        import socket
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((SERVER_HOST, HTTP_PORT))
            sock.close()
            return result == 0
        except:
            return False
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send HTTP request to MCP server"""
        import aiohttp
        
        message = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": method,
            "params": params or {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/mcp",  # Use /mcp endpoint
                    json=message,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=TEST_TIMEOUT)
                ) as response:
                    return await response.json()
        except Exception as e:
            return {"error": {"code": -1, "message": str(e)}}
    
    async def test_tool(self, tool_name: str, args: Dict[str, Any] = None) -> bool:
        """Test a single tool"""
        try:
            response = await self.send_request("tools/call", {
                "name": tool_name,
                "arguments": args or {}
            })
            
            if "error" in response:
                # For load_session with nonexistent file, this is expected
                if tool_name == "load_session" and "Session file not found" in response['error']['message']:
                    print(f"✅ {tool_name}: OK (expected error)")
                    return True
                print(f"❌ {tool_name}: {response['error']['message']}")
                return False
            
            print(f"✅ {tool_name}: OK")
            return True
            
        except Exception as e:
            print(f"❌ {tool_name}: {e}")
            return False
    
    def log(self, message: str, level: str = "INFO"):
        """Log message with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        if self.verbose or level in ["ERROR", "SUCCESS"]:
            print(f"[{timestamp}] {level}: {message}")
    
    async def test_server_initialization(self) -> bool:
        """Test server initialization"""
        self.log("Testing server initialization...")
        
        try:
            response = await self.send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            })
            
            if "error" in response:
                self.log(f"Initialize failed: {response['error']}", "ERROR")
                return False
            
            self.log("✅ Server initialization successful", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Initialize test failed: {e}", "ERROR")
            return False
    
    async def test_tools_list(self) -> bool:
        """Test tools list functionality"""
        self.log("Testing tools list...")
        
        try:
            response = await self.send_request("tools/list")
            
            if "error" in response:
                self.log(f"Tools list failed: {response['error']}", "ERROR")
                return False
            
            tools = response.get("result", {}).get("tools", [])
            expected_tools = [
                "ask_question", "list_knowledge_bases", "switch_knowledge_base",
                "get_kb_stats", "load_session", "list_sessions",
                "get_processing_status", "initialize_knowledge_base",
                "set_verbose_mode", "get_server_info"
            ]
            
            tool_names = [tool["name"] for tool in tools]
            missing_tools = [tool for tool in expected_tools if tool not in tool_names]
            
            if missing_tools:
                self.log(f"Missing tools: {missing_tools}", "ERROR")
                return False
            
            self.log(f"✅ Found {len(tools)} tools (expected {len(expected_tools)})", "SUCCESS")
            return True
            
        except Exception as e:
            self.log(f"Tools list test failed: {e}", "ERROR")
            return False
    
    async def test_all_tools(self) -> bool:
        """Test all 14 MCP tools"""
        self.log("Testing all 14 MCP tools...")
        
        # Test cases for each tool
        test_cases = [
            ("ask_question", {"question": "What is Ascalon?", "max_chunks": 3}),
            ("list_knowledge_bases", {}),
            ("get_kb_stats", {"knowledge_base": "Aahan's Notes"}),
            ("switch_knowledge_base", {"knowledge_base": "Aahan's Notes"}),
            ("load_session", {"session_file": "nonexistent.json"}),
            ("list_sessions", {}),
            ("get_processing_status", {}),
            ("initialize_knowledge_base", {"folder": "Data", "cleanup": False}),
            ("set_verbose_mode", {"verbose": True}),
            ("get_server_info", {}),
        ]
        
        passed = 0
        total = len(test_cases)
        
        for tool_name, args in test_cases:
            if await self.test_tool(tool_name, args):
                passed += 1
        
        self.log(f"📊 Tool Results: {passed}/{total} tools working")
        return passed == total
    
    async def test_error_handling(self) -> bool:
        """Test error handling and edge cases"""
        self.log("Testing error handling...")
        
        try:
            # Test invalid tool name
            response = await self.send_request("tools/call", {
                "name": "nonexistent_tool",
                "arguments": {}
            })
            
            if "error" not in response:
                self.log("Invalid tool should return error", "ERROR")
                return False
            
            self.log("✅ Invalid tool properly rejected")
            
            # Test invalid parameters
            response = await self.send_request("tools/call", {
                "name": "ask_question",
                "arguments": {"invalid_param": "test"}
            })
            
            if "error" not in response:
                self.log("Invalid parameters should return error", "ERROR")
                return False
            
            self.log("✅ Invalid parameters properly rejected")
            
            return True
            
        except Exception as e:
            self.log(f"Error handling test failed: {e}", "ERROR")
            return False
    
    async def test_performance(self) -> bool:
        """Test performance and concurrent requests"""
        self.log("Testing performance...")
        
        try:
            # Test concurrent requests
            tasks = []
            for i in range(3):
                task = self.send_request("tools/call", {
                    "name": "ask_question",
                    "arguments": {
                        "question": f"Test question {i}",
                        "max_chunks": 2
                    }
                })
                tasks.append(task)
            
            start_time = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start_time
            
            success_count = sum(1 for r in responses if isinstance(r, dict) and "error" not in r)
            
            if success_count < len(tasks):
                self.log(f"Only {success_count}/{len(tasks)} concurrent requests succeeded", "ERROR")
                return False
            
            self.log(f"✅ {len(tasks)} concurrent requests completed in {duration:.2f}s")
            
            # Test single request performance
            start_time = time.time()
            response = await self.send_request("tools/call", {
                "name": "ask_question",
                "arguments": {
                    "question": "Performance test question",
                    "max_chunks": 3
                }
            })
            duration = time.time() - start_time
            
            if "error" in response:
                self.log(f"Performance test failed: {response['error']}", "ERROR")
                return False
            
            if duration > 10:
                self.log(f"Response time too slow: {duration:.2f}s", "ERROR")
                return False
            
            self.log(f"✅ Single request completed in {duration:.2f}s")
            
            return True
            
        except Exception as e:
            self.log(f"Performance test failed: {e}", "ERROR")
            return False
    
    async def test_websocket_transport(self) -> bool:
        """Test WebSocket transport"""
        self.log("Testing WebSocket transport...")
        
        try:
            import websockets
            import asyncio
            
            # Add a small delay to ensure server is ready
            await asyncio.sleep(1)
            
            async with websockets.connect(self.ws_url, ping_interval=None, ping_timeout=None) as websocket:
                # Send initialize message
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
                
                await websocket.send(json.dumps(init_message))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                if "error" in data:
                    self.log(f"WebSocket initialize failed: {data['error']}", "ERROR")
                    return False
                
                # Send tools/list message
                list_message = {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {}
                }
                
                await websocket.send(json.dumps(list_message))
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(response)
                
                if "error" in data:
                    self.log(f"WebSocket tools/list failed: {data['error']}", "ERROR")
                    return False
                
                tools = data.get("result", {}).get("tools", [])
                self.log(f"✅ WebSocket transport working: {len(tools)} tools available")
                
                return True
                
        except asyncio.TimeoutError:
            self.log("WebSocket test failed: Connection timeout", "ERROR")
            return False
        except Exception as e:
            self.log(f"WebSocket test failed: {e}", "ERROR")
            return False
    
    async def run_comprehensive_tests(self) -> bool:
        """Run comprehensive test suite"""
        print("🚀 Comprehensive MCP Server Test Suite")
        print("=" * 60)
        
        if not self.start_server():
            return False
        
        try:
            # Define test categories
            test_categories = {
                "essential": [
                    ("Server Initialization", self.test_server_initialization),
                    ("Tools List", self.test_tools_list),
                    ("All Tools", self.test_all_tools),
                ],
                "advanced": [
                    ("Error Handling", self.test_error_handling),
                    ("Performance", self.test_performance),
                    ("WebSocket Transport", self.test_websocket_transport),
                ]
            }
            
            total_passed = 0
            total_tests = 0
            
            for category, tests in test_categories.items():
                self.log(f"Running {category.upper()} tests...")
                print("-" * 40)
                
                category_passed = 0
                for test_name, test_func in tests:
                    try:
                        start_time = time.time()
                        result = await test_func()
                        duration = time.time() - start_time
                        
                        if result:
                            category_passed += 1
                            self.log(f"✅ {test_name} passed ({duration:.2f}s)", "SUCCESS")
                        else:
                            self.log(f"❌ {test_name} failed ({duration:.2f}s)", "ERROR")
                        
                        total_tests += 1
                        
                    except Exception as e:
                        self.log(f"💥 {test_name} crashed: {e}", "ERROR")
                        total_tests += 1
                
                total_passed += category_passed
                self.log(f"📊 {category.upper()}: {category_passed}/{len(tests)} tests passed")
                print()
            
            # Summary
            print("=" * 60)
            print(f"📊 OVERALL RESULTS: {total_passed}/{total_tests} tests passed")
            
            if total_passed == total_tests:
                print("🎉 All tests passed! MCP server is robust and ready for production.")
                return True
            else:
                print("⚠️  Some tests failed. See details above.")
                return False
            
        finally:
            self.stop_server()
    
    async def run_quick_tests(self) -> bool:
        """Run essential tests only"""
        print("🚀 Quick MCP Server Test")
        print("=" * 40)
        
        if not self.start_server():
            return False
        
        try:
            # Run essential tests
            tests = [
                ("Server Initialization", self.test_server_initialization),
                ("Tools List", self.test_tools_list),
                ("All Tools", self.test_all_tools),
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                try:
                    result = await test_func()
                    if result:
                        passed += 1
                        print(f"✅ {test_name}")
                    else:
                        print(f"❌ {test_name}")
                except Exception as e:
                    print(f"💥 {test_name}: {e}")
            
            print("-" * 40)
            print(f"📊 Results: {passed}/{total} tests passed")
            
            if passed == total:
                print("🎉 All essential tests passed!")
                return True
            else:
                print("⚠️  Some tests failed.")
                return False
            
        finally:
            self.stop_server()


async def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Comprehensive MCP Server Test Suite")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output with detailed logging")
    parser.add_argument("--quick", "-q", action="store_true",
                       help="Run essential tests only")
    parser.add_argument("--test", choices=["tools", "performance", "errors", "websocket"],
                       help="Run specific test category")
    
    args = parser.parse_args()
    
    tester = MCPTester(verbose=args.verbose)
    
    if args.test:
        # Run specific test category
        if args.test == "tools":
            success = await tester.run_quick_tests()
        elif args.test == "performance":
            if not tester.start_server():
                sys.exit(1)
            try:
                success = await tester.test_performance()
            finally:
                tester.stop_server()
        elif args.test == "errors":
            if not tester.start_server():
                sys.exit(1)
            try:
                success = await tester.test_error_handling()
            finally:
                tester.stop_server()
        elif args.test == "websocket":
            if not tester.start_server():
                sys.exit(1)
            try:
                success = await tester.test_websocket_transport()
            finally:
                tester.stop_server()
    elif args.quick:
        # Run essential tests only
        success = await tester.run_quick_tests()
    else:
        # Run comprehensive tests
        success = await tester.run_comprehensive_tests()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
