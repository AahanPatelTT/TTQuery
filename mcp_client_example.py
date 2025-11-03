#!/usr/bin/env python3
"""
Synapse MCP Client Example

This script demonstrates how to interact with the Synapse MCP server
using HTTP transport. It provides examples of all available MCP tools 
and shows how to maintain conversation sessions.

Usage:
    python mcp_client_example.py --host localhost --port 8880
    python mcp_client_example.py --demo
"""

import argparse
import asyncio
import json
import sys
import time
from typing import Dict, Any, Optional

# HTTP client
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

class MCPClient:
    """MCP Client for interacting with Synapse MCP server"""
    
    def __init__(self, host: str = "localhost", port: int = 8880):
        self.host = host
        self.port = port
        self.message_id = 0
        
    def get_next_id(self) -> int:
        """Get next message ID"""
        self.message_id += 1
        return self.message_id
    
    def create_message(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create MCP JSON-RPC message"""
        return {
            "jsonrpc": "2.0",
            "id": self.get_next_id(),
            "method": method,
            "params": params or {}
        }
    
    async def send_http_request(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send HTTP request to MCP server"""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for HTTP transport")
        
        url = f"http://{self.host}:{self.port}/mcp"
        
        try:
            response = requests.post(url, json=message, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": {"code": -32603, "message": f"HTTP request failed: {e}"}}
    
    async def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Send message via HTTP"""
        return await self.send_http_request(message)
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize MCP session"""
        message = self.create_message("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}, "logging": {}},
            "clientInfo": {"name": "synapse-mcp-client", "version": "1.0.0"}
        })
        
        response = await self.send_message(message)
        
        # Send initialized notification
        notify_message = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self.send_message(notify_message)
        
        return response
    
    async def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        message = self.create_message("tools/list")
        return await self.send_message(message)
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP tool"""
        message = self.create_message("tools/call", {
            "name": tool_name,
            "arguments": arguments
        })
        return await self.send_message(message)
    
    async def close(self):
        """Close connection (no-op for HTTP)"""
        pass


async def demo_basic_functionality(client: MCPClient):
    """Demonstrate basic MCP functionality"""
    print("🚀 Starting MCP Client Demo")
    print("="*50)
    
    # Initialize
    print("\n1. Initializing MCP session...")
    init_response = await client.initialize()
    if "error" in init_response:
        print(f"❌ Initialization failed: {init_response['error']['message']}")
        return
    print("✅ Initialized successfully")
    
    # List tools
    print("\n2. Listing available tools...")
    tools_response = await client.list_tools()
    if "error" in tools_response:
        print(f"❌ Failed to list tools: {tools_response['error']['message']}")
        return
    
    tools = tools_response["result"]["tools"]
    print(f"✅ Found {len(tools)} tools:")
    for tool in tools:
        print(f"   • {tool['name']}: {tool['description']}")
    
    # Get server info
    print("\n3. Getting server information...")
    server_info = await client.call_tool("get_server_info", {})
    if "error" in server_info:
        print(f"❌ Failed to get server info: {server_info['error']['message']}")
    else:
        info = json.loads(server_info["result"]["content"][0]["text"])
        print(f"✅ Server: {info['server_name']} v{info['version']}")
        print(f"   Current KB: {info['current_kb']}")
        print(f"   Available KBs: {info['available_kbs']}")
        print(f"   Active sessions: {info['active_sessions']}")
    
    # List knowledge bases
    print("\n4. Listing knowledge bases...")
    kb_response = await client.call_tool("list_knowledge_bases", {})
    if "error" in kb_response:
        print(f"❌ Failed to list KBs: {kb_response['error']['message']}")
    else:
        kbs = json.loads(kb_response["result"]["content"][0]["text"])
        if kbs:
            print(f"✅ Available knowledge bases:")
            for kb in kbs:
                current = " (current)" if kb['is_current'] else ""
                print(f"   • {kb['display_name']}: {kb['chunk_count']:,} chunks{current}")
        else:
            print("⚠️  No knowledge bases found")
    
    return tools


async def demo_conversation(client: MCPClient):
    """Demonstrate conversation functionality"""
    print("\n" + "="*50)
    print("💬 CONVERSATION DEMO")
    print("="*50)
    
    # Create a session
    print("\n1. Creating new conversation session...")
    session_response = await client.call_tool("create_session", {"session_id": "demo_session"})
    if "error" in session_response:
        print(f"❌ Failed to create session: {session_response['error']['message']}")
        return
    
    session_info = json.loads(session_response["result"]["content"][0]["text"])
    session_id = session_info["session_id"]
    print(f"✅ Created session: {session_id}")
    
    # Ask some questions
    questions = [
        "What is Synapse and what does it do?",
        "How does the RAG pipeline work?",
        "What are the key features of the system?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n2.{i} Asking: {question}")
        
        answer_response = await client.call_tool("ask_question", {
            "question": question,
            "session_id": session_id,
            "verbose": False
        })
        
        if "error" in answer_response:
            print(f"❌ Error: {answer_response['error']['message']}")
            continue
        
        answer_data = json.loads(answer_response["result"]["content"][0]["text"])
        answer = answer_data["answer"]
        sources = answer_data.get("sources", [])
        
        print(f"🤖 Answer: {answer[:200]}...")
        if sources:
            print(f"📚 Sources: {len(sources)} documents")
    
    # Get session history
    print(f"\n3. Getting session history...")
    history_response = await client.call_tool("get_session_history", {
        "session_id": session_id,
        "limit": 5
    })
    
    if "error" in history_response:
        print(f"❌ Failed to get history: {history_response['error']['message']}")
    else:
        history_data = json.loads(history_response["result"]["content"][0]["text"])
        print(f"✅ Session has {history_data['total_exchanges']} exchanges")
        
        for i, exchange in enumerate(history_data["history"], 1):
            print(f"   Q{i}: {exchange['question'][:50]}...")
            print(f"   A{i}: {exchange['answer'][:50]}...")


async def demo_knowledge_base_switching(client: MCPClient):
    """Demonstrate knowledge base switching"""
    print("\n" + "="*50) 
    print("🔄 KNOWLEDGE BASE SWITCHING DEMO")
    print("="*50)
    
    # List available KBs
    kb_response = await client.call_tool("list_knowledge_bases", {})
    if "error" in kb_response:
        print(f"❌ Failed to list KBs: {kb_response['error']['message']}")
        return
    
    kbs = json.loads(kb_response["result"]["content"][0]["text"])
    if len(kbs) < 2:
        print("⚠️  Need at least 2 knowledge bases to demo switching")
        return
    
    # Switch to a different KB
    target_kb = None
    for kb in kbs:
        if not kb['is_current']:
            target_kb = kb
            break
    
    if target_kb:
        print(f"\n1. Switching to KB: {target_kb['display_name']}")
        switch_response = await client.call_tool("switch_knowledge_base", {
            "knowledge_base": target_kb['name']
        })
        
        if "error" in switch_response:
            print(f"❌ Switch failed: {switch_response['error']['message']}")
        else:
            switch_data = json.loads(switch_response["result"]["content"][0]["text"])
            print(f"✅ Switched to: {switch_data['display_name']}")
            print(f"   Chunks: {switch_data['chunk_count']:,}")


async def demo_processing_status(client: MCPClient):
    """Demonstrate processing status monitoring"""
    print("\n" + "="*50)
    print("📊 PROCESSING STATUS DEMO") 
    print("="*50)
    
    print("\n1. Getting processing status...")
    status_response = await client.call_tool("get_processing_status", {})
    
    if "error" in status_response:
        print(f"❌ Status check failed: {status_response['error']['message']}")
        print("   (This is normal if the fast initialization system isn't set up)")
        return
    
    status_data = json.loads(status_response["result"]["content"][0]["text"])
    print(f"✅ Processing status for {status_data['total_folders']} folders:")
    
    for folder_status in status_data["folder_status"]:
        folder = folder_status["folder"]
        docs = folder_status["completed_docs"]
        chunks = folder_status["total_chunks"]
        embedded = folder_status["embedded_chunks"]
        completion = folder_status["completion_rate"]
        
        print(f"   📁 {folder}: {docs} docs, {chunks} chunks, {embedded} embedded ({completion:.1f}%)")


async def interactive_mode(client: MCPClient):
    """Interactive mode for testing MCP tools"""
    print("\n" + "="*50)
    print("🔧 INTERACTIVE MODE")
    print("="*50)
    print("Type 'help' for available commands, 'quit' to exit")
    
    session_id = "interactive_session"
    
    # Create session
    await client.call_tool("create_session", {"session_id": session_id})
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit']:
                break
                
            if user_input.lower() == 'help':
                print("\nAvailable commands:")
                print("  ask <question>     - Ask a question")
                print("  list-kbs          - List knowledge bases")
                print("  switch-kb <name>  - Switch knowledge base")
                print("  history           - Show conversation history")
                print("  status            - Show processing status")
                print("  server-info       - Show server information")
                print("  quit              - Exit interactive mode")
                continue
            
            # Parse command
            parts = user_input.split(None, 1)
            command = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if command == "ask":
                if not arg:
                    print("❌ Please provide a question")
                    continue
                
                response = await client.call_tool("ask_question", {
                    "question": arg,
                    "session_id": session_id,
                    "verbose": False
                })
                
                if "error" in response:
                    print(f"❌ Error: {response['error']['message']}")
                else:
                    data = json.loads(response["result"]["content"][0]["text"])
                    print(f"\n🤖 {data['answer']}")
                    if data.get('sources'):
                        print(f"\n📚 Sources: {len(data['sources'])} documents")
            
            elif command == "list-kbs":
                response = await client.call_tool("list_knowledge_bases", {})
                if "error" in response:
                    print(f"❌ Error: {response['error']['message']}")
                else:
                    kbs = json.loads(response["result"]["content"][0]["text"])
                    print("\nKnowledge Bases:")
                    for kb in kbs:
                        current = " (current)" if kb['is_current'] else ""
                        print(f"  • {kb['name']}: {kb['chunk_count']:,} chunks{current}")
            
            elif command == "switch-kb":
                if not arg:
                    print("❌ Please provide a KB name")
                    continue
                
                response = await client.call_tool("switch_knowledge_base", {
                    "knowledge_base": arg
                })
                
                if "error" in response:
                    print(f"❌ Error: {response['error']['message']}")
                else:
                    data = json.loads(response["result"]["content"][0]["text"])
                    print(f"✅ Switched to: {data['display_name']}")
            
            elif command == "history":
                response = await client.call_tool("get_session_history", {
                    "session_id": session_id,
                    "limit": 5
                })
                
                if "error" in response:
                    print(f"❌ Error: {response['error']['message']}")
                else:
                    data = json.loads(response["result"]["content"][0]["text"])
                    print(f"\nConversation History ({data['total_exchanges']} total):")
                    for i, exchange in enumerate(data["history"], 1):
                        print(f"  Q{i}: {exchange['question']}")
                        print(f"  A{i}: {exchange['answer'][:100]}...")
                        print()
            
            elif command == "status":
                response = await client.call_tool("get_processing_status", {})
                if "error" in response:
                    print(f"❌ Error: {response['error']['message']}")
                else:
                    data = json.loads(response["result"]["content"][0]["text"])
                    print(f"\nProcessing Status ({data['total_folders']} folders):")
                    for folder in data["folder_status"]:
                        print(f"  📁 {folder['folder']}: {folder['completion_rate']:.1f}% complete")
            
            elif command == "server-info":
                response = await client.call_tool("get_server_info", {})
                if "error" in response:
                    print(f"❌ Error: {response['error']['message']}")
                else:
                    data = json.loads(response["result"]["content"][0]["text"])
                    print(f"\nServer Info:")
                    print(f"  Name: {data['server_name']} v{data['version']}")
                    print(f"  Current KB: {data['current_kb']}")
                    print(f"  Available KBs: {', '.join(data['available_kbs'])}")
                    print(f"  Active Sessions: {data['active_sessions']}")
                    print(f"  Uptime: {data['uptime']:.1f}s")
            
            else:
                print(f"❌ Unknown command: {command}")
                print("Type 'help' for available commands")
        
        except (EOFError, KeyboardInterrupt):
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Goodbye!")


async def main():
    parser = argparse.ArgumentParser(
        description="Synapse MCP Client Example - Test and demonstrate MCP server functionality",
        epilog="""
Examples:
  %(prog)s --demo                                    # Run complete demo
  %(prog)s --interactive                             # Interactive mode for testing
  %(prog)s --port 8080 --demo                        # Demo with custom port
  
The MCP server should be running before using this client:
  ./launch.sh --mcp                                  # Start MCP server
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--host", default="localhost", 
                      help="MCP server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8880,
                      help="Server port (default: 8880)")
    parser.add_argument("--demo", action="store_true", 
                      help="Run comprehensive demo of all MCP functionality")
    parser.add_argument("--interactive", action="store_true", 
                      help="Start interactive mode for manual testing")
    
    args = parser.parse_args()
    
    # Create client
    client = MCPClient(args.host, args.port)
    
    try:
        print(f"🔗 Connecting to MCP server at {args.host}:{args.port}")
        
        if args.demo:
            # Run full demo
            await demo_basic_functionality(client)
            await demo_conversation(client)
            await demo_knowledge_base_switching(client)
            await demo_processing_status(client)
        elif args.interactive:
            # Run basic setup then interactive mode
            await demo_basic_functionality(client)
            await interactive_mode(client)
        else:
            # Just basic functionality
            await demo_basic_functionality(client)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
