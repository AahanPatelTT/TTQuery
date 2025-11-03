#!/usr/bin/env python3
"""Simple test for FastMCP endpoints"""
import subprocess
import time
import requests
import sys
import os

# Start server
print("🚀 Starting MCP server...")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
proc = subprocess.Popen(
    [sys.executable, os.path.join(project_root, "mcp_server.py"), "--http-port", "8880"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=os.environ.copy(),
    cwd=project_root
)

# Wait for server
for i in range(10):
    time.sleep(1)
    try:
        r = requests.get("http://localhost:8880/health", timeout=1)
        if r.status_code == 200:
            print("✅ Server started")
            break
    except:
        pass
    print(f"⏳ Waiting... ({i+1}/10)")

try:
    # Test 1: /mcp endpoint with session header
    print("\n1️⃣ Testing /mcp with session header...")
    response = requests.post(
        "http://localhost:8880/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "get_server_info",
                "arguments": {}
            }
        },
        headers={"mcp-session-id": "test-session-123"}
    )
    
    print(f"   Status: {response.status_code}")
    print(f"   Session header sent: test-session-123")
    print(f"   Session header received: {response.headers.get('mcp-session-id', 'NOT FOUND')}")
    
    if response.status_code == 200:
        print("   ✅ /mcp endpoint working")
        if response.headers.get('mcp-session-id'):
            print("   ✅ Session header returned")
        else:
            print("   ⚠️  Session header NOT returned in response")
    else:
        print(f"   ❌ /mcp endpoint failed: {response.text}")
    
    # Test 2: /sse endpoint
    print("\n2️⃣ Testing /sse endpoint...")
    response = requests.post(
        "http://localhost:8880/sse",
        json={
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "get_server_info",
                "arguments": {}
            }
        },
        headers={"mcp-session-id": "sse-session-456"},
        stream=True
    )
    
    print(f"   Status: {response.status_code}")
    print(f"   Content-Type: {response.headers.get('Content-Type', 'NOT FOUND')}")
    print(f"   Session header sent: sse-session-456")
    print(f"   Session header received: {response.headers.get('mcp-session-id', 'NOT FOUND')}")
    
    if response.status_code == 200:
        print("   ✅ /sse endpoint working")
        
        # Read SSE stream
        events = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"   📨 {line[:80]}")
                events.append(line)
                if len(events) >= 5:  # Limit output
                    break
        
        content_type = response.headers.get('Content-Type', '')
        if 'text/event-stream' in content_type:
            print("   ✅ Correct Content-Type (text/event-stream)")
        else:
            print(f"   ⚠️  Wrong Content-Type: {content_type}")
    else:
        print(f"   ❌ /sse endpoint failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
    
    # Test 3: ask_question with session header (test session injection)
    print("\n3️⃣ Testing ask_question with session header...")
    response = requests.post(
        "http://localhost:8880/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "ask_question",
                "arguments": {
                    "question": "What is Ascalon?",
                    "max_chunks": 2
                }
            }
        },
        headers={"mcp-session-id": "question-session-789"}
    )
    
    print(f"   Status: {response.status_code}")
    print(f"   Session header sent: question-session-789")
    print(f"   Session header received: {response.headers.get('mcp-session-id', 'NOT FOUND')}")
    
    if response.status_code == 200:
        result = response.json()
        if "error" not in result:
            print("   ✅ ask_question working")
            # Parse the content to check if session was used
            content_text = result.get("result", {}).get("content", [{}])[0].get("text", "{}")
            import json
            try:
                parsed = json.loads(content_text)
                if parsed.get("session_id") == "question-session-789":
                    print("   ✅ Session ID correctly used in ask_question")
                else:
                    print(f"   ⚠️  Session ID mismatch: {parsed.get('session_id')}")
            except:
                pass
        else:
            print(f"   ❌ Error: {result.get('error', {}).get('message', 'Unknown')}")
    else:
        print(f"   ❌ Request failed: {response.status_code}")
    
finally:
    print("\n🛑 Stopping server...")
    proc.terminate()
    proc.wait()
    print("✅ Server stopped")

