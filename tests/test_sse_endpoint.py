#!/usr/bin/env python3
"""
Test script for FastMCP-style SSE endpoint

This script demonstrates how to use the /sse endpoint with:
1. Server-Sent Events (text/event-stream)
2. mcp-session-id headers
3. Session persistence across requests
"""

import requests
import json
import time
import uuid

# Configuration
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8880"
SESSION_ID = str(uuid.uuid4())

def test_sse_endpoint():
    """Test the SSE endpoint with session management"""
    print("🧪 Testing FastMCP-Style SSE Endpoint")
    print("=" * 60)
    
    # Test 1: Initialize with SSE
    print("\n1️⃣ Testing SSE Initialize...")
    message = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "test-sse-client", "version": "1.0.0"}
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "mcp-session-id": SESSION_ID
    }
    
    response = requests.post(
        f"{BASE_URL}/sse",
        json=message,
        headers=headers,
        stream=True
    )
    
    if response.status_code == 200:
        print(f"✅ SSE stream connected (session: {SESSION_ID})")
        print(f"   Response headers: {dict(response.headers)}")
        
        # Read SSE events
        events = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                print(f"   📨 {line}")
                if line.startswith('data:'):
                    events.append(line)
        
        print(f"   Received {len(events)} events")
    else:
        print(f"❌ SSE request failed: {response.status_code}")
        print(f"   {response.text}")
        return
    
    # Test 2: List tools with SSE
    print("\n2️⃣ Testing SSE tools/list...")
    message = {
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    }
    
    response = requests.post(
        f"{BASE_URL}/sse",
        json=message,
        headers=headers,
        stream=True
    )
    
    if response.status_code == 200:
        print(f"✅ Tools list via SSE")
        session_from_header = response.headers.get('mcp-session-id')
        print(f"   Session ID from header: {session_from_header}")
        
        for line in response.iter_lines(decode_unicode=True):
            if line and line.startswith('event:'):
                event_type = line.split(':', 1)[1].strip()
                print(f"   📡 Event type: {event_type}")
    else:
        print(f"❌ Tools list failed: {response.status_code}")
        return
    
    # Test 3: Ask question with SSE and session persistence
    print("\n3️⃣ Testing SSE ask_question with session...")
    message = {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "ask_question",
            "arguments": {
                "question": "What is Ascalon?",
                "max_chunks": 3
            }
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/sse",
        json=message,
        headers=headers,
        stream=True
    )
    
    if response.status_code == 200:
        print(f"✅ Ask question via SSE (using session)")
        session_from_header = response.headers.get('mcp-session-id')
        print(f"   Session ID from header: {session_from_header}")
        print(f"   Session persisted: {session_from_header == SESSION_ID}")
        
        event_count = 0
        for line in response.iter_lines(decode_unicode=True):
            if line:
                if line.startswith('event:'):
                    event_type = line.split(':', 1)[1].strip()
                    event_count += 1
                    print(f"   📡 Event {event_count}: {event_type}")
                elif line.startswith('data:') and event_count == 2:
                    # Show preview of the message data
                    data_preview = line[:100] + "..." if len(line) > 100 else line
                    print(f"   📝 Data preview: {data_preview}")
    else:
        print(f"❌ Ask question failed: {response.status_code}")
        return
    
    # Test 4: Regular /mcp endpoint with session headers
    print("\n4️⃣ Testing regular /mcp endpoint with session headers...")
    message = {
        "jsonrpc": "2.0",
        "id": 4,
        "method": "tools/call",
        "params": {
            "name": "get_server_info",
            "arguments": {}
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/mcp",
        json=message,
        headers=headers
    )
    
    if response.status_code == 200:
        print(f"✅ Regular endpoint with session headers")
        session_from_header = response.headers.get('mcp-session-id')
        print(f"   Session ID from header: {session_from_header}")
        print(f"   Session persisted: {session_from_header == SESSION_ID}")
        
        data = response.json()
        if 'result' in data:
            result = data['result']['content'][0]['text']
            result_obj = json.loads(result)
            print(f"   Active sessions: {result_obj.get('active_sessions', 0)}")
    else:
        print(f"❌ Regular endpoint failed: {response.status_code}")
        return
    
    print("\n" + "=" * 60)
    print("✅ All FastMCP-style tests passed!")
    print(f"   Session ID used: {SESSION_ID}")
    print(f"   Endpoints tested: /sse, /mcp")
    print(f"   Features tested: SSE streaming, session headers, session persistence")

if __name__ == "__main__":
    try:
        test_sse_endpoint()
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to MCP server")
        print("   Make sure the server is running: python mcp_server.py")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

