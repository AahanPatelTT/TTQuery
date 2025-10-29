#!/usr/bin/env python3
"""
Synapse MCP (Model Context Protocol) Server

This module implements an MCP server that exposes all Synapse CLI functionality
via HTTP and WebSocket transports. It provides tools for knowledge base queries,
session management, document processing, and more.

Features:
- HTTP transport with JSON-RPC 2.0 protocol
- WebSocket transport for real-time communication
- All CLI commands exposed as MCP tools
- Comprehensive logging and error handling
- Session persistence and management
- Knowledge base switching and management
- Document processing status monitoring

Usage:
    python mcp_server.py --transport http --port 3000
    python mcp_server.py --transport websocket --port 3001
    python mcp_server.py --transport both --http-port 3000 --ws-port 3001
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import traceback

# HTTP server imports
try:
    from flask import Flask, request, jsonify, Response
    from flask_cors import CORS
    import threading
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# WebSocket server imports
try:
    import websockets
    import websockets.server
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import Synapse components
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chat import ChatSession, load_default_config, save_default_config
from pipeline.query import (
    load_corpus, build_indices, load_query_encoder, encode_query,
    dense_search, sparse_search, rrf, per_doc_cap, rerank, mmr_select,
    format_citations, build_prompt, call_gemini_via_litellm,
    list_available_knowledge_bases, answer
)
from pipeline.database import SynapseDB


# MCP Protocol Types and Classes
@dataclass
class MCPMessage:
    """Base MCP message structure"""
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None


@dataclass
class MCPError:
    """MCP error structure"""
    code: int
    message: str
    data: Optional[Any] = None


@dataclass  
class MCPTool:
    """MCP tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]


class MCPServer:
    """Main MCP Server implementation"""
    
    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = artifacts_dir
        self.sessions: Dict[str, ChatSession] = {}
        self.current_embeddings_path = None
        self.corpus = None
        self.indices = None
        self.query_encoder = None
        self.query_encoder_name = None
        self.knowledge_bases = {}
        self.current_kb = None
        self.verbose_mode = False
        
        # Setup logging
        self.setup_logging()
        
        # Initialize default knowledge base
        self.initialize_knowledge_base()
        
        # Define MCP tools
        self.tools = self.define_tools()
        
    def setup_logging(self):
        """Setup comprehensive logging for MCP operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [MCP-%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('mcp_server.log')
            ]
        )
        self.logger = logging.getLogger('MCPServer')
        self.logger.info("MCP Server initialized")
        
    def initialize_knowledge_base(self):
        """Initialize the default knowledge base and load available KBs"""
        try:
            # Load available knowledge bases
            self.knowledge_bases = {
                kb['name']: kb for kb in list_available_knowledge_bases(self.artifacts_dir)
            }
            
            if self.knowledge_bases:
                # Set the first KB as current and load it
                self.current_kb = next(iter(self.knowledge_bases.keys()))
                kb_info = self.knowledge_bases[self.current_kb]
                
                # Load the embeddings for the current KB
                embeddings_path = kb_info['embeddings_path']
                if os.path.exists(embeddings_path):
                    self.current_embeddings_path = embeddings_path
                    self.corpus = load_corpus(embeddings_path)
                    idx_sum, idx_full, E_sum, E_full, bm25 = build_indices(self.corpus)
                    self.indices = {
                        'faiss_sum': idx_sum,
                        'faiss_full': idx_full,
                        'E_sum': E_sum,
                        'E_full': E_full,
                        'bm25': bm25
                    }
                    self.query_encoder_name, self.query_encoder = load_query_encoder()
                    self.logger.info(f"Loaded knowledge base: {self.current_kb} ({embeddings_path})")
                else:
                    self.logger.error(f"Embeddings file not found: {embeddings_path}")
                
                self.logger.info(f"Available KBs: {list(self.knowledge_bases.keys())}")
            else:
                self.logger.warning("No knowledge bases found")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge base: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
    
    def define_tools(self) -> List[MCPTool]:
        """Define all available MCP tools based on CLI functionality"""
        return [
            # Core Query Tools
            MCPTool(
                name="ask_question",
                description="Retrieve relevant context from knowledge base(s) for external response generation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to search for"},
                        "session_id": {"type": "string", "description": "Session ID for conversation context"},
                        "verbose": {"type": "boolean", "description": "Enable detailed retrieval information"},
                        "timeout": {"type": "number", "description": "Query timeout in seconds", "default": 60},
                        "knowledge_bases": {
                            "type": "array", 
                            "description": "List of knowledge base names to search (optional - uses current KB if not specified)",
                            "items": {"type": "string"}
                        },
                        "search_mode": {
                            "type": "string", 
                            "description": "Multi-KB search mode: 'concatenated' (unified search) or 'separate' (individual searches)",
                            "enum": ["concatenated", "separate"],
                            "default": "concatenated"
                        },
                        "context_format": {
                            "type": "string",
                            "description": "Format for returned context",
                            "enum": ["structured", "raw", "markdown"],
                            "default": "structured"
                        },
                        "max_chunks": {
                            "type": "number",
                            "description": "Maximum number of chunks to retrieve",
                            "default": 10,
                            "minimum": 1,
                            "maximum": 50
                        }
                    },
                    "required": ["question"]
                }
            ),
            
            # Knowledge Base Management
            MCPTool(
                name="list_knowledge_bases",
                description="List all available knowledge bases with statistics",
                inputSchema={"type": "object", "properties": {}}
            ),
            MCPTool(
                name="switch_knowledge_base",
                description="Switch to a different knowledge base",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "knowledge_base": {"type": "string", "description": "Name of the knowledge base to switch to"}
                    },
                    "required": ["knowledge_base"]
                }
            ),
            MCPTool(
                name="get_kb_stats",
                description="Get detailed statistics about current or specified knowledge base",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "knowledge_base": {"type": "string", "description": "KB name (optional, uses current if not specified)"}
                    }
                }
            ),
            
            # Session Management
            MCPTool(
                name="create_session",
                description="Create a new conversation session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Optional custom session ID"}
                    }
                }
            ),
            MCPTool(
                name="load_session",
                description="Load an existing conversation session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_file": {"type": "string", "description": "Path to session file"}
                    },
                    "required": ["session_file"]
                }
            ),
            MCPTool(
                name="list_sessions",
                description="List all available conversation sessions",
                inputSchema={"type": "object", "properties": {}}
            ),
            MCPTool(
                name="get_session_history",
                description="Get conversation history for a session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "limit": {"type": "number", "description": "Max number of exchanges to return", "default": 10}
                    },
                    "required": ["session_id"]
                }
            ),
            MCPTool(
                name="clear_session_history",
                description="Clear conversation history for a session",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"}
                    },
                    "required": ["session_id"]
                }
            ),
            MCPTool(
                name="export_session",
                description="Export session conversation to JSON file",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Session ID"},
                        "output_file": {"type": "string", "description": "Output file path"}
                    },
                    "required": ["session_id", "output_file"]
                }
            ),
            
            # Document Processing
            MCPTool(
                name="get_processing_status",
                description="Get document processing status from fast initialization system",
                inputSchema={"type": "object", "properties": {}}
            ),
            MCPTool(
                name="initialize_knowledge_base",
                description="Run knowledge base initialization (fast incremental)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "folder": {"type": "string", "description": "Specific folder to process (optional)"},
                        "cleanup": {"type": "boolean", "description": "Run cleanup before processing", "default": False}
                    }
                }
            ),
            
            # Configuration
            MCPTool(
                name="set_verbose_mode",
                description="Enable or disable verbose retrieval mode globally",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "verbose": {"type": "boolean", "description": "Enable verbose mode"}
                    },
                    "required": ["verbose"]
                }
            ),
            MCPTool(
                name="get_server_info", 
                description="Get MCP server information and status",
                inputSchema={"type": "object", "properties": {}}
            )
        ]
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming MCP message"""
        try:
            # Validate message structure
            if not isinstance(message, dict) or message.get('jsonrpc') != '2.0':
                return self.create_error_response(None, -32600, "Invalid Request")
            
            method = message.get('method')
            params = message.get('params', {})
            msg_id = message.get('id')
            
            self.logger.info(f"Handling method: {method} with params: {params}")
            
            # Handle MCP protocol methods
            if method == 'initialize':
                return self.handle_initialize(msg_id, params)
            elif method == 'tools/list':
                return self.handle_list_tools(msg_id)
            elif method == 'tools/call':
                return await self.handle_tool_call(msg_id, params)
            elif method == 'notifications/initialized':
                return None  # No response needed for notifications
            else:
                return self.create_error_response(msg_id, -32601, f"Method not found: {method}")
                
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self.logger.error(traceback.format_exc())
            return self.create_error_response(message.get('id'), -32603, f"Internal error: {str(e)}")
    
    def create_error_response(self, msg_id: Optional[Union[str, int]], code: int, message: str, data: Any = None) -> Dict[str, Any]:
        """Create MCP error response"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": code,
                "message": message,
                "data": data
            }
        }
    
    def create_success_response(self, msg_id: Union[str, int], result: Any) -> Dict[str, Any]:
        """Create MCP success response"""
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": result
        }
    
    def handle_initialize(self, msg_id: Union[str, int], params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP initialize request"""
        return self.create_success_response(msg_id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "logging": {}
            },
            "serverInfo": {
                "name": "synapse-rag-server",
                "version": "1.0.0"
            }
        })
    
    def handle_list_tools(self, msg_id: Union[str, int]) -> Dict[str, Any]:
        """Handle tools/list request"""
        return self.create_success_response(msg_id, {
            "tools": [asdict(tool) for tool in self.tools]
        })
    
    async def handle_tool_call(self, msg_id: Union[str, int], params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle tools/call request"""
        tool_name = params.get('name')
        arguments = params.get('arguments', {})
        
        if not tool_name:
            return self.create_error_response(msg_id, -32602, "Missing tool name")
        
        # Route to appropriate tool handler
        tool_handlers = {
            'ask_question': self.handle_ask_question,
            'list_knowledge_bases': self.handle_list_knowledge_bases,
            'switch_knowledge_base': self.handle_switch_knowledge_base,
            'get_kb_stats': self.handle_get_kb_stats,
            'create_session': self.handle_create_session,
            'load_session': self.handle_load_session,
            'list_sessions': self.handle_list_sessions,
            'get_session_history': self.handle_get_session_history,
            'clear_session_history': self.handle_clear_session_history,
            'export_session': self.handle_export_session,
            'get_processing_status': self.handle_get_processing_status,
            'initialize_knowledge_base': self.handle_initialize_knowledge_base,
            'set_verbose_mode': self.handle_set_verbose_mode,
            'get_server_info': self.handle_get_server_info
        }
        
        if tool_name not in tool_handlers:
            return self.create_error_response(msg_id, -32601, f"Unknown tool: {tool_name}")
        
        try:
            result = await tool_handlers[tool_name](arguments)
            return self.create_success_response(msg_id, {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]})
        except Exception as e:
            self.logger.error(f"Tool execution error for {tool_name}: {e}")
            self.logger.error(traceback.format_exc())
            return self.create_error_response(msg_id, -32603, f"Tool execution failed: {str(e)}")
    
    # Tool Implementation Methods
    async def handle_ask_question(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle ask_question tool - returns retrieval context instead of generated response"""
        question = args.get('question')
        session_id = args.get('session_id', 'default')
        verbose = args.get('verbose', self.verbose_mode)
        timeout = args.get('timeout', 60)
        knowledge_bases = args.get('knowledge_bases', [])
        search_mode = args.get('search_mode', 'concatenated')
        context_format = args.get('context_format', 'structured')  # structured, raw, markdown
        max_chunks = args.get('max_chunks', 10)
        
        if not question:
            raise ValueError("Question is required")
        
        # Get or create session
        if session_id not in self.sessions:
            self.sessions[session_id] = ChatSession(auto_continue=False)
        
        session = self.sessions[session_id]
        
        # Get conversation context for retrieval
        conv_context = session.get_context(last_n=3) if hasattr(session, 'get_context') else ""
        
        # Perform retrieval based on KB selection
        if knowledge_bases and len(knowledge_bases) > 1:
            # Multi-KB search
            retrieval_result = await self._perform_multi_kb_retrieval(
                question, knowledge_bases, conv_context, verbose, timeout, max_chunks
            )
        elif knowledge_bases and len(knowledge_bases) == 1:
            # Single specified KB
            kb_name = knowledge_bases[0]
            if kb_name not in self.knowledge_bases:
                raise ValueError(f"Knowledge base '{kb_name}' not found. Available: {list(self.knowledge_bases.keys())}")
            
            kb_info = self.knowledge_bases[kb_name]
            embeddings_path = kb_info['embeddings_path']
            retrieval_result = await self._perform_single_kb_retrieval(
                question, embeddings_path, kb_name, conv_context, verbose, timeout, max_chunks
            )
        else:
            # Use current KB
            if not self.current_embeddings_path:
                raise ValueError("No knowledge base loaded")
            
            retrieval_result = await self._perform_single_kb_retrieval(
                question, self.current_embeddings_path, self.current_kb, conv_context, verbose, timeout, max_chunks
            )
        
        # Format context based on requested format
        formatted_context = self._format_retrieval_context(retrieval_result, context_format)
        
        # Add to session history (store retrieval context instead of generated answer)
        session.add_exchange(question, f"[RETRIEVAL] {len(retrieval_result['chunks'])} chunks found", 
                           f"Retrieved from: {', '.join(retrieval_result['knowledge_bases'])}")
        
        return {
            "retrieval_context": {
                "query": question,
                "retrieved_chunks": retrieval_result['chunks'],
                "metadata": retrieval_result['metadata']
            },
            "formatted_context": formatted_context,
            "search_metadata": retrieval_result['search_metadata'],
            "session_id": session_id,
            "verbose_mode": verbose,
            "knowledge_bases": retrieval_result['knowledge_bases'],
            "search_mode": search_mode
        }
    
    async def _perform_single_kb_retrieval(self, question: str, embeddings_path: str, kb_name: str, 
                                         conv_context: str, verbose: bool, timeout: int, max_chunks: int) -> Dict[str, Any]:
        """Perform retrieval on a single knowledge base"""
        import time
        start_time = time.time()
        
        # Load corpus and build indices if not already loaded
        if not self.corpus or self.current_embeddings_path != embeddings_path:
            self.corpus = load_corpus(embeddings_path)
            idx_sum, idx_full, E_sum, E_full, bm25 = build_indices(self.corpus)
            self.indices = {
                'faiss_sum': idx_sum,
                'faiss_full': idx_full,
                'E_sum': E_sum,
                'E_full': E_full,
                'bm25': bm25
            }
            self.query_encoder_name, self.query_encoder = load_query_encoder()
        
        # Encode query
        query_embedding = encode_query(question, self.query_encoder)
        
        # Perform dense and sparse search
        dense_results = dense_search(query_embedding, self.indices['faiss_sum'], max_chunks*2)
        sparse_results = sparse_search(question, self.indices['bm25'], max_chunks*2)
        
        # Combine results with RRF
        combined_results = rrf([dense_results, sparse_results], weights=[1.0, 0.8], rrf_k=max_chunks*2)
        
        # Apply per-document capping
        capped_results = per_doc_cap(combined_results, self.corpus, 3)
        
        # Rerank results
        reranked_results = rerank(question, capped_results, self.corpus)
        
        # Take top results (skip MMR for now to avoid complexity)
        final_indices = reranked_results[:max_chunks]
        
        # Extract chunks with metadata
        chunks = []
        for i, idx in enumerate(final_indices):
            if idx < len(self.corpus):
                item = self.corpus[idx]
                chunk_data = {
                    "chunk_id": item.get('id', f'chunk_{idx}'),
                    "content": item.get('full_text', ''),
                    "summary": item.get('summary_text', ''),
                    "source": item.get('source_path', ''),
                    "source_name": os.path.basename(item.get('source_path', '')),
                    "page": item.get('metadata', {}).get('page_number', None),
                    "section": item.get('metadata', {}).get('section', None),
                    "relevance_score": 0.8 - (i * 0.1),  # Simple scoring based on rank
                    "chunk_index": i,
                    "metadata": item.get('metadata', {})
                }
                chunks.append(chunk_data)
        
        search_time = (time.time() - start_time) * 1000
        
        return {
            "chunks": chunks,
            "knowledge_bases": [kb_name],
            "metadata": {
                "total_chunks_found": len(final_indices),
                "search_time_ms": round(search_time, 2),
                "query_embedding_dim": len(query_embedding),
                "corpus_size": len(self.corpus)
            },
            "search_metadata": {
                "dense_results": len(dense_results),
                "sparse_results": len(sparse_results),
                "combined_results": len(combined_results),
                "reranked_results": len(reranked_results),
                "final_k": len(final_indices)
            }
        }
    
    async def _perform_multi_kb_retrieval(self, question: str, knowledge_bases: List[str], 
                                        conv_context: str, verbose: bool, timeout: int, max_chunks: int) -> Dict[str, Any]:
        """Perform retrieval across multiple knowledge bases"""
        from chat import enhanced_answer_multi_kb
        
        # Validate knowledge bases
        for kb_name in knowledge_bases:
            if kb_name not in self.knowledge_bases:
                raise ValueError(f"Knowledge base '{kb_name}' not found. Available: {list(self.knowledge_bases.keys())}")
        
        # Use the existing multi-KB function but extract retrieval data
        # We'll need to modify this to return retrieval context instead of generated answer
        # For now, let's implement a simplified version
        
        all_chunks = []
        all_kb_names = []
        total_search_time = 0
        
        for kb_name in knowledge_bases:
            kb_info = self.knowledge_bases[kb_name]
            embeddings_path = kb_info['embeddings_path']
            
            # Perform retrieval on this KB
            kb_result = await self._perform_single_kb_retrieval(
                question, embeddings_path, kb_name, conv_context, verbose, timeout, max_chunks // len(knowledge_bases)
            )
            
            all_chunks.extend(kb_result['chunks'])
            all_kb_names.append(kb_name)
            total_search_time += kb_result['metadata']['search_time_ms']
        
        # Sort by relevance score and take top chunks
        all_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        final_chunks = all_chunks[:max_chunks]
        
        return {
            "chunks": final_chunks,
            "knowledge_bases": all_kb_names,
            "metadata": {
                "total_chunks_found": len(final_chunks),
                "search_time_ms": round(total_search_time, 2),
                "kbs_searched": len(knowledge_bases)
            },
            "search_metadata": {
                "total_dense_results": sum(1 for _ in all_chunks),
                "total_sparse_results": sum(1 for _ in all_chunks),
                "final_k": len(final_chunks)
            }
        }
    
    def _format_retrieval_context(self, retrieval_result: Dict[str, Any], context_format: str) -> Dict[str, Any]:
        """Format retrieval context based on requested format"""
        chunks = retrieval_result['chunks']
        
        if context_format == 'raw':
            # Return raw chunks as-is
            return {
                "raw_chunks": chunks,
                "format": "raw"
            }
        
        elif context_format == 'markdown':
            # Format as markdown
            markdown_content = []
            citations = []
            
            for i, chunk in enumerate(chunks, 1):
                source_name = chunk['source_name']
                page = chunk.get('page', '')
                page_ref = f":{page}" if page else ""
                
                markdown_content.append(f"## Chunk {i}\n")
                markdown_content.append(f"**Source:** {source_name}{page_ref}\n")
                markdown_content.append(f"**Relevance:** {chunk['relevance_score']:.3f}\n")
                markdown_content.append(f"**Content:**\n{chunk['content']}\n")
                
                citations.append(f"[{i}] {source_name}{page_ref}")
            
            return {
                "markdown_content": "\n".join(markdown_content),
                "citations": citations,
                "format": "markdown"
            }
        
        else:  # structured (default)
            # Format as structured context ready for LLM
            structured_parts = []
            citations = []
            images = []
            
            for i, chunk in enumerate(chunks, 1):
                source_name = chunk['source_name']
                page = chunk.get('page', '')
                page_ref = f":{page}" if page else ""
                
                # Add to structured content
                structured_parts.append(f"[{i}] {chunk['content']}")
                citations.append(f"[{i}] {source_name}{page_ref}")
                
                # Check for images in metadata
                if 'image_path' in chunk.get('metadata', {}):
                    images.append({
                        "path": chunk['metadata']['image_path'],
                        "description": chunk['metadata'].get('image_description', ''),
                        "context": f"Referenced in {source_name}{page_ref}",
                        "chunk_id": chunk['chunk_id']
                    })
            
            return {
                "structured_text": "\n\n".join(structured_parts),
                "citations": citations,
                "images": images,
                "summary": f"Retrieved {len(chunks)} relevant chunks from {len(retrieval_result['knowledge_bases'])} knowledge base(s)",
                "format": "structured"
            }
    
    async def handle_list_knowledge_bases(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle list_knowledge_bases tool"""
        return [
            {
                "name": kb_info['name'],
                "display_name": kb_info['display_name'],
                "chunk_count": kb_info['chunk_count'],
                "file_size": kb_info['file_size'],
                "embeddings_path": kb_info['embeddings_path'],
                "is_current": kb_info['name'] == self.current_kb
            }
            for kb_info in self.knowledge_bases.values()
        ]
    
    async def handle_switch_knowledge_base(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle switch_knowledge_base tool"""
        kb_name = args.get('knowledge_base')
        
        if not kb_name:
            raise ValueError("Knowledge base name is required")
        
        if kb_name not in self.knowledge_bases:
            available_kbs = list(self.knowledge_bases.keys())
            raise ValueError(f"Knowledge base '{kb_name}' not found. Available: {available_kbs}")
        
        kb_info = self.knowledge_bases[kb_name]
        embeddings_path = kb_info['embeddings_path']
        
        if not os.path.exists(embeddings_path):
            raise ValueError(f"Embeddings file not found: {embeddings_path}")
        
        # Load the new knowledge base
        self.current_embeddings_path = embeddings_path
        self.corpus = load_corpus(embeddings_path)
        idx_sum, idx_full, E_sum, E_full, bm25 = build_indices(self.corpus)
        self.indices = {
            'faiss_sum': idx_sum,
            'faiss_full': idx_full,
            'E_sum': E_sum,
            'E_full': E_full,
            'bm25': bm25
        }
        self.query_encoder_name, self.query_encoder = load_query_encoder()
        self.current_kb = kb_name
        
        self.logger.info(f"Switched to knowledge base: {kb_name}")
        
        return {
            "switched_to": kb_name,
            "display_name": kb_info['display_name'],
            "chunk_count": kb_info['chunk_count'],
            "file_size": kb_info['file_size']
        }
    
    async def handle_get_kb_stats(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_kb_stats tool"""
        kb_name = args.get('knowledge_base', self.current_kb)
        
        if not kb_name or kb_name not in self.knowledge_bases:
            raise ValueError(f"Knowledge base '{kb_name}' not found")
        
        kb_info = self.knowledge_bases[kb_name]
        
        # Get additional stats from database if available
        stats = dict(kb_info)
        
        try:
            db = SynapseDB()
            folder_stats = db.get_folder_stats(kb_name)
            stats.update(folder_stats)
        except Exception as e:
            self.logger.warning(f"Could not get database stats: {e}")
        
        return stats
    
    async def handle_create_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle create_session tool"""
        session_id = args.get('session_id') or str(uuid.uuid4())
        
        if session_id in self.sessions:
            raise ValueError(f"Session '{session_id}' already exists")
        
        self.sessions[session_id] = ChatSession(auto_continue=False)
        
        return {
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
    
    async def handle_load_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle load_session tool"""
        session_file = args.get('session_file')

        if not session_file:
            raise ValueError("Session file path is required")

        # Always look for session files in the Sessions folder in the root directory
        root_dir = Path(__file__).parent.resolve()
        sessions_dir = root_dir / "Sessions"
        session_path = Path(session_file)
        if not session_path.is_absolute():
            session_path = sessions_dir / session_file

        if not session_path.exists():
            raise ValueError(f"Session file not found: {session_path}")

        session_id = session_path.stem

        # If already loaded, just return info about the existing session
        if session_id in self.sessions:
            session = self.sessions[session_id]
        else:
            session = ChatSession(str(session_path), auto_continue=True)
            self.sessions[session_id] = session

        return {
            "session_id": session_id,
            "session_file": str(session_path),
            "history_count": len(session.history),
            "status": "loaded"
        }
    
    async def handle_list_sessions(self, args: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle list_sessions tool"""
        sessions_dir = Path("sessions")
        active_sessions = []
        
        # Add currently active sessions
        for session_id, session in self.sessions.items():
            active_sessions.append({
                "session_id": session_id,
                "history_count": len(session.history),
                "status": "active",
                "session_file": session.session_file if session.session_file else None
            })
        
        # Add available session files
        if sessions_dir.exists():
            for session_file in sessions_dir.glob("*.json"):
                session_id = session_file.stem
                if session_id not in [s["session_id"] for s in active_sessions]:
                    try:
                        with open(session_file) as f:
                            data = json.load(f)
                        active_sessions.append({
                            "session_id": session_id,
                            "history_count": len(data.get('history', [])),
                            "status": "saved",
                            "session_file": str(session_file),
                            "modified_time": datetime.fromtimestamp(session_file.stat().st_mtime).isoformat()
                        })
                    except Exception as e:
                        self.logger.warning(f"Could not read session file {session_file}: {e}")
        
        return active_sessions
    
    async def handle_get_session_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_session_history tool"""
        session_id = args.get('session_id')
        limit = args.get('limit', 10)
        
        if not session_id:
            raise ValueError("Session ID is required")
        
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' not found")
        
        session = self.sessions[session_id]
        history = session.history[-limit:] if session.history else []
        
        return {
            "session_id": session_id,
            "history": history,
            "total_exchanges": len(session.history),
            "returned_exchanges": len(history)
        }
    
    async def handle_clear_session_history(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle clear_session_history tool"""
        session_id = args.get('session_id')
        
        if not session_id:
            raise ValueError("Session ID is required")
        
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' not found")
        
        session = self.sessions[session_id]
        history_count = len(session.history)
        session.clear_history()
        
        return {
            "session_id": session_id,
            "cleared_exchanges": history_count,
            "status": "cleared"
        }
    
    async def handle_export_session(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle export_session tool"""
        session_id = args.get('session_id')
        output_file = args.get('output_file')
        
        if not session_id or not output_file:
            raise ValueError("Session ID and output file are required")
        
        if session_id not in self.sessions:
            raise ValueError(f"Session '{session_id}' not found")
        
        session = self.sessions[session_id]
        session.export_session(output_file)
        
        return {
            "session_id": session_id,
            "output_file": output_file,
            "exported_exchanges": len(session.history),
            "status": "exported"
        }

    async def handle_get_processing_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_processing_status tool"""
        from initialize_fast import SynapseDB, FastEmbeddingService
        
        db = SynapseDB()
        embedding_service = FastEmbeddingService()
        
        folders = db.get_all_folders()
        
        status = {
            "total_folders": len(folders),
            "folder_status": []
        }
        
        for folder_key in folders:
            folder_stats = db.get_folder_stats(folder_key)
            embed_status = embedding_service.get_embedding_status(folder_key)
            
            status["folder_status"].append({
                "folder": folder_key,
                "completed_docs": folder_stats['completed_docs'],
                "pending_docs": folder_stats['pending_docs'],
                "failed_docs": folder_stats['failed_docs'],
                "total_chunks": folder_stats['total_chunks'],
                "embedded_chunks": folder_stats['embedded_chunks'],
                "completion_rate": embed_status['completion_rate']
            })
        
        return status
    
    async def handle_initialize_knowledge_base(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize_knowledge_base tool"""
        folder = args.get('folder')
        cleanup = args.get('cleanup', False)
        
        import subprocess
        
        cmd = [sys.executable, "initialize_fast.py", "--verbose"]
        
        if folder:
            cmd.extend(["--folder", folder])
        
        if cleanup:
            cmd.append("--cleanup")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        return {
            "command": " ".join(cmd),
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "status": "success" if result.returncode == 0 else "failed"
        }
    
    async def handle_set_verbose_mode(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle set_verbose_mode tool"""
        verbose = args.get('verbose')
        
        if verbose is None:
            raise ValueError("Verbose flag is required")
        
        self.verbose_mode = bool(verbose)
        
        return {
            "verbose_mode": self.verbose_mode,
            "status": "updated"
        }
    
    async def handle_get_server_info(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle get_server_info tool"""
        return {
            "server_name": "Synapse MCP Server",
            "version": "1.0.0",
            "current_kb": self.current_kb,
            "available_kbs": list(self.knowledge_bases.keys()),
            "active_sessions": len(self.sessions),
            "verbose_mode": self.verbose_mode,
            "artifacts_dir": self.artifacts_dir,
            "transports": ["http", "websocket"],
            "uptime": time.time() - getattr(self, 'start_time', time.time())
        }


# Transport Layer Implementations

class HTTPTransport:
    """HTTP transport for MCP server"""
    
    def __init__(self, mcp_server: MCPServer, port: int = 3000, cors: bool = True):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for HTTP transport. Install with: pip install flask flask-cors")
        
        self.mcp_server = mcp_server
        self.port = port
        self.app = Flask(__name__)
        
        if cors:
            CORS(self.app)
        
        self.setup_routes()
        
    def setup_routes(self):
        """Setup Flask routes for MCP protocol"""
        
        @self.app.route('/mcp', methods=['POST'])
        def handle_mcp_request():
            """Handle MCP JSON-RPC requests"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify(self.mcp_server.create_error_response(None, -32700, "Parse error")), 400
                
                # Handle batch requests
                if isinstance(data, list):
                    responses = []
                    for message in data:
                        response = asyncio.run(self.mcp_server.handle_message(message))
                        if response:
                            responses.append(response)
                    return jsonify(responses)
                else:
                    response = asyncio.run(self.mcp_server.handle_message(data))
                    if response:
                        return jsonify(response)
                    else:
                        return '', 204  # No Content for notifications
                        
            except Exception as e:
                self.mcp_server.logger.error(f"HTTP request error: {e}")
                return jsonify(self.mcp_server.create_error_response(None, -32603, "Internal error")), 500
        
        @self.app.route('/mcp', methods=['GET'])  
        def handle_mcp_info():
            """Get MCP server information"""
            return jsonify({
                "name": "Synapse MCP Server",
                "version": "1.0.0",
                "protocol": "Model Context Protocol",
                "transports": ["http", "websocket"],
                "tools_count": len(self.mcp_server.tools)
            })
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
        
    def run(self, debug: bool = False):
        """Run the HTTP server"""
        self.mcp_server.logger.info(f"Starting HTTP transport on port {self.port}")
        self.mcp_server.start_time = time.time()
        self.app.run(host='0.0.0.0', port=self.port, debug=debug, threaded=True)


class WebSocketTransport:
    """WebSocket transport for MCP server"""
    
    def __init__(self, mcp_server: MCPServer, port: int = 3001):
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets is required for WebSocket transport. Install with: pip install websockets")
        
        self.mcp_server = mcp_server
        self.port = port
        self.clients = set()
        
    async def handle_client(self, websocket, path):
        """Handle WebSocket client connection"""
        self.clients.add(websocket)
        client_addr = websocket.remote_address
        self.mcp_server.logger.info(f"WebSocket client connected: {client_addr}")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    response = await self.mcp_server.handle_message(data)
                    
                    if response:
                        await websocket.send(json.dumps(response))
                        
                except json.JSONDecodeError:
                    error_response = self.mcp_server.create_error_response(None, -32700, "Parse error")
                    await websocket.send(json.dumps(error_response))
                except Exception as e:
                    self.mcp_server.logger.error(f"WebSocket message error: {e}")
                    error_response = self.mcp_server.create_error_response(None, -32603, "Internal error")
                    await websocket.send(json.dumps(error_response))
                    
        except websockets.exceptions.ConnectionClosed:
            self.mcp_server.logger.info(f"WebSocket client disconnected: {client_addr}")
        finally:
            self.clients.remove(websocket)
    
    async def run(self):
        """Run the WebSocket server"""
        self.mcp_server.logger.info(f"Starting WebSocket transport on port {self.port}")
        self.mcp_server.start_time = time.time()
        
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            self.mcp_server.logger.info(f"WebSocket server running on ws://0.0.0.0:{self.port}")
            # Keep the server running
            await asyncio.Future()  # run forever


def main():
    """Main entry point for MCP server"""
    parser = argparse.ArgumentParser(description="Synapse MCP Server")
    parser.add_argument("--transport", choices=["http", "websocket", "both"], default="http", 
                      help="Transport protocol to use")
    parser.add_argument("--http-port", type=int, default=3000, 
                      help="Port for HTTP transport")
    parser.add_argument("--ws-port", type=int, default=3001,
                      help="Port for WebSocket transport")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts",
                      help="Directory containing knowledge base artifacts")
    parser.add_argument("--debug", action="store_true",
                      help="Enable debug mode")
    parser.add_argument("--cors", action="store_true", default=True,
                      help="Enable CORS for HTTP transport")
    
    args = parser.parse_args()
    
    # Check environment
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL")
    if not api_key or not base_url:
        print("❌ Environment variables not set!")
        print("Set LITELLM_API_KEY and LITELLM_BASE_URL")
        return 1
    
    # Initialize MCP server
    try:
        mcp_server = MCPServer(artifacts_dir=args.artifacts_dir)
    except Exception as e:
        print(f"❌ Failed to initialize MCP server: {e}")
        return 1
    
    # Start transport(s)
    if args.transport == "http":
        transport = HTTPTransport(mcp_server, port=args.http_port, cors=args.cors)
        transport.run(debug=args.debug)
        
    elif args.transport == "websocket":
        transport = WebSocketTransport(mcp_server, port=args.ws_port)
        asyncio.run(transport.run())
        
    elif args.transport == "both":
        # Run both transports
        def run_http():
            transport = HTTPTransport(mcp_server, port=args.http_port, cors=args.cors)
            transport.run(debug=args.debug)
        
        async def run_websocket():
            transport = WebSocketTransport(mcp_server, port=args.ws_port)
            await transport.run()
        
        # Start HTTP in a thread
        http_thread = threading.Thread(target=run_http, daemon=True)
        http_thread.start()
        
        # Run WebSocket in main thread  
        asyncio.run(run_websocket())
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
