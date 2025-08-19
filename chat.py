#!/usr/bin/env python3
"""
Synapse Interactive Chat Interface

A user-friendly chat CLI with conversation memory, context-aware responses,
and detailed retrieval insights. Provides an interactive experience for
querying your knowledge base with full conversation history.

Features:
- Conversation memory and context
- Verbose retrieval mode showing all ranking steps
- Interactive commands (/help, /clear, /history, /verbose)
- Rich context display with source attribution
- Session management and export
- Graceful error handling

Usage:
    python chat.py [--embeddings artifacts/embedded_with_images.npz] [--verbose] [--session session.json]
    python chat.py --test_gui  # Launch local web app GUI
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Import pipeline components
from pipeline.query import (
    load_corpus, build_indices, load_query_encoder, encode_query,
    dense_search, sparse_search, rrf, per_doc_cap, rerank, mmr_select,
    format_citations, build_prompt, call_gemini_via_litellm
)


def load_default_config():
    """Load default configuration from file."""
    default_config_path = Path("default_config.json")
    default_config = {
        "system_prompt": (
            "You are an engineering assistant. Keep your responses relevant and based on the context. "
            "Provide as much detail as possible but keep it concise. When asked for tables - reconstruct tables based on retrieved context. "
            "IMPORTANT: Never include any citation numbers, brackets like [1], [2], or references to 'Context Document X' in your response. "
            "Write naturally without any citation markers - the system will add citations separately. "
            "When interpreting OCR text from technical diagrams, be careful about common OCR errors: '2x' may appear as part of adjacent text, 'I2C' may appear as '12C', 'x4' formatting may be inconsistent. "
            "Always double-check technical specifications and component counts against the visual context when available."
        ),
        "topk": 10,
        "per_doc": 8,
        "lambda_mmr": 0.8,
        "timeout": 60,
        "verbose": False,
        "max_images": 2,
        "images_enabled": True,
    }
    
    try:
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                saved_config = json.load(f)
                default_config.update(saved_config)
    except Exception as e:
        print(f"⚠️  Could not load default config: {e}")
    
    return default_config


def save_default_config(config):
    """Save default configuration to file."""
    default_config_path = Path("default_config.json")
    try:
        with open(default_config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"⚠️  Could not save default config: {e}")
        return False


class ChatSession:
    """Manages conversation history, context, and configuration."""
    
    def __init__(self, session_file: Optional[str] = None, auto_continue: bool = True, default_config: Optional[Dict] = None):
        # Auto-generate or find existing session file if not provided
        if session_file is None:
            session_dir = Path("sessions")
            session_dir.mkdir(exist_ok=True)
            
            # Try to find and continue the most recent session
            if auto_continue:
                recent_session = self._find_most_recent_session(session_dir)
                if recent_session:
                    session_file = str(recent_session)
                    print(f"📚 Continuing previous session: {recent_session.name}")
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    session_file = str(session_dir / f"chat_session_{timestamp}.json")
                    print(f"🆕 Starting new session: {Path(session_file).name}")
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_file = str(session_dir / f"chat_session_{timestamp}.json")
        
        self.session_file = session_file
        self.history: List[Dict] = []
        self.created_at = datetime.now().isoformat()
        
        # Initialize config - use provided default or load from file
        if default_config is None:
            default_config = load_default_config()
        self.config = default_config.copy()
        
        self.load_session()
    
    def _find_most_recent_session(self, session_dir: Path) -> Optional[Path]:
        """Find the most recently modified session file."""
        try:
            session_files = list(session_dir.glob("chat_session_*.json"))
            if not session_files:
                return None
            
            # Find the most recently modified session file
            most_recent = max(session_files, key=lambda f: f.stat().st_mtime)
            
            # Only continue if the session was modified within the last 24 hours
            # This prevents continuing very old sessions accidentally
            from datetime import timedelta
            now = datetime.now()
            file_time = datetime.fromtimestamp(most_recent.stat().st_mtime)
            if now - file_time < timedelta(hours=24):
                return most_recent
            
            return None
        except Exception:
            return None
    
    def add_exchange(self, question: str, answer: str, sources: str, retrieval_info: Optional[Dict] = None):
        """Add a question-answer exchange to history."""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "sources": sources,
            "retrieval_info": retrieval_info or {}
        }
        self.history.append(exchange)
        self.save_session()
    
    def get_context(self, last_n: int = 3) -> str:
        """Get recent conversation context for continuity."""
        if not self.history:
            return ""
        
        context_parts = []
        for exchange in self.history[-last_n:]:
            context_parts.append(f"Previous Q: {exchange['question']}")
            # Truncate long answers for context
            answer = exchange['answer']
            if len(answer) > 200:
                answer = answer[:200] + "..."
            context_parts.append(f"Previous A: {answer}")
        
        return "\n".join(context_parts)
    
    def save_session(self):
        """Save session to file."""
        if not self.session_file:
            return
        
        session_data = {
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat(),
            "history": self.history,
            "config": self.config
        }
        
        try:
            with open(self.session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
        except Exception as e:
            print(f"⚠️  Could not save session: {e}")
    
    def load_session(self):
        """Load session from file."""
        if not self.session_file or not os.path.exists(self.session_file):
            return
        
        try:
            with open(self.session_file, 'r') as f:
                session_data = json.load(f)
            
            self.created_at = session_data.get("created_at", self.created_at)
            self.history = session_data.get("history", [])
            
            # Load session-specific config if available, otherwise keep current config
            if "config" in session_data:
                self.config.update(session_data["config"])
                print(f"📚 Loaded session with {len(self.history)} previous exchanges and custom config")
            else:
                print(f"📚 Loaded session with {len(self.history)} previous exchanges (using default config)")
                
        except Exception as e:
            print(f"⚠️  Could not load session: {e}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.save_session()
    
    def update_config(self, new_config: Dict):
        """Update session configuration and save."""
        self.config.update(new_config)
        self.save_session()
    
    def get_config(self) -> Dict:
        """Get current session configuration."""
        return self.config.copy()
    
    def export_session(self, filepath: str):
        """Export session to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "created_at": self.created_at,
                    "exported_at": datetime.now().isoformat(),
                    "total_exchanges": len(self.history),
                    "history": self.history,
                    "config": self.config
                }, f, indent=2)
            print(f"📝 Session exported to {filepath}")
        except Exception as e:
            print(f"❌ Export failed: {e}")


class VerboseRetrieval:
    """Handles verbose retrieval output for debugging and transparency."""
    
    @staticmethod
    def print_header(step: str):
        """Print a step header."""
        print(f"\n{'='*60}")
        print(f"🔍 {step}")
        print('='*60)
    
    @staticmethod
    def print_query_encoding(question: str, model_name: str):
        """Show query encoding details."""
        VerboseRetrieval.print_header("QUERY ENCODING")
        print(f"Question: {question}")
        print(f"Model: {model_name}")
        print(f"Query prefix: 'query: {question[:50]}{'...' if len(question) > 50 else ''}'")
    
    @staticmethod
    def print_dense_results(summary_results: List[int], full_results: List[int], items: List[Dict]):
        """Show dense retrieval results."""
        VerboseRetrieval.print_header("DENSE RETRIEVAL")
        
        print(f"📊 Summary vector results (top 10):")
        for i, idx in enumerate(summary_results[:10]):
            item = items[idx]
            source = Path(item['source_path']).name
            summary = item['summary_text'][:100] + '...' if len(item['summary_text']) > 100 else item['summary_text']
            print(f"  {i+1:2d}. [{idx:4d}] {source} - {summary}")
        
        print(f"\n📊 Full content vector results (top 10):")
        for i, idx in enumerate(full_results[:10]):
            item = items[idx]
            source = Path(item['source_path']).name
            content = item['full_text'][:100] + '...' if len(item['full_text']) > 100 else item['full_text']
            print(f"  {i+1:2d}. [{idx:4d}] {source} - {content}")
    
    @staticmethod
    def print_sparse_results(sparse_results: List[int], items: List[Dict]):
        """Show sparse (BM25) retrieval results."""
        VerboseRetrieval.print_header("SPARSE RETRIEVAL (BM25)")
        
        print(f"📊 BM25 keyword results (top 10):")
        for i, idx in enumerate(sparse_results[:10]):
            item = items[idx]
            source = Path(item['source_path']).name
            content = item['full_text'][:100] + '...' if len(item['full_text']) > 100 else item['full_text']
            print(f"  {i+1:2d}. [{idx:4d}] {source} - {content}")
    
    @staticmethod
    def print_fusion_results(fused_results: List[int], items: List[Dict]):
        """Show RRF fusion results."""
        VerboseRetrieval.print_header("RECIPROCAL RANK FUSION (RRF)")
        
        print(f"📊 Fused results after per-document capping (top 15):")
        for i, idx in enumerate(fused_results[:15]):
            item = items[idx]
            source = Path(item['source_path']).name
            doc_id = str(item['document_id'])[:8] + '...'
            print(f"  {i+1:2d}. [{idx:4d}] {source} (doc: {doc_id})")
    
    @staticmethod
    def print_rerank_results(reranked_results: List[int], items: List[Dict]):
        """Show reranking results."""
        VerboseRetrieval.print_header("RERANKING")
        
        print(f"📊 Cross-encoder reranked results (top 15):")
        for i, idx in enumerate(reranked_results[:15]):
            item = items[idx]
            source = Path(item['source_path']).name
            content = item['full_text'][:150] + '...' if len(item['full_text']) > 150 else item['full_text']
            print(f"  {i+1:2d}. [{idx:4d}] {source}")
            print(f"       {content}")
    
    @staticmethod
    def print_final_context(final_indices: List[int], items: List[Dict]):
        """Show final context selection."""
        VerboseRetrieval.print_header("FINAL CONTEXT (MMR DIVERSIFIED)")
        
        print(f"📊 Selected contexts for LLM ({len(final_indices)} chunks):")
        for i, idx in enumerate(final_indices):
            item = items[idx]
            source = Path(item['source_path']).name
            meta = item.get('metadata', {})
            page = meta.get('page_number') or meta.get('slide_number')
            page_info = f" (page {page})" if page else ""
            
            content = item['full_text'][:200] + '...' if len(item['full_text']) > 200 else item['full_text']
            print(f"  [{i+1}] {source}{page_info}")
            print(f"      {content}")
            print()


def add_automatic_citations(text: str, final_indices: List[int], items: List[Dict]) -> str:
    """Add automatic citations to text based on content relevance."""
    import re
    
    # Simple heuristic: add citations at the end of sentences that contain specific keywords
    sentences = re.split(r'(?<=[.!?])\s+', text)
    result_sentences = []
    
    for sentence in sentences:
        if sentence.strip():
            # For now, add citation [1] to the first significant sentence
            # This is a simplified approach - could be made more sophisticated
            if len(result_sentences) == 0 and len(sentence) > 20:
                sentence = sentence.strip() + " [1]"
            result_sentences.append(sentence)
    
    return ' '.join(result_sentences)


def enhanced_answer(
    question: str,
    embeddings_path: str,
    conversation_context: str = "",
    verbose: bool = False,
    k_dense_sum: int = 60,
    k_dense_full: int = 60,
    k_sparse: int = 60,
    per_doc: int = 4,
    final_k: int = 8,
    lambda_mmr: float = 0.7,
    timeout: int = 60,
    system_override: Optional[str] = None,
    chunked_path: Optional[str] = None,
    images_enabled: bool = True,
    max_images: int = 2,
) -> Tuple[str, str, Dict, List[str]]:
    """Enhanced answer function with verbose output and conversation context."""
    
    # Load corpus and build indices
    items = load_corpus(embeddings_path)
    if not items:
        raise RuntimeError("No embedding records found. Run initialize.py first.")

    idx_sum, idx_full, E_sum, E_full, bm25 = build_indices(items)
    model_name, q_encoder = load_query_encoder()
    
    # Encode query
    if verbose:
        VerboseRetrieval.print_query_encoding(question, model_name)
    
    qv = encode_query(question, q_encoder)

    # Dense search
    c_sum = dense_search(qv, idx_sum, k_dense_sum)
    c_full = dense_search(qv, idx_full, k_dense_full)
    
    if verbose:
        VerboseRetrieval.print_dense_results(c_sum, c_full, items)

    # Sparse search
    c_sparse = sparse_search(question, bm25, k_sparse)
    
    if verbose:
        VerboseRetrieval.print_sparse_results(c_sparse, items)

    # Fusion
    fused = rrf([c_sum, c_full, c_sparse], weights=[0.9, 1.2, 0.8], rrf_k=60, base=60)
    fused = per_doc_cap(fused, items, per_doc)
    
    if verbose:
        VerboseRetrieval.print_fusion_results(fused, items)

    # Rerank
    reranked = rerank(question, fused[:100], items, topn=20)
    
    if verbose:
        VerboseRetrieval.print_rerank_results(reranked, items)

    # MMR diversification
    final_indices = mmr_select(qv[0], reranked, E_full, lambda_mmr, min(final_k, len(reranked) or 0))
    
    if verbose:
        VerboseRetrieval.print_final_context(final_indices, items)

    # Extract relevant images first so we can include descriptions in the LLM prompt
    image_paths = []
    image_descriptions = []
    if chunked_path and images_enabled and max_images > 0:
        try:
            from pipeline.query import extract_relevant_images
            # Extract images with the specified limit
            all_image_paths = extract_relevant_images(question, final_indices, items)
            image_paths = all_image_paths[:max_images]  # Limit to max_images
            
            # Get image descriptions for LLM context
            for img_path in image_paths:
                # Find corresponding image metadata
                for item in items:
                    metadata = item.get("metadata", {})
                    if (metadata.get("content_format") == "extracted_image" and 
                        metadata.get("image_path") == img_path):
                        # Get OCR text and context
                        ocr_text = metadata.get("ocr_text", "")
                        doc_context = metadata.get("document_context", "")
                        img_type = metadata.get("image_type", "diagram")
                        
                        desc = f"Technical {img_type}"
                        if ocr_text:
                            desc += f" containing: {ocr_text[:200]}"
                        if doc_context:
                            desc += f" (Context: {doc_context[:100]})"
                        
                        image_descriptions.append(desc)
                        break
                else:
                    # Fallback description based on filename
                    img_name = os.path.basename(img_path)
                    image_descriptions.append(f"Technical diagram: {img_name}")
                    
        except Exception as img_error:
            if verbose:
                print(f"⚠️  Image extraction failed: {img_error}")
    elif verbose:
        print("⚠️  No chunked file provided - skipping image extraction")

    # Build prompt with conversation context
    sources_block, cmap = format_citations(final_indices, items)
    system, user = build_prompt(question, final_indices, items)
    
    # Add image descriptions to the prompt if available
    if image_descriptions:
        image_context = "\n\nRelevant visual content available:\n" + "\n".join(
            f"- {desc}" for desc in image_descriptions
        )
        user += image_context
        if verbose:
            print(f"📷 Added {len(image_descriptions)} image descriptions to LLM context")
    
    # Add conversation context if available
    if conversation_context:
        system += "\n\nFor context, here is our recent conversation:\n" + conversation_context
    # Admin/system override from GUI
    if system_override and system_override.strip():
        if verbose:
            print(f"🔄 SYSTEM PROMPT OVERRIDE APPLIED")
            print(f"   Original: {system[:100]}...")
            print(f"   Override: {system_override[:100]}...")
        system = str(system_override)
    elif verbose:
        print(f"🔧 USING DEFAULT SYSTEM PROMPT")
    
    if verbose:
        VerboseRetrieval.print_header("LLM GENERATION")
        print(f"System prompt: {system[:500]}{'...' if len(system) > 500 else ''}")
        print(f"User prompt: {user[:300]}{'...' if len(user) > 300 else ''}")
        print(f"Timeout: {timeout}s")
    
    # Generate answer
    start_time = time.time()
    out = call_gemini_via_litellm(system, user, timeout=timeout)
    generation_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Generation completed in {generation_time:.1f}s")

    # Images were already extracted before LLM generation above
    
    if verbose and image_paths:
        print(f"📷 FOUND {len(image_paths)} RELEVANT IMAGES:")
        for i, img_path in enumerate(image_paths, 1):
            print(f"   [{i}] {os.path.basename(img_path)}")
    elif verbose:
        print(f"📷 No relevant images found for this query")

    # Prepare retrieval info for session storage
    retrieval_info = {
        "model_used": model_name,
        "dense_summary_results": len(c_sum),
        "dense_full_results": len(c_full), 
        "sparse_results": len(c_sparse),
        "fused_results": len(fused),
        "reranked_results": len(reranked),
        "final_contexts": len(final_indices),
        "generation_time_s": generation_time,
        "sources_count": len(final_indices),
        "images_found": len(image_paths)
    }

    return out, sources_block, retrieval_info, image_paths


def print_welcome(session: Optional[ChatSession] = None):
    """Print welcome message and instructions."""
    print("\n" + "="*80)
    print("💬 Synapse Interactive Chat")
    print("="*80)
    print("Welcome! Ask questions about your knowledge base.")
    
    if session:
        session_name = Path(session.session_file).name
        if session.history:
            print(f"📚 Session: {session_name} ({len(session.history)} previous exchanges)")
        else:
            print(f"🆕 Session: {session_name} (new session)")
    
    print("Type /help for commands, /quit to exit.")
    print("💡 Your conversations are automatically saved and can be resumed!")
    print("="*80)


def print_help():
    """Print help information."""
    print("\n📋 Available Commands:")
    print("  /help         - Show this help message")
    print("  /quit, /exit  - Exit the chat")
    print("  /clear        - Clear conversation history")
    print("  /history      - Show conversation history")
    print("  /verbose      - Toggle verbose retrieval mode")
    print("  /export FILE  - Export session to file")
    print("  /stats        - Show session statistics")
    print("  /sessions     - List all available sessions")
    print("  /new          - Start a new session")
    print("\n💡 Tips:")
    print("  • Ask follow-up questions - I remember our conversation!")
    print("  • Use /verbose to see detailed retrieval steps")
    print("  • Questions can reference previous answers")
    print("  • Sources are provided for fact verification")
    print("  • Sessions auto-save and auto-continue within 24 hours")


def print_stats(session: ChatSession, items: List[Dict]):
    """Print session and knowledge base statistics."""
    print(f"\n📊 Session Statistics:")
    print(f"  • Exchanges: {len(session.history)}")
    print(f"  • Session started: {session.created_at}")
    print(f"  • Session file: {Path(session.session_file).name}")
    print(f"  • Knowledge base: {len(items):,} chunks")
    
    if session.history:
        # Analyze question types
        total_chars = sum(len(ex['question']) + len(ex['answer']) for ex in session.history)
        avg_q_len = sum(len(ex['question']) for ex in session.history) / len(session.history)
        avg_a_len = sum(len(ex['answer']) for ex in session.history) / len(session.history)
        
        print(f"  • Total conversation: {total_chars:,} characters")
        print(f"  • Average question length: {avg_q_len:.0f} characters")
        print(f"  • Average answer length: {avg_a_len:.0f} characters")


def list_sessions():
    """List all available session files."""
    session_dir = Path("sessions")
    if not session_dir.exists():
        print("📝 No sessions directory found")
        return
    
    session_files = sorted(session_dir.glob("chat_session_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
    
    if not session_files:
        print("📝 No session files found")
        return
    
    print(f"\n📚 Available Sessions ({len(session_files)} total):")
    
    for i, session_file in enumerate(session_files[:10], 1):  # Show last 10 sessions
        try:
            stat = session_file.stat()
            modified = datetime.fromtimestamp(stat.st_mtime)
            size = stat.st_size
            
            # Try to load session to get exchange count
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                exchanges = len(session_data.get("history", []))
                exchange_info = f" ({exchanges} exchanges)"
            except:
                exchange_info = ""
            
            # Check if this is recent (within 24 hours)
            from datetime import timedelta
            is_recent = datetime.now() - modified < timedelta(hours=24)
            recent_marker = " 🔥" if is_recent else ""
            
            print(f"  {i:2d}. {session_file.name}{exchange_info}")
            print(f"      Modified: {modified.strftime('%Y-%m-%d %H:%M:%S')}{recent_marker}")
            print(f"      Size: {size:,} bytes")
            print()
            
        except Exception as e:
            print(f"  {i:2d}. {session_file.name} (error reading: {e})")
    
    if len(session_files) > 10:
        print(f"  ... and {len(session_files) - 10} more sessions")
    
    print("💡 Use --session <filename> to load a specific session")


def start_new_session(default_config: Optional[Dict] = None) -> ChatSession:
    """Start a new session, forcing creation of a new file."""
    session_dir = Path("sessions")
    session_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = str(session_dir / f"chat_session_{timestamp}.json")
    print(f"🆕 Started new session: {Path(session_file).name}")
    
    if default_config is None:
        default_config = load_default_config()
    
    return ChatSession(session_file, auto_continue=False, default_config=default_config)


def run_gui(embeddings_path: str, default_timeout: int = 60) -> int:
    try:
        from flask import Flask, request, jsonify, render_template
    except Exception as exc:
        print("❌ Flask is required. Run: pip install flask")
        return 2
    try:
        import markdown  # type: ignore
    except Exception as exc:
        print("❌ 'markdown' package is required for proper rendering. Run: pip install markdown")
        return 2

    # Load KB once
    try:
        items = load_corpus(embeddings_path)
        if not items:
            print("❌ No embeddings found in file")
            return 1
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return 1

    # Derive chunked file path for image support
    chunked_path = None
    try:
        from pathlib import Path
        embeddings_file = Path(embeddings_path)
        potential_chunked = embeddings_file.with_name("chunked.jsonl")
        if potential_chunked.exists():
            chunked_path = str(potential_chunked)
            print(f"✅ Found chunked file for image support: {chunked_path}")
        else:
            # Try alternative naming
            alt_chunked = embeddings_file.parent / "chunked_with_images.jsonl"
            if alt_chunked.exists():
                chunked_path = str(alt_chunked)
                print(f"✅ Found chunked file for image support: {chunked_path}")
            else:
                print("⚠️  No chunked file found - image display will be limited")
    except Exception as e:
        print(f"⚠️  Could not check for chunked file: {e}")
        chunked_path = None

    # Load default configuration
    default_config = load_default_config()
    default_config["timeout"] = default_timeout  # Override timeout with CLI argument
    
    # Persistent session using CLI's ChatSession with default config
    session = ChatSession(auto_continue=True, default_config=default_config)
    session_history: List[Dict] = []  # mirrored in-memory for quick UI rendering

    app = Flask(__name__)

    @app.get("/")
    def index():
        # Get current session config for template
        current_config = session.get_config()
        return render_template(
            'index.html',
            system_prompt=current_config["system_prompt"],
            topk=current_config["topk"],
            per_doc=current_config["per_doc"],
            lambda_mmr=current_config["lambda_mmr"],
            timeout=current_config["timeout"],
            embeddings_path=os.path.abspath(embeddings_path),
        )

    @app.get("/api/config")
    def get_config():
        return jsonify({**session.get_config()})
    
    @app.get("/api/config/default")
    def get_default_config():
        """Get the default configuration."""
        return jsonify(load_default_config())



    @app.post("/api/config")
    def set_config():
        """Update current session configuration."""
        data = request.get_json(force=True)
        valid_keys = ["system_prompt","topk","per_doc","lambda_mmr","timeout","verbose","max_images","images_enabled"]
        config_update = {}
        
        for k in valid_keys:
            if k in data and data[k] is not None:
                config_update[k] = data[k]
        
        if config_update:
            session.update_config(config_update)
        
        return jsonify({"ok": True})
    
    @app.post("/api/config/save-as-default")
    def save_config_as_default():
        """Save current session config as the new default."""
        try:
            current_config = session.get_config()
            if save_default_config(current_config):
                return jsonify({"ok": True, "message": "Configuration saved as default"})
            else:
                return jsonify({"ok": False, "error": "Failed to save default configuration"}), 500
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
    
    @app.post("/api/config/reset-to-default")
    def reset_to_default():
        """Reset current session config to default values."""
        try:
            default_config = load_default_config()
            default_config["timeout"] = default_timeout  # Preserve CLI timeout setting
            session.update_config(default_config)
            return jsonify({"ok": True, "message": "Configuration reset to default"})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.get("/api/history")
    def get_history():
        # Load from persistent session on first call
        nonlocal session_history
        if not session_history and session.history:
            for ex in session.history:
                # Convert stored plain text to HTML for UI rendering
                a_html = markdown.markdown(str(ex.get("answer", "")), extensions=["tables", "fenced_code", "nl2br", "toc", "attr_list"])
                s_html = markdown.markdown(str(ex.get("sources", "")), extensions=["tables", "fenced_code", "nl2br"])
                session_history.append({
                    "q": ex.get("question", ""),
                    "a_html": a_html,
                    "sources_html": s_html,
                    "ts": ex.get("timestamp", "")
                })
        return jsonify({"history": session_history, "session_file": session.session_file})

    @app.get("/api/sessions")
    def list_sessions_api():
        from datetime import timedelta
        session_dir = Path("sessions")
        result: List[Dict] = []
        if session_dir.exists():
            files = sorted(session_dir.glob("chat_session_*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            for fpath in files[:50]:
                try:
                    stat = fpath.stat()
                    with open(fpath, 'r') as f:
                        data = json.load(f)
                    result.append({
                        "name": fpath.name,
                        "path": str(fpath.resolve()),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "exchanges": len(data.get("history", []))
                    })
                except Exception:
                    continue
        return jsonify({"sessions": result, "current": session.session_file})

    @app.post("/api/session/new")
    def new_session_api():
        nonlocal session, session_history
        # Create new session with current default config
        current_default = load_default_config()
        current_default["timeout"] = default_timeout
        session = start_new_session(current_default)
        session_history = []
        return jsonify({"ok": True, "session_file": session.session_file})

    @app.post("/api/session/load")
    def load_session_api():
        nonlocal session, session_history
        data = request.get_json(force=True)
        fname = str(data.get("filename") or "").strip()
        if not fname:
            return jsonify({"ok": False, "error": "filename required"}), 400
        # Resolve path inside sessions dir
        target = Path("sessions") / fname
        if not target.exists():
            return jsonify({"ok": False, "error": "session file not found"}), 404
        
        # Load session with default config as fallback
        default_config = load_default_config()
        default_config["timeout"] = default_timeout
        session = ChatSession(str(target), auto_continue=False, default_config=default_config)
        session_history = []
        return jsonify({"ok": True, "session_file": session.session_file})

    @app.post("/api/ask")
    def ask():
        data = request.get_json(force=True)
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"answer":"", "sources":""})

        # Get image settings from request (with fallback to config defaults)
        images_enabled = data.get("images_enabled", True)
        max_images = data.get("max_images", 2)

        # Conversation context for parity with CLI
        conv_ctx = session.get_context(last_n=3)
        try:
            # Get current session config
            current_config = session.get_config()
            
            answer_text, sources_block, retrieval_info, image_paths = enhanced_answer(
                question=question,
                embeddings_path=embeddings_path,
                conversation_context=conv_ctx,
                verbose=bool(current_config.get("verbose", False)),
                per_doc=int(current_config.get("per_doc", 8)),
                final_k=int(current_config.get("topk", 10)),
                lambda_mmr=float(current_config.get("lambda_mmr", 0.8)),
                timeout=int(current_config.get("timeout", default_timeout)),
                system_override=current_config.get("system_prompt"),
                chunked_path=chunked_path,
                images_enabled=images_enabled,
                max_images=max_images,
            )
        except Exception as e:
            print(f"❌ Error in /api/ask: {e}")
            return jsonify({"ok": False, "error": str(e)}), 500

        # Add to persistent session
        session.add_exchange(question, answer_text, sources_block, retrieval_info)

        # Keep in-memory history for UI
        # Process answer with better markdown handling
        processed_answer = str(answer_text)
        # For now, skip automatic citations as they need more sophisticated implementation
        # processed_answer = add_automatic_citations(processed_answer, final_indices, items)
        
        a_html = markdown.markdown(processed_answer, extensions=["tables", "fenced_code", "nl2br", "toc", "attr_list", "def_list"])
        s_html = markdown.markdown(str(sources_block), extensions=["tables", "fenced_code", "nl2br"])
        session_history.append({"q": question, "a_html": a_html, "sources_html": s_html, "ts": time.time()})

        # Convert absolute paths to relative paths for the web interface
        relative_image_paths = []
        for img_path in image_paths:
            try:
                # Convert to relative path from project root
                rel_path = os.path.relpath(img_path, start=os.getcwd())
                relative_image_paths.append(rel_path)
            except Exception:
                # Fallback to absolute path if relative conversion fails
                relative_image_paths.append(img_path)

        resp = {
            "answer": answer_text, 
            "sources": sources_block, 
            "answer_html": a_html, 
            "sources_html": s_html,
            "images": relative_image_paths
        }
        if session.get_config().get("verbose"):
            resp["retrieval_info"] = retrieval_info
        return jsonify(resp)

    @app.post("/api/clear")
    def clear():
        session_history.clear()
        session.clear_history()
        return jsonify({"ok": True})

    @app.get("/api/image/<path:image_path>")
    def serve_image(image_path: str):
        """Serve images from the file system with security checks."""
        try:
            from flask import send_file, abort
            
            # Convert relative path back to absolute
            if not os.path.isabs(image_path):
                abs_path = os.path.abspath(os.path.join(os.getcwd(), image_path))
            else:
                abs_path = image_path
            
            # Security check: ensure file exists and is actually an image
            if not os.path.isfile(abs_path):
                abort(404)
            
            # Check file extension
            allowed_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
            file_ext = os.path.splitext(abs_path)[1].lower()
            if file_ext not in allowed_extensions:
                abort(403)
            
            # Additional security: ensure path is within the project directory or Data directory
            project_root = os.path.abspath(os.getcwd())
            if not abs_path.startswith(project_root):
                abort(403)
                
            return send_file(abs_path)
            
        except Exception as e:
            print(f"Error serving image {image_path}: {e}")
            abort(500)

    @app.get("/api/export")
    def export():
        from flask import Response
        data = json.dumps({"history": session_history}, indent=2)
        return Response(data, mimetype='application/json')

    app.run(host="127.0.0.1", port=7860, debug=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Synapse Interactive Chat")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="artifacts/embedded_with_images.npz",
        help="Path to embeddings file"
    )
    parser.add_argument("--session", type=str, help="Specific session file for conversation history")
   
    parser.add_argument("--new-session", action="store_true", help="Force start a new session")
    parser.add_argument("--verbose", action="store_true", help="Start with verbose retrieval mode enabled")
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout in seconds")
    parser.add_argument("--test_gui", action="store_true", help="Launch local test GUI instead of CLI")

    args = parser.parse_args()

    api_key = os.getenv("LITELLM_API_KEY"); base_url = os.getenv("LITELLM_BASE_URL")
    if not api_key or not base_url:
        print("❌ Environment variables not set!\nSet LITELLM_API_KEY and LITELLM_BASE_URL")
        return 1

    embeddings_path = os.path.abspath(args.embeddings)
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings file not found: {embeddings_path}\nRun 'python initialize.py' first.")
        return 1

    if args.test_gui:
        return run_gui(embeddings_path, default_timeout=int(args.timeout))

    # ------------- existing CLI startup -------------
    print("🔄 Loading knowledge base...")
    try:
        items = load_corpus(embeddings_path)
        if not items:
            print("❌ No embeddings found in file"); return 1
        print(f"✅ Loaded {len(items):,} chunks from knowledge base")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}"); return 1

    auto_continue = not args.new_session
    session = ChatSession(args.session, auto_continue=auto_continue)
    verbose_mode = args.verbose
    print_welcome(session)

    try:
        while True:
            try:
                question = input("\n💬 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!"); break
            if not question: continue
            if question.startswith('/'):
                cmd_parts = question[1:].split(None, 1)
                cmd = cmd_parts[0].lower()
                if cmd in ['quit','exit']: print("👋 Goodbye!"); break
                elif cmd=='help': print_help()
                elif cmd=='clear': session.clear_history(); print("🗑️  Conversation history cleared")
                elif cmd=='history':
                    if not session.history: print("📝 No conversation history yet")
                    else:
                        print(f"\n📚 Conversation History ({len(session.history)} exchanges):")
                        for i, ex in enumerate(session.history[-10:], 1):
                            timestamp = ex['timestamp'][:19].replace('T',' ')
                            print(f"\n[{timestamp}] Q{i}: {ex['question']}")
                            answer = ex['answer'][:150] + '...' if len(ex['answer'])>150 else ex['answer']
                            print(f"[{timestamp}] A{i}: {answer}")
                elif cmd=='verbose': verbose_mode = not verbose_mode; print(f"🔍 Verbose mode: {'ON' if verbose_mode else 'OFF'}")
                elif cmd=='export':
                    if len(cmd_parts)>1: session.export_session(cmd_parts[1])
                    else:
                        default_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"; session.export_session(default_file)
                elif cmd=='stats': items2 = load_corpus(embeddings_path); print_stats(session, items2)
                elif cmd=='sessions': list_sessions()
                elif cmd=='new': session = start_new_session(); print("🔄 Switched to new session. Previous context cleared.")
                else:
                    print(f"❓ Unknown command: /{cmd}\nType /help for available commands")
                continue
            # Process question
            print("🤔 Thinking..."); start_time = time.time()
            try:
                context = session.get_context(last_n=3)
                answer, sources, retrieval_info, image_paths = enhanced_answer(
                    question=question, embeddings_path=embeddings_path, conversation_context=context,
                    verbose=verbose_mode, timeout=args.timeout, chunked_path=None,
                    images_enabled=True, max_images=2)
                total_time = time.time() - start_time
                print(f"\n🤖 Assistant ({total_time:.1f}s):\n{answer.strip()}\n\n📚 Sources:\n{sources}")
                
                # Show relevant images if found
                if image_paths:
                    print(f"\n🖼️  Relevant Images ({len(image_paths)}):")
                    for i, img_path in enumerate(image_paths, 1):
                        print(f"  [{i}] {os.path.basename(img_path)}")
                        print(f"      {img_path}")
                
                session.add_exchange(question, answer, sources, retrieval_info)
            except Exception as e:
                print(f"❌ Error: {e}")
                if verbose_mode:
                    import traceback; traceback.print_exc()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
