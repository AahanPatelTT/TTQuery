#!/usr/bin/env python3
"""
TTQuery Interactive Chat Interface

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
    python chat.py [--embeddings artifacts/embeddings.jsonl] [--verbose] [--session session.json]
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


class ChatSession:
    """Manages conversation history and context."""
    
    def __init__(self, session_file: Optional[str] = None, auto_continue: bool = True):
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
            "history": self.history
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
            
            print(f"📚 Loaded session with {len(self.history)} previous exchanges")
        except Exception as e:
            print(f"⚠️  Could not load session: {e}")
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        self.save_session()
    
    def export_session(self, filepath: str):
        """Export session to a file."""
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    "created_at": self.created_at,
                    "exported_at": datetime.now().isoformat(),
                    "total_exchanges": len(self.history),
                    "history": self.history
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
) -> Tuple[str, str, Dict]:
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

    # Build prompt with conversation context
    sources_block, cmap = format_citations(final_indices, items)
    system, user = build_prompt(question, final_indices, items)
    
    # Add conversation context if available
    if conversation_context:
        system += "\n\nFor context, here is our recent conversation:\n" + conversation_context
    # Admin/system override from GUI
    if system_override:
        system = str(system_override)
    
    if verbose:
        VerboseRetrieval.print_header("LLM GENERATION")
        print(f"System prompt: {system[:200]}{'...' if len(system) > 200 else ''}")
        print(f"User prompt: {user[:300]}{'...' if len(user) > 300 else ''}")
        print(f"Timeout: {timeout}s")
    
    # Generate answer
    start_time = time.time()
    out = call_gemini_via_litellm(system, user, timeout=timeout)
    generation_time = time.time() - start_time
    
    if verbose:
        print(f"✅ Generation completed in {generation_time:.1f}s")

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
        "sources_count": len(final_indices)
    }

    return out, sources_block, retrieval_info


def print_welcome(session: Optional[ChatSession] = None):
    """Print welcome message and instructions."""
    print("\n" + "="*80)
    print("💬 TTQuery Interactive Chat")
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


def start_new_session() -> ChatSession:
    """Start a new session, forcing creation of a new file."""
    session_dir = Path("sessions")
    session_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_file = str(session_dir / f"chat_session_{timestamp}.json")
    print(f"🆕 Started new session: {Path(session_file).name}")
    return ChatSession(session_file, auto_continue=False)


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

    # Persistent session using CLI's ChatSession
    session = ChatSession(auto_continue=True)
    session_history: List[Dict] = []  # mirrored in-memory for quick UI rendering

    # Admin-configurable RAG params (mutable via UI)
    rag_cfg = {
        "system_prompt": (
            "You are a precise engineering assistant. Use ONLY the provided context. "
            "Write a concise, coherent answer. When multiple chunks from the SAME document are provided, stitch them into a single cohesive section. "
            "Quote exact phrases for key claims where appropriate. Use inline citations like [1], [2] immediately after the claims they support. If the answer is not found, say you don't know."
        ),
        "topk": 10,
        "per_doc": 8,
        "lambda_mmr": 0.8,
        "timeout": default_timeout,
        "verbose": False,
    }

    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template(
            'index.html',
            system_prompt=rag_cfg["system_prompt"],
            topk=rag_cfg["topk"],
            per_doc=rag_cfg["per_doc"],
            lambda_mmr=rag_cfg["lambda_mmr"],
            timeout=rag_cfg["timeout"],
            embeddings_path=os.path.abspath(embeddings_path),
        )

    @app.get("/api/config")
    def get_config():
        return jsonify({**rag_cfg})

    @app.post("/api/config")
    def set_config():
        data = request.get_json(force=True)
        for k in ["system_prompt","topk","per_doc","lambda_mmr","timeout","verbose"]:
            if k in data and data[k] is not None:
                rag_cfg[k] = data[k]
        return jsonify({"ok": True})

    @app.get("/api/history")
    def get_history():
        # Load from persistent session on first call
        nonlocal session_history
        if not session_history and session.history:
            for ex in session.history:
                # Convert stored plain text to HTML for UI rendering
                a_html = markdown.markdown(str(ex.get("answer", "")), extensions=["tables", "fenced_code"])
                s_html = markdown.markdown(str(ex.get("sources", "")), extensions=["tables", "fenced_code"])
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
        session = start_new_session()
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
        session = ChatSession(str(target), auto_continue=False)
        session_history = []
        return jsonify({"ok": True, "session_file": session.session_file})

    @app.post("/api/ask")
    def ask():
        data = request.get_json(force=True)
        question = (data.get("question") or "").strip()
        if not question:
            return jsonify({"answer":"", "sources":""})

        # Conversation context for parity with CLI
        conv_ctx = session.get_context(last_n=3)
        try:
            answer_text, sources_block, retrieval_info = enhanced_answer(
                question=question,
                embeddings_path=embeddings_path,
                conversation_context=conv_ctx,
                verbose=bool(rag_cfg.get("verbose", False)),
                per_doc=int(rag_cfg.get("per_doc", 8)),
                final_k=int(rag_cfg.get("topk", 10)),
                lambda_mmr=float(rag_cfg.get("lambda_mmr", 0.8)),
                timeout=int(rag_cfg.get("timeout", default_timeout)),
                system_override=rag_cfg.get("system_prompt") if rag_cfg.get("system_prompt") else None,
            )
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

        # Add to persistent session
        session.add_exchange(question, answer_text, sources_block, retrieval_info)

        # Keep in-memory history for UI
        a_html = markdown.markdown(str(answer_text), extensions=["tables", "fenced_code"])
        s_html = markdown.markdown(str(sources_block), extensions=["tables", "fenced_code"])
        session_history.append({"q": question, "a_html": a_html, "sources_html": s_html, "ts": time.time()})

        resp = {"answer": answer_text, "sources": sources_block, "answer_html": a_html, "sources_html": s_html}
        if rag_cfg.get("verbose"):
            resp["retrieval_info"] = retrieval_info
        return jsonify(resp)

    @app.post("/api/clear")
    def clear():
        session_history.clear()
        session.clear_history()
        return jsonify({"ok": True})

    @app.get("/api/export")
    def export():
        from flask import Response
        data = json.dumps({"history": session_history}, indent=2)
        return Response(data, mimetype='application/json')

    app.run(host="127.0.0.1", port=7860, debug=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="TTQuery Interactive Chat")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="artifacts/embeddings.jsonl",
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
                answer, sources, retrieval_info = enhanced_answer(
                    question=question, embeddings_path=embeddings_path, conversation_context=context,
                    verbose=verbose_mode, timeout=args.timeout)
                total_time = time.time() - start_time
                print(f"\n🤖 Assistant ({total_time:.1f}s):\n{answer.strip()}\n\n📚 Sources:\n{sources}")
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
