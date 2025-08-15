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


def main() -> int:
    parser = argparse.ArgumentParser(description="TTQuery Interactive Chat")
    parser.add_argument(
        "--embeddings",
        type=str,
        default="artifacts/embeddings.jsonl",
        help="Path to embeddings file"
    )
    parser.add_argument(
        "--session",
        type=str,
        help="Specific session file for conversation history (default: auto-continue recent session)"
    )
    parser.add_argument(
        "--new-session",
        action="store_true",
        help="Force start a new session instead of continuing previous one"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Start with verbose retrieval mode enabled"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="LLM timeout in seconds"
    )
    
    args = parser.parse_args()
    
    # Check environment
    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL")
    
    if not api_key or not base_url:
        print("❌ Environment variables not set!")
        print("Please set LITELLM_API_KEY and LITELLM_BASE_URL")
        print("\nExample:")
        print("export LITELLM_API_KEY=your_key")
        print("export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/")
        return 1
    
    # Check embeddings file
    embeddings_path = os.path.abspath(args.embeddings)
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings file not found: {embeddings_path}")
        print("Run 'python initialize.py' first to prepare your knowledge base.")
        return 1
    
    # Load knowledge base
    print("🔄 Loading knowledge base...")
    try:
        items = load_corpus(embeddings_path)
        if not items:
            print("❌ No embeddings found in file")
            return 1
        print(f"✅ Loaded {len(items):,} chunks from knowledge base")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        return 1
    
    # Initialize session with auto-continue behavior
    auto_continue = not args.new_session  # Don't auto-continue if user wants new session
    session = ChatSession(args.session, auto_continue=auto_continue)
    verbose_mode = args.verbose
    
    print_welcome(session)
    
    try:
        while True:
            # Get user input
            try:
                question = input("\n💬 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
            
            if not question:
                continue
            
            # Handle commands
            if question.startswith('/'):
                cmd_parts = question[1:].split(None, 1)
                cmd = cmd_parts[0].lower()
                
                if cmd in ['quit', 'exit']:
                    print("👋 Goodbye!")
                    break
                elif cmd == 'help':
                    print_help()
                elif cmd == 'clear':
                    session.clear_history()
                    print("🗑️  Conversation history cleared")
                elif cmd == 'history':
                    if not session.history:
                        print("📝 No conversation history yet")
                    else:
                        print(f"\n📚 Conversation History ({len(session.history)} exchanges):")
                        for i, ex in enumerate(session.history[-10:], 1):  # Show last 10
                            timestamp = ex['timestamp'][:19].replace('T', ' ')
                            print(f"\n[{timestamp}] Q{i}: {ex['question']}")
                            answer = ex['answer'][:150] + '...' if len(ex['answer']) > 150 else ex['answer']
                            print(f"[{timestamp}] A{i}: {answer}")
                elif cmd == 'verbose':
                    verbose_mode = not verbose_mode
                    print(f"🔍 Verbose mode: {'ON' if verbose_mode else 'OFF'}")
                elif cmd == 'export':
                    if len(cmd_parts) > 1:
                        session.export_session(cmd_parts[1])
                    else:
                        default_file = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        session.export_session(default_file)
                elif cmd == 'stats':
                    print_stats(session, items)
                elif cmd == 'sessions':
                    list_sessions()
                elif cmd == 'new':
                    # Start a new session
                    session = start_new_session()
                    print("🔄 Switched to new session. Previous context cleared.")
                else:
                    print(f"❓ Unknown command: /{cmd}")
                    print("Type /help for available commands")
                
                continue
            
            # Process question
            print("🤔 Thinking...")
            start_time = time.time()
            
            try:
                # Get conversation context
                context = session.get_context(last_n=3)
                
                # Get answer with enhanced retrieval
                answer, sources, retrieval_info = enhanced_answer(
                    question=question,
                    embeddings_path=embeddings_path,
                    conversation_context=context,
                    verbose=verbose_mode,
                    timeout=args.timeout
                )
                
                total_time = time.time() - start_time
                
                # Display answer
                print(f"\n🤖 Assistant ({total_time:.1f}s):")
                print(answer.strip())
                print(f"\n📚 Sources:")
                print(sources)
                
                # Save to session
                session.add_exchange(question, answer, sources, retrieval_info)
                
            except Exception as e:
                print(f"❌ Error: {e}")
                if verbose_mode:
                    import traceback
                    traceback.print_exc()
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
