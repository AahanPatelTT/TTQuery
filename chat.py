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
    python chat.py [--embeddings path_or_kb_name] [--verbose] [--session session.json]
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


def print_verbose_step(step: str, data: Dict = None):
    """Print verbose retrieval step information."""
    print(f"\n{'='*60}")
    print(f"🔍 {step}")
    print('='*60)
    
    if data:
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                print(f"📊 {key}: {len(value)} items")
                for i, item in enumerate(value[:5]):  # Show top 5
                    if isinstance(item, dict):
                        source = Path(item.get('source_path', '')).name
                        content = str(item.get('full_text', item.get('summary_text', '')))[:100]
                        print(f"  {i+1:2d}. {source} - {content}...")
                    else:
                        print(f"  {i+1:2d}. {item}")
                if len(value) > 5:
                    print(f"  ... and {len(value) - 5} more")
            else:
                print(f"📊 {key}: {value}")




def enhanced_answer_multi_kb(
    question: str,
    selected_knowledge_bases: List[str],
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
    images_enabled: bool = True,
    max_images: int = 2,
    artifacts_dir: str = "artifacts"
) -> Tuple[str, str, Dict, List[str]]:
    """Enhanced answer function that searches across multiple knowledge bases."""
    from pipeline.query import list_available_knowledge_bases
    
    # Get all available knowledge bases
    available_kbs = list_available_knowledge_bases(artifacts_dir)
    
    # If no specific KBs selected, use all available
    if not selected_knowledge_bases:
        selected_knowledge_bases = [kb['name'] for kb in available_kbs]
    
    # Filter to only valid knowledge bases
    valid_kbs = []
    for kb_name in selected_knowledge_bases:
        for kb in available_kbs:
            if kb['name'] == kb_name:
                valid_kbs.append(kb)
                break
    
    if not valid_kbs:
        raise ValueError("No valid knowledge bases found")
    
    # If only one KB selected, use the original function
    if len(valid_kbs) == 1:
        kb = valid_kbs[0]
        return enhanced_answer(
            question=question,
            embeddings_path=kb['embeddings_path'],
            conversation_context=conversation_context,
            verbose=verbose,
            k_dense_sum=k_dense_sum,
            k_dense_full=k_dense_full,
            k_sparse=k_sparse,
            per_doc=per_doc,
            final_k=final_k,
            lambda_mmr=lambda_mmr,
            timeout=timeout,
            system_override=system_override,
            chunked_path=kb.get('chunked_path'),
            images_enabled=images_enabled,
            max_images=max_images
        )
    
    # Multi-KB search: concatenate all knowledge bases into unified corpus
    if verbose:
        print(f"🔗 CONCATENATING {len(valid_kbs)} KNOWLEDGE BASES")
        for kb in valid_kbs:
            print(f"   - {kb['display_name']} ({kb['name']})")
    
    # Load and concatenate all knowledge bases
    combined_items = []
    all_image_paths = []
    
    for kb in valid_kbs:
        from pipeline.query import load_corpus
        kb_items = load_corpus(kb['embeddings_path'])
        
        # Add KB metadata to each item
        for item in kb_items:
            if 'metadata' not in item:
                item['metadata'] = {}
            item['metadata']['source_kb'] = kb['name']
            item['metadata']['source_kb_display'] = kb['display_name']
            combined_items.append(item)
    
    if not combined_items:
        raise ValueError("No items found in any knowledge base")
    
    # Perform unified search on the concatenated corpus
    from pipeline.query import build_indices, load_query_encoder, encode_query
    from pipeline.query import dense_search, sparse_search, rrf, per_doc_cap, rerank, mmr_select
    from pipeline.query import format_citations, build_prompt, call_gemini_via_litellm
    
    dense_index_s, dense_index_f, E_sum, E_full, sparse_index = build_indices(combined_items)
    model_name, query_encoder = load_query_encoder("BAAI/bge-large-en-v1.5")
    
    # Encode query
    query_text = f"{conversation_context} {question}" if conversation_context else question
    query_vec = encode_query(query_text, query_encoder)
    
    # Search the concatenated corpus
    dense_results_s = dense_search(query_vec, dense_index_s, k=k_dense_sum)
    dense_results_f = dense_search(query_vec, dense_index_f, k=k_dense_full)
    sparse_results = sparse_search(question, sparse_index, k=k_sparse)
    
    # Fusion and ranking
    fused_indices = rrf([dense_results_s, dense_results_f, sparse_results], weights=[0.9, 1.2, 0.8])
    capped_indices = per_doc_cap(fused_indices, combined_items, per_doc)
    
    # Rerank and select final contexts
    if len(capped_indices) > final_k:
        reranked_indices = rerank(question, capped_indices, combined_items, topn=40)
        final_indices = mmr_select(query_vec, reranked_indices, E_full, lambda_mmr, final_k)
    else:
        final_indices = capped_indices[:final_k]
    
    # Extract final contexts and build citations
    final_contexts = [combined_items[i] for i in final_indices]
    
    # Group sources by KB for better attribution
    sources_by_kb = {}
    for ctx in final_contexts:
        kb_name = ctx['metadata'].get('source_kb_display', 'Unknown')
        if kb_name not in sources_by_kb:
            sources_by_kb[kb_name] = []
        sources_by_kb[kb_name].append(ctx)
    
    # Format citations with KB attribution
    citations_parts = []
    for kb_name, kb_contexts in sources_by_kb.items():
        kb_indices = [final_indices[i] for i, ctx in enumerate(final_contexts) 
                     if ctx['metadata'].get('source_kb_display') == kb_name]
        kb_citations, _ = format_citations(kb_indices, combined_items)
        if kb_citations.strip():
            citations_parts.append(f"**From {kb_name}:**\n{kb_citations}")
    
    combined_sources_block = "\n\n---\n\n".join(citations_parts)
    
    # Extract images from contexts
    if images_enabled:
        for ctx in final_contexts:
            img_paths = ctx.get('metadata', {}).get('image_paths', [])
            if isinstance(img_paths, list):
                all_image_paths.extend(img_paths[:max_images//len(final_contexts)+1])
    
    # Limit total images
    if len(all_image_paths) > max_images:
        all_image_paths = all_image_paths[:max_images]
    
    # Build prompt and generate answer
    context_text = "\n\n".join([
        f"Source {i+1}: {ctx.get('full_text', ctx.get('content', ''))}" 
        for i, ctx in enumerate(final_contexts)
    ])
    
    kb_names = [kb['display_name'] for kb in valid_kbs]
    enhanced_question = f"Based on information from multiple knowledge bases ({', '.join(kb_names)}): {question}"
    
    user_prompt = build_prompt(enhanced_question, context_text, conversation_context)
    
    # Generate answer
    system_prompt = "You are an engineering assistant. Keep your responses relevant and based on the context. Provide as much detail as possible but keep it concise. When asked for tables - reconstruct tables based on retrieved context. IMPORTANT: Never include any citation numbers, brackets like [1], [2], or references to 'Context Document X' in your response. Write naturally without any citation markers - the system will add citations separately. When interpreting OCR text from technical diagrams, be careful about common OCR errors: '2x' may appear as part of adjacent text, 'I2C' may appear as '12C', 'x4' formatting may be inconsistent. Always double-check technical specifications and component counts against the visual context when available."
    
    if system_override and system_override.strip():
        system_prompt = system_override
    
    answer_text = call_gemini_via_litellm(system_prompt, user_prompt, timeout=timeout)
    
    # Build retrieval info
    retrieval_info_combined = {
        'knowledge_bases_used': [kb['display_name'] for kb in valid_kbs],
        'total_items_searched': len(combined_items),
        'final_contexts': len(final_contexts),
        'sources_by_kb': {kb: len(contexts) for kb, contexts in sources_by_kb.items()},
        'concatenated_search': True
    }
    
    return answer_text, combined_sources_block, retrieval_info_combined, all_image_paths


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
        raise RuntimeError("No embedding records found. Run initialize_fast.py first.")

    idx_sum, idx_full, E_sum, E_full, bm25 = build_indices(items)
    model_name, q_encoder = load_query_encoder()
    
    # Encode query
    qv = encode_query(question, q_encoder)

    # Dense search
    c_sum = dense_search(qv, idx_sum, k_dense_sum)
    c_full = dense_search(qv, idx_full, k_dense_full)
    
    if verbose:
        print_verbose_step("DENSE RETRIEVAL", {
            "summary_results": [items[i] for i in c_sum[:10]],
            "full_results": [items[i] for i in c_full[:10]]
        })

    # Sparse search
    c_sparse = sparse_search(question, bm25, k_sparse)
    
    if verbose:
        print_verbose_step("SPARSE RETRIEVAL (BM25)", {
            "sparse_results": [items[i] for i in c_sparse[:10]]
        })

    # Fusion
    fused = rrf([c_sum, c_full, c_sparse], weights=[0.9, 1.2, 0.8], rrf_k=60, base=60)
    fused = per_doc_cap(fused, items, per_doc)
    
    if verbose:
        print_verbose_step("RECIPROCAL RANK FUSION (RRF)", {
            "fused_results": [items[i] for i in fused[:15]]
        })

    # Rerank
    reranked = rerank(question, fused[:100], items, topn=20)
    
    if verbose:
        print_verbose_step("RERANKING", {
            "reranked_results": [items[i] for i in reranked[:15]]
        })

    # MMR diversification
    final_indices = mmr_select(qv[0], reranked, E_full, lambda_mmr, min(final_k, len(reranked) or 0))
    
    if verbose:
        print_verbose_step("FINAL CONTEXT (MMR DIVERSIFIED)", {
            "final_contexts": [items[i] for i in final_indices]
        })

    # Extract relevant images
    image_paths = []
    image_descriptions = []
    if chunked_path and images_enabled and max_images > 0:
        from pipeline.query import extract_relevant_images
        all_image_paths = extract_relevant_images(question, final_indices, items)
        image_paths = all_image_paths[:max_images]
        
        # Get image descriptions for LLM context
        for img_path in image_paths:
            for item in items:
                metadata = item.get("metadata", {})
                if (metadata.get("content_format") == "extracted_image" and 
                    metadata.get("image_path") == img_path):
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
                img_name = os.path.basename(img_path)
                image_descriptions.append(f"Technical diagram: {img_name}")

    # Build prompt with conversation context
    sources_block, cmap = format_citations(final_indices, items)
    system, user = build_prompt(question, final_indices, items)
    
    # Add image descriptions to the prompt if available
    if image_descriptions:
        image_context = "\n\nRelevant visual content available:\n" + "\n".join(
            f"- {desc}" for desc in image_descriptions
        )
        user += image_context
    
    # Add conversation context if available
    if conversation_context:
        system += "\n\nFor context, here is our recent conversation:\n" + conversation_context
    
    # System override from GUI
    if system_override and system_override.strip():
        system = str(system_override)
    
    if verbose:
        print_verbose_step("LLM GENERATION", {
            "system_prompt": system[:500] + "..." if len(system) > 500 else system,
            "user_prompt": user[:300] + "..." if len(user) > 300 else user,
            "timeout": f"{timeout}s",
            "images_found": len(image_paths)
        })
    
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
        "sources_count": len(final_indices),
        "images_found": len(image_paths)
    }

    return out, sources_block, retrieval_info, image_paths


def print_welcome(session: Optional[ChatSession] = None, kb_name: Optional[str] = None):
    """Print welcome message and instructions."""
    print("\n" + "="*80)
    print("💬 Synapse Interactive Chat")
    print("="*80)
    print("Welcome! Ask questions about your knowledge base.")
    
    if kb_name:
        print(f"🔍 Knowledge Base: {kb_name}")
    
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
    print("  /kb           - List available knowledge bases")
    print("  /switch-kb    - Switch to a different knowledge base")
    print("\n💡 Tips:")
    print("  • Ask follow-up questions - I remember our conversation!")
    print("  • Use /verbose to see detailed retrieval steps")
    print("  • Questions can reference previous answers")
    print("  • Sources are provided for fact verification")
    print("  • Sessions auto-save and auto-continue within 24 hours")
    print("  • Use /kb to see available knowledge bases")
    print("  • Use /switch-kb to change knowledge bases mid-conversation")


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


def run_gui(embeddings_path: str, default_timeout: int = 60, selected_kb_name: Optional[str] = None) -> int:
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
        
        # Get selected knowledge bases from request
        selected_kbs = data.get("selected_knowledge_bases", [])

        # Conversation context for parity with CLI
        conv_ctx = session.get_context(last_n=3)
        try:
            # Get current session config
            current_config = session.get_config()
            
            # Use multi-KB search if knowledge bases are selected, otherwise use single KB
            if selected_kbs:
                artifacts_dir = os.path.dirname(embeddings_path)
                answer_text, sources_block, retrieval_info, image_paths = enhanced_answer_multi_kb(
                    question=question,
                    selected_knowledge_bases=selected_kbs,
                    conversation_context=conv_ctx,
                    verbose=bool(current_config.get("verbose", False)),
                    per_doc=int(current_config.get("per_doc", 8)),
                    final_k=int(current_config.get("topk", 10)),
                    lambda_mmr=float(current_config.get("lambda_mmr", 0.8)),
                    timeout=int(current_config.get("timeout", default_timeout)),
                    system_override=current_config.get("system_prompt"),
                    images_enabled=images_enabled,
                    max_images=max_images,
                    artifacts_dir=artifacts_dir
                )
            else:
                # Fallback to single KB search
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
        processed_answer = str(answer_text)
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

    @app.get("/api/knowledge-bases")
    def list_knowledge_bases():
        """List available knowledge bases."""
        try:
            from pipeline.query import list_available_knowledge_bases
            artifacts_dir = os.path.dirname(embeddings_path)
            knowledge_bases = list_available_knowledge_bases(artifacts_dir)
            
            # Mark the current knowledge base
            current_kb = None
            for kb in knowledge_bases:
                if kb['embeddings_path'] == embeddings_path:
                    current_kb = kb['name']
                    break
            
            return jsonify({
                "knowledge_bases": knowledge_bases,
                "current_kb": current_kb,
                "current_kb_name": selected_kb_name
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.post("/api/switch-kb")
    def switch_knowledge_base():
        """Switch to a different knowledge base."""
        nonlocal embeddings_path, chunked_path, selected_kb_name, items
        
        try:
            data = request.get_json(force=True)
            kb_name = str(data.get("kb_name", "")).strip()
            
            if not kb_name:
                return jsonify({"ok": False, "error": "Knowledge base name required"}), 400
            
            # Find the knowledge base
            from pipeline.query import list_available_knowledge_bases
            artifacts_dir = os.path.dirname(embeddings_path)
            knowledge_bases = list_available_knowledge_bases(artifacts_dir)
            
            selected_kb = None
            for kb in knowledge_bases:
                if kb['name'] == kb_name:
                    selected_kb = kb
                    break
            
            if not selected_kb:
                return jsonify({"ok": False, "error": f"Knowledge base '{kb_name}' not found"}), 404
            
            # Switch to the new knowledge base
            embeddings_path = selected_kb['embeddings_path']
            chunked_path = selected_kb['chunked_path']
            selected_kb_name = selected_kb['display_name']
            
            # Reload the corpus
            items = load_corpus(embeddings_path)
            if not items:
                return jsonify({"ok": False, "error": "Failed to load new knowledge base"}), 500
            
            return jsonify({
                "ok": True,
                "message": f"Switched to knowledge base: {selected_kb_name}",
                "kb_name": selected_kb_name,
                "chunks_loaded": len(items)
            })
            
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.post("/api/upload")
    def upload_documents():
        """Upload documents to a specific knowledge base folder with duplicate detection and auto-processing."""
        try:
            from flask import request
            from werkzeug.utils import secure_filename
            import threading
            
            if 'files' not in request.files:
                return jsonify({'error': 'No files provided'}), 400
            
            files = request.files.getlist('files')
            folder = request.form.get('folder', 'uploads')
            
            if not files or all(file.filename == '' for file in files):
                return jsonify({'error': 'No files selected'}), 400
            
            # Sanitize folder name
            folder = secure_filename(folder) or 'uploads'
            folder_path = os.path.join('Data', folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Initialize database for duplicate checking
            db = None
            try:
                from pipeline.database import SynapseDB
                db = SynapseDB()
            except ImportError:
                logging.warning("Database not available for duplicate detection")
            
            # Allowed file extensions
            allowed_extensions = {
                'pdf', 'pptx', 'docx', 'txt', 'md', 'csv',
                'png', 'jpg', 'jpeg', 'tiff', 'bmp'
            }
            
            def allowed_file(filename):
                return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions
            
            uploaded_files = []
            failed_files = []
            duplicate_files = []
            
            for file in files:
                if file and file.filename and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(folder_path, filename)
                    
                    try:
                        # Check for duplicates before saving
                        duplicate_by_name = None
                        duplicate_by_content = None
                        
                        if db:
                            # Check if file with same name already exists
                            duplicate_by_name = db.find_duplicate_by_name(file_path, folder)
                        
                        # If file already exists, check if content is different
                        if os.path.exists(file_path):
                            # Save to temporary location to check content
                            temp_path = file_path + '.tmp'
                            file.save(temp_path)
                            
                            if db:
                                duplicate_by_content = db.find_duplicate_by_content(temp_path, folder)
                            
                            # Check if content is identical to existing file
                            if os.path.exists(file_path):
                                import filecmp
                                if filecmp.cmp(temp_path, file_path, shallow=False):
                                    # Identical content - this is a duplicate
                                    os.unlink(temp_path)
                                    duplicate_files.append({
                                        'filename': filename,
                                        'reason': 'Identical file already exists',
                                        'existing_path': file_path
                                    })
                                    continue
                                else:
                                    # Different content - replace the file
                                    os.replace(temp_path, file_path)
                            else:
                                # File doesn't exist, move temp to final location
                                os.rename(temp_path, file_path)
                        else:
                            # New file - save normally
                            file.save(file_path)
                        
                        # Check for content duplicates in database
                        if db and duplicate_by_content:
                            duplicate_files.append({
                                'filename': filename,
                                'reason': 'Same content exists in different file',
                                'existing_path': duplicate_by_content
                            })
                            os.unlink(file_path)  # Remove the duplicate
                            continue
                        
                        uploaded_files.append({
                            'filename': filename,
                            'path': file_path,
                            'folder': folder,
                            'size': os.path.getsize(file_path)
                        })
                        logging.info(f"Uploaded file: {file_path}")
                        
                    except Exception as e:
                        failed_files.append({
                            'filename': file.filename,
                            'error': str(e)
                        })
                        logging.error(f"Failed to upload {file.filename}: {e}")
                else:
                    failed_files.append({
                        'filename': file.filename or 'Unknown',
                        'error': 'Invalid file type or empty file'
                    })
            
            # Trigger auto-reinitialization if files were uploaded
            processing_started = False
            if uploaded_files:
                try:
                    # Start background processing
                    def process_uploaded_files():
                        try:
                            logging.info(f"Starting auto-reinitialization for folder: {folder}")
                            
                            # Import and run fast initialization for this folder
                            import subprocess
                            result = subprocess.run([
                                'python3', 'initialize_fast.py', 
                                '--folder', folder,
                                '--skip-images'  # Fast processing
                            ], capture_output=True, text=True, cwd=os.getcwd())
                            
                            if result.returncode == 0:
                                logging.info(f"Auto-reinitialization completed for folder: {folder}")
                            else:
                                logging.error(f"Auto-reinitialization failed for {folder}: {result.stderr}")
                                
                        except Exception as e:
                            logging.error(f"Background processing error: {e}")
                    
                    # Start processing in background thread
                    processing_thread = threading.Thread(target=process_uploaded_files, daemon=True)
                    processing_thread.start()
                    processing_started = True
                    
                except Exception as e:
                    logging.warning(f"Failed to start background processing: {e}")
            
            # Build response message
            message_parts = []
            if uploaded_files:
                message_parts.append(f"Uploaded {len(uploaded_files)} files to '{folder}'")
            if duplicate_files:
                message_parts.append(f"Skipped {len(duplicate_files)} duplicates")
            if failed_files:
                message_parts.append(f"{len(failed_files)} files failed")
            
            message = ", ".join(message_parts) if message_parts else "No files processed"
            
            if processing_started:
                message += ". Processing started in background."
            
            return jsonify({
                'success': True,
                'message': message,
                'uploaded_files': uploaded_files,
                'failed_files': failed_files,
                'duplicate_files': duplicate_files,
                'folder': folder,
                'auto_processing': processing_started
            })
        
        except Exception as e:
            logging.error(f"Upload error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.get("/api/upload/status")
    def get_upload_status():
        """Get upload and processing status."""
        try:
            # Try to get status from the fast database system
            try:
                from pipeline.database import SynapseDB
                from pipeline.fast_embed import FastEmbeddingService
                
                db = SynapseDB()
                embedding_service = FastEmbeddingService()
                
                folders = db.get_all_folders()
                folder_status = []
                
                for folder_key in folders:
                    stats = db.get_folder_stats(folder_key)
                    embed_status = embedding_service.get_embedding_status(folder_key)
                    
                    folder_status.append({
                        'name': folder_key,
                        'display_name': folder_key.replace('_', ' ').replace('hash_', '#'),
                        'total_docs': stats['completed_docs'],
                        'pending_docs': stats['pending_docs'],
                        'failed_docs': stats['failed_docs'],
                        'total_chunks': stats['total_chunks'],
                        'embedded_chunks': stats['embedded_chunks'],
                        'pending_embeddings': stats['pending_embeddings'],
                        'completion_rate': embed_status['completion_rate']
                    })
                
                return jsonify({
                    'folders': folder_status,
                    'total_folders': len(folders),
                    'fast_system_available': True
                })
                
            except ImportError:
                # Fallback to basic folder listing if database system not available
                data_dir = Path('Data')
                folders = []
                
                if data_dir.exists():
                    for folder_path in data_dir.iterdir():
                        if folder_path.is_dir() and not folder_path.name.startswith('.'):
                            file_count = len([f for f in folder_path.rglob('*') if f.is_file() and not f.name.startswith('.')])
                            folders.append({
                                'name': folder_path.name,
                                'display_name': folder_path.name,
                                'total_docs': file_count,
                                'pending_docs': 0,
                                'failed_docs': 0,
                                'total_chunks': 0,
                                'embedded_chunks': 0,
                                'pending_embeddings': 0,
                                'completion_rate': 0
                            })
                
                return jsonify({
                    'folders': folders,
                    'total_folders': len(folders),
                    'fast_system_available': False
                })
        
        except Exception as e:
            logging.error(f"Status error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.get("/api/upload/progress/<folder_key>")
    def get_upload_progress(folder_key):
        """Get detailed upload and processing progress for a specific folder."""
        try:
            from pipeline.database import SynapseDB
            from pipeline.fast_embed import FastEmbeddingService
            
            db = SynapseDB()
            embedding_service = FastEmbeddingService()
            
            # Get folder statistics
            stats = db.get_folder_stats(folder_key)
            embed_status = embedding_service.get_embedding_status(folder_key)
            
            # Calculate progress percentages
            parsing_progress = 0
            if stats['completed_docs'] + stats['pending_docs'] > 0:
                parsing_progress = (stats['completed_docs'] / (stats['completed_docs'] + stats['pending_docs'])) * 100
            
            embedding_progress = embed_status['completion_rate']
            
            # Overall progress (weighted average)
            overall_progress = (parsing_progress * 0.3) + (embedding_progress * 0.7)
            
            return jsonify({
                'folder': folder_key,
                'parsing': {
                    'completed': stats['completed_docs'],
                    'pending': stats['pending_docs'],
                    'failed': stats['failed_docs'],
                    'progress': parsing_progress
                },
                'embedding': {
                    'completed': stats['embedded_chunks'],
                    'pending': stats['pending_embeddings'],
                    'total': stats['total_chunks'],
                    'progress': embedding_progress
                },
                'overall_progress': overall_progress,
                'status': 'processing' if stats['pending_docs'] > 0 or stats['pending_embeddings'] > 0 else 'complete'
            })
        
        except ImportError:
            return jsonify({'error': 'Fast processing system not available'}), 500
        except Exception as e:
            logging.error(f"Progress check error: {e}")
            return jsonify({'error': str(e)}), 500

    @app.post("/api/process/<folder_key>")
    def trigger_processing(folder_key):
        """Manually trigger processing for a specific folder."""
        try:
            from pipeline.database import SynapseDB
            from pipeline.fast_embed import FastEmbeddingService
            
            db = SynapseDB()
            embedding_service = FastEmbeddingService()
            
            # Process the folder immediately
            processed = embedding_service.process_folder_immediately(folder_key)
            
            return jsonify({
                'success': True,
                'message': f'Processed {processed} chunks for folder {folder_key}',
                'processed_count': processed
            })
        
        except ImportError:
            return jsonify({
                'success': False, 
                'error': 'Fast processing system not available. Run: pip install sqlite3'
            }), 500
        except Exception as e:
            logging.error(f"Processing trigger error: {e}")
            return jsonify({'error': str(e)}), 500

    app.run(host="127.0.0.1", port=7860, debug=False)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Synapse Interactive Chat")
    parser.add_argument(
        "--embeddings",
        type=str,
        default=None,
        help="Path to embeddings file or knowledge base name (optional - will auto-detect available KBs)"
    )
    parser.add_argument("--session", type=str, help="Specific session file for conversation history")
    parser.add_argument("--kb", type=str, help="Use specific knowledge base by name")
    parser.add_argument("--select-kb", action="store_true", help="Interactively select knowledge base")
    parser.add_argument("--list-kb", action="store_true", help="List available knowledge bases and exit")
    parser.add_argument("--new-session", action="store_true", help="Force start a new session")
    parser.add_argument("--verbose", action="store_true", help="Start with verbose retrieval mode enabled")
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout in seconds")
    parser.add_argument("--test_gui", action="store_true", help="Launch local test GUI instead of CLI")

    args = parser.parse_args()

    api_key = os.getenv("LITELLM_API_KEY"); base_url = os.getenv("LITELLM_BASE_URL")
    if not api_key or not base_url:
        print("❌ Environment variables not set!\nSet LITELLM_API_KEY and LITELLM_BASE_URL")
        return 1

    # Handle knowledge base selection (import here to avoid circular imports)
    from pipeline.query import list_available_knowledge_bases, interactive_knowledge_base_selector
    
    # Handle knowledge base listing
    if args.list_kb:
        if args.embeddings is None:
            artifacts_dir = "artifacts"
        else:
            artifacts_dir = os.path.dirname(os.path.abspath(args.embeddings))
        knowledge_bases = list_available_knowledge_bases(artifacts_dir)
        
        if not knowledge_bases:
            print("No knowledge bases found. Run folder-based initialization first:")
            print("  python initialize_fast.py")
            return 1
        
        print("Available Knowledge Bases:")
        print("=" * 50)
        for kb in knowledge_bases:
            size_mb = kb['file_size'] / (1024 * 1024) if kb['file_size'] > 0 else 0
            chunk_info = f"{kb['chunk_count']:,} chunks" if kb['chunk_count'] > 0 else "unknown chunks"
            size_info = f"{size_mb:.1f} MB" if size_mb > 0 else "unknown size"
            
            print(f"📚 {kb['display_name']} ({kb['name']})")
            print(f"   📊 {chunk_info}, {size_info}")
            print(f"   📄 Embeddings: {os.path.basename(kb['embeddings_path'])}")
            if kb['chunked_path']:
                print(f"   🧩 Chunks: {os.path.basename(kb['chunked_path'])}")
            print()
        
        print("Usage:")
        print(f"  python chat.py --kb \"{knowledge_bases[0]['name']}\"")
        print(f"  python chat.py --select-kb")
        return 0
    
    # Handle knowledge base selection
    embeddings_path = args.embeddings
    chunked_path = None
    selected_kb_name = None
    
    # If no embeddings path provided, try to auto-detect available knowledge bases
    if embeddings_path is None:
        # Try to find artifacts directory
        artifacts_dir = "artifacts"
        if not os.path.exists(artifacts_dir):
            print("❌ No embeddings path provided and no artifacts directory found.")
            print("Please specify --embeddings or run initialization first:")
            print("  python initialize_fast.py")
            return 1
        
        knowledge_bases = list_available_knowledge_bases(artifacts_dir)
        if not knowledge_bases:
            print("❌ No knowledge bases found. Run initialization first:")
            print("  python initialize_fast.py")
            return 1
        
        # Use the first available knowledge base as default
        selected_kb = knowledge_bases[0]
        embeddings_path = selected_kb['embeddings_path']
        chunked_path = selected_kb['chunked_path']
        selected_kb_name = selected_kb['display_name']
        print(f"🔍 Auto-selected knowledge base: {selected_kb_name}")
        print(f"📄 Embeddings: {os.path.basename(embeddings_path)}")
        if chunked_path:
            print(f"🧩 Chunks: {os.path.basename(chunked_path)}")
        print()
    
    # Interactive knowledge base selection
    if args.select_kb:
        if embeddings_path is None:
            artifacts_dir = "artifacts"
        else:
            artifacts_dir = os.path.dirname(os.path.abspath(embeddings_path))
        selected_kb = interactive_knowledge_base_selector(artifacts_dir)
        
        if selected_kb is None:
            print("Using default knowledge base...")
        else:
            args.kb = selected_kb['name']
    
    # Handle specific knowledge base selection
    if args.kb:
        if embeddings_path is None:
            artifacts_dir = "artifacts"
        else:
            artifacts_dir = os.path.dirname(os.path.abspath(embeddings_path))
        knowledge_bases = list_available_knowledge_bases(artifacts_dir)
        
        selected_kb = None
        for kb in knowledge_bases:
            if kb['name'] == args.kb or kb['display_name'].lower() == args.kb.lower():
                selected_kb = kb
                break
        
        if not selected_kb:
            print(f"Knowledge base '{args.kb}' not found. Available options:")
            for kb in knowledge_bases:
                print(f"  - {kb['name']} ({kb['display_name']})")
            return 1
        
        embeddings_path = selected_kb['embeddings_path']
        chunked_path = selected_kb['chunked_path']
        selected_kb_name = selected_kb['display_name']
        print(f"🔍 Using knowledge base: {selected_kb_name}")
        print(f"📄 Embeddings: {os.path.basename(embeddings_path)}")
        if chunked_path:
            print(f"🧩 Chunks: {os.path.basename(chunked_path)}")
        print()

    # Ensure embeddings_path is not None before calling abspath
    if embeddings_path is None:
        print("❌ No embeddings path available. Please specify --embeddings or run initialization first.")
        return 1
    
    embeddings_path = os.path.abspath(embeddings_path)
    # For GUI mode, check if we have any knowledge bases available
    if args.test_gui:
        artifacts_dir = os.path.dirname(embeddings_path)
        try:
            from pipeline.query import list_available_knowledge_bases
            available_kbs = list_available_knowledge_bases(artifacts_dir)
            if not available_kbs:
                print(f"❌ No knowledge bases found in {artifacts_dir}")
                print("Run 'python initialize_fast.py' first to create knowledge bases.")
                return 1
            
            # Use the first available KB as default, or the specified one
            default_kb = None
            if selected_kb_name:
                for kb in available_kbs:
                    if kb['name'] == selected_kb_name:
                        default_kb = kb
                        break
            if not default_kb:
                default_kb = available_kbs[0]  # Use first available
            
            print(f"🚀 Starting GUI with default knowledge base: {default_kb['display_name']}")
            return run_gui(default_kb['embeddings_path'], default_timeout=int(args.timeout), selected_kb_name=default_kb['name'])
        except Exception as e:
            print(f"❌ Failed to load knowledge bases: {e}")
            return 1

    # For CLI mode, check the specific embeddings file
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings file not found: {embeddings_path}\nRun 'python initialize_fast.py' first.")
        return 1

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
    print_welcome(session, selected_kb_name)

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
                elif cmd=='kb':
                    # List available knowledge bases
                    artifacts_dir = os.path.dirname(embeddings_path)
                    knowledge_bases = list_available_knowledge_bases(artifacts_dir)
                    
                    if not knowledge_bases:
                        print("❌ No knowledge bases found. Run initialize_fast.py first.")
                    else:
                        print(f"\n📚 Available Knowledge Bases ({len(knowledge_bases)} total):")
                        current_kb = None
                        for kb in knowledge_bases:
                            if kb['embeddings_path'] == embeddings_path:
                                current_kb = kb
                                break
                        
                        for i, kb in enumerate(knowledge_bases, 1):
                            is_current = "🔍 " if kb == current_kb else "   "
                            size_mb = kb['file_size'] / (1024 * 1024) if kb['file_size'] > 0 else 0
                            chunk_info = f"{kb['chunk_count']:,} chunks" if kb['chunk_count'] > 0 else "unknown chunks"
                            size_info = f"{size_mb:.1f} MB" if size_mb > 0 else "unknown size"
                            
                            print(f"{is_current}{i:2d}. 📁 {kb['display_name']}")
                            print(f"       📊 {chunk_info}, {size_info}")
                        print("\n💡 Use /switch-kb to change knowledge base")
                        
                elif cmd=='switch-kb':
                    # Interactive knowledge base switching
                    artifacts_dir = os.path.dirname(embeddings_path)
                    selected_kb = interactive_knowledge_base_selector(artifacts_dir)
                    
                    if selected_kb is not None:
                        # Update the embeddings path and reload
                        embeddings_path = selected_kb['embeddings_path']
                        chunked_path = selected_kb['chunked_path']
                        selected_kb_name = selected_kb['display_name']
                        
                        print(f"🔄 Loading new knowledge base: {selected_kb_name}")
                        try:
                            items = load_corpus(embeddings_path)
                            if not items:
                                print("❌ Failed to load new knowledge base")
                            else:
                                print(f"✅ Loaded {len(items):,} chunks from {selected_kb_name}")
                                print("🔄 Knowledge base switched successfully!")
                        except Exception as e:
                            print(f"❌ Failed to load new knowledge base: {e}")
                    else:
                        print("Knowledge base not changed.")
                        
                else:
                    print(f"❓ Unknown command: /{cmd}\nType /help for available commands")
                continue
            # Process question
            print("🤔 Thinking..."); start_time = time.time()
            try:
                context = session.get_context(last_n=3)
                answer, sources, retrieval_info, image_paths = enhanced_answer(
                    question=question, embeddings_path=embeddings_path, conversation_context=context,
                    verbose=verbose_mode, timeout=args.timeout, chunked_path=chunked_path,
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
