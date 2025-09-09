#!/usr/bin/env python3
"""
Fast incremental embedding generator for real-time document updates.

This module provides background embedding processing for chunks that have been
parsed but not yet embedded, enabling near real-time updates to the knowledge base.
"""

import asyncio
import threading
import time
import logging
from typing import List, Dict, Optional, Callable
import numpy as np
from pathlib import Path

try:
    from .database import SynapseDB
    from .embed import (
        LocalEmbedder, OpenAIEmbedder, ColBERTEmbedder, BertEmbedder,
        build_page_index, build_full_text, heuristic_summary,
        _infer_doc_prefix
    )
except ImportError:
    # Handle direct execution or different import contexts
    from database import SynapseDB
    from embed import (
        LocalEmbedder, OpenAIEmbedder, ColBERTEmbedder, BertEmbedder,
        build_page_index, build_full_text, heuristic_summary,
        _infer_doc_prefix
    )


class FastEmbeddingService:
    """Real-time embedding service for incremental updates."""
    
    def __init__(self, 
                 db_path: str = "artifacts/synapse.db",
                 provider: str = "local",
                 model_name: str = "BAAI/bge-large-en-v1.5",
                 batch_size: int = 32,
                 max_workers: int = 2):
        """Initialize the fast embedding service.
        
        Args:
            db_path: Path to the Synapse database
            provider: Embedding provider ('local', 'openai', 'colbert', 'bert')
            model_name: Model name for the embedding provider
            batch_size: Batch size for embedding generation
            max_workers: Maximum number of worker threads
        """
        self.db = SynapseDB(db_path)
        self.provider = provider
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        self.processing_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker_threads = []
        
        # Initialize embedder
        self.embedder = self._create_embedder()
        self.doc_prefix = _infer_doc_prefix(model_name)
        
        logging.info(f"FastEmbeddingService initialized with {provider} provider ({model_name})")
    
    def _create_embedder(self):
        """Create the appropriate embedder based on provider."""
        try:
            if self.provider == "local":
                return LocalEmbedder(model_name=self.model_name)
            elif self.provider == "openai":
                return OpenAIEmbedder(model_name=self.model_name)
            elif self.provider == "colbert":
                return ColBERTEmbedder(model_name=self.model_name)
            elif self.provider == "bert":
                return BertEmbedder(model_name=self.model_name)
            else:
                logging.warning(f"Unknown provider {self.provider}, falling back to local")
                return LocalEmbedder()
        except Exception as e:
            logging.error(f"Failed to initialize {self.provider} embedder: {e}")
            logging.info("Falling back to local embedder")
            return LocalEmbedder()
    
    def process_pending_embeddings(self, folder_key: Optional[str] = None, 
                                 max_chunks: Optional[int] = None) -> int:
        """Process pending embeddings for a folder or all folders.
        
        Args:
            folder_key: Specific folder to process, or None for all folders
            max_chunks: Maximum number of chunks to process in this call
            
        Returns:
            Number of chunks processed
        """
        with self.processing_lock:
            # Get chunks without embeddings
            pending_chunks = self.db.get_chunks_without_embeddings(folder_key)
            
            if not pending_chunks:
                return 0
            
            if max_chunks:
                pending_chunks = pending_chunks[:max_chunks]
            
            logging.info(f"Processing embeddings for {len(pending_chunks)} chunks" + 
                        (f" in folder '{folder_key}'" if folder_key else ""))
            
            processed_count = 0
            
            # Process in batches
            for i in range(0, len(pending_chunks), self.batch_size):
                if self.stop_event.is_set():
                    break
                    
                batch = pending_chunks[i:i + self.batch_size]
                
                try:
                    # Prepare texts for embedding
                    summary_texts = []
                    full_texts = []
                    
                    for chunk in batch:
                        content = chunk['content']
                        metadata = chunk['metadata']
                        
                        # Build full text representation (same as in embed.py)
                        full_text = self._build_full_text_for_chunk(content, metadata)
                        
                        # Generate summary text
                        summary_text = self._generate_summary_text(content, metadata, full_text)
                        
                        # Add document prefix if needed
                        if self.doc_prefix:
                            summary_texts.append(f"{self.doc_prefix}{summary_text}")
                            full_texts.append(f"{self.doc_prefix}{full_text}")
                        else:
                            summary_texts.append(summary_text)
                            full_texts.append(full_text)
                    
                    # Generate embeddings
                    if self.provider == "colbert":
                        # ColBERT has special handling
                        summary_embeddings = self.embedder.embed_pooled(summary_texts)
                        full_embeddings = self.embedder.embed_pooled(full_texts)
                    else:
                        summary_embeddings = self.embedder.embed(summary_texts)
                        full_embeddings = self.embedder.embed(full_texts)
                    
                    # Update database
                    for chunk, sum_emb, full_emb in zip(batch, summary_embeddings, full_embeddings):
                        self.db.update_chunk_embeddings(chunk['id'], sum_emb, full_emb)
                        processed_count += 1
                    
                    logging.debug(f"Processed embedding batch {i//self.batch_size + 1}/{(len(pending_chunks) + self.batch_size - 1)//self.batch_size}")
                    
                except Exception as e:
                    logging.error(f"Failed to process embedding batch: {e}")
                    continue
            
            if processed_count > 0:
                logging.info(f"✅ Generated embeddings for {processed_count} chunks")
            
            return processed_count
    
    def _build_full_text_for_chunk(self, content: str, metadata: Dict) -> str:
        """Build full text representation for a chunk (simplified version of embed.py logic)."""
        source_type = metadata.get('source_type', '')
        content_format = metadata.get('content_format', '')
        
        # Handle CSV tables
        if content_format == "csv" or source_type == "csv":
            return self._summarize_csv_content(content)
        
        # Handle images
        if source_type == "image":
            return content  # Already processed in parse.py
        
        # Handle markdown with heading paths
        heading_path = metadata.get('heading_path', [])
        if heading_path and isinstance(heading_path, list):
            prefix = " > ".join([str(h) for h in heading_path if h])
            return f"Section: {prefix}.\n{content}" if content else f"Section: {prefix}."
        
        return content
    
    def _generate_summary_text(self, content: str, metadata: Dict, full_text: str, max_chars: int = 280) -> str:
        """Generate summary text for a chunk (simplified version of embed.py logic)."""
        source_type = metadata.get('source_type', '')
        content_format = metadata.get('content_format', '')
        
        if content_format == "csv" or source_type == "csv":
            cols = metadata.get('columns', [])
            nrows = metadata.get('num_rows')
            if isinstance(cols, list) and cols:
                col_list = ", ".join([str(c) for c in cols[:10]])
            else:
                col_list = "table columns"
            nrows_str = f"{int(nrows)}" if isinstance(nrows, int) else "multiple"
            return f"This table summarizes {nrows_str} rows across columns {col_list}."
        
        if source_type == "image":
            if "Caption:" in full_text:
                first = full_text.split("\n", 1)[0]
                return first.replace("Figure caption:", "This figure").strip().rstrip(".") + "."
            return "This figure provides visual information relevant to the surrounding section."
        
        # Default: first sentence/line trimmed
        first_line = full_text.strip().split("\n")[0]
        if len(first_line) > max_chars:
            first_line = first_line[:max_chars].rstrip() + "…"
        return first_line
    
    def _summarize_csv_content(self, csv_text: str, max_sample_rows: int = 10) -> str:
        """Summarize CSV content (simplified version of embed.py logic)."""
        lines = [ln for ln in csv_text.splitlines() if ln.strip()]
        if not lines:
            return "Empty table."
            
        header = [h.strip() for h in lines[0].split(",")]
        rows = [r.split(",") for r in lines[1:]]
        
        col_desc = ", ".join(header[:20])  # cap to avoid overlong headers
        parts = [f"Table with {len(rows)} rows and {len(header)} columns: {col_desc}."]
        
        if rows:
            sample = rows[:max_sample_rows]
            for idx, r in enumerate(sample, start=1):
                kv = ", ".join(
                    f"{header[i]}={r[i].strip()}" for i in range(min(len(header), len(r))) if header[i]
                )
                parts.append(f"Row {idx}: {kv}.")
            if len(rows) > max_sample_rows:
                parts.append(f"… {len(rows) - max_sample_rows} more rows not shown.")
                
        return " ".join(parts)
    
    def start_background_processing(self, check_interval: int = 30) -> List[threading.Thread]:
        """Start background embedding processing threads.
        
        Args:
            check_interval: How often to check for pending work (seconds)
            
        Returns:
            List of started worker threads
        """
        def background_worker(worker_id: int):
            """Background worker function."""
            logging.info(f"Started background embedding worker {worker_id}")
            
            while not self.stop_event.is_set():
                try:
                    # Process pending embeddings for all folders
                    processed = self.process_pending_embeddings(max_chunks=self.batch_size * 4)
                    
                    if processed == 0:
                        # No work found, sleep longer
                        self.stop_event.wait(check_interval)
                    else:
                        # Work found, check again soon
                        self.stop_event.wait(5)
                        
                except Exception as e:
                    logging.error(f"Background worker {worker_id} error: {e}")
                    self.stop_event.wait(check_interval)
            
            logging.info(f"Background embedding worker {worker_id} stopped")
        
        # Start worker threads
        self.worker_threads = []
        for i in range(self.max_workers):
            thread = threading.Thread(target=background_worker, args=(i,), daemon=True)
            thread.start()
            self.worker_threads.append(thread)
        
        logging.info(f"Started {len(self.worker_threads)} background embedding workers")
        return self.worker_threads
    
    def stop_background_processing(self):
        """Stop background processing threads."""
        logging.info("Stopping background embedding workers...")
        self.stop_event.set()
        
        # Wait for threads to finish
        for thread in self.worker_threads:
            thread.join(timeout=10)
        
        self.worker_threads = []
        logging.info("Background embedding workers stopped")
    
    def process_folder_immediately(self, folder_key: str) -> int:
        """Process all pending embeddings for a specific folder immediately.
        
        Args:
            folder_key: The folder to process
            
        Returns:
            Number of chunks processed
        """
        logging.info(f"Processing folder '{folder_key}' immediately...")
        return self.process_pending_embeddings(folder_key)
    
    def get_embedding_status(self, folder_key: Optional[str] = None) -> Dict:
        """Get embedding processing status.
        
        Args:
            folder_key: Specific folder to check, or None for all folders
            
        Returns:
            Dictionary with status information
        """
        if folder_key:
            stats = self.db.get_folder_stats(folder_key)
            pending_chunks = len(self.db.get_chunks_without_embeddings(folder_key))
            
            return {
                'folder': folder_key,
                'total_chunks': stats['total_chunks'],
                'embedded_chunks': stats['embedded_chunks'],
                'pending_embeddings': pending_chunks,
                'completion_rate': (stats['embedded_chunks'] / max(1, stats['total_chunks'])) * 100
            }
        else:
            all_folders = self.db.get_all_folders()
            total_chunks = 0
            total_embedded = 0
            total_pending = 0
            
            for folder in all_folders:
                stats = self.db.get_folder_stats(folder)
                pending = len(self.db.get_chunks_without_embeddings(folder))
                
                total_chunks += stats['total_chunks']
                total_embedded += stats['embedded_chunks']
                total_pending += pending
            
            return {
                'total_folders': len(all_folders),
                'total_chunks': total_chunks,
                'embedded_chunks': total_embedded,
                'pending_embeddings': total_pending,
                'completion_rate': (total_embedded / max(1, total_chunks)) * 100
            }


def create_embedding_service(provider: str = "local", 
                           model_name: str = "BAAI/bge-large-en-v1.5",
                           **kwargs) -> FastEmbeddingService:
    """Create and return a configured FastEmbeddingService."""
    return FastEmbeddingService(
        provider=provider,
        model_name=model_name,
        **kwargs
    )


if __name__ == "__main__":
    # Test the embedding service
    import tempfile
    import os
    
    # Create a temporary database for testing
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        test_db_path = tmp.name
    
    try:
        # Initialize service
        service = FastEmbeddingService(db_path=test_db_path, batch_size=2)
        
        # Add some test data to the database
        db = service.db
        
        # Create a test document and chunks
        test_file = "test_doc.txt"
        doc_id = db.mark_document_processing(test_file, "test_folder")
        
        test_chunks = [
            {
                'id': 'chunk1',
                'content': 'This is the first test chunk with some content.',
                'metadata': {'chunk_index': 1, 'source_type': 'txt'}
            },
            {
                'id': 'chunk2', 
                'content': 'This is the second test chunk with different content.',
                'metadata': {'chunk_index': 2, 'source_type': 'txt'}
            }
        ]
        
        db.save_document_chunks(test_file, test_chunks)
        db.mark_document_completed(test_file)
        
        print("Test data created")
        
        # Test embedding processing
        processed = service.process_pending_embeddings("test_folder")
        print(f"Processed {processed} chunks")
        
        # Test status
        status = service.get_embedding_status("test_folder")
        print(f"Status: {status}")
        
        print("FastEmbeddingService test completed successfully")
        
    finally:
        # Clean up
        os.unlink(test_db_path)
        if os.path.exists(test_file):
            os.unlink(test_file)
