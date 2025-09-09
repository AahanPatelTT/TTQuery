#!/usr/bin/env python3
"""
Database layer for fast incremental document updates and real-time embedding generation.

This module provides SQLite-based caching and tracking for documents, chunks, and embeddings
to enable fast incremental updates without full re-initialization.
"""

import sqlite3
import hashlib
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import threading
import logging
from pathlib import Path


@dataclass
class DocumentRecord:
    """Database record for a document."""
    id: str
    file_path: str
    file_hash: str
    file_size: int
    file_mtime: float
    last_processed: float
    status: str  # 'pending', 'processing', 'completed', 'failed'
    metadata: Dict
    folder_key: str
    error_message: Optional[str] = None


@dataclass
class ChunkRecord:
    """Database record for a chunk."""
    id: str
    document_id: str
    chunk_index: int
    content: str
    metadata: Dict
    embedding_summary: Optional[List[float]] = None
    embedding_full: Optional[List[float]] = None
    last_updated: float = None


class SynapseDB:
    """Fast database for incremental document processing."""
    
    def __init__(self, db_path: str = "artifacts/synapse.db"):
        self.db_path = os.path.abspath(db_path)
        self.lock = threading.Lock()
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            # Documents table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT UNIQUE NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    file_mtime REAL NOT NULL,
                    last_processed REAL NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    folder_key TEXT NOT NULL,
                    error_message TEXT
                )
            """)
            
            # Chunks table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    embedding_summary TEXT,
                    embedding_full TEXT,
                    last_updated REAL NOT NULL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Processing queue table for background tasks
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processing_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    priority INTEGER DEFAULT 0,
                    created_at REAL NOT NULL,
                    started_at REAL,
                    completed_at REAL,
                    status TEXT DEFAULT 'pending',
                    error_message TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            """)
            
            # Indexes for fast lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_folder ON documents(folder_key)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_mtime ON documents(file_mtime)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_updated ON chunks(last_updated)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_queue_status ON processing_queue(status, priority)")
            
            conn.commit()
            
        logging.info(f"Initialized Synapse database at {self.db_path}")
    
    def compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of file content for change detection."""
        hasher = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            logging.warning(f"Failed to compute hash for {file_path}: {e}")
            return ""
    
    def get_document_status(self, file_path: str) -> Optional[DocumentRecord]:
        """Get document status from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM documents WHERE file_path = ?", 
                (os.path.abspath(file_path),)
            )
            row = cursor.fetchone()
            if row:
                return DocumentRecord(
                    id=row[0], file_path=row[1], file_hash=row[2],
                    file_size=row[3], file_mtime=row[4], last_processed=row[5],
                    status=row[6], metadata=json.loads(row[7]), folder_key=row[8],
                    error_message=row[9]
                )
        return None
    
    def is_document_unchanged(self, file_path: str) -> bool:
        """Check if document has changed since last successful processing."""
        record = self.get_document_status(file_path)
        if not record or record.status != 'completed':
            return False
        
        try:
            stat = os.stat(file_path)
            # Check both mtime and size for fast detection, plus hash for accuracy
            if stat.st_mtime != record.file_mtime or stat.st_size != record.file_size:
                return False
            
            # If mtime/size match but we want to be extra sure, check hash
            current_hash = self.compute_file_hash(file_path)
            return current_hash == record.file_hash
            
        except OSError:
            return False

    def find_duplicate_by_content(self, file_path: str, folder_key: str) -> Optional[str]:
        """Find if a document with the same content already exists in the folder."""
        try:
            current_hash = self.compute_file_hash(file_path)
            if not current_hash:
                return None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path FROM documents 
                    WHERE folder_key = ? AND file_hash = ? AND file_path != ? AND status = 'completed'
                    LIMIT 1
                """, (folder_key, current_hash, os.path.abspath(file_path)))
                
                row = cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            logging.warning(f"Failed to check for duplicates: {e}")
            return None

    def find_duplicate_by_name(self, file_path: str, folder_key: str) -> Optional[str]:
        """Find if a document with the same name already exists in the folder."""
        try:
            filename = os.path.basename(file_path)
            folder_path = os.path.join(os.path.dirname(file_path), '')  # Ensure trailing slash
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path FROM documents 
                    WHERE folder_key = ? AND file_path LIKE ? AND file_path != ? AND status = 'completed'
                    LIMIT 1
                """, (folder_key, f'%/{filename}', os.path.abspath(file_path)))
                
                row = cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            logging.warning(f"Failed to check for name duplicates: {e}")
            return None
    
    def mark_document_processing(self, file_path: str, folder_key: str) -> str:
        """Mark document as being processed and return document ID."""
        abs_path = os.path.abspath(file_path)
        doc_id = hashlib.sha256(abs_path.encode()).hexdigest()
        
        try:
            stat = os.stat(file_path)
            file_hash = self.compute_file_hash(file_path)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO documents 
                    (id, file_path, file_hash, file_size, file_mtime, last_processed, status, metadata, folder_key, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    doc_id, abs_path, file_hash, stat.st_size, stat.st_mtime,
                    time.time(), 'processing', '{}', folder_key, None
                ))
                conn.commit()
                
            logging.debug(f"Marked document for processing: {file_path}")
            return doc_id
            
        except Exception as e:
            logging.error(f"Failed to mark document processing: {e}")
            raise
    
    def mark_document_completed(self, file_path: str) -> bool:
        """Mark document processing as completed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    UPDATE documents SET status = ?, last_processed = ? 
                    WHERE file_path = ?
                """, ('completed', time.time(), os.path.abspath(file_path)))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    logging.debug(f"Marked document completed: {file_path}")
                    return True
                    
        except Exception as e:
            logging.error(f"Failed to mark document completed: {e}")
            
        return False
    
    def mark_document_failed(self, file_path: str, error_message: str):
        """Mark document processing as failed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE documents SET status = ?, last_processed = ?, error_message = ?
                    WHERE file_path = ?
                """, ('failed', time.time(), error_message, os.path.abspath(file_path)))
                conn.commit()
                
            logging.warning(f"Marked document failed: {file_path} - {error_message}")
            
        except Exception as e:
            logging.error(f"Failed to mark document failed: {e}")
    
    def save_document_chunks(self, file_path: str, chunks: List[Dict]) -> bool:
        """Save processed chunks to database."""
        try:
            abs_path = os.path.abspath(file_path)
            
            with sqlite3.connect(self.db_path) as conn:
                # Get document ID
                cursor = conn.execute("SELECT id FROM documents WHERE file_path = ?", (abs_path,))
                row = cursor.fetchone()
                if not row:
                    logging.error(f"Document not found in database: {file_path}")
                    return False
                
                doc_id = row[0]
                
                # Delete existing chunks for this document
                conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
                
                # Insert new chunks
                current_time = time.time()
                for chunk in chunks:
                    conn.execute("""
                        INSERT INTO chunks 
                        (id, document_id, chunk_index, content, metadata, last_updated)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        chunk['id'], doc_id, 
                        chunk.get('metadata', {}).get('chunk_index', 0),
                        chunk['content'], 
                        json.dumps(chunk.get('metadata', {})),
                        current_time
                    ))
                
                conn.commit()
                logging.debug(f"Saved {len(chunks)} chunks for {file_path}")
                return True
                
        except Exception as e:
            logging.error(f"Failed to save chunks for {file_path}: {e}")
            return False
    
    def get_pending_documents(self, folder_key: Optional[str] = None) -> List[str]:
        """Get list of documents that need processing."""
        with sqlite3.connect(self.db_path) as conn:
            if folder_key:
                cursor = conn.execute("""
                    SELECT file_path FROM documents 
                    WHERE folder_key = ? AND status IN ('pending', 'failed')
                    ORDER BY last_processed ASC
                """, (folder_key,))
            else:
                cursor = conn.execute("""
                    SELECT file_path FROM documents 
                    WHERE status IN ('pending', 'failed')
                    ORDER BY last_processed ASC
                """)
            return [row[0] for row in cursor.fetchall()]
    
    def get_cached_chunks(self, file_path: str) -> List[Dict]:
        """Get cached chunks for a file."""
        abs_path = os.path.abspath(file_path)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.metadata,
                       c.embedding_summary, c.embedding_full, d.file_path, d.folder_key
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.file_path = ? AND d.status = 'completed'
                ORDER BY c.chunk_index
            """, (abs_path,))
            
            chunks = []
            for row in cursor.fetchall():
                # Reconstruct chunk in the format expected by the pipeline
                metadata = json.loads(row[4]) if row[4] else {}
                
                chunk = {
                    'id': row[0],
                    'document_id': row[1],
                    'source_path': row[7],
                    'source_type': metadata.get('source_type', ''),
                    'content': row[3],
                    'metadata': metadata
                }
                
                # Add embeddings if available
                if row[5]:  # embedding_summary
                    chunk['embedding_summary'] = json.loads(row[5])
                if row[6]:  # embedding_full
                    chunk['embedding_full'] = json.loads(row[6])
                    
                chunks.append(chunk)
                
            return chunks
    
    def get_all_chunks_for_folder(self, folder_key: str) -> List[Dict]:
        """Get all chunks for a folder (for building embeddings)."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT c.id, c.document_id, c.chunk_index, c.content, c.metadata,
                       c.embedding_summary, c.embedding_full, d.file_path, d.folder_key
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.folder_key = ? AND d.status = 'completed'
                ORDER BY d.file_path, c.chunk_index
            """, (folder_key,))
            
            chunks = []
            for row in cursor.fetchall():
                metadata = json.loads(row[4]) if row[4] else {}
                
                chunk = {
                    'id': row[0],
                    'document_id': row[1],
                    'source_path': row[7],
                    'source_type': metadata.get('source_type', ''),
                    'content': row[3],
                    'metadata': metadata
                }
                
                if row[5]:  # embedding_summary
                    chunk['embedding_summary'] = json.loads(row[5])
                if row[6]:  # embedding_full
                    chunk['embedding_full'] = json.loads(row[6])
                    
                chunks.append(chunk)
                
            return chunks
    
    def update_chunk_embeddings(self, chunk_id: str, summary_embedding: List[float], full_embedding: List[float]):
        """Update embeddings for a specific chunk."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE chunks 
                    SET embedding_summary = ?, embedding_full = ?, last_updated = ?
                    WHERE id = ?
                """, (
                    json.dumps(summary_embedding),
                    json.dumps(full_embedding),
                    time.time(),
                    chunk_id
                ))
                conn.commit()
                
        except Exception as e:
            logging.error(f"Failed to update embeddings for chunk {chunk_id}: {e}")
    
    def get_chunks_without_embeddings(self, folder_key: Optional[str] = None) -> List[Dict]:
        """Get chunks that need embeddings generated."""
        with sqlite3.connect(self.db_path) as conn:
            if folder_key:
                cursor = conn.execute("""
                    SELECT c.id, c.content, c.metadata, d.folder_key
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.folder_key = ? AND d.status = 'completed' 
                    AND (c.embedding_summary IS NULL OR c.embedding_full IS NULL)
                    ORDER BY c.last_updated ASC
                """, (folder_key,))
            else:
                cursor = conn.execute("""
                    SELECT c.id, c.content, c.metadata, d.folder_key
                    FROM chunks c
                    JOIN documents d ON c.document_id = d.id
                    WHERE d.status = 'completed' 
                    AND (c.embedding_summary IS NULL OR c.embedding_full IS NULL)
                    ORDER BY c.last_updated ASC
                """)
            
            chunks = []
            for row in cursor.fetchall():
                chunks.append({
                    'id': row[0],
                    'content': row[1],
                    'metadata': json.loads(row[2]) if row[2] else {},
                    'folder_key': row[3]
                })
            
            return chunks
    
    def get_folder_stats(self, folder_key: str) -> Dict[str, int]:
        """Get statistics for a folder."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(CASE WHEN status = 'completed' THEN 1 END) as completed_docs,
                    COUNT(CASE WHEN status = 'processing' THEN 1 END) as processing_docs,
                    COUNT(CASE WHEN status = 'failed' THEN 1 END) as failed_docs,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_docs
                FROM documents WHERE folder_key = ?
            """, (folder_key,))
            
            row = cursor.fetchone()
            doc_stats = {
                'completed_docs': row[0] or 0,
                'processing_docs': row[1] or 0,
                'failed_docs': row[2] or 0,
                'pending_docs': row[3] or 0
            }
            
            # Get chunk statistics
            cursor = conn.execute("""
                SELECT 
                    COUNT(c.id) as total_chunks,
                    COUNT(CASE WHEN c.embedding_summary IS NOT NULL AND c.embedding_full IS NOT NULL THEN 1 END) as embedded_chunks
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.folder_key = ? AND d.status = 'completed'
            """, (folder_key,))
            
            row = cursor.fetchone()
            chunk_stats = {
                'total_chunks': row[0] or 0,
                'embedded_chunks': row[1] or 0,
                'pending_embeddings': (row[0] or 0) - (row[1] or 0)
            }
            
            return {**doc_stats, **chunk_stats}
    
    def get_all_folders(self) -> List[str]:
        """Get list of all folder keys in the database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT DISTINCT folder_key FROM documents ORDER BY folder_key")
            return [row[0] for row in cursor.fetchall()]
    
    def cleanup_old_entries(self, days_old: int = 30):
        """Clean up old failed/orphaned entries."""
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        with sqlite3.connect(self.db_path) as conn:
            # Remove old failed documents
            cursor = conn.execute("""
                DELETE FROM documents 
                WHERE status = 'failed' AND last_processed < ?
            """, (cutoff_time,))
            
            deleted_docs = cursor.rowcount
            
            # Remove orphaned chunks
            cursor = conn.execute("""
                DELETE FROM chunks 
                WHERE document_id NOT IN (SELECT id FROM documents)
            """)
            
            deleted_chunks = cursor.rowcount
            conn.commit()
            
            if deleted_docs > 0 or deleted_chunks > 0:
                logging.info(f"Cleaned up {deleted_docs} old documents and {deleted_chunks} orphaned chunks")


def get_default_db() -> SynapseDB:
    """Get the default database instance."""
    return SynapseDB()


if __name__ == "__main__":
    # Test the database
    db = SynapseDB("test_synapse.db")
    print("Database initialized successfully")
    
    # Test basic operations
    test_file = "test_document.txt"
    with open(test_file, 'w') as f:
        f.write("Test content")
    
    try:
        # Test document tracking
        doc_id = db.mark_document_processing(test_file, "test_folder")
        print(f"Document marked for processing: {doc_id}")
        
        # Test chunk saving
        test_chunks = [{
            'id': 'test_chunk_1',
            'content': 'Test chunk content',
            'metadata': {'chunk_index': 1, 'source_type': 'txt'}
        }]
        
        db.save_document_chunks(test_file, test_chunks)
        db.mark_document_completed(test_file)
        
        # Test retrieval
        cached = db.get_cached_chunks(test_file)
        print(f"Retrieved {len(cached)} cached chunks")
        
        # Test stats
        stats = db.get_folder_stats("test_folder")
        print(f"Folder stats: {stats}")
        
    finally:
        os.unlink(test_file)
        os.unlink("test_synapse.db")
        print("Test completed successfully")
