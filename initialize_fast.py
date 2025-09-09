#!/usr/bin/env python3
"""
Fast initialization script for Synapse with database-driven incremental updates.

This script uses the new database system to enable much faster re-initialization
when documents are added, modified, or removed. It only processes changed files
and generates embeddings incrementally.

Usage:
    python initialize_fast.py                    # Process all folders
    python initialize_fast.py --folder "MyDocs"  # Process specific folder
    python initialize_fast.py --status           # Show processing status
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

# Add pipeline to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pipeline'))

from pipeline.database import SynapseDB
from pipeline.fast_embed import FastEmbeddingService
from pipeline.parse import get_folder_structure
import pipeline.parse as parse_module
import pipeline.embed as embed_module


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("fast_init.log")
        ]
    )


def check_dependencies():
    """Check if all required dependencies are available."""
    missing = []
    
    try:
        import sqlite3
    except ImportError:
        missing.append("sqlite3 (should be included with Python)")
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        missing.append("sentence-transformers")
    
    try:
        import faiss
    except ImportError:
        missing.append("faiss-cpu")
    
    if missing:
        print("❌ Missing required dependencies:")
        for dep in missing:
            print(f"   - {dep}")
        print("\nInstall missing dependencies:")
        print("   pip install numpy sentence-transformers faiss-cpu")
        return False
    
    return True


def initialize_database() -> SynapseDB:
    """Initialize and return the Synapse database."""
    db = SynapseDB()
    logging.info("✅ Database initialized")
    return db


def process_folder_incremental(folder_key: str, folder_files: List[str], 
                             db: SynapseDB, args) -> Dict:
    """Process a single folder incrementally."""
    logging.info(f"🔄 Processing folder: {folder_key}")
    
    # Check which files need processing
    changed_files = []
    unchanged_files = []
    
    for file_path in folder_files:
        if db.is_document_unchanged(file_path):
            unchanged_files.append(file_path)
        else:
            changed_files.append(file_path)
            db.mark_document_processing(file_path, folder_key)
    
    if not changed_files:
        logging.info(f"✅ No changes detected in {folder_key} ({len(unchanged_files)} files up to date)")
        return {
            'folder_key': folder_key,
            'total_files': len(folder_files),
            'changed_files': 0,
            'unchanged_files': len(unchanged_files),
            'new_chunks': 0,
            'processing_time': 0
        }
    
    start_time = time.time()
    logging.info(f"📄 Processing {len(changed_files)} changed files (skipping {len(unchanged_files)} unchanged)")
    
    # Parse changed files
    total_chunks = 0
    failed_files = []
    
    for file_path in changed_files:
        try:
            logging.debug(f"Parsing: {os.path.basename(file_path)}")
            
            # Skip images if requested for speed
            file_ext = os.path.splitext(file_path)[1].lower()
            if args.skip_images and file_ext in {'.png', '.jpg', '.jpeg', '.tiff', '.bmp'}:
                logging.debug(f"Skipping image: {os.path.basename(file_path)}")
                db.mark_document_completed(file_path)
                continue
            
            # Parse the file using the existing pipeline
            if args.engine == "unstructured":
                from pipeline.parse import UnstructuredOptions, parse_file_unstructured
                uopts = UnstructuredOptions(
                    pdf_strategy=args.pdf_strategy,
                    ocr_languages=args.ocr_languages,
                    infer_table_structure=not args.no_infer_tables,
                    pdf_tables_as_csv=not args.no_pdf_tables_as_csv,
                )
                chunks = parse_file_unstructured(file_path, uopts, image_captioner=None)
            else:
                from pipeline.parse import parse_file_basic
                chunks = parse_file_basic(file_path, image_captioner=None)
            
            if chunks:
                # Convert ParsedChunk objects to dicts for database storage
                from dataclasses import asdict
                chunk_dicts = [asdict(chunk) for chunk in chunks]
                
                # Save to database
                db.save_document_chunks(file_path, chunk_dicts)
                db.mark_document_completed(file_path)
                
                total_chunks += len(chunks)
                logging.debug(f"✅ Parsed {len(chunks)} chunks from {os.path.basename(file_path)}")
            else:
                logging.warning(f"⚠️  No chunks extracted from {file_path}")
                
        except Exception as e:
            logging.error(f"❌ Failed to parse {file_path}: {e}")
            db.mark_document_failed(file_path, str(e))
            failed_files.append(file_path)
    
    processing_time = time.time() - start_time
    
    result = {
        'folder_key': folder_key,
        'total_files': len(folder_files),
        'changed_files': len(changed_files),
        'unchanged_files': len(unchanged_files),
        'new_chunks': total_chunks,
        'failed_files': len(failed_files),
        'processing_time': processing_time
    }
    
    logging.info(f"✅ Completed {folder_key}: {total_chunks} new chunks in {processing_time:.1f}s")
    return result


def process_embeddings_incremental(db: SynapseDB, embedding_service: FastEmbeddingService, 
                                  folder_keys: Optional[List[str]] = None) -> Dict:
    """Process embeddings incrementally for specified folders or all folders."""
    logging.info("🔮 Processing embeddings incrementally...")
    
    start_time = time.time()
    total_processed = 0
    
    if folder_keys:
        folders_to_process = folder_keys
    else:
        folders_to_process = db.get_all_folders()
    
    folder_results = {}
    
    for folder_key in folders_to_process:
        logging.info(f"🔮 Processing embeddings for: {folder_key}")
        
        try:
            processed_count = embedding_service.process_folder_immediately(folder_key)
            total_processed += processed_count
            
            # Get status
            status = embedding_service.get_embedding_status(folder_key)
            folder_results[folder_key] = {
                'processed_count': processed_count,
                'total_chunks': status['total_chunks'],
                'embedded_chunks': status['embedded_chunks'],
                'completion_rate': status['completion_rate']
            }
            
            if processed_count > 0:
                logging.info(f"✅ Generated {processed_count} new embeddings for {folder_key}")
            else:
                logging.info(f"✅ All embeddings up to date for {folder_key}")
                
        except Exception as e:
            logging.error(f"❌ Failed to process embeddings for {folder_key}: {e}")
            folder_results[folder_key] = {'error': str(e)}
    
    processing_time = time.time() - start_time
    
    return {
        'total_processed': total_processed,
        'processing_time': processing_time,
        'folder_results': folder_results
    }


def export_embeddings_to_jsonl(db: SynapseDB, output_dir: str = "artifacts") -> Dict:
    """Export embeddings from database to JSONL files for compatibility."""
    logging.info("📤 Exporting embeddings to JSONL files...")
    
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}
    
    folders = db.get_all_folders()
    
    for folder_key in folders:
        # Get all chunks with embeddings
        chunks = db.get_all_chunks_for_folder(folder_key)
        embedded_chunks = [chunk for chunk in chunks if 
                          chunk.get('embedding_summary') and chunk.get('embedding_full')]
        
        if not embedded_chunks:
            logging.warning(f"No embedded chunks found for {folder_key}")
            continue
        
        # Export to JSONL
        safe_folder_name = folder_key.replace('/', '_').replace('#', 'hash_')
        output_file = os.path.join(output_dir, f"embedded_{safe_folder_name}.jsonl")
        
        # Convert to the format expected by the query system
        outputs = []
        for chunk in embedded_chunks:
            output_record = {
                'id': chunk['id'],
                'document_id': chunk['document_id'],
                'source_path': chunk['source_path'],
                'source_type': chunk['source_type'],
                'metadata': chunk['metadata'],
                'summary_text': chunk['content'][:280],  # Approximate summary
                'full_text': chunk['content'],
                'embedding_summary': chunk['embedding_summary'],
                'embedding_full': chunk['embedding_full']
            }
            outputs.append(output_record)
        
        # Write JSONL file
        from pipeline.embed import write_jsonl
        write_jsonl(outputs, output_file)
        
        exported_files[folder_key] = {
            'file': output_file,
            'chunks': len(outputs)
        }
        
        logging.info(f"📤 Exported {len(outputs)} embeddings to {os.path.basename(output_file)}")
    
    return exported_files


def show_status(db: SynapseDB, embedding_service: FastEmbeddingService):
    """Show current processing status."""
    print("\n" + "="*80)
    print("📊 SYNAPSE FAST INITIALIZATION STATUS")
    print("="*80)
    
    folders = db.get_all_folders()
    
    if not folders:
        print("❌ No folders found in database. Run initialization first.")
        return
    
    total_docs = 0
    total_chunks = 0
    total_embedded = 0
    
    for folder_key in sorted(folders):
        stats = db.get_folder_stats(folder_key)
        embed_status = embedding_service.get_embedding_status(folder_key)
        
        total_docs += stats['completed_docs']
        total_chunks += stats['total_chunks']
        total_embedded += stats['embedded_chunks']
        
        print(f"📁 {folder_key}:")
        print(f"   📄 Documents: {stats['completed_docs']} completed, {stats['pending_docs']} pending, {stats['failed_docs']} failed")
        print(f"   🧩 Chunks: {stats['total_chunks']}")
        print(f"   🔮 Embeddings: {stats['embedded_chunks']}/{stats['total_chunks']} ({embed_status['completion_rate']:.1f}%)")
        
        if stats['pending_embeddings'] > 0:
            print(f"   ⏳ Pending embeddings: {stats['pending_embeddings']}")
        print()
    
    print(f"📊 TOTALS:")
    print(f"   📂 Folders: {len(folders)}")
    print(f"   📄 Documents: {total_docs}")
    print(f"   🧩 Chunks: {total_chunks}")
    print(f"   🔮 Embeddings: {total_embedded}/{total_chunks} ({(total_embedded/max(1,total_chunks))*100:.1f}%)")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Fast incremental initialization for Synapse")
    parser.add_argument("--input", type=str, default="Data", help="Input directory containing documents")
    parser.add_argument("--folder", type=str, help="Process specific folder only")
    parser.add_argument("--status", action="store_true", help="Show current processing status")
    parser.add_argument("--export-only", action="store_true", help="Only export embeddings to JSONL (no processing)")
    parser.add_argument("--engine", type=str, choices=["basic", "unstructured"], default="basic", 
                       help="Parsing engine")
    parser.add_argument("--pdf-strategy", type=str, default="hi_res", choices=["hi_res", "fast"])
    parser.add_argument("--ocr-languages", type=str, default="eng")
    parser.add_argument("--no-infer-tables", action="store_true")
    parser.add_argument("--no-pdf-tables-as-csv", action="store_true")
    parser.add_argument("--embed-provider", type=str, choices=["local", "openai", "colbert", "bert"], 
                       default="local", help="Embedding provider")
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-large-en-v1.5", 
                       help="Embedding model name")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old/failed entries before processing")
    parser.add_argument("--skip-images", action="store_true", help="Skip image processing for faster initialization")
    parser.add_argument("--fast-images", action="store_true", help="Use fast image processing (no captioning, simple OCR)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Initialize database
    try:
        db = initialize_database()
    except Exception as e:
        logging.error(f"Failed to initialize database: {e}")
        return 1
    
    # Initialize embedding service
    try:
        embedding_service = FastEmbeddingService(
            provider=args.embed_provider,
            model_name=args.embed_model,
            batch_size=64
        )
    except Exception as e:
        logging.error(f"Failed to initialize embedding service: {e}")
        return 1
    
    # Clean up old entries if requested
    if args.cleanup:
        logging.info("🧹 Cleaning up old entries...")
        db.cleanup_old_entries()
    
    # Show status and exit if requested
    if args.status:
        show_status(db, embedding_service)
        return 0
    
    # Export only if requested
    if args.export_only:
        export_results = export_embeddings_to_jsonl(db)
        logging.info(f"✅ Exported {len(export_results)} embedding files")
        return 0
    
    # Get folder structure
    input_dir = os.path.abspath(args.input)
    if not os.path.isdir(input_dir):
        logging.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    folder_groups = get_folder_structure(input_dir)
    
    if args.folder:
        # Process specific folder only
        if args.folder not in folder_groups:
            logging.error(f"Folder '{args.folder}' not found. Available folders: {list(folder_groups.keys())}")
            return 1
        folder_groups = {args.folder: folder_groups[args.folder]}
    
    # Process folders
    start_time = time.time()
    folder_results = []
    
    for folder_key, folder_files in folder_groups.items():
        result = process_folder_incremental(folder_key, folder_files, db, args)
        folder_results.append(result)
    
    # Process embeddings
    embedding_results = process_embeddings_incremental(
        db, embedding_service, list(folder_groups.keys())
    )
    
    # Export embeddings to JSONL files
    export_results = export_embeddings_to_jsonl(db)
    
    total_time = time.time() - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("🚀 FAST INITIALIZATION COMPLETED")
    print("="*80)
    print(f"⏱️  Total time: {total_time:.1f}s")
    print(f"📂 Folders processed: {len(folder_results)}")
    
    total_new_chunks = sum(r['new_chunks'] for r in folder_results)
    total_new_embeddings = embedding_results['total_processed']
    
    print(f"🧩 New chunks: {total_new_chunks}")
    print(f"🔮 New embeddings: {total_new_embeddings}")
    print(f"📤 Exported files: {len(export_results)}")
    
    print("\n📊 FOLDER BREAKDOWN:")
    for result in sorted(folder_results, key=lambda x: x['new_chunks'], reverse=True):
        if result['new_chunks'] > 0 or result['changed_files'] > 0:
            print(f"   📁 {result['folder_key']}: {result['new_chunks']} new chunks from {result['changed_files']} changed files")
    
    print("\n🔮 EMBEDDING BREAKDOWN:")
    for folder_key, result in embedding_results['folder_results'].items():
        if 'error' not in result and result['processed_count'] > 0:
            print(f"   📁 {folder_key}: {result['processed_count']} new embeddings ({result['completion_rate']:.1f}% complete)")
    
    print("="*80)
    print("✅ FAST INITIALIZATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
