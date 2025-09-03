#!/usr/bin/env python3
"""
Folder-Based Synapse Initialization Script

This script runs the complete folder-based Parse → Chunk → Embed pipeline
to create specialized knowledge bases for each folder in your Data directory.

Usage:
    python initialize_folders.py [--provider local] [--verbose] [--force-reprocess]
    
Features:
- Automatic folder-based pipeline execution
- Creates separate knowledge bases for each folder
- Special handling for '#' prefixed directories (creates KB per subfolder)
- Smart caching per folder
- Progress tracking with detailed status
- Knowledge base discovery and listing
"""

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Import the existing initialization functions
from initialize import (
    setup_logging, print_banner, check_dependencies, check_environment,
    run_command, check_file_exists
)


def run_folder_based_pipeline(
    data_dir: str,
    artifacts_dir: str,
    provider: str = "local",
    embed_model: str = "BAAI/bge-large-en-v1.5",
    verbose: bool = False,
    force_reprocess: bool = False,
    disable_image_deduplication: bool = False
) -> bool:
    """Run the complete folder-based pipeline."""
    
    print(f"\n🗂️  STARTING FOLDER-BASED PIPELINE")
    print("="*80)
    
    success = True
    
    # Step 1: Folder-based parsing
    print(f"\n📖 STEP 1: FOLDER-BASED PARSING")
    cmd = [
        sys.executable, "pipeline/parse.py",
        "--input", data_dir,
        "--output", os.path.join(artifacts_dir, "parsed.jsonl"),
        "--folder-based",
        "--extract-images"
    ]
    if verbose:
        cmd.append("--verbose")
    if force_reprocess:
        cmd.append("--no-cache")
    if disable_image_deduplication:
        cmd.append("--disable-image-deduplication")
    
    if not run_command(cmd, "FOLDER-BASED PARSING", force_reprocess):
        return False
    
    # Step 2: Folder-based chunking
    print(f"\n🔨 STEP 2: FOLDER-BASED CHUNKING")
    cmd = [
        sys.executable, "pipeline/chunk.py",
        "--folder-based"
    ]
    if verbose:
        cmd.append("--verbose")
    
    if not run_command(cmd, "FOLDER-BASED CHUNKING", force_reprocess):
        return False
    
    # Step 3: Folder-based embedding
    print(f"\n🧠 STEP 3: FOLDER-BASED EMBEDDING")
    cmd = [
        sys.executable, "pipeline/embed.py",
        "--folder-based",
        "--provider", provider,
        "--embed-model", embed_model
    ]
    if verbose:
        cmd.append("--verbose")
    if force_reprocess:
        cmd.append("--no-cache")
    
    if not run_command(cmd, "FOLDER-BASED EMBEDDING", force_reprocess):
        return False
    
    return True


def list_created_knowledge_bases(artifacts_dir: str) -> None:
    """List the knowledge bases that were created."""
    import glob
    
    # Find all embedding files
    pattern = os.path.join(artifacts_dir, "embedded_*.jsonl")
    embedding_files = glob.glob(pattern)
    
    if not embedding_files:
        print("⚠️  No knowledge bases found after processing")
        return
    
    print(f"\n📚 CREATED KNOWLEDGE BASES ({len(embedding_files)}):")
    print("="*60)
    
    for embed_file in sorted(embedding_files):
        basename = os.path.basename(embed_file)
        if basename.startswith("embedded_") and basename.endswith(".jsonl"):
            folder_name = basename[9:-6]  # Remove "embedded_" and ".jsonl"
            display_name = folder_name.replace('_', ' ').replace('hash_', '#')
            
            # Check for corresponding files
            chunked_file = os.path.join(artifacts_dir, f"chunked_{folder_name}.jsonl")
            parsed_file = os.path.join(artifacts_dir, f"parsed_{folder_name}.jsonl")
            
            print(f"📁 {display_name}")
            print(f"   ID: {folder_name}")
            print(f"   📄 Files: {os.path.basename(embed_file)}")
            if os.path.exists(chunked_file):
                print(f"          {os.path.basename(chunked_file)}")
            if os.path.exists(parsed_file):
                print(f"          {os.path.basename(parsed_file)}")
            
            # Get file sizes for reference
            try:
                embed_size = os.path.getsize(embed_file)
                print(f"   📊 Size: {embed_size:,} bytes")
            except:
                pass
            
            print()


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize folder-based Synapse knowledge bases")
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="Data", 
        help="Input directory containing documents (default: Data)"
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Output directory for artifacts (default: artifacts)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "openai", "colbert", "bert"],
        default="local",
        help="Embedding provider (default: local)"
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model name"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Force reprocessing (disable caching)"
    )
    parser.add_argument(
        "--disable-image-deduplication",
        action="store_true",
        help="Disable image deduplication (will save all images even if identical)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    print_banner()
    
    # Setup paths
    data_dir = os.path.abspath(args.data_dir)
    artifacts_dir = os.path.abspath(args.artifacts_dir)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Artifacts directory: {artifacts_dir}")
    print(f"🤖 Embedding provider: {args.provider}")
    print(f"🧠 Embedding model: {args.embed_model}")
    print(f"🗂️  Mode: Folder-based processing")
    
    # Pre-flight checks
    print(f"\n🔍 Running pre-flight checks...")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"❌ Data directory not found: {data_dir}")
        return 1
    
    # Count files in data directory
    data_files = list(Path(data_dir).rglob("*"))
    data_files = [f for f in data_files if f.is_file() and not f.name.startswith('.')]
    print(f"   📊 Found {len(data_files)} files in {data_dir}")
    
    # Check dependencies
    deps_ok, missing_deps = check_dependencies()
    if not deps_ok:
        print(f"❌ Missing dependencies: {', '.join(missing_deps)}")
        print("   Run: pip install -r requirements.txt")
        return 1
    print("   ✅ All dependencies installed")
    
    # Check environment variables
    env_ok, missing_env = check_environment()
    if not env_ok:
        print(f"⚠️  Missing environment variables: {', '.join(missing_env)}")
        print("   These are required for the query step but not for initialization")
    else:
        print("   ✅ Environment variables configured")
    
    # Run folder-based pipeline
    success = run_folder_based_pipeline(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        provider=args.provider,
        embed_model=args.embed_model,
        verbose=args.verbose,
        force_reprocess=args.force_reprocess,
        disable_image_deduplication=args.disable_image_deduplication
    )
    
    # Final status
    print("\n" + "="*80)
    if success:
        print("🎉 FOLDER-BASED INITIALIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # List created knowledge bases
        list_created_knowledge_bases(artifacts_dir)
        
        print("🚀 USAGE EXAMPLES:")
        print("   # List all knowledge bases:")
        print("   python pipeline/query.py --list-kb")
        print("   python chat.py --list-kb")
        print()
        print("   # Query specific knowledge base:")
        print("   python pipeline/query.py --kb \"Aahan_s_Notes\" --question \"What is RISC-V?\"")
        print("   python pipeline/query.py --kb \"hash_Confluence_IPS\" --question \"IPS requirements\"")
        print()
        print("   # Interactive knowledge base selection:")
        print("   python pipeline/query.py --select-kb --question \"Your question\"")
        print("   python chat.py --select-kb")
        print()
        print("   # Start chat with specific knowledge base:")
        print("   python chat.py --kb \"Aahan_s_Notes\"")
        print()
        print("   # Start GUI with knowledge base selection:")
        print("   python chat.py --test_gui --kb \"Aahan_s_Notes\"")
        print()
        print("💡 KNOWLEDGE BASE FEATURES:")
        print("   • Each folder becomes a specialized knowledge base")
        print("   • Use /kb and /switch-kb commands in interactive chat")
        print("   • GUI supports dynamic knowledge base switching")
        print("   • Image deduplication prevents saving duplicate images")
        print("   • See KNOWLEDGE_BASE_USAGE.md and IMAGE_DEDUPLICATION_GUIDE.md for guides")
        
        print("="*80)
        return 0
    else:
        print("❌ FOLDER-BASED INITIALIZATION FAILED!")
        print("="*80)
        print("🔧 TROUBLESHOOTING:")
        print("   1. Check error messages above")
        print("   2. Verify input data directory exists and contains folders")
        print("   3. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   4. Try running individual pipeline steps manually")
        print("   5. Use --verbose for more detailed output")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())

