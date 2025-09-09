#!/usr/bin/env python3
"""
Synapse Folder-Based Initialization Script (Legacy)

⚠️  NOTICE: This is the legacy initialization script. For better performance,
        use 'initialize_fast.py' which offers 10-100x faster incremental updates.

This script runs the complete folder-based Parse → Chunk → Embed pipeline
to create specialized knowledge bases for each folder in your Data directory.

Usage:
    python initialize.py [--data-dir Data] [--force-reprocess] [--provider local]
    
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
from typing import List, Optional, Tuple


def setup_logging(verbose: bool = False) -> None:
    """Configure logging for the initialization process."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )


def print_banner() -> None:
    """Print the Synapse initialization banner."""
    print("\n" + "="*80)
    print("██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗")
    print("██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝")
    print("██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   ")
    print("██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   ")
    print("██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   ")
    print("╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   ")
    print("                                                          ")
    print("              KNOWLEDGE BASE INITIALIZATION                   ")
    print("="*80)


def check_dependencies() -> Tuple[bool, List[str]]:
    """Check if required dependencies are installed."""
    missing = []
    # Map package names to their import names
    required_packages = {
        'unstructured': 'unstructured',
        'sentence_transformers': 'sentence_transformers', 
        'faiss-cpu': 'faiss',
        'rank_bm25': 'rank_bm25',
        'litellm': 'litellm',
        'langchain_text_splitters': 'langchain_text_splitters'
    }
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name.replace('-', '_'))
        except ImportError:
            missing.append(package_name)
    
    return len(missing) == 0, missing


def check_environment() -> Tuple[bool, List[str]]:
    """Check if required environment variables are set."""
    missing = []
    required_env = ['LITELLM_API_KEY', 'LITELLM_BASE_URL']
    
    for var in required_env:
        if not os.getenv(var):
            missing.append(var)
    
    return len(missing) == 0, missing


def run_command(cmd: List[str], stage_name: str, no_cache: bool = False) -> bool:
    """Run a pipeline command with error handling and progress tracking."""
    if no_cache:
        cmd.append('--no-cache')
    
    print(f"\n🔄 Starting {stage_name}...")
    print(f"   Command: {' '.join(cmd)}")
    print("   Progress will be shown below:")
    print()
    
    start_time = time.time()
    
    try:
        # Run command with real-time output for progress tracking
        result = subprocess.run(
            cmd,
            capture_output=False,  # Allow real-time output
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"\n✅ {stage_name} completed successfully in {elapsed:.1f}s")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {stage_name} failed after {elapsed:.1f}s")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Output: {e.stdout}")
        if e.stderr:
            print(f"   Error output: {e.stderr}")
        return False


def check_file_exists(filepath: str, stage_name: str) -> bool:
    """Check if output file exists and show info."""
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"   📄 {stage_name} output: {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"   ❌ {stage_name} output missing: {filepath}")
        return False


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
    if args.verbose:
        cmd.append("--verbose")
    if args.force_reprocess:
        cmd.append("--no-cache")
    if args.disable_image_deduplication:
        cmd.append("--disable-image-deduplication")
    
    if not run_command(cmd, "FOLDER-BASED PARSING", args.force_reprocess):
        success = False
    
    # Step 2: Folder-based chunking
    if success:
        print(f"\n🔨 STEP 2: FOLDER-BASED CHUNKING")
        cmd = [
            sys.executable, "pipeline/chunk.py",
            "--folder-based"
        ]
        if args.verbose:
            cmd.append("--verbose")
        
        if not run_command(cmd, "FOLDER-BASED CHUNKING", args.force_reprocess):
            success = False
    
    # Step 3: Folder-based embedding
    if success:
        print(f"\n🧠 STEP 3: FOLDER-BASED EMBEDDING")
        cmd = [
            sys.executable, "pipeline/embed.py",
            "--folder-based",
            "--provider", args.provider,
            "--embed-model", args.embed_model
        ]
        if args.verbose:
            cmd.append("--verbose")
        if args.force_reprocess:
            cmd.append("--no-cache")
        
        if not run_command(cmd, "FOLDER-BASED EMBEDDING", args.force_reprocess):
            success = False
    
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
        print("   • See Docs/KNOWLEDGE_BASE_USAGE.md and Docs/IMAGE_DEDUPLICATION_GUIDE.md for guides")
        
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
