#!/usr/bin/env python3
"""
TTQuery Initialization Script

This script automatically runs the complete Parse → Chunk → Embed pipeline
to prepare your knowledge base for querying. It handles the entire setup
process with intelligent defaults and progress tracking.

Usage:
    python initialize.py [--data-dir Data] [--force-reprocess] [--provider local]
    
Features:
- Automatic pipeline execution with dependency checking
- Smart caching (only reprocess changed files)
- Progress tracking with detailed status
- Configurable embedding providers
- Error handling and recovery
- Setup validation
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
    """Print the TTQuery initialization banner."""
    print("\n" + "="*80)
    print("████████╗████████╗ ██████╗ ██╗   ██╗███████╗██████╗ ██╗   ██╗")
    print("╚══██╔══╝╚══██╔══╝██╔═══██╗██║   ██║██╔════╝██╔══██╗╚██╗ ██╔╝")
    print("   ██║      ██║   ██║   ██║██║   ██║█████╗  ██████╔╝ ╚████╔╝ ")
    print("   ██║      ██║   ██║▄▄ ██║██║   ██║██╔══╝  ██╔══██╗  ╚██╔╝  ")
    print("   ██║      ██║   ╚██████╔╝╚██████╔╝███████╗██║  ██║   ██║   ")
    print("   ╚═╝      ╚═╝    ╚══▀▀═╝  ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ")
    print("                                                              ")
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
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        print(f"✅ {stage_name} completed successfully in {elapsed:.1f}s")
        
        # Show key output lines
        if result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines[-5:]:  # Show last 5 lines
                if line.strip() and ('completed' in line.lower() or 'wrote' in line.lower() or '=' in line):
                    print(f"   {line.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"❌ {stage_name} failed after {elapsed:.1f}s")
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Initialize TTQuery knowledge base")
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
        "--skip-parse",
        action="store_true", 
        help="Skip parsing step (use existing parsed.jsonl)"
    )
    parser.add_argument(
        "--skip-chunk",
        action="store_true",
        help="Skip chunking step (use existing chunked.jsonl)"
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding step (use existing embeddings.jsonl)"
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
    
    parsed_path = os.path.join(artifacts_dir, "parsed.jsonl")
    chunked_path = os.path.join(artifacts_dir, "chunked.jsonl") 
    embeddings_path = os.path.join(artifacts_dir, "embeddings.jsonl")
    
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Artifacts directory: {artifacts_dir}")
    print(f"🤖 Embedding provider: {args.provider}")
    print(f"🧠 Embedding model: {args.embed_model}")
    
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
    
    # Pipeline execution
    success = True
    
    # Step 1: Parse
    if not args.skip_parse:
        cmd = [
            sys.executable, "pipeline/parse.py",
            "--input", data_dir,
            "--output", parsed_path
        ]
        if args.verbose:
            cmd.append("--verbose")
        
        if not run_command(cmd, "PARSING", args.force_reprocess):
            success = False
        else:
            check_file_exists(parsed_path, "parsing")
    else:
        print("\n⏭️  Skipping parsing step")
        if not check_file_exists(parsed_path, "parsing"):
            success = False
    
    # Step 2: Chunk
    if success and not args.skip_chunk:
        cmd = [
            sys.executable, "pipeline/chunk.py",
            "--input", parsed_path,
            "--output", chunked_path
        ]
        if args.verbose:
            cmd.append("--verbose")
        
        if not run_command(cmd, "CHUNKING", args.force_reprocess):
            success = False
        else:
            check_file_exists(chunked_path, "chunking")
    elif not args.skip_chunk:
        print("\n⏭️  Skipping chunking due to previous failure")
    else:
        print("\n⏭️  Skipping chunking step")
        if not check_file_exists(chunked_path, "chunking"):
            success = False
    
    # Step 3: Embed
    if success and not args.skip_embed:
        cmd = [
            sys.executable, "pipeline/embed.py",
            "--input", chunked_path,
            "--output", embeddings_path,
            "--provider", args.provider,
            "--embed-model", args.embed_model
        ]
        if args.verbose:
            cmd.append("--verbose")
        
        if not run_command(cmd, "EMBEDDING", args.force_reprocess):
            success = False
        else:
            check_file_exists(embeddings_path, "embedding")
    elif not args.skip_embed:
        print("\n⏭️  Skipping embedding due to previous failure")
    else:
        print("\n⏭️  Skipping embedding step")
        if not check_file_exists(embeddings_path, "embedding"):
            success = False
    
    # Final status
    print("\n" + "="*80)
    if success:
        print("🎉 INITIALIZATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("📋 SUMMARY:")
        print(f"   📄 Parsed documents: {parsed_path}")
        print(f"   🔨 Chunked content: {chunked_path}")
        print(f"   🧠 Embeddings ready: {embeddings_path}")
        print("\n🚀 NEXT STEPS:")
        print("   1. Set environment variables (if not already done):")
        print("      export LITELLM_API_KEY=your_key")
        print("      export LITELLM_BASE_URL=https://litellm-proxy--tenstorrent.workload.tenstorrent.com/")
        print("   2. Start chatting:")
        print("      python chat.py")
        print("   3. Or run a single query:")
        print(f"      python pipeline/query.py --question 'Your question' --embeddings {embeddings_path}")
        print("="*80)
        return 0
    else:
        print("❌ INITIALIZATION FAILED!")
        print("="*80)
        print("🔧 TROUBLESHOOTING:")
        print("   1. Check error messages above")
        print("   2. Verify input data directory exists and contains files")
        print("   3. Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   4. Try running individual pipeline steps manually")
        print("   5. Use --verbose for more detailed output")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
