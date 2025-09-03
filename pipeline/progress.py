#!/usr/bin/env python3
"""
Simple CLI progress tracker for the Synapse pipeline.

Provides fast, lightweight progress tracking for Parse -> Chunk -> Embed steps
without fancy animations or heavy dependencies.
"""

import sys
import time
from typing import Optional, Callable


class ProgressTracker:
    """Simple CLI progress tracker that shows current process and completion counts."""
    
    def __init__(self, total_items: int, process_name: str = "Processing"):
        self.total_items = total_items
        self.process_name = process_name
        self.completed = 0
        self.start_time = time.time()
        self.current_item = ""
        self.last_update = 0
        
    def update(self, current_item: str, completed_count: Optional[int] = None):
        """Update progress with current item being processed."""
        if completed_count is not None:
            self.completed = completed_count
        else:
            self.completed += 1
            
        self.current_item = current_item
        
        # Only update display every 0.1 seconds to avoid flickering
        current_time = time.time()
        if current_time - self.last_update < 0.1 and self.completed < self.total_items:
            return
        self.last_update = current_time
        
        self._display_progress()
    
    def _display_progress(self):
        """Display current progress on a single line."""
        # Calculate percentage and timing
        percent = (self.completed / max(1, self.total_items)) * 100
        elapsed = time.time() - self.start_time
        
        # Estimate remaining time
        if self.completed > 0:
            rate = self.completed / elapsed
            remaining = (self.total_items - self.completed) / rate if rate > 0 else 0
            eta = f" ETA: {int(remaining)}s" if remaining > 0 else ""
        else:
            eta = ""
        
        # Create progress bar (simple text version)
        bar_width = 20
        filled = int(bar_width * percent / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        
        # Format current item path (truncate if too long)
        item_display = self.current_item
        if len(item_display) > 60:
            item_display = "..." + item_display[-57:]
        
        # Print progress line
        progress_line = f"\r{self.process_name}: [{bar}] {self.completed}/{self.total_items} ({percent:.1f}%){eta}"
        item_line = f"\nCurrent: {item_display}"
        
        # Clear previous lines and print new status
        sys.stdout.write("\033[2K")  # Clear current line
        sys.stdout.write(progress_line)
        sys.stdout.write("\033[1A\033[2K")  # Move up and clear
        sys.stdout.write(item_line)
        sys.stdout.flush()
    
    def finish(self, success: bool = True):
        """Complete the progress tracking."""
        elapsed = time.time() - self.start_time
        status = "✅" if success else "❌"
        
        sys.stdout.write("\033[2K\r")  # Clear line and return to start
        if success:
            print(f"{status} {self.process_name} completed: {self.completed}/{self.total_items} in {elapsed:.1f}s")
        else:
            print(f"{status} {self.process_name} failed after {elapsed:.1f}s")
        sys.stdout.flush()


def create_progress_callback(tracker: ProgressTracker) -> Callable[[str], None]:
    """Create a callback function for progress updates."""
    def callback(current_item: str):
        tracker.update(current_item)
    return callback


def count_files_to_process(input_dir: str) -> int:
    """Count the number of files that will be processed."""
    import os
    from pathlib import Path
    
    # Supported extensions from parse.py
    supported_extensions = {".pdf", ".pptx", ".docx", ".md", ".csv", ".txt", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
    
    count = 0
    for file_path in Path(input_dir).rglob("*"):
        if file_path.is_file() and not file_path.name.startswith('.'):
            if file_path.suffix.lower() in supported_extensions:
                count += 1
    
    return count
